$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\AI\pixart_local_generation"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheScript = Join-Path $ProjectRoot "distillation\cache_teacher_trajectories.py"
$TrainScript = Join-Path $ProjectRoot "distillation\train_student_lora.py"
$TargetCache = Join-Path $ProjectRoot "distillation\target_caches\teacher_trajectories_20to10_g2_n260_r4_seed2026.pt"
$OutputRoot = Join-Path $ProjectRoot "models\distilled_students"
$RunName = "student_20to10_teacher_trajectories_r4_train16000_g2_seed42"
$RunDirectory = Join-Path $OutputRoot $RunName
$StatusPath = Join-Path $ProjectRoot "distillation\stage1_v2_status.json"

function Write-StageStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$Message
    )
    [ordered]@{
        stage = $Stage
        message = $Message
        updated_at = (Get-Date).ToString("o")
        target_cache = $TargetCache
        run_directory = $RunDirectory
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }

    if (-not (Test-Path -LiteralPath $TargetCache)) {
        Write-StageStatus -Stage "caching" -Message "Building 10400 real Teacher trajectory targets."
        & $PythonExe $CacheScript `
            --teacher-guidance 2 `
            --source-steps 20 `
            --student-steps 10 `
            --trajectories-per-prompt 4 `
            --seed 2026 `
            --output $TargetCache
        if ($LASTEXITCODE -ne 0) {
            throw "Teacher trajectory caching exited with code $LASTEXITCODE"
        }
    }

    $TargetMetadata = [System.IO.Path]::ChangeExtension($TargetCache, ".metadata.json")
    if (-not (Test-Path -LiteralPath $TargetCache) -or -not (Test-Path -LiteralPath $TargetMetadata)) {
        throw "Teacher target cache validation failed after generation."
    }

    $FinalAdapter = Join-Path $RunDirectory "final_adapter\adapter_model.safetensors"
    if (-not (Test-Path -LiteralPath $FinalAdapter)) {
        if ((Test-Path -LiteralPath $RunDirectory) -and (Get-ChildItem -LiteralPath $RunDirectory -Force | Select-Object -First 1)) {
            throw "Student output directory exists but is incomplete: $RunDirectory"
        }
        Write-StageStatus -Stage "training" -Message "Training the 10-step Student for 16000 optimizer updates."
        & $PythonExe $TrainScript `
            --target-cache $TargetCache `
            --output-root $OutputRoot `
            --run-name $RunName `
            --max-train-steps 16000 `
            --checkpointing-steps 4000 `
            --learning-rate 1e-6 `
            --batch-size 1 `
            --gradient-accumulation-steps 1 `
            --seed 42
        if ($LASTEXITCODE -ne 0) {
            throw "Student training exited with code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path -LiteralPath $FinalAdapter)) {
        throw "Final Student adapter was not found after training: $FinalAdapter"
    }
    Write-StageStatus -Stage "complete" -Message "Stage 1 v2 cache and Student training completed successfully."
    exit 0
}
catch {
    Write-StageStatus -Stage "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
