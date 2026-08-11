$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64,garbage_collection_threshold:0.8"

$ProjectRoot = "D:\AI\pixart_local_generation"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheScript = Join-Path $ProjectRoot "distillation\cache_teacher_trajectories.py"
$TrainScript = Join-Path $ProjectRoot "distillation\train_student_lora.py"
$TeacherAdapter = Join-Path $ProjectRoot "models\distilled_students\student_20to10_teacher_trajectories_r4_train16000_g2_seed42\checkpoints\step_012000"
$TargetCache = Join-Path $ProjectRoot "distillation\target_caches\teacher_trajectories_10to5_g1_n260_r4_seed3026.pt"
$OutputRoot = Join-Path $ProjectRoot "models\distilled_students"
$RunName = "student_10to5_teacher_step12000_trajectories_r4_train8000_g1_seed43"
$RunDirectory = Join-Path $OutputRoot $RunName
$StatusPath = Join-Path $ProjectRoot "distillation\stage2_10to5_status.json"

function Write-StageStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$Message
    )
    [ordered]@{
        stage = $Stage
        message = $Message
        updated_at = (Get-Date).ToString("o")
        teacher_adapter = $TeacherAdapter
        target_cache = $TargetCache
        run_directory = $RunDirectory
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }

    if (-not (Test-Path -LiteralPath $TargetCache)) {
        Write-StageStatus -Stage "caching" -Message "Building 5200 real 10-step Teacher trajectory targets for the 5-step Student."
        & $PythonExe $CacheScript `
            --teacher-adapter $TeacherAdapter `
            --source-steps 10 `
            --student-steps 5 `
            --teacher-guidance 1 `
            --trajectories-per-prompt 4 `
            --seed 3026 `
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
        Write-StageStatus -Stage "training" -Message "Training the 5-step Student for 8000 optimizer updates."
        & $PythonExe $TrainScript `
            --target-cache $TargetCache `
            --init-adapter $TeacherAdapter `
            --output-root $OutputRoot `
            --run-name $RunName `
            --max-train-steps 8000 `
            --checkpointing-steps 2000 `
            --learning-rate 1e-6 `
            --batch-size 1 `
            --gradient-accumulation-steps 1 `
            --seed 43
        if ($LASTEXITCODE -ne 0) {
            throw "Student training exited with code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path -LiteralPath $FinalAdapter)) {
        throw "Final 5-step Student adapter was not found: $FinalAdapter"
    }
    Write-StageStatus -Stage "complete" -Message "Stage 2 10-to-5 cache and Student training completed successfully."
    exit 0
}
catch {
    Write-StageStatus -Stage "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
