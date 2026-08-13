# System Architecture & Workflow Diagrams: Efficient PixArt-Sigma LoRA Distillation

This document visualizes the complete system architecture, model block composition, multi-stage distillation math flow, and runtime inference pathways for **Efficient PixArt-Sigma LoRA**.

---

## 1. End-to-End Training & Distillation Pipeline Flow

```mermaid
flowchart TD
    subgraph S0["Stage 0: Data & Feature Preparation"]
        D1["Raw Ink-Wash Scrapes / Kaggle Data<br/>(260 Images)"] --> D2["Florence-2 / JoyCaption<br/>Auto-Captioning"]
        D2 --> D3["Deterministic Filtering<br/>(209 Plant Captions)"]
        D3 --> D4["Precompute SDXL VAE Latents<br/>(4 x 64 x 64, FP16)"]
        D3 --> D5["Precompute T5-XXL Embeddings<br/>(300 tokens, 4096-dim, FP16)"]
        D4 --> D6[("data/archives/clean_latents_512.zip")]
        D5 --> D7[("data/features/t5_embeddings_n260.pt")]
    end

    subgraph S1["Stage 1: 20-Step Style Teacher Training"]
        D6 & D7 --> T1["Base PixArt-Sigma XL-2 (Frozen)"]
        T1 --> T2["Attach LoRA (Rank 16, Alpha 16)<br/>12 Official Target Modules"]
        T2 --> T3["Train 10,000 steps (AdamW, LR=1e-5)<br/>Standard DDPM MSE Loss"]
        T3 --> T4["outputs/style_teacher/...<br/>20-step Style Teacher LoRA"]
        T4 --> T5{"Validate Provenance &<br/>574 FP32 Tensors (13.76M Params)"}
        T5 -->|PASS| T6[("teacher_manifest.json")]
    end

    subgraph S2["Stage 2: Trajectory Caching"]
        T6 --> C1["build_distill_prompt_cache.py<br/>627 Bank Prompts + 30 Eval Prompts"]
        C1 --> C2[("distill_t5_plant627.pt")]
        C2 & T6 --> C3["cache_teacher_trajectories.py<br/>20-step Denoising Rollout (2 Replicas)"]
        C3 --> C4[("Trajectory Cache<br/>(1,254 trajectories x 21 states)")]
    end

    subgraph S3["Stage 3: 4-Step Joint LoRA Distillation"]
        C4 & C2 & D6 & T6 --> S3_1["Initialize from Style Teacher LoRA"]
        S3_1 --> S3_2["Train 2,000 steps (LR=5e-6)<br/>Phased Intervals: (0,5), (5,10), (10,15), (15,20)"]
        S3_2 --> S3_3["Loss = 80% Pseudo-Huber Jump Loss<br/>+ 20% Clean Latent Anchor Loss"]
        S3_3 --> S3_4[("Student 4-Step Best Adapter")]
    end

    subgraph S4["Stage 4: 4-Step Quality Gate Verification"]
        S3_4 --> Q1["generate_evaluation_set.py<br/>(30 Prompts x 4 Seeds = 120 Images)"]
        Q1 --> Q2["evaluate_distilled.py<br/>CLIPScore & CMMD Computation"]
        Q2 --> Q3{"Quality Gate Checks:<br/>1. CLIPScore >= 90% Teacher<br/>2. CMMD <= 1.5x Teacher<br/>3. Speedup >= 5.0x"}
        Q3 -->|FAIL| Q4["Reject / Retrain"]
        Q3 -->|PASS| Q5["Approved 4-Step Student"]
    end

    subgraph S5["Stage 5: 2-Step Joint LoRA Distillation"]
        Q5 & C4 & C2 & D6 --> S5_1["Initialize from Approved 4-Step Student"]
        S5_1 --> S5_2["Train 7,000 steps (LR=2e-6)<br/>Phased Intervals: (0,10), (10,20)"]
        S5_2 --> S5_3["Loss = 80% Jump Loss (50% On-Policy Rollout)<br/>+ 20% Clean Latent Anchor Loss"]
        S5_3 --> S5_4[("Student 2-Step Best Adapter")]
    end

    subgraph S6["Stage 6: Final 2-Step Quality Gate & Benchmark"]
        S5_4 --> F1["generate_evaluation_set.py<br/>(120 Images @ 2 Steps, guidance=1.0)"]
        F1 --> F2["evaluate_distilled.py &<br/>eval_30prompts_cmmd.py"]
        F2 --> F3{"Final Gate:<br/>Speedup >= 11x<br/>CLIPScore >= 95%<br/>Exact 2 DiT Calls"}
        F3 -->|PASS| F4["Production Ready Model<br/>(Latency: 0.244s, Speedup: 11.78x)"]
    end

    style S0 fill:#e8f4f8,stroke:#2b6cb0,stroke-width:2px;
    style S1 fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    style S2 fill:#ede9fe,stroke:#7c3aed,stroke-width:2px;
    style S3 fill:#ecfdf5,stroke:#059669,stroke-width:2px;
    style S4 fill:#fff1f2,stroke:#e11d48,stroke-width:2px;
    style S5 fill:#ecfdf5,stroke:#059669,stroke-width:2px;
    style S6 fill:#f0fdf4,stroke:#16a34a,stroke-width:2px;
```

---

## 2. PixArt-Sigma Diffusion Transformer & LoRA Block Architecture

```mermaid
flowchart TB
    subgraph InputEmbeddings["1. Conditioning & Latent Inputs"]
        X["Noisy Latent Input x_t<br/>(B, 4, 64, 64)"] --> PatchEmbed["PatchEmbed (Patch Size 2x2)<br/>Project to (B, 1024, 1152)"]
        T["Timestep t<br/>(B,)"] --> TimestepEmbed["Timestep Embedding + MLP<br/>(B, 1152)"]
        Prompt["T5-XXL Text Embeddings<br/>(B, 300, 4096)"] --> TextProj["Linear Projection<br/>(B, 300, 1152)"]
        Meta["Resolution & Aspect Ratio"] --> MetaEmbed["AdaLN-Single Modulation<br/>(B, 1152)"]
    end

    subgraph DiTBlock["2. DiT Transformer Block (x28 Layers)"]
        direction TB
        Modulation["Modulation Parameters (scale, shift, gate)<br/>from Timestep + Condition Vector"]
        
        subgraph SelfAttn["Self-Attention Sub-Block"]
            SA_Norm["LayerNorm"]
            SA_Q["Linear to_q + LoRA_A/B (r=16)"]
            SA_K["Linear to_k + LoRA_A/B (r=16)"]
            SA_V["Linear to_v + LoRA_A/B (r=16)"]
            SA_Score["Multi-Head Scaled Dot-Product"]
            SA_Out["Linear to_out.0 + LoRA_A/B (r=16)"]
            
            SA_Norm --> SA_Q & SA_K & SA_V --> SA_Score --> SA_Out
        end

        subgraph CrossAttn["Cross-Attention Sub-Block"]
            CA_Norm["LayerNorm"]
            CA_Q["Linear proj_in + LoRA_A/B (r=16)"]
            CA_K["Linear proj + LoRA_A/B (r=16)"]
            CA_V["Linear linear / linear_1 / linear_2 + LoRA_A/B (r=16)"]
            CA_Score["Cross Multi-Head Attention"]
            CA_Out["Linear proj_out + LoRA_A/B (r=16)"]
            
            CA_Norm --> CA_Q & CA_K & CA_V --> CA_Score --> CA_Out
        end

        subgraph FFN["Feed-Forward Sub-Block"]
            FF_Norm["LayerNorm"]
            FF_In["Linear ff.net.0.proj + LoRA_A/B (r=16)"]
            FF_Act["GELU Activation"]
            FF_Out["Linear ff.net.2 + LoRA_A/B (r=16)"]
            
            FF_Norm --> FF_In --> FF_Act --> FF_Out
        end

        Modulation -.-> SA_Norm & CA_Norm & FF_Norm
        Modulation -.-> SA_Out & CA_Out & FF_Out
    end

    subgraph OutputHead["3. Final Modulation & Output Projection"]
        FinalNorm["Final LayerNorm + AdaLN Modulation"]
        FinalProj["Linear Projection -> (B, 1024, 2x2x8)"]
        Unpatchify["Unpatchify Operation"]
        OutputSplit["Split Output Channels (4 + 4)"]
        EpsilonPred["Denoised Epsilon Prediction (4 channels)"]
        SigmaPred["Learned Variance Sigma (4 channels)"]

        FinalNorm --> FinalProj --> Unpatchify --> OutputSplit
        OutputSplit --> EpsilonPred & SigmaPred
    end

    PatchEmbed --> DiTBlock
    TextProj --> CrossAttn
    TimestepEmbed & MetaEmbed --> Modulation
    DiTBlock --> FinalNorm

    style DiTBlock fill:#f8fafc,stroke:#475569,stroke-width:2px;
    style SelfAttn fill:#f0fdf4,stroke:#16a34a,stroke-width:1px;
    style CrossAttn fill:#eff6ff,stroke:#2563eb,stroke-width:1px;
    style FFN fill:#faf5ff,stroke:#9333ea,stroke-width:1px;
```

---

## 3. Mathematical Trajectory Jump & Distillation Loss Flow

```mermaid
flowchart TD
    subgraph LossFormulation["Loss Computation in Student Distillation Step"]
        Dice{"Sample Loss Branch<br/>Random Coin Flip"}
        
        Dice -->|Probability = 0.20| AnchorBranch["Anchor Loss Branch (Clean Latent Regularization)"]
        Dice -->|Probability = 0.80| TrajectoryBranch["Phased Trajectory Jump Branch"]
        
        subgraph AnchorLoss["Clean Latent Anchor Loss"]
            Z0["Clean Ground-Truth Latent x_0<br/>from clean_latents_512.zip"]
            Noise["Sample Random Gaussian Noise ε ~ N(0, I)"]
            T_rand["Sample Random Timestep t in [0, 999]"]
            Zt["Forward DDPM Noising:<br/>x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε"]
            Pred_eps["Student Forward Pass:<br/>ε_pred = Student(x_t, t, Prompt)"]
            Loss_anchor["MSE Loss:<br/>L_anchor = || ε_pred - ε ||²"]
            
            Z0 & Noise & T_rand --> Zt --> Pred_eps --> Loss_anchor
        end

        subgraph TrajLoss["Phased Deterministic Jump Loss"]
            Cache["Trajectory Cache Shard<br/>Sequence of States: x_{t_0}, x_{t_1}, ..., x_{t_20}"]
            Phase["Select Stage Phase Jump:<br/>4-Step: (0,5), (5,10), (10,15), (15,20)<br/>2-Step: (0,10), (10,20)"]
            OnPolicy{"On-Policy Rollout?<br/>(2-Step Phase 2, p=0.5)"}
            
            OnPolicy -->|Yes| Rollout["Compute Jump 1 using Current Student:<br/>x_{t_10} = Jump(x_{t_0}, Student(x_{t_0}, t_0))"]
            OnPolicy -->|No| CacheState["Load Exact Start State x_{t_start}<br/>from Teacher Cache"]
            
            StartLatent["Starting Latent x_{t_start}"]
            TargetLatent["Target Teacher Latent x_{t_target}"]
            
            Rollout --> StartLatent
            CacheState --> StartLatent
            
            StudentJump["Student Forward:<br/>ε_θ = Student(x_{t_start}, t_start, Prompt)<br/>x_hat_{t_target} = Deterministic_Jump(x_{t_start}, ε_θ, t_start, t_target)"]
            
            Loss_traj["Pseudo-Huber Loss:<br/>L_traj = sqrt( || x_hat_{t_target} - x_{t_target} ||² + c² ) - c<br/>(c = 0.001)"]
            
            StartLatent --> StudentJump
            StudentJump & TargetLatent --> Loss_traj
        end

        AnchorBranch --> AnchorLoss
        TrajectoryBranch --> TrajLoss
        
        Loss_anchor --> Backprop["Backward Pass & AdamW Optimizer Step (Weight Decay 0.01)"]
        Loss_traj --> Backprop
    end

    style LossFormulation fill:#f8fafc,stroke:#334155,stroke-width:2px;
    style AnchorLoss fill:#eff6ff,stroke:#1d4ed8,stroke-width:1px;
    style TrajLoss fill:#fdf2f8,stroke:#be185d,stroke-width:1px;
```

---

## 4. Timestep Mapping & Sampling Schedules

```mermaid
gantt
    title Discrete Diffusion Timestep Progression
    dateFormat X
    axisFormat %s

    section Teacher 20-Step (CFG 1.5, 40 passes)
    999 to 949 :active, 0, 50
    949 to 899 :active, 50, 100
    899 to 849 :active, 100, 150
    849 to 799 :active, 150, 200
    799 to 749 :active, 200, 250
    749 to 699 :active, 250, 300
    699 to 649 :active, 300, 350
    649 to 599 :active, 350, 400
    599 to 549 :active, 400, 450
    549 to 500 :active, 450, 500
    500 to 450 :active, 500, 550
    450 to 400 :active, 550, 600
    400 to 350 :active, 600, 650
    350 to 300 :active, 650, 700
    300 to 250 :active, 700, 750
    250 to 200 :active, 750, 800
    200 to 150 :active, 800, 850
    150 to 100 :active, 850, 900
    100 to 50  :active, 900, 950
    50 to 0    :active, 950, 1000

    section Student 4-Step (CFG 1.0, 4 passes)
    Jump 1 (999 -> 749) :crit, 0, 250
    Jump 2 (749 -> 500) :crit, 250, 500
    Jump 3 (500 -> 250) :crit, 500, 750
    Jump 4 (250 -> 0)   :crit, 750, 1000

    section Student 2-Step (CFG 1.0, 2 passes)
    Jump 1 (999 -> 500) :done, 0, 500
    Jump 2 (500 -> 0)   :done, 500, 1000
```

---

## 5. Denoising Forward-Pass & Latency Comparison

```mermaid
flowchart LR
    subgraph Teacher["Teacher (20 Steps, CFG=1.5)"]
        T_Loop["20 Iterations<br/>Cond Pass + Uncond Pass"]
        T_Calls["40 DiT Forward Calls"]
        T_Time["Median Latency: 2.871 s<br/>Speedup: 1.00x"]
        T_Loop --> T_Calls --> T_Time
    end

    subgraph FourStep["Student 4-Step (CFG=1.0)"]
        S4_Loop["4 Deterministic Jumps<br/>No Unconditional Branch"]
        S4_Calls["4 DiT Forward Calls"]
        S4_Time["Median Latency: 0.480 s<br/>Speedup: 5.99x (83.3% Latency Drop)"]
        S4_Loop --> S4_Calls --> S4_Time
    end

    subgraph TwoStep["Student 2-Step (CFG=1.0)"]
        S2_Loop["2 Deterministic Jumps<br/>No Unconditional Branch"]
        S2_Calls["2 DiT Forward Calls"]
        S2_Time["Median Latency: 0.244 s<br/>Speedup: 11.78x (91.5% Latency Drop)"]
        S2_Loop --> S2_Calls --> S2_Time
    end

    style Teacher fill:#fee2e2,stroke:#dc2626,stroke-width:2px;
    style FourStep fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    style TwoStep fill:#dcfce7,stroke:#16a34a,stroke-width:2px;
```

---

## 6. Formal Quality Gate Validation Engine

```mermaid
flowchart TD
    EvalStart["Run Evaluation Suite on 30 Held-out Prompts x 4 Seeds = 120 Images"] --> MetricGen["Extract CLIPScore & CMMD (Kernel Bandwidth σ=10.0)"]
    
    MetricGen --> Gate1{"Gate 1: CLIP Retention<br/>Student CLIPScore / Teacher CLIPScore >= 0.90?"}
    Gate1 -->|No| Fail["REJECT CHECKPOINT (Gate Status: FAIL)"]
    Gate1 -->|Yes| Gate2{"Gate 2: CMMD Distribution<br/>Student CMMD <= 1.5 * Teacher CMMD?"}
    
    Gate2 -->|No| Fail
    Gate2 -->|Yes| Gate3{"Gate 3: Speedup Factor<br/>4-Step: >= 5.0x<br/>2-Step: >= 10.0x?"}
    
    Gate3 -->|No| Fail
    Gate3 -->|Yes| Gate4{"Gate 4: Metadata Integrity<br/>Exact Forward Calls (4 or 2)?<br/>No CFG Branch?<br/>Finite Tensors?"}
    
    Gate4 -->|No| Fail
    Gate4 -->|Yes| Pass["PASS: Write evaluation_summary.json<br/>Proceed to Next Stage / Release Checkpoint"]

    style Fail fill:#fee2e2,stroke:#ef4444,stroke-width:2px;
    style Pass fill:#dcfce7,stroke:#22c55e,stroke-width:2px;
```
