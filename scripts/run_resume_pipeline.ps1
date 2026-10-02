Write-Host "=========================================================="
Write-Host "1. Executing Stage 4B (Soft Routing GRPO) on Kathbath Train Dataset"
Write-Host "   (5 Epochs @ 84k samples, batch 96 = ~4,380 steps)"
Write-Host "=========================================================="
.venv311\Scripts\python.exe moe\train_moe_grpo.py --moe_ckpt runs\moe_hard_routing\checkpoints\stage2_moe\best_model.pt --steps 4500 --batch_size 12 --grad_accum_steps 8 --run_dir runs\moe_grpo_kathbath

Write-Host "=========================================================="
Write-Host "2. Executing Stage 5B (Final Alignment Sequence Polish) on OpenSLR64"
Write-Host "   (~50 Epochs, GRPO polishing the full sequence)"
Write-Host "=========================================================="
.venv311\Scripts\python.exe final_alignment\train_grpo_polish.py --moe_ckpt runs\moe_grpo_kathbath\checkpoints\stage4b_moe_grpo\best_model.pt --steps 3500 --batch_size 12 --grad_accum_steps 4 --run_dir runs\stage5_openslr_polish

Write-Host "=========================================================="
Write-Host "3. Dynamically Launching Final Benchmark Evaluation on RESPIN Test"
Write-Host "   (Outputting to inference/evaluation2)"
Write-Host "=========================================================="
New-Item -ItemType Directory -Force -Path inference\evaluation2
.venv311\Scripts\python.exe inference\pipeline.py --checkpoint runs\stage5_openslr_polish\checkpoints\stage5_grpo_polish\best_model.pt --output_dir inference\evaluation2

Write-Host "Pipeline Complete! All models evaluated and compared!"
