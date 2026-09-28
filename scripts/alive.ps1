Write-Output ("NOW: " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Output ("Telemetry last write: " + (Get-Item D:\marathi-asr\runs\no-splitformer-moe-only\logs\stage1_pretrain_telemetry.jsonl).LastWriteTime)
Get-Content D:\marathi-asr\runs\no-splitformer-moe-only\logs\stage1_pretrain_telemetry.jsonl -Tail 1 | Out-String | Write-Output
Write-Output "--- python procs ---"
$p = Get-Process python -ErrorAction SilentlyContinue
if ($p) {
  foreach ($x in $p) {
    Write-Output ("PID " + $x.Id + " | " + [math]::Round($x.WorkingSet64/1MB) + " MB | started " + $x.StartTime)
  }
} else { Write-Output "NONE" }
Write-Output "--- GPU compute apps ---"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | Out-String | Write-Output
Write-Output "--- GPU summary ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | Out-String | Write-Output