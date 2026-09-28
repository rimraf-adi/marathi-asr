# 30-min monitoring snapshot for marathi-asr stage1_pretrain
$ErrorActionPreference = "Continue"
$root = "D:\marathi-asr"
$run = "$root\runs\no-splitformer-moe-only"

Write-Output ("[MONITOR " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "]")
Write-Output "--- git ---"
Write-Output (git -C $root log -1 --format="%h %s")

Write-Output "--- python procs ---"
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) { $py | Select-Object Id, @{N='MB';E={[math]::Round($_.WorkingSet64/1MB)}} | Format-Table -AutoSize | Out-String | Write-Output } else { Write-Output "NO python processes - TRAINING DEAD" }

Write-Output "--- gpu ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | Out-String | Write-Output

$latest_telemetry = Get-ChildItem "$run\logs\*telemetry*.jsonl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest_telemetry) {
    Write-Output ("--- telemetry tail (" + $latest_telemetry.Name + ") ---")
    Get-Content $latest_telemetry.FullName -Tail 3 -ErrorAction SilentlyContinue | Out-String | Write-Output
} else {
    Write-Output "--- telemetry tail: none ---"
}

$latest_log = Get-ChildItem "$run\logs\*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest_log) {
    Write-Output ("--- log tail (" + $latest_log.Name + ") ---")
    Get-Content $latest_log.FullName -Tail 15 -ErrorAction SilentlyContinue | Out-String | Write-Output
} else {
    Write-Output "--- log tail: none ---"
}

Write-Output "--- chunks ---"
$chunks = Get-ChildItem "$root\data_cache\pretrain" -Filter "chunk_*.pt" -ErrorAction SilentlyContinue
if ($chunks) {
    $sum = ($chunks | Measure-Object -Property Length -Sum).Sum
    Write-Output ("chunks: " + $chunks.Count + " | size: " + [math]::Round($sum/1GB,2) + " GB")
} else { Write-Output "no chunks" }

Write-Output "--- disk free ---"
$d = Get-PSDrive D
Write-Output ("D: free " + [math]::Round($d.Free/1GB,2) + " GB")

Write-Output "--- pipeline status ---"
if (Test-Path "$run\pipeline_status.json") { Get-Content "$run\pipeline_status.json" | Out-String | Write-Output } else { Write-Output "no status file" }

Write-Output "--- telegram update ---"
# Automated telegram updates are handled asynchronously by the agent cronjob using log analysis (no template msgs)
# python "$root\send_telegram_update.py" | Out-String | Write-Output

Write-Output "--- git auto-push ---"
try {
    git -C "$root" add runs/
    $st = git -C "$root" status --porcelain runs/
    if ($st) {
        $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        git -C "$root" commit -m "chore(runs): sync training telemetry and logs [$now]"
        git -C "$root" push origin main
        Write-Output "Git pushed run updates to GitHub successfully."
    } else {
        Write-Output "No changes in runs/ to commit."
    }
} catch {
    Write-Output "Git push error: $_"
}