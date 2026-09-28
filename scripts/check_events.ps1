$cut = (Get-Date).AddHours(-4)
Write-Output "=== System power/sleep events (last 4h) ==="
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$cut} -ErrorAction SilentlyContinue |
  Where-Object { $_.Id -in 41,42,1074,6008,1073 } |
  Select-Object -First 12 TimeCreated, Id, @{N='Msg';E={$_.Message.Substring(0, [Math]::Min(140, $_.Message.Length))}} |
  ForEach-Object { Write-Output ($_.TimeCreated.ToString('HH:mm:ss') + ' | ID ' + $_.Id + ' | ' + $_.Msg) }

Write-Output ""
Write-Output "=== Checkpoints ==="
Get-ChildItem 'D:\marathi-asr\runs\no-splitformer-moe-only\checkpoints\stage1_pretrain' -ErrorAction SilentlyContinue |
  ForEach-Object { Write-Output ($_.Name + ' | ' + $_.LastWriteTime.ToString('HH:mm:ss') + ' | ' + [math]::Round($_.Length/1MB,1) + ' MB') }