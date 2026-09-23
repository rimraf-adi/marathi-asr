$files = Get-ChildItem D:\huggingface_cache -Recurse -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 20
foreach ($f in $files) {
    Write-Output ($f.FullName + "  " + [math]::Round($f.Length/1MB, 2) + " MB")
}
Write-Output "---"
Get-ChildItem D:\huggingface_cache\datasets -Recurse -Depth 3 -ErrorAction SilentlyContinue | Select-Object FullName | ForEach-Object { $_.FullName }