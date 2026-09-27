param([string]$Target)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($Target)) {
    $Target = Read-Host 'Путь к корню LRC_Extandator'
}
$Target = [System.IO.Path]::GetFullPath($Target.Trim('"'))
if (-not (Test-Path (Join-Path $Target 'app.py'))) {
    throw "В $Target не найден app.py. Укажи корень репозитория LRC_Extandator."
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $Target "_v6_backup_$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null

$files = @(
  'app.py','lrc_maker.py','lrc_formats.py','alignment_engine.py','alignment_quality.py','alignment_cache.py','benchmark.py',
  'requirements.txt','requirements-experimental.txt','start.ps1','install.bat','launcher.bat','run.bat','setup.bat','run_tests.bat',
  'README_V6.md','PATCH_README.txt',
  'static\v6-enhancer.js','static\v6-enhancer.css',
  'tests\test_lrc_formats.py','tests\test_alignment_quality.py','tests\test_anchor_alignment.py','tests\test_benchmark.py',
  'benchmarks\README.md','benchmarks\manifest.example.json'
)

foreach ($relative in $files) {
    $src = Join-Path $PatchRoot $relative
    if (-not (Test-Path $src)) { throw "В patch-архиве отсутствует $relative" }
    $dst = Join-Path $Target $relative
    if (Test-Path $dst) {
        $bak = Join-Path $backup $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $bak) -Force | Out-Null
        Copy-Item $dst $bak -Force
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
    Copy-Item $src $dst -Force
}

Write-Host ''
Write-Host 'V6 файлы применены.' -ForegroundColor Green
Write-Host "Backup: $backup"
Write-Host ''
Write-Host 'Теперь запусти install.bat в репозитории.' -ForegroundColor Cyan
