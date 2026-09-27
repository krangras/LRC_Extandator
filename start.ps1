param(
    [switch]$Install,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$VenvDir = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements.txt'
$App = Join-Path $Root 'app.py'

function Fail([string]$Message, [int]$Code = 1) {
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit $Code
}

function Test-Python([string]$Exe, [string[]]$Prefix = @()) {
    try {
        $out = & $Exe @Prefix '-c' 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"); raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)' 2>$null
        return [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Version = ($out | Select-Object -Last 1); Exe = $Exe; Prefix = $Prefix }
    } catch {
        return [pscustomobject]@{ Ok = $false; Version = ''; Exe = $Exe; Prefix = $Prefix }
    }
}

function Find-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        foreach ($minor in @('3.14','3.13','3.12','3.11','3.10')) {
            $test = Test-Python $py.Source @("-$minor")
            if ($test.Ok) { return $test }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $test = Test-Python $python.Source
        if ($test.Ok) { return $test }
    }
    return $null
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

function Has-NvidiaGpu {
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) { return $false }
    try {
        & $nvidia.Source '--query-gpu=name' '--format=csv,noheader' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

if ($Install) {
    $system = Find-Python
    if ($null -eq $system) {
        Write-Host 'Нужен Python 3.10-3.14 x64.' -ForegroundColor Yellow
        Write-Host 'Для твоей текущей системы подходит установленный Python 3.14.'
        exit 2
    }
    Write-Host "Python $($system.Version)" -ForegroundColor Cyan

    if (Test-Path $VenvPython) {
        $venvTest = Test-Python $VenvPython
        if (-not $venvTest.Ok) {
            Write-Host 'Старое окружение несовместимо, пересоздаю .venv...' -ForegroundColor Yellow
            Remove-Item $VenvDir -Recurse -Force
        }
    }
    if (-not (Test-Path $VenvPython)) {
        $venvArgs = @($system.Prefix) + @('-m','venv',$VenvDir)
        & $system.Exe @venvArgs
        if ($LASTEXITCODE -ne 0) { Fail 'Не удалось создать .venv' 3 }
    }

    Invoke-Checked $VenvPython @('-m','pip','install','--upgrade','pip','setuptools','wheel')

    # Torch ставим отдельно, чтобы RTX использовала CUDA и чтобы зависимости
    # Demucs не подменили его случайным CPU build.
    $TorchIndex = $env:LRC_TORCH_INDEX_URL
    if ([string]::IsNullOrWhiteSpace($TorchIndex)) {
        if (Has-NvidiaGpu) {
            $TorchIndex = 'https://download.pytorch.org/whl/cu128'
            Write-Host 'NVIDIA GPU найдена: ставлю PyTorch CUDA 12.8.' -ForegroundColor Green
        } else {
            $TorchIndex = 'https://download.pytorch.org/whl/cpu'
            Write-Host 'NVIDIA GPU не найдена: ставлю CPU PyTorch.' -ForegroundColor Yellow
        }
    }
    Invoke-Checked $VenvPython @('-m','pip','install','--index-url',$TorchIndex,'torch==2.11.0','torchaudio==2.11.0')
    Invoke-Checked $VenvPython @('-m','pip','install','-r',$Requirements)

    Write-Host 'Пробую установить опциональный Demucs...' -ForegroundColor Cyan
    & $VenvPython '-m' 'pip' 'install' '-r' (Join-Path $Root 'requirements-demucs.txt')
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Demucs не установился. Forced alignment продолжит работать по исходному mix; Demucs можно поставить позже.' -ForegroundColor Yellow
    }

    if (Get-Command npm.exe -ErrorAction SilentlyContinue) {
        Write-Host 'Устанавливаю track-dl...' -ForegroundColor Cyan
        npm install
        if ($LASTEXITCODE -ne 0) { Write-Host 'npm install не удался; локальный forced alignment всё равно будет работать.' -ForegroundColor Yellow }
    }

    Write-Host ''
    Write-Host 'Проверяю Forced Alignment runtime...' -ForegroundColor Cyan
    Invoke-Checked $VenvPython @('doctor.py','--preload')

    Write-Host ''
    Write-Host 'Запускаю тесты ядра...' -ForegroundColor Cyan
    Invoke-Checked $VenvPython @('-m','unittest','discover','-s','tests','-v')

    Write-Host ''
    Write-Host 'Установка Forced Alignment v7 завершена.' -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $VenvPython)) { Fail 'Нет .venv. Сначала запусти install.bat.' 5 }
if (-not (Test-Path $App)) { Fail 'app.py не найден.' 6 }

Invoke-Checked $VenvPython @('-m','py_compile','app.py','lrc_maker.py','lrc_formats.py','alignment_engine.py','alignment_quality.py','alignment_cache.py','doctor.py')

$env:LRC_STUDIO_HOST = if ($env:LRC_STUDIO_HOST) { $env:LRC_STUDIO_HOST } else { '127.0.0.1' }
$env:LRC_STUDIO_PORT = if ($env:LRC_STUDIO_PORT) { $env:LRC_STUDIO_PORT } else { '5000' }

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($Url)
        Start-Sleep -Seconds 2
        Start-Process $Url
    } -ArgumentList "http://$($env:LRC_STUDIO_HOST):$($env:LRC_STUDIO_PORT)" | Out-Null
}

& $VenvPython $App
$rc = $LASTEXITCODE
if ($null -eq $rc) { $rc = 0 }
exit $rc
