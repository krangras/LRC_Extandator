param(
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Fail([string]$Message, [int]$Code = 1) {
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit $Code
}

function Test-CompatiblePython([string]$File, [string[]]$PrefixArgs = @()) {
    try {
        $code = 'import sys; ok=(3,10) <= sys.version_info[:2] < (3,13); print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"); raise SystemExit(0 if ok else 1)'
        $output = & $File @PrefixArgs '-c' $code 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Compatible = $true; Version = ($output | Select-Object -Last 1) }
        }
        return [pscustomobject]@{ Compatible = $false; Version = ($output | Select-Object -Last 1) }
    }
    catch {
        return [pscustomobject]@{ Compatible = $false; Version = $null }
    }
}

function Get-CompatibleSystemPython {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        foreach ($minor in @('3.12', '3.11', '3.10')) {
            $test = Test-CompatiblePython $py.Source @("-$minor")
            if ($test.Compatible) {
                return [pscustomobject]@{ File = $py.Source; Args = @("-$minor"); Version = $test.Version }
            }
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $test = Test-CompatiblePython $python.Source @()
        if ($test.Compatible) {
            return [pscustomobject]@{ File = $python.Source; Args = @(); Version = $test.Version }
        }
    }

    return $null
}

function Invoke-Python($Python, [string[]]$Arguments) {
    $allArgs = @()
    $allArgs += @($Python.Args)
    $allArgs += @($Arguments)
    & $Python.File $allArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Get-VenvVersion([string]$PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $null
    }
    try {
        $v = & $PythonPath '-c' 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($v | Select-Object -Last 1)
    }
    catch { return $null }
}

function Test-VersionStringCompatible([string]$Version) {
    if ([string]::IsNullOrWhiteSpace($Version)) { return $false }
    try {
        $parts = $Version.Split('.')
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        return ($major -eq 3 -and $minor -ge 10 -and $minor -le 12)
    }
    catch { return $false }
}

$VenvDir = Join-Path $Root '.venv'
$VenvPythonPath = Join-Path $VenvDir 'Scripts\python.exe'
$RequirementsPath = Join-Path $Root 'requirements.txt'
$AppPath = Join-Path $Root 'app.py'

if ($Install) {
    $systemPython = Get-CompatibleSystemPython
    if ($null -eq $systemPython) {
        Write-Host 'This project requires Python 3.10, 3.11, or 3.12.' -ForegroundColor Yellow
        Write-Host 'Python 3.13/3.14 cannot install py-roller 0.8.x.' -ForegroundColor Yellow
        Write-Host ''
        Write-Host 'Recommended installation command:'
        Write-Host '  winget install -e --id Python.Python.3.12'
        Write-Host ''
        Write-Host 'After Python 3.12 is installed, run install.bat again.'
        exit 2
    }

    Write-Host "Using Python $($systemPython.Version)"

    $existingVersion = Get-VenvVersion $VenvPythonPath
    if ($null -ne $existingVersion -and -not (Test-VersionStringCompatible $existingVersion)) {
        Write-Host "Existing .venv uses incompatible Python $existingVersion. Recreating it..." -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $VenvPythonPath -PathType Leaf)) {
        Write-Host 'Creating virtual environment...'
        Invoke-Python $systemPython @('-m', 'venv', $VenvDir)
    }

    if (-not (Test-Path -LiteralPath $VenvPythonPath -PathType Leaf)) {
        Fail 'Virtual environment was not created.' 3
    }

    $venvPython = [pscustomobject]@{ File = $VenvPythonPath; Args = @() }
    Write-Host 'Updating pip/setuptools/wheel...'
    Invoke-Python $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')

    if (Test-Path -LiteralPath $RequirementsPath -PathType Leaf) {
        Write-Host 'Installing project dependencies...'
        Invoke-Python $venvPython @('-m', 'pip', 'install', '-r', $RequirementsPath)
    }
    else {
        Fail 'requirements.txt was not found.' 4
    }

    Write-Host ''
    Write-Host 'Installation completed successfully.' -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf)) {
    Fail "app.py was not found in: $Root" 4
}

if (-not (Test-Path -LiteralPath $VenvPythonPath -PathType Leaf)) {
    Fail 'Virtual environment is missing. Run install.bat first.' 5
}

$venvVersion = Get-VenvVersion $VenvPythonPath
if (-not (Test-VersionStringCompatible $venvVersion)) {
    Fail "The existing .venv uses incompatible Python $venvVersion. Run install.bat to recreate it with Python 3.12." 6
}

$runtimePython = [pscustomobject]@{ File = $VenvPythonPath; Args = @() }
Write-Host "Project directory: $Root"
Write-Host "Python: $venvVersion"
Write-Host 'Checking Python syntax...'
Invoke-Python $runtimePython @('-m', 'py_compile', $AppPath)

Write-Host 'Starting LRC Extandator...'
& $runtimePython.File $AppPath
$rc = $LASTEXITCODE
if ($null -eq $rc) { $rc = 0 }
exit $rc
