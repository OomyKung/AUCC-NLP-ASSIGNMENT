<#
.SYNOPSIS
    Start the Thai News Intelligence Dashboard: backend API and frontend together.

.DESCRIPTION
    One command to get the whole system running, including first-time setup.

    It is idempotent: it creates the virtualenv, installs dependencies and seeds
    the database only when those things are missing, so the second run is fast.

    The API and the dev server each get their own window so their logs stay
    readable. Closing this window, or pressing Ctrl+C in it, stops both.

.PARAMETER Reset
    Rebuild the database from scratch before starting.

.PARAMETER SkipInstall
    Skip the dependency checks. Use when you know everything is installed.

.PARAMETER NoBrowser
    Do not open the dashboard in a browser.

.PARAMETER Quiet
    Keep the API and dev server logs in this window's child processes without
    opening new windows (useful over SSH or in a terminal multiplexer).

.EXAMPLE
    .\run.ps1
    First run: sets everything up, then starts both servers.

.EXAMPLE
    .\run.ps1 -Reset
    Rebuild the database, then start.

.NOTES
    If PowerShell refuses to run this file ("running scripts is disabled on this
    system"), use run.cmd instead - it bypasses the policy for this one script
    without changing any machine setting.
#>

[CmdletBinding()]
param(
    [switch]$Reset,
    [switch]$SkipInstall,
    [switch]$NoBrowser,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$VenvPython = Join-Path $Backend '.venv\Scripts\python.exe'
$Database = Join-Path $Backend 'thai_news.db'

$ApiUrl = 'http://127.0.0.1:8000'
$WebUrl = 'http://localhost:5173'

# Child processes, stopped in the finally block no matter how we exit.
$script:Children = @()

function Write-Step($message) {
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Note($message) {
    Write-Host "    $message" -ForegroundColor DarkGray
}

function Write-Problem($message) {
    Write-Host "!!! $message" -ForegroundColor Red
}

function Assert-Tool($name, $versionArgs, $help) {
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $found) {
        Write-Problem "$name was not found on PATH."
        Write-Host "    $help"
        exit 1
    }
    $version = (& $name $versionArgs 2>&1 | Select-Object -First 1)
    Write-Note "$name $version"
}

function Wait-ForUrl($url, $label, $timeoutSeconds) {
    # Poll rather than sleep a fixed amount: uvicorn's first start has to build
    # the Thai tokeniser trie and load the models, which is slow on a cold disk
    # but fast afterwards. A fixed sleep is either too short or wasteful.
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
            # Not up yet. Keep waiting.
        }
        # Surface a dead child immediately instead of waiting out the timeout.
        foreach ($child in $script:Children) {
            if ($child -and $child.HasExited) {
                Write-Problem "$label process exited early (code $($child.ExitCode)). Check its window for the error."
                return $false
            }
        }
        Start-Sleep -Milliseconds 700
    }
    Write-Problem "$label did not come up within $timeoutSeconds seconds."
    return $false
}

function Start-Child($workingDirectory, $file, $arguments, $title) {
    if ($Quiet) {
        $process = Start-Process -FilePath $file -ArgumentList $arguments `
            -WorkingDirectory $workingDirectory -PassThru -NoNewWindow
    } else {
        # A separate window per server, so their logs do not interleave.
        $inner = "`$host.UI.RawUI.WindowTitle = '$title'; & '$file' $arguments"
        $process = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList '-NoExit', '-NoProfile', '-Command', $inner `
            -WorkingDirectory $workingDirectory -PassThru
    }
    $script:Children += $process
    return $process
}

function Stop-Children {
    foreach ($child in $script:Children) {
        if ($child -and -not $child.HasExited) {
            # Kill the tree: uvicorn --reload and vite both spawn workers that
            # would survive (and hold ports 8000/5173) if only the parent died.
            & taskkill.exe /PID $child.Id /T /F 2>&1 | Out-Null
        }
    }
}

try {
    Write-Host ''
    Write-Host 'Thai News Intelligence Dashboard' -ForegroundColor White
    Write-Host '--------------------------------' -ForegroundColor DarkGray

    # ---------------------------------------------------------------- checks
    Write-Step 'Checking tools'
    Assert-Tool 'python' '--version' 'Install Python 3.11+ from https://python.org and tick "Add to PATH".'
    Assert-Tool 'node' '--version' 'Install Node.js 20+ from https://nodejs.org.'

    # ------------------------------------------------------------ dependencies
    if (-not $SkipInstall) {
        if (-not (Test-Path $VenvPython)) {
            Write-Step 'Creating the Python virtualenv (first run only)'
            & python -m venv (Join-Path $Backend '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'could not create the virtualenv' }
        }

        # Cheap marker: fastapi importing means requirements are in place.
        & $VenvPython -c 'import fastapi' 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Step 'Installing backend dependencies (first run only, ~2 minutes)'
            & $VenvPython -m pip install --quiet --upgrade pip
            & $VenvPython -m pip install -r (Join-Path $Backend 'requirements.txt')
            if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
        } else {
            Write-Note 'backend dependencies present'
        }

        if (-not (Test-Path (Join-Path $Frontend 'node_modules'))) {
            Write-Step 'Installing frontend dependencies (first run only, ~1 minute)'
            # npm.cmd, not npm: the .ps1 shim is blocked by the default
            # execution policy, which is the error most people hit here.
            Push-Location $Frontend
            try {
                & npm.cmd install
                if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
            } finally {
                Pop-Location
            }
        } else {
            Write-Note 'frontend dependencies present'
        }
    }

    # --------------------------------------------------------------- database
    if ($Reset -or -not (Test-Path $Database)) {
        if ($Reset) {
            Write-Step 'Rebuilding the database (--reset)'
            $seedArgs = @('seed.py', '--reset')
        } else {
            Write-Step 'Seeding the database (first run only, ~60 seconds)'
            $seedArgs = @('seed.py')
        }
        Write-Note '25,928 real Thai chat messages into 116 analysed windows. No network needed.'
        Push-Location $Backend
        try {
            & $VenvPython $seedArgs
            if ($LASTEXITCODE -ne 0) { throw 'seeding failed' }
        } finally {
            Pop-Location
        }
    } else {
        Write-Note 'database present (use -Reset to rebuild)'
    }

    # ---------------------------------------------------------------- servers
    Write-Step "Starting the API on $ApiUrl"
    Start-Child $Backend $VenvPython '-m uvicorn app.main:app --reload --port 8000' 'API - uvicorn' | Out-Null
    if (-not (Wait-ForUrl "$ApiUrl/api/health" 'API' 120)) { exit 1 }
    Write-Note 'API is up'

    Write-Step "Starting the dashboard on $WebUrl"
    Start-Child $Frontend 'npm.cmd' 'run dev' 'Dashboard - vite' | Out-Null
    if (-not (Wait-ForUrl $WebUrl 'Dashboard' 90)) { exit 1 }
    Write-Note 'dashboard is up'

    Write-Host ''
    Write-Host '  Dashboard : ' -NoNewline; Write-Host $WebUrl -ForegroundColor Green
    Write-Host '  API       : ' -NoNewline; Write-Host $ApiUrl -ForegroundColor Green
    Write-Host '  API docs  : ' -NoNewline; Write-Host "$ApiUrl/docs" -ForegroundColor Green
    Write-Host ''

    if (-not $NoBrowser) {
        Start-Process $WebUrl | Out-Null
    }

    Write-Host 'Both servers are running. Press Ctrl+C here to stop them.' -ForegroundColor Yellow
    Write-Host ''

    # Hold the window open, and notice if either server dies on its own.
    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($child in $script:Children) {
            if ($child -and $child.HasExited) {
                Write-Problem "A server stopped (exit code $($child.ExitCode)). Shutting down the other."
                exit 1
            }
        }
    }
} finally {
    Write-Host ''
    Write-Step 'Stopping servers'
    Stop-Children
    Write-Note 'done'
}
