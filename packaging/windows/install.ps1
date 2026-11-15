<#
.SYNOPSIS
Installs the Strix CUDA package from this extracted ZIP.

.DESCRIPTION
Copies bin\ and share\ to -Prefix (default: per-user, no administrator rights needed), creates the
configuration directory (default %ProgramData%\Strix) with serve.toml and serve.args unless they already exist
(an existing configuration is never overwritten), and optionally registers a "Strix Server" scheduled task that
starts strix-start.ps1 at logon of the installing user. Nothing is downloaded; no model is installed.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\install.ps1 -AddToPath -RegisterTask
#>
[CmdletBinding()]
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\Strix'),
    [string]$ConfigDir = (Join-Path $env:ProgramData 'Strix'),
    [switch]$AddToPath,
    [switch]$RegisterTask
)
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
if (-not (Test-Path (Join-Path $src 'bin'))) { throw "install.ps1: run it from the extracted package (no bin\ next to it)" }

New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
foreach ($dir in 'bin', 'share') {
    if (Test-Path (Join-Path $src $dir)) { Copy-Item -Recurse -Force -Path (Join-Path $src $dir) -Destination $Prefix }
}
Copy-Item -Force -Path (Join-Path $src '*.ps1') -Destination $Prefix

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
foreach ($name in 'serve.toml', 'serve.args', 'devices.example.toml') {
    $dst = Join-Path $ConfigDir $name
    if (Test-Path $dst) { Write-Host "kept existing $dst" }
    else { Copy-Item -Path (Join-Path $src "etc\strix\$name") -Destination $dst; Write-Host "wrote $dst" }
}

if ($AddToPath) {
    $bin = Join-Path $Prefix 'bin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($userPath -split ';') -contains $bin)) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ';' + $bin).TrimStart(';')), 'User')
        Write-Host "added $bin to the user PATH (new shells)"
    }
}

if ($RegisterTask) {
    $start = Join-Path $Prefix 'strix-start.ps1'
    $arg = '-NoProfile -ExecutionPolicy Bypass -File "' + $start + '" -ConfigDir "' + $ConfigDir + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $Prefix
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries
    Register-ScheduledTask -TaskName 'Strix Server' -Action $action -Trigger $trigger -Settings $settings `
        -Description 'Strix inference server (strix-start.ps1)' -Force | Out-Null
    Write-Host "registered scheduled task 'Strix Server' (at logon of $env:USERNAME; not started now)"
}

$receipt = @("prefix: $Prefix", "config: $ConfigDir")
$build = Get-ChildItem -Path (Join-Path $Prefix 'share\strix\support') -Filter 'build-*.txt' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($build) { $receipt += Get-Content $build.FullName -TotalCount 4 }
Set-Content -Path (Join-Path $Prefix 'install.txt') -Value $receipt
Write-Host "installed to $Prefix; edit $ConfigDir\serve.args (model paths), then run strix-start.ps1"
Write-Host "smoke test: powershell -ExecutionPolicy Bypass -File `"$Prefix\smoke.ps1`""
