<#
.SYNOPSIS
Removes a Strix installation made by install.ps1: the scheduled task, the PATH entry and -Prefix.
The configuration directory (default %ProgramData%\Strix) and any model files are kept.
#>
[CmdletBinding()]
param([string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\Strix'))
$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName 'Strix Server' -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName 'Strix Server' -Confirm:$false
    Write-Host "removed scheduled task 'Strix Server'"
}
$bin = Join-Path $Prefix 'bin'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -and (($userPath -split ';') -contains $bin)) {
    [Environment]::SetEnvironmentVariable('Path', ((($userPath -split ';') | Where-Object { $_ -ne $bin }) -join ';'), 'User')
    Write-Host "removed $bin from the user PATH"
}
if (Test-Path (Join-Path $Prefix 'install.txt')) {
    Remove-Item -Recurse -Force -Path $Prefix -Confirm:$false
    Write-Host "removed $Prefix"
} else {
    Write-Host "no install.txt in ${Prefix}: not a Strix installation, left untouched"
}
