function Get-GpuBusyEvent {
    param([Parameter(Mandatory=$true)][string]$CoordinationFile)
    # Quoted references in prose and ACKs are not start/end events.
    $last = Get-Content -LiteralPath $CoordinationFile |
        Select-String '^[-*] \d{2}:[\dx]{2} [^:]+: (\*\*)?GPU-BUSY (start|end)\b(?!/)' |
        Select-Object -Last 1
    if (!$last) { throw 'No GPU-BUSY start/end event' }
    return $last.Line
}
function Wait-PCIdle {
    param([Parameter(Mandatory=$true)][string]$CoordinationFile)
    do {
        $last = Get-GpuBusyEvent -CoordinationFile $CoordinationFile
        Write-Host $last
        if ($last -match 'GPU-BUSY start\b') { Start-Sleep -Seconds 60; Start-Sleep -Seconds 60 }
    } while ($last -match 'GPU-BUSY start\b')
}
