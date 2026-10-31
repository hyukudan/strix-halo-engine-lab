param([Parameter(Mandatory=$true)][string]$BuildDir,[switch]$Fresh,
    [Parameter(Mandatory=$true)][string]$TierstreamSource,
    [Parameter(Mandatory=$true)][string]$TierstreamBuild,
    [Parameter(Mandatory=$true)][string]$CoordinationFile)
. (Join-Path $PSScriptRoot 'pc_idle.ps1')
$ErrorActionPreference = 'Stop'
$build = [IO.Path]::GetFullPath($BuildDir)
$tierBuild = [IO.Path]::GetFullPath($TierstreamBuild)
$tierSource = [IO.Path]::GetFullPath($TierstreamSource)
if ($Fresh -and ((Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt')) -or (Test-Path -LiteralPath (Join-Path $tierBuild 'CMakeCache.txt')))) { throw '-Fresh requires new GLM and tierstream directories' }
if ((git -C $tierSource rev-parse HEAD) -ne '873b00453cec123ac9b7a8a8ebee7e60e20c643c') { throw 'Wrong tierstream pin' }
if ((git -C $tierSource symbolic-ref -q HEAD 2>$null)) { throw 'Tierstream pin must be detached' }
if (git -C $tierSource status --porcelain) { throw 'Tierstream checkout must remain clean/read-only' }
New-Item -ItemType Directory -Force $tierBuild | Out-Null
New-Item -ItemType Directory -Force $build,(Join-Path $build 'tmp') | Out-Null
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$freshOption = if ($Fresh) { '--fresh' } else { '' }
$lines = @('@echo off', 'set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"', "set `"TMP=$build\tmp`"", "set `"TEMP=$build\tmp`"", 'set "CCACHE_DISABLE=1"', 'set "VSLANG=1033"',
    "call `"$vs`" >nul", 'if errorlevel 1 exit /b 1', 'chcp 65001 >nul',
    "cmake $freshOption -S `"$PSScriptRoot`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=`"$($cuda -replace '\\','/')/bin/nvcc.exe`" -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_CUDA_COMPILER_LAUNCHER= -DTIERSTREAM_SOURCE=`"$($tierSource -replace '\\','/')`" -DTIERSTREAM_LIBRARY=`"$($tierBuild -replace '\\','/')/tierstream.lib`"",
    'if errorlevel 1 exit /b 1', "cmake --build `"$build`" --parallel 2", 'exit /b %errorlevel%')
$script = Join-Path $build 'producer-build.cmd'
$lines | Set-Content -LiteralPath $script -Encoding ascii
$process = Get-Process -Id $PID
$oldPriority = $process.PriorityClass
try {
    $process.PriorityClass = 'Idle'
    Wait-PCIdle -CoordinationFile $CoordinationFile
    $tierScript = Join-Path $tierBuild 'tierstream-build.cmd'
    @('@echo off', 'set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"', 'chcp 65001 >nul', "call `"$vs`" >nul", 'if errorlevel 1 exit /b 1', 'set "CCACHE_DISABLE=1"', 'set "VSLANG=1033"',
      "set `"TMP=$build\tmp`"", "set `"TEMP=$build\tmp`"",
      "cmake $freshOption -S `"$tierSource`" -B `"$tierBuild`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_CXX_COMPILER_LAUNCHER= -DTS_CUDA=OFF -DTS_HIP=OFF -DTS_TESTS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL",
      'if errorlevel 1 exit /b 1', "cmake --build `"$tierBuild`" --target tierstream --parallel 2", 'exit /b %errorlevel%') | Set-Content -LiteralPath $tierScript -Encoding ascii
    & cmd /d /c $tierScript 2>&1 | Tee-Object (Join-Path $tierBuild 'tierstream-build.log')
    if ($LASTEXITCODE -ne 0) { throw 'Tierstream build failed' }
    Wait-PCIdle -CoordinationFile $CoordinationFile
    & cmd /d /c $script 2>&1 | Tee-Object (Join-Path $build 'producer-build.log')
    if ($LASTEXITCODE -ne 0) { throw 'Producer build failed' }
    # Compiler artifacts only. Never load a service/benchmark or CUDA runtime.
    & python -B (Join-Path $PSScriptRoot 'inspect_build.py') --build $build --cuobjdump (Join-Path $cuda 'bin/cuobjdump.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Producer/bench SASS check failed' }
} finally { $process.PriorityClass = $oldPriority }
Get-FileHash (Join-Path $build 'glm_native_service.exe'),(Join-Path $build 'glm_native_integrity.exe') -Algorithm SHA256
