@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0
nvcc -DNOMINMAX -DWIN32_LEAN_AND_MEAN -O3 --generate-code=arch=compute_120a,code=sm_120a --fmad=false -prec-div=true -prec-sqrt=true --ftz=false -std=c++20 -Xcompiler "/fp:strict /EHsc /O2 /Zc:preprocessor /utf-8" -I..\..\.. ^
    fp32_identity.cu mm_rtx_kernels.cu mm_rtx_prefill_main.cu mm_rtx_pack.cpp mm_rtx_weights.cpp ^
    ..\..\core\dense\dense.cu ..\..\core\dense\dense_mma.cu ..\..\core\dense\nvfp4.cu ^
    ..\..\..\bp_codec.cpp ..\..\..\handoff.cpp -I..\..\..\..\include -lws2_32 -o mm-rtx-prefill.exe
