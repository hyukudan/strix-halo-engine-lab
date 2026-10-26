@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0
nvcc -O3 -arch=sm_120 --fmad=false -prec-div=true -prec-sqrt=true -std=c++17 -Xcompiler "/fp:strict /EHsc" -I..\..\.. fp32_identity.cu mm_fp32_identity_tests.cu -o mm_fp32_identity_tests.exe
