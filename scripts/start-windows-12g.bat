@echo off
REM ============================================================
REM  Local LLM service launcher -- 12 GB card / 16 GB RAM
REM  Machine-adapted: RTX 4070 SUPER (sm_89), 15.7 GB host RAM
REM
REM  EDIT THESE TWO PATHS BEFORE FIRST RUN:
REM    ENGINE = path to your OpenAI-compatible serve binary
REM    MODEL  = path to your model file
REM
REM  Why this config (details in ../README.md):
REM    * 128K context needs 4-bit KV (rk4v4); fp8 does not fit
REM    * host-side KV pinning reduced from 8 GiB default to 2048 MiB
REM      (default overflows 16 GB RAM -> cudaMallocHost OOM)
REM    * single concurrency: long requests own the engine
REM
REM  ASCII-only on purpose: batch files are parsed in the OEM/ANSI
REM  codepage; non-ASCII text corrupts commands. No CJK here.
REM ============================================================

REM  CUDA toolkit is NOT required (driver-side runtime).
REM  Set CUDA_BIN only if the engine dies with 0xC0000135.
set "CUDA_BIN="
if not defined CUDA_BIN (
  for /d %%D in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13*") do set "CUDA_BIN=%%~fD\bin"
)
if not defined CUDA_BIN (
  for /d %%D in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12*") do set "CUDA_BIN=%%~fD\bin"
)
if not defined CUDA_BIN set "CUDA_BIN=<PATH_TO_YOUR_ENGINE_DIR>"
set "PATH=%CUDA_BIN%;%CUDA_BIN%\x64;%PATH%"

set "ENGINE=<PATH_TO_ENGINE_EXE>"
set "MODEL=<PATH_TO_MODEL_FILE>"

echo.
echo   URL   : http://127.0.0.1:8085
echo   Ready : curl.exe -s -o NUL -w "%%{http_code}" http://127.0.0.1:8085/v1/models
echo   Expect: 200
echo.

"%ENGINE%" "%MODEL%" ^
  --host 127.0.0.1 --port 8085 --model-id <MODEL_ID> ^
  --max-context 131072 --kv-capacity 131072 --kv-dtype <YOUR_4BIT_KV_DTYPE> ^
  --max-concurrency 1 --prefill-chunk 1024 ^
  --host-kv-mib 2048 ^
  --speculative <YOUR_SPEC_MODE> --draft-tokens 3 ^
  --cors --tolerant-tool-calls --vision

echo.
echo [engine exited] errorlevel=%ERRORLEVEL%
pause