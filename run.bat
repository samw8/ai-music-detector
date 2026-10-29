@echo off
rem One-word launcher so nobody has to remember where Python lives.
rem   run setup                    install everything (first time only)
rem   run check "C:\Music\song.mp3"   check one song or a folder
rem   run all                      re-run the whole research pipeline, start to finish
rem   run <step>                   one step: download, prepare, tag, features, embed, melspec,
rem                                train, baselines, evaluate, leakage, robustness, fingerprints,
rem                                harmony, modelcard, report, demo
setlocal
set "HERE=%~dp0"
if "%AIMD_HOME%"=="" set "AIMD_HOME=%USERPROFILE%\.aimd"
set "PY=%AIMD_HOME%\venv\Scripts\python.exe"
set "HF_HOME=%AIMD_HOME%\hf"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"

if /i "%~1"=="setup" goto setup
if not exist "%PY%" (
  echo Python environment not found. Run:  run setup
  exit /b 1
)
if /i "%~1"=="check"        ( "%PY%" -W ignore "%HERE%predict.py" %2 %3 %4 & exit /b )
if /i "%~1"=="download"     ( "%PY%" "%HERE%scripts\download.py" %2 %3 %4 & exit /b )
if /i "%~1"=="prepare"      ( "%PY%" "%HERE%scripts\prepare.py" %2 %3 & exit /b )
if /i "%~1"=="tag"          ( "%PY%" -W ignore "%HERE%scripts\tag.py" & exit /b )
if /i "%~1"=="features"     ( "%PY%" "%HERE%scripts\extract_features.py" %2 & exit /b )
if /i "%~1"=="embed"        ( "%PY%" -W ignore "%HERE%scripts\embed.py" & exit /b )
if /i "%~1"=="melspec"      ( "%PY%" "%HERE%scripts\melspec.py" & exit /b )
if /i "%~1"=="train"        ( "%PY%" -W ignore "%HERE%scripts\run_experiments.py" %2 %3 %4 & exit /b )
if /i "%~1"=="features-mp3" ( "%PY%" "%HERE%scripts\extract_features.py" --codec mp3 & exit /b )
if /i "%~1"=="train-mp3"    ( "%PY%" -W ignore "%HERE%scripts\run_experiments.py" --variant mp3 & exit /b )
if /i "%~1"=="modelcard"    ( "%PY%" -W ignore "%HERE%scripts\write_model_card.py" & exit /b )
if /i "%~1"=="report"       ( "%PY%" -W ignore "%HERE%scripts\make_report.py" & exit /b )
if /i "%~1"=="harmony"      ( "%PY%" -W ignore "%HERE%scripts\harmony_probe.py" & exit /b )
if /i "%~1"=="test"         ( "%PY%" "%HERE%tests\test_splits.py" & exit /b )
if /i "%~1"=="baselines"    ( "%PY%" -W ignore "%HERE%scripts\baseline_pretrained.py" & exit /b )
if /i "%~1"=="evaluate"     ( "%PY%" -W ignore "%HERE%scripts\evaluate.py" & exit /b )
if /i "%~1"=="leakage"      ( "%PY%" -W ignore "%HERE%scripts\leakage_check.py" & exit /b )
if /i "%~1"=="robustness"   ( "%PY%" -W ignore "%HERE%scripts\robustness.py" %2 %3 & exit /b )
if /i "%~1"=="fingerprints" ( "%PY%" -W ignore "%HERE%scripts\fingerprints.py" & exit /b )
if /i "%~1"=="demo"         ( "%PY%" -W ignore "%HERE%scripts\demo.py" & exit /b )
if /i "%~1"=="all" (
  call "%~f0" download || exit /b 1
  for %%S in (prepare embed tag features melspec leakage train features-mp3 train-mp3 baselines robustness fingerprints harmony evaluate modelcard report) do (
    echo.
    echo ===== %%S =====
    call "%~f0" %%S || exit /b 1
  )
  exit /b
)
echo Unknown command "%~1". See the top of run.bat or README.md.
exit /b 1

:setup
rem Uses uv (https://docs.astral.sh/uv/) to install a private Python 3.12 that does not touch the rest of the PC.
where uv >nul 2>nul || powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "UV_PYTHON_INSTALL_DIR=%AIMD_HOME%\python"
uv venv "%AIMD_HOME%\venv" --python 3.12 --managed-python || exit /b 1
uv pip install --python "%PY%" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu || exit /b 1
uv pip install --python "%PY%" -r "%HERE%requirements.txt" || exit /b 1
uv pip install --python "%PY%" --no-deps "git+https://github.com/awsaf49/sonics.git@9156ffad151f797c71556923c4a02fa01fa8fc91"
echo Setup finished.
exit /b
