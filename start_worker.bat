@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON="
set "PYTHON_ARGS="

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=%ROOT%.venv\Scripts\python.exe"
    goto launch
  )
)
for %%V in (313 312 311) do (
  if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
    "%LocalAppData%\Programs\Python\Python%%V\python.exe" -c "import numpy" >nul 2>nul
    if not errorlevel 1 (
      set "PYTHON=%LocalAppData%\Programs\Python\Python%%V\python.exe"
      goto launch
    )
  )
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=py"
    set "PYTHON_ARGS=-3"
    goto launch
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=python"
    goto launch
  )
)
where python3 >nul 2>nul
if not errorlevel 1 (
  python3 -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=python3"
    goto launch
  )
)

echo Python 3.11-3.13 was not found.
echo Install the core with: python -m pip install -e "%ROOT%."
exit /b 2

:launch
"%PYTHON%" %PYTHON_ARGS% -B "%ROOT%worker.py" %*
exit /b %errorlevel%
