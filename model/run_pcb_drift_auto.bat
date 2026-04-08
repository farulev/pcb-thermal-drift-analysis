@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%pcb_thermal_drift_prototype.py"
set "OUTPUT_DIR=%SCRIPT_DIR%pcb_drift_outputs"
pushd "%SCRIPT_DIR%"

if exist "C:\Users\Forev\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\Forev\AppData\Local\Programs\Python\Python311\python.exe" "%SCRIPT_PATH%" --output-dir "%OUTPUT_DIR%"
    goto :done
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%SCRIPT_PATH%" --output-dir "%OUTPUT_DIR%"
    goto :done
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%SCRIPT_PATH%" --output-dir "%OUTPUT_DIR%"
    goto :done
)

echo Python was not found.
echo Install Python 3.11+ and then run:
echo     python -m pip install -r "%SCRIPT_DIR%requirements.txt"
popd
exit /b 1

:done
popd

echo.
echo Output folder: %OUTPUT_DIR%
echo To score your own boards automatically, place pcb_input.csv next to this BAT file and run it again.
