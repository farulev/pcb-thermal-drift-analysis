@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=C:\Users\Forev\AppData\Local\Programs\Python\Python311\python.exe"
set "SCRIPT_PATH=%SCRIPT_DIR%pcb_thermal_drift_prototype.py"
set "OUTPUT_DIR=%SCRIPT_DIR%pcb_drift_outputs"

"%PYTHON_EXE%" "%SCRIPT_PATH%" --output-dir "%OUTPUT_DIR%"

echo.
echo Output folder: %OUTPUT_DIR%
echo To score your own boards automatically, place pcb_input.csv next to this BAT file and run it again.
