PCB Thermal Drift Package

What is included
- pcb_thermal_drift_prototype.py
- run_pcb_drift_auto.bat
- requirements.txt
- pcb_input_template.csv
- pcb_drift_outputs\pcb_drift_regression_model.joblib
- pcb_drift_outputs\pcb_drift_classifier_model.joblib
- pcb_drift_outputs\pcb_drift_report.html
- pcb_drift_outputs\pcb_demo_scenario_predictions.csv
- pcb_drift_outputs\simulated_pcb_thermal_drift_data.csv

How to use on another computer
1. Install Python 3.11 or newer.
2. Open Command Prompt in this folder.
3. Run:
   python -m pip install -r requirements.txt
4. Run:
   run_pcb_drift_auto.bat

How to score your own boards automatically
1. Open pcb_input_template.csv
2. Fill in your board rows
3. Save a copy named pcb_input.csv in the same folder
4. Run run_pcb_drift_auto.bat again

Where results are written
- pcb_drift_outputs\pcb_input_predictions.csv
- pcb_drift_outputs\pcb_drift_report.html

Notes
- The batch file tries py -3 first, then python on PATH.
- If Python is installed but not added to PATH, run the script manually with the full python.exe path.
