# PCB Thermal Drift Analysis

Author: Rodion Farulev

Exploratory research note on calibration coefficient drift (3–5%)
in electronic boards after thermal cycling.

## Problem
Electronic boards may experience deformation and calibration drift
after repeated thermal cycles.

## Goal
Explore whether machine learning models could predict hardware drift
based on test data.

## Contents
- report.pdf — technical exploration

- ## Possible ML Approach

Potential machine learning pipeline:

1. Collect thermal cycling test data
2. Extract features:
   - temperature gradient
   - cycle count
   - board material
3. Train regression model to predict calibration drift
Potential extension:
Use time-series models (e.g., LSTM) to analyze degradation over multiple cycles.
Using machine learning to predict calibration drift of electronic boards after thermal cycling based on environmental and material parameters.
## ML Approach

The goal is to predict calibration coefficient drift (3–5%) in electronic boards after thermal cycling.

The model would analyze relationships between environmental conditions and resulting hardware degradation.

### Input features (possible):

- Temperature range (min / max)
- Number of thermal cycles
- Heating / cooling rate
- Material properties of PCB
- Time under stress conditions

### Output:

- Predicted drift of calibration coefficient
- Probability of deviation beyond acceptable threshold

### Model type:

- Regression model (e.g., Random Forest / Neural Network)
- Optional anomaly detection for unexpected behavior

### Idea:

Train the model on experimental test data from thermal cycling and use it to predict future hardware degradation and reduce failure risks.

### Model

Use run_pcb_drift_auto.bat as the one-click launcher. It runs pcb_thermal_drift_prototype.py, trains the models, saves artifacts, writes an HTML report, and auto-scores your boards if it finds pcb_input.csv.

How to use it:

Run run_pcb_drift_auto.bat.
If you want to score real boards, fill in pcb_input_template.csv, save a copy as C:\Users\Forev\Documents\pcb_input.csv, and run the BAT file again.
Open the results in pcb_drift_report.html. Your scored boards will be in pcb_input_predictions.csv.
What it now does automatically:

simulates and trains the hard PCB drift models
saves dataset and .joblib models into pcb_drift_outputs
creates a ready-to-fill input template
auto-computes the advanced derived features from your CSV
predicts drift percent, exceedance probability, and risk band
writes an HTML report with metrics, charts, and scenario results
I verified both paths:

normal one-click run
auto-scoring using the template as input
