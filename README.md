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
