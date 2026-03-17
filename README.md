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
