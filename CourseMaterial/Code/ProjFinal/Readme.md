# Project: Forecasting Vegetation Index and Rainfall in Sudan

## Overview
This project models, predicts, and reconstructs measurements for the vegetation index (NDVI) and precipitation in Sudan, using time series analysis techniques. The project focuses on data collected from the cities of El Geneina and Kassala. The analysis is performed through state-space modeling, Box-Jenkins models, and Kalman filtering to estimate and predict NDVI values and precipitation levels.

<p align="center">
  <img src="https://github.com/AliBakly/FMSN45/assets/21970392/df45d339-58e0-4d90-a235-c8ac91c7b494" width="450">
</p>
<h4 align="center">Figure: The project data.</h4>

## Project Structure
- **FMSN45_Project_Final.pdf** – Final project report detailing the analysis, methods, and results.
- **ProjFinal.m** – Main MATLAB script for data preprocessing, modeling, and validation.
- **KalmanBjStepSim.m** – Simulates a Box-Jenkins model with step input to test Kalman filter adaptation.
- **KalmanMod.m** – Implements a modified Box-Jenkins model with dependencies on lag 36.
- **KalmanModVaryingInput.m** – Extends the Kalman model by introducing dynamic input model parameters.
- **BJSIMDAT.mat** – Dataset for Box-Jenkins simulations.
- **proj23.mat** – Contains the original and interpolated NDVI and rainfall data.
- **proj23.pdf** – Describes the modeling process and project requirements.

## Objectives
1. **Reconstruct Rain Data** – Use Kalman filtering to reconstruct monthly rainfall data at a finer resolution (10-day intervals) based on AR(1) modeling.
2. **Model NDVI for El Geneina** – Develop SARMA models to predict NDVI values with and without rainfall input.
3. **Dynamic Parameter Estimation** – Enhance NDVI predictions by allowing model parameters to vary over time using Kalman filters.
4. **Generalization to Kassala** – Apply the best-performing model to data from Kassala to assess model transferability.

## Methodology
- **State-Space Modeling** – Monthly rainfall data is interpolated using Kalman filters. An AR(1) process is fitted recursively to estimate finer time-scale rainfall.
- **Box-Jenkins Models** – NDVI is modeled using SARMA processes, incorporating seasonal lags to reflect annual vegetation cycles.
- **Kalman Filtering** – Kalman filters are employed to adaptively update model parameters over time, enhancing the accuracy of NDVI predictions.
- **Validation and Testing** – Data is split into modeling (75%), validation (15%), and testing (10%) sets.

<p align="center">
  <img src="https://github.com/AliBakly/FMSN45/assets/21970392/37b408c1-066d-45c8-8acf-fed5b76eb42e" width="800">
</p>
<h4 align="center">Figure: Simulated Box-Jenkins model with dynamic parameters, estimated and predicted via a Kalman fiter.</h4>

## How to Run
1. Clone or download the repository.
2. Open MATLAB and navigate to the `ProjFinal` directory.
3. Run `ProjFinal.m` to execute the full analysis pipeline.
4. Modify the `city` variable in the script to switch between El Geneina and Kassala data.

## Key Results
- The reconstructed rainfall data closely approximates actual values, maintaining overall sums.
- Seasonal ARMA models with precipitation input improve NDVI predictions compared to naive models.
- Dynamic parameter estimation through Kalman filtering enhances prediction accuracy, especially in test data.

## Visualization
- **Rainfall Reconstruction** – Visual comparisons between original and reconstructed rainfall data.
- **NDVI Modeling** – ACF/PACF plots for NDVI modeling and prediction error analysis.
- **Model Performance** – Validation and test results for NDVI predictions.


