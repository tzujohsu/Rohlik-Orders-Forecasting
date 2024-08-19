

# Rohlik Orders Forecasting Challenge
### Use historical data to predict customer orders

https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/overview


## Project Overview

This project addresses a critical challenge in the e-grocery industry: accurately predicting order volumes for Rohlik Group. The goal is to forecast the number of grocery delivery orders for selected warehouses over a 60-day period.

### Business Goal

The primary business objective is to enhance operational efficiency and sustainability in Rohlik's e-grocery services. Accurate order forecasts are crucial for:

- Optimizing workforce allocation
- Improving delivery logistics
- Enhancing inventory management
- Streamlining supply chain operations.

### Technical Goal

From a technical perspective, the project aims to develop a robust machine learning model that can:

1. Process historical order data from multiple warehouses
2. Handle time-series forecasting for a 60-day period
3. Incorporate various features, including those not available at prediction time (e.g., precipitation, shutdown, user website activity)
4. Minimize the **Mean Absolute Percentage Error (MAPE)** between predicted and actual order volumes

$$ MAPE = (1/n) * Σ|(Actual - Forecast) / Actual| * 100 $$


## Methodology

My approach follows a comprehensive pipeline:

1. **Exploratory Data Analysis (EDA)**: To understand the data distribution, patterns, and potential insights.
2. **Preprocessing**: Data cleaning and feature engineering to prepare the dataset for modeling.
3. **Model Evaluation**: An overview of a diverse set of different models to identify promising algorithms.
4. **Hyperparameter Tuning**: Using Grid Search and Random Search to optimize model parameters.
5. **Feature Importance Analysis**: To understand which factors most significantly influence order volumes.
6. **Final Model Selection**: Based on performance metrics.
7. **Extra: Model Stacking**: Employing a meta-learner to combine predictions from multiple models.
8. **Result Visualization**: To present findings and predictions in an easily interpretable format.

## Models Explored

I evaluated a diverse range of regression models:

- XGBRegressor
- HistGradientBoostingRegressor
- LGBMRegressor
- RandomForestRegressor
- BaggingRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- DecisionTreeRegressor
- ExtraTreeRegressor
- AdaBoostRegressor
- KNeighborsRegressor

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- LightGBM
- skimpy (for EDA -> data summarization)
- Pandas
- NumPy
- Matplotlib/Seaborn (for visualization)
