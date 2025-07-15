# DM-DengAI-Prediction
Forecast weekly dengue cases in 2 cities using climate data, ensemble stacking, and negative binomial regression and the output is significantly better than the original. (I have implemented this Data Mining Lab as a project inspired by the DrivenData DengAI competition and find out new result.)


📊 DengAI: Predicting Disease Spread
Analysis using Ensemble Stacking & Negative Binomial Regression


📝 Overview
This project forecasts weekly dengue cases in San Juan (Puerto Rico) and Iquitos (Peru) using climate and epidemiological data from the DrivenData DengAI competition. It applies data mining, ensemble learning, and regression techniques to create predictive models that can support early-warning systems and public health planning.

⚙ Project Workflow
Data cleaning & preprocessing (missing value imputation, city-wise segmentation)

Feature engineering & selection

Model training:
XGBoost Regressor (city-specific models)
Ensemble Stacking Regressor
Negative Binomial Regression for baseline
Model evaluation with RMSE, MAE, MAPE, and R² score
Visualization: predicted vs. actual plots, correlation heatmaps, interactive charts

📈 Key Results

| Model             |   MAE | MAPE (%) | Acc-like (%) | R² Score |
| ----------------- | ----: | -------: | -----------: | -------: |
| Negative Binomial | 22.65 |    75.96 |        24.04 |    0.201 |
| Ensemble Stacker  | 12.09 |    40.92 |        59.08 |    0.779 |

Ensemble Stacker significantly outperformed baseline
Humidity, temperature, dew point, and seasonal features were the most influential


📂 Dataset
Source: DrivenData DengAI Competition
Features: temperature, precipitation, humidity, vegetation indices
Labels: weekly dengue case counts for San Juan and Iquitos

🛠 Installation & Usage

# Clone repository
git clone https://github.com/your-username/dengai-prediction.git
cd dengai-prediction

# Install dependencies
pip install -r requirements.txt

Run the main notebook:
jupyter notebook DM_Project_Dengue-Pred_v-2.ipynb
" Make sure dataset CSV files (dengue_features_train.csv, dengue_labels_train.csv, dengue_features_test.csv) are in the same folder or adjust paths accordingly."

📊 Visualizations
Correlation heatmaps between features & dengue cases

Actual vs. predicted cases plots (San Juan & Iquitos)

Interactive Plotly graphs comparing Negative Binomial vs Ensemble predictions

🚀 Applications
Early warning systems for dengue outbreaks

Public health resource planning

Government policy development in tropical regions


🔍 Limitations & Future Work
Limited features (no data on population movement, mosquito surveillance, socioeconomics)

Overfitting risk due to small dataset

Future work: incorporate satellite imagery, test time-aware cross-validation, explore LSTM/TCN models


👨‍💻 Author
Md. Shahidul Islam Prodhan
Student, Green University of Bangladesh
Course: Data Mining Lab (CSE-436)
Project submitted: Spring 2025

📜 References
DrivenData DengAI Competition
XGBoost paper
Scikit-learn, Pandas, StatsModels official docs
