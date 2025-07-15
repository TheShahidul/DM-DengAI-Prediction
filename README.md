# 📊 DengAI: Predicting Disease Spread  
_Analysis using Ensemble Stacking & Negative Binomial Regression_

---

## 📝 Overview

This project forecasts weekly dengue cases in **San Juan** (Puerto Rico) and **Iquitos** (Peru) using historical climate and epidemiological data from the DrivenData **DengAI** competition.  
The goal is to support **early warning systems** and public health planning through robust predictive modeling.

---

## ⚙️ Project Workflow

- ✅ Data cleaning & preprocessing (missing value handling, city-wise segmentation)
- ✅ Feature engineering & selection
- ✅ Model training:
  - XGBoost Regressor (separate models for each city)
  - Ensemble Stacking Regressor
  - Negative Binomial Regression (baseline)
- ✅ Evaluation with RMSE, MAE, MAPE, R²
- ✅ Visualizations: predicted vs. actual plots, correlation heatmaps, interactive charts

---

## 📈 Key Results

| Model                | MAE   | MAPE (%) | Accuracy-like (%) | R² Score |
|----------------------|-------|----------|-------------------|----------|
| Negative Binomial    | 22.65 | 75.96    | 24.04             | 0.201    |
| Ensemble Stacker     | 12.09 | 40.92    | 59.08             | 0.779    |

**Highlights:**  
- The **Ensemble Stacker** significantly outperformed the baseline.
- Most influential features: humidity, average temperature, dew point temperature, seasonal patterns.

---

## 📂 Dataset

- **Source:** [DrivenData DengAI Competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
- **Features:** Temperature, precipitation, humidity, vegetation indices.
- **Labels:** Weekly dengue case counts for San Juan and Iquitos.

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/TheShahidul/DengAI-Prediction.git
cd DengAI-Prediction

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
