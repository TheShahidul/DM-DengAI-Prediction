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

Run the Jupyter Notebook:

```bash
jupyter notebook DM_Project_Dengue-Pred_v-2.ipynb

```
...
## 📊 Visualizations (Check the Project Report for better understanding)

- 📌 **Correlation heatmaps**
- 📌 **Actual vs. predicted plots** for each city
- 📌 **Scatter plots & interactive Plotly comparisons**
- 📌 **Feature importance charts**

---

## 🚀 Applications

- 🧭 **Early warning systems** for dengue outbreaks.
- 🏥 **Public health resource planning** in tropical regions.
- 📌 **Policy design** for vector-borne disease prevention.

---

## 🔍 Limitations & Future Work

- ⚠️ No test labels due to competition rules → final validation limited.
- ⚠️ Missing real-world factors like population movement, socioeconomic data.

✅ **Future improvements:**
- Incorporate satellite & vector surveillance data.
- Apply time-aware cross-validation.
- Test deep learning (LSTM/TCN) for sequence modeling.
- Develop live web dashboards for real-time forecasting.

---

## 👨‍💻 Author

**Md. Shahidul Islam Prodhan**  
Green University of Bangladesh  
_B.Sc. in CSE — Data Mining Lab Project (Spring 2025)_

---

## 📜 References

- [DrivenData DengAI Competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)

---
