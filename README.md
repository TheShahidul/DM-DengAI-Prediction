# Dengue Case: Predicting Disease Spread

## Analysis using Ensemble Stack & Negative Binomial Regression

---

## 📝 Project Overview

This project focuses on **predicting weekly dengue fever outbreaks** in two distinct cities: San Juan (Puerto Rico) and Iquitos (Peru). Leveraging historical epidemiological records and climate data, the goal is to develop a robust forecasting model that can assist health officials in proactive resource planning and preventive actions.

Dengue fever, a mosquito-borne disease, poses a significant public health threat in tropical and subtropical regions. Accurate prediction of its outbreaks is crucial for timely intervention and resource allocation, especially as climate change is expected to influence its transmission dynamics.

## 🎯 Problem Definition

Given climate and disease surveillance data, the core problem is to **predict the number of dengue cases per week** for San Juan and Iquitos. The dataset includes various historical weather variables (temperature, humidity, dew point, precipitation, vegetation index) and weekly reported dengue cases.

### Complex Engineering Problem Criteria Addressed:

* **Depth of Knowledge**: Requires integration of meteorological analysis, epidemiological modeling, and advanced machine learning.
* **Conflicting Requirements**: Balancing model accuracy with generalization across two geographically and climatically different cities.
* **Depth of Analysis**: Involves cleaning, imputing, transforming data, and evaluating multiple regression models.
* **Familiarity of Issues**: Addresses a real-world public health issue with both structured data and domain uncertainty.

## 🚀 Design Goals / Objectives

* Forecast weekly dengue case counts with high accuracy.
* Apply advanced data mining techniques, including feature engineering and model ensembling.
* Validate model performance using appropriate metrics.

## 💡 Application

* **Public health preparedness**: Enables health agencies to prepare for potential outbreaks.
* **Early warning systems**: Provides timely alerts for epidemic outbreaks, allowing for rapid response.
* **Government policy planning**: Supports informed decision-making in tropical regions regarding disease control and resource allocation.

---

## 💻 Methodology

The project follows a structured data mining pipeline:

### 1. Data Acquisition and Preprocessing

* **Data Sources**: The dataset is sourced from the DrivenData DengAI competition, comprising `dengue_features_train.csv`, `dengue_labels_train.csv`, and `dengue_features_test.csv`.
* **Data Cleaning**:
    * The `week_start_date` column was dropped as it was redundant for modeling.
    * Missing values in climate data were filled using a **forward-fill method** (`ffill`), followed by a backward-fill, to maintain temporal continuity and prevent algorithmic failure.
* **City-Wise Segmentation**: Data was split into two subsets for San Juan (`sj`) and Iquitos (`iq`) to account for their distinct geographical, climatic, and epidemiological patterns.

### 2. Feature Engineering & Selection

* **Selected Features**: Initial analysis revealed strong correlations between `total_cases` and `reanalysis_specific_humidity_g_per_kg`, `reanalysis_dew_point_temp_k`, `station_avg_temp_c`, and `station_min_temp_c`. These were primarily used in the mosquito model.
* **Missing Value Handling**: Forward-fill was applied again after feature selection to ensure no NaNs remained.
* **Correlation Analysis**: Heatmaps were generated to visualize feature correlations for both San Juan and Iquitos.
    *     *     * **Observation**: Temperature data showed strong correlations, as expected. However, `total_cases` did not exhibit many obvious strong correlations with individual features initially.

### 3. Model Development

#### A. Negative Binomial Regression (Baseline)

* **Approach**: A Generalized Linear Model (GLM) with a **Negative Binomial family** was used. This model is well-suited for count data like dengue cases, which often exhibit overdispersion.
* **Hyperparameter Tuning**: `alpha` (dispersion parameter) was tuned using a grid search to find the best fit based on Mean Absolute Error (MAE).
* **Training**: Separate Negative Binomial models were trained for San Juan and Iquitos.

#### B. Ensemble Stacking Model (My Part)

* **Concept**: Stacking combines predictions from multiple diverse base models using a meta-learner to make a final prediction. This often leads to improved robustness and accuracy.
    *     * * **Base Estimators**:
    * **Linear Regression**: A simple linear model to capture linear relationships.
    * **RandomForestRegressor**: An ensemble tree-based model known for handling non-linear interactions and less sensitive to outliers.
    * **LGBMRegressor (LightGBM)**: A gradient boosting framework that uses tree-based learning algorithms, optimized for speed and accuracy.
* **Final Estimator (Meta-Learner)**: **GradientBoostingRegressor** was chosen as the meta-learner to combine the predictions of the base models.
* **Cross-Validation**: 5-fold cross-validation (`cv=5`) was used during stacking to prevent overfitting and ensure better generalization.

---

## 📊 Performance Evaluation

### Simulation Environment

All experiments were conducted on **Google Colaboratory** (Python 3.10) leveraging its cloud-based environment for rapid development. Key libraries included `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`, `lightgbm`, and `statsmodels`.

### Metric Calculation

The primary evaluation metric used is **Mean Absolute Error (MAE)**. Additionally, an **Improved Mean Absolute Percentage Error (MAPE)** and an **Accuracy-like Score** were calculated. The $R^2$ score was also included for completeness.

### Results: Negative Binomial vs. Ensemble Stacker

| Model               | MAE   | MAPE (%) | Acc-like (%) | R2 Score |
| :------------------ | :---- | :------- | :----------- | :------- |
| Negative Binomial   | 22.65 | 75.96    | 24.04        | 0.201    |
| **Ensemble Stacker** | **12.09** | **40.92** | **59.08** | **0.779** |

### 📈 Analysis & Output Visualization

The ensemble stacking model significantly **outperformed** the standalone Negative Binomial model across all metrics, particularly in **MAE (reduced by ~46%)** and **R2 Score (increased from 0.201 to 0.779)**. This indicates a much better fit and predictive capability.

* **Line Plot Comparison**:
    *     * **Observation**: The line plots clearly show that the **Ensemble model's predictions (green dashed-dot line)** track the `Actual Cases` (blue solid line) much more closely than the `Negative Binomial` predictions (orange dashed line), especially during periods of higher case counts.

* **Scatter Plot Comparison**:
    *     * **Observation**: The scatter plot demonstrates that the `Ensemble` predictions (green dots) are far more clustered around the ideal red diagonal line (where Actual = Predicted) compared to the `Negative Binomial` predictions (blue dots), which are more dispersed and tend to underestimate higher case numbers.

* **Seaborn Line Plot Comparison**:
    *     * **Observation**: This plot confirms the trend observed in the Matplotlib version, with the Ensemble model showing a superior alignment with actual cases.

* **Plotly Interactive Plot Comparison**:
    *     *     * **Observation**: These interactive plots (close-up views) vividly highlight the Ensemble model's ability to capture the actual trend, whereas the Negative Binomial model struggles to keep up, especially during peaks.

---

## 💡 Conclusion & Future Work

This project successfully demonstrates the effective use of data mining techniques, particularly **ensemble stacking**, for forecasting weekly dengue outbreaks. The model's promising results, especially for Iquitos, align with known epidemiological factors.

### Limitations:

* **Unavailable Test Labels**: The final performance on unseen test data could not be definitively validated due to competition rules.
* **Data Limitations**: Key features like population movement or socioeconomic factors were not included.
* **Overfitting Risk**: Performance was assessed primarily on training/validation data.

### Scope of Future Work:

* **Cross-Validation**: Implement time-aware validation (e.g., rolling or expanding window validation) for more robust generalization metrics.
* **External Data Sources**: Incorporate satellite imagery, land use data, and mosquito vector indices, especially relevant for contexts like Bangladesh.
* **Deep Learning Models**: Explore LSTM or Temporal Convolutional Networks (TCNs) for sequential modeling, especially with daily-level data.
* **Real-time Forecasting**: Integrate the model into a live web-based dashboard for actionable health alerts.

## 🔗 References

* DrivenData Competition Page. DengAI: Predicting Disease Spread.
* DrivenData Benchmark Blog. Predicting Dengue with XGBoost and Negative Binomial Regression.
* T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System, 2016.
* Scikit-learn Documentation. Machine Learning in Python.
* Pandas Documentation. Data Analysis and Manipulation Tool.
