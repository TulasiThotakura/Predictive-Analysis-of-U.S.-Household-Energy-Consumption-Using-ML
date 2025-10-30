# Predictive Analysis of U.S. Household Energy Consumption Using Machine Learning

<p align="center">
  <img src="./visuals/Banner.jpg" alt="Household Energy ML" width="600"/>
</p>

---

## 📄 Overview
This project focuses on analyzing and predicting **U.S. household energy consumption patterns** using **machine learning models**.  

Data is sourced from the **2024 Residential Energy Consumption Survey (RECS)** conducted by the U.S. Energy Information Administration (EIA), which represents a nationally representative sample of over 18,500 households. The dataset includes housing characteristics, appliance usage, energy consumption, and household demographics.

The main objective is to:  
- Explore features most strongly associated with annual electricity usage (kWh)  
- Predict total residential energy consumption using **linear and ensemble machine learning methods**  

---

## 🧰 Technologies Used
- **Languages:** Python  
- **Libraries & Tools:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, FeatureSelector  
- **ML Models:** Linear Regression, Lasso, Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost  
- **Other:** Jupyter Notebook  

---

## 🗂️ Dataset
- **Source:** U.S. Energy Information Administration (EIA) – Residential Energy Consumption Survey (RECS 2024)  
- **Description:** 18,500 households with 900+ variables on housing characteristics, energy usage, appliances, and costs.  
- **Goal:** Predict annual household electricity usage (kWh) using relevant features selected through PCA and feature selection.

---

## 🚀 Workflow
1. **Exploratory Data Analysis (EDA):**  
   - Examined data distributions, correlations, and relationships using scatter plots, boxplots, KDEs, and median kWh plots.  
   - Outliers (e.g., kWh > 80,000) were identified and removed.  

2. **Data Transformation & Preprocessing:**  
   - Merged low-frequency categorical levels, removed unnecessary columns, duplicate features, and NA-heavy columns.  
   - Resulted in 428 features pre-feature selection.  

3. **Dimensionality Reduction (PCA):**  
   - Principal Component Analysis (PCA) applied to identify features explaining 95% of variance (~200 features).  

4. **Feature Selection:**  
   - Used **FeatureSelector** to remove:  
     - Features with >60% missing values  
     - Highly collinear features (≥98% correlation)  
     - Features with zero importance after tree-based encoding  
     - Features with low cumulative importance (<90%)  
     - Single-value features  
   - Final dataset: 216 optimized features  

5. **Modeling & Evaluation:**  
   - Baseline models built using linear and ensemble methods.  
   - **Cross-validation** (k-fold) used to ensure robust evaluation.  
   - Hyperparameter tuning applied to improve predictive performance.  

---

## 📊 Key Results
| Model               | Avg RMSE   | Avg R²   |
|--------------------|------------|----------|
| Linear Regression   | 4624.60    | 0.625    |
| Lasso               | 4619.39    | 0.626    |
| Ridge               | 4623.13    | 0.625    |
| ElasticNet          | 4732.64    | 0.608    |
| Random Forest       | 4673.94    | 0.618    |
| Gradient Boosting   | 4489.18    | 0.647    |
| XGBoost             | 4658.06    | 0.620    |

- **Gradient Boosting** achieved the best performance with an **Avg RMSE of 4489 kWh** and **R² of 0.647**.  
- Cross-validation ensured the results are **stable and reliable** across multiple dataset splits.  

---

## 🔍 Insights
- PCA and feature selection helped reduce dimensionality without losing predictive power.  
- Ensemble models (Gradient Boosting, Random Forest, XGBoost) generally outperform linear models.  
- The most important features for predicting electricity usage were related to **household size, appliances, heating/cooling systems, and housing characteristics**.  

---

## 🔗 References
- **Dataset:** [EIA RECS 2024](https://www.eia.gov/consumption/residential/)  
- **Feature Selection Tool:** [FeatureSelector](https://github.com/WillKoehrsen/feature-selector)  
