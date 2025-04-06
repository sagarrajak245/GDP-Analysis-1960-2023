# 🌍 Global GDP Per Capita Analysis (1960–2023) | India Focus 🇮🇳

A comprehensive data science project that analyzes and predicts GDP per capita trends across countries, with an in-depth focus on India's economic progress under different leaderships and future potential. This includes visualizations, machine learning-based predictions, clustering insights, and leadership impact analysis.

---

## 📌 Project Overview

This project is divided into four major components:

---

### 1. 📊 Visual Analysis of Rich & Poor Countries (2022)

- **Objective**: Identify the top 10 richest and poorest countries based on GDP per capita in the year 2022.
- **Approach**:
  - Filter dataset for the year 2022
  - Use bar plots to display top 10 highest and lowest GDP per capita countries
- **Tools**: `pandas`, `matplotlib`, `seaborn`

---

### 2. 🔮 Predictive Modeling – GDP per Capita for 2023

- **Objective**: Predict GDP per capita for all countries for the year 2023.
- **Approach**:
  - Time-series forecasting using GDP history
  - Regression model (e.g., Linear Regression or Random Forest)
- **Evaluation Metrics**:
  - `Mean Absolute Error (MAE)`
  - `Root Mean Squared Error (RMSE)`
  - `R² Score`
- **Outcome**: Predicted GDP per capita values for 2023

---

### 3. 🔗 Clustering Future Leaders Based on 2023 GDP

- **Objective**: Identify which countries are likely to lead the global economy based on their 2023 GDP performance.
- **Approach**:
  - Apply K-Means Clustering on 2023 GDP data
  - Visualize clusters using PCA/t-SNE (if dimensionality reduction used)
- **Evaluation Metrics**:
  - `Silhouette Score`
  - `Inertia (within-cluster sum of squares)`
- **Outcome**: Country groups based on economic similarity and leadership potential

---

### 4. 🇮🇳 Leadership-Based GDP Trend & Future Prediction for India

- **Objective**: 
  - Visualize India's GDP per capita trend under various Prime Ministers (1960–2022)
  - Analyze growth rates by leadership and political party
  - Predict if India's GDP will continue to grow under current leadership

- **Approach**:
  - Use historical GDP data aligned with PM tenures
  - Calculate CAGR per PM & average party growth
  - Logistic Regression to predict **whether India’s GDP will grow in the next 4 years**
- **Evaluation Metrics** (for classification model):
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `Confusion Matrix`
- **Outcome**: 
  - Visual timeline of leadership and GDP trends
  - Prediction: Whether GDP per capita will rise under current government

---

## 📁 Dataset

- `pib_per_capita_countries_dataset.csv`
- Contains:
  - `country_name`
  - `year`
  - `gdp_per_capita`
- Source: World Bank (or similar global economic dataset)

---

## 🛠️ Technologies Used

- `Python 3.7+`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `statsmodels`
- `jupyter-notebook` / `VS Code`

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/global-gdp-analysis.git
cd global-gdp-analysis
pip install -r requirements.txt
