# Global GDP Per Capita Analysis (1960–2023) | India Focus

## Project Overview

This comprehensive data science project analyzes global GDP per capita trends from 1960 to 2023, with a special focus on India's economic performance under different political leaderships. The project combines descriptive analytics, predictive modeling, clustering analysis, and political economy insights to understand economic growth patterns and forecast future trends.

## Dataset

**Source:** `pib_per_capita_countries_dataset.csv`  
**Size:** 13,760 records covering 215+ countries  
**Time Period:** 1960-2023 (64 years of economic data)  
**Key Features:**
- `country_name`: Country identifier
- `year`: Year of observation
- `gdp_per_capita`: GDP per capita in current USD
- `gdp_variation`: Year-over-year GDP variation
- `region`, `sub_region`: Geographic classifications

## Methodology & Analysis

### 1. Exploratory Data Analysis (EDA)

**Data Preprocessing:**
```python
# Handle missing values
df_data['gdp_variation'] = df_data['gdp_variation'].fillna(0)

# Statistical analysis
df_data.describe()  # 13,760 observations, mean GDP: $7,654
```

**Key Findings:**
- GDP per capita ranges from $0 to $256,580
- Significant economic inequality globally
- Missing data handled using forward/backward fill methods

**Visualizations Created:**
- Scatter plot: GDP Variation vs GDP per Capita by region
- Histogram: Distribution of median GDP per capita across countries
- Missing data analysis and correlation matrices

### 2. Rich vs Poor Countries Classification (2023)

**Approach:**
```python
# Filter 2023 data and classify based on mean GDP
df_2023 = df_data[df_data['year'] == 2023].dropna(subset=['gdp_per_capita'])
mean_gdp_2023 = df_2023['gdp_per_capita'].mean()

# Binary classification
df_2023['category'] = df_2023['gdp_per_capita'].apply(
    lambda x: 'Rich (≥ Mean)' if x >= mean_gdp_2023 else 'Poor (< Mean)'
)
```

**Results:**
- Clear visualization of global economic disparity in 2023
- Countries categorized into rich/poor segments based on mean GDP threshold
- Bar chart showing distribution of countries across categories

### 3. Predictive Modeling for 2024 GDP

**Data Preparation:**
```python
# Create time-series dataset with 3-year lookback
def create_ts_data(df, lookback=3):
    X, y = [], []
    years = sorted([col for col in df.columns if isinstance(col, int)])
    for i in range(lookback, len(years)):
        X.append(df[years[i-lookback:i]].values)
        y.append(df[years[i]].values)
    return X, y, years[lookback:]

# Pivot data for time-series analysis
pivot_df = df_data.pivot(index='country_name', columns='year', values='gdp_per_capita')
```

**Models Implemented:**
1. **Ridge Regression** with L2 regularization
2. **Lasso Regression** with L1 regularization  
3. **XGBoost** ensemble method

**Model Performance:**
```
Ridge Regression:
- R²: 0.9520
- RMSE: 5,951.22

Lasso Regression:
- R²: 0.9520  
- RMSE: 5,954.13

XGBoost:
- R²: 0.9123
- RMSE: 8,043.60
```

**Feature Engineering:**
- 3-year historical GDP lookback windows
- Time-series cross-validation
- Consensus prediction using weighted average of all models

### 4. Future Economic Leaders Clustering (2024)

**Clustering Methodology:**
```python
# Use consensus prediction (average of all models)
pivot_df['2024_consensus'] = pivot_df[['2024_pred_Ridge', '2024_pred_Lasso', '2024_pred_XGBoost']].mean(axis=1)

# Feature engineering for clustering
pivot_df['growth_1yr'] = safe_growth(pivot_df['2024_consensus'], pivot_df[2023])
pivot_df['growth_3yr'] = safe_growth(pivot_df['2024_consensus'], pivot_df[2021])

# K-Means clustering with optimal k selection
optimal_k = 2  # Determined using silhouette analysis
```

**Cluster Analysis:**
- **Cluster 0 (High GDP - Stable Economies):** 200 countries, Avg GDP: $20,460, 7.7% growth
- **Cluster 1 (Medium GDP - Rapid Growth):** 15 countries, Avg GDP: -$930, 0% growth

**Visualizations:**
- 3D scatter plot with PCA dimensionality reduction
- Interactive Plotly visualization for cluster exploration
- Parallel coordinates plot for cluster characteristic comparison

### 5. India's Leadership-Based GDP Analysis

**Political Timeline Mapping:**
```python
# Comprehensive PM data with exact tenure dates
pm_data = [
    {"Name": "Jawaharlal Nehru", "Start": "1947-08-15", "End": "1964-05-27", "Party": "INC"},
    # ... detailed PM records through 2024
    {"Name": "Narendra Modi (3rd)", "Start": "2024-06-09", "End": "2029-06-09", "Party": "BJP"}
]
```

**Growth Rate Calculation:**
```python
# Calculate CAGR for each PM tenure
cagr = ((end_gdp / start_gdp) ** (1/tenure_years) - 1) * 100
```

**Key Historical Findings:**
- **BJP Average CAGR:** 6.00% (3 terms, 16 years total)
- **INC Average CAGR:** 3.44% (7 terms, 42 years total)
- **Overall Historical Average:** 3.75%

**Modi Government Performance:**
- **First Term (2014-2019):** 5.61% CAGR
- **Second Term (2019-2024):** 5.29% CAGR (including COVID-19 impact)
- **Projected Third Term (2024-2029):** 4.31% CAGR

**Economic Regime Analysis:**
1. **License Raj (1960-1980):** Lower growth period
2. **Early Reforms (1980-1991):** Gradual opening
3. **Liberalization (1991-2000):** Economic transformation
4. **Growth Acceleration (2000-2014):** Sustained high growth
5. **Modi Era (2014-2024):** Consistent performance despite global challenges

### 6. Advanced Predictive Analytics

**Logistic Regression for Growth Classification:**
```python
# Feature engineering for classification
features = ['gdp_lag1', 'gdp_growth_lag1', 'gdp_growth_lag2', 'gdp_3yr_avg',
            'is_bjp', 'is_inc', 'years_in_power', 'global_crisis']

# Model performance
accuracy = 72.73%  # on Modi era test data (2014-2024)
```

**Feature Importance Analysis:**
- **Most Important:** 3-year average GDP growth (coefficient: 2.42)
- **Political Factors:** Years in power, party affiliation
- **Economic Factors:** Lagged GDP values and growth rates

**Future Growth Probability (2024-2029):**
- Probability of above-average growth: 24-27% annually
- Classification: Below average growth expected
- Model suggests potential challenges requiring policy adjustments

### 7. Growth Stability Analysis

**Modi Terms Comparison:**
```python
# Coefficient of variation analysis
modi_stability = modi_years.groupby('term')['growth_rate'].agg(['mean', 'std', 'cv'])
```

**Results:**
- **1st Term:** 6.64% mean growth, CV: 0.82 (more stable)
- **2nd Term:** 5.28% mean growth, CV: 1.45 (more volatile, COVID impact)

## Technical Implementation

### Data Science Pipeline
```
Raw CSV → Data Cleaning → Feature Engineering → Model Training → Predictions → Visualizations
```

### Key Libraries Used
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn, plotly
- **Statistical Analysis:** scipy, statsmodels

### Model Validation Techniques
- Time-series cross-validation
- Train-test splits (pre-2014 vs Modi era)
- Silhouette analysis for clustering
- Multiple regression metrics (R², RMSE, MAE)

## Key Insights & Conclusions

### Historical Economic Performance
1. **Post-1991 liberalization** marked India's economic transformation
2. **BJP governments** have averaged higher growth rates than INC
3. **Policy continuity** appears more important than party affiliation for sustained growth

### Current Modi Government Assessment
1. **Strong first term** performance (5.61% CAGR)
2. **Resilient second term** despite global challenges (5.29% CAGR)
3. **Future projections** suggest moderation to 4.31% CAGR

### Predictive Model Insights
1. **3-year GDP trends** are the strongest predictor of future performance
2. **Political stability** (years in power) positively correlates with growth
3. **Global economic conditions** significantly impact growth probability

### Global Economic Context
1. **Clustering analysis** reveals India in the high GDP stable economies group
2. **Future leadership potential** based on projected 2024 GDP performance
3. **Economic inequality** remains a significant global challenge

## Limitations & Future Work

### Current Limitations
- GDP per capita doesn't capture inequality within countries
- Political factors simplified into party binary variables
- Global economic context modeled with basic crisis indicators
- Projections assume policy continuity

### Potential Enhancements
- Incorporate additional economic indicators (inflation, unemployment)
- Add geopolitical factors and trade relationships
- Implement deep learning models for complex pattern recognition
- Include real-time data integration via economic APIs

## Files Generated
- `india_gdp_by_pm_2024.png`: Historical GDP trends with PM timelines
- `india_pm_growth_rates_2024.png`: CAGR comparison across governments
- `india_economic_regimes.png`: Growth by policy periods
- `growth_probability_forecast.png`: Future growth probability predictions
- `modi_terms_growth_distribution.png`: Growth stability analysis

## Reproducibility
All code is documented with clear comments, and the analysis can be reproduced by running the Jupyter notebook with the provided dataset. Model parameters and random seeds are set for consistent results across runs.

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
