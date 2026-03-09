# World Happiness ML Dashboard

An interactive Shiny dashboard built in R to explore, analyze, and predict happiness levels across countries using the **World Happiness Report** dataset. The dashboard combines multiple machine learning models with an interactive UI for prediction, feature exploration, and country comparison.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Model Results](#model-results)
- [Dashboard Features](#dashboard-features)
- [Data Pipeline](#data-pipeline)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [Key Findings](#key-findings)
- [Dependencies](#dependencies)

---

## Overview

This project applies **5 classification models** and **4 regression models** to predict happiness outcomes for countries worldwide. A custom Shiny dashboard provides interactive visualisations, confusion matrices, feature importance exploration, a live prediction tool, and a country comparison tool.

---

## Dataset

**Source:** [World Happiness Report](https://worldhappiness.report/)

| Column | Description |
| --- | --- |
| `Country` | Country name |
| `Happiness.Rank` | Global happiness ranking |
| `Happiness.Score` | Composite happiness score (2.69 – 7.54) |
| `Economy` | GDP per capita contribution |
| `Family` | Social support contribution |
| `Health` | Healthy life expectancy contribution |
| `Freedom` | Freedom to make life choices |
| `Generosity` | Generosity contribution |
| `Corruption` | Trust in government (inverse of corruption) |
| `Job.Satisfaction` | Job satisfaction score (44.4 – 95.1) |
| `Region` | Geographic region (visualisation only) |

> **Note:** `Dystopia` was intentionally excluded — it is a WHR residual benchmark variable mathematically derived from the other features, not a real-world measurement. Including it would cause data leakage in regression models.

### Data Cleaning

- Removed 2 ghost rows (blank entries with no country name)
- Converted 2 `NaN` values in `Job.Satisfaction` (North Cyprus, South Sudan) to median
- Imputed impossible zero values: `Health = 0` (Lesotho), `Freedom = 0` (Greece, Angola), `Family = 0` (Central African Republic)
- Retained `Corruption = 0` (Bosnia) and near-zero `Economy` (Central African Republic) as plausible real-world values
- Final dataset: **151 countries**

---

## Project Structure

```text
Happiness-ML-Dashboard/
├── app.R                          # App entry point
├── ui.R                           # Full sidebar dashboard UI
├── server.R                       # Main server logic
├── data_loader.R                  # Data pipeline & feature engineering
├── utils.R                        # calculate_metrics(), render_confusion_matrix()
├── metrics_plotting.R             # plot_comparison_metric()
│
├── data/
│   ├── World Happiness Report.csv         # Raw dataset
│   └── World_Happiness_Cleaned.csv        # Cleaned dataset (app loads this)
│
├── models/
│   ├── model_knn.R
│   ├── model_naive_bayes.R
│   ├── model_decision_tree.R
│   ├── model_linear_regression.R
│   ├── model_random_forest.R
│   └── model_xgboost.R
│
└── modules/
    ├── mod_predict.R
    ├── mod_feature_importance.R
    └── mod_country_compare.R
```

---

## Machine Learning Models

### Classification — predicts `Happiness.Level` (Low / Medium / High)

Happiness levels are derived from `Happiness.Score` using:

| Level | Score Range |
| --- | --- |
| Low | < 5.0 |
| Medium | 5.0 – 7.0 |
| High | > 7.0 |

Models used: **KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost**

### Regression — predicts `Happiness.Score` (continuous)

Models used: **Linear Regression, Decision Tree, Random Forest, XGBoost**

---

## Model Results

### Train / Test Split

| | Count |
| --- | --- |
| Total countries | 151 |
| Train set | 120 (80%) |
| Test set | 31 (20%) |
| Seed | 123 |

**Class distribution:**

| Class | Train | Test |
| --- | --- | --- |
| Low | 44 | 12 |
| Medium | 67 | 15 |
| High | 9 | 4 |

> **Note:** The High class is underrepresented (only 13 countries total). F1 Score per class is a more reliable metric than overall accuracy for this dataset.

---

### Classification Results

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| **Naive Bayes** | **90.3%** | **87.5%** | **86.7%** | **87.0%** |
| KNN | 83.9% | 84.3% | 81.1% | 81.9% |
| Random Forest | 83.9% | 91.7% | 75.0% | 79.4% |
| XGBoost | 83.9% | 79.9% | 76.1% | 77.6% |
| Decision Tree | 77.4% | 75.0% | 76.7% | 75.3% |

**Per-class F1 Score (High class — hardest to predict):**

| Model | Low | Medium | High |
| --- | --- | --- | --- |
| Naive Bayes | 95.7% | 90.3% | 75.0% |
| KNN | 85.7% | 84.8% | 75.0% |
| Random Forest | 85.7% | 85.7% | 66.7% |
| Decision Tree | 81.8% | 77.4% | 66.7% |
| XGBoost | 91.7% | 83.9% | 57.1% |

> Naive Bayes outperforms all models including the more complex ensemble methods — a meaningful result on this small, low-correlation dataset.

---

### Regression Results

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| **Linear Regression** | **0.5260** | **0.4085** | **0.8263** |
| Random Forest | 0.5598 | 0.4383 | 0.8032 |
| XGBoost | 0.6203 | 0.4674 | 0.7584 |
| Decision Tree | 0.6766 | 0.5172 | 0.7126 |

> Linear Regression achieves the best R² of 0.826, explaining 82.6% of happiness score variance. XGBoost and Decision Tree underperform simpler models — a classic result on small datasets where model complexity does not always help.

---

## Dashboard Features

| Tab | Description |
| --- | --- |
| **Overview** | Dataset summary, score distribution, top/bottom 10 countries, correlation heatmap |
| **Classification** | Confusion matrices for all 5 classifiers, decision tree visualisation |
| **Regression** | Actual vs predicted and residual plots for all 4 regressors, metrics table |
| **Model Comparison** | Side-by-side bar charts for Accuracy, Precision, Recall, F1 Score |
| **Feature Importance** | Per-model importance charts (normalised 0–100), RF vs XGBoost comparison, cross-model summary table |
| **Predict** | Slider-based input form — get predictions from all 5 classifiers and 4 regressors simultaneously |
| **Country Compare** | Select up to 5 countries and compare feature profiles and happiness scores |

---

## Data Pipeline

```text
load_data()          → clean raw CSV, impute missing values, create Happiness.Level
split_data()         → 80/20 train/test split (set.seed = 123)
get_scaled()         → scale features for KNN and Naive Bayes, store center/scale params
get_xgb_matrices()   → build DMatrix objects and 0-based integer labels for XGBoost
```

All model files consume a **consistent return structure:**

```text
list(model, predictions, actuals, conf_matrix / residuals, importance, metrics, type)
```

---

## Installation & Setup

### 1. Install required packages

```r
install.packages(c(
  "shiny",
  "shinydashboard",
  "ggplot2",
  "plotly",
  "caTools",
  "class",
  "e1071",
  "rpart",
  "rpart.plot",
  "randomForest",
  "xgboost",
  "dplyr",
  "tidyr",
  "viridis"
))
```

### 2. Clone or download the project

```text
Happiness-ML-Dashboard/
```

### 3. Generate the cleaned dataset (one time only)

```r
setwd("path/to/Happiness-ML-Dashboard")
source("data_loader.R")
data <- load_data("data/World Happiness Report.csv")
write.csv(data, "data/World_Happiness_Cleaned.csv", row.names = FALSE)
```

---

## Running the App

```r
setwd("path/to/Happiness-ML-Dashboard")
shiny::runApp()
```

The app trains all models on startup. Expected load time is 10–20 seconds depending on hardware.

---

## Deployment

The app is deployed on ShinyApps.io:

```text
https://mohit26bm.shinyapps.io/happiness-ml-dashboard/
```

To redeploy:

```r
install.packages("rsconnect")
library(rsconnect)
rsconnect::deployApp()
```

---

## Key Findings

**Feature Importance — consistent across all 6 tree-based models:**

| Rank | Feature | Verdict |
| --- | --- | --- |
| 1 | Job Satisfaction | Dominant — top 2 in 5 out of 6 models |
| 2 | Health | Consistently top 3, strongest in XGBoost Regressor |
| 3 | Economy | Consistently top 3 across all models |
| 4 | Family | Mid tier |
| 5 | Freedom | Mid tier, stronger for regression than classification |
| 6 | Generosity | Weak signal |
| 7 | Corruption | Weakest — negative importance in RF Classifier |

**Notable ML insights:**

- Naive Bayes outperforms Random Forest and XGBoost for classification — simpler models win when features have low inter-correlation and the dataset is small
- Linear Regression outperforms XGBoost for regression on this dataset — complexity does not always improve performance
- The High happiness class (13 countries) is the hardest to predict across all models — XGBoost scores only 57.1% F1 on this class
- Job Satisfaction was added as a feature in this version and immediately became the strongest predictor, outranking Economy and Health in most models

---

## Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| shiny | latest | Web app framework |
| shinydashboard | latest | Sidebar dashboard layout |
| ggplot2 | latest | Static plots |
| plotly | latest | Interactive plots |
| caTools | latest | Train/test split |
| class | latest | KNN |
| e1071 | latest | Naive Bayes |
| rpart | latest | Decision Tree |
| rpart.plot | latest | Decision Tree visualisation |
| randomForest | latest | Random Forest |
| xgboost | latest | XGBoost |
| dplyr | latest | Data manipulation |
| tidyr | latest | pivot_longer for country compare |
| viridis | latest | Color scales for importance plots |
