# 🌍 Happiness Machine Learning Dashboard

An interactive Shiny dashboard built in R to explore, analyze, and compare machine learning models applied to the **World Happiness Report** dataset.

## 📊 Overview

This project analyzes the **happiness levels of countries** using four machine learning models:
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Linear Regression (for regression modeling)

A custom Shiny dashboard provides **visual comparisons**, **confusion matrices**, **decision tree visualizations**, and **regression insights**.

---

## 🧠 Machine Learning Models & Metrics

The categorical target variable (`Happiness.Level`) is derived from the `Happiness.Score` using:
- Low (<5), Medium (5–7), High (>7)

### 🔢 Model Comparison (Classification)

| Metric     | Naive Bayes | Decision Tree | KNN   |
|------------|-------------|----------------|-------|
| **Accuracy**   | 89.1%       | 82.6%          | 80.4% |
| **Recall**     | 88.4%       | 76.7%          | 78.6% |
| **Precision**  | 86.1%       | 82.5%          | 74.1% |
| **F1 Score**   | 86.8%       | 78.4%          | 75.3% |

Naive Bayes performs best across all evaluation metrics.

---

## 📈 Visualizations

- 🔹 **Confusion Matrices**: For each classification model.
- 🔹 **Bar Charts**: Compare metrics (Accuracy, Recall, Precision, F1 Score).
- 🔹 **Decision Tree Plot**: Graphical view of splits.
- 🔹 **Linear Regression Plots**:
  - Actual vs. Predicted
  - Residuals vs. Predicted

---

## 📁 Folder Structure

```
Happiness-ML-Dashboard/
├── app.R
├── ui.R
├── server.R
├── data/
│   └── World Happiness Report.csv
├── data_loader.R
├── model_knn.R
├── model_naive_bayes.R
├── model_decision_tree.R
├── model_linear_regression.R
├── utils.R
├── metrics_plotting.R
└── README.md
```

---

## 🚀 Run the Dashboard Locally

```r
# In R or RStudio
setwd("path/to/Happiness-ML-Dashboard")
shiny::runApp()
```

---

## ☁️ Deployment (ShinyApps.io)

1. Install & load `rsconnect`:
   ```r
   install.packages("rsconnect")
   library(rsconnect)
   ```

2. Authenticate with your ShinyApps.io account.

3. Deploy the app:
   ```r
   rsconnect::deployApp()
   ```

App is publicly accessible at:
```
https://mohit26bm.shinyapps.io/happiness-ml-dashboard/
```

---

## 📚 Dataset Source

- **World Happiness Report**  
  Available from [https://worldhappiness.report/](https://worldhappiness.report/)

---

## 👨‍💻 Author

This dashboard was developed by Mohit Bali as part of a machine learning and data visualization project using R and Shiny.

---


