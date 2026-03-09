# ui.R
library(shiny)
library(shinydashboard)
library(plotly)

ui <- dashboardPage(
  skin = "blue",
  
  # ── Header ──────────────────────────────────────────
  dashboardHeader(title = "Happiness ML Dashboard"),
  
  # ── Sidebar ─────────────────────────────────────────
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("globe")),
      menuItem(
        "Classification",
        tabName = "classif",
        icon = icon("robot")
      ),
      menuItem(
        "Regression",
        tabName = "regression",
        icon = icon("chart-line")
      ),
      menuItem(
        "Model Comparison",
        tabName = "comparison",
        icon = icon("chart-bar")
      ),
      menuItem(
        "Feature Importance",
        tabName = "features",
        icon = icon("search")
      ),
      menuItem("Predict", tabName = "predict", icon = icon("magic")),
      menuItem(
        "Country Compare",
        tabName = "compare",
        icon = icon("flag")
      )
    )
  ),
  
  # ── Body ────────────────────────────────────────────
  dashboardBody(
    tabItems(
      # ── 1. OVERVIEW ───────────────────────────────
      tabItem(
        tabName = "overview",
        h2("World Happiness Report — ML Dashboard"),
        p(
          "Explore happiness across countries using 5 classifiers and 4 regressors."
        ),
        fluidRow(
          # Summary stat boxes
          valueBoxOutput("total_countries", width = 3),
          valueBoxOutput("num_features", width = 3),
          valueBoxOutput("best_classifier", width = 3),
          valueBoxOutput("best_regressor", width = 3)
        ),
        fluidRow(
          box(
            title = "Happiness Score Distribution",
            width = 6,
            status = "primary",
            plotlyOutput("score_distribution", height = "300px")
          ),
          box(
            title = "Happiness Level Breakdown",
            width = 6,
            status = "info",
            plotlyOutput("level_pie", height = "300px")
          )
        ),
        fluidRow(
          box(
            title = "Top 10 Happiest Countries",
            width = 6,
            status = "success",
            plotlyOutput("top10_plot", height = "300px")
          ),
          box(
            title = "Bottom 10 Countries",
            width = 6,
            status = "danger",
            plotlyOutput("bottom10_plot", height = "300px")
          )
        ),
        fluidRow(
          box(
            title = "Feature Correlation Heatmap",
            width = 12,
            status = "warning",
            plotlyOutput("corr_heatmap", height = "400px")
          )
        )
      ),
      
      # ── 2. CLASSIFICATION ─────────────────────────
      tabItem(
        tabName = "classif",
        h2("Classification Models"),
        p(
          "Confusion matrices for all 5 classifiers predicting Happiness Level (Low / Medium / High)."
        ),
        fluidRow(
          box(
            title = "KNN",
            width = 6,
            status = "primary",
            plotlyOutput("knn_matrix", height = "350px")
          ),
          box(
            title = "Naive Bayes",
            width = 6,
            status = "primary",
            plotlyOutput("nb_matrix", height = "350px")
          )
        ),
        fluidRow(
          box(
            title = "Decision Tree",
            width = 4,
            status = "primary",
            plotlyOutput("dt_cls_matrix", height = "350px")
          ),
          box(
            title = "Random Forest",
            width = 4,
            status = "primary",
            plotlyOutput("rf_cls_matrix", height = "350px")
          ),
          box(
            title = "XGBoost",
            width = 4,
            status = "primary",
            plotlyOutput("xgb_cls_matrix", height = "350px")
          )
        ),
        fluidRow(
          box(
            title = "Decision Tree Visualisation",
            width = 12,
            status = "info",
            plotOutput("decision_tree_plot", height = "500px")
          )
        )
      ),
      
      # ── 3. REGRESSION ─────────────────────────────
      tabItem(
        tabName = "regression",
        h2("Regression Models"),
        p(
          "Predicting the actual Happiness Score (continuous) using 4 regression models."
        ),
        
        # Metrics summary table at top
        fluidRow(
          box(
            title = "Regression Metrics Summary",
            width = 12,
            status = "primary",
            tableOutput("reg_metrics_table")
          )
        ),
        
        # Linear Regression
        fluidRow(
          h3("Linear Regression", style = "padding-left:15px"),
          box(
            title = "Actual vs Predicted",
            width = 6,
            status = "info",
            plotlyOutput("lr_actual_pred", height = "300px")
          ),
          box(
            title = "Residuals vs Predicted",
            width = 6,
            status = "info",
            plotlyOutput("lr_residuals", height = "300px")
          )
        ),
        
        # Decision Tree Regressor
        fluidRow(
          h3("Decision Tree Regressor", style = "padding-left:15px"),
          box(
            title = "Actual vs Predicted",
            width = 6,
            status = "warning",
            plotlyOutput("dt_actual_pred", height = "300px")
          ),
          box(
            title = "Residuals vs Predicted",
            width = 6,
            status = "warning",
            plotlyOutput("dt_residuals", height = "300px")
          )
        ),
        
        # Random Forest Regressor
        fluidRow(
          h3("Random Forest Regressor", style = "padding-left:15px"),
          box(
            title = "Actual vs Predicted",
            width = 6,
            status = "success",
            plotlyOutput("rf_actual_pred", height = "300px")
          ),
          box(
            title = "Residuals vs Predicted",
            width = 6,
            status = "success",
            plotlyOutput("rf_residuals", height = "300px")
          )
        ),
        
        # XGBoost Regressor
        fluidRow(
          h3("XGBoost Regressor", style = "padding-left:15px"),
          box(
            title = "Actual vs Predicted",
            width = 6,
            status = "danger",
            plotlyOutput("xgb_actual_pred", height = "300px")
          ),
          box(
            title = "Residuals vs Predicted",
            width = 6,
            status = "danger",
            plotlyOutput("xgb_residuals", height = "300px")
          )
        )
      ),
      
      # ── 4. MODEL COMPARISON ───────────────────────
      tabItem(
        tabName = "comparison",
        h2("Model Comparison"),
        p("Side-by-side comparison of all classifier metrics."),
        fluidRow(
          box(
            title = "Accuracy",
            width = 6,
            status = "primary",
            plotlyOutput("accuracy_plot", height = "280px")
          ),
          box(
            title = "F1 Score",
            width = 6,
            status = "success",
            plotlyOutput("f1_plot", height = "280px")
          )
        ),
        fluidRow(
          box(
            title = "Recall",
            width = 6,
            status = "warning",
            plotlyOutput("recall_plot", height = "280px")
          ),
          box(
            title = "Precision",
            width = 6,
            status = "danger",
            plotlyOutput("precision_plot", height = "280px")
          )
        ),
        fluidRow(
          box(
            title = "Full Metrics Table",
            width = 12,
            status = "info",
            tableOutput("cls_metrics_table")
          )
        )
      ),
      
      # ── 5. FEATURE IMPORTANCE ─────────────────────
      tabItem(
        tabName = "features",
        h2("Feature Importance Explorer"),
        p(
          "Compare which features drive predictions across tree-based models."
        ),
        fluidRow(
          box(
            width = 3,
            status = "primary",
            selectInput(
              "feat_model_select",
              "Select Model",
              choices = c(
                "Decision Tree (Classifier)"  = "dt_cls",
                "Decision Tree (Regressor)"   = "dt_reg",
                "Random Forest (Classifier)"  = "rf_cls",
                "Random Forest (Regressor)"   = "rf_reg",
                "XGBoost (Classifier)"        = "xgb_cls",
                "XGBoost (Regressor)"         = "xgb_reg"
              )
            ),
            hr(),
            p("Gain = improvement in accuracy brought by a feature.", style = "font-size:12px; color:grey")
          ),
          box(
            title = "Feature Importance",
            width = 9,
            status = "primary",
            plotlyOutput("importance_plot", height = "420px")
          )
        ),
        fluidRow(
          box(
            title = "Side-by-Side Comparison (RF vs XGBoost — Classification)",
            width = 12,
            status = "info",
            plotlyOutput("importance_compare_plot", height = "380px")
          )
        ),
        fluidRow(
          box(
            title = "Cross-Model Importance Summary",
            width = 12,
            status = "warning",
            p("Higher = more important. Scales differ per model type."),
            tableOutput("importance_summary_table")
          )
        )
      ),
      # ── 6. PREDICT ────────────────────────────────
      tabItem(
        tabName = "predict",
        h2("Predict Happiness for a Custom Country"),
        p(
          "Adjust the sliders to set feature values and see what all models predict."
        ),
        fluidRow(
          box(
            title = "Input Features",
            width = 4,
            status = "primary",
            sliderInput("p_gdp", "Economy (GDP per Capita)", 0.0, 2.0, 1.0, step = 0.01),
            sliderInput("p_family", "Family (Social Support)", 0.0, 2.0, 1.0, step = 0.01),
            sliderInput("p_health", "Health (Life Expectancy)", 0.0, 1.5, 0.7, step = 0.01),
            sliderInput("p_freedom", "Freedom", 0.0, 1.0, 0.5, step = 0.01),
            sliderInput("p_generosity", "Generosity", 0.0, 0.8, 0.2, step = 0.01),
            sliderInput("p_corruption", "Trust in Government", 0.0, 0.5, 0.1, step = 0.01),
            sliderInput("p_jobsat", "Job Satisfaction", 50.0, 100.0, 75.0, step = 0.5),
            hr(),
            actionButton(
              "predict_btn",
              "Predict",
              class = "btn-success btn-lg btn-block",
              icon  = icon("play")
            )
          ),
          
          box(
            title = "Classification Results (Happiness Level)",
            width = 4,
            status = "success",
            div(
              style = "font-size:12px; color:#555; margin-bottom:10px;",
              p("Most influential features for classification:"),
              tags$ul(
                tags$li("Job Satisfaction — strongest signal across all classifiers"),
                tags$li("Health — most important in Random Forest"),
                tags$li("Economy — consistently top 3"),
                tags$li("Corruption and Generosity — weakest predictors, minimal effect")
              )
            ),
            hr(),
            tableOutput("pred_cls_table"),
            hr(),
            plotlyOutput("pred_cls_plot", height = "250px")
          ),
          
          box(
            title = "Regression Results (Happiness Score)",
            width = 4,
            status = "info",
            div(
              style = "font-size:12px; color:#555; margin-bottom:10px;",
              p("To move the predicted score meaningfully:"),
              tags$ul(
                tags$li("Increase Job Satisfaction above 85 for biggest impact"),
                tags$li("Increase Health above 0.7 — strong in XGBoost and RF"),
                tags$li("Increase Economy above 1.3 for higher score predictions"),
                tags$li("Freedom matters more for score than classification"),
                tags$li("Generosity and Corruption have minimal effect on score")
              )
            ),
            hr(),
            tableOutput("pred_reg_table"),
            hr(),
            plotlyOutput("pred_reg_plot", height = "250px")
          )
        )
      ),
      
      # ── 7. COUNTRY COMPARE ────────────────────────
      tabItem(
        tabName = "compare",
        h2("Country Comparison Tool"),
        p(
          "Select up to 5 countries and compare their happiness feature profiles."
        ),
        fluidRow(
          box(
            width = 3,
            status = "primary",
            selectizeInput(
              "cmp_countries",
              "Select Countries (max 5)",
              choices  = NULL,
              multiple = TRUE,
              options  = list(maxItems = 5)
            ),
            hr(),
            checkboxGroupInput(
              "cmp_features",
              "Features to Compare",
              choices  = c(
                "Economy",
                "Family",
                "Health",
                "Freedom",
                "Generosity",
                "Corruption",
                "Job.Satisfaction"
              ),
              selected = c("Economy", "Family", "Health", "Freedom")
            )
          ),
          box(
            title = "Feature Comparison",
            width = 9,
            status = "primary",
            plotlyOutput("cmp_bar_plot", height = "380px")
          )
        ),
        fluidRow(
          box(
            title = "Happiness Score & Rank",
            width = 6,
            status = "info",
            tableOutput("cmp_score_table")
          ),
          box(
            title = "Score Comparison",
            width = 6,
            status = "success",
            plotlyOutput("cmp_score_plot", height = "280px")
          )
        )
      )
      
    )
  )
)