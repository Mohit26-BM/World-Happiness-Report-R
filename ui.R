library(shiny)
library(shinydashboard) 
library(plotly)

ui <- dashboardPage(
  dashboardHeader(title = "Model Analysis Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Confusion Matrices", tabName = "confusion_tab", icon = icon("table")),
      menuItem("Comparison", tabName = "comparison_tab", icon = icon("chart-bar")),
      menuItem("Decision Tree", tabName = "decision_tree_tab", icon = icon("tree")),
      menuItem("Linear Regression", tabName = "linear_regression_tab", icon = icon("chart-line"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "confusion_tab",
        tabsetPanel(
          tabPanel("KNN", plotlyOutput("knn_matrix", height = "400px")),
          tabPanel("Naive Bayes", plotlyOutput("nb_matrix", height = "400px")),
          tabPanel("Decision Tree", plotlyOutput("dt_matrix", height = "400px"))
        )
      ),
      tabItem(
        tabName = "comparison_tab",
        fluidRow(
          box(title = "F1 Score Comparison", width = 12, plotlyOutput("f1_plot", height = "300px")),
          box(title = "Accuracy Comparison", width = 12, plotlyOutput("accuracy_plot", height = "300px")),
          box(title = "Recall Comparison", width = 12, plotlyOutput("recall_plot", height = "300px")),
          box(title = "Precision Comparison", width = 12, plotlyOutput("precision_plot", height = "300px"))
        )
      ),
      tabItem(
        tabName = "decision_tree_tab",
        fluidRow(
          box(title = "Decision Tree Visualization", width = 12, plotOutput("decision_tree_plot", height = "500px"))
        )
      ),
      tabItem(
        tabName = "linear_regression_tab",
        fluidRow(
          box(title = "Linear Regression: Actual vs Predicted", width = 12, plotlyOutput("lr_actual_pred", height = "300px")),
          box(title = "Linear Regression: Residuals vs Predicted", width = 12, plotlyOutput("lr_residuals_plot", height = "300px"))
        )
      )
    )
  )
)