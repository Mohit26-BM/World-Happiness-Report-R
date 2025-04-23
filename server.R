source("data_loader.R")
source("model_knn.R")
source("model_naive_bayes.R")
source("model_decision_tree.R")
source("model_linear_regression.R")
source("utils.R")
source("metrics_plotting.R")

server <- function(input, output) {
  data <- load_data()
  split <- split_data(data)
  
  knn_results <- run_knn(split$train_scaled, split$test_scaled, split$train$Happiness.Level, split$test$Happiness.Level)
  nb_results <- run_naive_bayes(split$train, split$test)
  dt_results <- run_decision_tree(split$train, split$test)
  lr_results <- run_linear_regression(split$train, split$test)
  
  knn_metrics <- calculate_metrics(knn_results$cm)
  nb_metrics <- calculate_metrics(nb_results$cm)
  dt_metrics <- calculate_metrics(dt_results$cm)
  
  metrics_df <- data.frame(
    Model = c("KNN", "Naive Bayes", "Decision Tree"),
    Accuracy = c(knn_metrics$accuracy, nb_metrics$accuracy, dt_metrics$accuracy) * 100,
    Recall = c(knn_metrics$mean_recall, nb_metrics$mean_recall, dt_metrics$mean_recall) * 100,
    Precision = c(knn_metrics$mean_precision, nb_metrics$mean_precision, dt_metrics$mean_precision) * 100,
    F1_Score = c(knn_metrics$mean_f1, nb_metrics$mean_f1, dt_metrics$mean_f1) * 100
  )
  
  render_confusion_matrix <- function(cm, title) {
    df <- as.data.frame(cm)
    colnames(df) <- c("Actual", "Predicted", "Freq")
    
    # Keep 'Actual' as row variable and reverse factor levels to fix heatmap direction
    df$Actual <- factor(df$Actual, levels = rev(sort(unique(df$Actual))))
    df$Predicted <- factor(df$Predicted, levels = sort(unique(df$Predicted)))
    
    p <- ggplot(df, aes(x = Predicted, y = Actual, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), color = "black", size = 5) +
      scale_fill_gradient(low = "white", high = "blue") +
      labs(title = title, x = "Predicted", y = "Actual") +
      theme_minimal()
    
    plotly::ggplotly(p)
  }
  
  
  output$knn_matrix <- renderPlotly({ render_confusion_matrix(knn_results$cm, "KNN Confusion Matrix") })
  output$nb_matrix <- renderPlotly({ render_confusion_matrix(nb_results$cm, "Naive Bayes Confusion Matrix") })
  output$dt_matrix <- renderPlotly({ render_confusion_matrix(dt_results$cm, "Decision Tree Confusion Matrix") })
  
  output$f1_plot <- renderPlotly({ plot_comparison_metric(metrics_df, "F1_Score", "F1 Score Comparison") })
  output$accuracy_plot <- renderPlotly({ plot_comparison_metric(metrics_df, "Accuracy", "Accuracy Comparison") })
  output$recall_plot <- renderPlotly({ plot_comparison_metric(metrics_df, "Recall", "Recall Comparison") })
  output$precision_plot <- renderPlotly({ plot_comparison_metric(metrics_df, "Precision", "Precision Comparison") })
  
  output$decision_tree_plot <- renderPlot({
    rpart.plot::rpart.plot(dt_results$model, main = "Decision Tree for Happiness Level", type = 3, extra = 102, fallen.leaves = TRUE)
  })
  
  output$lr_actual_pred <- renderPlotly({
    df <- data.frame(Actual = split$test$Happiness.Score, Predicted = lr_results$pred)
    p <- ggplot(df, aes(x = Actual, y = Predicted)) +
      geom_point() +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = "Actual vs Predicted", x = "Actual Happiness Score", y = "Predicted Happiness Score") +
      theme_minimal()
    plotly::ggplotly(p)
  })
  
  output$lr_residuals_plot <- renderPlotly({
    df <- data.frame(Predicted = lr_results$pred, Residuals = lr_results$residuals)
    p <- ggplot(df, aes(x = Predicted, y = Residuals)) +
      geom_point() +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      labs(title = "Residuals vs Predicted", x = "Predicted Happiness Score", y = "Residuals") +
      theme_minimal()
    plotly::ggplotly(p)
  })
}
