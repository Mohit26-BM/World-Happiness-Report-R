# utils.R

calculate_metrics <- function(cm) {
  TP <- diag(cm)
  FP <- colSums(cm) - TP
  FN <- rowSums(cm) - TP
  TN <- sum(cm) - (TP + FP + FN)
  
  accuracy  <- sum(TP) / sum(cm)
  recall    <- TP / (TP + FN)
  precision <- TP / (TP + FP)
  f1        <- 2 * (precision * recall) / (precision + recall)
  
  return(
    list(
      accuracy       = accuracy,
      recall         = recall,
      precision      = precision,
      f1             = f1,
      mean_recall    = mean(recall, na.rm = TRUE),
      mean_precision = mean(precision, na.rm = TRUE),
      mean_f1        = mean(f1, na.rm = TRUE)
    )
  )
}

render_confusion_matrix <- function(cm, title) {
  df <- as.data.frame(cm)
  colnames(df) <- c("Actual", "Predicted", "Freq")
  
  df$Actual    <- factor(df$Actual,
                         levels = rev(c("Low", "Medium", "High")))
  df$Predicted <- factor(df$Predicted,
                         levels = c("Low", "Medium", "High"))
  
  color_map <- list(
    "KNN"           = c("#EBF5FB", "#2E86C1"),  # soft blue
    "Naive Bayes"   = c("#EAFAF1", "#1E8449"),  # soft green
    "Decision Tree" = c("#FEF9E7", "#D4AC0D"),  # soft yellow
    "Random Forest" = c("#F5EEF8", "#7D3C98"),  # soft purple
    "XGBoost"       = c("#FDEDEC", "#C0392B")   # soft red
  )
  
  chosen_color <- if (title %in% names(color_map)) {
    color_map[[title]]
  } else {
    c("white", "steelblue")
  }
  
  p <- ggplot(df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = Freq), color = "black",
              size = 6, fontface = "bold") +
    scale_fill_gradient(low  = chosen_color[1],
                        high = chosen_color[2]) +
    labs(title = paste(title, "— Confusion Matrix"),
         x = "Predicted", y = "Actual") +
    theme_minimal() +
    theme(
      legend.position   = "none",
      plot.title        = element_text(hjust = 0.5, face = "bold"),
      axis.text         = element_text(size = 11),
      panel.grid        = element_blank()
    )
  
  plotly::ggplotly(p)
}