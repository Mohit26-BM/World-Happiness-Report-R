calculate_metrics <- function(cm) {
  TP <- diag(cm)
  FP <- colSums(cm) - TP
  FN <- rowSums(cm) - TP
  TN <- sum(cm) - (TP + FP + FN)
  
  accuracy <- sum(TP) / sum(cm)
  recall <- TP / (TP + FN)
  precision <- TP / (TP + FP)
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(list(
    accuracy = accuracy,
    recall = recall,
    precision = precision,
    f1 = f1,
    mean_recall = mean(recall, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE)
  ))
}
