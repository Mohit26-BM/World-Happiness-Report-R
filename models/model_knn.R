# models/model_knn.R
library(class)

run_knn <- function(train_scaled, test_scaled, k = 5) {
  
  # Labels are now attached to scaled data frames
  train_labels <- train_scaled$Happiness.Level
  test_labels  <- factor(test_scaled$Happiness.Level,
                         levels = c("Low", "Medium", "High"))
  
  pred <- class::knn(
    train = train_scaled[, MODEL_FEATURES],
    test  = test_scaled[,  MODEL_FEATURES],
    cl    = train_labels,
    k     = k
  )
  pred    <- factor(pred, levels = c("Low", "Medium", "High"))
  cm      <- table(Actual = test_labels, Predicted = pred)
  
  return(list(
    pred    = pred,
    actuals = test_labels,
    cm      = cm,
    k       = k
  ))
}