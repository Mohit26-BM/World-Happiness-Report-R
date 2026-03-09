# models/model_random_forest.R
# ─────────────────────────────────────────────
library(randomForest)

# ─────────────────────────────────────────────
# Classification  →  predicts Happiness.Level
# ─────────────────────────────────────────────
train_rf_classifier <- function(train, test, ntree = 100) {
  
  formula <- as.formula(
    paste("Happiness.Level ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model <- randomForest(
    formula,
    data       = train,
    ntree      = ntree,
    importance = TRUE,
    seed       = 123
  )
  
  preds      <- predict(model, test)
  preds      <- factor(preds, levels = c("Low", "Medium", "High"))
  actuals    <- factor(test$Happiness.Level, levels = c("Low", "Medium", "High"))
  conf_mat   <- table(Actual = actuals, Predicted = preds)
  
  # Importance as tidy data frame for plotting
  imp_df <- as.data.frame(importance(model))
  imp_df$Feature <- rownames(imp_df)
  rownames(imp_df) <- NULL
  # MeanDecreaseAccuracy = classification importance metric
  imp_df <- imp_df[order(-imp_df$MeanDecreaseAccuracy), ]
  
  return(list(
    model        = model,
    predictions  = preds,
    actuals      = actuals,
    conf_matrix  = conf_mat,
    importance   = imp_df,
    type         = "classification"
  ))
}

# ─────────────────────────────────────────────
# Regression  →  predicts Happiness.Score
# ─────────────────────────────────────────────
train_rf_regressor <- function(train, test, ntree = 100) {
  
  formula <- as.formula(
    paste("Happiness.Score ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model <- randomForest(
    formula,
    data       = train,
    ntree      = ntree,
    importance = TRUE,
    seed       = 123
  )
  
  preds   <- predict(model, test)
  actuals <- test$Happiness.Score
  
  # Regression importance metric = %IncMSE
  imp_df <- as.data.frame(importance(model))
  imp_df$Feature <- rownames(imp_df)
  rownames(imp_df) <- NULL
  imp_df <- imp_df[order(-imp_df$`%IncMSE`), ]
  
  # Regression metrics
  residuals <- actuals - preds
  rmse      <- sqrt(mean(residuals^2))
  mae       <- mean(abs(residuals))
  ss_res    <- sum(residuals^2)
  ss_tot    <- sum((actuals - mean(actuals))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  return(list(
    model       = model,
    predictions = preds,
    actuals     = actuals,
    residuals   = residuals,
    importance  = imp_df,
    metrics     = list(
      RMSE      = round(rmse, 4),
      MAE       = round(mae, 4),
      R_Squared = round(r_squared, 4)
    ),
    type        = "regression"
  ))
}