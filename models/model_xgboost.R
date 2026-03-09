# models/model_XGboost.R
# ─────────────────────────────────────────────
library(xgboost)

# ─────────────────────────────────────────────
# Classification  →  predicts Happiness.Level
# ─────────────────────────────────────────────
train_xgb_classifier <- function(xgb_data, nrounds = 100) {
  # xgb_data comes from get_xgb_matrices() in data_loader.R
  dtrain <- xgb.DMatrix(data  = xgb_data$x_train, label = xgb_data$y_train_cls)
  dtest  <- xgb.DMatrix(data  = xgb_data$x_test, label = xgb_data$y_test_cls)
  
  params <- list(
    objective   = "multi:softmax",
    num_class   = 3,
    # Low / Medium / High
    max_depth   = 4,
    eta         = 0.1,
    subsample   = 0.8,
    eval_metric = "merror",
    # multi class error
    seed    = 123
  )
  
  model <- xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = nrounds,
    verbose = 0,
  )
  
  # Decode 0-based integers back to labels
  inv_map      <- c("0" = "Low", "1" = "Medium", "2" = "High")
  raw_preds    <- predict(model, dtest)
  preds        <- factor(inv_map[as.character(raw_preds)], levels = c("Low", "Medium", "High"))
  actuals      <- factor(inv_map[as.character(xgb_data$y_test_cls)], levels = c("Low", "Medium", "High"))
  conf_mat     <- table(Actual = actuals, Predicted = preds)
  
  # Importance
  imp_df <- xgb.importance(feature_names = MODEL_FEATURES, model         = model)
  # imp_df already a data.frame with Feature, Gain, Cover, Frequency cols
  
  return(
    list(
      model       = model,
      predictions = preds,
      actuals     = actuals,
      conf_matrix = conf_mat,
      importance  = imp_df,
      inv_map     = inv_map,
      type        = "classification"
    )
  )
}

# ─────────────────────────────────────────────
# Regression  →  predicts Happiness.Score
# ─────────────────────────────────────────────
train_xgb_regressor <- function(xgb_data, nrounds = 100) {
  dtrain <- xgb.DMatrix(data  = xgb_data$x_train, label = xgb_data$y_train_reg)
  dtest  <- xgb.DMatrix(data  = xgb_data$x_test, label = xgb_data$y_test_reg)
  
  params <- list(
    objective   = "reg:squarederror",
    max_depth   = 4,
    eta         = 0.1,
    subsample   = 0.8,
    eval_metric = "rmse",
    seed    = 123
  )
  
  model <- xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  preds     <- predict(model, dtest)
  actuals   <- xgb_data$y_test_reg
  residuals <- actuals - preds
  
  rmse      <- sqrt(mean(residuals^2))
  mae       <- mean(abs(residuals))
  ss_res    <- sum(residuals^2)
  ss_tot    <- sum((actuals - mean(actuals))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  imp_df <- xgb.importance(feature_names = MODEL_FEATURES, model         = model)
  
  return(
    list(
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
    )
  )
}