# models/model_linear_regression.R

run_linear_regression <- function(train, test) {
  
  formula <- as.formula(
    paste("Happiness.Score ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model     <- lm(formula, data = train)
  pred      <- predict(model, newdata = test)
  actuals   <- test$Happiness.Score
  residuals <- actuals - pred
  
  ss_res    <- sum(residuals^2)
  ss_tot    <- sum((actuals - mean(actuals))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  rmse      <- sqrt(mean(residuals^2))
  mae       <- mean(abs(residuals))
  
  return(list(
    model     = model,
    pred      = pred,
    actuals   = actuals,
    residuals = residuals,
    metrics   = list(
      RMSE      = round(rmse, 4),
      MAE       = round(mae, 4),
      R_Squared = round(r_squared, 4)
    )
  ))
}