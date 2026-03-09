# models/model_decision_tree.R
library(rpart)
library(rpart.plot)

# ─────────────────────────────────────────────
# Classification  →  predicts Happiness.Level
# ─────────────────────────────────────────────
run_decision_tree_classifier <- function(train, test) {
  
  formula <- as.formula(
    paste("Happiness.Level ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model   <- rpart::rpart(formula, data = train, method = "class",
                          control = rpart.control(minsplit = 10, cp = 0.01))
  pred    <- predict(model, newdata = test, type = "class")
  pred    <- factor(pred,               levels = c("Low", "Medium", "High"))
  actuals <- factor(test$Happiness.Level, levels = c("Low", "Medium", "High"))
  cm      <- table(Actual = actuals, Predicted = pred)
  
  return(list(
    model   = model,
    pred    = pred,
    actuals = actuals,
    cm      = cm,
    type    = "classification"
  ))
}

# ─────────────────────────────────────────────
# Regression  →  predicts Happiness.Score
# ─────────────────────────────────────────────
run_decision_tree_regressor <- function(train, test) {
  
  formula <- as.formula(
    paste("Happiness.Score ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model     <- rpart::rpart(formula, data = train, method = "anova",
                            control = rpart.control(minsplit = 10, cp = 0.01))
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
    ),
    type      = "regression"
  ))
}