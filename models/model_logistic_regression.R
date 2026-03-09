# models/model_logistic_regression.R
# ─────────────────────────────────────────────
library(nnet)

run_logistic_regression <- function(train, test) {
  
  formula <- as.formula(
    paste("Happiness.Level ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  # Multinomial logistic regression
  # MaxNWts increased for 7 features x 3 classes
  model <- nnet::multinom(
    formula,
    data    = train,
    maxit   = 500,
    MaxNWts = 2000,
    trace   = FALSE    # suppress training output
  )
  
  pred    <- predict(model, newdata = test)
  pred    <- factor(pred,               levels = c("Low", "Medium", "High"))
  actuals <- factor(test$Happiness.Level, levels = c("Low", "Medium", "High"))
  cm      <- table(Actual = actuals, Predicted = pred)
  
  # ── Coefficients as feature importance ──────────
  coef_mat <- as.data.frame(coef(model))
  coef_mat$Class   <- rownames(coef_mat)
  rownames(coef_mat) <- NULL
  
  # Tidy format for plotting
  coef_long <- tidyr::pivot_longer(
    coef_mat,
    cols      = -Class,
    names_to  = "Feature",
    values_to = "Coefficient"
  )
  # Drop intercept row
  coef_long <- coef_long[coef_long$Feature != "(Intercept)", ]
  
  return(list(
    model      = model,
    pred       = pred,
    actuals    = actuals,
    cm         = cm,
    coef_long  = coef_long,
    type       = "classification"
  ))
}