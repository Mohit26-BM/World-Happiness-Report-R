# models/model_naive_bayes.R
library(e1071)

run_naive_bayes <- function(train, test) {
  
  formula <- as.formula(
    paste("Happiness.Level ~", paste(MODEL_FEATURES, collapse = " + "))
  )
  
  model   <- e1071::naiveBayes(formula, data = train)
  pred    <- predict(model, newdata = test)
  pred    <- factor(pred,          levels = c("Low", "Medium", "High"))
  actuals <- factor(test$Happiness.Level, levels = c("Low", "Medium", "High"))
  cm      <- table(Actual = actuals, Predicted = pred)
  
  return(list(
    model   = model,       # needed by Predict module
    pred    = pred,
    actuals = actuals,
    cm      = cm
  ))
}