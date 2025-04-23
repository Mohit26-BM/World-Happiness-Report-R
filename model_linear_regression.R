run_linear_regression <- function(train, test) {
  model <- lm(Happiness.Score ~ Economy + Family + Health + Freedom + Generosity + Corruption, data = train)
  pred <- predict(model, newdata = test)
  residuals <- test$Happiness.Score - pred
  return(list(model = model, pred = pred, residuals = residuals))
}
