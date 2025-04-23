run_naive_bayes <- function(train, test) {
  model <- e1071::naiveBayes(Happiness.Level ~ Economy + Family + Health + Freedom + Generosity + Corruption, data = train)
  pred <- predict(model, newdata = test)
  cm <- table(Predicted = pred, Actual = test$Happiness.Level)
  return(list(pred = pred, cm = cm))
}
