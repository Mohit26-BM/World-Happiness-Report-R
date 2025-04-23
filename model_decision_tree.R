run_decision_tree <- function(train, test) {
  model <- rpart::rpart(Happiness.Level ~ Economy + Family + Health + Freedom + Generosity + Corruption, data = train, method = "class")
  pred <- predict(model, newdata = test, type = "class")
  cm <- table(Predicted = pred, Actual = test$Happiness.Level)
  return(list(model = model, pred = pred, cm = cm))
}
