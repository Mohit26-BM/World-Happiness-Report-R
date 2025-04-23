run_knn <- function(train_scaled, test_scaled, train_labels, test_labels, k = 5) {
  pred <- class::knn(train = train_scaled, test = test_scaled, cl = train_labels, k = k)
  pred <- factor(pred, levels = levels(test_labels))  # Align factor levels!
  cm <- table(Predicted = pred, Actual = test_labels)
  return(list(pred = pred, cm = cm))
}
