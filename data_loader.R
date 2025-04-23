load_data <- function(path = "data/World Happiness Report.csv") {
  data <- read.csv(path)
  data$Happiness.Level <- ifelse(data$Happiness.Score < 5, "Low",
                                 ifelse(data$Happiness.Score < 7, "Medium", "High"))
  data$Happiness.Level <- as.factor(data$Happiness.Level)
  return(data)
}

split_data <- function(data) {
  set.seed(123)
  split <- caTools::sample.split(data$Happiness.Score, SplitRatio = 0.7)
  train <- subset(data, split == TRUE)
  test <- subset(data, split == FALSE)
  
  train_scaled <- scale(train[, c("Economy", "Family", "Health", "Freedom", "Generosity", "Corruption")])
  test_scaled <- scale(test[, c("Economy", "Family", "Health", "Freedom", "Generosity", "Corruption")],
                       center = attr(train_scaled, "scaled:center"),
                       scale = attr(train_scaled, "scaled:scale"))
  
  return(list(train = train, test = test, train_scaled = train_scaled, test_scaled = test_scaled))
}
