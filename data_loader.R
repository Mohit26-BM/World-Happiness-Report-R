# data_loader.R
# ─────────────────────────────────────────────
# Packages
# ─────────────────────────────────────────────
library(caTools)

# ─────────────────────────────────────────────
# Constants — single source of truth
# ─────────────────────────────────────────────

# Features used in ALL models
MODEL_FEATURES <- c("Economy", "Family", "Health",
                    "Freedom", "Generosity", "Corruption",
                    "Job.Satisfaction")

# Columns kept for visualization / UI only (never fed into models)
META_COLS <- c("Country", "Region",
               "Happiness.Rank", "Happiness.Score", "Happiness.Level")

# ─────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────
load_data <- function(path = "data/World_Happiness_Cleaned.csv") {
  
  data <- read.csv(path, stringsAsFactors = FALSE)
  
  # Drop Dystopia — it is a WHR residual, not a real measured variable
  data$Dystopia.Residual <- NULL      # drop if present under this name
  data$Dystopia          <- NULL      # drop if present under this name
  
  # Create classification target
  data$Happiness.Level <- cut(
    data$Happiness.Score,
    breaks = c(-Inf, 5, 7, Inf),
    labels = c("Low", "Medium", "High"),
    right  = FALSE
  )
  data$Happiness.Level <- factor(data$Happiness.Level,
                                 levels = c("Low", "Medium", "High"))
  
  # Coerce feature columns to numeric (safety net for CSV quirks)
  for (col in MODEL_FEATURES) {
    if (col %in% names(data)) {
      data[[col]] <- as.numeric(data[[col]])
    }
  }
  
  # Drop rows with any NA in model features or target
  data <- data[complete.cases(data[, c(MODEL_FEATURES,
                                       "Happiness.Score",
                                       "Happiness.Level")]), ]
  
  return(data)
}

# ─────────────────────────────────────────────
# 2. Train / test split  (80 / 20, fixed seed)
# ─────────────────────────────────────────────
split_data <- function(data, ratio = 0.8) {
  
  set.seed(123)
  idx   <- caTools::sample.split(data$Happiness.Score, SplitRatio = ratio)
  train <- subset(data, idx == TRUE)
  test  <- subset(data, idx == FALSE)
  
  return(list(train = train, test = test))
}

# ─────────────────────────────────────────────
# 3. Scaled versions  (for KNN & Naive Bayes only)
#    Scale is FIT on train, APPLIED to test
# ─────────────────────────────────────────────
get_scaled <- function(train, test) {
  
  train_scaled <- scale(train[, MODEL_FEATURES])
  
  # Reuse train's centre & scale so test never leaks into fit
  test_scaled  <- scale(test[, MODEL_FEATURES],
                        center = attr(train_scaled, "scaled:center"),
                        scale  = attr(train_scaled, "scaled:scale"))
  
  # Attach targets back so model files have everything in one object
  train_scaled_df           <- as.data.frame(train_scaled)
  train_scaled_df$Happiness.Level <- train$Happiness.Level
  train_scaled_df$Happiness.Score <- train$Happiness.Score
  
  test_scaled_df            <- as.data.frame(test_scaled)
  test_scaled_df$Happiness.Level  <- test$Happiness.Level
  test_scaled_df$Happiness.Score  <- test$Happiness.Score
  
  return(list(
    train = train_scaled_df,
    test  = test_scaled_df,
    # keep scaling params for the Predict module (user input must be scaled too)
    center = attr(train_scaled, "scaled:center"),
    scale  = attr(train_scaled, "scaled:scale")
  ))
}

# ─────────────────────────────────────────────
# 4. Feature matrix helpers  (for XGBoost)
#    Returns plain matrices, no factor columns
# ─────────────────────────────────────────────
get_xgb_matrices <- function(train, test) {
  
  x_train <- as.matrix(train[, MODEL_FEATURES])
  x_test  <- as.matrix(test[,  MODEL_FEATURES])
  
  # Classification labels: 0-based integer
  label_map    <- c("Low" = 0L, "Medium" = 1L, "High" = 2L)
  y_train_cls  <- label_map[as.character(train$Happiness.Level)]
  y_test_cls   <- label_map[as.character(test$Happiness.Level)]
  
  # Regression labels: raw score
  y_train_reg  <- train$Happiness.Score
  y_test_reg   <- test$Happiness.Score
  
  return(list(
    x_train     = x_train,   x_test     = x_test,
    y_train_cls = y_train_cls, y_test_cls = y_test_cls,
    y_train_reg = y_train_reg, y_test_reg = y_test_reg,
    label_map   = label_map
  ))
}