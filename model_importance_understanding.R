source("data_loader.R")
source("utils.R")
source("models/model_decision_tree.R")
source("models/model_random_forest.R")
source("models/model_xgboost.R")

data        <- load_data()
splits      <- split_data(data)
xgb_data    <- get_xgb_matrices(splits$train, splits$test)
train       <- splits$train
test        <- splits$test

dt_cls_res  <- run_decision_tree_classifier(train, test)
dt_reg_res  <- run_decision_tree_regressor(train, test)
rf_cls_res  <- train_rf_classifier(train, test)
rf_reg_res  <- train_rf_regressor(train, test)
xgb_cls_res <- train_xgb_classifier(xgb_data)
xgb_reg_res <- train_xgb_regressor(xgb_data)

cat("\n── Decision Tree Classifier ─────────────────────\n")
print(sort(dt_cls_res$model$variable.importance, decreasing = TRUE))

cat("\n── Decision Tree Regressor ──────────────────────\n")
print(sort(dt_reg_res$model$variable.importance, decreasing = TRUE))

cat("\n── Random Forest Classifier ─────────────────────\n")
print(rf_cls_res$importance[order(-rf_cls_res$importance$MeanDecreaseAccuracy), c("Feature", "MeanDecreaseAccuracy")])

cat("\n── Random Forest Regressor ──────────────────────\n")
print(rf_reg_res$importance[order(-rf_reg_res$importance$`%IncMSE`), c("Feature", "%IncMSE")])

cat("\n── XGBoost Classifier ───────────────────────────\n")
print(xgb_cls_res$importance[, c("Feature", "Gain")])

cat("\n── XGBoost Regressor ────────────────────────────\n")
print(xgb_reg_res$importance[, c("Feature", "Gain")])