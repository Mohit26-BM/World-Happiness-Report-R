# server.R
library(tidyr)
library(dplyr)
source("data_loader.R")
source("utils.R")
source("metrics_plotting.R")
source("models/model_knn.R") 
source("models/model_logistic_regression.R") #######
source("models/model_naive_bayes.R")
source("models/model_decision_tree.R")
source("models/model_linear_regression.R")
source("models/model_random_forest.R")
source("models/model_xgboost.R")
source("modules/mod_predict.R")
source("modules/mod_feature_importance.R")
source("modules/mod_country_compare.R")

server <- function(input, output, session) {
  # ── 1. Data pipeline ────────────────────────────────
  data        <- load_data()
  splits      <- split_data(data)
  scaled      <- get_scaled(splits$train, splits$test)
  xgb_data    <- get_xgb_matrices(splits$train, splits$test)
  train       <- splits$train
  test        <- splits$test
  
  # ── 2. Train all models once ────────────────────────
  knn_res     <- run_knn(scaled$train, scaled$test)
  nb_res      <- run_naive_bayes(train, test)
  dt_cls_res  <- run_decision_tree_classifier(train, test)
  dt_reg_res  <- run_decision_tree_regressor(train, test)
  lr_cls_res <- run_logistic_regression(train, test) # Logistic Regression
  lr_res      <- run_linear_regression(train, test)
  rf_cls_res  <- train_rf_classifier(train, test)
  rf_reg_res  <- train_rf_regressor(train, test)
  xgb_cls_res <- train_xgb_classifier(xgb_data)
  xgb_reg_res <- train_xgb_regressor(xgb_data)
  
  # ── 3. Classification metrics ────────────────────────
  knn_m     <- calculate_metrics(knn_res$cm)
  nb_m      <- calculate_metrics(nb_res$cm)
  dt_m      <- calculate_metrics(dt_cls_res$cm)        
  rf_cls_m  <- calculate_metrics(rf_cls_res$conf_matrix)
  xgb_cls_m <- calculate_metrics(xgb_cls_res$conf_matrix)
  lr_cls_m <- calculate_metrics(lr_cls_res$cm) # Logistic REgression metrics
  
  metrics_df <- data.frame(
    Model     = c(
      "KNN",
      "Naive Bayes",
      "Decision Tree",
      "Random Forest",
      "XGBoost", "Logistic Regression"
    ),
    Accuracy  = c(
      knn_m$accuracy,
      nb_m$accuracy,
      dt_m$accuracy,lr_cls_m$accuracy,
      rf_cls_m$accuracy,
      xgb_cls_m$accuracy
    )   * 100,
    Recall    = c(
      knn_m$mean_recall,
      nb_m$mean_recall,
      dt_m$mean_recall,
      rf_cls_m$mean_recall,lr_cls_m$mean_recall,
      xgb_cls_m$mean_recall
    ) * 100,
    Precision = c(
      knn_m$mean_precision,
      nb_m$mean_precision,
      dt_m$mean_precision,
      rf_cls_m$mean_precision,lr_cls_m$mean_precision,
      xgb_cls_m$mean_precision
    ) * 100,
    F1_Score  = c(
      knn_m$mean_f1,
      nb_m$mean_f1,
      dt_m$mean_f1,
      rf_cls_m$mean_f1,lr_cls_m$mean_f1,
      xgb_cls_m$mean_f1
    )    * 100
  )
  
  # ── 4. Regression metrics ────────────────────────────
  reg_metrics_df <- data.frame(
    Model     = c(
      "Linear Regression",
      "Decision Tree",
      # ← fixed
      "Random Forest",
      "XGBoost"
    ),
    RMSE      = c(
      lr_res$metrics$RMSE,
      dt_reg_res$metrics$RMSE,
      rf_reg_res$metrics$RMSE,
      xgb_reg_res$metrics$RMSE
    ),
    MAE       = c(
      lr_res$metrics$MAE,
      dt_reg_res$metrics$MAE,
      rf_reg_res$metrics$MAE,
      xgb_reg_res$metrics$MAE
    ),
    R_Squared = c(
      lr_res$metrics$R_Squared,
      dt_reg_res$metrics$R_Squared,
      rf_reg_res$metrics$R_Squared,
      xgb_reg_res$metrics$R_Squared
    )
  )
  
  # ── 5. Confusion matrices ────────────────────────────
  output$knn_matrix     <- renderPlotly({
    render_confusion_matrix(knn_res$cm, "KNN")
  })
  output$nb_matrix      <- renderPlotly({
    render_confusion_matrix(nb_res$cm, "Naive Bayes")
  })
  output$dt_cls_matrix  <- renderPlotly({
    render_confusion_matrix(dt_cls_res$cm, "Decision Tree")
  })
  output$rf_cls_matrix  <- renderPlotly({
    render_confusion_matrix(rf_cls_res$conf_matrix, "Random Forest")
  })
  output$xgb_cls_matrix <- renderPlotly({
    render_confusion_matrix(xgb_cls_res$conf_matrix, "XGBoost")
  })
  output$lr_cls_matrix <- renderPlotly({
    render_confusion_matrix(lr_cls_res$cm, "Logistic Regression")
  })
  
  output$lr_cls_coef_plot <- renderPlotly({
    p <- ggplot(lr_cls_res$coef_long,
                aes(x = reorder(Feature, Coefficient),
                    y = Coefficient, fill = Class)) +
      geom_col(position = "dodge") +
      coord_flip() +
      scale_fill_manual(values = c("Medium" = "steelblue",
                                   "High"   = "coral")) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 color = "black", linewidth = 0.5) +
      labs(title = "Logistic Regression — Coefficients by Class",
           x = "Feature", y = "Coefficient",
           caption = "Positive = increases probability of that class") +
      theme_minimal() +
      theme(legend.position = "top")
    plotly::ggplotly(p)
  })
  
  # ── 6. Metric comparison charts ──────────────────────
  output$accuracy_plot  <- renderPlotly({
    plot_comparison_metric(metrics_df, "Accuracy", "Accuracy Comparison")
  })
  output$recall_plot    <- renderPlotly({
    plot_comparison_metric(metrics_df, "Recall", "Recall Comparison")
  })
  output$precision_plot <- renderPlotly({
    plot_comparison_metric(metrics_df, "Precision", "Precision Comparison")
  })
  output$f1_plot        <- renderPlotly({
    plot_comparison_metric(metrics_df, "F1_Score", "F1 Score Comparison")
  })
  
  # ── 7. Metrics tables ────────────────────────────────
  output$cls_metrics_table <- renderTable({
    metrics_df
  }, digits = 1, striped = TRUE, hover = TRUE)
  
  output$reg_metrics_table <- renderTable({
    reg_metrics_df
  }, digits = 4, striped = TRUE, hover = TRUE)
  
  # ── 8. Decision tree visualisation ───────────────────
  output$decision_tree_plot <- renderPlot({
    rpart.plot::rpart.plot(
      dt_cls_res$model,
      # ← fixed
      main          = "Decision Tree — Happiness Level",
      type          = 3,
      extra         = 102,
      fallen.leaves = TRUE
    )
  })
  
  # ── 9. Regression plots ──────────────────────────────
  make_actual_pred_plot <- function(actuals, preds, model_name) {
    df <- data.frame(Actual = actuals, Predicted = preds)
    p  <- ggplot(df, aes(x = Actual, y = Predicted)) +
      geom_point(color = "steelblue", alpha = 0.7) +
      geom_abline(
        slope = 1,
        intercept = 0,
        color = "red",
        linetype = "dashed"
      ) +
      labs(
        title = paste(model_name, "— Actual vs Predicted"),
        x = "Actual Happiness Score",
        y = "Predicted Happiness Score"
      ) +
      theme_minimal()
    plotly::ggplotly(p)
  }
  
  make_residual_plot <- function(preds, residuals, model_name) {
    df <- data.frame(Predicted = preds, Residuals = residuals)
    p  <- ggplot(df, aes(x = Predicted, y = Residuals)) +
      geom_point(color = "coral", alpha = 0.7) +
      geom_hline(yintercept = 0,
                 color = "red",
                 linetype = "dashed") +
      labs(
        title = paste(model_name, "— Residuals vs Predicted"),
        x = "Predicted Happiness Score",
        y = "Residuals"
      ) +
      theme_minimal()
    plotly::ggplotly(p)
  }
  
  output$lr_actual_pred  <- renderPlotly({
    make_actual_pred_plot(lr_res$actuals, lr_res$pred, "Linear Regression")
  })
  output$lr_residuals    <- renderPlotly({
    make_residual_plot(lr_res$pred, lr_res$residuals, "Linear Regression")
  })
  output$dt_actual_pred  <- renderPlotly({
    # ← new
    make_actual_pred_plot(dt_reg_res$actuals, dt_reg_res$pred, "Decision Tree")
  })
  output$dt_residuals    <- renderPlotly({
    # ← new
    make_residual_plot(dt_reg_res$pred, dt_reg_res$residuals, "Decision Tree")
  })
  output$rf_actual_pred  <- renderPlotly({
    make_actual_pred_plot(rf_reg_res$actuals,
                          rf_reg_res$predictions,
                          "Random Forest")
  })
  output$rf_residuals    <- renderPlotly({
    make_residual_plot(rf_reg_res$predictions,
                       rf_reg_res$residuals,
                       "Random Forest")
  })
  output$xgb_actual_pred <- renderPlotly({
    make_actual_pred_plot(xgb_reg_res$actuals, xgb_reg_res$predictions, "XGBoost")
  })
  output$xgb_residuals   <- renderPlotly({
    make_residual_plot(xgb_reg_res$predictions,
                       xgb_reg_res$residuals,
                       "XGBoost")
  })
  
  # ── 10. Overview outputs ─────────────────────────────
  output$total_countries <- renderValueBox({
    valueBox(nrow(data),
             "Countries",
             icon = icon("globe"),
             color = "blue")
  })
  output$num_features    <- renderValueBox({
    valueBox(
      length(MODEL_FEATURES),
      "Features",
      icon = icon("list"),
      color = "purple"
    )
  })
  output$best_classifier <- renderValueBox({
    valueBox(
      "Naive Bayes",
      "Best Classifier",
      icon = icon("robot"),
      color = "green"
    )
  })
  output$best_regressor  <- renderValueBox({
    valueBox(
      "Linear Regression",
      "Best Regressor",
      icon = icon("chart-line"),
      color = "orange"
    )
  })
  
  output$score_distribution <- renderPlotly({
    p <- ggplot(data, aes(x = Happiness.Score)) +
      geom_histogram(
        bins = 20,
        fill = "steelblue",
        color = "white",
        alpha = 0.8
      ) +
      geom_vline(
        xintercept = 5,
        color = "orange",
        linetype = "dashed",
        linewidth = 0.8
      ) +
      geom_vline(
        xintercept = 7,
        color = "red",
        linetype = "dashed",
        linewidth = 0.8
      ) +
      annotate(
        "text",
        x = 3.8,
        y = Inf,
        label = "Low",
        vjust = 2,
        color = "orange"
      ) +
      annotate(
        "text",
        x = 6.0,
        y = Inf,
        label = "Medium",
        vjust = 2,
        color = "steelblue"
      ) +
      annotate(
        "text",
        x = 7.3,
        y = Inf,
        label = "High",
        vjust = 2,
        color = "red"
      ) +
      labs(x = "Happiness Score", y = "Count") +
      theme_minimal()
    plotly::ggplotly(p)
  })
  
  output$level_pie <- renderPlotly({
    pie_df <- as.data.frame(table(data$Happiness.Level))
    colnames(pie_df) <- c("Level", "Count")
    plotly::plot_ly(
      pie_df,
      labels = ~ Level,
      values = ~ Count,
      type = "pie",
      marker = list(colors = c("tomato", "steelblue", "gold"))
    ) %>% plotly::layout(title = "Happiness Level Breakdown")
  })
  
  output$top10_plot <- renderPlotly({
    top10 <- head(data[order(-data$Happiness.Score), ], 10)
    p <- ggplot(top10,
                aes(
                  x = reorder(Country, Happiness.Score),
                  y = Happiness.Score,
                  fill = Happiness.Score
                )) +
      geom_col() +
      coord_flip() +
      scale_fill_gradient(low = "gold", high = "darkgreen") +
      labs(x = "", y = "Happiness Score") +
      theme_minimal() +
      theme(legend.position = "none")
    plotly::ggplotly(p)
  })
  
  output$bottom10_plot <- renderPlotly({
    bottom10 <- head(data[order(data$Happiness.Score), ], 10)
    p <- ggplot(bottom10,
                aes(
                  x = reorder(Country, Happiness.Score),
                  y = Happiness.Score,
                  fill = Happiness.Score
                )) +
      geom_col() +
      coord_flip() +
      scale_fill_gradient(low = "tomato", high = "gold") +
      labs(x = "", y = "Happiness Score") +
      theme_minimal() +
      theme(legend.position = "none")
    plotly::ggplotly(p)
  })
  
  output$corr_heatmap <- renderPlotly({
    corr_mat <- round(cor(data[, MODEL_FEATURES]), 2)
    plotly::plot_ly(
      x      = colnames(corr_mat),
      y      = rownames(corr_mat),
      z      = corr_mat,
      type   = "heatmap",
      colors = colorRamp(c("tomato", "white", "steelblue"))
    ) %>% plotly::layout(title = "Feature Correlation Heatmap")
  })
  
  # ── 11. Shiny modules ────────────────────────────────
  models_bundle <- list(
    nb           = nb_res$model,
    dt           = dt_cls_res$model,
    # ← fixed
    knn_train    = scaled$train,
    knn_k        = knn_res$k,
    rf           = rf_cls_res$model,
    xgb          = xgb_cls_res$model,
    xgb_inv_map  = xgb_cls_res$inv_map,
    lr_reg       = lr_res$model,
    # ← new
    dt_reg       = dt_reg_res$model,
    # ← new
    rf_reg       = rf_reg_res$model,
    # ← new
    xgb_reg      = xgb_reg_res$model,
    # ← new
    scale_center = scaled$center,
    scale_scale  = scaled$scale
  )
  
  importance_bundle <- list(
    dt_cls  = dt_cls_res$model$variable.importance,
    dt_reg  = dt_reg_res$model$variable.importance,
    rf_cls  = rf_cls_res$importance,
    rf_reg  = rf_reg_res$importance,
    xgb_cls = xgb_cls_res$importance,
    xgb_reg = xgb_reg_res$importance
  )
  
  output$importance_plot <- renderPlotly({
    req(input$feat_model_select)
    
    imp_raw <- importance_bundle[[input$feat_model_select]]
    
    if (is.null(imp_raw) || length(imp_raw) == 0) {
      return(NULL)
    } else if (is.data.frame(imp_raw) &&
               "Gain" %in% colnames(imp_raw)) {
      imp_df <- data.frame(Feature = imp_raw$Feature, Score   = imp_raw$Gain)
    } else if (is.data.frame(imp_raw) &&
               "MeanDecreaseAccuracy" %in% colnames(imp_raw)) {
      imp_df <- data.frame(Feature = imp_raw$Feature,
                           Score   = imp_raw$MeanDecreaseAccuracy)
    } else if (is.data.frame(imp_raw) &&
               "%IncMSE" %in% colnames(imp_raw)) {
      imp_df <- data.frame(Feature = imp_raw$Feature,
                           Score   = imp_raw$`%IncMSE`)
    } else {
      imp_df <- data.frame(
        Feature          = names(imp_raw),
        Score            = as.numeric(imp_raw),
        stringsAsFactors = FALSE
      )
    }
    
    imp_df <- imp_df[order(-imp_df$Score), ]
    
    p <- ggplot(imp_df, aes(
      x = reorder(Feature, Score),
      y = Score,
      fill = Score
    )) +
      geom_col() +
      coord_flip() +
      scale_fill_viridis_c(option = "plasma") +
      labs(
        title = paste("Feature Importance —", input$feat_model_select),
        x = "Feature",
        y = "Importance Score"
      ) +
      theme_minimal() +
      theme(legend.position = "none")
    
    plotly::ggplotly(p)
  })
  output$importance_compare_plot <- renderPlotly({
    # Random Forest
    rf_imp <- data.frame(
      Feature = rf_cls_res$importance$Feature,
      Score   = rf_cls_res$importance$MeanDecreaseAccuracy,
      Model   = "Random Forest",
      stringsAsFactors = FALSE
    )
    
    # XGBoost
    xgb_imp <- data.frame(
      Feature = xgb_cls_res$importance$Feature,
      Score   = xgb_cls_res$importance$Gain,
      Model   = "XGBoost",
      stringsAsFactors = FALSE
    )
    
    # Decision Tree — named numeric vector
    dt_imp <- data.frame(
      Feature = names(dt_cls_res$model$variable.importance),
      Score   = as.numeric(dt_cls_res$model$variable.importance),
      Model   = "Decision Tree",
      stringsAsFactors = FALSE
    )
    
    # KNN has no feature importance — skip
    # Naive Bayes has no feature importance — skip
    
    # Normalise each model to 0-1 independently
    rf_imp$Score  <- rf_imp$Score  / max(rf_imp$Score, na.rm = TRUE)
    xgb_imp$Score <- xgb_imp$Score / max(xgb_imp$Score, na.rm = TRUE)
    dt_imp$Score  <- dt_imp$Score  / max(dt_imp$Score, na.rm = TRUE)
    
    combined <- rbind(rf_imp, xgb_imp, dt_imp)
    combined <- combined[!is.na(combined$Score) &
                           !is.nan(combined$Score), ]
    
    # Fix factor order by average score across models
    feature_order <- combined %>%
      group_by(Feature) %>%
      summarise(avg = mean(Score, na.rm = TRUE)) %>%
      arrange(avg) %>%
      pull(Feature)
    
    combined$Feature <- factor(combined$Feature, levels = feature_order)
    
    p <- ggplot(combined, aes(x = Feature, y = Score, fill = Model)) +
      geom_col(position = "dodge") +
      coord_flip() +
      scale_fill_manual(
        values = c(
          "Random Forest" = "steelblue",
          "XGBoost"       = "coral",
          "Decision Tree" = "#F39C12"
        )
      ) +
      labs(title = "Feature Importance — Decision Tree vs Random Forest vs XGBoost", x = "Feature", y = "Normalised Importance (0-1)") +
      theme_minimal() +
      theme(legend.position = "top")
    
    plotly::ggplotly(p)
  })
  
  output$importance_summary_table <- renderTable({
    # ← separate block
    data.frame(
      Feature = c(
        "Job.Satisfaction",
        "Health",
        "Economy",
        "Family",
        "Freedom",
        "Generosity",
        "Corruption"
      ),
      DT_Cls  = c(37.98, 22.36, 18.27, 18.72, 10.21, 0.97, 0.55),
      DT_Reg  = c(102.42, 73.08, 67.71, 54.59, 40.39, 11.39, 16.89),
      RF_Cls  = c(9.27, 9.84, 8.18, 7.15, 4.28, 2.99, -0.49),
      RF_Reg  = c(10.76, 8.06, 8.35, 6.48, 5.38, 2.56, 1.94),
      XGB_Cls = c(0.383, 0.163, 0.145, 0.074, 0.126, 0.057, 0.051),
      XGB_Reg = c(0.281, 0.383, 0.162, 0.057, 0.060, 0.023, 0.033)
    )
  }, striped = TRUE, hover = TRUE, digits = 3)       # ← summary table ends here
  
  # ── Predict ───────────────────────────────────────────
  observeEvent(input$predict_btn, {
    new_data <- data.frame(
      Economy          = input$p_gdp,
      Family           = input$p_family,
      Health           = input$p_health,
      Freedom          = input$p_freedom,
      Generosity       = input$p_generosity,
      Corruption       = input$p_corruption,
      Job.Satisfaction = input$p_jobsat
    )
    
    # Scale for KNN
    new_scaled    <- scale(new_data,
                           center = scaled$center,
                           scale  = scaled$scale)
    new_scaled_df <- as.data.frame(new_scaled)
    
    # Classification predictions
    pred_nb  <- as.character(predict(nb_res$model, newdata = new_data))
    pred_dt  <- as.character(predict(dt_cls_res$model, newdata = new_data, type = "class"))
    pred_knn <- as.character(
      class::knn(
        train = scaled$train[, MODEL_FEATURES],
        test  = new_scaled_df[, MODEL_FEATURES],
        cl    = scaled$train$Happiness.Level,
        k     = knn_res$k
      )
    )
    pred_rf  <- as.character(predict(rf_cls_res$model, newdata = new_data))
    
    xgb_mat  <- xgb.DMatrix(as.matrix(new_data))
    pred_xgb <- xgb_cls_res$inv_map[as.character(predict(xgb_cls_res$model, xgb_mat))]
    
    cls_results <- data.frame(
      Model      = c(
        "Naive Bayes",
        "Decision Tree",
        "KNN",
        "Random Forest",
        "XGBoost"
      ),
      Prediction = c(pred_nb, pred_dt, pred_knn, pred_rf, pred_xgb),
      stringsAsFactors = FALSE
    )
    
    # Regression predictions
    pred_lr_reg  <- round(predict(lr_res$model, newdata = new_data), 3)
    pred_dt_reg  <- round(predict(dt_reg_res$model, newdata = new_data), 3)
    pred_rf_reg  <- round(predict(rf_reg_res$model, newdata = new_data), 3)
    xgb_reg_mat  <- xgb.DMatrix(as.matrix(new_data))
    pred_xgb_reg <- round(predict(xgb_reg_res$model, xgb_reg_mat), 3)
    
    reg_results <- data.frame(
      Model           = c(
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
      ),
      Predicted_Score = c(pred_lr_reg, pred_dt_reg, pred_rf_reg, pred_xgb_reg),
      stringsAsFactors = FALSE
    )
    
    output$pred_cls_table <- renderTable({
      cls_results
    }, striped = TRUE, hover = TRUE)
    
    output$pred_cls_plot  <- renderPlotly({
      counts     <- as.data.frame(table(cls_results$Prediction))
      colnames(counts) <- c("Level", "Votes")
      all_levels <- data.frame(Level = c("Low", "Medium", "High"))
      counts     <- merge(all_levels, counts, by = "Level", all.x = TRUE)
      counts$Votes[is.na(counts$Votes)] <- 0
      
      p <- ggplot(counts, aes(x = Level, y = Votes, fill = Level)) +
        geom_col(width = 0.5) +
        scale_fill_manual(values = c(
          "Low"    = "tomato",
          "Medium" = "gold",
          "High"   = "steelblue"
        )) +
        labs(title = "Model Consensus", x = "Happiness Level", y = "Number of Models") +
        theme_minimal() +
        theme(legend.position = "none")
      plotly::ggplotly(p)
    })
    
    output$pred_reg_table <- renderTable({
      reg_results
    }, striped = TRUE, hover = TRUE)
    
    output$pred_reg_plot  <- renderPlotly({
      p <- ggplot(reg_results,
                  aes(
                    x = reorder(Model, Predicted_Score),
                    y = Predicted_Score,
                    fill = Predicted_Score
                  )) +
        geom_col(width = 0.5) +
        geom_text(aes(label = round(Predicted_Score, 2)),
                  hjust = -0.2,
                  size = 3.5) +
        coord_flip() +
        scale_fill_gradient(low = "gold", high = "steelblue") +
        scale_y_continuous(limits = c(0, 8)) +
        labs(title = "Predicted Happiness Score", x = "", y = "Score") +
        theme_minimal() +
        theme(legend.position = "none")
      plotly::ggplotly(p)
    })
  })
  
  # ── Country Compare ───────────────────────────────────
  observe({
    updateSelectizeInput(session,
                         "cmp_countries",
                         choices = sort(unique(data$Country)),
                         server  = TRUE)
  })
  
  output$cmp_bar_plot <- renderPlotly({
    req(input$cmp_countries, input$cmp_features)
    
    plot_data <- data %>%
      filter(Country %in% input$cmp_countries) %>%
      select(Country, all_of(input$cmp_features)) %>%
      pivot_longer(-Country, names_to  = "Feature", values_to = "Value")
    
    p <- ggplot(plot_data, aes(x = Feature, y = Value, fill = Country)) +
      geom_col(position = "dodge") +
      labs(title = "Feature Comparison Across Countries", x = "Feature", y = "Value") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
    
    plotly::ggplotly(p)
  })
  
  output$cmp_score_table <- renderTable({
    req(input$cmp_countries)
    data %>%
      filter(Country %in% input$cmp_countries) %>%
      select(Country, Happiness.Rank, Happiness.Score, Happiness.Level) %>%
      arrange(Happiness.Rank)
  }, striped = TRUE, hover = TRUE)
  
  output$cmp_score_plot <- renderPlotly({
    req(input$cmp_countries)
    filtered <- data %>%
      filter(Country %in% input$cmp_countries)
    
    p <- ggplot(filtered,
                aes(
                  x = reorder(Country, Happiness.Score),
                  y = Happiness.Score,
                  fill = Happiness.Level
                )) +
      geom_col(width = 0.5) +
      geom_text(aes(label = round(Happiness.Score, 2)), hjust = -0.2, size = 3.5) +
      coord_flip() +
      scale_fill_manual(values = c(
        "Low"    = "tomato",
        "Medium" = "gold",
        "High"   = "steelblue"
      )) +
      scale_y_continuous(limits = c(0, 8)) +
      labs(title = "Happiness Score Comparison", x = "", y = "Happiness Score") +
      theme_minimal()
    
    plotly::ggplotly(p)
  })
}