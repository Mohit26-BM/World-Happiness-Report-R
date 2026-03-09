# modules/mod_predict.R
# ─────────────────────────────────────────────
library(class)

mod_predict_server <- function(id, models) {
  moduleServer(id, function(input, output, session) {
    
    # ── Prediction logic (fires on button click) ─────
    prediction <- eventReactive(input$predict_btn, {
      
      # Build raw input data frame
      new_data <- data.frame(
        Economy        = input$p_gdp,
        Family         = input$p_family,
        Health         = input$p_health,
        Freedom        = input$p_freedom,
        Generosity     = input$p_generosity,
        Corruption     = input$p_corruption,
        Job.Satisfaction = input$p_jobsat
      )
      
      # ── Scaled version for KNN & Naive Bayes ───────
      new_scaled <- scale(
        new_data,
        center = models$scale_center,
        scale  = models$scale_scale
      )
      new_scaled_df <- as.data.frame(new_scaled)
      
      # ── Classification predictions ──────────────────
      pred_nb  <- as.character(predict(models$nb, newdata = new_data))
      
      pred_dt  <- as.character(predict(models$dt,
                                       newdata = new_data, type = "class"))
      
      pred_knn <- as.character(class::knn(
        train = models$knn_train[, MODEL_FEATURES],
        test  = new_scaled_df[,   MODEL_FEATURES],
        cl    = models$knn_train$Happiness.Level,
        k     = models$knn_k
      ))
      
      pred_rf  <- as.character(predict(models$rf,  newdata = new_data))
      
      # XGBoost needs matrix input
      xgb_mat  <- xgb.DMatrix(as.matrix(new_data))
      pred_xgb <- models$xgb_inv_map[
        as.character(predict(models$xgb, xgb_mat))
      ]
      
      cls_results <- data.frame(
        Model      = c("Naive Bayes", "Decision Tree", "KNN",
                       "Random Forest", "XGBoost"),
        Prediction = c(pred_nb, pred_dt, pred_knn,
                       pred_rf, pred_xgb),
        stringsAsFactors = FALSE
      )
      
      # ── Regression predictions ──────────────────────
      # Linear Regression needs model stored — add to bundle
      # RF and XGBoost regressor stored separately
      pred_lr_reg  <- as.numeric(predict(models$lr_reg,
                                         newdata = new_data))
      pred_dt_reg  <- as.numeric(predict(models$dt_reg,
                                         newdata = new_data))
      pred_rf_reg  <- as.numeric(predict(models$rf_reg,
                                         newdata = new_data))
      xgb_reg_mat  <- xgb.DMatrix(as.matrix(new_data))
      pred_xgb_reg <- as.numeric(predict(models$xgb_reg, xgb_reg_mat))
      
      reg_results <- data.frame(
        Model = c("Linear Regression", "Decision Tree",
                  "Random Forest",     "XGBoost"),
        Predicted_Score = round(c(pred_lr_reg, pred_dt_reg,
                                  pred_rf_reg, pred_xgb_reg), 3),
        stringsAsFactors = FALSE
      )
      
      list(cls = cls_results, reg = reg_results)
    })
    
    # ── Classification table ─────────────────────────
    output$pred_cls_table <- renderTable({
      req(prediction())
      prediction()$cls
    }, striped = TRUE, hover = TRUE)
    
    # ── Classification consensus plot ────────────────
    output$pred_cls_plot <- renderPlotly({
      req(prediction())
      df  <- prediction()$cls
      counts <- as.data.frame(table(df$Prediction))
      colnames(counts) <- c("Level", "Votes")
      
      # Ensure all 3 levels always appear
      all_levels <- data.frame(Level = c("Low", "Medium", "High"))
      counts <- merge(all_levels, counts, by = "Level", all.x = TRUE)
      counts$Votes[is.na(counts$Votes)] <- 0
      
      p <- ggplot(counts,
                  aes(x = Level, y = Votes, fill = Level)) +
        geom_col(width = 0.5) +
        scale_fill_manual(values = c("Low"    = "tomato",
                                     "Medium" = "gold",
                                     "High"   = "steelblue")) +
        labs(title = "Model Consensus",
             x = "Happiness Level", y = "Number of Models") +
        theme_minimal() +
        theme(legend.position = "none")
      plotly::ggplotly(p)
    })
    
    # ── Regression table ─────────────────────────────
    output$pred_reg_table <- renderTable({
      req(prediction())
      prediction()$reg
    }, striped = TRUE, hover = TRUE)
    
    # ── Regression score plot ────────────────────────
    output$pred_reg_plot <- renderPlotly({
      req(prediction())
      df <- prediction()$reg
      p  <- ggplot(df,
                   aes(x = reorder(Model, Predicted_Score),
                       y = Predicted_Score,
                       fill = Predicted_Score)) +
        geom_col(width = 0.5) +
        geom_text(aes(label = round(Predicted_Score, 2)),
                  hjust = -0.2, size = 3.5) +
        coord_flip() +
        scale_fill_gradient(low = "gold", high = "steelblue") +
        scale_y_continuous(limits = c(0, 8)) +
        labs(title = "Predicted Happiness Score",
             x = "", y = "Score") +
        theme_minimal() +
        theme(legend.position = "none")
      plotly::ggplotly(p)
    })
  })
}