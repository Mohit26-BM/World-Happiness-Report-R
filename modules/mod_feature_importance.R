# modules/mod_feature_importance.R
# ─────────────────────────────────────────────
library(ggplot2)

mod_feature_importance_server <- function(id, importance_bundle) {
  moduleServer(id, function(input, output, session) {
    
    # ── Single model importance plot ─────────────────
    output$importance_plot <- renderPlotly({
      
      imp_raw <- importance_bundle[[input$feat_model_select]]
      
      # Normalise different importance formats into Feature + Score
      if ("Gain" %in% colnames(imp_raw)) {
        # XGBoost format
        imp_df <- data.frame(
          Feature = imp_raw$Feature,
          Score   = imp_raw$Gain
        )
      } else if ("MeanDecreaseAccuracy" %in% colnames(imp_raw)) {
        # RF classifier format
        imp_df <- data.frame(
          Feature = imp_raw$Feature,
          Score   = imp_raw$MeanDecreaseAccuracy
        )
      } else if ("%IncMSE" %in% colnames(imp_raw)) {
        # RF regressor format
        imp_df <- data.frame(
          Feature = imp_raw$Feature,
          Score   = imp_raw$`%IncMSE`
        )
      } else {
        # Decision Tree — variable importance from rpart
        imp_df <- data.frame(
          Feature = names(imp_raw),
          Score   = as.numeric(imp_raw)
        )
      }
      
      imp_df <- imp_df[order(-imp_df$Score), ]
      
      p <- ggplot(imp_df,
                  aes(x = reorder(Feature, Score),
                      y = Score, fill = Score)) +
        geom_col() +
        coord_flip() +
        scale_fill_viridis_c(option = "plasma") +
        labs(title = paste("Feature Importance —",
                           input$feat_model_select),
             x = "Feature", y = "Importance Score") +
        theme_minimal() +
        theme(legend.position = "none")
      
      plotly::ggplotly(p)
    })
    
    # ── Side-by-side RF vs XGBoost comparison ────────
    output$importance_compare_plot <- renderPlotly({
      
      rf_imp <- data.frame(
        Feature = importance_bundle$rf_cls$Feature,
        Score   = importance_bundle$rf_cls$MeanDecreaseAccuracy,
        Model   = "Random Forest"
      )
      
      xgb_imp <- data.frame(
        Feature = importance_bundle$xgb_cls$Feature,
        Score   = importance_bundle$xgb_cls$Gain,
        Model   = "XGBoost"
      )
      
      # Normalise scores to 0-1 so they're comparable
      rf_imp$Score  <- rf_imp$Score  / max(rf_imp$Score)
      xgb_imp$Score <- xgb_imp$Score / max(xgb_imp$Score)
      
      combined <- rbind(rf_imp, xgb_imp)
      
      p <- ggplot(combined,
                  aes(x = reorder(Feature, Score),
                      y = Score, fill = Model)) +
        geom_col(position = "dodge") +
        coord_flip() +
        scale_fill_manual(values = c("Random Forest" = "steelblue",
                                     "XGBoost"       = "coral")) +
        labs(title = "RF vs XGBoost — Normalised Feature Importance",
             x = "Feature", y = "Normalised Importance") +
        theme_minimal()
      
      plotly::ggplotly(p)
    })
  })
}