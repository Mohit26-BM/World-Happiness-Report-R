plot_comparison_metric <- function(metrics, metric_name, title) {
  p <- ggplot(metrics, aes(x = Model, y = .data[[metric_name]], fill = Model)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = paste0(round(.data[[metric_name]], 1), "%")), vjust = -0.3) +
    labs(title = title, y = paste0(metric_name, " (%)")) +
    theme_minimal() +
    theme(legend.position = "none")
  plotly::ggplotly(p)
}
