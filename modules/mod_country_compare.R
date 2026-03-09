# modules/mod_country_compare.R
# ─────────────────────────────────────────────
library(dplyr)
library(tidyr)
library(ggplot2)

mod_country_compare_server <- function(id, happiness_data) {
  moduleServer(id, function(input, output, session) {
    
    # Populate country dropdown from actual data
    observe({
      updateSelectizeInput(
        session, "cmp_countries",
        choices  = sort(unique(happiness_data$Country)),
        server   = TRUE
      )
    })
    
    # Filtered data reactive
    filtered <- reactive({
      req(input$cmp_countries)
      happiness_data %>%
        filter(Country %in% input$cmp_countries)
    })
    
    # ── Feature comparison bar chart ─────────────────
    output$cmp_bar_plot <- renderPlotly({
      req(filtered(), input$cmp_features)
      
      plot_data <- filtered() %>%
        select(Country, all_of(input$cmp_features)) %>%
        pivot_longer(-Country,
                     names_to  = "Feature",
                     values_to = "Value")
      
      p <- ggplot(plot_data,
                  aes(x = Feature, y = Value, fill = Country)) +
        geom_col(position = "dodge") +
        labs(title = "Feature Comparison Across Countries",
             x = "Feature", y = "Value") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 30, hjust = 1))
      
      plotly::ggplotly(p)
    })
    
    # ── Score and rank table ──────────────────────────
    output$cmp_score_table <- renderTable({
      req(filtered())
      filtered() %>%
        select(Country, Happiness.Rank,
               Happiness.Score, Happiness.Level) %>%
        arrange(Happiness.Rank)
    }, striped = TRUE, hover = TRUE)
    
    # ── Score comparison bar chart ────────────────────
    output$cmp_score_plot <- renderPlotly({
      req(filtered())
      
      p <- ggplot(filtered(),
                  aes(x = reorder(Country, Happiness.Score),
                      y = Happiness.Score,
                      fill = Happiness.Level)) +
        geom_col(width = 0.5) +
        geom_text(aes(label = round(Happiness.Score, 2)),
                  hjust = -0.2, size = 3.5) +
        coord_flip() +
        scale_fill_manual(values = c("Low"    = "tomato",
                                     "Medium" = "gold",
                                     "High"   = "steelblue")) +
        scale_y_continuous(limits = c(0, 8)) +
        labs(title = "Happiness Score Comparison",
             x = "", y = "Happiness Score") +
        theme_minimal()
      
      plotly::ggplotly(p)
    })
  })
}