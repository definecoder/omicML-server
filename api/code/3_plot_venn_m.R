# Function to install and load required packages
install_and_load <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
  }
  library(package, character.only = TRUE)
}

# Function to create Venn diagram, compute intersections, and save results
plot_and_save_venn <- function(gene_lists, reg) {
  # Check if at least two lists are provided
  if (length(gene_lists) < 2) {
    message("You need at least two gene lists to generate a Venn diagram and compute intersections.")
    return(NULL)
  }

  # Install and load required packages
  install_and_load("ggplot2")
  install_and_load("ggVennDiagram")

  # Generate the Venn diagram with custom hex colors
  venn_plot <- ggVennDiagram(
    gene_lists,
    label       = "count",
    label_geom  = "label",
    label_alpha = 0.85,
    label_fill  = "#1f2937",   # dark gray box background
    label_color = "#f9fafb",   # almost white text
    label_size  = 3.6,
    edge_size   = 0.8
  ) 
  
  
  # +
  #   # Apply gradient fill using hex colors
  #   scale_fill_gradient(
  #     low = "#93c5fd",  # light blue
  #     high = "#1e3a8a", # deep navy blue
  #     name = "Gene Count"
  #   ) +
  #   theme_void() +
  #   theme(
  #     legend.position = "right",
  #     legend.title = element_text(size = 10),
  #     legend.text  = element_text(size = 9)
  #   )

  # Save the plot in different formats with reg in the filename
  ggsave(paste0("figures/venn_diagram_", reg, ".png"), plot = venn_plot, width = 8, height = 6, dpi = 300)
  ggsave(paste0("figures/venn_diagram_", reg, ".pdf"), plot = venn_plot, width = 8, height = 6)
}
