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

  # Generate the Venn diagram
  venn_plot <- ggVennDiagram(
    gene_lists,
    label       = "count",     # show counts
    label_geom  = "label",     # draw counts in boxes (like pic 2)
    label_alpha = 0.85,        # box opacity
    label_fill  = "grey85",    # box color behind numbers
    label_color = "grey10",    # text color
    label_size  = 3.6,         # text size
    edge_size   = 0.8          # circle edge thickness
  # ) +
  #   scale_fill_gradient(low = "#c6dbef", high = "#08306b", name = "count") +
  #   theme_void() +                       # clean white background (no dark bg)
  #   theme(
  #     legend.position = "right",
  #     legend.title = element_text(size = 10),
  #     legend.text  = element_text(size = 9)
    )

  # Save the plot in different formats with reg in the filename
  ggsave(paste0("figures/venn_diagram_", reg, ".png"), plot = venn_plot, width = 8, height = 6)
  ggsave(paste0("figures/venn_diagram_", reg, ".pdf"), plot = venn_plot, width = 8, height = 6)
  #ggsave(paste0("figures/venn_diagram_", reg, ".tiff"), plot = venn_plot, width = 8, height = 6)
  #ggsave(paste0("figures/venn_diagram_", reg, ".jpg"), plot = venn_plot, width = 8, height = 6)
}
