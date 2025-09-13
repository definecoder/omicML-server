batch_effect_correction <- function(input_file, output_dir, user_id) {
  library(jsonlite)
  library(sva)
  
  safe_boxpanels <- function(before_mat, after_mat, sample_names) {
    par(mfrow = c(1, 2), mar = c(10, 5, 4, 2))
    boxplot(before_mat,
            main = "Before Batch Correction",
            las = 2,
            col = "lightblue",
            outline = FALSE,
            ylab = "Expression Levels",
            cex.axis = 0.7,
            names = sample_names)
    boxplot(after_mat,
            main = "After Batch Correction",
            las = 2,
            col = "lightgreen",
            outline = FALSE,
            ylab = "Expression Levels",
            cex.axis = 0.7,
            names = sample_names)
  }
  
  tryCatch({
    # Read and preprocess (preserve names exactly)
    merged_df_data <- read.csv(input_file, header = TRUE, row.names = 1, check.names = FALSE)
    merged_df_data <- na.omit(merged_df_data)
    colnames(merged_df_data) <- make.unique(colnames(merged_df_data))
    
    # Split meta vs expression
    condition_info <- merged_df_data$condition
    data_t <- t(merged_df_data[, !(colnames(merged_df_data) %in% c("condition", "batch"))])
    
    # Keep names
    feature_names <- rownames(data_t)
    sample_names  <- colnames(data_t)
    
    # ComBat
    batch_info  <- merged_df_data$batch
    data_combat <- ComBat(dat = as.matrix(data_t), batch = batch_info, par.prior = TRUE, prior.plots = FALSE)
    rownames(data_combat) <- feature_names
    
    # Save corrected table
    output_file <- file.path(output_dir, paste0("batch_", basename(input_file)))
    data_corrected <- t(data_combat)
    out_df <- cbind(condition = condition_info, data_corrected)
    write.csv(out_df, output_file, row.names = TRUE, quote = TRUE, na = "", fileEncoding = "UTF-8")
    
    # Consistent device settings
    width_in  <- 12
    height_in <- 6
    point_sz  <- 12
    dpi       <- 300
    
    # PDF (vector)
    pdf_file <- file.path(output_dir, "batch_correction_boxplots.pdf")
    pdf(pdf_file, width = width_in, height = height_in, family = "sans", useDingbats = FALSE, pointsize = point_sz)
    safe_boxpanels(data_t, data_combat, sample_names)
    dev.off()
    
    # PNG (bitmap) — use Cairo for consistent text/line rendering
    png_file <- file.path(output_dir, "batch_correction_boxplots.png")
    png(png_file, width = width_in, height = height_in, units = "in", res = dpi, type = "cairo", pointsize = point_sz)
    safe_boxpanels(data_t, data_combat, sample_names)
    dev.off()
    
    cat("Batch effect correction completed. Corrected data saved to:", output_file, "\n")
    cat("Boxplots saved to:\n  -", pdf_file, "\n  -", png_file, "\n")
  }, error = function(e) {
    cat("Error: ", e$message, "\n")
  })
}

# Example command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- args[2]
user_id <- args[3]

batch_effect_correction(input_file, output_dir, user_id)
