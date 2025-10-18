setup_stringdb_environment <- function() {
  # Load required libraries (all pre-installed in Docker)
  library(STRINGdb)
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(stringr)

  # Set timeout to 6000 seconds (100 minutes)
  options(timeout = 6000)

  # Inform the user that the setup is complete
  message("Libraries loaded and environment set up successfully.")
}

# Initialize STRINGdb object
initialize_stringdb <- function(species_unique_id, score_threshold = 150) {
  STRINGdb$new(version = "12.0", species = species_unique_id, score_threshold = score_threshold)
}

# Define the function to extract taxon_ID by Species Name
get_taxon_id <- function(data, species_name) {
  # Filter the data to find the species
  result <- data[data$Species_Name == species_name, "taxon_ID"]

  # Check if a match was found
  if (length(result) == 0) {
    return("Species not found")
  } else {
    return(result)
  }
}
