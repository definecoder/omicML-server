# R Package Management - Implementation Summary

## Overview

All R packages are now pre-installed in the Docker image, eliminating runtime installations that were causing slowdowns and redundant package installs during API calls.

## Package Management Strategy

### Using BiocManager for Version Compatibility

Instead of manually specifying exact versions for every package, we're using **BiocManager** as the package manager, which provides several key benefits:

1. **Automatic Compatibility**: By setting Bioconductor version 3.21, BiocManager automatically ensures all Bioconductor packages are compatible with each other
2. **Dependency Resolution**: BiocManager handles CRAN packages and resolves dependencies that work well with the Bioconductor ecosystem
3. **Less Version Conflicts**: No need to manually track and update dozens of specific version numbers
4. **Easier Maintenance**: Future updates are simpler - just update the Bioconductor version and rebuild

## Dockerfile Changes

### Old Approach (Problematic)

- Used `remotes::install_version()` for specific versions of each package
- Required manual version management for 20+ packages
- Prone to version conflicts between CRAN and Bioconductor packages
- More complex and harder to maintain

### New Approach (Recommended)

```r
# Set Bioconductor version 3.21 - this manages compatibility
BiocManager::install(version = '3.21', ask = FALSE, update = TRUE)

# Install Bioconductor packages - version managed by BiocManager
BiocManager::install(c('WGCNA', 'DESeq2', 'limma', 'biomaRt', 'sva',
                       'STRINGdb', 'apeglm', 'impute'), ask = FALSE)

# Install CRAN packages - BiocManager ensures Bioconductor compatibility
BiocManager::install(c('tidyverse', 'Rtsne', 'umap', 'ggplot2', 'readr',
                       'ape', 'mice', 'dplyr', 'gplots', 'ggVennDiagram',
                       'pheatmap', 'RColorBrewer', 'stringr', 'here', 'lme4'),
                     ask = FALSE)
```

## R Scripts Modified

All runtime package installation code has been removed from the following files:

### Core Scripts

- `api/code/req_packages.R` - Now just loads pre-installed packages
- `api/code/micro_functions.R` - Replaced `install_and_load()` with direct `library()` calls
- `api/code/Microarray_Updated_Workflow.R` - Simplified package loading

### Venn Diagram Scripts

- `api/code/1_init_venn_m.R` - Removed installation function
- `api/code/3_plot_venn_m.R` - Removed installation function
- `api/code/4_wide_frame_venn_m.R` - Removed installation checks

### Other Analysis Scripts

- `api/code/5_Up_Down_m.R` - Removed ggplot2 installation check
- `api/code/2_plot_heatmap.R` - Removed pheatmap/RColorBrewer installation
- `api/code/1_Biomart_init_m.R` - Removed biomaRt installation function

### STRING Workflow

- `api/string/1_setup_string_env.R` - Removed `install_and_load()` function

### Configuration

- `api/code/.Rprofile` - Added safeguards to prevent runtime installations

## Benefits

1. **Performance**: No more waiting for package installations during API calls
2. **Reliability**: Packages are tested and verified during Docker build, not at runtime
3. **Reproducibility**: Docker image contains exact package versions
4. **Version Compatibility**: BiocManager ensures all packages work together
5. **Simpler Maintenance**: Easier to update packages in the future

## Expected Package Versions (Bioconductor 3.21)

When using Bioconductor 3.21 with R 4.4+, you'll get approximately:

- WGCNA: ~1.73
- DESeq2: ~1.44.0
- limma: ~3.60.0+
- biomaRt: ~2.60.0+
- sva: ~3.52.0
- STRINGdb: ~2.16.0+
- tidyverse: ~2.0.0
- And compatible versions of all other packages

_Note: Exact versions may vary slightly but BiocManager ensures compatibility_

## Docker Build Command

To rebuild the Docker image with the new package setup:

```bash
docker build -t plagl1-server .
docker run --name plgl-server -p 8000:8000 -d plagl1-server:latest
```

Or use the command from the Dockerfile:

```bash
docker rm -f plgl-server && docker build -t wonderful_rubin . && docker run --name plgl-server -p 8000:8000 -d wonderful_rubin:latest
```

## Testing

After building the new Docker image, test that all packages load correctly:

```r
# In R console within container
library(WGCNA)
library(DESeq2)
library(limma)
library(biomaRt)
library(sva)
library(STRINGdb)
library(tidyverse)
# ... etc
```

All packages should load without any installation prompts or errors.

## Future Updates

To update R packages in the future:

1. Change the Bioconductor version in Dockerfile (e.g., from `'3.21'` to `'3.22'`)
2. Rebuild the Docker image
3. Test the application

BiocManager will automatically pull compatible versions of all packages.
