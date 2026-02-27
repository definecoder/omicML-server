# omicML: a bioinformatics and machine learning based graphical user interface for transcriptomic biomarker identification

An intuitive graphical user interface that combines transcriptomic data analysis with ML-based classification via integrating R and Python packages/libraries. It supports both RNA-Seq and microarray data following automatic preprocessing and differential expression analysis. It annotates differentially expressed genes with descriptions, gene ontology, and pathway information and incorporates comparative analysis. The extensive ML pipeline sequentially enables both unsupervised and supervised learning, benchmarks multiple ML classifiers, assesses feature importance, develops single-gene and multi-gene predictive models, and systematically finalizes the biomarker algorithm.


- **Production instance:** https://backend.omicml.org
- **Frontend:** https://omicml.org
- **Repository:** https://github.com/definecoder/omicML-server

---
## Preprint

**bioRxiv**

**omicML: An Integrative Bioinformatics and Machine Learning Framework for Transcriptomic Biomarker Identification**

Joy Prokash Debnath, Kabir Hossen, Md. Sayeam Khandaker, Shawon Majid, Md Mehrajul Islam, Siam Arefin, Preonath Chondrow Dev, Saifuddin Sarker, Tanvir Hossain

bioRxiv, Cold Spring Harbor Laboratory (2025)

DOI: [10.1101/2025.10.25.684517](https://doi.org/10.1101/2025.10.25.684517)

[View PDF](https://www.biorxiv.org/content/10.1101/2025.10.25.684517v1.full.pdf)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Option 1 — Docker (Recommended)](#option-1--docker-recommended)
  - [Option 2 — Local Development](#option-2--local-development)
- [Configuration](#configuration)
- [API Overview](#api-overview)
- [Test Dataset & Quickstart](#test-dataset--quickstart)
- [Interactive API Docs](#interactive-api-docs)
- [Citation](#citation)

---

## Features

| Module | Capabilities |
|---|---|
| **RNA-seq Analysis** | Upload count + metadata, log2 normalization, quantile normalization, DESeq2 DGE, volcano plots |
| **Microarray Analysis** | Limma-based DGE, before/after normalization QC |
| **Quality Control** | PCA, t-SNE, UMAP, hierarchical clustering, K-means, boxplots |
| **Batch Effect Correction** | SVA (Surrogate Variable Analysis) |
| **Machine Learning** | Feature selection, model benchmarking (XGBoost, Random Forest) |
| **Gene Annotation** | BiomaRt-based ID mapping across multiple organisms and ID types |
| **Venn Diagrams** | Multi-set gene list comparison |
| **Heatmaps** | Publication-ready annotated heatmaps |
| **STRING Networks** | Protein–protein interaction queries and enrichment analysis |

---

## Architecture

```
omicML-server/
├── api/
│   ├── main.py                  # FastAPI application entry point
│   ├── requirements.txt         # Python dependencies
│   ├── database.py              # SQLite database (SQLAlchemy)
│   ├── routers/                 # API route handlers
│   │   ├── auth_router.py       # User registration & JWT login
│   │   ├── operation_router.py  # RNA-seq analysis pipeline
│   │   ├── micro_router.py      # Microarray pipeline
│   │   ├── annotation_router.py # Gene annotation (BiomaRt)
│   │   ├── venn_router.py       # Venn diagram generation
│   │   ├── heatmap_router.py    # Heatmap generation
│   │   └── file_router.py       # File serving
│   ├── code/                    # R analysis scripts
│   ├── string/                  # STRING database scripts
│   ├── models/                  # Pydantic schemas & ORM models
│   └── core/                    # Security (JWT), constants
├── Dockerfile
└── README.md
```

The server bridges Python (FastAPI) and R (rpy2) to execute statistical analyses. Each user's data is stored in isolated per-user directories under `api/code/{user_id}/`.

---

## System Requirements

### Minimum

- CPU: 2 cores
- RAM: 8 GB (16 GB recommended for large datasets)
- Disk: 10 GB free (Docker image is ~6 GB due to R packages)

### Software

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11.x | Required for local development |
| R | 4.4+ | Required for local development |
| Docker | 20.10+ | Required for Docker deployment |

> **Note:** The Docker image bundles Python 3.11, R 4.5, and all required packages. Local development requires you to install Python and R separately.

---

## Installation

### Option 1 — Docker (Recommended)

Docker is the recommended installation method. It bundles all Python and R dependencies, eliminating manual environment setup.

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) installed and running.

```bash
# 1. Clone the repository
git clone https://github.com/definecoder/omicML-server.git
cd omicML-server

# 2. Build the Docker image
#    (installs all Python and R packages — takes ~15–30 minutes on first build)
docker build -t omicml-server .

# 3. Run the container
docker run --name omicml-server -p 8000:8000 -d omicml-server:latest

# 4. Verify the server is running
curl http://localhost:8000/
# Expected: {"message": "PLAGL1 Server is running..."}
```

The server will be available at `http://localhost:8000`.

To stop the container:

```bash
docker stop omicml-server
```

To rebuild after code changes:

```bash
docker rm -f omicml-server && docker build -t omicml-server . && \
docker run --name omicml-server -p 8000:8000 -d omicml-server:latest
```

---

### Option 2 — Local Development

#### Step 1 — Install Python 3.11

Download from [python.org](https://www.python.org/downloads/) or via your system package manager.

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

#### Step 2 — Install R 4.4+

```bash
# Ubuntu/Debian (CRAN repository)
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
  sudo gpg --dearmor -o /usr/share/keyrings/cran.gpg
echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" | \
  sudo tee /etc/apt/sources.list.d/cran.list
sudo apt-get update && sudo apt-get install -y r-base

# macOS (via Homebrew)
brew install r

# Windows: download installer from https://cran.r-project.org/bin/windows/base/
```

#### Step 3 — Install system libraries (Ubuntu/Debian only)

```bash
sudo apt-get install -y \
  libssl-dev libcurl4-openssl-dev libxml2-dev libgit2-dev libglpk-dev \
  libgmp-dev libgsl-dev libfreetype6-dev libpng-dev libtiff5-dev \
  libjpeg-dev libcairo2-dev libharfbuzz-dev libfribidi-dev cmake
```

#### Step 4 — Install R packages

Open an R console and run:

```r
# Install BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# Set Bioconductor version (ensures compatibility across all packages)
BiocManager::install(version = "3.21", ask = FALSE, update = TRUE)

# Install Bioconductor packages
BiocManager::install(
  c("WGCNA", "DESeq2", "limma", "biomaRt", "sva",
    "STRINGdb", "apeglm", "impute"),
  ask = FALSE, update = FALSE
)

# Install CRAN packages
BiocManager::install(
  c("tidyverse", "Rtsne", "umap", "ggplot2", "readr", "ape",
    "mice", "dplyr", "gplots", "ggVennDiagram", "pheatmap",
    "RColorBrewer", "stringr", "here", "lme4"),
  ask = FALSE
)
```

Expected versions (Bioconductor 3.21): DESeq2 ~1.44, limma ~3.60, biomaRt ~2.60, sva ~3.52, WGCNA ~1.73, STRINGdb ~2.16.

#### Step 5 — Set up the Python virtual environment

```bash
# From the repository root
python3.11 -m venv venv

# Activate
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Install Python dependencies
cd api
pip install --upgrade pip
pip install -r requirements.txt
```

**Python dependencies summary:**

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.112.0 | Web framework |
| uvicorn | 0.30.5 | ASGI server |
| rpy2 | 3.5.16 | Python–R bridge |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.6.1 | Machine learning utilities |
| xgboost | 2.1.4 | Gradient boosting |
| umap-learn | 0.5.7 | UMAP dimensionality reduction |
| matplotlib | 3.7.1 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| SQLAlchemy | 2.0.32 | ORM / SQLite |
| python-jose | 3.3.0 | JWT authentication |
| passlib | 1.7.4 | Password hashing |
| pydantic | 2.8.2 | Data validation |

Full pinned list: [`api/requirements.txt`](api/requirements.txt)

#### Step 6 — Run the development server

```bash
# From the api/ directory (with venv activated)
fastapi dev main.py
```

The server starts at `http://localhost:8000`.

---

## Configuration

The following settings can be modified in `api/core/consts.py`:

| Variable | Default | Description |
|---|---|---|
| `BASE_URL` | `http://localhost:8000` | Base URL used in file-link responses. Set to your production domain. |

The JWT secret key is defined in `api/core/security.py`. For production deployments, replace the default key with a strong random secret and manage it via an environment variable or secrets manager.

---

## API Overview

All protected endpoints require a Bearer JWT token in the `Authorization` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | — | Register a new user |
| `POST` | `/auth/login` | — | Login and obtain a JWT token |
| `POST` | `/operations/init` | ✓ | Upload count data and metadata |
| `GET` | `/operations/analyze` | ✓ | Run normalization and QC visualizations |
| `GET` | `/operations/dge-analysis` | ✓ | Differential gene expression + volcano plots |
| `GET` | `/operations/visualize-dimensions` | ✓ | PCA / t-SNE / UMAP |
| `GET` | `/operations/find-best-model` | ✓ | Benchmark ML models |
| `POST` | `/operations/annotation/annotate_genes` | ✓ | Annotate genes via BiomaRt |
| `POST` | `/operations/venn/venn_diagram` | ✓ | Generate Venn diagram |
| `GET` | `/operations/heatmap/heatmap_diagram` | ✓ | Generate heatmap |
| `GET` | `/figures/{user_id}/{file}` | — | Retrieve a generated figure (PNG/PDF) |
| `GET` | `/files/{user_id}/{file}` | — | Retrieve a generated data file (CSV) |

Full interactive documentation is available at `http://localhost:8000/docs`.

---

## Test Dataset & Quickstart

### Download Test Data

Two CSV files from the Mpox / PLAGL1 study are provided as test data:

| File | Description | Download |
|---|---|---|
| `count_data.csv` | RNA-seq raw gene count matrix (genes × samples) | [Download from Google Drive](https://drive.google.com/file/d/16Emw5cmXdokykDTVZoP_enItHpdZ34fv/view?usp=share_link) |
| `metadata.csv` | Sample metadata with condition labels | [Download from Google Drive](https://drive.google.com/file/d/1Jj1KjYTEi3YpLsM7VXhL7Rllq7JlxSWl/view?usp=share_link) |

### Input Format

**`count_data.csv`** — Gene count matrix:
- Row names (first column): gene identifiers (e.g., Ensembl IDs or gene symbols)
- Column names (first row): sample IDs (must match the row names of `metadata.csv`)
- Values: non-negative integers (raw counts)

```
,Sample1,Sample2,Sample3,Sample4
ENSG00000000003,1200,985,1102,1340
ENSG00000000005,450,520,490,610
...
```

**`metadata.csv`** — Sample annotation:
- Row names (first column): sample IDs (must match column names of `count_data.csv`)
- Columns: sample attributes; the `condition` column defines groups for DGE analysis

```
,condition
Sample1,control
Sample2,control
Sample3,infected
Sample4,infected
...
```

---

### Step-by-Step Quickstart (cURL)

Replace `http://localhost:8000` with `https://backend.omicml.org` to test against the production server.

#### 1. Register a user account

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword", "name": "Your Name"}'
```

**Response:**
```json
{
  "message": "User created successfully",
  "token": "<JWT_TOKEN>"
}
```

#### 2. Login (if account already exists)

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword"}'
```

**Response:**
```json
{
  "message": "User logged in successfully",
  "token": "<JWT_TOKEN>"
}
```

Save the token for subsequent requests:
```bash
TOKEN="<JWT_TOKEN>"
```

#### 3. Upload count data and metadata

```bash
curl -X POST http://localhost:8000/operations/init \
  -H "Authorization: Bearer $TOKEN" \
  -F "count_data=@count_data.csv" \
  -F "meta_data=@metadata.csv"
```

**Response:**
```json
{
  "message": "file uploaded & Processed successfully!",
  "count_data": "/path/to/count_data.csv",
  "meta_data": "/path/to/meta_data.csv"
}
```

#### 4. Run normalization and quality control analysis

```bash
curl -X GET http://localhost:8000/operations/analyze \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "message": "Analysis completed successfully!",
  "results": {
    "boxplot_denorm_img": "http://localhost:8000/figures/{user_id}/Boxplot_denorm.png",
    "boxplot_norm_img":   "http://localhost:8000/figures/{user_id}/Boxplot_norm.png",
    "pca_denorm_img":     "http://localhost:8000/figures/{user_id}/PCA_denorm.png",
    "pca_norm_img":       "http://localhost:8000/figures/{user_id}/PCA_norm.png",
    "tsne_norm_img":      "http://localhost:8000/figures/{user_id}/tSNE_norm.png",
    "umap_norm_img":      "http://localhost:8000/figures/{user_id}/UMAP_norm.png",
    "htree_norm_img":     "http://localhost:8000/figures/{user_id}/htree_norm.png"
  }
}
```

Each URL in the response points to a downloadable PNG or PDF figure.

#### 5. Run differential gene expression analysis

```bash
curl -X GET "http://localhost:8000/operations/dge-analysis" \
  -H "Authorization: Bearer $TOKEN"
```

Runs DESeq2 and returns URLs to volcano plot figures showing significantly up- and down-regulated genes per condition contrast.

#### 6. Retrieve a generated figure

```bash
# Replace {user_id} with your numeric user ID from the analysis response
curl -O http://localhost:8000/figures/{user_id}/PCA_norm.png
```

---

### Expected Output Summary

| Endpoint | Output files | Description |
|---|---|---|
| `GET /operations/analyze` | `Boxplot_denorm.png/pdf` | Sample expression distributions before normalization |
| `GET /operations/analyze` | `Boxplot_norm.png/pdf` | Sample expression distributions after normalization |
| `GET /operations/analyze` | `PCA_denorm.png/pdf` | PCA before normalization |
| `GET /operations/analyze` | `PCA_norm.png/pdf` | PCA after normalization |
| `GET /operations/analyze` | `tSNE_norm.png/pdf` | t-SNE dimensionality reduction |
| `GET /operations/analyze` | `UMAP_norm.png/pdf` | UMAP dimensionality reduction |
| `GET /operations/analyze` | `htree_norm.png/pdf` | Hierarchical clustering dendrogram |
| `GET /operations/dge-analysis` | `volcano_*.png/pdf` | Volcano plots per condition comparison |

All figures are generated in both PNG (for web display) and PDF (for publication) formats.

---

## Interactive API Docs

Once the server is running, the full interactive Swagger UI is available at:

```
http://localhost:8000/docs
```

This lets you explore and test all endpoints directly from your browser with request/response schemas, without needing cURL or Postman.

A [Postman collection](postman_collection.json) is also included in the repository. Import it via **Postman > Import** to run all endpoints with pre-configured requests.

---

## Citation

If you use omicML in your research, please cite:

> Debnath, J.P., Hossen, K., Khandaker, M.S., Majid, S., Islam, M.M., Arefin, S., Chondrow Dev, P., Sarker, S. and Hossain, T., 2025. omicML: An Integrative Bioinformatics and Machine Learning Framework for Transcriptomic Biomarker Identification. *bioRxiv*, pp.2025-10.

---

## License

License terms to be determined. Please contact the authors before use in derivative works.
