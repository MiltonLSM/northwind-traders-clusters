# Client Segmentation Clustering Project

## Overview
This project uses unsupervised learning to segment clients based on their yearly report data. I compare three clustering techniques (K-Means, DBSCAN, and Hierarchical Clustering) to identify the best client segments.

## Project Structure
```
clustering-project/
├── data/
│   ├── raw/          # Original client data
│   └── processed/    # Scaled data
├── notebooks/        # Jupyter notebooks for analysis
├── src/              # Reusable Python functions
├── models/           # Saved models and scaler
└── reports/figures/  # Visualizations
```

## Setup

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install as package (optional but recommended):**
   ```bash
   pip install -e .
   ```

## Workflow

### 1. Exploratory Data Analysis
- Load and inspect the client data
- Check for missing values and outliers
- Visualize distributions and correlations
- Understand data characteristics

### 2. Clustering Analysis
- Scale the features
- Apply K-Means clustering
- Apply DBSCAN clustering
- Apply Hierarchical clustering
- Compare results using multiple metrics
- Select the best clustering approach

### 3. Prediction
- Load saved models
- Predict cluster for new clients
- Analyze cluster characteristics

## Clustering Techniques

### K-Means
- Good for spherical clusters
- Requires specifying number of clusters (k)
- Uses elbow method and silhouette scores to find optimal k

### DBSCAN
- Density-based clustering
- Can find arbitrary-shaped clusters
- Automatically determines number of clusters
- Can identify outliers as noise

### Hierarchical Clustering
- Creates dendrogram of cluster relationships
- Can choose number of clusters by cutting dendrogram
- Good for understanding cluster hierarchy

## Evaluation Metrics

- **Silhouette Score** (higher is better, range: -1 to 1)

## Results

Results and visualizations are saved in `reports/figures/`:
- Elbow plot for K-Means
- Silhouette scores comparison
- Cluster visualizations
- Cluster profiles

## Author
Milton Silva

## Date
2025