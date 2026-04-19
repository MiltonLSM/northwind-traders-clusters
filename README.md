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
Open `notebooks/01_eda.ipynb` to:
- Load and inspect the client data
- Check for missing values and outliers
- Visualize distributions and correlations
- Understand data characteristics

### 2. Clustering Analysis
Open `notebooks/02_clustering.ipynb` to:
- Scale the features
- Apply K-Means clustering
- Apply DBSCAN clustering
- Apply Hierarchical clustering
- Compare results using multiple metrics
- Select the best clustering approach

### 3. Prediction
Open `notebooks/03_prediction.ipynb` to:
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

## Usage Example

```python
from src.utils import load_model, predict_new_client
import pandas as pd

# Load saved model and scaler
scaler = load_model('models/scaler.pkl')
model = load_model('models/kmeans_model.pkl')

# New client data
new_client = {
    'age': 35,
    'income': 75000,
    'purchase_frequency': 12,
    'avg_transaction': 150
}

# Predict cluster
cluster = predict_new_client(new_client, scaler, model, 
                            feature_names=['age', 'income', 'purchase_frequency', 'avg_transaction'])

print(f"New client belongs to cluster: {cluster}")
```

## Author
Milton Silva

## Date
2025