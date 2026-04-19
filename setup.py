from setuptools import setup, find_packages
 
setup(
    name="clustering-northwind-traders",
    version="0.1.0",
    description="Client Segmentation Clustering Project",
    author="Milton Silva",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
    ],
)