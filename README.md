# Parallel Processing for Smarter Netflix Predictions

This research project explores how parallel computing and distributed training can optimize large-scale machine learning pipelines. We built an end-to-end system that processes over 1.5 million Netflix user reviews to predict 5-star ratings using machine learning and deep learning models. The solution incorporates Dask, PyTorch (DDP & FSDP), multiprocessing, and CUDA profiling to achieve high performance.

---

##  Project Goals

- Efficient data loading and preprocessing using Dask and multiprocessing
- Parallelized feature engineering and transformation
- Scalable model training with PyTorch DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP)
- Profiling and evaluation of multi-core CPU and multi-GPU setups

---

##  Tools & Technologies

- **Languages**: Python
- **Parallel Data Tools**: Dask, multiprocessing, Pandas
- **ML/Modeling**: Scikit-learn, XGBoost, PyTorch, DDP, FSDP
- **Profiling**: torch.profiler, cProfile, timeit
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Environment**: Discovery HPC Cluster, NVIDIA Tesla P100 GPUs, Intel Xeon CPUs

---

##  Dataset

- **Source**: [Kaggle - Netflix Google Store Reviews](https://www.kaggle.com/datasets/bwandowando/1-5-million-netflix-google-store-reviews/data)
- **Size**: 1.5M reviews × 11 columns
- **Target Variable**: Binary classification — whether review_rating = 5
- **Key Features**:
  - Review text
  - Review time
  - App version
  - Review likes

---

##  Methodology

### 1. Data Handling
- Loaded large datasets with Dask (12× faster than Pandas)
- Preprocessing with Pandas post-filtering
- Profiling with `cProfile`, `timeit` for validation

### 2. EDA & Visualization
- Used Dask groupby operations and lazy aggregations
- Visualized with Matplotlib, Plotly after converting to Pandas

### 3. Feature Engineering
- Parallel one-hot encoding and missing value imputation
- Reduced features to top 30 for scalability
- Parallelized with `joblib` and Dask

### 4. ML Models (CPU)
- Trained Logistic Regression, Random Forest, SVM using Scikit-learn
- Hyperparameter tuning with GridSearchCV (n_jobs=-1)
- Best XGBoost model achieved RMSE of **1.53**

### 5. Deep Learning Models (GPU)
- Built custom MLP "NetflixNet" for binary classification
- Trained with DDP and FSDP across 1–4 GPUs
- CUDA profiling for kernel efficiency and memory bottlenecks

---

##  Environment

- **Cluster**: Discovery HPC, Northeastern University
- **GPUs**: NVIDIA Tesla P100 (up to 4 GPUs used)
- **CPUs**: Intel Xeon E5-2680 v4 (up to 12 cores tested)

---

##  Results Summary

| Model      | Platform      | Training Time | Accuracy | Speedup |
|------------|---------------|---------------|----------|---------|
| Random Forest | 10-CPU DDP  | ~440s         | 67%      | 1.2×    |
| NetflixNet (1 GPU) | DDP     | ~298s         | 61.1%    | —       |
| NetflixNet (3 GPUs) | DDP    | ~112s         | 61.2%    | 2.7×    |
| NetflixNet (FSDP) | 4 GPUs   | ~502s         | 61.1%    | ↓       |

>  FSDP showed degraded performance due to fallback to `NO_SHARD` mode. DDP scaled better with more GPUs.

---

##  Key Takeaways

- **Dask** enabled a **12× speedup** in loading large datasets
- **DDP outperformed FSDP** in our GPU environment due to better utilization
- **Parallelism improved performance**, but scalability plateaued due to Amdahl’s Law and memory bottlenecks
- **Feature engineering** significantly improved correlation with the target variable

---

## Future Work

- Improve FSDP sharding strategy and memory tuning
- Integrate SMOTE for handling class imbalance
- Deploy model with Flask/Streamlit for real-time inference
- Add experiment tracking with MLflow or Weights & Biases


