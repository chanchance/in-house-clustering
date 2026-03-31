# Clustering Methods for Semiconductor Layout Feature Analysis

> In-House Clustering Tool Suite — Algorithm Reference
> Target: 126-dimensional layout feature vectors extracted from GDS/OAS patterns

---

## Overview

| # | Method | Category | k Required | Scalability | Key Strength |
|---|--------|----------|-----------|-------------|--------------|
| 01 | Decision Tree Clustering | Rule-based | Yes (depth) | ★★★★★ | Interpretable, fast inference |
| 02 | K-Means (MiniBatch) | Partition | Yes | ★★★★★ | Simple, scalable, baseline |
| 03 | Autoencoder + K-Means | Deep Learning | Yes | ★★★★☆ | Nonlinear feature compression |
| 04 | GMM | Probabilistic | Yes | ★★★☆☆ | Soft assignment, covariance |
| 05 | Bisecting K-Means | Hierarchical | Yes | ★★★★☆ | Balanced, top-down |
| 06 | Agglomerative Ward | Hierarchical | Yes | ★★☆☆☆ | Dendrogram, deterministic |
| 07 | HDBSCAN | Density-based | No | ★★★☆☆ | Arbitrary shape, noise |
| 08 | Spectral Clustering | Graph-based | Yes | ★★☆☆☆ | Non-convex manifold |
| 09 | Isolation Forest + K-Means | Hybrid | Yes | ★★★★☆ | Anomaly-robust |
| 10 | VAE + K-Means | Deep Learning | Yes | ★★★★☆ | Probabilistic latent space |

---

## Method 01 — Decision Tree Clustering

### Core Idea
A decision tree is grown on the feature matrix, and each terminal leaf node is treated as a distinct cluster. The depth of the tree controls cluster granularity.

### Algorithm
1. Fit a `DecisionTreeClassifier` or unsupervised variant on 126D feature vectors
2. Extract leaf node indices via `tree.apply(X)` → each unique leaf ID = one cluster
3. Depth hyperparameter controls the number of clusters (2^depth max leaves)
4. Optional: prune leaves with fewer than `min_samples_leaf` points

### Mathematical Basis
- Splitting criterion: Gini impurity or variance reduction
- Intra-cluster variance minimized at each split
- Leaf assignment: hard, deterministic

### Key Properties
- **Interpretability**: each split = human-readable rule (e.g., `CD_mean ≤ 0.72`)
- **Speed**: O(n · depth) inference — suitable for real-time use
- **Limitation**: axis-aligned boundaries; deep trees overfit noisy features

### Reference
Breiman, L. et al. *Classification and Regression Trees* (1984).

---

## Method 02 — K-Means (MiniBatch)

### Core Idea
Partition n data points into k spherical clusters by minimizing within-cluster sum of squares (inertia). MiniBatch variant processes random subsets per iteration for scalability.

### Algorithm
1. Initialize k centroids with **k-means++** (spread initialization)
2. Sample a mini-batch B ⊂ X each iteration
3. Assign each point in B to nearest centroid (Euclidean distance)
4. Update centroids using exponential moving average
5. Repeat until convergence or `max_iter` reached

### Mathematical Basis
$$\min_{\mu_1...\mu_k} \sum_{i=1}^{n} \min_{j} \|x_i - \mu_j\|^2$$

Decision boundary = **Voronoi diagram** of centroids.

### Key Properties
- Hard assignment; assumes convex, isotropic clusters
- Sensitive to initialization → k-means++ mitigates
- Scales to millions of points; GPU-friendly
- Elbow method / silhouette score for optimal k

### Reference
Sculley, D. *Web-Scale K-Means Clustering*. WWW 2010.

---

## Method 03 — Autoencoder + K-Means

### Core Idea
A deep autoencoder compresses 126D layout features into a 16D latent representation, removing redundancy and noise. K-Means is applied on the compact latent vectors.

### Architecture
```
Input (126D)
  → Linear(128) + BatchNorm + LeakyReLU
  → Linear(64)  + BatchNorm + LeakyReLU
  → Linear(32)  + BatchNorm + LeakyReLU
  → Latent z (16D)             ← K-Means applied here
  → Linear(32)  + LeakyReLU
  → Linear(64)  + LeakyReLU
  → Linear(128) + LeakyReLU
  → Output (126D)
```

### Training
- Loss: MSE reconstruction `L = ||x - x̂||²`
- Optimizer: Adam, lr=1e-3, 100 epochs
- After training: extract z for all samples → fit K-Means

### Key Properties
- 87% dimensionality reduction (126D → 16D)
- Captures nonlinear manifold structure
- GPU-accelerated (PyTorch)
- t-SNE on z reveals well-separated clusters

### Reference
Hinton, G. E. & Salakhutdinov, R. *Reducing the Dimensionality of Data with Neural Networks*. Science 2006.
Xie, J. et al. *Unsupervised Deep Embedding for Clustering Analysis* (DEC). ICML 2016.

---

## Method 04 — Gaussian Mixture Model (GMM)

### Core Idea
Model the data as a mixture of K multivariate Gaussian distributions. Unlike K-Means, GMM assigns **soft (probabilistic) memberships** and captures ellipsoidal cluster shapes via full covariance matrices.

### Algorithm (Expectation-Maximization)
1. **Initialize**: k means μ_k, covariances Σ_k, weights π_k
2. **E-step**: compute responsibility r_ik = P(cluster k | x_i)
3. **M-step**: update μ_k, Σ_k, π_k using weighted counts
4. Iterate until log-likelihood converges
5. **Scalability**: fit on 50K subsample, predict full dataset

### Mathematical Basis
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x\,|\,\mu_k, \Sigma_k)$$

Covariance types: `full` > `tied` > `diag` > `spherical`

### Key Properties
- Soft assignment enables uncertainty quantification
- Full covariance handles correlated, elongated clusters
- BIC/AIC for model selection (optimal k)
- 50K subsample for tractability on large datasets

### Reference
Reynolds, D. *Gaussian Mixture Models*. Encyclopedia of Biometrics, 2009.
Dempster, A. et al. *Maximum Likelihood from Incomplete Data via the EM Algorithm*. JRSS-B 1977.

---

## Method 05 — Bisecting K-Means

### Core Idea
A top-down divisive approach: start with one cluster containing all data, then repeatedly bisect the cluster with the **highest SSE** (Sum of Squared Errors) using 2-means.

### Algorithm
1. Initialize: all n points in cluster C₀
2. Select cluster C* with maximum SSE
3. Apply K-Means(k=2) to split C* into C_a, C_b
4. Replace C* with {C_a, C_b} in the cluster list
5. Repeat steps 2–4 until k clusters are reached

### Mathematical Basis
At each step:
$$C^* = \arg\max_{C_j} \text{SSE}(C_j), \quad \text{SSE}(C) = \sum_{x \in C}\|x - \mu_C\|^2$$

### Key Properties
- More balanced than vanilla K-Means
- Produces a binary splitting tree (implicit hierarchy)
- O(k · n · d) total complexity
- Better global optimum than flat K-Means on many datasets

### Reference
Steinbach, M. et al. *A Comparison of Document Clustering Techniques*. KDD Workshop 2000.

---

## Method 06 — Agglomerative Hierarchical Clustering (Ward)

### Core Idea
Bottom-up merging: each point starts as its own cluster; pairs are merged iteratively by **Ward linkage** (minimize variance increase from merging). Produces a full **dendrogram** — cut at any level for flat clusters.

### Algorithm
1. Initialize: n singleton clusters
2. Compute pairwise Ward distance matrix
3. Merge the pair (C_i, C_j) with minimum ΔVariance:
   ΔV(i,j) = (n_i·n_j)/(n_i+n_j) · ||μ_i − μ_j||²
4. Update distance matrix (Lance-Williams formula)
5. Repeat until one cluster remains
6. Cut dendrogram at threshold → k flat clusters

### Key Properties
- Deterministic (no random initialization)
- Full dendrogram enables visual inspection of hierarchy
- Ward linkage tends to produce compact, equal-sized clusters
- O(n² log n) memory; use on subsample for large n

### Reference
Ward, J. H. *Hierarchical Grouping to Optimize an Objective Function*. JASA 1963.

---

## Method 07 — HDBSCAN

### Core Idea
Hierarchical extension of DBSCAN. Builds a **condensed cluster tree** from density levels, then extracts the most stable (persistent) clusters. Points in low-density regions are labeled as **noise (−1)**.

### Algorithm
1. Compute core distances: d_core(x) = distance to k-th nearest neighbor
2. Define mutual reachability distance: d_mreach(a,b) = max(d_core(a), d_core(b), d(a,b))
3. Build minimum spanning tree on mutual reachability graph
4. Condense hierarchy by removing points falling below `min_cluster_size`
5. Extract clusters by maximizing **cluster stability** (sum of λ_death − λ_birth)

### Mathematical Basis
$$\lambda = 1/\text{distance}; \quad \text{stability}(C) = \sum_{p \in C} (\lambda_{\text{death}} - \lambda_p)$$

### Key Properties
- No need to specify k
- Handles clusters of varying density and arbitrary shape
- Robust to outliers/noise
- Soft cluster membership probabilities available

### Reference
Campello, R. J. et al. *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD 2013.
McInnes, L. et al. *hdbscan: Hierarchical density based clustering*. JOSS 2017.

---

## Method 08 — Spectral Clustering

### Core Idea
Map data to a low-dimensional eigenspace of the **graph Laplacian** (capturing manifold structure), then apply K-Means in that space. Excels at finding non-convex, ring-shaped, or manifold clusters.

### Algorithm
1. Build affinity matrix **W**: W_ij = exp(−||x_i − x_j||² / 2σ²)  (RBF kernel)
2. Compute degree matrix **D**: D_ii = Σ_j W_ij
3. Normalized Laplacian: **L** = D^(−½)(D − W)D^(−½)
4. Extract top-k eigenvectors of **L** → embedding matrix **U** ∈ ℝ^(n×k)
5. Row-normalize **U**, apply K-Means

### Key Properties
- Finds clusters unreachable by distance-based methods
- Works on graph/similarity data (no explicit feature space needed)
- σ (bandwidth) and k are key hyperparameters
- O(n³) eigendecomposition → use on subsample (≤10K)

### Reference
Ng, A. et al. *On Spectral Clustering: Analysis and an Algorithm*. NIPS 2001.
Von Luxburg, U. *A Tutorial on Spectral Clustering*. Statistics and Computing 2007.

---

## Method 09 — Isolation Forest + K-Means

### Core Idea
Two-stage pipeline: first detect and remove anomalies using **Isolation Forest**, then apply K-Means on the cleaned inlier set. Outliers are preserved as cluster label **−1**.

### Stage 1: Isolation Forest
- Build an ensemble of random isolation trees
- Anomaly score = normalized average path length to isolate a point
- Short paths → easy to isolate → outlier
- `contamination` parameter controls outlier fraction

### Stage 2: K-Means on Inliers
- Filter: retain only points with score > threshold
- Apply K-Means (MiniBatch) on the cleaned dataset
- Re-assign filtered outliers: label = −1

### Mathematical Basis
$$s(x, n) = 2^{-E[h(x)]/c(n)}, \quad c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where h(x) = path length, c(n) = expected path length for n samples.

### Key Properties
- Robust clustering: outliers don't distort centroids
- Interpretable anomaly score per data point
- Anomaly detection + clustering in one pipeline
- Particularly effective when layout data has defective/rare patterns

### Reference
Liu, F. T. et al. *Isolation Forest*. ICDM 2008.

---

## Method 10 — Variational Autoencoder + K-Means (VAE)

### Core Idea
Extends the deterministic autoencoder to a **generative probabilistic model**. The encoder outputs a distribution q(z|x) = N(μ(x), σ²(x)), and z is sampled via the reparameterization trick. K-Means is applied on the sampled latent vectors.

### Architecture
```
Input (126D)
  → Encoder → μ(x), log σ²(x)       ← reparameterization: z = μ + ε·σ
  → z (16D, sampled)                 ← K-Means applied here
  → Decoder → x̂ (126D)
```

### Training Objective (ELBO)
$$\mathcal{L} = \underbrace{\mathbb{E}_{q}[\log p(x|z)]}_{\text{Reconstruction (MSE)}} - \underbrace{\beta \cdot D_{KL}(q(z|x) \| p(z))}_{\text{Regularization (KL)}}$$

- β > 1 (β-VAE): stronger disentanglement
- KL term forces z ~ N(0,I): smooth, structured latent space

### Key Properties vs Autoencoder
| | AE | VAE |
|--|--|--|
| Latent z | Deterministic | Stochastic (sampled) |
| Latent space | Arbitrary | Regularized N(0,I) |
| Clustering | Compact but irregular | Smooth, more separable |
| Generative | No | Yes (can synthesize x̂) |

### Reference
Kingma, D. P. & Welling, M. *Auto-Encoding Variational Bayes*. ICLR 2014.
Jiang, Z. et al. *Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering* (VaDE). IJCAI 2017.
