# Critical Reproduction and Optimization of Multi-level Contrastive Learning for Sequential Recommendation (MCLSR)

I just don't have any social media, it's instead of a blog post for Nebius 😭

**Authors:** [Aksinya Bykova](https://github.com/Aksinya-Bykova), [Pavel Krasnov](https://github.com/hookjabber) 

**Affiliation:** Yandex School of Data Analysis (YSDA)  

## Problem
We found SOTA model, but it didn't have open source code implementation

[Multi-level Contrastive Learning Framework for Sequential
Recommendation](https://arxiv.org/pdf/2208.13007)

The result is actually great

---

### Abstract
We present a reproduction study of the MCLSR framework (CIKM '22) for sequential recommendation. While MCLSR addresses data sparsity through multi-level contrastive learning, our study identifies significant inconsistencies in the original reported metrics and implementation. Focusing on the Amazon-Clothing dataset, we provide a corrected evaluation pipeline and an optimized independent implementation. Our results demonstrate that MCLSR significantly and consistently outperforms the SASRec baseline across all evaluation metrics, thereby clarifying the true potential and effectiveness of the multi-level contrastive learning approach.

### 1. Introduction
Sequential recommendation (SR) focuses on modeling historical interactions, but often struggles with the **data sparsity problem**. The Multi-level Contrastive Learning (MCLSR) framework [1] addresses this by incorporating collaborative and co-action signals through cross-view contrastive learning at both interest and feature levels.

In this work, we reproduce MCLSR on the *Amazon-Clothing* dataset to investigate reported **numerical inconsistencies** and validate the model's actual potential. By developing the architecture **from scratch** and ensuring a mathematically sound evaluation pipeline, we demonstrate that our **refined implementation** of MCLSR significantly outperforms purely sequential baselines like SASRec [2], establishing a more reliable performance bound for the framework.

### 2. Methodology
MCLSR integrates collaborative and co-action signals from four views (sequential, user-item, user-user, and item-item) using two contrastive learning levels: (1) **Interest-level CL** aligns current intent with general preference, and (2) **Feature-level CL** captures co-occurrence signals by contrasting embeddings across graph views. The architecture employs a LightGCN-based graph encoder [3] and a self-attention sequential encoder.

### 3. Reproduction Setup
**Implementation:** The framework was re-implemented in PyTorch based on the methodological descriptions in [1]. While we strictly adhered to the reported hyperparameters ($\alpha=0.5, \beta=1, \gamma=0.05$, embedding size 64), the **lack of implementation details** necessitated independent derivation of several critical components. Specifically, we had to resolve ambiguities regarding the **stochastic sampling strategy** and the **structural integration of the co-action views**. 

**Evaluation Pipeline:** We utilized a **rigorous full-ranking protocol** to ensure mathematical consistency and transparency. All metrics (Recall@N, NDCG@N, Hit@N) are reported within the standard $[0, 1]$ interval. This approach allows for a direct and reliable comparison between our independent implementation and the SASRec [2] baseline.

### 4. Results and Critical Analysis
We evaluate MCLSR against the original study [1] and a SASRec [2] baseline. Our implementation was developed independently, based on the methodological descriptions provided in the original paper

#### Table 1: Original results for Amazon-Clothing reported in [1].
| Metric | Rec@20 | NDCG@20 | Hit@20 | Rec@50 | NDCG@50 | Hit@50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Amazon-Clothing | 3.138 | 1.081 | 5.138 | 5.352 | 1.464 | 8.503 |

There are some errors in original papers (metrics can't be > 1.0), so we didn't compare metrics our MCLSR implementation with theirs  

**Reproduction Performance:** Table 2 shows our results using a standardized $[0, 1]$ pipeline. Our implementation demonstrates a significant performance boost over the sequential baseline.

#### Table 2: Reproduction (independent implementation) vs. SASRec baseline on Amazon-Clothing
| Metric | SASRec (Baseline) | MCLSR (Reproduction) | Improvement |
| :--- | :--- | :--- | :--- |
| Recall@20 | 0.0184 | **0.0420** | +128.3% |
| NDCG@20 | 0.0076 | **0.0189** | +148.7% |
| HitRate@20 | 0.0272 | **0.0660** | +142.6% |
| Recall@50 | 0.0314 | **0.0704** | +124.2% |
| NDCG@50 | 0.0106 | **0.0254** | +139.6% |
| HitRate@50 | 0.0490 | **0.1089** | +122.2% |


![Hit Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-21-15.png)

![NDCG Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-27-15.png)

![Recall Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-28-52.png)

Preliminary experiments on the Amazon-Books dataset also showed a similar trend, where our implementation consistently outperformed the SASRec baseline, confirming the robustness of the multi-level contrastive learning approach across different product categories.

**Discussion and Critical Observations:** Our reproduction confirms that while the MCLSR framework is conceptually powerful, its original implementation details and reported results are **highly unreliable**. By resolving methodological ambiguities — specifically the lack of co-action weights and incorrect tensor shapes — we achieved a Recall@20 of **0.0420**. 

This is not just an improvement; it is a **total recalibration** of the model's potential. The fact that our independent reproduction exceeds the original (scaled) metrics by such a wide margin proves that the original paper significantly under-reported the framework's capability due to suboptimal structural modeling.

### 5. Conclusion
This work provides an independent validation of the MCLSR framework. By implementing the model from scratch and utilizing a rigorous evaluation pipeline, we confirmed the consistent superiority of multi-level contrastive learning over purely sequential modeling (SASRec). Our results establish a reliable and reproducible performance benchmark for the *Amazon-Clothing* dataset, providing a more transparent basis for future research in sequential recommendation.

### References
1. Wang, Z., et al. 2022. Multi-level Contrastive Learning Framework for Sequential Recommendation. In *CIKM*.
2. Kang, W.-C., & McAuley, J. 2018. Self-attentive sequential recommendation. In *ICDM*.
3. He, X., et al. 2020. LightGCN: Simplifying and Powering graph convolution network for recommendation. In *SIGIR*.

# LogQ Correction
We evaluate our refined MCLSR implementation against both a pure SASRec baseline and an enhanced SASRec version utilizing LogQ correction. All results are reported on the *Amazon-Clothing* dataset using a rigorous full-ranking protocol.

**Table 3: Performance comparison across different architectures and enhancements**
| Model | Recall@20 | NDCG@20 | HitRate@20 | Recall@50 | NDCG@50 | HitRate@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SASRec (Pure Baseline) | 0.0184 | 0.0076 | 0.0272 | 0.0314 | 0.0106 | 0.0490 |
| SASRec (+ LogQ) | 0.0339 | 0.0144 | 0.0528 | 0.0588 | 0.0199 | 0.0898 |
| **MCLSR (Our Refined)** | **0.0420** | **0.0189** | **0.0660** | **0.0704** | **0.0254** | **0.1089** |

![Hit Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-32-02.png)

![NDCG Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-33-32.png)

![Recall Metrics](https://github.com/Aksinya-Bykova/mclsr-abstract/blob/main/assets/Screenshot%20from%202026-04-03%2023-35-09.png)

Our current work investigates the broader applicability of **Sampling-Bias Correction (LogQ)** [4] within different sequential recommendation architectures. This method, originally proposed by Google researchers, addresses the inherent bias in non-uniform negative sampling by adjusting model logits based on item frequency:

$$s^c(x, y) = s(x, y) - \lambda \log(p_j)$$

Our preliminary experiments confirm that popularity bias is a significant bottleneck for standard sequential models. Specifically, integrating LogQ correction into the **SASRec** architecture yielded a substantial performance boost, nearly doubling the baseline metrics on the *Amazon-Clothing* dataset (as shown in Table 3).

### Additional References
[4] Yi, X., et al. 2019. [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/). In *RecSys*.


# Optimization Tricks (filter top k)

For our research goals, the main thing is to test hypotheses as quickly as possible. On one hand, the code needs to run fast; on the other hand, we don't want to waste time developing code that will never go into production. Here, however, the optimization made sense: it may take N hours -> a few minutes on CPU


## Old version:

```
def _filter_matrix_by_top_k(matrix, k):
    # --- OLD IMPLEMENTATION ---
    # mat = matrix.tolil()
    # for i in range(mat.shape[0]):
    #     if len(mat.rows[i]) <= k:
    #         continue
    #     data = np.array(mat.data[i])
    #     top_k_indices = np.argpartition(data, -k)[-k:]
    #     mat.data[i] = [mat.data[i][j] for j in top_k_indices]
    #     mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]
    # return mat.tocsr()
```

## Optimized:

```
def filter_matrix_by_top_k(matrix, k):
    mat = matrix.tocsr()

    for i in range(mat.shape[0]):
        start = mat.indptr[i]
        end = mat.indptr[i + 1]

        if end - start > k:
            row_view = mat.data[start:end]

            threshold = np.partition(row_view, -k)[-k]

            row_view[row_view < threshold] = 0

    mat.eliminate_zeros()

    return mat
```
### Explanation
```
matrix.tolil() vs matrix.tocsr()
```
LIL - separated user list, CSR - only 3 arrays (data, indices, indptr). In the second access is much faster. CSR doesn't provide fast insertion, but we don't need that here. We work with user-user graph. Data - stores the floating-point values of interaction counts (co-action weights). Indices - stores the column index for each value in data, these are the IDs of the neighbor users. Indptr: Maps the rows to the data and indices arrays

Example: user 0 has data[0-4], user 1 has data[5-9] etc. For example data[7] = 15.0, indices[7] = 101 which means weight beetween user 1 (because it has data[5-9], which includes 7) and user 101 is 15.0. We will figure out why it is important later 

```
# for i in range(mat.shape[0]):
#     if len(mat.rows[i]) <= k:
#         continue
#     data = np.array(mat.data[i])
#     top_k_indices = np.argpartition(data, -k)[-k:]
#     mat.data[i] = [mat.data[i][j] for j in top_k_indices]
#     mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]
# return mat.tocsr()
```
Old implementation. Here mat.rows - array of neighbour IDs, mat.data - array of c-action weight of these neighbours
```
for i in range(mat.shape[0]):
```
Take every user, i - user ID
```
if len(mat.rows[i]) <= k: continue
```
Stopping condition (we need top-k)
```
data = np.array(mat.data[i])
```
Convert user weights array to numpy array
```
top_k_indices = np.argpartition(data, -k)[-k:]
mat.data[i] = [mat.data[i][j] for j in top_k_indices]
mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]
```
It doesn't fully sort; it just moves the top-k elements to the end, then takes them

**Why old imlemetation is bad practice?**
LIL stores lists sparsely - so the processor has to execute mat.data[i][j] very slowly in python! Complexity is O(VN), where V - number of of rows, N - the row dencity. Firstly find [i] then [j]. Unlike C++, in Python array's elements are not contiguous in memory. So elemets are sparesed and thehe are cache misses: processor just is just waiting for RAM. Also in the loop are redundant objects: object creation and garbage collection - it's not ok

**What do instead?**
```
row_view[...] = 0
```
Inplace operation. Don't have to create new object

In CSR, all (9,000,000) numbers are stored tightly packed together. When the CPU accesses an element, it fetches data in 64-byte cache lines, automatically pre-loading multiple subsequent weights into the L1 cache. This significantly reduces memory latency by minimizing direct RAM access, effectively aligning the implementation with the underlying hardware architecture

I did [something similar in C++ in 2024 year](https://github.com/Aksinya-Bykova/Integral-OMP) aligning the structure for a multithreading program

---

We can actually see cache sizes:
```
mrass@mrass-RedmiBook-13-R:~$ lscpu | grep -E 'L1|L2|L3'
L1d cache:                            192 KiB (6 instances)
L1i cache:                            192 KiB (6 instances)
L2 cache:                             3 MiB (6 instances)
L3 cache:                             8 MiB (2 instances)
Vulnerability L1tf:                   Not affected
mrass@mrass-RedmiBook-13-R:~$ 
```

We physically can't store such big data (9 * 10^6 * 8 bytes = 72mb) without RAM

---

```
threshold = np.partition(row_slice, -k)[-k]
row_view[row_view < threshold] = 0
```

Physically removing an element from a contiguous array - bad idea. It means shifting every subsequent element by one position to fill the gap. Performing such shifts for every means O(E^2) complexity (where E - all edges)
 
Insead we use trick: mark all useless elements 0. So we avoid these redundant memory shifts during the iteration. ```np.partition``` finds element with mean time O(n)

```row_view[row_view < threshold] = 0``` - NumPy operation needs O(n)

Then
```
mat.eliminate_zeros()
```
After loop it removes all non-zero element to a new place tightly packed together  

Item-Item Graph works the same way also using this method


# Matrix Optimization Tricks
To treat users and items as nodes in a single unified graph, we construct a bipartite adjacency matrix $\mathbf{A} \in \mathbb{R}^{(N+M) \times (N+M)}$, where $N$ is the number of users and $M$ is the number of items.

$$
\mathbf{A} = \begin{pmatrix} \mathbf{0} & \mathbf{R} 
\\ 
\mathbf{R}^T & \mathbf{0} \end{pmatrix}
$$

*   $\mathbf{R} \in \mathbb{R}^{N \times M}$ is the User-Item interaction matrix.
*   The off-diagonal blocks represent connections between different sets (User-to-Item and Item-to-User).
*   The diagonal blocks are zero because we assume no initial self-loops or intra-set edges.
```
R = sparse_matrix.tocsr()

upper_right = R
lower_left = R.T

upper_left = sp.csr_matrix((fst_dim, fst_dim))
lower_right = sp.csr_matrix((snd_dim, snd_dim))

adj_mat = sp.bmat([[upper_left, upper_right], [lower_left, lower_right]])
```

Then we have to calculate 

$$
\mathcal{L} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}
$$

There is a big optimization problem

## Old Version
```
# it's ok
rowsum = np.array(adj_mat.sum(1))
d_inv = np.power(rowsum, -0.5).flatten()
d_inv[np.isinf(d_inv)] = 0.

# bad practice
# d_mat_inv = sp.diags(d_inv)
# norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
```

1. ``rowsum = np.array(adj_mat.sum(1))``

For each node $i$ we calculate its degree $d_i$:

$$
d_i = \sum_{j} \mathbf{A}_{ij}
$$

2. `d_inv = np.power(rowsum, -0.5).flatten()` and `d_inv[np.isinf(d_inv)] = 0.`

We compute the scaling coefficient vector $v_i$, handling cases of zero-degree nodes (isolated nodes) to avoid division by zero:

$$
v_i = 
\begin{cases} 
d_i^{  -1/2} & \text{if } d_i > 0 \\ 
0 & \text{if } d_i = 0 
\end{cases}
$$

3. `d_mat_inv = sp.diags(d_inv)`
We construct a sparse diagonal matrix $\mathbf{D}^{-1/2}$ with the computed coefficients along the main diagonal:

$$
\mathbf{D}^{-1/2} = \text{diag}(v_0, v_1, \dots, v_{n-1})
$$

Why is it bad? There is an allocation + copy + indexing (to take elements on diagonal)

4. `norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)`

Finally

$$
\mathcal{L} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
$$


## Optimized 3-4 steps
```
norm_adj = adj_mat.multiply(d_inv[:, np.newaxis]).multiply(d_inv)
```

There is a trick:

$$
\mathcal{L}_{ij} = (\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2})_{ij} = v_i \cdot \mathbf{A}_{ij} \cdot v_j = \frac{\mathbf{A}_{ij}}{\sqrt{d_i \cdot d_j}}
$$

How does it work?

$\mathbf{D}^{-1/2} \mathbf{A}$

`adj_mat.multiply(d_inv[:, np.newaxis])`

`[:, np.newaxis]` - reshape `d_inv` from (N, ) to (N, 1)

$\dots \mathbf{D}^{-1/2}$

`.multiply(d_inv)` - element-wise multiplication instead of matrix multiplication

