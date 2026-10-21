
# Initial Preprocessing Considerations on the Beijing Multi-Site Air Quality Dataset

For a spatial-temporal series forecasting problem utilizing the **Beijing Multi-Site Air Quality** dataset, preprocessing is highly challenging because the dataset is a hybrid of three distinct data distributions:
1. **Highly right-skewed, heavy-tailed pollutants** (PM2.5, PM10, SO2, NO2, CO, O3) which feature massive, sporadic smog spikes.
2. **Sparsely zero-inflated physical variables** (RAIN) which are mostly 0 but have sudden, intense positive spikes.
3. **Symmetric, seasonally cyclical variables** (TEMP, DEWP, PRES). Crucially, TEMP and DEWP drop below $0^\circ\text{C}$ (negative values) during Beijing winters, while PRES (Atmospheric Pressure) is strictly positive with a narrow range (always hovering around $1013\text{ hPa}$).

To optimize a spatial-temporal model (such as a Graph Convolutional LSTM or GNN-GRU) on this dataset, the normalization methods are ranked below from **most suitable** to **least suitable**.

---

### Rank 1: Standardization (Z-score Normalization) — *Most Suitable*

* **Why it fits here:** 
  Standardization is the standard baseline choice for this dataset. Because it centers all variables around $0$ with a standard deviation of $1$, it is highly compatible with the optimization mechanics of deep spatial-temporal neural networks.
* **Dataset-Specific Performance:** 
  * It seamlessly handles both negative meteorological features (TEMP, DEWP) and positive features (pollutants) without requiring arbitrary shift parameters.
  * When applied **globally** across all 12 stations, it preserves spatial gradients (i.e., the relative spatial variances and means between stations), allowing spatial-temporal networks to accurately model how pollution and weather disperse across Beijing.
  * It is resilient to extreme smog events. A PM2.5 spike will simply register as a high positive Z-score (e.g., $+4.5$), without crushing the variance of cleaner days.
* **Primary Drawback:** 
  It shifts true zeroes to non-zero values, which can slightly obscure the binary "rain vs. no rain" threshold for the RAIN feature.

---

### Rank 2: Robust Scaling (Median & IQR)

* **Why it fits here:** 
  Beijing's air quality is notorious for extreme, transient pollution spikes (e.g., PM2.5 occasionally skyrocketing past $800\ \mu\text{g/m}^3$ while the median sits around $50\ \mu\text{g/m}^3$). 
* **Dataset-Specific Performance:** 
  * By scaling using the Interquartile Range (IQR) and centering on the median, Robust Scaling ensures that the baseline daily weather and moderate air quality patterns are scaled with high precision, completely unskewed by extreme smog anomalies or sensor errors.
* **Primary Drawback:** 
  It does not compress the scale of outliers. Extreme smog days will yield very large normalized values (e.g., $>15.0$), which can cause exploding gradients in LSTMs unless the inputs are explicitly clipped before being fed into the network.

---

### Rank 3: Power Transformations (specifically Yeo-Johnson)

* **Why it fits here:** 
  Many of the target variables (pollutants and wind speed) are heavily non-normal and right-skewed. Power transformations force these distributions into symmetric, Gaussian curves.
* **Dataset-Specific Performance:** 
  * Transforming skewed pollutants into Gaussian curves aligns them with standard regression loss functions (such as Mean Squared Error).
  * The Yeo-Johnson variant must be used (rather than Box-Cox) because it mathematically supports the negative values present in winter temperatures (TEMP) and dew points (DEWP).
* **Primary Drawback:** 
  The inverse transformation (needed to convert the model's predictions back into real-world units like $\mu\text{g/m}^3$ for evaluation) is highly non-linear. This can warp prediction errors and make back-transformation mathematically complex and prone to bias.

---

### Rank 4: Min-Max Scaling

* **Why it fits here:** 
  It is incredibly common in simple tutorial workflows and is straightforward to invert. 
* **Dataset-Specific Performance:** 
  * It preserves physical zeroes (perfect for RAIN) and restricts all features to a strict, numerically stable $[0, 1]$ range.
* **Why it ranks lower:** 
  Because of Beijing's extreme pollution spikes, a global or local Min-Max scaler will heavily compress the dataset. If the maximum PM2.5 recorded is $800\ \mu\text{g/m}^3$, a normal day of $50\ \mu\text{g/m}^3$ gets mapped to a tiny value ($\approx 0.06$). Consequently, the network is forced to learn subtle, daily spatial-temporal variations within an extremely narrow, squashed interval, which degrades prediction accuracy.

---

### Rank 5: Logarithmic Transformation

* **Why it fits here:** 
  Applying a $\ln(x+1)$ transformation is highly effective at stabilizing variance and linearizing exponential propagation for right-skewed features like pollutants and wind speed.
* **Why it ranks lower:** 
  * It **cannot** be applied uniformly to the entire dataset. Because Beijing winters feature sub-zero temperatures (TEMP and DEWP drop below $0^\circ\text{C}$), a log transform will fail mathematically. 
  * To use it, you must construct a disjoint pipeline: log-transforming the pollutants, while using Z-score or Min-Max for the meteorological features. This hybrid approach increases pipeline complexity. Furthermore, back-transforming log predictions introduces systemic underestimation bias.

---

### Rank 6: Max Absolute Scaling (MaxAbs) — *Least Suitable*

* **Why it fits here:** 
  It is mathematically simple, preserves physical zeroes, and retains signs.
* **Why it is the least suitable:** 
  * It suffers from the same outlier-compression vulnerability as Min-Max scaling for pollutants.
  * **The major failure point is Atmospheric Pressure (PRES):** In Beijing, pressure is strictly positive and fluctuates in a narrow band (typically between $990\text{ hPa}$ and $1040\text{ hPa}$). Because MaxAbs does not center the data, dividing PRES by its absolute maximum ($\approx 1040$) maps the entire feature to a tiny, flat interval between $[0.95, 1.0]$. Without zero-centering, the network will struggle to extract any meaningful gradients from atmospheric pressure, effectively rendering a crucial meteorological feature useless.

# Is using a different scaling / preprocessing pipeline per feature a good idea?

Using a **hybrid scaling strategy**—where each feature in a dataset is processed using its own mathematically optimal scaler—is **conceptually excellent and highly effective for handling physical heterogeneity, but it introduces major optimization challenges if implemented naively.** 

In professional machine learning workflows, this approach is highly recommended, provided it is coupled with a secondary step to align the value ranges. 

Below is a formal, multi-dimensional mathematical analysis of the implications of this strategy.

---

### 1. The Theoretical Justification: Why it is a Good Idea
In a complex dataset like the Beijing Multi-Site Air Quality dataset, features represent fundamentally different physical phenomena. Treating them with a single, uniform scaling function forces mathematical compromises. A hybrid strategy resolves this by matching the transformation to the underlying probability density function (PDF):

* **Information Preservation:** Right-skewed variables like $PM2.5$ span orders of magnitude. A logarithmic or Yeo-Johnson transform maps them to a normal distribution, preventing high-concentration days from compressing low-concentration variations.
* **Preserving Physical Boundaries:** Variables like $\text{RAIN}$ are sparse and bounded at $0$. A Min-Max or Max-Abs scaler preserves the exact physical zero, allowing the network to use zero as a discrete indicator of "no rain."
* **Stabilizing Cyclical Signals:** Meteorological variables like $\text{TEMP}$ are continuous and cross zero. Standardization (Z-score) is optimal here as it preserves the negative-to-positive transition without distorting the variance.

---

### 2. The Optimization Cost: Mismatched Value Ranges
While a hybrid strategy optimizes the *distribution shape* of each feature, it results in **mismatched value ranges** (e.g., some features are strictly in $[0, 1]$, some are in $[-1, 1]$, and others are Z-scores in $(-\infty, \infty)$). This range mismatch negatively impacts gradient-based optimization in neural networks.

#### A. Ill-Conditioning of the Loss Hessian ($\mathbf{H}$)
Let $E(\mathbf{w})$ be the loss function of a neural network with weight vector $\mathbf{w}$. The speed of convergence of gradient descent is dictated by the **condition number** of the Hessian matrix of the loss function:

$$\kappa(\mathbf{H}) = \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}(\mathbf{H})}$$

Where $\lambda_{\max}$ and $\lambda_{\min}$ are the maximum and minimum eigenvalues of $\mathbf{H}$. 
* If features have radically different scales (e.g., Feature 1 is compressed to $[0, 0.05]$ by a poor scaler, while Feature 2 is in $[-3, 3]$), the gradients with respect to the weights of Feature 1 will be exceedingly small compared to Feature 2.
* This makes the Hessian **ill-conditioned** ($\kappa \gg 1$), creating highly elongated, elliptical "canyons" in the loss landscape. Gradient descent will oscillate wildly along the high-gradient dimensions and make negligible progress along the low-gradient dimensions, severely slowing down convergence.

#### B. Isotropic Weight Initialization Mismatch
Modern neural networks initialize their weights $\mathbf{w}$ isotropically (typically using Xavier/Glorot or He initialization). These frameworks assume that all input features $\mathbf{x}$ are zero-centered with unit variance ($\mu = 0, \sigma^2 = 1$). 
* If Feature A is bounded in $[0, 1]$ (mean $\approx 0.5$, variance $\approx 0.08$) and Feature B is standardized (mean $0$, variance $1$), the initialized network will naturally be biased toward, and dominated by, the gradients of Feature B.

---

### 3. Impact on Spatial-Temporal Architectures

#### A. Spatial Convolution and Aggregation (GNNs)
In multi-site forecasting, Graph Neural Networks (GNNs) aggregate feature vectors across neighboring stations:

$$\mathbf{h}_v = \text{AGGREGATE}\left(\{ \mathbf{x}_u : u \in \mathcal{N}(v) \}\right)$$

If different features have mismatched ranges, features with larger absolute scales will dominate the aggregation operations. This forces the GNN to learn to scale down those features, wasting parameter capacity.

#### B. Dot-Product Self-Attention
If using Spatial-Temporal Transformers, the attention scores are computed via scaled dot-products of Query ($\mathbf{Q}$) and Key ($\mathbf{K}$) matrices:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

If the feature scales within the input vector are mismatched, the dot products will be dominated by the high-magnitude features. This pushes the inputs to the `softmax` function into regions of extreme values, saturating the function and causing **vanishing gradients** during backpropagation.

---

### 4. The Golden Rule: Target vs. Auxiliary Variables
When implementing a hybrid strategy, you must mathematically separate your variables into two categories:

1. **Target Variables (e.g., $PM2.5$):** The variables the model is trying to forecast. 
   * **Constraint:** Any scaler applied to the target variable **must be strictly invertible** with a stable, closed-form inverse function (e.g., standard Z-score or Min-Max). Applying highly complex non-linear transforms (like Yeo-Johnson with extreme $\lambda$) to targets makes back-transforming predictions highly unstable, magnifying tiny model errors into huge physical-scale errors.
2. **Auxiliary Inputs (e.g., $\text{TEMP}$, $\text{PRES}$, $\text{RAIN}$):** Variables used strictly as context.
   * **Constraint:** These variables are never predicted; hence, they **never need to be inverted**. You can apply highly complex, non-invertible, or non-linear scaling methods to these variables without any risk to the output evaluation.

---

### 5. The Solution: The Two-Step Hybrid Pipeline
To harness the benefits of feature-specific scaling without suffering from the optimization issues of mismatched ranges, the industry-standard solution is to use a **Two-Step Preprocessing Pipeline**:

```
Raw Feature ──> Step 1: Distribution Alignment ──> Step 2: Range Uniformity ──> Model Input
                (Feature-Specific Scaler)          (Global Standardization)
```

1. **Step 1 (Feature-Specific Transformation):** Apply the optimal mathematical transform to reshape the PDF of each feature independently (e.g., Log-plus-one for pollutants, Identity for temperature, Box-Cox for wind speed).
2. **Step 2 (Universal Standardization):** Apply a standard Z-score scaling ($\mu=0, \sigma=1$) to **all** the resulting transformed features.

#### Why this is the optimal approach:
* **Step 1** corrects the non-linear skewness, eliminates heavy tails, and stabilizes variance.
* **Step 2** maps all those newly reshaped distributions to a uniform mathematical range centered at $0$ with unit variance. 

This gives the neural network perfectly shaped, symmetric input distributions with a perfectly conditioned Hessian, maximizing gradient flow and convergence speed.

# Final Pipeline

Applying the **Two-Step Hybrid Pipeline** to the Beijing Multi-Site Air Quality dataset requires categorizing its 18 features into distinct groups based on their physical behavior, mathematical distributions, and whether they serve as target or auxiliary variables. 

Below are all the highly meaningful and useful instances of this pipeline designed specifically for this dataset.

---

### Group 1: Target Pollutants (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`)
* **Characteristics:** Highly right-skewed, strictly non-negative, and prone to extreme smog-related spikes. Because these are your forecast targets, **Step 1 must be strictly and stably invertible.**

```
Raw Pollutant ──> Step 1: Log-plus-one ──> Step 2: Global Z-score ──> Network Input
```

#### Step 1: Distribution Alignment (Logarithmic Transform)
To compress the heavy right tail without losing fine-grained variance during low-pollution days, we use the natural log-plus-one transformation:

$$y_i = \ln(x_i + 1)$$

This is highly stable and easily invertible. If $\hat{y}_i'$ is the network's normalized prediction, we can recover the physical concentration ($\hat{x}_i$) without numerical instability:

$$\hat{x}_i = \exp\left(\hat{y}_i' \cdot \sigma_{y} + \mu_{y}\right) - 1$$

#### Step 2: Range Uniformity (Global Z-Score Standardization)
To align the transformed pollutant concentrations with the rest of the network's inputs:

$$z_i = \frac{y_i - \mu_{\text{global}}}{\sigma_{\text{global}} + \epsilon}$$

*Note: $\mu_{\text{global}}$ and $\sigma_{\text{global}}$ must be calculated across all 12 stations combined to preserve the absolute spatial differences between clean suburban and highly polluted urban sites.*

---

### Group 2: Cyclical Meteorological Variables (`TEMP`, `DEWP`)
* **Characteristics:** Continuously crossing zero (negative in winter, positive in summer), roughly symmetric, and Gaussian. No non-linear reshaping is required.

```
Raw TEMP/DEWP ──> Step 1: Identity Transform ──> Step 2: Global Z-score ──> Network Input
```

#### Step 1: Distribution Alignment (Identity Transform)
Since these variables already exhibit symmetric, near-Gaussian bell curves, any non-linear transformation would distort their natural physical meaning.

$$y_i = x_i$$

#### Step 2: Range Uniformity (Global Z-Score Standardization)
To remove the physical units ($^\circ\text{C}$) and map them to a unit-variance scale centered at zero:

$$z_i = \frac{x_i - \mu_{\text{global}}}{\sigma_{\text{global}} + \epsilon}$$

Using global parameters preserves the critical spatial gradient (e.g., mountainous stations being colder than downtown stations).

---

### Group 3: High-Offset, Low-Variance Continuous Variables (`PRES`)
* **Characteristics:** Strictly positive, with a huge baseline offset ($\sim 1013\text{ hPa}$) but very narrow relative variance (typically fluctuating only between $990\text{ hPa}$ and $1045\text{ hPa}$).

```
Raw PRES ──> Step 1: Identity Transform ──> Step 2: Global Z-score ──> Network Input
```

#### Step 1: Distribution Alignment (Identity Transform)
The distribution of atmospheric pressure is already highly Gaussian, so no non-linear transformation is needed:

$$y_i = x_i$$

#### Step 2: Range Uniformity (Global Z-Score Standardization)
This step is **paramount** for `PRES`. Because the baseline is so far from zero, a simple division by the maximum (like MaxAbs) would compress all values into an unusable interval like $[0.95, 1.0]$. Applying the Z-score:

$$z_i = \frac{x_i - \mu_{\text{global}}}{\sigma_{\text{global}} + \epsilon}$$

This completely strips away the $1000\text{ hPa}$ offset, centering the daily high- and low-pressure systems precisely around $0$, allowing the network's weights to easily capture pressure gradients.

---

### Group 4: Extreme Zero-Inflated Sparse Variables (`RAIN`)
* **Characteristics:** Extremely sparse (often $>95\%$ of values are exactly $0.0\text{ mm}$). When non-zero, it is extremely right-skewed. Traditional continuous transforms fail because of the massive point-mass at zero.

```
                  ┌──> Step 1a: Binary Indicator (0 or 1) ──────────┐
Raw RAIN ──> Split│                                                 ├──> Concatenate ──> Step 2: Standardize Intensity ──> Input
                  └──> Step 1b: Log-plus-one (Magnitude if > 0) ────┘
```

#### Step 1: Distribution Alignment (Feature Decomposition)
Instead of forcing a single scaler onto this binary-continuous hybrid, we decompose `RAIN` into two distinct features:
1. **Precipitation Indicator ($y_{\text{ind}}$):** A binary flag indicating whether it rained.
   
   $$y_{\text{ind}, i} = \begin{cases} 1 & \text{if } x_i > 0 \\ 0 & \text{if } x_i = 0 \end{cases}$$

2. **Precipitation Intensity ($y_{\text{int}}$):** The log-transformed magnitude of rain, mapped to zero if dry:
   
   $$y_{\text{int}, i} = \begin{cases} \ln(x_i) & \text{if } x_i > 0 \\ 0 & \text{if } x_i = 0 \end{cases}$$

#### Step 2: Range Uniformity
* For $y_{\text{ind}}$, **no standardization** is applied. It remains a clean $[0, 1]$ binary input.
* For $y_{\text{int}}$, we standardize only the non-zero active rain events:
  
  $$z_{\text{int}, i} = \frac{y_{\text{int}, i} - \mu_{\text{active}}}{\sigma_{\text{active}} + \epsilon}$$

---

### Group 5: Bounded Skewed Variables (`WSPM` - Wind Speed)
* **Characteristics:** Strictly non-negative, bounded at $0$ (during calm wind conditions), and moderately right-skewed.

```
Raw WSPM ──> Step 1: Box-Cox / Yeo-Johnson ──> Step 2: Global Z-score ──> Network Input
```

#### Step 1: Distribution Alignment (Power Transform)
Because wind speed cannot be negative but is not as heavily skewed as pollutants, a Yeo-Johnson power transform is highly effective at normalizing its distribution:

$$y_i = \psi(x_i, \lambda)$$

Since `WSPM` is an auxiliary input (never predicted), we do not need to worry about inverting this transform. We can let maximum likelihood estimation (MLE) find the mathematically perfect $\lambda$.

#### Step 2: Range Uniformity (Global Z-Score Standardization)
We standardize the power-transformed wind speed to align its variance with the rest of the inputs:

$$z_i = \frac{y_i - \mu_{\text{global}}}{\sigma_{\text{global}} + \epsilon}$$

---

### Group 6: Temporal and Spatial Cyclical Variables (`hour`, `month`, `wd` - Wind Direction)
* **Characteristics:** Bounded, circular integer or categorical variables. For example, `hour` 23 is physically adjacent to `hour` 0, and Wind Direction (`wd` e.g., N, NNE, NE) represents angles on a $360^\circ$ circle. Linear scaling fails because it treats the boundaries as opposites.

```
Raw Cyclical ──> Step 1: Coordinate Projection (Sine & Cosine) ──> Step 2: Skip (Already Bounded) ──> Network Input
```

#### Step 1: Distribution Alignment (Trigonometric Circular Projection)
We map these 1D cyclical variables into 2D Cartesian coordinates:
1. **Convert to Radians ($\theta$):**
   * For Hour ($t \in [0, 23]$): $\theta_i = \frac{2\pi \cdot t_i}{24}$
   * For Month ($t \in [1, 12]$): $\theta_i = \frac{2\pi \cdot (t_i - 1)}{12}$
   * For Wind Direction (`wd`): Map the 16 cardinal directions to angles (e.g., $N = 0^\circ$, $E = 90^\circ$), then convert to radians: $\theta_i = \frac{\text{degrees}_i \cdot \pi}{188}$

2. **Project to Sine and Cosine Components:**
   
   $$y_{\sin, i} = \sin(\theta_i), \quad y_{\cos, i} = \cos(\theta_i)$$

This preserves the physical circular proximity: the distance between hour 23 and hour 0 is now identical to the distance between hour 12 and hour 13.

#### Step 2: Range Uniformity
**Skip.** Because sine and cosine naturally output values strictly within the range $[-1, 1]$ centered at $0$, they are already perfectly scaled and optimized for neural network weight matrices.

---

### Group 7: Discrete Categorical Identifiers (`station`)
* **Characteristics:** High-cardinality discrete identifiers representing the 12 monitoring sites.

```
Station Name (String) ──> Step 1: Integer ID ──> Step 2: Learned Entity Embedding ──> Network Input
```

#### Step 1: Distribution Alignment (Label Encoding)
Map each of the 12 unique station strings to an integer index:

$$y_i \in \{0, 1, 2, \dots, 11\}$$

#### Step 2: Range Uniformity (Learned Entity Embeddings)
Instead of using One-Hot encoding (which creates sparse, high-dimensional vectors), we feed these integer IDs into a trainable **Embedding Layer** within the neural network:

$$\mathbf{E} = \text{EmbeddingLayer}(y_i) \in \mathbb{R}^{12 \times d}$$

During backpropagation, the network learns a continuous, dense $d$-dimensional vector for each station. Because these embedding weights are initialized using standard Gaussian methods (like Xavier initialization) and updated via weight decay, the output vectors are naturally self-standardizing and zero-centered.