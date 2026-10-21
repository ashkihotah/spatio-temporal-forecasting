To normalize numerical vectors (features) for machine learning, several mathematical frameworks exist. In spatial-temporal series forecasting—where data points are ordered across both time (temporal) and space (multi-site/spatial grids)—the choice of normalization directly impacts spatial gradients, temporal trends, gradient flow, and structural preservation.

Below is a formal, deep, and mathematically rigorous analysis of the primary normalization methods, structured according to your requirements.

---

### Mathematical Notation
Let $\mathbf{x} = [x_1, x_2, \dots, x_N]^T \in \mathbb{R}^N$ represent a continuous float vector corresponding to a specific feature observed over $N$ samples (which may span multiple time steps and spatial locations). Let $\mathbf{x}'$ represent the normalized vector.

---

### 1. Min-Max Scaling (Linear Range Bounding)

#### A. General Idea
Min-Max scaling linearly projects the original data range into a user-defined closed interval, most commonly $[0, 1]$ or $[-1, 1]$. It preserves the exact relative distances between all data points while enforcing strict upper and lower boundaries.

#### B. Formal Assumptions
1. **Bounded Support:** The underlying physical process has strict, known, or historically stable minimum ($\min(\mathbf{x})$) and maximum ($\max(\mathbf{x})$) values.
2. **Stationarity of Bounds:** Future unseen test observations will not violate the historical bounds ($x_{\text{test}} \in [\min(\mathbf{x}_{\text{train}}), \max(\mathbf{x}_{\text{train}})]$).
3. **Absence of Heavy Tails/Outliers:** The distribution does not contain extreme, transient anomalies that would compress the nominal data into an imperceptibly narrow interval.

#### C. Formal Computation
For a target range $[a, b]$ (where $a < b$):

$$x_i' = a + \frac{(x_i - \min(\mathbf{x}))(b - a)}{\max(\mathbf{x}) - \min(\mathbf{x}) + \epsilon}$$

Where $\epsilon > 0$ is an infinitesimally small smoothing parameter (e.g., $10^{-8}$) to prevent division by zero in zero-variance vectors. For mapping to $[0, 1]$:

$$x_i' = \frac{x_i - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x}) + \epsilon}$$

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Preservation of Sparsity:** If the lower bound is $0$ and the target range is $[0, 1]$, true physical zeroes (e.g., zero precipitation, zero wind speed) remain exactly $0$.
    *   **Activation Function Compatibility:** Bound-constrained outputs map to the active regions of saturating activation functions (e.g., Sigmoid, Tanh).
    *   **Linear Scale Preservation:** It does not distort the linear relationships or ratios between temporal steps.

*   **Cons:**
    *   **Extreme Vulnerability to Outliers:** In environmental sensing, a single sensor anomaly or extreme weather event (e.g., an unprecedented storm) artificially inflates the denominator. This permanently squashes the regular, meaningful variations of the time series into a flat, low-variance signal.
    *   **Out-of-Bounds Concept Drift:** If a warming climate or extreme event causes a future temperature to exceed the historical maximum, the normalized value will exceed $1.0$. This can destabilize neural network weights that expect input values strictly within $[0, 1]$.

---

### 2. Standardization (Z-score Normalization)

#### A. General Idea
Standardization translates and scales the vector so that it exhibits a mean of $0$ and a standard deviation of $1$. It centers the feature space around the origin without imposing hard boundaries on the output.

#### B. Formal Assumptions
1. **Gaussian Distribution:** The data distribution is approximately normal (Gaussian), or at least symmetric and unimodal.
2. **Stationarity of Moments:** The first moment (mean, $\mu$) and the second moment (variance, $\sigma^2$) are stationary over time and space.

#### C. Formal Computation
Let the sample mean $\mu$ and standard deviation $\sigma$ be defined as:

$$\mu = \frac{1}{N}\sum_{i=1}^N x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$$

The normalized elements are computed as:

$$x_i' = \frac{x_i - \mu}{\sigma + \epsilon}$$

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Optimal Gradient Behavior:** Centering variables at zero prevents the "zig-zagging" gradient updates in backpropagation. This accelerates convergence in deep recurrent and convolutional spatial-temporal models (such as LSTMs and GNNs).
    *   **Outlier Resilience:** Unlike Min-Max scaling, Z-score does not squash the bulk of the data when an anomaly occurs. Anomalies simply manifest as high absolute Z-scores (e.g., $|x_i'| > 3$).
    *   **Spatial Comparison under Global Scaling:** If standardizing globally across all stations, it maintains the relative spatial variance and mean differences between stations while stabilizing the feature's variance.

*   **Cons:**
    *   **Destruction of Sparsity:** Subtracting $\mu$ shifts physical zero values to non-zero values (e.g., a dry hour of $0.0\text{ mm}$ rain becomes $-0.21$). This makes it harder for the model to learn threshold-based physical behaviors (such as precipitation vs. no precipitation).
    *   **Unbounded Domain:** The output range is $(-\infty, +\infty)$, which can lead to numerical instability if a deep network layer generates excessively large outputs during extreme events.

---

### 3. Robust Scaling (Quantile/IQR-based Scaling)

#### A. General Idea
Robust Scaling centers and scales the data using statistics that are highly resilient to anomalies: the median and the Interquartile Range (IQR). It maps the middle $50\%$ of the data to a standardized range, rendering the scaling parameters impervious to extreme values.

#### B. Formal Assumptions
1. **Anomalous/Heavy-Tailed Distribution:** The data contains high levels of noise, sensor errors, or highly skewed, non-Gaussian behavior.
2. **Representative Core:** The central half of the distribution (between the 25th and 75th percentiles) represents the true nominal state of the system.

#### C. Formal Computation
Let $Q_1(\mathbf{x})$, $Q_2(\mathbf{x})$ (median), and $Q_3(\mathbf{x})$ represent the 25th, 50th, and 75th percentiles of $\mathbf{x}$, respectively. The Interquartile Range ($IQR$) is defined as:

$$IQR = Q_3(\mathbf{x}) - Q_1(\mathbf{x})$$

The transformation is given by:

$$x_i' = \frac{x_i - Q_2(\mathbf{x})}{IQR + \epsilon}$$

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Extreme Outlier Immunity:** If a sensor frequently malfunctions (generating extreme spikes or drops), the scale parameters ($Q_2$ and $IQR$) remain completely unaffected. The "normal" physical dynamics of the spatial-temporal series remain cleanly scaled.
    *   **Stationary Representation of Seasonal Fluctuations:** Transient seasonal extremes do not distort the normalized scale of baseline conditions.

*   **Cons:**
    *   **Loss of Outlier Scale Compression:** Because the scaling factor is based strictly on the $IQR$, extreme outliers can result in very large normalized values (e.g., $x_i' = 85.0$). This can trigger exploding gradients in Recurrent Neural Networks (RNNs) if not clipped.
    *   **Destruction of Sparsity:** Similar to standardization, subtracting the median shifts true zero values to non-zero values.

---

### 4. Max Absolute Scaling (MaxAbs)

#### A. General Idea
Max Absolute scaling divides each element of the vector by the maximum absolute value in the vector. This maps the data to the interval $[-1, 1]$ without shifting the center of the distribution.

#### B. Formal Assumptions
1. **Significance of Zero:** The zero value ($0$) represents a physical baseline or absolute minimum that must be preserved.
2. **Bounded Support and Absence of Outliers:** The distribution is bounded, and the maximum absolute value is not an anomaly.

#### C. Formal Computation
The transformation is defined as:

$$x_i' = \frac{x_i}{\max_{j} |x_j| + \epsilon}$$

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Sparsity Preservation:** Because there is no translation step (no subtraction), $0$ maps exactly to $0$. This is useful for naturally sparse spatial-temporal features, such as solar radiation, rainfall, or localized traffic surges.
    *   **Preservation of Vector Signs:** If the vector contains both positive and negative values (e.g., wind velocity components $u$ and $v$, or temperature anomalies), the sign is preserved.

*   **Cons:**
    *   **Highly Sensitive to Outliers:** A single extreme absolute value in the historical series will compress all other values toward $0$.
    *   **Suboptimal Centering for Biased Features:** If a feature is strictly positive and far from zero (e.g., atmospheric pressure around $1013\text{ hPa}$), MaxAbs maps all values to a tight cluster near $1.0$ without centering them, which slows down gradient descent.

---

### 5. Logarithmic Transformation

#### A. General Idea
A logarithmic transform applies a non-linear mapping to compress positive, highly right-skewed data. It converts multiplicative, exponential growth patterns into linear, additive relationships.

#### B. Formal Assumptions
1. **Log-Normal Distribution:** The data is highly right-skewed and strictly positive ($x_i > 0$, or $x_i \geq 0$ with a manual shift).
2. **Multiplicative Dynamics:** The variance of the time series scales proportionally with its mean (heteroscedasticity).

#### C. Formal Computation
For strictly positive vectors:

$$x_i' = \ln(x_i)$$

For vectors containing zero values (log-plus-one):

$$x_i' = \ln(x_i + 1)$$

For vectors with arbitrary negative values, a shift parameter $c$ is introduced such that $\min(\mathbf{x}) + c > 0$:

$$x_i' = \ln(x_i + c)$$

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Stabilization of Variance (Homoscedasticity):** In many physical processes (e.g., wind gusts, rainfall volume, epidemic spread), higher values exhibit higher variance. Log-transformation stabilizes this variance, making model loss functions (like MSE) more stable across different scales.
    *   **Linearizing Exponential Trends:** Many temporal propagation processes are multiplicative. Log-transforms linearize these trends, allowing linear components of neural networks or state-space models to capture them easily.

*   **Cons:**
    *   **Mathematical Undefined Domain:** It cannot handle negative values or zero directly without arbitrary shift factors ($c$), which can introduce bias.
    *   **Prediction Back-Transformation Bias:** Neural networks predict in the transformed space. Applying the inverse transform (exponential function $\exp(\hat{x}_i')$) to generate the physical forecast introduces systematic bias, because:
        $$E[\exp(Y)] \neq \exp(E[Y])$$
        This can lead to underestimation of peak values in the original scale unless mathematical bias-correction factors are applied.

---

### 6. Power Transformations (Box-Cox and Yeo-Johnson)

#### A. General Idea
Power transformations are a family of parametric, non-linear transformations that stabilize variance and transform highly skewed, non-Gaussian distributions into near-normal distributions. The optimal parameter $\lambda$ is determined dynamically via maximum likelihood estimation (MLE).

#### B. Formal Assumptions
1. **Unimodal, Skewed Distribution:** The data can be successfully mapped to a Gaussian distribution via a power parameter $\lambda$.
2. **Box-Cox Specific:** The input vector must be strictly positive ($x_i > 0$).
3. **Yeo-Johnson Specific:** Relaxes the positive constraint, allowing zero and negative values.

#### C. Formal Computation

##### Box-Cox Transformation
Given $\lambda$:

$$\psi(x_i, \lambda) = \begin{cases} \frac{x_i^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(x_i) & \text{if } \lambda = 0 \end{cases}$$

##### Yeo-Johnson Transformation
Allows $x_i \leq 0$:

$$\psi(x_i, \lambda) = \begin{cases} \frac{(x_i + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x_i \geq 0 \\ \ln(x_i + 1) & \text{if } \lambda = 0, x_i \geq 0 \\ -\frac{(-x_i + 1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, x_i < 0 \\ -\ln(-x_i + 1) & \text{if } \lambda = 2, x_i < 0 \end{cases}$$

The parameter $\lambda$ is estimated by maximizing the profile log-likelihood function of the transformed variable under a normal assumption.

#### D. Formal Pros & Cons in Spatial-Temporal Forecasting

*   **Pros:**
    *   **Optimal Mapping to Gaussian Assumptions:** Many regression objectives (e.g., Mean Squared Error) assume Gaussian distributed errors. Power transforms actively force highly non-linear, skewed physical phenomena (like wind speed or precipitation) into a symmetric, bell-shaped distribution, which improves model performance.
    *   **Data-Driven Versatility:** The transform adjusts its parameters dynamically to fit the skew of each unique feature.

*   **Cons:**
    *   **Spatial Inconsistency under Local Fitting:** If $\lambda$ is fitted independently per station, the physical relationship of the transformed values is broken across space. If Station 1 has $\lambda = 0.5$ and Station 2 has $\lambda = 0.2$, a normalized value of `1.5` represents different raw states, which can confuse spatial layers (e.g., GNNs).
    *   **Computational Complexity:** Finding $\lambda$ via MLE adds computational overhead during the data preprocessing pipeline.
    *   **Complex Inverse Transformations:** Projecting multi-step predictions back to the original physical scale requires evaluating complex, non-linear inverse formulas. This can amplify small prediction errors in the transformed space into large, unstable errors in the physical space.