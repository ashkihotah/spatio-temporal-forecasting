# Question 1

So during the training of a RNN-AE the loss to minimize is computed using the normalized target values while validation and test errors are computed on the ground truth recovered real target values ( the ones directly observed from sensors). Am i right? Or is it possible to use normalized target values also for computing regression metrics in validation / testing phases?

## Answer

Your understanding of the training process is highly accurate, but the choice of scale for validation and testing depends on the specific metric you are computing and the purpose of the evaluation. 

Here is a formal breakdown of how targets are handled in each phase, along with a deep mathematical explanation of why evaluating on the raw scale is usually necessary.

---

### 1. Training Phase: Strictly Normalized Scale
During training, the loss function (e.g., MSE or MAE) is computed **strictly in the normalized space**:

$$\mathcal{L} = \frac{1}{M}\sum_{i=1}^M \left( \hat{y}_{\text{norm}, i} - y_{\text{norm}, i} \right)^2$$

* **Why?** If you computed loss on the raw physical scale, pollutants with large magnitudes (like $PM10$ or $PM2.5$, which range up to $800$) would generate massive gradients. These would completely overwhelm the gradients of lower-magnitude or narrow-variance features (like $CO$ or $WSPM$). The optimizer would focus entirely on minimizing the error of the high-magnitude features, leading to extremely poor convergence for the rest of the network.

---

### 2. Validation and Testing Phases: Normalized vs. Real Scale

You can technically compute metrics on both scales, but they serve different purposes. 

#### A. When is it acceptable to use the Normalized Scale?
You can use the normalized target values for computing **Validation Loss** during the training loop (e.g., for early stopping or learning rate decay). 
* **The Reason:** Calculating the validation loss on the normalized scale is computationally fast (no inverse transform needed) and provides a direct, unwarped analog to the training loss. This tells you if the network's optimization process is generalizing well.

#### B. When must you use the Recovered Real Scale (Ground Truth)?
For final **Model Evaluation, Testing, and Reporting** (using metrics like RMSE, MAE, or MAPE), you **should compute the metrics on the original, recovered physical scale**.

There are two primary reasons for this:

##### 1. Physical Interpretability
A normalized MAE of `0.045` or a normalized RMSE of `0.12` has no physical meaning to domain experts or stakeholders. Reporting an RMSE of `12.4 ug/m^3` for $PM2.5$ is physically interpretable and allows direct comparison with existing air quality forecasting models in the literature.

##### 2. Mathematical Distortion from Non-Linear Scalers (The Critical Reason)
If you used a non-linear scaler in Step 1 of your pipeline—such as the log-plus-one transform ($\ln(x+1)$)—the error distances in the normalized space **do not scale linearly** to error distances in the physical space. 

Let's look at this mathematically. Suppose we have a model predicting $PM2.5$ using the log-plus-one transform. We compare two prediction scenarios:

* **Scenario A (Low Concentrations):**
  * Ground Truth ($x$): $10\ \mu\text{g/m}^3$ $\rightarrow$ Normalized ($y$): $\ln(11) \approx 2.398$
  * Predicted ($\hat{x}$): $20\ \mu\text{g/m}^3$ $\rightarrow$ Normalized ($\hat{y}$): $\ln(21) \approx 3.045$
  * **Normalized Absolute Error:** $|y - \hat{y}| = 0.647$
  * **Physical Absolute Error:** $|x - \hat{x}| = \mathbf{10\ \mu\text{g/m}^3}$

* **Scenario B (High Concentrations - Extreme Smog Event):**
  * Ground Truth ($x$): $400\ \mu\text{g/m}^3$ $\rightarrow$ Normalized ($y$): $\ln(401) \approx 5.994$
  * Predicted ($\hat{x}$): $724\ \mu\text{g/m}^3$ $\rightarrow$ Normalized ($\hat{y}$): $\ln(725) \approx 6.586$
  * **Normalized Absolute Error:** $|y - \hat{y}| = 0.592$
  * **Physical Absolute Error:** $|x - \hat{x}| = \mathbf{324\ \mu\text{g/m}^3}$

#### The Implication:
If you compute your test metrics (like MAE) on the **normalized scale**, Scenario B looks *better* (error of $0.592$) than Scenario A (error of $0.647$). 
However, in the **physical world**, the prediction error in Scenario B is an unacceptable $324\ \mu\text{g/m}^3$, while Scenario A is off by only $10\ \mu\text{g/m}^3$. 

Evaluating on the normalized scale would heavily mask massive prediction errors occurring at high values, leading to a highly overconfident and physically unreliable evaluation.

---

### Summary of Best Practices

| Phase | Metric Type | Scale to Use | Reason |
| :--- | :--- | :--- | :--- |
| **Training** | Training Loss (MSE/MAE) | **Normalized** | Essential for stable gradient updates and equitable feature weighting. |
| **Validation** | Validation Loss / Early Stopping | **Normalized** | Computationally efficient; directly tracks optimization convergence. |
| **Testing** | Final Regression Metrics (RMSE/MAE/MAPE) | **Physical (Inverse Transformed)** | Ensures physical interpretability and prevents non-linear mathematical distortion of errors. |

# Question 2

Returning to the discussion of when we should compute losses in normalized space or in ground truth spaces, i have two questions:
1. if we must compute testing losses on the raw ground truth value space, then can the loss explode and overflow due to a large dataset and large values of singular losses? If yes how can we mitigate this issue?
2. you made me an example of when evaluating a loss computed on normalized space for pm2.5 can be misleading with respect comparing it in raw ground truth value space. However you did this example with log normalizer. Is this discussion the same for every scling method we use e.g. also for min max scalers? If yes make me an example with min max scalers

## Answer

Below is a detailed, mathematically rigorous analysis of your two questions regarding the calculation of test losses on raw vs. normalized scales.

---

### Question 1: Numerical Explosion and Overflow in Raw Value Space

If you compute testing losses (specifically **Mean Squared Error (MSE)**) on the raw physical scale of large datasets, **numerical issues can definitely occur.** 

While a 32-bit single-precision float (`float32`) can represent numbers up to $\approx 3.4 \times 10^{38}$ (meaning it is highly unlikely to suffer a hard mathematical overflow), you face a major risk of **floating-point precision loss** and **catastrophic cancellation**.

#### The Mathematical Problem
When evaluating a large dataset (e.g., $N = 100,000$ sequence steps across $12$ stations), you accumulate a running sum of squared errors:

$$\text{Sum of Squared Errors (SSE)} = \sum_{i=1}^N (y_{\text{raw}, i} - \hat{y}_{\text{raw}, i})^2$$

If a model makes a prediction error of $400\ \mu\text{g/m}^3$ during an extreme smog event, the squared error for that single step is:

$$(400)^2 = 160,000$$

Across millions of spatial-temporal evaluation points, the running sum can easily reach $10^{10}$ or $10^{11}$. 
* In `float32`, numbers are represented with only **24 bits of precision** (about 7 decimal digits of accuracy). 
* When you add a small squared error (e.g., $1.2$) to a very large running sum (e.g., $15,000,000.0$), the lower bits of the small number are truncated and lost entirely. This is called **underflow during addition** or **precision loss**. Over a large dataset, your final evaluation metric will become highly inaccurate.

#### Mitigation Strategies

To prevent this in production pipelines, apply these three techniques:

1.  **Upcast to Double Precision (`float64`) During Evaluation:**
    Keep your model weights in `float32` for fast training, but explicitly upcast your prediction and target tensors to `float64` inside your evaluation script before computing metrics. `float64` provides 53 bits of precision (15–17 decimal digits), eliminating truncation errors.
    ```python
    # Convert to float64 before math operations
    y_true_phys = tf.cast(y_true_phys, tf.float64)
    y_pred_phys = tf.cast(y_pred_phys, tf.float64)
    ```
2.  **Evaluate using MAE and RMSE instead of MSE:**
    *   **Mean Absolute Error (MAE):** Because errors are not squared ($|y - \hat{y}|$), the running sum remains small and linearly scaled, reducing the risk of truncation.
    *   **Root Mean Squared Error (RMSE):** RMSE brings the final metric back to the physical scale ($\mu\text{g/m}^3$), making it highly interpretable.
3.  **Welford’s Algorithm for Running Metrics:**
    Instead of summing all errors and dividing by $N$ at the end, use Welford's algorithm or batch-wise incremental averaging. This computes the running mean of the loss dynamically, ensuring the values being added are always of similar magnitudes.

---

### Question 2: Is Min-Max Scaling Misleading?

For a **single feature using a global scaler**, a linear transformation like Min-Max preserves the relative rankings of errors. If Error A is larger than Error B on the raw scale, it is guaranteed to be larger on the normalized scale.

However, **Min-Max scaling is highly misleading when comparing across different entities (local scaling) or across different features (multi-variable scaling).** 

Below is a concrete mathematical example demonstrating how local Min-Max scaling can severely mask critical model failures.

#### The Setup: Local Min-Max Scaling on Two Stations
Suppose we deploy our model to two different monitoring stations in Beijing:
1.  **Station A (Urban/Industrial):** Highly polluted. 
    *   $PM2.5$ historical range: $[0,\ 800]\ \mu\text{g/m}^3 \implies (\max - \min) = 800$
2.  **Station B (Suburban/Mountainous):** Clean.
    *   $PM2.5$ historical range: $[0,\ 50]\ \mu\text{g/m}^3 \implies (\max - \min) = 50$

The local Min-Max formula is:

$$y_{\text{norm}} = \frac{y_{\text{raw}}}{\max - \min}$$

#### The Evaluation Mismatch

Let's evaluate the model’s prediction errors on a specific day for both stations:

| Metric | Station A (Urban) | Station B (Suburban) |
| :--- | :--- | :--- |
| **Ground Truth Raw ($y_{\text{raw}}$)** | $400.0\ \mu\text{g/m}^3$ | $10.0\ \mu\text{g/m}^3$ |
| **Predicted Raw ($\hat{y}_{\text{raw}}$)** | $480.0\ \mu\text{g/m}^3$ | $15.0\ \mu\text{g/m}^3$ |
| **Normalized Ground Truth ($y_{\text{norm}}$)** | $\frac{400}{800} = \mathbf{0.5}$ | $\frac{10}{50} = \mathbf{0.2}$ |
| **Normalized Prediction ($\hat{y}_{\text{norm}}$)** | $\frac{480}{800} = \mathbf{0.6}$ | $\frac{15}{50} = \mathbf{0.3}$ |
| **Normalized Absolute Error** | $\|0.5 - 0.6\| = \mathbf{0.1} $ | $ \|0.2 - 0.3\| = \mathbf{0.1}$ |
| **Physical Absolute Error** | $\|400 - 480\| = \mathbf{80.0\ \mu\text{g/m}^3}$ | $\|10 - 15\| = \mathbf{5.0\ \mu\text{g/m}^3}$ |

#### The Implication
If you evaluate your validation metrics using **normalized Min-Max space**:
* The model appears to perform **exactly the same** at both stations, as both exhibit a normalized absolute error of **$0.1$**.

If you evaluate on the **raw physical space**:
* **Station B's error is $5\ \mu\text{g/m}^3$:** This is physically negligible and well within sensor noise tolerance.
* **Station A's error is $80\ \mu\text{g/m}^3$:** This is a massive, hazardous prediction failure. An error of $80\ \mu\text{g/m}^3$ can easily misclassify the air quality index (AQI) from "Moderate" to "Unhealthy," leading to a failure to trigger city-wide smog alerts.

### Summary
Evaluating validation or test performance in normalized space (whether Logarithmic, Standardized, or Min-Max) **homogenizes the error scales**. While this homogeneity is beneficial for training gradients, it hides the true physical consequences of prediction errors in the real world. Evaluation must always occur on the physical scale.

# Question 3

So the loss computed on normalized predictions and ground truths cannot either be used to interpret how much good is the model right? 

# Answer

**Correct. You cannot use the loss computed on normalized data to interpret the physical, real-world quality of the model.** 

An air quality manager, a policy maker, or a software engineer cannot look at a normalized Mean Squared Error of `0.025` and understand whether the model is safe to deploy. 

To formalize this, we must distinguish between **Absolute Interpretability** (which normalized loss lacks) and **Relative Interpretability** (which normalized loss provides).

---

### 1. Why Normalized Loss Lacks Absolute Interpretability

#### A. The Loss of Units
Physical decisions require physical units. 
* A normalized loss of `0.025` has **no units**. 
* It does not tell you if the predictions are off by $\pm 5\ \mu\text{g/m}^3$ (highly accurate) or $\pm 50\ \mu\text{g/m}^3$ (poor accuracy).

#### B. The Distortion of Error Variance
Because different features have different scales, a normalized loss treats errors across features as mathematically equivalent, even when their physical impacts are radically different. 
* A normalized MAE of `0.05` for $PM2.5$ might translate to a physical error of $40\ \mu\text{g/m}^3$.
* A normalized MAE of `0.05` for $CO$ might translate to a physical error of $0.5\ \text{mg/m}^3$.
* Averaging these normalized losses together into a single "global model loss" creates an abstract number that has no physical or chemical meaning.

---

### 2. What Normalized Loss *Is* Useful For (Relative Interpretability)

While normalized loss cannot be used for physical interpretation, it is highly useful for **relative performance tracking** and mathematical diagnostics during the training phase:

#### A. Model Benchmarking
If you are comparing two different model architectures (e.g., an LSTM vs. a Transformer) trained on the **exact same dataset with the exact same scalers**:
* If Model A achieves a normalized validation loss of `0.015` and Model B achieves `0.030`, you can interpret that **Model A is better optimized and has learned the data patterns more effectively than Model B**.

#### B. Overfitting and Convergence Diagnostics
You use the normalized loss curves to monitor the training dynamics:
* If the normalized training loss continues to decrease but the normalized validation loss begins to rise, you know the model is **overfitting**.
* If the normalized loss curve flattens out, you know the model has **converged**.

---

### MLOps Best Practice: Separation of Optimization and Evaluation

In professional machine learning engineering, we manage this by strictly separating our metrics into two classes:

1.  **Optimization Loss (Normalized Space):**
    *   *Target Audience:* The Optimizer (Adam/SGD).
    *   *Purpose:* Guides gradient descent with stable, well-scaled mathematical boundaries.
2.  **Business Metrics / KPIs (Physical Space):**
    *   *Target Audience:* Humans (Engineers, Domain Experts, Stakeholders).
    *   *Purpose:* Evaluates real-world safety, accuracy, and deployment readiness (measured in physical units like $\text{ug/m}^3$ or $^\circ\text{C}$).