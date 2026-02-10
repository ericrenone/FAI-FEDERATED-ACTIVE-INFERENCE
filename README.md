# Federated-Agentic-Inference-Framework

## Core

This framework moves beyond standard gradient descent, treating learning as a **geodesic path** on a statistical manifold.
1.  **Active Inference:** Agents don't just "learn"; they minimize **Free Energy ($F$)**—a trade-off between accuracy (likelihood) and complexity (divergence).
2.  **Fisher-Shannon Geodesics:** Learning is optimized by following the natural gradient of the **Fisher Information Metric**, ensuring stable convergence in probability space.
3.  **Bayesian Model Reduction:** Complex agents dynamically switch to simpler generative models if it reduces local surprise, mimicking biological energy efficiency.

---

## 🛠 Technical Architecture

### 1. Variational Free Energy (FAI)
Each agent operates as a self-organizing system. The objective is to minimize the **Action Integral** of Free Energy:

$$F = D_{KL}[q(s) || p(s)] - E_{q(s)}[\ln p(o|s)]$$

* **Complexity:** Divergence between current beliefs $q(s)$ and priors $p(s)$.
* **Accuracy:** Expected log-likelihood of observations given the hidden state.



### 2. Information Geometry & Fisher Metric
The learner treats the parameter space as a **Riemannian Manifold**. 
* **Fisher Information Matrix ($G$):** Quantifies the curvature of the manifold.
* **Natural Gradient:** Updates follow $\theta_{t+1} = \theta_t - \eta G^{-1} \nabla L$, ensuring the update step is aware of the statistical geometry, preventing catastrophic "drift" in decentralized settings.



### 3. Federated Aggregation (FedAvg-KL)
To synchronize 1,000+ nodes, the framework employs **KL-Optimal Aggregation**, minimizing the global divergence across the population:


---

## 📊 Real-Time Analytics Dashboard

The framework includes a high-speed telemetry suite (Matplotlib & ASCII) to track the "thermodynamics" of the network:

| Metric | Purpose | System Signal |
| :--- | :--- | :--- |
| **Average Free Energy** | Measures collective "surprise." | Downward trend = Convergence. |
| **Fisher Information** | Measures model "peakiness" or certainty. | Upward trend = Structural learning. |
| **System Entropy ($H$)** | Measures statistical disorder in the swarm. | Downward trend = Consensus. |
| **Relational Abstraction**| Measures the transition from data to concepts. | High ratio = Efficient compression. |



