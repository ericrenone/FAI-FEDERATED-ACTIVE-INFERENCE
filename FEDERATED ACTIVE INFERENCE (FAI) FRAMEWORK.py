"""
UNIFIED FEDERATED ACTIVE INFERENCE (FAI) FRAMEWORK
--------------------------------------------------
Combines:
1. Variational Model Selection (Simple vs. Complex Agents)
2. Scalable Federated Learning (1,000 Nodes)
3. Integrated Real-time Analytics (Matplotlib Dashboard)
"""

import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Math Utilities ---
def softmax(x):
    max_x = max(x)
    e_x = [math.exp(i - max_x) for i in x]
    sum_e = sum(e_x)
    return [i / sum_e for i in e_x]

def kl_divergence(p, q):
    """Compute KL divergence between two distributions."""
    return sum(p[i] * math.log(p[i] / (q[i] + 1e-10) + 1e-10) for i in range(len(p)) if p[i] > 0)

# --- Core Simulation Classes ---
class ActiveInferenceAgent:
    """Agent minimizing Free Energy with the ability to switch generative models."""
    def __init__(self, agent_id, model_type='complex'):
        self.agent_id = agent_id
        self.model_type = model_type
        # Complex models start with random priors; simple models are peaked
        self.prior = normalize([random.random() for _ in range(3)]) if model_type == 'complex' else [0.1, 0.8, 0.1]
        self.fe = 0.0
        self.qs = [1/3] * 3

    def infer(self, obs_idx, A):
        """Update beliefs q(s) using current model parameters."""
        likelihood = [A[obs_idx][s] for s in range(3)]
        log_qs = [math.log(self.prior[s] + 1e-10) + math.log(likelihood[s] + 1e-10) for s in range(3)]
        self.qs = softmax(log_qs)
        
        accuracy = -sum(self.qs[s] * math.log(likelihood[s] + 1e-10) for s in range(3))
        complexity = kl_divergence(self.qs, self.prior)
        self.fe = complexity + accuracy
        return self.fe

    def try_switch_model(self, A, obs_idx):
        """Bayesian Model Reduction: Switch to simple model if it reduces local Free Energy."""
        if self.model_type == 'complex':
            old_fe = self.fe
            # Test simple hypothesis
            test_prior = [0.1, 0.8, 0.1]
            likelihood = [A[obs_idx][s] for s in range(3)]
            log_qs = [math.log(test_prior[s] + 1e-10) + math.log(likelihood[s] + 1e-10) for s in range(3)]
            test_qs = softmax(log_qs)
            test_fe = kl_divergence(test_qs, test_prior) - sum(test_qs[s] * math.log(likelihood[s] + 1e-10) for s in range(3))
            
            if test_fe < old_fe - 0.05: # Minimal evidence threshold
                self.model_type = 'simple'
                self.prior = test_prior
                return True
        return False

def normalize(vec):
    s = sum(vec)
    return [v/s for v in vec]

class FederatedNode:
    """Local node optimizing the Likelihood Matrix (A) via gradient descent."""
    def __init__(self):
        self.w = [random.uniform(-0.1, 0.1) for _ in range(9)]
        
    def get_A(self):
        """Reshapes weights into a valid 3x3 Likelihood Matrix (Observations x States)."""
        matrix = [softmax(self.w[i:i+3]) for i in range(0, 9, 3)]
        return list(map(list, zip(*matrix)))

    def compute_gradient(self, agent, obs, eps=1e-3):
        """Finite difference gradient of Free Energy w.r.t model weights."""
        original_w = self.w[:]
        grads = []
        
        def eval_fe(weights):
            self.w = weights
            A = self.get_A()
            return agent.infer(obs, A)

        base_fe = eval_fe(original_w)
        for i in range(9):
            temp_w = original_w[:]
            temp_w[i] += eps
            grads.append((eval_fe(temp_w) - base_fe) / eps)
        
        self.w = original_w
        return grads

# --- Global Simulation Parameters ---
NUM_NODES = 1000
NUM_ROUNDS = 100
nodes = [FederatedNode() for _ in range(NUM_NODES)]
agents = [ActiveInferenceAgent(i, 'complex' if i % 4 != 0 else 'simple') for i in range(NUM_NODES)]
global_w = [random.random() for _ in range(9)]

# Visual Metrics Storage
history = {'fe': [], 'drift': [], 'simple_ratio': []}

# --- Dashboard Setup ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)

def animate(round_idx):
    global global_w
    if round_idx >= NUM_ROUNDS: return

    node_updates = []
    total_fe = 0
    switches = 0
    lr = 0.05
    
    # Environment Truth (State mapping bias)
    observations = [random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0] for _ in range(NUM_NODES)]

    for i in range(NUM_NODES):
        nodes[i].w = global_w[:] # Synchronize
        
        # 1. Active Inference & Learning
        A = nodes[i].get_A()
        total_fe += agents[i].infer(observations[i], A)
        
        # 2. Gradient Computation
        grad = nodes[i].compute_gradient(agents[i], observations[i])
        node_updates.append([nodes[i].w[j] - lr * grad[j] for j in range(9)])
        
        # 3. Model Selection (Bayesian Model Reduction)
        if agents[i].try_switch_model(A, observations[i]):
            switches += 1

    # 4. Federated Aggregation (FedAvg)
    prev_w = global_w[:]
    global_w = [sum(u[j] for u in node_updates) / NUM_NODES for j in range(9)]
    
    # Record Metrics
    drift = math.sqrt(sum((global_w[j] - prev_w[j])**2 for j in range(9)))
    simple_ratio = sum(1 for a in agents if a.model_type == 'simple') / NUM_NODES
    
    history['fe'].append(total_fe / NUM_NODES)
    history['drift'].append(drift)
    history['simple_ratio'].append(simple_ratio)

    # Refresh Visuals
    ax1.clear(); ax1.plot(history['fe'], color='#1f77b4', lw=2)
    ax1.set_title(f"Population Average Free Energy (Surprise Reduction)"); ax1.set_ylabel("F")
    
    ax2.clear(); ax2.plot(history['drift'], color='#d62728', lw=2)
    ax2.set_title("Global Model Convergence (Weight Drift)"); ax2.set_ylabel("Delta W")
    
    ax3.clear(); ax3.stackplot(range(len(history['simple_ratio'])), history['simple_ratio'], color='#2ca02c', alpha=0.3)
    ax3.plot(history['simple_ratio'], color='#2ca02c', lw=2)
    ax3.set_title("Simple Model Adoption Rate (Structural Learning)"); ax3.set_ylim(0, 1)

    print(f"Round {round_idx:2d} | FE: {history['fe'][-1]:.4f} | Simple Model Ratio: {simple_ratio:.1%}")

ani = FuncAnimation(fig, animate, frames=NUM_ROUNDS, repeat=False, interval=50)
print("Launching Federated Active Inference Dashboard...")
plt.show()