#!/usr/bin/env python3


import math
import random
import tkinter as tk

# -------------------------
# CONFIG / PARAMETERS
# -------------------------
SEED = 42
random.seed(SEED)

N_STRUCT = 17
N_RES = 33
T_STEPS = 100
ALPHA = 0.1
BETA = 0.1
ATTENTION_BUDGET = 5.0
LAMBDA = 0.05

# -------------------------
# INITIALIZATION
# -------------------------
rho = 1.0
phi = [random.uniform(0.5, 1.5) for _ in range(N_STRUCT)]
psi = [random.uniform(0.0, 1.0) for _ in range(N_RES)]

history = []

# -------------------------
# INFORMATION GEOMETRY METRIC
# -------------------------
def fisher_metric(theta):
    return [[1.0 if i==j else 0.0 for j in range(len(theta))] for i in range(len(theta))]

def dot_vec(a,b):
    return sum(x*y for x,y in zip(a,b))

def norm_metric(v,g):
    return math.sqrt(sum(v[i]*g[i][j]*v[j] for i in range(len(v)) for j in range(len(v))))

def kl_approx(theta_new, theta, g):
    delta = [theta_new[i]-theta[i] for i in range(len(theta))]
    return 0.5*sum(delta[i]*g[i][j]*delta[j] for i in range(len(theta)) for j in range(len(theta)))

# -------------------------
# DISTRIBUTION OPERATOR
# -------------------------
def optimize_resonance(rho, phi, psi, g, C, lambda_scale):
    psi_new = psi[:]
    for k in range(len(psi)):
        best_val = -float('inf')
        best_candidate = psi[k]
        for delta in [-0.05, 0, 0.05]:
            candidate = max(0.0, min(1.0, psi[k]+delta))
            trial = psi_new[:]; trial[k]=candidate
            utility = ALPHA*dot_vec(phi,phi) - LAMBDA*kl_approx(trial, psi, g)
            if utility>best_val:
                best_val=utility
                best_candidate=candidate
        psi_new[k]=best_candidate
    # enforce attention budget
    ri_cost = kl_approx(psi_new, psi, g)
    if ri_cost>C:
        scale = C/(ri_cost+1e-12)
        psi_new = [psi[i]+(psi_new[i]-psi[i])*scale for i in range(len(psi))]
    return psi_new

# -------------------------
# GUI SETUP FOR REAL-TIME PLOTTING
# -------------------------
WIDTH, HEIGHT = 800, 400
MARGIN = 50

root = tk.Tk()
root.title("Canonical REF Simulation Metrics")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

metric_keys = ['rho','resonance_norm','brittleness','volatility']
colors = {'rho':'red','resonance_norm':'blue','brittleness':'orange','volatility':'purple'}

def draw_axes():
    canvas.delete("all")
    canvas.create_line(MARGIN, HEIGHT-MARGIN, WIDTH-MARGIN, HEIGHT-MARGIN, width=2)
    canvas.create_line(MARGIN, HEIGHT-MARGIN, MARGIN, MARGIN, width=2)
    canvas.create_text(WIDTH/2, HEIGHT-20, text="Time step")
    canvas.create_text(20, HEIGHT/2, text="Metric", angle=90)

def draw_lines():
    if not history: return
    max_val = max(max(entry[k] for k in metric_keys) for entry in history)*1.1
    min_val = 0
    scale_x = (WIDTH-2*MARGIN)/T_STEPS
    scale_y = (HEIGHT-2*MARGIN)/(max_val-min_val+1e-12)
    for key in metric_keys:
        points=[]
        for t, entry in enumerate(history):
            x = MARGIN + t*scale_x
            y = HEIGHT-MARGIN - (entry[key]-min_val)*scale_y
            points.append((x,y))
        for i in range(len(points)-1):
            canvas.create_line(points[i][0],points[i][1],points[i+1][0],points[i+1][1],fill=colors[key],width=2)
    # legend
    for idx,(key,color) in enumerate(colors.items()):
        canvas.create_rectangle(WIDTH-150, MARGIN+idx*20, WIDTH-140, MARGIN+idx*20+10, fill=color)
        canvas.create_text(WIDTH-130, MARGIN+idx*20+5, text=key, anchor='w')

# -------------------------
# SIMULATION LOOP
# -------------------------
def update_simulation(t):
    global rho, phi, psi
    if t>=T_STEPS:
        print_summary()
        return
    # accumulation
    rho += ALPHA*sum(phi)
    phi = [max(0.0, p+ALPHA*(random.uniform(-0.05,0.05))) for p in phi]
    # distribution
    g = fisher_metric(psi)
    psi = optimize_resonance(rho, phi, psi, g, ATTENTION_BUDGET, LAMBDA)
    # metrics
    resonance_norm = norm_metric(psi,g)
    brittleness = max(phi)-min(phi)
    volatility = max(psi)-min(psi)
    history.append({'rho':rho,'resonance_norm':resonance_norm,
                    'brittleness':brittleness,'volatility':volatility})
    draw_axes()
    draw_lines()
    root.after(100, lambda:update_simulation(t+1)) # 100ms per step

# -------------------------
# SUMMARY PRINTOUT
# -------------------------
def print_summary():
    final = history[-1]
    print("\nSimulation Complete")
    print("Final Metrics:")
    for k in metric_keys:
        values = [h[k] for h in history]
        print(f"{k:15}: final={final[k]:.3f}, min={min(values):.3f}, max={max(values):.3f}, mean={sum(values)/len(values):.3f}")
    # final system matrix
    system_matrix = [[phi[i]*psi[j] for j in range(N_RES)] for i in range(N_STRUCT)]
    print("\nFinal 17x33 System Matrix (first 3x3 entries):")
    for row in system_matrix[:3]:
        print([round(x,3) for x in row[:3]])

# -------------------------
# START SIMULATION
# -------------------------
draw_axes()
update_simulation(0)
root.mainloop()
