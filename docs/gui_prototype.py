"""
MEC Offloading GUI Prototype - Matplotlib Version
This generates a static reference image showing what the GUI should look like.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'MEC Task Offloading Visualization', 
        fontsize=20, fontweight='bold', ha='center')
ax.text(7, 9.1, 'Episode 12,340 / 20,000  |  Slot 7 / 10', 
        fontsize=12, ha='center', color='gray')

# ═══════════════════════════════════════════════════
# Mobile Devices (Left Side)
# ═══════════════════════════════════════════════════
ax.text(2.5, 8.5, 'Mobile Devices', fontsize=14, fontweight='bold', ha='center')

devices = [
    {'name': 'MD0', 'size': '4.2 Mb', 'action': 'LOCAL', 'color': '#90EE90'},
    {'name': 'MD1', 'size': '3.5 Mb', 'action': 'ES2', 'color': '#FFB6C1'},
    {'name': 'MD2', 'size': '3.0 Mb', 'action': 'ES2', 'color': '#FFB6C1'},
]

for i, dev in enumerate(devices):
    y = 7.2 - i * 1.5
    # Device box
    box = FancyBboxPatch((0.5, y-0.4), 2.2, 1.0, 
                         boxstyle="round,pad=0.1", 
                         facecolor=dev['color'], edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(1.6, y+0.2, dev['name'], fontsize=11, fontweight='bold', ha='center')
    ax.text(1.6, y-0.1, dev['size'], fontsize=9, ha='center')
    ax.text(1.6, y-0.3, f'[{dev["action"]}]', fontsize=8, ha='center', 
            color='darkgreen' if dev['action'] == 'LOCAL' else 'darkred')

# ═══════════════════════════════════════════════════
# Edge Servers (Right Side)
# ═══════════════════════════════════════════════════
ax.text(10.5, 8.5, 'Edge Servers', fontsize=14, fontweight='bold', ha='center')

servers = [
    {'name': 'ES1', 'cpu': '6 GHz', 'queue': 2, 'load': 40, 'color': '#87CEEB'},
    {'name': 'ES2', 'cpu': '12 GHz', 'queue': 1, 'load': 30, 'color': '#87CEEB'},
    {'name': 'ES3', 'cpu': '12 GHz', 'queue': 0, 'load': 20, 'color': '#87CEEB'},
]

for i, srv in enumerate(servers):
    y = 7.2 - i * 1.5
    # Server box
    box = FancyBboxPatch((9.0, y-0.4), 2.2, 1.0, 
                         boxstyle="round,pad=0.1", 
                         facecolor=srv['color'], edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(10.1, y+0.2, srv['name'], fontsize=11, fontweight='bold', ha='center')
    ax.text(10.1, y-0.05, f'CPU: {srv["cpu"]}', fontsize=8, ha='center')
    ax.text(10.1, y-0.25, f'Queue: {srv["queue"]}', fontsize=8, ha='center')
    
    # Load bar
    bar_width = 1.8 * (srv['load'] / 100)
    ax.barh(y-0.45, bar_width, height=0.08, left=9.2, color='green', alpha=0.7)
    ax.text(10.1, y-0.45, f'Load: {srv["load"]}%', fontsize=7, ha='center', va='center')

# ═══════════════════════════════════════════════════
# Arrows (Offloading Decisions)
# ═══════════════════════════════════════════════════
# MD0 -> LOCAL (no arrow, just text)
ax.text(3.5, 7.2, 'Local Execution', fontsize=9, style='italic', color='green')

# MD1 -> ES2
arrow1 = FancyArrowPatch((2.7, 5.7), (9.0, 5.7),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=3, color='red', alpha=0.7)
ax.add_patch(arrow1)

# MD2 -> ES2
arrow2 = FancyArrowPatch((2.7, 4.2), (9.0, 4.2),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=3, color='red', alpha=0.7)
ax.add_patch(arrow2)

# ═══════════════════════════════════════════════════
# Real-time Metrics (Bottom)
# ═══════════════════════════════════════════════════
ax.text(7, 2.8, 'Real-time Metrics', fontsize=14, fontweight='bold', ha='center')

# Metric boxes
metrics = [
    {'label': 'Avg Cost', 'value': '0.412', 'color': '#FFD700'},
    {'label': 'Completion', 'value': '98.6%', 'color': '#98FB98'},
    {'label': 'Avg Latency', 'value': '0.38s', 'color': '#DDA0DD'},
]

for i, met in enumerate(metrics):
    x = 2.5 + i * 3.5
    box = FancyBboxPatch((x-1.0, 1.8), 2.0, 0.7, 
                         boxstyle="round,pad=0.05", 
                         facecolor=met['color'], edgecolor='black', alpha=0.8)
    ax.add_patch(box)
    ax.text(x, 2.25, met['label'], fontsize=9, ha='center', fontweight='bold')
    ax.text(x, 2.0, met['value'], fontsize=12, ha='center', fontweight='bold')

# ═══════════════════════════════════════════════════
# Cost Curve (Bottom)
# ═══════════════════════════════════════════════════
ax.text(7, 1.4, 'Training Cost Curve (Last 100 Episodes)', 
        fontsize=11, ha='center', style='italic')

# Generate fake curve
x_curve = np.linspace(2, 12, 100)
y_curve = 1.5 * np.exp(-x_curve/3) + 0.4 + np.random.normal(0, 0.02, 100)
y_curve = np.convolve(y_curve, np.ones(5)/5, mode='same')  # Smooth

ax.plot(x_curve, y_curve, 'b-', linewidth=2, alpha=0.7)
ax.fill_between(x_curve, y_curve, alpha=0.2)
ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Target')
ax.set_xlim(2, 12)
ax.set_ylim(0.3, 1.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Cost')
ax.legend()

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:/MARL-IPPOAndMore/docs/gui_prototype.png', dpi=150, bbox_inches='tight')
print("GUI prototype saved to docs/gui_prototype.png")
plt.show()
