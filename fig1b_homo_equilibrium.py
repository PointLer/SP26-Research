# Figure 2: Stability region of neutral equilibrium in homogeneous symmetric model (b, mu)
import numpy as np
import matplotlib.pyplot as plt

# b axis range [-3, 0]
b_range = np.linspace(-3, 0, 100)
mu_range = np.linspace(0.01, 0.5, 100)
B, MU = np.meshgrid(b_range, mu_range)

Z = B * (0.5)**B - 2 + 2/MU

fig, ax = plt.subplots(figsize=(8, 7))

# Two solid colors
stable_color   = '#B0C4DE'
unstable_color = '#E8B4B8'

ax.contourf(B, MU, Z, levels=[Z.min(), 0, Z.max()],
            colors=[unstable_color, stable_color])

# Zero contour line
ax.contour(B, MU, Z, levels=[0], colors='black', linewidths=2)

# Stable Region / Unstable Region labels
ax.text(-1.0, 0.2, 'Stable Region', fontsize=26, color='#375093',
        ha='center', va='center', fontfamily='Times New Roman',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax.text(-2.1, 0.35, 'Unstable Region', fontsize=26, color='#831A21',
        ha='center', va='center', fontfamily='Times New Roman',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

ax.set_xlabel(r'Bias parameter $b$', fontsize=22, fontfamily='Times New Roman')
ax.set_ylabel(r'Convergence parameter $\mu$', fontsize=22, fontfamily='Times New Roman')

# Tick spacing 0.5 for b-axis
ax.set_xticks(np.arange(-3.0, 0.01, 0.5))
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')

plt.tight_layout()
plt.savefig('../Images/homo_equilibrium.png', dpi=200, bbox_inches='tight')
plt.show()
