# Figure 1: Stability region of intermediate equilibria in asymmetric model (b,c)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import warnings
warnings.filterwarnings('ignore')

def find_a_satisfying_constraint(b, c):
    def equation(a):
        if 0 < a < 1:
            return b * np.log(a) - c * np.log(1-a)
        else:
            return np.inf
    try:
        sol = root_scalar(equation, bracket=[0.001, 0.999], method='brentq')
        return sol.root if sol.converged else np.nan
    except:
        return np.nan

def evaluate_function(b, c, a, mu):
    term1 = 2*(1-mu)**2
    term2 = mu*(1-mu)*a**b*(2+b*(1-a)+c*a)
    term3 = mu**2*a**(2*b)*((1-a)*b+c*a)
    return term1 + term2 + term3

mu = 0.5

b_range = np.linspace(-2, 0, 100)
c_range = np.linspace(-2, 0, 100)
B, C = np.meshgrid(b_range, c_range)
Z = np.zeros_like(B)

for i in range(len(b_range)):
    for j in range(len(c_range)):
        b, c = b_range[i], c_range[j]
        a = find_a_satisfying_constraint(b, c)
        if not np.isnan(a):
            Z[j, i] = evaluate_function(b, c, a, mu)
        else:
            Z[j, i] = 1

fig, ax = plt.figure(figsize=(8, 7)), plt.gca()

# Two solid colors — light blue (stable) and light red (unstable)
stable_color   = '#B0C4DE'
unstable_color = '#E8B4B8'

ax.contourf(B, C, Z, levels=[Z.min(), 0, Z.max()],
            colors=[unstable_color, stable_color])

# Zero contour line
ax.contour(B, C, Z, levels=[0], colors='black', linewidths=2)

# (-1, -1) point marker only (no yellow highlight box)
ax.plot(-1, -1, 'o', markersize=10, markeredgecolor='black',
        markerfacecolor='#F1F77D', markeredgewidth=2)
ax.text(-0.9, -0.92, r'$(-1,-1)$', ha='center', fontsize=16,
        fontfamily='Times New Roman')

# Stable Region / Unstable Region labels
ax.text(-0.5, -0.5, 'Stable Region', fontsize=26, color='#375093',
        ha='center', va='center', fontfamily='Times New Roman',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax.text(-1.5, -1.5, 'Unstable Region', fontsize=26, color='#831A21',
        ha='center', va='center', fontfamily='Times New Roman',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

ax.set_xlabel(r'Bias parameter $b$', fontsize=22, fontfamily='Times New Roman')
ax.set_ylabel(r'Bias parameter $c$', fontsize=22, fontfamily='Times New Roman')

# Tick spacing 0.5
ax.set_xticks(np.arange(-2.0, 0.01, 0.5))
ax.set_yticks(np.arange(-2.0, 0.01, 0.5))
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')

plt.tight_layout()
plt.savefig('../Images/hetero_equilibrium.png', dpi=200, bbox_inches='tight')
plt.show()
