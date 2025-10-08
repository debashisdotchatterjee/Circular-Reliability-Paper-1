# Python code to generate professional circular plots requested by the user.
# - A circular histogram (rose diagram) of failure angles θ on a polar axis
# - A circular reliability curve R(θ) on a polar axis
#
# Notes:
# * Uses matplotlib (no seaborn).
# * Each chart is its own figure.
# * The plots are saved to /mnt/data/Figures/ as high-resolution PNG and PDF.
# * If you have your own angles array, replace the synthetic generator block with real data.
#
# -----------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.special import i0  # modified Bessel function of the first kind, order 0

# -----------------------------
# 1) Prepare output folder
# -----------------------------
out_dir = "/mnt/data/Figures"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# 2) Example data (replace with your own θ in [0, 2π) as needed)
#    Here we simulate a unimodal von Mises-like cluster with a few outliers
# -----------------------------
rng = np.random.default_rng(42)
n = 320
mu_true = 1.25 * np.pi      # mean direction
kappa_true = 3.2            # concentration
# von Mises sampling via numpy
thetas = rng.vonmises(mu=mu_true, kappa=kappa_true, size=n) % (2*np.pi)
# add a few dispersed outliers
outliers = rng.uniform(0, 2*np.pi, size=12)
thetas = np.concatenate([thetas, outliers]) % (2*np.pi)

# If you already have angles, comment the generator above and uncomment:
# thetas = np.asarray(your_theta_array) % (2*np.pi)

# -----------------------------
# 3) Helper functions
# -----------------------------

def circ_mean(thetas):
    """Sample mean direction on [0,2π)."""
    C = np.mean(np.cos(thetas))
    S = np.mean(np.sin(thetas))
    ang = np.arctan2(S, C)
    return ang % (2*np.pi)

def circ_resultant_length(thetas):
    C = np.mean(np.cos(thetas))
    S = np.mean(np.sin(thetas))
    return np.hypot(C, S)

def vonmises_kde(theta_grid, thetas, kappa=4.0):
    """
    Von Mises kernel density estimator on [0, 2π).
    f(θ) = (1/n) Σ K(θ - θ_i), with K the von Mises density of concentration kappa.
    """
    # Ensure broadcast: (grid_len, 1) - (1, n)
    diffs = theta_grid[:, None] - thetas[None, :]
    # vm kernel (no mean shift)
    kern = np.exp(kappa * np.cos(diffs)) / (2*np.pi * i0(kappa))
    f = kern.mean(axis=1)
    return f

def choose_kappa_rule_of_thumb(thetas):
    """
    Simple rule-of-thumb for KDE concentration:
    Use the sample resultant length Rbar to pick a single von Mises concentration
    for a baseline smooth. This is not optimal, but yields a professional-looking curve.
    """
    Rbar = circ_resultant_length(thetas)
    # Invert A(kappa) ~ Rbar (roughly) with quick approximations
    # (use standard approximations for kappa given Rbar)
    if Rbar < 0.53:
        kappa = 2*Rbar + Rbar**3 + 5*Rbar**5/6
    elif Rbar < 0.85:
        kappa = -0.4 + 1.39*Rbar + 0.43/(1 - Rbar)
    else:
        kappa = 1/(Rbar**3 - 4*Rbar**2 + 3*Rbar)
    # Make it a tad smaller for smoother KDE (acts like larger bandwidth)
    return max(kappa*0.8, 0.5)

# -----------------------------
# 4) Circular Histogram (Rose Diagram)
# -----------------------------

def plot_circular_histogram(thetas, nbins=24, filename_prefix="circ_hist_failureTimes"):
    # bins on [0,2π)
    bins = np.linspace(0, 2*np.pi, nbins+1)
    counts, _ = np.histogram(thetas, bins=bins)
    widths = np.diff(bins)
    # bar positions: center of each bin
    centers = (bins[:-1] + bins[1:]) / 2
    
    # Styling
    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax = fig.add_subplot(111, projection='polar')
    # Make 0 at top, clockwise positive angles (common in circular plots)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Professional colors: gradient bars with edge lines
    # Normalize heights for alpha mapping
    if counts.max() > 0:
        alphas = 0.35 + 0.55*(counts / counts.max())
    else:
        alphas = np.ones_like(counts)*0.5
    
    bars = ax.bar(centers, counts, width=widths, align='center',
                  linewidth=0.8, edgecolor='black',
                  color='#3B82F6', alpha=0.85)  # blue tone
    
    # Optional: color intensity by counts via alpha
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    
    # Mean direction line
    mu_hat = circ_mean(thetas)
    ax.plot([mu_hat, mu_hat], [0, counts.max()*1.05 if counts.max()>0 else 1],
            lw=2.2, color='#EF4444', alpha=0.95)  # red accent
    
    # Radius limit with some headroom
    rmax = (counts.max()*1.15) if counts.max()>0 else 1.0
    ax.set_rlim(0, rmax)
    
    # Radial grid + labels
    ax.set_rlabel_position(225)  # move radial labels away from overlap
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_title("Circular Histogram of Failure Angles", pad=14, fontsize=13)
    
    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{filename_prefix}.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path

# -----------------------------
# 5) Circular Reliability Curve R(θ) on polar axis
#    We compute a smoothed density via von Mises KDE, integrate to get F(θ), then R=1-F.
# -----------------------------

def plot_circular_reliability(thetas, grid_size=720, filename_prefix="circular_Reliability"):
    theta_grid = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    kappa_kde = choose_kappa_rule_of_thumb(thetas)
    f = vonmises_kde(theta_grid, thetas, kappa=kappa_kde)
    # Normalize numerically in case of tiny drift
    f = f / (np.trapz(f, theta_grid))
    
    # CDF and Reliability
    F = np.cumsum((f[:-1] + f[1:]) / 2) * (theta_grid[1] - theta_grid[0])
    F = np.concatenate([[0.0], F])
    # ensure ends at ~1
    F = F / (F[-1] if F[-1] != 0 else 1.0)
    R = 1.0 - F  # Reliability
    
    # Plot on polar: show both R(θ) and optionally density on a secondary style
    fig = plt.figure(figsize=(7,7), dpi=200)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Reliability curve
    ax.plot(theta_grid, R, lw=2.2, color='#10B981', alpha=0.95, label=r'$R(\theta)$')  # teal
    
    # Light density overlay (scaled to max(R) for visual context)
    Rmax = float(np.max(R)) if np.max(R)>0 else 1.0
    density_scaled = (f / f.max()) * (0.3 * Rmax)
    ax.plot(theta_grid, density_scaled, lw=1.4, color='#6366F1', alpha=0.85,
            label=r'$f(\theta)$ (scaled)')  # indigo
    
    # Mean direction
    mu_hat = circ_mean(thetas)
    ax.plot([mu_hat, mu_hat], [0, Rmax*1.02], lw=1.8, color='#EF4444', alpha=0.9, linestyle='--',
            label=r'$\hat{\mu}$')
    
    # Limits and grid
    ax.set_rlim(0, max(1.0, Rmax*1.1))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_rlabel_position(225)
    ax.set_title("Circular Reliability Function $R(\\theta)$", pad=14, fontsize=13)
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), frameon=False)
    
    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{filename_prefix}.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path

# -----------------------------
# Run and save both plots
# -----------------------------
hist_png, hist_pdf = plot_circular_histogram(thetas, nbins=24, filename_prefix="circ_hist_failureTimes")
reli_png, reli_pdf = plot_circular_reliability(thetas, grid_size=720, filename_prefix="circular_Reli")

(hist_png, hist_pdf, reli_png, reli_pdf)
