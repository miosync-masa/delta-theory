#!/usr/bin/env python3
"""
Universal Lindemann Law Validation
===================================
δ_L = (√48 / Z) × √(k_B T_m / E_coh)

No fitting parameters. Pure topology + thermodynamics.

Author: Masamichi Iizumi / Miosync Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Physical constants
k_B = 8.617333262e-5  # eV/K (Boltzmann constant)

# Geometric constant
C_geom = np.sqrt(48)  # = 4√3 ≈ 6.928 (FCC reference: 12²/3)


@dataclass
class Metal:
    """Metal properties for Lindemann calculation"""
    name: str
    structure: str
    Z: int           # Coordination number
    T_m: float       # Melting temperature [K]
    E_coh: float     # Cohesive energy [eV]
    r_nn: float      # Nearest neighbor distance [Å]
    delta_exp: float # Experimental Lindemann ratio


# =============================================================================
# Experimental Data (7 metals: BCC, FCC, HCP)
# =============================================================================
metals = [
    Metal("Fe", "BCC", 8,  1811, 4.28, 2.48, 0.180),
    Metal("W",  "BCC", 8,  3695, 8.90, 2.74, 0.160),
    Metal("Cu", "FCC", 12, 1357, 3.49, 2.56, 0.100),
    Metal("Al", "FCC", 12,  933, 3.39, 2.86, 0.100),
    Metal("Ni", "FCC", 12, 1728, 4.44, 2.49, 0.110),
    Metal("Ti", "HCP", 12, 1941, 4.85, 2.95, 0.100),
    Metal("Mg", "HCP", 12,  923, 1.51, 3.21, 0.117),
]


def universal_lindemann(Z: int, T_m: float, E_coh: float) -> float:
    """
    Universal Lindemann Law (parameter-free)
    
    δ_L = (√48 / Z) × √(k_B T_m / E_coh)
    
    Args:
        Z: Coordination number (8 for BCC, 12 for FCC/HCP)
        T_m: Melting temperature [K]
        E_coh: Cohesive energy [eV]
    
    Returns:
        Lindemann ratio δ_L (dimensionless)
    """
    return (C_geom / Z) * np.sqrt(k_B * T_m / E_coh)


def rms_displacement(delta_L: float, r_nn: float) -> float:
    """
    RMS atomic displacement at melting point
    
    √<u²> = δ_L × r_nn
    
    Args:
        delta_L: Lindemann ratio
        r_nn: Nearest neighbor distance [Å]
    
    Returns:
        RMS displacement [Å]
    """
    return delta_L * r_nn


def validate_all():
    """Calculate and display results for all metals"""
    
    print("=" * 80)
    print("UNIVERSAL LINDEMANN LAW VALIDATION")
    print("δ_L = (√48 / Z) × √(k_B T_m / E_coh)")
    print("=" * 80)
    print(f"\nGeometric constant: √48 = {C_geom:.4f} (= 12²/3, FCC reference)")
    print(f"Boltzmann constant: k_B = {k_B:.6e} eV/K\n")
    
    # Results storage
    results = []
    
    # Table header
    print("-" * 80)
    print(f"{'Metal':>6} {'Struct':>6} {'Z':>3} {'T_m[K]':>8} {'E_coh[eV]':>10} "
          f"{'δ_exp':>7} {'δ_pred':>7} {'Err[%]':>8}")
    print("-" * 80)
    
    for m in metals:
        # Calculate predicted Lindemann ratio
        delta_pred = universal_lindemann(m.Z, m.T_m, m.E_coh)
        
        # Calculate RMS displacement
        rms_pred = rms_displacement(delta_pred, m.r_nn)
        rms_exp = rms_displacement(m.delta_exp, m.r_nn)
        
        # Error
        error = (delta_pred - m.delta_exp) / m.delta_exp * 100
        
        # Store results
        results.append({
            'metal': m,
            'delta_pred': delta_pred,
            'rms_pred': rms_pred,
            'rms_exp': rms_exp,
            'error': error
        })
        
        print(f"{m.name:>6} {m.structure:>6} {m.Z:>3} {m.T_m:>8.0f} {m.E_coh:>10.2f} "
              f"{m.delta_exp:>7.3f} {delta_pred:>7.3f} {error:>+8.1f}")
    
    print("-" * 80)
    
    # Statistics
    errors = [abs(r['error']) for r in results]
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    delta_exp_arr = np.array([r['metal'].delta_exp for r in results])
    delta_pred_arr = np.array([r['delta_pred'] for r in results])
    correlation = np.corrcoef(delta_exp_arr, delta_pred_arr)[0, 1]
    
    print(f"\nStatistics:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}%")
    print(f"  Root Mean Square Error:    {rmse:.2f}%")
    print(f"  Correlation coefficient:   {correlation:.4f}")
    
    # RMS Displacement Table
    print("\n" + "-" * 80)
    print("RMS ATOMIC DISPLACEMENTS AT MELTING POINT")
    print("-" * 80)
    print(f"{'Metal':>6} {'r_nn[Å]':>8} {'RMS_exp[Å]':>12} {'RMS_pred[Å]':>12}")
    print("-" * 80)
    
    for r in results:
        m = r['metal']
        print(f"{m.name:>6} {m.r_nn:>8.2f} {r['rms_exp']:>12.3f} {r['rms_pred']:>12.3f}")
    
    print("-" * 80)
    
    # Structure-dependent analysis
    print("\n" + "-" * 80)
    print("STRUCTURE-DEPENDENT ANALYSIS")
    print("-" * 80)
    
    for struct in ["BCC", "FCC", "HCP"]:
        struct_results = [r for r in results if r['metal'].structure == struct]
        if struct_results:
            Z = struct_results[0]['metal'].Z
            avg_exp = np.mean([r['metal'].delta_exp for r in struct_results])
            avg_pred = np.mean([r['delta_pred'] for r in struct_results])
            struct_mae = np.mean([abs(r['error']) for r in struct_results])
            cage_ratio = 12 / Z
            print(f"  {struct} (Z={Z}): δ_exp={avg_exp:.3f}, δ_pred={avg_pred:.3f}, "
                  f"MAE={struct_mae:.1f}%, Cage ratio (12/Z)={cage_ratio:.2f}")
    
    return results


def create_figure(results):
    """Create validation figure with 3 panels"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Color scheme by structure
    colors = {'BCC': '#E74C3C', 'FCC': '#3498DB', 'HCP': '#2ECC71'}
    markers = {'BCC': 's', 'FCC': 'o', 'HCP': '^'}
    
    # ==========================================================================
    # Panel (a): Predicted vs Experimental δ_L
    # ==========================================================================
    ax1 = axes[0]
    
    for r in results:
        m = r['metal']
        ax1.scatter(m.delta_exp, r['delta_pred'], 
                   c=colors[m.structure], marker=markers[m.structure],
                   s=120, edgecolors='black', linewidths=0.5, zorder=5)
        ax1.annotate(m.name, (m.delta_exp, r['delta_pred']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Identity line
    lims = [0.05, 0.20]
    ax1.plot(lims, lims, 'k--', lw=1.5, label='Identity', zorder=1)
    
    # ±10% bounds
    ax1.fill_between(lims, [x*0.9 for x in lims], [x*1.1 for x in lims],
                     alpha=0.15, color='gray', label='±10%')
    
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(r'Experimental $\delta_L$', fontsize=11)
    ax1.set_ylabel(r'Predicted $\delta_L$', fontsize=11)
    ax1.set_title('(a) Universal Lindemann Law Validation', fontsize=12)
    ax1.set_aspect('equal')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel (b): Structure-dependent trends
    # ==========================================================================
    ax2 = axes[1]
    
    x_positions = {'BCC': 0, 'FCC': 1, 'HCP': 2}
    
    for struct in ['BCC', 'FCC', 'HCP']:
        struct_results = [r for r in results if r['metal'].structure == struct]
        x = x_positions[struct]
        
        for i, r in enumerate(struct_results):
            m = r['metal']
            offset = (i - len(struct_results)/2 + 0.5) * 0.15
            
            # Experimental (filled)
            ax2.scatter(x + offset - 0.05, m.delta_exp, 
                       c=colors[struct], marker=markers[struct],
                       s=100, edgecolors='black', linewidths=0.5)
            
            # Predicted (open)
            ax2.scatter(x + offset + 0.05, r['delta_pred'], 
                       c='white', marker=markers[struct],
                       s=100, edgecolors=colors[struct], linewidths=2)
            
            ax2.annotate(m.name, (x + offset, max(m.delta_exp, r['delta_pred']) + 0.005),
                        ha='center', fontsize=8)
    
    # Predicted averages
    for struct in ['BCC', 'FCC', 'HCP']:
        struct_results = [r for r in results if r['metal'].structure == struct]
        Z = struct_results[0]['metal'].Z
        avg_pred = np.mean([r['delta_pred'] for r in struct_results])
        x = x_positions[struct]
        ax2.axhline(y=avg_pred, xmin=(x-0.3)/3+0.05, xmax=(x+0.3)/3+0.05,
                   color=colors[struct], linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['BCC\n(Z=8)', 'FCC\n(Z=12)', 'HCP\n(Z=12)'])
    ax2.set_ylabel(r'Lindemann ratio $\delta_L$', fontsize=11)
    ax2.set_title('(b) Structure Dependence', fontsize=12)
    ax2.set_ylim([0.05, 0.22])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Legend for panel b
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Experimental'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markeredgewidth=2, markersize=10, label='Predicted')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # ==========================================================================
    # Panel (c): Energy ratio analysis
    # ==========================================================================
    ax3 = axes[2]
    
    for r in results:
        m = r['metal']
        energy_ratio = np.sqrt(k_B * m.T_m / m.E_coh)
        
        ax3.scatter(energy_ratio, m.delta_exp,
                   c=colors[m.structure], marker=markers[m.structure],
                   s=120, edgecolors='black', linewidths=0.5, zorder=5)
        ax3.annotate(m.name, (energy_ratio, m.delta_exp),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Theoretical lines
    x_range = np.linspace(0.05, 0.25, 100)
    
    # BCC line (Z=8)
    ax3.plot(x_range, (C_geom/8) * x_range, 
             color=colors['BCC'], linestyle='-', linewidth=2, 
             label=f'BCC: δ = {C_geom/8:.3f} × √(k_BT_m/E_coh)')
    
    # FCC/HCP line (Z=12)
    ax3.plot(x_range, (C_geom/12) * x_range,
             color=colors['FCC'], linestyle='-', linewidth=2,
             label=f'FCC/HCP: δ = {C_geom/12:.3f} × √(k_BT_m/E_coh)')
    
    ax3.set_xlabel(r'$\sqrt{k_B T_m / E_{coh}}$', fontsize=11)
    ax3.set_ylabel(r'Lindemann ratio $\delta_L$', fontsize=11)
    ax3.set_title('(c) Energy Scaling', fontsize=12)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0.05, 0.22])
    ax3.set_ylim([0.05, 0.22])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('/content/universal_lindemann_validation.png', dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.savefig('/content/universal_lindemann_validation.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("\nFigures saved:")
    print("  - universal_lindemann_validation.png")
    print("  - universal_lindemann_validation.pdf")
    
    return fig


def print_latex_table(results):
    """Generate LaTeX table for paper"""
    
    print("\n" + "=" * 80)
    print("LaTeX TABLE")
    print("=" * 80)
    
    print(r"""
\begin{table}[htbp]
\centering
\caption{Validation of the Universal Lindemann Law: 
$\delta_L = \frac{\sqrt{48}}{Z}\sqrt{\frac{k_B T_m}{E_{coh}}}$.
No fitting parameters.}
\label{tab:universal_lindemann}
\begin{tabular}{lcccccccc}
\toprule
Metal & Structure & $Z$ & $T_m$ [K] & $E_{coh}$ [eV] & $r_{nn}$ [\AA] & $\delta_L^{exp}$ & $\delta_L^{pred}$ & Error [\%] \\
\midrule""")
    
    for r in results:
        m = r['metal']
        error_str = f"+{r['error']:.1f}" if r['error'] > 0 else f"{r['error']:.1f}"
        print(f"{m.name} & {m.structure} & {m.Z} & {m.T_m:.0f} & {m.E_coh:.2f} & "
              f"{m.r_nn:.2f} & {m.delta_exp:.3f} & {r['delta_pred']:.3f} & {error_str} \\\\")
    
    # Statistics
    errors = [abs(r['error']) for r in results]
    mae = np.mean(errors)
    
    print(r"""\midrule
\multicolumn{9}{l}{\textbf{Mean Absolute Error (MAE): """ + f"{mae:.1f}" + r"""\%}} \\
\bottomrule
\end{tabular}
\end{table}
""")


def print_rms_latex_table(results):
    """Generate LaTeX table for RMS displacements"""
    
    print("\n" + "=" * 80)
    print("LaTeX TABLE - RMS DISPLACEMENTS")
    print("=" * 80)
    
    print(r"""
\begin{table}[htbp]
\centering
\caption{RMS atomic displacements at the melting point.}
\label{tab:rms_displacement}
\begin{tabular}{lccccc}
\toprule
Metal & Structure & $r_{nn}$ [\AA] & $\sqrt{\langle u^2 \rangle}^{exp}$ [\AA] & $\sqrt{\langle u^2 \rangle}^{pred}$ [\AA] \\
\midrule""")
    
    for r in results:
        m = r['metal']
        print(f"{m.name} & {m.structure} & {m.r_nn:.2f} & {r['rms_exp']:.3f} & {r['rms_pred']:.3f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    """Main execution"""
    
    # Validate all metals
    results = validate_all()
    
    # Create figure
    fig = create_figure(results)
    
    # Print LaTeX tables
    print_latex_table(results)
    print_rms_latex_table(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    Universal Lindemann Law:
    
        δ_L = (√48 / Z) × √(k_B T_m / E_coh)
            = ({C_geom:.3f} / Z) × √(k_B T_m / E_coh)
    
    Physical meaning:
        √48 = 12/√3 ≈ 6.928  :  3D FCC cage geometry (12²/3)
        1/Z                   :  Topological cage size
        √(k_B T_m / E_coh)    :  Energy ratio (thermal/cohesive)
    
    Results:
        - 7 metals validated (Fe, W, Cu, Al, Ni, Ti, Mg)
        - 3 crystal structures (BCC, FCC, HCP)
        - Mean Absolute Error: {np.mean([abs(r['error']) for r in results]):.1f}%
        - Fitting parameters: ZERO
    
    The Lindemann criterion is NOT an empirical rule—
    it is a geometric identity arising from topological cage confinement.
    """)
    
    return results


if __name__ == "__main__":
    results = main()
