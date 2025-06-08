import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Dati
frequenze = np.array([1.2, 2.5, 5, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35, 40, 50, 60, 80, 90, 150, 300])
omega = 2 * np.pi * frequenze * 1000

Va = np.array([1.98, 1.98, 1.98, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
               2.00, 1.98, 1.99, 1.98, 2.00, 2.01, 2.02, 2.02])

Va_err = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01, 0.05, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01])

Va_Vb = np.array([0.12, 0.23, 0.43, 0.76, 0.92, 1.02, 1.12, 1.24, 1.32, 1.32, 1.40, 1.40, 1.46, 1.48, 1.48,
                  1.56, 1.56, 1.680, 1.74, 1.82, 1.84, 1.94, 1.96, 2, 2])

Va_Vb_err = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                      0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01])

rapporti = Va_Vb / Va
err_rapporti = np.sqrt((Va_Vb_err / Va) ** 2 + (Va_err * Va_Vb / Va ** 2) ** 2)

# Modello
def modello_modulo(omega, R, L):
    return omega * L / np.sqrt(R**2 + (omega * L)**2)

# Fit con Minuit
least_squares = LeastSquares(omega, rapporti, err_rapporti, modello_modulo)
m = Minuit(least_squares, R=9850, L=0.067)
m.limits["R"] = (8000, 10000)
m.limits["L"] = (0.063, 0.073)

m.migrad()
m.hesse()

# Estrazione risultati
R_fit, L_fit = m.values["R"], m.values["L"]
R_err, L_err = m.errors["R"], m.errors["L"]

# Calcolo chi² ridotto e p-value
chi2_val = m.fval
ndof = len(omega) - 2
chi2_red = chi2_val / ndof
p_value = 1 - chi2.cdf(chi2_val, ndof)

# Plot
omega_fit = np.linspace(min(omega), max(omega), 500)
modulo_fit = modello_modulo(omega_fit, R_fit, L_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
plt.plot(omega_fit, modulo_fit, 'r-', label=f'Fit con Minuit\n'
                                            f'R = ({R_fit*1e-3:.1f} ± {R_err*1e-3:.1f}) kΩ\n'
                                            f'L = ({L_fit*1e3:.1f} ± {L_err*1e3:.1f}) mH')
plt.xlabel('ω (rad/s)')
plt.ylabel('|Vc/Va|')
plt.grid(True)
plt.legend()
plt.text(0.7, 0.7, f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

# Log scale plot
plt.figure(figsize=(10, 6))
plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
plt.plot(omega_fit, modulo_fit, 'r-', label=f'Fit\n'
                                            f'R = ({R_fit*1e-3:.3f} ± {R_err*1e-3:.3f}) kΩ\n'
                                            f'L = ({L_fit*1e3:.1f} ± {L_err*1e3:.1f}) mH')
plt.xscale('log')
plt.xlabel('ω (rad/s)')
plt.ylabel('|Vc/Va|')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.text(0.7, 0.15, f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.title('Scala logaritmica per ω')
plt.tight_layout()
plt.show()
