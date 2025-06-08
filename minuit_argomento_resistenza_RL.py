import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Dati
frequenze = np.array([1.2, 2.5, 5, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35, 40, 50, 60, 80, 90, 150, 300])
omega = 2 * np.pi * frequenze * 1000

fase_gen_R = np.array([-2.2, -4.5, -14.1, -23.4, -26.5, -31.4, -34.2, -38.3, -40.5, -43.2, -46.3, -43.2, -46.0,
                       -48.5, -49.1, -51.3, -54.6, -57.6, -60.5, -67.9, -71.6, -72.8, -78.3, -83.2, -86.3])
fase_gen_R_err = np.array([1.2, 1.7, 1.1, 0.8, 0.6, 0.5, 1.1, 1.0, 0.9, 0.8, 1.4, 0.5, 2.0,
                           0.6, 1.1, 1.2, 0.9, 0.8, 2.0, 2.3, 1.7, 3.1, 4.2, 4, 3.9])
print(len (frequenze), len(fase_gen_R), len(fase_gen_R_err))
# Modello di fase
def modello_fase(omega, R, L):
    return -(180 / np.pi) * np.arctan((omega * L) / R)

# Cost function
least_squares = LeastSquares(omega, fase_gen_R, fase_gen_R_err, modello_fase)

# Fit con Minuit
m = Minuit(least_squares, R=9850, L=0.067)
m.limits["R"] = (8000, 10000)
m.limits["L"] = (0.055, 0.075)
m.migrad()
m.hesse()

# Estrazione risultati
R_fit, L_fit = m.values["R"], m.values["L"]
R_err, L_err = m.errors["R"], m.errors["L"]

# Calcolo chi2 ridotto e p-value
chi2_val = m.fval
ndof = len(omega) - 2
chi2_red = chi2_val / ndof
from scipy.stats import chi2 as chi2dist
p_value = 1 - chi2dist.cdf(chi2_val, ndof)

# Plot
omega_fit = np.linspace(min(omega), max(omega), 500)
fase_fit = modello_fase(omega_fit, R_fit, L_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(omega, fase_gen_R, yerr=fase_gen_R_err, fmt='o', label='Dati')
plt.plot(omega_fit, fase_fit, 'r-', label=f'Fit\n'
                                          f'R = ({R_fit*1e-3:.2f} ± {R_err*1e-3:.2f}) kΩ\n'
                                          f'L = ({L_fit*1e3:.2f} ± {L_err*1e3:.2f}) mH')
plt.xlabel('ω (rad/s)')
plt.ylabel('fase (°)')
plt.grid(True)
plt.legend()
plt.text(0.05, 0.15, f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

# -----------------------
# PLOT SCALA LOGARITMICA
# -----------------------
plt.figure(figsize=(10, 6))
plt.errorbar(omega, fase_gen_R, yerr=fase_gen_R_err, fmt='o', label='Dati')
plt.plot(omega_fit, fase_fit, 'r-', label=f'Fit\n'
                                          f'R = ({R_fit*1e-3:.2f} ± {R_err*1e-3:.2f}) kΩ\n'
                                          f'L = ({L_fit*1e3:.2f} ± {L_err*1e3:.2f}) mH')
plt.xscale('log')
plt.xlabel('ω (rad/s)')
plt.ylabel('fase (°)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.text(0.05, 0.15, f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
#plt.title('Scala logaritmica per ω')
plt.tight_layout()
plt.show()