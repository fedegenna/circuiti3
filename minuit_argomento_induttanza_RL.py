import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2 as chi2dist

# Dati
frequenze = np.array([1.2, 2.5, 5, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35, 40, 50, 60, 80, 90, 150, 300])
omega = 2 * np.pi * frequenze * 1000

fase_gen_L = np.array([91.3, 88.1, 75.9, 65.2, 64.6, 57.6, 54.9, 51.8, 49.7, 47.7, 47.8, 48.2, 45.0,
                       42.8, 40.8, 40.5, 36.4, 34.2, 33.7, 23.2, 21.8, 16.1, 12.1, 7.1, 4.2])

fase_gen_L_err = np.array([1.2, 1.5, 0.9, 2.4, 1.3, 1.2, 2.9, 2.4, 2.9, 1.4, 2.6, 2.0, 1.9,
                           0.6, 0.9, 1.2, 1.0, 2.4, 2.0, 0.9, 1.8, 2.8, 3.0, 1.4, 2.5])

# Modello
def modello_modulo(omega, R, L):
    return (180 / np.pi) * np.arctan(R / (omega * L))

# Funzione di costo
least_squares = LeastSquares(omega, fase_gen_L, fase_gen_L_err, modello_modulo)

# Fit con Minuit
m = Minuit(least_squares, R=9850, L=0.067)
m.limits["R"] = (8000, 10100)
m.limits["L"] = (0.063, 0.070)
m.migrad()
m.hesse()

# Risultati
R_fit, L_fit = m.values["R"], m.values["L"]
R_err, L_err = m.errors["R"], m.errors["L"]
chi2_val = m.fval
ndof = len(omega) - 2
chi2_red = chi2_val / ndof
p_value = 1 - chi2dist.cdf(chi2_val, ndof)

# Predizione modello
omega_fit = np.linspace(min(omega), max(omega), 500)
fase_fit = modello_modulo(omega_fit, R_fit, L_fit)

# ----------------------------
# Plot normale
# ----------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(omega, fase_gen_L, yerr=fase_gen_L_err, fmt='o', label='Dati')
plt.plot(omega_fit, fase_fit, 'r-', label=f'Fit\n'
                                          f'R = ({R_fit*1e-3:.2f} ± {R_err*1e-3:.2f}) kΩ\n'
                                          f'L = ({L_fit*1e3:.2f} ± {L_err*1e3:.2f}) mH')
plt.xlabel('ω (rad/s)')
plt.ylabel('fase (°)')
plt.grid(True)
plt.legend()
plt.text(omega[0], max(fase_gen_L)*0.8,
         f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
#plt.title("Fit della fase in circuito RL")
plt.show()

# ----------------------------
# Plot scala logaritmica
# ----------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(omega, fase_gen_L, yerr=fase_gen_L_err, fmt='o', label='Dati')
plt.plot(omega_fit, fase_fit, 'r-', label=f'Fit\n'
                                          f'R = ({R_fit*1e-3:.2f} ± {R_err*1e-3:.2f}) kΩ\n'
                                          f'L = ({L_fit*1e3:.2f} ± {L_err*1e3:.2f}) mH')
plt.xscale('log')
plt.xlabel('ω (rad/s)')
plt.ylabel('fase (°)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.text(omega[0], max(fase_gen_L)*0.8,
         f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}",
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
#plt.title("Fit della fase in scala logaritmica")
plt.tight_layout()
plt.show()
