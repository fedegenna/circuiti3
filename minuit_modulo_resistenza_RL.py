import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
# -----------------------
# DATI
# -----------------------
frequenze = np.array([1.2, 2.5, 5, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35, 40, 50, 60, 80, 90, 150, 300])
omega = 2 * np.pi * frequenze * 1000

Va = np.array([1.98, 1.98, 1.98, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 1.98, 1.99, 1.98, 2.00, 2.01, 2.02, 2.02])

Va_err =  np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01])

Vb = np.array ([1.98, 1.98, 1.95, 1.86, 1.80, 1.74, 1.66, 1.60, 1.54, 1.50, 1.48, 1.44, 1.40, 1.38, 1.36, 1.30, 1.24,
    1.12, 1.00, 0.85, 0.68, 0.52, 0.50, 0.24, 0.17])

Vb_err =  np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.01, 0.04, 0.02, 0.03, 0.01])


rapporti = Vb / Va
err_rapporti = np.sqrt((Vb_err / Va)**2 + (Va_err * Vb / Va**2)**2)

# -----------------------
# MODELLO
# -----------------------
def modello(omega, R, L):
    return R / np.sqrt(R**2 + (omega * L)**2)

# -----------------------
# FIT con Minuit
# -----------------------
least_squares = LeastSquares(omega, rapporti, err_rapporti, modello)

m = Minuit(least_squares, R=980, L=0.065)  # stime iniziali plausibili
m.limits["R"] = (8050, 10500)
m.limits["L"] = (0.05, 0.1)
m.strategy = 2
m.errordef = 1

m.migrad()
m.hesse()


# RISULTATI
# -----------------------
R_fit = m.values["R"]
L_fit = m.values["L"]
R_err = m.errors["R"]
L_err = m.errors["L"]
chi2_red = m.fval / (len(omega) - 2)
ndof = len(omega) - 2
p_value = 1 - chi2.cdf(m.fval, ndof)
# -----------------------
# PLOT
# -----------------------
omega_fit = np.linspace(min(omega), max(omega), 500)
modulo_fit = modello(omega_fit, R_fit, L_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
plt.plot(omega_fit, modulo_fit, 'r-', label='Fit')

plt.xlabel('ω (rad/s)')
plt.ylabel('|Vb/Va|')
plt.grid(True)
plt.legend()

textstr = (
    f"R = {R_fit*0.001:.1f} ± {R_err*0.001:.1f} kΩ\n"
    f"L = {L_fit*1000:.0f} ± {L_err*1000:.0f} mH\n"
    f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n"
    f"$p$-value = {p_value:.3f}"
)
plt.text(0.8, 0.85, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

# -----------------------
# PLOT SCALA LOGARITMICA
# -----------------------
plt.figure(figsize=(10, 6))
plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
plt.plot(omega_fit, modulo_fit, 'r-', label='Fit')

plt.xscale('log')
plt.xlabel('ω (rad/s)')
plt.ylabel('|Vb/Va|')
plt.grid(True, which='both', ls='--')
plt.legend()

plt.text(0.05, 0.80, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

#plt.title('Scala logaritmica per ω')
plt.tight_layout()
plt.show()
