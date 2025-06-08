import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def modello_modulo(omega, R, C):
    return (180 / np.pi) * np.arctan(1 / (omega * R * C))


def main():
    frequenze = np.array([6,  12,  25,  50,  70,  90, 110, 120, 130, 140, 150,
   160, 165, 166, 167, 170, 180, 200, 250, 300, 350, 400, 500,
 1000, 2000, 4000])
    omega = 2 * np.pi * frequenze
    
    fase_gen_R = np.array([
 89.2, 84.7, 80.5, 70.5, 70.4, 60.4, 58.3, 54.5, 50.4,
 49.3, 49.1, 47.5, 46.7, 46.4, 46.6, 46.2, 41.5, 41.2, 35.1,
 28.1, 25.5, 24.0, 19.4, 9.4, 5.4, 3.5
])
    fase_gen_R_err = np.array([
 1.5, 1.2, 1.1, 1.5, 1.1, 1.2, 1.0, 1.4, 1.7,
 1.3, 0.7, 0.4, 0.6, 1.4, 0.9, 0.7, 1.5, 1.3, 0.9,
 1.0, 1.1, 0.6, 0.4, 0.6, 0.8, 2.5
])
    
    # Stime iniziali
    p0 = [9.85 *1000,  98 * 10 ** (-9)]  # R, C
    bounds = ([8000, 0], [10000000, 100000000])  # vincoli positivi realistici

    # Fit
    popt, pcov = curve_fit(
        modello_modulo, omega, fase_gen_R,
        sigma=fase_gen_R_err, absolute_sigma=True,
        p0=p0, bounds=bounds, maxfev=10000000000
    )
    R_fit, C_fit = popt
    R_err, C_err = np.sqrt(np.diag(pcov))

    # Chi² e p-value
    residuals = fase_gen_R - modello_modulo(omega, *popt)
    chi2_val = np.sum((residuals / fase_gen_R_err)**2)
    ndof = len(omega) - len(popt)
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

        # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(omega, fase_gen_R, yerr=fase_gen_R_err, fmt='o', label='Dati')
    omega_fit = np.linspace(min(omega), max(omega), 500)
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label=
             f'R = ({R_fit*0.001:.1f} ± {R_err*0.001:.1f}) kΩ\n'
             f'C = ({C_fit * 10 ** (8):.1f} ± {C_err * 10 ** (8):.1f}) $10^{{-8}}$ F\n')

    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('fase (°)')
    
    #plt.title('Fit con modello sovrasmorzato')
    plt.grid(True)
    plt.legend()

    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(omega[0], max(fase_gen_R)*0.8, textstr, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    # Plot con asse x in scala logaritmica
    plt.figure(figsize=(10, 6))
    plt.errorbar(omega, fase_gen_R, yerr=fase_gen_R_err, fmt='o', label='Dati')
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label=
             'Fit')
    plt.xscale('log')
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('fase (°)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.title('Scala logaritmica')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()