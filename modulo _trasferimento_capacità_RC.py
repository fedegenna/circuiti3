import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def modello_modulo(omega, R, C):
    return 1/((omega*C)* np.sqrt(R**2 + (1 / (omega  * C))**2))


def main():


    frequenze = np.array([6,  12,  25,  50,  70,  90, 110, 120, 130, 140, 150,
   160, 165, 166, 167, 170, 180, 200, 250, 300, 350, 400, 500,
 1000, 2000, 4000])
    omega = 2 * np.pi * frequenze

    Va = np.array([
 2.02, 2.02, 2.02, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.98
])
    Va_err = np.array([
 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02
])
    
    Va_Vb = np.array([
 2.00, 2.00, 1.96, 1.90, 1.86, 1.74, 1.64, 1.64, 1.56,
 1.56, 1.48, 1.42, 1.40, 1.40, 1.40, 1.40, 1.34, 1.24, 1.08,
 0.98, 0.84, 0.76, 0.66, 0.35, 0.20, 0.12
])
    Va_Vb_err = np.array([
 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01,
 0.02, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02
])
    
    rapporti = Va_Vb / Va
    err_rapporti = np.sqrt((Va_Vb_err / Va)**2 + (Va_err * Va_Vb / Va**2)**2)
    
    # Stime iniziali
    p0 = [9.85 *1000,  98 * 10 ** (-9)]  # R, C
    bounds = ([8000, 0], [10000000, 100000000])  # vincoli positivi realistici

    # Fit
    popt, pcov = curve_fit(
        modello_modulo, omega, rapporti,
        sigma=err_rapporti, absolute_sigma=True,
        p0=p0, bounds=bounds, maxfev=10000000000
    )
    R_fit, C_fit = popt
    R_err, C_err = np.sqrt(np.diag(pcov))

    # Chi² e p-value
    residuals = rapporti - modello_modulo(omega, *popt)
    chi2_val = np.sum((residuals / err_rapporti)**2)
    ndof = len(omega) - len(popt)
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

        # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
    omega_fit = np.linspace(min(omega), max(omega), 500)
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label=
             f'R = ({R_fit*0.001:.1f} ± {R_err*0.001:.1f}) kOhm \n'
             f'C = ({C_fit*10**(8):.1f} ± {C_err*10**(8):.1f}) $10^{{-8}}$ F\n')

    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('|(Va-Vb)/Va|')
    
    #plt.title('Fit con modello sovrasmorzato')
    plt.grid(True)
    plt.legend()

    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(0.95, 0.8, textstr, fontsize=12,
             transform=plt.gca().transAxes,
             ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    # Plot con asse x in scala logaritmica
    plt.figure(figsize=(10, 6))
    plt.errorbar(omega, rapporti, yerr=err_rapporti, fmt='o', label='Dati')
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label='Fit')
    plt.xscale('log')
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('|(Va-Vb)/Va|')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.title('Scala logaritmica')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()