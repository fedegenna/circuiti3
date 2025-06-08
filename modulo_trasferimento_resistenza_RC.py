import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def modello_modulo(omega, R, C):
    return 1 / np.sqrt(1 + (1 / (omega * R * C))**2)


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

    Vb = np.array([
 0.12, 0.23, 0.35, 0.61, 0.78, 0.94, 1.10, 1.17, 1.22,
 1.27, 1.33, 1.38, 1.38, 1.38, 1.40, 1.41, 1.46, 1.53, 1.66,
 1.74, 1.80, 1.86, 1.89, 1.97, 2.00, 1.98
])
    Vb_err = np.array([
 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
])

    rapporti = Vb/Va
    err_rapporti = np.sqrt((Vb_err / Va)**2 + (Va_err * Vb / Va**2)**2)
    '''
    Va_Vb = np.array([
 2.00, 2.00, 1.96, 1.90, 1.86, 1.74, 1.64, 1.64, 1.64, 1.56,
 1.56, 1.48, 1.42, 1.40, 1.40, 1.40, 1.34, 1.34, 1.24, 0.94,
 0.98, 0.84, 0.76, 0.68, 0.44, 0.24, 0.22
])
    Va_Vb_err = np.array([
 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01,
 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02
])
    fase_gen_R = np.array([
 89.2, 84.7, 80.5, 70.5, 70.4, 60.4, 58.3, 54.8, 54.5, 50.4,
 49.3, 49.1, 47.5, 46.7, 46.4, 46.8, 46.2, 41.5, 41.2, 35.1,
 28.1, 25.5, 24.0, 19.4, 9.4, 6.9, 3.5
])
    fase_gen_R_err = np.array([
 1.5, 1.2, 1.1, 2.5, 2.1, 2.2, 1.4, 1.1, 2.4, 1.7,
 1.3, 0.7, 0.4, 0.6, 1.4, 0.9, 0.2, 1.5, 1.3, 0.9,
 1.1, 1.4, 0.8, 0.4, 0.6, 0.8, 2.5
])
    fase_gen_C = np.array([
  0.8,  5.3,  9.5, 15.9, 21.3, 28.3, 30.5, 32.6, 32.8, 38.0,
 38.4, 37.2, 41.8, 42.3, 44.3, 44.2, 44.3, 47.2, 46.8, 54.3,
 60.3, 60.8, 62.6, 69.4, 81.0, 84.6, 86.5
])
    fase_gen_C_err = np.array([0.8, 0.3, 0.4, 0.5, 1.3, 1.5, 1.4, 1.4, 1.4, 0.8,
 1.2, 0.8, 0.4, 1.1, 1.4, 1.7, 2.3, 0.8, 0.9, 1.2,
 0.4, 2.1, 1.1, 1.1, 1.3, 1.7, 2.5
])
    '''
    # Stime iniziali
    p0 = [9.85 *1000,  98 * 10 ** (-9)]  # R, C
    bounds = ([0, 0], [10000000, 100000000])  # vincoli positivi realistici

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
             f'R = ({R_fit * 0.001:.1f} ± {R_err * 0.001:.1f}) kΩ \n'
             f'C = ({C_fit* 10 ** (8):.1f} ± {C_err * 10 ** (8):.1f}) $10^{{-8}}$ F\n')

    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('|Vb/Va|')
    
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
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label='fit')
    plt.xscale('log')
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('|Vb/Va|')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.title('Scala logaritmica')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()