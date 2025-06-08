import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def modello_modulo(omega, R, C):
    return -(180 / np.pi) * np.arctan(omega * R * C)


def main():
    frequenze = np.array([6,  12,  25,  50,  70,  90, 110, 120, 130, 140, 150,
   160, 165, 166, 167, 170, 180, 200, 250, 300, 350, 400, 500,
 1000, 2000, 4000])
    omega = 2 * np.pi * frequenze
    '''
    Va = 2* np.ones(20)
    Va_err =  0.01 *np.ones(20)

    Vb = np.array([0.610, 0.78, 0.94, 1.10, 1.17, 1.22, 1.27, 1.33, 1.38, 1.38, 1.38, 1.40, 1.41, 1.46, 1.53, 1.66, 1.74, 1.80, 1.86, 1.89])
    Vb_err = 0.01 * np.ones(20)

    rapporti = Vb/Va
    err_rapporti = np.sqrt((Vb_err / Va)**2 + (Va_err * Vb / Va**2)**2)

    Va_Vb = [1.90, 1.86, 1.74, 1.64, 1.64, 1.56, 1.56, 1.48, 1.42, 1.40, 1.40, 1.40, 1.40, 1.34, 1.24, 1.08, 0.98, 0.84, 0.76, 0.68]
    Va_Vb_err = [0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.06, 0.04, 0.01, 0.01]
    
    fase_gen_R = [70.5, 70.4, 60.4, 58.3, 54.5, 50.4, 49.3, 49.1, 47.5, 46.7, 46.4, 46.6, 46.2, 41.5, 41.2, 35.1, 28.1, 25.5, 24.0, 19.4]
    fase_gen_R_err = [2.5, 2.1, 2.2, 1.0, 2.4, 1.7, 1.3, 0.7, 0.4, 0.6, 1.4, 0.9, 0.7, 1.5, 1.3, 0.9, 1.0, 1.1, 0.6, 0.4]
    '''
    fase_gen_C = (-1)*np.array([
  0.8,  5.3,  9.5, 15.9, 21.3, 28.3, 32.5, 34.6, 38.0,
 38.4, 39.2, 41.8, 42.3, 44.3, 44.2, 44.3, 47.2, 48.8, 54.3,
 60.3, 64.2, 66, 69.4, 80.6, 84.6, 86.5
])
    fase_gen_C_err = np.array([1.2, 0.6, 0.6, 0.5, 1.3, 1.0, 1.4, 1.0, 0.8,
 1.2, 0.8, 1, 1.0, 1.4, 1.2, 1.3, 0.8, 0.9, 1.2,
 0.4, 1.1, 1.3, 1.1, 1.3, 1.4, 1.0
])
    
    # Stime iniziali
    p0 = [9.85 *1000,  98 * 10 ** (-9)]  # R, C
    bounds = ([8000, 0], [10000000, 100000000])  # vincoli positivi realistici

    # Fit
    popt, pcov = curve_fit(
        modello_modulo, omega, fase_gen_C,
        sigma=fase_gen_C_err, absolute_sigma=True,
        p0=p0, bounds=bounds, maxfev=10000000000
    )
    R_fit, C_fit = popt
    R_err, C_err = np.sqrt(np.diag(pcov))

    # Chi² e p-value
    residuals = fase_gen_C - modello_modulo(omega, *popt)
    chi2_val = np.sum((residuals / fase_gen_C_err)**2)
    ndof = len(omega) - len(popt)
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

        # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(omega, fase_gen_C, yerr=fase_gen_C_err, fmt='o', label='Dati')
    omega_fit = np.linspace(min(omega), max(omega), 500)
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label=
             f'R = ({R_fit * 0.001:.1f} ± {R_err* 0.001:.1f}) kΩ\n'
             f'C = ({C_fit * 10 ** (8):.1f} ± {C_err * 10 ** (8):.1f}) $10^{{-8}}$ F\n')

    plt.xlabel('$\omega$  (rad/s)')
    plt.ylabel('fase (°)')
    
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
    plt.errorbar(omega, fase_gen_C, yerr=fase_gen_C_err, fmt='o', label='Dati')
    plt.plot(omega_fit, modello_modulo(omega_fit, *popt), 'r-', label=
             'Fit')
    plt.xscale('log')
    plt.xlabel('$\omega$  (rad/s)')
    plt.ylabel('fase (°)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.title('Scala logaritmica')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()