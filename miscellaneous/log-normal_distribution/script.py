from math import pi, sqrt
from scipy.special import erf
from numpy import log, exp, array, arange
import matplotlib.pyplot as plt
from statistics import stdev


vecteur_x = arange(0.1, 3.1, 0.1)
sigma = array([1./8., 1./6., 1./4., 0.5, 1, 1.5, 10])
mu = 0


plt.figure()
for s, sig in enumerate(sigma):
    density_proba = (1. / (vecteur_x * sig * sqrt(2*pi))) * exp(- (log(vecteur_x) - mu)**2 / (2*sig**2))
    print(f'{sig:6.3f}, sigma = {stdev(density_proba):.3e}, nu = {exp(stdev(density_proba)**2) - 1:.3e}, '
          f'mu_eff = {exp(stdev(density_proba)**2*2.5)}')
    plt.plot(vecteur_x, density_proba, label=sig)


plt.legend(loc=0)
plt.savefig('densite_de_probalite.png')


