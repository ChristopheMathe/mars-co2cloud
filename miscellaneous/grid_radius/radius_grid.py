#!/bin/bash python3
from math import log, exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from statistics import stdev


def remplissage(nbin, N, rayon, grille, sigma):
	n_aer = np.zeros(shape=nbin)
	cst = 1. / (sqrt(2.) * sigma)
	for i in range(nbin):
		n_aer[i] = (N/2.) * (erf(log(grille[i+1]/rayon)*cst) - erf(log(grille[i]/rayon)*cst))
	return n_aer


# H2O MICROPHYSICS
rbmin_cld = 0.0001e-6
rbmax_cld = 1e-2
rmin_cld  = 0.1e-6
rmax_cld = 10e-6
nbin_cld = 5
rad_cld = np.zeros(nbin_cld)
rb_cld = np.zeros(nbin_cld+1) # boundary

vrat_cld = exp(log(rmax_cld/rmin_cld) / (nbin_cld-1) * 3.)
print(f"vrat_cld = {vrat_cld}")

rb_cld[0] = rbmin_cld
rad_cld[0] = rmin_cld
for i in range(nbin_cld-1):
	rad_cld[i+1] = rad_cld[i] * vrat_cld**(1./3.)

for i in range(nbin_cld):
	rb_cld[i+1] = ((2.*vrat_cld) / (vrat_cld+1.))**(1./3.) * rad_cld[i]
rb_cld[nbin_cld] = rbmax_cld
print(stdev(rb_cld))
rb_cld = np.log(rb_cld)

print(stdev(rb_cld))
nuice_sed = 0.1
sigma_ice = sqrt(log(1. + nuice_sed))
dev3 = 1. / (sqrt(2.) * sigma_ice)
print(sigma_ice, dev3)
print((1+nuice_sed), (1+nuice_sed)**(1.5), (1+nuice_sed)**(-1.5), exp(-1.5*nuice_sed**2))

print('================================')
# CO2 MICROPHYSICS
rbmin_cldco2 = 1e-10
rbmax_cldco2 = 2.e-4
rmin_cldco2 = 1e-9
rmax_cldco2 = 5e-5
nbin_cldco2 = 100
rad_cldco2 = np.zeros(nbin_cldco2)
rb_cldco2 = np.zeros(nbin_cldco2 + 1)

vrat_cldco2 = exp(log(rmax_cldco2/rmin_cldco2) / (nbin_cldco2-1) * 3.)
print(f"vrat_cldco2 = {vrat_cldco2}")
rb_cldco2[0] = rbmin_cldco2
rad_cldco2[0] = rmin_cldco2
for i in range(nbin_cldco2-1):
	rad_cldco2[i+1] = rad_cldco2[i] * vrat_cldco2**(1./3.)

for i in range(nbin_cldco2):
	rb_cldco2[i+1] = ((2.*vrat_cldco2) / (vrat_cldco2+1.))**(1./3.) * rad_cldco2[i]
rb_cldco2[nbin_cldco2] = rbmax_cldco2

reff = 1e-6
Nccnco2 = 1e8
nuiceco2_sed = 0.5
print(f'Pour un rayon effectif de {reff:.1e} m')
print(f'Standard deviation sigma = {stdev(rb_cldco2):.1e}')
print(f'Variance effective = {exp(stdev(rb_cldco2)**2) - 1:.1e}')
print(f'Dans le GCM actuel')
print(f'\tvariance effective = 0.2')
sigma_iceco2 = sqrt(log(1. + nuiceco2_sed))
r0 = reff/exp(5*sigma_iceco2**2/2)
print(f'rm = {r0*1e6*exp(-sigma_iceco2**2):f}')
print(f'r0 = {r0*1e6:f}')
print(f'reff = {reff*1e6:f}')

n_aer = remplissage(nbin=nbin_cldco2, N=Nccnco2, rayon=r0, grille=rb_cldco2,
					sigma=sigma_iceco2)


# FIGURE
fig, ax = plt.subplots(ncols=1, figsize=(11,11))
ax.plot(rad_cldco2*1e6, n_aer/Nccnco2, color='black')

#ax.vlines(rb_cld*1e6, 0.5, 1, color='blue', ls='-', label='H2O - mean rad')
#ax.vlines(rad_cld*1e6, 0.5, 1, color='blue', ls='--', label='H2O - boundaries')
ax.vlines(rb_cldco2*1e6, 0, 0.5, color='red', ls='-', label='CO2 - mean rad')
ax.vlines(rad_cldco2*1e6, 0, 0.5, color='red', ls='--', label='CO2 - boundaries')
ax.vlines(r0*1e6, 0, 1, color='black')
ax.vlines(reff*1e6, 0, 1, color='cyan')
ax.vlines(r0*1e6*exp(-sigma_iceco2**2), 0, 1, ls='--', color='blue')
ax.vlines(r0*1e6*(1+nuiceco2_sed), 0, 1, color='green')
ax.legend(loc='best')
ax.set_xlabel('Size (m)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-5, )
fig.savefig('test.png')
#fig.savefig(f'grid_radius_gcm_nbincld_co2_{nbin_cldco2:.0f}_rbmin_cldco2_{rbmin_cldco2:.0e}_rmin_cldco2_'
#			f'{rmin_cldco2:.0e}.png')
