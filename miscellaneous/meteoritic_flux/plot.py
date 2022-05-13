#!/bin/bash python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import interpolate
import math
from statistics import stdev

rmin_cld = 1e-9
rmax_cld = 5e-6
nbinco2_cld = 100

vrat_cld = np.exp(np.log(rmax_cld/rmin_cld) / float(nbinco2_cld-1) * 3.)
bins = np.zeros(nbinco2_cld)
bins[0] = rmin_cld
for i in range(1, nbinco2_cld):
	bins[i] = bins[i-1] * vrat_cld**(1./3.)

# CURRENT FILE IN GCM
filename = 'Meteo_flux_Plane.dat'
dataset = np.loadtxt(filename)

pressure = dataset[:130] # First 130 values are pressures

data_ref = dataset[130:].reshape(130, 100) # 13000 remains values are either 130 pressures x 100 bin
data_ref = np.ma.masked_values(data_ref, 0)

fig, ax = plt.subplots()
ax.set_title('Meteo_flux_plane.dat')
ctf = ax.pcolormesh(bins, pressure, data_ref, norm=LogNorm(), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('Inc')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_ylim(pressure[0], pressure[-1])
fig.savefig('meteoflux_plane.png', bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_title('Meteo_flux_plane en mass')
data_ref2 =  (4./3.) * math.pi * 2500 * data_ref * bins**3
ctf = ax.pcolormesh(bins, pressure, data_ref2, norm=LogNorm(), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('Inc')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_ylim(pressure[0], pressure[-1])
fig.savefig('meteoflux_plane_mass.png', bbox_inches='tight')



# FILE FROM JOHN PLANE
filename = 'Mars_dust.txt'
dataset = np.loadtxt(filename) # Altitude / km;    Bin size / m;  Differential number density (dN / dlog10(r))
data_altitude = np.unique(dataset[:, 0])
data_pressure = np.loadtxt('mars_pressure.txt')[:, 1]
data_bins = np.unique(dataset[:, 1])
data =  np.zeros((data_altitude.shape[0], data_bins.shape[0]))
#data = dataset[:, 2].reshape(130, 20) * 10**0.01 # to convert dn/dlog10(r) to dN
data = dataset[:, 2].reshape(130, 20) * 0.1 # to convert dn/dlog10(r) to dN

fig, ax = plt.subplots()
ax.set_title('Mars_dust.txt')
ctf = ax.pcolormesh(data_bins, data_pressure, data, norm=LogNorm(vmin=1e-10, vmax=1e3), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('dN')
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(pressure[0], pressure[-1])
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Pressure (Pa)")
fig.savefig('mars_dust.png', bbox_inches='tight')


# INTERPOLATION
f = interpolate.interp2d(data_bins, data_pressure, np.log10(data), kind='linear', fill_value=np.nan, bounds_error=False)
data_new = f(bins, pressure)
data_new = np.flip(10**(data_new), axis=0)
fig, ax = plt.subplots()
ax.set_title('Interpolation to GCM format')
data_new = np.where(np.isnan(data_new), 0, data_new)
pcm = ax.pcolormesh(bins, pressure, data_new, norm=LogNorm(vmin=1e-10, vmax=1e3), shading='auto')
cbar = fig.colorbar(pcm)
cbar.ax.set_title('dN')
cbar.ax.set_label('Number density')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(pressure[0], pressure[-1])
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Pressure (Pa)")
fig.savefig('interp.png', bbox_inches='tight')


fin = open("Meteo_flux_Plane_v2.dat", "w")
fin.write('#Pressures : 130 values\n')
for i in range(pressure.shape[0]):
	fin.write(f'{pressure[i]:.6e}\n')
fin.write("#End pressure, now the table:\n")
for i in range(pressure.shape[0]):
	for j in range(bins.shape[0]):
		fin.write(f'{data_new[i, j]:.6e}\n')
fin.close()

total_number = np.sum(data_new, axis=1)
print(np.sum(total_number))

rho_dust = 2500 # kg.m-3
total_mass = np.zeros(data_new.shape[0])
for j in range(data_new.shape[0]):
	for i in range(bins.shape[0]):
		total_mass[j] += (4./3.) * math.pi * rho_dust * data_new[j,i] * bins[i]**3

fin = np.savetxt(f"Meteo_flux_Plane_total_number_between_{bins[0]:.1e}_{bins[-1]:.1e}.dat", np.c_[pressure, total_number])
fin = np.savetxt(f"Meteo_flux_Plane_total_mass_between_{bins[0]:.1e}_{bins[-1]:.1e}.dat", np.c_[pressure, total_mass])


# CHECK NEW FILE INTERPOLATED
filename = 'Meteo_flux_Plane_v2.dat'
dataset = np.loadtxt(filename)

pressure = dataset[:130] # First 130 values are pressures

data_ref = dataset[130:].reshape(130, 100) # 13000 remains values are either 130 pressures x 100 bin
data_ref = np.ma.masked_values(data_ref, 0)

fig, ax = plt.subplots()
#ax.set_title('Meteo_flux_plane_v2.dat')
ctf = ax.pcolormesh(bins, pressure, data_ref, norm=LogNorm(vmin=1e-10, vmax=1e3), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('#.m$^{-3}$', horizontalalignment='center')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_ylim(pressure[0], pressure[-1])
fig.savefig('meteoflux_plane_v2.png', bbox_inches='tight')



# GET MASS from MARS_DUST.TXT
fig, ax = plt.subplots()
ax.set_title('Mars_dust.txt')
data =  (4./3.) * math.pi * 2500 * data * data_bins**3
ctf = ax.pcolormesh(data_bins, data_pressure, data, norm=LogNorm(), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('dN')
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(pressure[0], pressure[-1])
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Pressure (Pa)")
fig.savefig('mars_dust_mass.png', bbox_inches='tight')
