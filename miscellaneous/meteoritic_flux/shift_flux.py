import matplotlib.pyplot as plt
from numpy import abs, loadtxt, ma, sum, zeros
from statistics import stdev
from math import log, exp, pi
from matplotlib.colors import LogNorm

rmin_cld = 1e-9
rmax_cld = 5e-6
nbinco2_cld = 100
shift_value = 5e-7

vrat_cld = exp(log(rmax_cld / rmin_cld) / float(nbinco2_cld - 1) * 3.)
bins = zeros(nbinco2_cld)
bins[0] = rmin_cld
for i in range(1, nbinco2_cld):
    bins[i] = bins[i - 1] * vrat_cld ** (1. / 3.)

idx = abs(bins - shift_value).argmin()
filename = 'Meteo_flux_Plane_v2.dat'
dataset = loadtxt(filename)
pressure = dataset[:130]  # First 130 values are pressures
data_ref = dataset[130:].reshape(130, 100)  # 13000 remains values are either 130 pressures x 100 bin
data_new = zeros(shape=data_ref.shape)

print(f'Total number of particules = {sum(data_ref)}')
for p, value_p in enumerate(pressure):
    print(f'\tAt pressure = {value_p:.2e} Pa')
    print(f'\t\tNumber of particules = {sum(data_ref[p, :]):.2e}, standard deviation = {stdev(data_ref[p, :]):.2e}')
    a = zeros(shape=100)
    a[idx:] = data_ref[p, :-idx]
    data_new[p, :] = a

fin = open(f"Meteo_flux_Plane_v2_shifted_{str(shift_value):s}.dat", "w")
fin.write('#Pressures : 130 values\n')
for i in range(pressure.shape[0]):
    fin.write(f'{pressure[i]:.6e}\n')
fin.write("#End pressure, now the table:\n")
for i in range(pressure.shape[0]):
    for j in range(bins.shape[0]):
        fin.write(f'{data_new[i, j]:.6e}\n')
fin.close()

data_new = ma.masked_values(data_new, 0)
fig, ax = plt.subplots()
ctf = ax.pcolormesh(bins, pressure, data_new, norm=LogNorm(vmin=1e-10, vmax=1e3), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('dN')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_ylim(pressure[0], pressure[-1])
plt.savefig(f'meteoflux_plane_v2_shifted_{str(shift_value):s}.png', bbox_inches='tight')

rho_dust = 2500  # kg.m-3
total_mass = zeros(shape=data_new.shape[0])
distrib = zeros(shape=data_new.shape)
for j in range(data_new.shape[0]):
    for i in range(bins.shape[0]):
        total_mass[j] += (4. / 3.) * pi * rho_dust * data_ref[j, i] * bins[i] ** 3
    print(f'Total mass at {pressure[j]:.2e} Pa: {total_mass[j]:.2e} kg')
    ratio = data_new[j,:]/sum(data_new[j,:])
    ratio_mass = ratio * total_mass[j]
    for i in range(bins.shape[0]):
        distrib[j,i] = ratio_mass[i] * (3. / (4. * pi * rho_dust * bins[i]**3))

fig, ax = plt.subplots()
ctf = ax.pcolormesh(bins, pressure, distrib, norm=LogNorm(vmin=1e-10, vmax=1e3), shading='auto')
cb = fig.colorbar(ctf)
cb.ax.set_title('dN')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_ylim(pressure[0], pressure[-1])
plt.savefig(f'meteoflux_plane_v2_shifted_{str(shift_value):s}_massconserv.png', bbox_inches='tight')

fin = open(f"Meteo_flux_Plane_v2_shifted_{str(shift_value):s}_massconserv.dat", "w")
fin.write('#Pressures : 130 values\n')
for i in range(pressure.shape[0]):
    fin.write(f'{pressure[i]:.6e}\n')
fin.write("#End pressure, now the table:\n")
for i in range(pressure.shape[0]):
    for j in range(bins.shape[0]):
        fin.write(f'{distrib[i, j]:.6e}\n')
fin.close()