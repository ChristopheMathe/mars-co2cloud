import matplotlib.pyplot as plt
from numpy import arange, loadtxt, genfromtxt, array, append, asarray
from os import mkdir

ncol = 5
tab_freq = array([])
tab_radius = array([])
tab_coef = array([])

#list_filename = ['optprop_co2ice_ir_n50.dat', 'optprop_co2ice_vis_n50.dat',
#                 'optprop_iceir_n50.dat', 'optprop_icevis_n50.dat']

list_filename = ['optprop_co2_ir_fait_avec_co2_ref_index_IR_ddelta1.dat']
for f, filename in enumerate(list_filename):
    directory_save = filename[:-4] + '/'
    try:
        mkdir(directory_save)
    except FileExistsError:
        pass

    with open(file=filename, mode='r') as fread:
        fread.readline()                 # Commented line
        nfreq = int(fread.readline())    # Number of frequencies
        print(f"nfreq = {nfreq}")
        fread.readline()                 # Commented line
        nradius = int(fread.readline())  # Number of radius
        print(f"nradius = {nradius}")
        fread.readline()                 # Commented line
        nline_freq = int(nfreq/ncol) + min(nfreq % ncol, 1)
        print(f"nline_freq = {nline_freq}")
        for i in range(nline_freq):
            tab_freq = append(tab_freq, fread.readline().split())
        tab_freq = asarray(tab_freq, dtype=float)
        fread.readline()                 # Commented line
        nline = int(nradius/ncol)
        for i in range(nline):
            tab_radius = append(tab_radius, fread.readline().split())
        tab_radius = asarray(tab_radius, dtype=float)
        fread.readline()                 # Commented line
        fread.readline()                 # Commented line
        for j in range(nradius):
            print(f'radius = {tab_radius[j]}, {j+1}')
            for i in range(nline_freq):
                tab_coef = append(tab_coef, fread.readline().split())
                tab_coef = asarray(tab_coef, dtype=float)
            fread.readline()  # Commented line
            plt.figure(figsize=(11, 11))
            plt.title(f'Extinction coefficient, radius particle = {tab_radius[j]:.2e} m')
            plt.plot(tab_freq, tab_coef, color='black')
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Extinction coef (SI)')
            plt.xscale('log')
            plt.savefig(f'{directory_save}coef_ext_radius_{j+1}_{tab_radius[j]:.2e}m.png')
            tab_coef = array([])

    tab_freq = array([])
    tab_radius = array([])
    tab_coef = array([])
