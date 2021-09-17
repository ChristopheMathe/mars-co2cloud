#!/bin/bash python3
from packages.lib_function import *
from packages.ncdump import *
from netCDF4 import Dataset
from matplotlib.colors import DivergingNorm, LinearSegmentedColormap
from numpy.ma import masked_inside
from os import listdir
from pylab import *
import matplotlib.pyplot as plt
from sys import exit


class NonLinearColormap(LinearSegmentedColormap):
    """A nonlinear colormap"""
    name_cmap = 'jet'

    def __init__(self, cmap, levels, name_cmap, segment_data):
        """
        """
        super().__init__(name_cmap, segment_data)
        self.cmap = cmap
        self.name = name_cmap
        self.monochrome = self.cmap.monochrome
        self.levels = asarray(levels, dtype='float64')
        self._x = self.levels - self.levels.min()
        self._x /= self._x.max()
        self._y = linspace(0, 1, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)


def compute_error(data_ref, data):
    shape_data = data.shape
    delta = zeros(shape_data)

    for i in range(shape_data[0]):
        for j in range(shape_data[1]):
            if data_ref[i, j] == 0:
                delta[i, j] = 0
            else:
                delta[i, j] = (data[i, j] - data_ref[i, j]) * 100 / data_ref[i, j]

    return delta


def extract_data(data, mvals):
    if mvals is True:
        data_h2o_ice_s = data.variables['h2o_ice_s'][:, :, :]
        data_icetot = data.variables['icetot'][:, :, :]
        data_tau_tes = data.variables['tauTES'][:, :, :]
        data_mtot = data.variables['mtot'][:, :, :]
        data_ps = data.variables['ps'][:, :, :]
        data_tsurf = data.variables['tsurf'][:, :, :]
        data_co2ice = data.variables['co2ice'][:, :, :]
    else:
        data_time = data.variables['Time'][:]
        data_local_time, idx, stats_file = check_local_time(data_time=data_time, selected_time=14)
        data_h2o_ice_s = data.variables['h2o_ice_s'][idx::len(data_local_time), :, :]
        data_icetot = data.variables['icetot'][idx::len(data_local_time), :, :]
        data_tau_tes = data.variables['tauTES'][idx::len(data_local_time), :, :]
        data_mtot = data.variables['mtot'][idx::len(data_local_time), :, :]
        data_ps = data.variables['ps'][idx::len(data_local_time), :, :]
        data_tsurf = data.variables['tsurf'][idx::len(data_local_time), :, :]
        data_co2ice = data.variables['co2ice'][idx::len(data_local_time), :, :]

    # correct data
    data_h2o_ice_s = correction_value(data=data_h2o_ice_s, operator='inf', threshold=1e-13)
    data_icetot = correction_value(data=data_icetot, operator='inf', threshold=1e-13)
    data_tau_tes = correction_value(data=data_tau_tes, operator='inf', threshold=1e-13)
    data_mtot = correction_value(data=data_mtot, operator='inf', threshold=1e-13)
    data_ps = correction_value(data=data_ps, operator='inf', threshold=1e-13)
    data_tsurf = correction_value(data=data_tsurf, operator='inf', threshold=1e-13)
    data_co2ice = correction_value(data=data_co2ice, operator='inf', threshold=1e-13)

    # compute zonal mean
    data_h2o_ice_s = mean(data_h2o_ice_s, axis=2)
    data_icetot = mean(data_icetot, axis=2)
    data_tau_tes = mean(data_tau_tes, axis=2)
    data_mtot = mean(data_mtot, axis=2)
    data_ps = mean(data_ps, axis=2)
    data_tsurf = mean(data_tsurf, axis=2)
    data_co2ice = mean(data_co2ice, axis=2)

    # rotate data
    data_h2o_ice_s, data_icetot, data_tau_tes, data_mtot, data_ps, data_tsurf, data_co2ice = rotate_data(
        data_h2o_ice_s, data_icetot, data_tau_tes, data_mtot, data_ps, data_tsurf, data_co2ice, do_flip=True)
    return data_h2o_ice_s, data_icetot, data_tau_tes, data_mtot, data_ps, data_tsurf, data_co2ice


def plot_figure(data_ref, data, delta, levels, sup_title, unit, fmt, ndx, data_latitude, save_name):

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey='col', figsize=(11, 8))
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.2)
    fig.suptitle(sup_title)

    dim1, dim2 = where(data_ref == 0)
    data_ref[dim1, dim2] = None

    dim1, dim2 = where(data == 0)
    data[dim1, dim2] = None

    dim1, dim2 = where(delta == 0)
    delta[dim1, dim2] = None

    ax[0].set_title('Simu ref (MV)')
    pc1 = ax[0].contourf(data_ref, levels=levels, cmap='plasma')
    ax[1].set_facecolor('white')

    ax[1].set_title('Our simu')
    ax[1].contourf(data, levels=levels, cmap='plasma')
    ax[1].set_facecolor('white')

    ax[2].set_title('Relative change (simu - ref)')
    pc2 = ax[2].contourf(delta, norm=DivergingNorm(vmin=-100, vcenter=0, vmax=100), levels=arange(-10, 12, 2) * 10,
                         cmap='seismic')

    ax[0].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[0].set_yticklabels(labels=data_latitude[::6])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_xticklabels(labels=[0, 90, 180, 270, 359])
    ax[1].set_xticklabels(labels=['', 90, 180, 270, 359])
    ax[2].set_xticklabels(labels=['', 90, 180, 270, 359])

    pos1 = ax[0].get_position()
    pos3 = ax[2].get_position()
    cbar_ax1 = fig.add_axes([pos1.x0 + 0.02, 0.05, pos3.x0 - pos1.x0 - 0.04, 0.03])
    cbar1 = fig.colorbar(pc1, cax=cbar_ax1, orientation="horizontal", format=fmt)
    cbar1.ax.set_title(unit)

    cbar_ax2 = fig.add_axes([pos3.x0 + 0.02, 0.05, pos3.x1 - pos3.x0 - 0.04, 0.03])
    cbar2 = fig.colorbar(pc2, cax=cbar_ax2, orientation="horizontal")
    cbar2.ax.set_title('%')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.15, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.close(fig)


def main():
    from numpy import min, max

    # Data from Margaux Vals
    data_ref = Dataset('../simu_ref_cycle_eau_mvals/simu_ref_cycle_eau_mvals/concat_vars_3D_LT_14h_Ls.nc', "r",
                       format="NETCDF4")
    data_ref_h2o_ice_s, data_ref_icetot, data_ref_tau_tes, data_ref_mtot, data_ref_ps, data_ref_tsurf, data_ref_co2ice \
        = extract_data(data_ref, mvals=True)
    data_time_mvals = data_ref.variables['Time']

    # My data
    files = listdir('.')

    directory_store = None
    try:
        directory_store = [file for file in files if 'occigen' in file][0] + '/'
    except not directory_store:
        directory_store = None

    if directory_store is None:
        directory_store = ''
    else:
        files = listdir(directory_store)

    filename = getfilename(files)
    filename = directory_store + filename
    data_3d = Dataset(filename, "r", format="NETCDF4")
    data_h2o_ice_s, data_icetot, data_tau_tes, data_mtot, data_ps, data_tsurf, data_co2ice = extract_data(data_3d,
                                                                                                          mvals=False)

    # Get ndx and axis_ls
    data_latitude = data_3d.variables['latitude']
    data_latitude = data_latitude[::-1]
    data_time = data_3d.variables['Time']
    data_localtime, idx, stats = check_local_time(data_time, selected_time=14)
    data_time = data_time[idx::len(data_localtime)]
    ndx, axis_ls, ls_lin = get_ls_index(data_time)
    if ls_lin:
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        data_ls = data_ls[idx::len(data_localtime)]
        data_h2o_ice_s, data_time = linearize_ls(data=data_h2o_ice_s, data_ls=data_ls)
        data_icetot, data_time = linearize_ls(data=data_icetot, data_ls=data_ls)
        data_tau_tes, data_time = linearize_ls(data=data_tau_tes, data_ls=data_ls)
        data_mtot, data_time = linearize_ls(data=data_mtot, data_ls=data_ls)
        data_ps, data_time = linearize_ls(data=data_ps, data_ls=data_ls)
        data_tsurf, data_time = linearize_ls(data=data_tsurf, data_ls=data_ls)
        data_co2ice, data_time = linearize_ls(data=data_co2ice, data_ls=data_ls)
        ndx, axis_ls, ls_lin = get_ls_index(data_time)

    # Compute the difference between Margaux and me
    delta_h2o_ice_s = compute_error(data_ref_h2o_ice_s, data_h2o_ice_s)
    delta_icetot = compute_error(data_ref_icetot, data_icetot)
    delta_tau_tes = compute_error(data_ref_tau_tes, data_tau_tes)
    delta_mtot = compute_error(data_ref_mtot, data_mtot)
    delta_ps = compute_error(data_ref_ps, data_ps)
    delta_tsurf = compute_error(data_ref_tsurf, data_tsurf)
    delta_co2ice = compute_error(data_ref_co2ice, data_co2ice)

    # Plot: tsurf
    print(f'tsurf: min = {min(data_tsurf):.0e}, max = {max(data_tsurf):.0e}')
    plot_figure(data_ref_tsurf, data_tsurf, delta_tsurf, levels=arange(140, 270, 10),
                sup_title='Zonal mean of surface temperature', unit='K', fmt='%.2f', ndx=ndx,
                data_latitude=data_latitude, save_name='check_water_cycle_relative_error_tsurf')

    # Plot: ps
    print(f'ps: min = {min(data_ps):.0e}, max = {max(data_ps):.0e}')
    plot_figure(data_ref_ps, data_ps, delta_ps, levels=arange(300, 1200, 100),
                sup_title='Zonal mean of surface pressure', unit='Pa', fmt='%d', ndx=ndx, data_latitude=data_latitude,
                save_name='check_water_cycle_relative_error_ps')

    # Plot: mtot
    print(f'mtot: min = {min(data_mtot):.0e}, max = {max(data_mtot):.0e}')
    plot_figure(data_ref_mtot, data_mtot, delta_mtot, levels=None,
                sup_title='Zonal mean of total atmospheric mass', unit='kg/m$^2$', fmt='%.1e', ndx=ndx,
                data_latitude=data_latitude, save_name='check_water_cycle_relative_error_mtot')

    # Plot: icetot
    print(f'icetot: min = {min(data_icetot):.0e}, max = {max(data_icetot):.0e}')
    plot_figure(data_ref_icetot, data_icetot, delta_icetot, levels=linspace(0, 0.05, 10),
                sup_title='Zonal mean of total water ice mass', unit='kg/m$^2$', fmt='%.1e', ndx=ndx,
                data_latitude=data_latitude, save_name='check_water_cycle_relative_error_icetot')

    # Plot: h2o_ice_s
    print(f'h2o_ice_s: min = {min(data_h2o_ice_s):.0e}, max = {max(data_h2o_ice_s):.0e}')
    plot_figure(data_ref_h2o_ice_s, data_h2o_ice_s, delta_h2o_ice_s, levels=arange(0, 9, 1),
                sup_title='Zonal mean of surface water ice', unit='kg/m$^2$', fmt='%d', ndx=ndx,
                data_latitude=data_latitude, save_name='check_water_cycle_relative_error_h2o_ice_s')

    # Plot: tauTES
    print(f'tauTES: min = {min(data_tau_tes):.0e}, max = {max(data_tau_tes):.0e}')
    plot_figure(data_ref_tau_tes, data_tau_tes, delta_tau_tes, levels=arange(0, 7, 1),
                sup_title='Zonal mean of opacity at 825 cm$^{-1}$', unit='', fmt='%.d', ndx=ndx,
                data_latitude=data_latitude, save_name='check_water_cycle_relative_error_tauTES')

    # Plot: co2ice
    print(f'co2ice: min = {min(data_co2ice):.0e}, max = {max(data_co2ice):.0e}')
    plot_figure(data_ref_co2ice, data_co2ice, delta_co2ice, levels=None, sup_title='CO$_2$ ice at the surface',
                unit='kg/m$^2$', fmt='%.1e', ndx=ndx, data_latitude=data_latitude,
                save_name='check_water_cycle_relative_error_co2ice')

    # Compare with TES obs
    # To compare to Figure 2 of Navarro2014
    # Do water_vapor zonal mean
    # Do tau_ice  zonal mean
    directory_tes = '/home/mathe/Documents/owncloud/GCM/TES/'

    data_tes = Dataset(directory_tes + 'TES.SeasonalClimatology.nc', "r", format="NETCDF4")
    data_tes_tauice = data_tes.variables['tauice']
    data_tes_h2o_vap = data_tes.variables['water']
    data_tes_tsurf_day = data_tes.variables['Tsurf_day']

    data_tes_time = data_tes.variables['time']
    data_tes_latitude = data_tes.variables['latitude']
    zonal_mean_tes_tauice = mean(data_tes_tauice, axis=2).T
    zonal_mean_tes_h2o_vap = mean(data_tes_h2o_vap, axis=2).T
    zonal_mean_tes_tsurf_day = mean(data_tes_tsurf_day, axis=2).T

    cmap_lin = get_cmap('jet')
    levels1 = [0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.75, 2.0, 4.0]
    cmap_non_lin1 = NonLinearColormap(cmap=cmap_lin, levels=levels1, name_cmap='jet', segment_data=100)

    levels2 = [0, 5, 15, 30, 50, 70, 90, 130]
    cmap_non_lin2 = NonLinearColormap(cmap=cmap_lin, levels=levels2, name_cmap='jet', segment_data=100)

    idx1 = abs(data_tes_time[:] - 360).argmin()
    idx2 = abs(data_tes_time[:] - 720).argmin()

    # FIRST PLOTS: TAU-TES
    gridspec = {'width_ratios': [1, 1, 1, 0.1]}
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 8), gridspec_kw=gridspec)
    fig.subplots_adjust(wspace=0.05)
    ax[0].set_title('TES', fontsize=18)
    ctf = ax[0].contourf(data_tes_time[idx1:idx2] - 360, data_tes_latitude[:], zonal_mean_tes_tauice[:, idx1:idx2],
                         levels=levels1, cmap=cmap_non_lin1)

    ax[1].set_title('M. VALS', fontsize=18)
    ax[1].contourf(data_time_mvals[:], data_latitude[:], data_ref_tau_tes, levels=levels1, cmap=cmap_non_lin1)

    ax[2].set_title('Our', fontsize=18)
    ax[2].contourf(data_time[:], data_latitude[:], data_tau_tes, levels=levels1, cmap=cmap_non_lin1)

    cbar = fig.colorbar(ctf, cax=ax[3])
    cbar.set_label('Cloud opacity at 825 cm$^{-1}$', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax[0].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[0].set_yticklabels(labels=[-90, -60, -30, 0, 30, 60, 90], fontsize=18)
    ax[1].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_yticklabels(labels='')
    ax[2].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[2].set_yticklabels(labels='')

    ax[0].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[0].set_xticklabels(labels=[0, 90, 180, 270, 360], fontsize=18)
    ax[1].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[1].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)
    ax[2].set_xticks(ticks=ndx)
    ax[2].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)

    ax[0].grid(color='black')
    ax[1].grid(color='black')
    ax[2].grid(color='black')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.05, 'Solar longitude (°)', ha='center', va='center', fontsize=18)
    fig.savefig('check_water_cycle_tes_mvals_me_tauice.png', bbox_inches='tight')
    plt.close(fig)

    # PLOT 2 : H2O_VAP
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 8), gridspec_kw=gridspec)
    fig.subplots_adjust(wspace=0.05)
    ax[0].set_title('TES', fontsize=18)
    ctf = ax[0].contourf(data_tes_time[idx1:idx2] - 360, data_tes_latitude[:], zonal_mean_tes_h2o_vap[:, idx1:idx2],
                         levels=levels2, cmap=cmap_non_lin2)
    ctr = ax[0].contour(data_tes_time[idx1:idx2] - 360, data_tes_latitude[:], zonal_mean_tes_h2o_vap[:, idx1:idx2],
                        levels=levels2, colors='white', linewidths=1)
    ax[0].clabel(ctr, inline=1, fontsize=12, fmt='%d')

    ax[1].set_title('M. VALS', fontsize=18)
    ax[1].contourf(data_time_mvals[:], data_latitude[:], data_ref_mtot * 1e3, levels=levels2, cmap=cmap_non_lin2)
    ctr1 = ax[1].contour(data_time_mvals[:], data_latitude[:], data_ref_mtot * 1e3, levels=levels2, colors='white',
                         linewidths=1)
    ax[1].clabel(ctr1, inline=1, fontsize=12, fmt='%d')

    ax[2].set_title('Our', fontsize=18)
    ax[2].contourf(data_time[:], data_latitude[:], data_mtot * 1e3, levels=levels2, cmap=cmap_non_lin2)
    ctr2 = ax[2].contour(data_time[:], data_latitude[:], data_mtot * 1e3, levels=levels2, colors='white', linewidths=1)
    ax[2].clabel(ctr2, inline=1, fontsize=12, fmt='%d')

    cbar = fig.colorbar(ctf, cax=ax[3])
    cbar.set_label('pr.µm', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax[0].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[0].set_yticklabels(labels=[-90, -60, -30, 0, 30, 60, 90], fontsize=18)
    ax[1].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_yticklabels(labels='')
    ax[2].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[2].set_yticklabels(labels='')

    ax[0].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[0].set_xticklabels(labels=[0, 90, 180, 270, 360], fontsize=18)
    ax[1].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[1].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)
    ax[2].set_xticks(ticks=ndx)
    ax[2].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)

    ax[0].grid(color='black')
    ax[1].grid(color='black')
    ax[2].grid(color='black')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.05, 'Solar longitude (°)', ha='center', va='center', fontsize=18)
    plt.savefig('check_water_cycle_tes_mvals_me_h2o_vap.png')
    plt.close()

    # PLOT 3 : Tsurf day
    levels3 = arange(125, 350, 25)
    cmap3 = 'inferno'
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 8), gridspec_kw=gridspec)
    fig.subplots_adjust(wspace=0.05)
    ax[0].set_title('TES', fontsize=18)
    ctf = ax[0].contourf(data_tes_time[idx1:idx2] - 360, data_tes_latitude[:], zonal_mean_tes_tsurf_day[:, idx1:idx2],
                         levels=levels3, cmap=cmap3)
    ctr = ax[0].contour(data_tes_time[idx1:idx2] - 360, data_tes_latitude[:], zonal_mean_tes_tsurf_day[:, idx1:idx2],
                        levels=levels3, colors='white', linewidths=1)
    ax[0].clabel(ctr, inline=1, fontsize=12, fmt='%d')

    ax[1].set_title('M. VALS', fontsize=18)
    ax[1].contourf(data_time_mvals[:], data_latitude[:], data_ref_tsurf, levels=levels3, cmap=cmap3)
    ctr1 = ax[1].contour(data_time_mvals[:], data_latitude[:], data_ref_tsurf, levels=levels3, colors='white',
                         linewidths=1)
    ax[1].clabel(ctr1, inline=1, fontsize=12, fmt='%d')

    ax[2].set_title('Our', fontsize=18)
    ax[2].contourf(data_time[:], data_latitude[:], data_tsurf, levels=levels3, cmap=cmap3)
    ctr2 = ax[2].contour(data_time[:], data_latitude[:], data_tsurf, levels=levels3, colors='white', linewidths=1)
    ax[2].clabel(ctr2, inline=1, fontsize=12, fmt='%d')

    cbar = fig.colorbar(ctf, cax=ax[3])
    cbar.set_label('K', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax[0].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[0].set_yticklabels(labels=[-90, -60, -30, 0, 30, 60, 90], fontsize=18)
    ax[1].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_yticklabels(labels='')
    ax[2].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[2].set_yticklabels(labels='')

    ax[0].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[0].set_xticklabels(labels=[0, 90, 180, 270, 360], fontsize=18)
    ax[1].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[1].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)
    ax[2].set_xticks(ticks=ndx)
    ax[2].set_xticklabels(labels=['', 90, 180, 270, 360], fontsize=18)

    ax[0].grid(color='black')
    ax[1].grid(color='black')
    ax[2].grid(color='black')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.05, 'Solar longitude (°)', ha='center', va='center', fontsize=18)
    plt.savefig('check_water_cycle_tes_mvals_me_tsurf_day.png')
    plt.close()

    exit()
    # Compare with PFS obs
    # To compare to Figure 2 of Navarro2014
    # Do water_vapor zonal mean
    # Do tau_ice  zonal mean
    directory_pfs = '/home/mathe/Documents/owncloud/GCM/PFS/PFS_dataset_20793/PFS_data/PFS_data.nc'

    data_pfs = Dataset(directory_pfs, "r", format="NETCDF4")
    data_pfs_tauice = data_pfs.variables['ice']

    data_pfs_time = data_pfs.variables['Time']
    data_pfs_latitude = data_pfs.variables['latitude']

    max_martian_year = int(ceil(data_pfs_time[-1] / 360))
    martian_year = arange(0, 360 * (max_martian_year + 1), 360)

    zonal_mean = zeros((max_martian_year, 360, data_pfs_latitude.shape[0]))

    for j in range(martian_year.shape[0] - 1):
        print(f'{j / martian_year.shape[0] * 100}%')
        idx1 = abs(data_pfs_time[:] - martian_year[j]).argmin()
        idx2 = abs(data_pfs_time[:] - martian_year[j + 1]).argmin()
        one_year_time = data_pfs_time[idx1:idx2 + 1] - 360 * j
        print('\t', one_year_time[0], one_year_time[-1])
        one_year_data = data_pfs_tauice[idx1:idx2 + 1, :, :]
        one_year_data = correction_value(data=one_year_data, operator='inf', threshold=1e-13)
        one_year_data = mean(one_year_data, axis=2)
        one_year_data = correction_value(data=one_year_data, operator='inf', threshold=1e-13)
        for i in range(359):
            mask = masked_inside(one_year_time, i, i + 1)
            if mask.mask.any():
                for lat in range(data_pfs_latitude[:].shape[0]):
                    a = mean(one_year_data[mask.mask, lat])
                    if math.isnan(a):
                        print(one_year_data[mask.mask, lat])
                    else:
                        zonal_mean[j, i, lat] = a
            del mask
        del one_year_time, one_year_data

    zonal_mean = correction_value(data=zonal_mean, operator='inf', threshold=1e-13)
    zonal_mean_final = mean(zonal_mean, axis=0)

    # FIRST PLOTS: TAU-TES
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))
    fig.subplots_adjust(wspace=0.1)
    ax[0].set_title('PFS')
    ctf = ax[0].contourf(arange(360), data_pfs_latitude[:], zonal_mean_final.T, levels=levels1, cmap=cmap_non_lin1)
    cbar = fig.colorbar(ctf)
    cbar.set_label('Cloud opacity at 825 cm$^{-1}$')

    ax[1].set_title('M. VALS')
    ax[1].contourf(data_time_mvals[:], data_latitude[:], data_ref_tau_tes, levels=levels1, cmap=cmap_non_lin1)

    ax[2].set_title('Our')
    ax[2].contourf(data_time[:], data_latitude[:], data_tau_tes, levels=levels1, cmap=cmap_non_lin1)

    ax[0].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[0].set_yticklabels(labels=[-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_yticklabels(labels='')
    ax[2].set_yticks(ticks=[-90, -60, -30, 0, 30, 60, 90])
    ax[2].set_yticklabels(labels='')

    ax[0].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[0].set_xticklabels(labels=[0, 90, 180, 270, 360])
    ax[1].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[1].set_xticklabels(labels=['', 90, 180, 270, 360])
    ax[2].set_xticks(ticks=[0, 90, 180, 270, 360])
    ax[2].set_xticklabels(labels=['', 90, 180, 270, 360])

    ax[0].grid(color='black')
    ax[1].grid(color='black')
    ax[2].grid(color='black')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.05, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
    fig.savefig('check_water_cycle_pfs_mvals_me_tauice.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
