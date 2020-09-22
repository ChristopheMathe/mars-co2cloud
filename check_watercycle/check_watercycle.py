#!/bin/bash python3
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm, LogNorm
from numpy import mean, abs, min, max, zeros, where, ones, concatenate, flip, arange, unravel_index, argmax, array, \
    count_nonzero, diff, std, savetxt, c_, append, loadtxt, asarray, power, linspace, logspace


def correction_data(data, threshold):
    for i in range(data.shape[0]):
        dim2, dim3 = where(data[i, :, :] <= threshold)
        data[i, dim2, dim3] = 0
    return data


def extract_data(data, zonal_mean=None):
    data_h2o_ice_s = data.variables['h2o_ice_s'][:,:,:]
    data_icetot = data.variables['icetot'][:,:,:]
    data_tauTES = data.variables['tauTES'][:,:,:]
    data_mtot = data.variables['mtot'][:,:,:]
    data_ps = data.variables['ps'][:,:,:]
    data_tsurf = data.variables['tsurf'][:,:,:]

    data_h2o_ice_s = correction_data(data_h2o_ice_s, threshold=1e-5)

    data_icetot = correction_data(data_icetot, threshold=1e-5)

    data_tauTES = correction_data(data_tauTES, threshold=1e-5)

    data_mtot = correction_data(data_mtot, threshold=1e-5)

    if zonal_mean:
        data_h2o_ice_s = flip(mean(data_h2o_ice_s, axis=2).T, axis=0)
        data_icetot = flip(mean(data_icetot, axis=2).T, axis=0)
        data_tauTES = flip(mean(data_tauTES, axis=2).T, axis=0)
        data_mtot = flip(mean(data_mtot, axis=2).T, axis=0)
        data_ps = flip(mean(data_ps, axis=2).T, axis=0)
        data_tsurf = flip(mean(data_tsurf, axis=2).T, axis=0)

    return data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf


def compute_error(data_ref, data):
    shape_data = data.shape
    delta = zeros((shape_data))

    for i in range(shape_data[0]):
        for j in range(shape_data[1]):
            if data_ref[i,j] == 0:
                delta[i,j] == 0
            else:
                delta[i,j] = (data[i,j] - data_ref[i,j])*100 / data_ref[i,j]

    return delta


def linearize_ls(data, dim_time, dim_latitude, interp_time):
    from numpy import arange
    from scipy.interpolate import interp2d

    # interpolation to get linear Ls
    f = interp2d(x=arange(dim_time), y=arange(dim_latitude), z=data, kind='linear')

    data = f(interp_time, arange(dim_latitude))

    return data


def plot_figure(data_ref, data, delta, levels, title, unit, format, ndx, data_latitude, savename):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.2)
    fig.suptitle(title)

    dim1, dim2 = where(data_ref == 0)
    data_ref[dim1,dim2] = None

    dim1, dim2 = where(data == 0)
    data[dim1,dim2] = None

    dim1, dim2 = where(delta == 0)
    delta[dim1,dim2] = None

    axes[0].set_title('Simu ref (MV)')
    pc1 = axes[0].contourf(data_ref, levels=levels, cmap='plasma')
    axes[1].set_facecolor('white')

    axes[1].set_title('Our simu')
    axes[1].contourf(data, levels=levels, cmap='plasma')
    axes[1].set_facecolor('white')

    axes[2].set_title('Relative change (simu - ref)')
    pc2 = axes[2].contourf(delta, norm=DivergingNorm(vmin=-100, vcenter=0, vmax=100), levels=arange(-10, 12, 2)*10,
                           cmap='seismic')

    axes[0].set_yticks(ticks=arange(0, len(data_latitude), 6))
    axes[0].set_yticklabels(labels=data_latitude[::6])
    axes[0].set_xticks(ticks=ndx)
    axes[0].set_xticklabels(labels=[0, 90, 180, 270, 360])
    axes[1].set_xticklabels(labels=['', 90, 180, 270, 360])
    axes[2].set_xticklabels(labels=['', 90, 180, 270, 360])

    pos1 = axes[0].get_position()
    pos2 = axes[1].get_position()
    pos3 = axes[2].get_position()
    cbar_ax1 = fig.add_axes([pos1.x0 + 0.02, 0.05, pos3.x0 - pos1.x0 - 0.04, 0.03])
    cbar1 = fig.colorbar(pc1, cax=cbar_ax1, orientation="horizontal", format=format)
    cbar1.ax.set_title(unit)

    cbar_ax2 = fig.add_axes([pos3.x0 + 0.02, 0.05, pos3.x1 - pos3.x0 - 0.04, 0.03])
    cbar2 = fig.colorbar(pc2, cax=cbar_ax2, orientation="horizontal")
    cbar2.ax.set_title('%')

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.15, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
    plt.savefig(savename+'.png', bbox_inches='tight')
    plt.close(fig)


def linear_grid_ls(data):
    from numpy import linspace, searchsorted

    axis_ls = linspace(0, 360, data.shape[0])
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data, axis_ls)

    return interp_time, axis_ls, ndx


def main():
    # Data from Margaux Vals
    data_ref = Dataset('/home/mathe/Documents/runs_analysis/simu_ref_cycle_eau_mvals/concat.nc', "r", format="NETCDF4")
    data_ref_h2o_ice_s, data_ref_icetot, data_ref_tauTES, data_ref_mtot, data_ref_ps, data_ref_tsurf = \
                                                                                extract_data(data_ref, zonal_mean=True)

    data = Dataset('concat_ls_vars_cycle_h2o.nc', "r", format="NETCDF4")
    data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf = extract_data(data, zonal_mean=True)

    data_latitude = data.variables['latitude']
    data_time = data.variables['Time']
    interp_time, axis_ls, ndx = linear_grid_ls(data_time)

    delta_h2o_ice_s = compute_error(data_ref_h2o_ice_s, data_h2o_ice_s)
    delta_h2o_ice_s = linearize_ls(delta_h2o_ice_s, data_time.shape[0], data_latitude.shape[0], interp_time)

    delta_icetot = compute_error(data_ref_icetot, data_icetot)
    delta_icetot = linearize_ls(delta_icetot, data_time.shape[0], data_latitude.shape[0], interp_time)

    delta_tauTES = compute_error(data_ref_tauTES, data_tauTES)
    delta_tauTES = linearize_ls(delta_tauTES, data_time.shape[0], data_latitude.shape[0], interp_time)

    delta_mtot = compute_error(data_ref_mtot, data_mtot)
    delta_mtot = linearize_ls(delta_mtot, data_time.shape[0], data_latitude.shape[0], interp_time)

    delta_ps = compute_error(data_ref_ps, data_ps)
    delta_ps = linearize_ls(delta_ps, data_time.shape[0], data_latitude.shape[0], interp_time)

    delta_tsurf = compute_error(data_ref_tsurf, data_tsurf)
    delta_tsurf = linearize_ls(delta_tsurf, data_time.shape[0], data_latitude.shape[0], interp_time)

    # top - left: tsurf
    print(min(data_tsurf), max(data_tsurf))
    plot_figure(data_ref_tsurf, data_tsurf, delta_tsurf, levels=arange(140,270,10),
                title='Zonal mean of surface temperature', unit='K', format='%.2f', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_tsurf')

    # top - right: ps
    print(min(data_ps), max(data_ps))
    plot_figure(data_ref_ps, data_ps, delta_ps, levels=arange(300, 1200, 100),
                title='Zonal mean of surface pressure', unit='Pa', format='%d', ndx=ndx, data_latitude=data_latitude,
                savename='check_watercycle_relative_error_ps')

    # middle - left: mtot
    print(min(data_mtot), max(data_mtot))
    plot_figure(data_ref_mtot, data_mtot, delta_mtot, levels=None,#logspace(-5, -1, 10),
                title='Zonal mean of total atmospheric mass', unit='kg/m$^2$', format='%.1e', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_mtot')

    # middle - right: icetot
    print(min(data_icetot), max(data_icetot))
    plot_figure(data_ref_icetot, data_icetot, delta_icetot, levels=linspace(0, 0.05, 10),
                title='Zonal mean of total water ice mass', unit='kg/m$^2$', format='%.1e', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_icetot')

    # bot - left: h2o_ice_s
    print(min(data_h2o_ice_s), max(data_h2o_ice_s))
    plot_figure(data_ref_h2o_ice_s, data_h2o_ice_s, delta_h2o_ice_s, levels=arange(0, 9, 1),
                title='Zonal mean of surface water ice', unit='kg/m$^2$', format='%d', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_h2o_ice_s')

    # bot - right: tauTES
    print(min(data_tauTES), max(data_tauTES))
    plot_figure(data_ref_tauTES, data_tauTES, delta_tauTES, levels=arange(0, 7, 1),
                title='Zonal mean of opacity at 825 cm$^{-1}$', unit='', format='%.d',ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_tauTES')


    try:
        data_ref = Dataset('/home/mathe/Documents/runs_analysis/simu_ref_cycle_eau_mvals/ref_watercycle_co2ice.nc',
                           "r", format='NETCDF4')
    except:
        print('No ref_watercycle_co2ice.nc exists!')
        exit()

    try:
        data = Dataset('concat_co2ice.nc', "r", format='NETCDF4')
    except:
        print('No concat_co2ice.nc exists!')
        exit()

    data_ref_co2ice = data_ref.variables['co2ice'][:,:,:]
    data_co2ice = data.variables['co2ice'][:,:,:]

    data_ref_co2ice = correction_data(data_ref_co2ice, threshold=1e-5)
    data_ref_co2ice = flip(mean(data_ref_co2ice, axis=2).T, axis=0)

    data_co2ice = correction_data(data_co2ice, threshold=1e-5)
    data_co2ice = flip(mean(data_co2ice, axis=2).T, axis=0)

    delta_co2ice = compute_error(data_ref_co2ice, data_co2ice)
    delta_co2ice = linearize_ls(delta_co2ice, data_time.shape[0], data_latitude.shape[0], interp_time)

    plot_figure(data_ref_co2ice, data_co2ice, delta_co2ice, levels=None, title='CO$_2$ ice at the surface',
                unit='kg/m$^2$', format='%.1e', ndx=ndx, data_latitude=data_latitude,
                savename='check_watercycle_relative_error_co2ice')


if __name__ == '__main__':
    main()