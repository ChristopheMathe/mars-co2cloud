#!/bin/bash python3
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm, LogNorm
from numpy import mean, abs, min, max, zeros, where, ones, concatenate, flip, arange, unravel_index, argmax, array, \
    count_nonzero, diff, std, savetxt, c_, append, loadtxt, asarray, power, linspace, logspace

def correction_value(data, threshold):
    from numpy import ma

    data = ma.masked_where(data <= threshold, data)

    return data


def rotate_data(*list_data, doflip):
    from numpy import flip
    list_data = list(list_data)

    for i, value in enumerate(list_data):
        list_data[i] = list_data[i].T  # get Ls on x-axis
        if doflip:
            list_data[i] = flip(list_data[i], axis=0)  # reverse to get North pole on top of the fig

    return list_data


def extract_data(data):
    data_h2o_ice_s = data.variables['h2o_ice_s'][:, :, :]
    data_icetot = data.variables['icetot'][:, :, :]
    data_tauTES = data.variables['tauTES'][:, :, :]
    data_mtot = data.variables['mtot'][:, :, :]
    data_ps = data.variables['ps'][:, :, :]
    data_tsurf = data.variables['tsurf'][:, :, :]
    data_co2ice = data.variables['co2ice'][:,:,:,]

    # correct data
    data_h2o_ice_s = correction_value(data_h2o_ice_s, threshold=1e-13)
    data_icetot = correction_value(data_icetot, threshold=1e-13)
    data_tauTES = correction_value(data_tauTES, threshold=1e-13)
    data_mtot = correction_value(data_mtot, threshold=1e-13)
    data_ps = correction_value(data_ps, threshold=1e-13)
    data_tsurf = correction_value(data_tsurf, threshold=1e-13)
    data_co2ice = correction_value(data_co2ice, threshold=1e-13)

    # compute zonal mean
    data_h2o_ice_s = mean(data_h2o_ice_s, axis=2)
    data_icetot = mean(data_icetot, axis=2)
    data_tauTES = mean(data_tauTES, axis=2)
    data_mtot = mean(data_mtot, axis=2)
    data_ps = mean(data_ps, axis=2)
    data_tsurf = mean(data_tsurf, axis=2)
    data_co2ice = mean(data_co2ice, axis=2)

    # rotate data
    data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf, data_co2ice = rotate_data(
        data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf, data_co2ice, doflip=True)
    return data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf, data_co2ice


def compute_error(data_ref, data):
    shape_data = data.shape
    delta = zeros((shape_data))

    for i in range(shape_data[0]):
        for j in range(shape_data[1]):
            if data_ref[i,j] == 0:
                delta[i,j] == 0
            else:
                delta[i,j] = (data[i,j] - data_ref[i,j]) * 100 / data_ref[i,j]

    return delta


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


def get_ls_index(data_time):
    from numpy import array, searchsorted, max

    axis_ls = array([0, 90, 180, 270, 360])
    if max(data_time) > 361:
        # ls = 0, 90, 180, 270, 360
        idx = searchsorted(data_time[:], [0, 193.47, 371.99, 514.76, 669])
    else:
        idx = searchsorted(data_time[:], axis_ls)

    return idx, axis_ls


def main():
    # Data from Margaux Vals
    data_ref = Dataset('../simu_ref_cycle_eau_mvals/concat_vars_3D_Ls.nc', "r", format="NETCDF4")
    data_ref_h2o_ice_s, data_ref_icetot, data_ref_tauTES, data_ref_mtot, data_ref_ps, data_ref_tsurf, data_ref_co2ice\
        = extract_data(data_ref)

    # My data
    data_3D = Dataset('occigen_test_64x48x32_1years_Tµphy_para_start_simu_ref_Margaux/concat_vars_3D_Ls.nc',
                      "r",
                      format="NETCDF4")
    data_h2o_ice_s, data_icetot, data_tauTES, data_mtot, data_ps, data_tsurf, data_co2ice = extract_data(data_3D)

    # Get ndx and axis_ls
    data_latitude = data_3D.variables['latitude']
    data_latitude = data_latitude[::-1]
    data_time = data_3D.variables['Time']
    ndx, axis_ls = get_ls_index(data_time)

    # Compute the difference between Margaux and me
    delta_h2o_ice_s = compute_error(data_ref_h2o_ice_s, data_h2o_ice_s)
    delta_icetot = compute_error(data_ref_icetot, data_icetot)
    delta_tauTES = compute_error(data_ref_tauTES, data_tauTES)
    delta_mtot = compute_error(data_ref_mtot, data_mtot)
    delta_ps = compute_error(data_ref_ps, data_ps)
    delta_tsurf = compute_error(data_ref_tsurf, data_tsurf)
    delta_co2ice = compute_error(data_ref_co2ice, data_co2ice)

    # Plot: tsurf
    print('tsurf: min = {:.0f}, max = {:.0f}'.format(min(data_tsurf), max(data_tsurf)))
    plot_figure(data_ref_tsurf, data_tsurf, delta_tsurf, levels=arange(140,270,10),
                title='Zonal mean of surface temperature', unit='K', format='%.2f', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_tsurf')

    # Plot: ps
    print('ps: min = {:.0f}, max = {:.0f}'.format(min(data_ps), max(data_ps)))
    plot_figure(data_ref_ps, data_ps, delta_ps, levels=arange(300, 1200, 100),
                title='Zonal mean of surface pressure', unit='Pa', format='%d', ndx=ndx, data_latitude=data_latitude,
                savename='check_watercycle_relative_error_ps')

    # Plot: mtot
    print('mtot: min = {:.0f}, max = {:.0f}'.format(min(data_mtot), max(data_mtot)))
    plot_figure(data_ref_mtot, data_mtot, delta_mtot, levels=None,#logspace(-5, -1, 10),
                title='Zonal mean of total atmospheric mass', unit='kg/m$^2$', format='%.1e', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_mtot')

    # Plot: icetot
    print('icetot: min = {:.0f}, max = {:.0f}'.format(min(data_icetot), max(data_icetot)))
    plot_figure(data_ref_icetot, data_icetot, delta_icetot, levels=linspace(0, 0.05, 10),
                title='Zonal mean of total water ice mass', unit='kg/m$^2$', format='%.1e', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_icetot')

    # Plot: h2o_ice_s
    print('h2o_ice_s: min = {:.0f}, max = {:.0f}'.format(min(data_h2o_ice_s), max(data_h2o_ice_s)))
    plot_figure(data_ref_h2o_ice_s, data_h2o_ice_s, delta_h2o_ice_s, levels=arange(0, 9, 1),
                title='Zonal mean of surface water ice', unit='kg/m$^2$', format='%d', ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_h2o_ice_s')

    # Plot: tauTES
    print('tauTES: min = {:.0f}, max = {:.0f}'.format(min(data_tauTES), max(data_tauTES)))
    plot_figure(data_ref_tauTES, data_tauTES, delta_tauTES, levels=arange(0, 7, 1),
                title='Zonal mean of opacity at 825 cm$^{-1}$', unit='', format='%.d',ndx=ndx,
                data_latitude=data_latitude, savename='check_watercycle_relative_error_tauTES')

    # Plot: co2ice
    print('co2ice: min = {:.0f}, max = {:.0f}'.format(min(data_co2ice), max(data_co2ice)))
    plot_figure(data_ref_co2ice, data_co2ice, delta_co2ice, levels=None, title='CO$_2$ ice at the surface',
                unit='kg/m$^2$', format='%.1e', ndx=ndx, data_latitude=data_latitude,
                savename='check_watercycle_relative_error_co2ice')



if __name__ == '__main__':
    main()