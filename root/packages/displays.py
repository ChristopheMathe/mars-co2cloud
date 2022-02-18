import matplotlib.pyplot as plt
from .DataObservation import *
from .DataProcessed import *
from .constant_parameter import *


def colormap_idl_rainbow_plus_white():
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [[0., 0., 0.],
               [0.0156863, 0., 0.0117647],
               [0.0352941, 0., 0.027451],
               [0.0509804, 0., 0.0392157],
               [0.0705882, 0., 0.054902],
               [0.0862745, 0., 0.0745098],
               [0.105882, 0., 0.0901961],
               [0.121569, 0., 0.109804],
               [0.141176, 0., 0.12549],
               [0.156863, 0., 0.14902],
               [0.176471, 0., 0.168627],
               [0.196078, 0., 0.188235],
               [0.227451, 0., 0.231373],
               [0.239216, 0., 0.247059],
               [0.25098, 0., 0.266667],
               [0.266667, 0., 0.282353],
               [0.270588, 0., 0.301961],
               [0.282353, 0., 0.317647],
               [0.290196, 0., 0.337255],
               [0.301961, 0., 0.356863],
               [0.309804, 0., 0.372549],
               [0.313725, 0., 0.392157],
               [0.321569, 0., 0.407843],
               [0.32549, 0., 0.427451],
               [0.329412, 0., 0.462745],
               [0.337255, 0., 0.478431],
               [0.341176, 0., 0.498039],
               [0.345098, 0., 0.517647],
               [0.337255, 0., 0.533333],
               [0.341176, 0., 0.552941],
               [0.341176, 0., 0.568627],
               [0.341176, 0., 0.588235],
               [0.333333, 0., 0.603922],
               [0.329412, 0., 0.623529],
               [0.329412, 0., 0.639216],
               [0.329412, 0., 0.658824],
               [0.309804, 0., 0.694118],
               [0.305882, 0., 0.713725],
               [0.301961, 0., 0.729412],
               [0.298039, 0., 0.74902],
               [0.278431, 0., 0.764706],
               [0.27451, 0., 0.784314],
               [0.266667, 0., 0.8],
               [0.258824, 0., 0.819608],
               [0.235294, 0., 0.839216],
               [0.227451, 0., 0.854902],
               [0.215686, 0., 0.87451],
               [0.180392, 0., 0.909804],
               [0.168627, 0., 0.92549],
               [0.156863, 0., 0.945098],
               [0.141176, 0., 0.960784],
               [0.129412, 0., 0.980392],
               [0.0980392, 0., 1.],
               [0.0823529, 0., 1.],
               [0.0627451, 0., 1.],
               [0.0470588, 0., 1.],
               [0.0156863, 0., 1.],
               [0., 0., 1.],
               [0., 0.0156863, 1.],
               [0., 0.0627451, 1.],
               [0., 0.0823529, 1.],
               [0., 0.0980392, 1.],
               [0., 0.113725, 1.],
               [0., 0.14902, 1.],
               [0., 0.164706, 1.],
               [0., 0.180392, 1.],
               [0., 0.2, 1.],
               [0., 0.215686, 1.],
               [0., 0.247059, 1.],
               [0., 0.262745, 1.],
               [0., 0.282353, 1.],
               [0., 0.329412, 1.],
               [0., 0.34902, 1.],
               [0., 0.364706, 1.],
               [0., 0.380392, 1.],
               [0., 0.415686, 1.],
               [0., 0.431373, 1.],
               [0., 0.447059, 1.],
               [0., 0.466667, 1.],
               [0., 0.498039, 1.],
               [0., 0.513725, 1.],
               [0., 0.529412, 1.],
               [0., 0.54902, 1.],
               [0., 0.596078, 1.],
               [0., 0.615686, 1.],
               [0., 0.631373, 1.],
               [0., 0.647059, 1.],
               [0., 0.682353, 1.],
               [0., 0.698039, 1.],
               [0., 0.713725, 1.],
               [0., 0.733333, 1.],
               [0., 0.764706, 1.],
               [0., 0.780392, 1.],
               [0., 0.796078, 1.],
               [0., 0.847059, 1.],
               [0., 0.862745, 1.],
               [0., 0.882353, 1.],
               [0., 0.898039, 1.],
               [0., 0.913725, 1.],
               [0., 0.94902, 1.],
               [0., 0.964706, 1.],
               [0., 0.980392, 1.],
               [0., 1., 1.],
               [0., 1., 0.964706],
               [0., 1., 0.94902],
               [0., 1., 0.933333],
               [0., 1., 0.882353],
               [0., 1., 0.862745],
               [0., 1., 0.847059],
               [0., 1., 0.831373],
               [0., 1., 0.796078],
               [0., 1., 0.780392],
               [0., 1., 0.764706],
               [0., 1., 0.74902],
               [0., 1., 0.733333],
               [0., 1., 0.698039],
               [0., 1., 0.682353],
               [0., 1., 0.666667],
               [0., 1., 0.615686],
               [0., 1., 0.596078],
               [0., 1., 0.580392],
               [0., 1., 0.564706],
               [0., 1., 0.529412],
               [0., 1., 0.513725],
               [0., 1., 0.498039],
               [0., 1., 0.482353],
               [0., 1., 0.447059],
               [0., 1., 0.431373],
               [0., 1., 0.415686],
               [0., 1., 0.4],
               [0., 1., 0.34902],
               [0., 1., 0.329412],
               [0., 1., 0.313725],
               [0., 1., 0.298039],
               [0., 1., 0.262745],
               [0., 1., 0.247059],
               [0., 1., 0.231373],
               [0., 1., 0.215686],
               [0., 1., 0.180392],
               [0., 1., 0.164706],
               [0., 1., 0.14902],
               [0., 1., 0.0980392],
               [0., 1., 0.0823529],
               [0., 1., 0.0627451],
               [0., 1., 0.0470588],
               [0., 1., 0.0313725],
               [0., 1., 0.],
               [0.0156863, 1., 0.],
               [0.0313725, 1., 0.],
               [0.0470588, 1., 0.],
               [0.0823529, 1., 0.],
               [0.0980392, 1., 0.],
               [0.113725, 1., 0.],
               [0.164706, 1., 0.],
               [0.180392, 1., 0.],
               [0.2, 1., 0.],
               [0.215686, 1., 0.],
               [0.247059, 1., 0.],
               [0.262745, 1., 0.],
               [0.282353, 1., 0.],
               [0.298039, 1., 0.],
               [0.313725, 1., 0.],
               [0.34902, 1., 0.],
               [0.364706, 1., 0.],
               [0.380392, 1., 0.],
               [0.431373, 1., 0.],
               [0.447059, 1., 0.],
               [0.466667, 1., 0.],
               [0.482353, 1., 0.],
               [0.513725, 1., 0.],
               [0.529412, 1., 0.],
               [0.54902, 1., 0.],
               [0.564706, 1., 0.],
               [0.6, 1., 0.],
               [0.615686, 1., 0.],
               [0.631373, 1., 0.],
               [0.647059, 1., 0.],
               [0.698039, 1., 0.],
               [0.713725, 1., 0.],
               [0.733333, 1., 0.],
               [0.74902, 1., 0.],
               [0.780392, 1., 0.],
               [0.796078, 1., 0.],
               [0.815686, 1., 0.],
               [0.831373, 1., 0.],
               [0.866667, 1., 0.],
               [0.882353, 1., 0.],
               [0.898039, 1., 0.],
               [0.94902, 1., 0.],
               [0.964706, 1., 0.],
               [0.980392, 1., 0.],
               [1., 1., 0.],
               [1., 0.980392, 0.],
               [1., 0.94902, 0.],
               [1., 0.933333, 0.],
               [1., 0.913725, 0.],
               [1., 0.898039, 0.],
               [1., 0.866667, 0.],
               [1., 0.847059, 0.],
               [1., 0.831373, 0.],
               [1., 0.780392, 0.],
               [1., 0.764706, 0.],
               [1., 0.74902, 0.],
               [1., 0.733333, 0.],
               [1., 0.698039, 0.],
               [1., 0.682353, 0.],
               [1., 0.666667, 0.],
               [1., 0.647059, 0.],
               [1., 0.631373, 0.],
               [1., 0.6, 0.],
               [1., 0.580392, 0.],
               [1., 0.564706, 0.],
               [1., 0.513725, 0.],
               [1., 0.498039, 0.],
               [1., 0.482353, 0.],
               [1., 0.466667, 0.],
               [1., 0.431373, 0.],
               [1., 0.415686, 0.],
               [1., 0.4, 0.],
               [1., 0.380392, 0.],
               [1., 0.34902, 0.],
               [1., 0.333333, 0.],
               [1., 0.313725, 0.],
               [1., 0.298039, 0.],
               [1., 0.247059, 0.],
               [1., 0.231373, 0.],
               [1., 0.215686, 0.],
               [1., 0.2, 0.],
               [1., 0.164706, 0.],
               [1., 0.14902, 0.],
               [1., 0.133333, 0.],
               [1., 0.113725, 0.],
               [1., 0.0823529, 0.],
               [1., 0.0666667, 0.],
               [1., 0.0470588, 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 1., 1.]]

    test_cm = LinearSegmentedColormap.from_list('test', cm_data)
    return test_cm


def display_co2ice_at_viking_lander_site(data_at_vk1, data_at_vk2, data_time):
    plt.figure(figsize=figsize_1graph)
    plt.plot(data_time, data_at_vk1, color='blue', label='At VK1')
    plt.plot(data_time, data_at_vk2, color='red', label='At VK2')
    plt.legend(loc=0)
    plt.xlabel('Time (sols)')
    plt.ylabel('CO2 ice at surface (kg)')
    plt.xlim(0, 669)
    plt.savefig('co2ice_at_viking_lander_site.png', bbox_inches='tight')

    dict_var = [{'data': data_at_vk1, 'varname': 'CO2 ice quantity at the surface at Viking Lander 1 site',
                 'units': 'kg', 'shortname': 'CO2ICE_VK1'},
                {'data': data_at_vk2, 'varname': 'CO2 ice quantity at the surface at Viking Lander 2 site',
                 'units': 'kg', 'shortname': 'CO2ICE_VK2'},
                {'data': data_time[:], 'varname': 'Time in sols', 'units': 'sols', 'shortname': 'TIME'},
                ]

    save_figure_data(list_dict_var=dict_var, savename='co2ice_at_viking_lander_site')


def display_co2_ice_mola(info_netcdf):
    from matplotlib.colors import LogNorm
    from numpy import logspace

    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                        selected_time=info_netcdf.local_time)

    data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
        if info_netcdf.data_dim.time.units != 'deg':
            data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = info_netcdf.data_dim.time[idx::len(data_local_time)]

        info_netcdf.data_target, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
    else:
        data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
        data_time = data_ls[idx::len(data_local_time)]
        info_netcdf.data_target, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])

    mola_latitude, mola_ls, mola_altitude = observation_mola()
    mola_altitude = correction_value(mola_altitude, operator='inf', value=0)
    mola_altitude = correction_value(mola_altitude, operator='sup', value=1e4)

    fig, ax = plt.subplots(ncols=2, figsize=figsize_1graph)
    ax[0].set_title('Zonal mean of column density \n of CO$_2$ ice (kg.m$^{-2}$)', loc='center')
    ctf = ax[0].contourf(data_time, info_netcdf.data_dim.latitude[:], info_netcdf.data_target,
                         norm=LogNorm(), levels=logspace(-13, 2, 16), cmap='inferno')
    plt.colorbar(ctf, ax=ax[0])

    ax[1].set_title('Top altitude of the CO$_2$ cloud \nobserved from MOLA (km)')
    ctf2 = ax[1].contourf(mola_ls[:], mola_latitude[:], mola_altitude[:, :], levels=arange(-4, 11), cmap='inferno')
    plt.colorbar(ctf2, ax=ax[1])

    for axes in ax.reshape(-1):
        axes.set_xlim(0, 360)
        axes.set_ylim(-90, 90)
        axes.set_xticks(ticks=ndx)
        axes.set_xticklabels(axis_ls, fontsize=fontsize)
        axes.set_yticks(ticks=info_netcdf.data_dim.latitude[::4])
        axes.set_yticklabels(labels=[str(int(x)) for x in info_netcdf.data_dim.latitude[::4]], fontsize=fontsize)
        axes.set_ylabel('Latitude (°N)')
        axes.set_xlabel('Solar longitude (°)')
    plt.savefig('DARI_co2_ice_density_column_MOLA.png', bbox_inches='tight')
    plt.show()
    return


def display_co2_ice_distribution_altitude_latitude_polar(info_netcdf, distribution_north, distribution_south,
                                                         north_latitude, south_latitude, save_name):
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0, vmax=2000)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_2graph_rows)

    ax[0].set_title('North pole', fontsize=fontsize)
    pc = ax[0].pcolormesh(north_latitude, info_netcdf.data_dim.altitude / 1e3, distribution_north, norm=norm,
                          cmap='Greys', shading='auto')
    ax[0].set_ylim(0, 40)
    ax[0].tick_params(labelsize=fontsize)
    ax[1].set_title('South pole', fontsize=fontsize)
    ax[1].pcolormesh(south_latitude, info_netcdf.data_dim.altitude / 1e3, distribution_south, norm=norm,
                     cmap='Greys', shading='auto')
    ax[1].set_ylim(0, 40)
    ax[1].tick_params(labelsize=fontsize)

    plt.draw()
    p0 = ax[0].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0.95, p0[2] - p0[0], 0.025])  # left, bottom, width, height
    cbar = plt.colorbar(pc, cax=ax_cbar, orientation='horizontal')
    cbar.ax.set_title('count', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    fig.text(0.02, 0.5, f'{info_netcdf.data_dim.altitude.name} above areoid (k{info_netcdf.data_dim.altitude.units})',
             ha='center', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=fontsize)

    fig.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_co2_ice_cloud_evolution_latitude(info_netcdf, data_satuco2, data_temp, data_riceco2, data_ccnco2,
                                             data_h2o_ice, latitude_selected):
    from numpy import arange, logspace, array
    from matplotlib.colors import LogNorm, Normalize
    dirsave = 'cloud_evolution/'

    cpt = 0
    for i, value_i in enumerate(info_netcdf.data_dim.time):
        if (i / info_netcdf.data_dim.time.shape[0] * 100) > (cpt * 5):
            print(f'{i / info_netcdf.data_dim.time.shape[0] * 100:.0f} %')
            cpt = cpt + 1
        if info_netcdf.data_target[i, :, :].any():
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize_1graph)
            fig.subplots_adjust(wspace=0.4)
            fig.suptitle(f'Sols: {value_i:.0f}, local time: {value_i * 24 % 24:.0f} h, zonal mean', fontsize=fontsize)

            ax[0, 0].title.set_text('CO$_2$ ice mmr', fontsize=fontsize)
            pc0 = ax[0, 0].contourf(latitude_selected, info_netcdf.data_dim.altitude[:],
                                    info_netcdf.data_target[i, :, :], norm=LogNorm(vmin=1e-13, vmax=1),
                                    levels=logspace(-13, 0, 14), cmap='inferno')
            cbar0 = plt.colorbar(pc0, ax=ax[0, 0])
            cbar0.ax.set_title('kg.kg$^{-1}$', fontsize=fontsize)
            cbar0.ax.tick_params(labelsize=fontsize)

            ax[0, 1].title.set_text('Temperature', fontsize=fontsize)
            pc1 = ax[0, 1].contourf(latitude_selected, info_netcdf.data_dim.altitude[:], data_temp[i, :, :],
                                    vmin=80, vmax=240, levels=arange(80, 260, 20), cmap='inferno')
            cbar1 = plt.colorbar(pc1, ax=ax[0, 1])
            cbar1.ax.set_title('K', fontsize=fontsize)
            cbar1.ax.tick_params(labelsize=fontsize)

            ax[1, 0].title.set_text('Saturation of CO$_2$ ice', fontsize=fontsize)
            pc2 = ax[1, 0].contourf(latitude_selected, info_netcdf.data_dim.altitude[:], data_satuco2[i, :, :],
                                    norm=Normalize(vmin=1, vmax=1000),
                                    levels=array([1, 5, 10, 25, 50, 75, 100, 500, 1000]), cmap='inferno', extend='max')
            cbar2 = plt.colorbar(pc2, ax=ax[1, 0])
            cbar2.ax.set_title('')
            cbar2.ax.tick_params(labelsize=fontsize)

            ax[1, 1].title.set_text('Radius of CO$_2$ ice', fontsize=fontsize)
            pc3 = ax[1, 1].contourf(latitude_selected, info_netcdf.data_dim.altitude[:], data_riceco2[i, :, :] * 1e6,
                                    norm=LogNorm(vmin=1e-3, vmax=1e3), levels=logspace(-3, 3, 7), cmap='inferno')
            cbar3 = plt.colorbar(pc3, ax=ax[1, 1])
            cbar3.ax.set_title('µm', fontsize=fontsize)
            cbar3.ax.tick_params(labelsize=fontsize)

            ax[0, 2].title.set_text('Number of\ncondensation nuclei', fontsize=fontsize)
            pc4 = ax[0, 2].contourf(latitude_selected, info_netcdf.data_dim.altitude[:], data_ccnco2[i, :, :],
                                    norm=LogNorm(vmin=1, vmax=1e10), levels=logspace(0, 10, 11), cmap='inferno',
                                    extend='max')
            cbar4 = plt.colorbar(pc4, ax=ax[0, 2])
            cbar4.ax.set_title('#.m$^{-3}$', fontsize=fontsize)
            cbar4.ax.tick_params(labelsize=fontsize)

            ax[1, 2].title.set_text('H2O ice mmr', fontsize=fontsize)
            pc5 = ax[1, 2].contourf(latitude_selected, info_netcdf.data_dim.altitude[:], data_h2o_ice[i, :, :],
                                    norm=LogNorm(vmin=1e-13, vmax=1), levels=logspace(-13, 0, 14), cmap='inferno')
            cbar5 = plt.colorbar(pc5, ax=ax[1, 2])
            cbar5.ax.set_title('kg.kg$^{-1}$', fontsize=fontsize)
            cbar5.ax.tick_params(labelsize=fontsize)

            for axes in ax.reshape(-1):
                axes.set_yscale('log')
                axes.invert_yaxis()

            fig.text(0.02, 0.5, 'Altitude (Pa)', ha='center', va='center', rotation='vertical', fontsize=fontsize)
            fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=fontsize)

            save_name = f'cloud_evolution_latitude_sols_{info_netcdf.data_dim.time[i]:.0f}_' \
                        f'{info_netcdf.data_dim.time[i] * 24 % 24:.0f}h.png'
            plt.savefig(dirsave + save_name, bbox_inches='tight')
            plt.close()
    return


def display_co2_ice_max_longitude_altitude(info_netcdf, max_mmr, max_alt, max_temp, max_satu, max_radius, max_ccn_n,
                                           unit):
    from matplotlib.colors import LogNorm, DivergingNorm, Normalize, BoundaryNorm
    from numpy import arange, logspace, max, min
    from math import log10

    cmap = plt.get_cmap('inferno')

    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                        selected_time=0)

    data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
        if info_netcdf.data_dim.time.units != 'deg':
            data_ls = data_ls[idx::len(data_local_time)]
        else:
            data_ls = info_netcdf.data_dim.time[idx::len(data_local_time)]
        max_mmr, data_time = linearize_ls(data=max_mmr, data_ls=data_ls)
        max_alt, data_time = linearize_ls(data=max_alt, data_ls=data_ls)
        max_temp, data_time = linearize_ls(data=max_temp, data_ls=data_ls)
        max_satu, data_time = linearize_ls(data=max_satu, data_ls=data_ls)
        max_radius, data_time = linearize_ls(data=max_radius, data_ls=data_ls)
        max_ccn_n, data_time = linearize_ls(data=max_ccn_n, data_ls=data_ls)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
    else:
        data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
        data_ls = data_ls[idx::len(data_local_time)]
        max_mmr, data_time = linearize_ls(data=max_mmr, data_ls=data_ls)
        max_alt, data_time = linearize_ls(data=max_alt, data_ls=data_ls)
        max_temp, data_time = linearize_ls(data=max_temp, data_ls=data_ls)
        max_satu, data_time = linearize_ls(data=max_satu, data_ls=data_ls)
        max_radius, data_time = linearize_ls(data=max_radius, data_ls=data_ls)
        max_ccn_n, data_time = linearize_ls(data=max_ccn_n, data_ls=data_ls)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])

    if info_netcdf.data_dim.altitude.units == 'Pa':
        max_alt = correction_value(data=max_alt, operator='sup', value=1e3)
    # PLOT
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize_6graph_2rows_3cols)

    # plot 1
    ax[0, 0].set_title(f'Max {info_netcdf.target_name} in altitude/longitude', fontsize=fontsize)
    pc = ax[0, 0].pcolormesh(max_mmr, norm=LogNorm(vmin=1e-13, vmax=1e0), cmap=cmap, shading='flat')
    ax[0, 0].set_facecolor('white')
    cbar = plt.colorbar(pc, ax=ax[0, 0])
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # plot 2
    ax[0, 1].set_title('Altitude at co2_ice mmr max', fontsize=fontsize)
    pc2 = ax[0, 1].pcolormesh(max_alt, norm=LogNorm(vmin=1e-3, vmax=1e3), cmap=cmap, shading='flat')
    ax[0, 1].set_facecolor('white')
    cbar2 = plt.colorbar(pc2, ax=ax[0, 1])
    cbar2.ax.set_title('km', fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    # plot 3
    ax[0, 2].set_title('Temperature at co2_ice mmr max', fontsize=fontsize)
    pc3 = ax[0, 2].pcolormesh(max_temp, norm=Normalize(vmin=100, vmax=180), cmap=cmap, shading='flat')
    ax[0, 2].set_facecolor('white')
    cbar3 = plt.colorbar(pc3, ax=ax[0, 2])
    cbar3.ax.set_title('K', fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)

    # plot 4
    cmap_satu = plt.get_cmap('coolwarm')
    ax[1, 0].set_title('Saturation at co2_ice mmr max', fontsize=fontsize)
    pc4 = ax[1, 0].pcolormesh(max_satu, norm=DivergingNorm(vmin=0, vcenter=1, vmax=5), cmap=cmap_satu, shading='flat')
    ax[1, 0].set_facecolor('white')
    cbar4 = plt.colorbar(pc4, ax=ax[1, 0])
    cbar4.ax.set_title(' ')
    cbar4.ax.tick_params(labelsize=fontsize)

    # plot 5
    ax[1, 1].set_title('Radius of co2_ice at co2_ice mmr max', fontsize=fontsize)
    pc5 = ax[1, 1].pcolormesh(max_radius, cmap=cmap, shading='flat', norm=LogNorm(vmin=1e-3, vmax=1e3))
    ax[1, 1].set_facecolor('white')
    cbar5 = plt.colorbar(pc5, ax=ax[1, 1])
    cbar5.ax.set_title(u'µm', fontsize=fontsize)
    cbar5.ax.tick_params(labelsize=fontsize)

    # plot 6
    cmap_ccn = plt.get_cmap('inferno')
    levels = logspace(0, log10(max(max_ccn_n)), int(log10(max(max_ccn_n)) + 1))
    norm = BoundaryNorm(levels, ncolors=cmap_ccn.N, clip=False)
    ax[1, 2].set_title('CCN number at co2_ice mmr max', fontsize=fontsize)
    pc6 = ax[1, 2].pcolormesh(max_ccn_n, cmap=cmap_ccn, norm=norm, shading='flat')
    ax[1, 2].set_facecolor('white')
    cbar6 = plt.colorbar(pc6, ax=ax[1, 2], ticks=levels)
    cbar6.ax.set_title('nb/kg', fontsize=fontsize)
    cbar6.ax.set_yticklabels(['{:.0e}'.format(x) for x in levels], fontsize=fontsize)

    for axes in ax.reshape(-1):
        axes.set_ylabel('Latitude (°N)', fontsize=fontsize)
        axes.set_xlabel('Solar Longitude (°)', fontsize=fontsize)
        axes.set_yticks(ticks=arange(0, len(info_netcdf.data_dim.latitude), 6))
        axes.set_yticklabels(labels=info_netcdf.data_dim.latitude[::6], fontsize=fontsize)
        axes.set_xticks(ndx)
        axes.set_xticklabels(axis_ls, fontsize=fontsize)

    if len(info_netcdf.local_time) == 1:
        fig.savefig(f'max_{info_netcdf.target_name}_in_altitude_longitude{info_netcdf.local_time[0]:2.0f}h.png',
                    bbox_inches='tight')
    else:
        fig.savefig(f'max_{info_netcdf.target_name}_in_altitude_longitude_diurnal_mean.png',
                    bbox_inches='tight')
    plt.show()
    return


def display_co2_ice_density_column_evolution_polar_region(info_netcdf, time, latitude):
    from numpy import logspace
    from math import floor
    from matplotlib import cm
    from matplotlib.colors import BoundaryNorm
    import cartopy.crs as crs
    from os import mkdir
    from matplotlib.ticker import FuncFormatter

    plate_carree = crs.PlateCarree(central_longitude=0)

    if latitude[0] > 0:
        orthographic = crs.Orthographic(central_longitude=0, central_latitude=90)
        title = 'North polar region'
        pole = 'north'
        dir_name = 'co2_ice_density_column_evolution_northern_polar_region/'
    else:
        orthographic = crs.Orthographic(central_longitude=0, central_latitude=-90)
        title = 'South polar region'
        pole = 'south'
        dir_name = 'co2_ice_density_column_evolution_southern_polar_region/'

    try:
        mkdir(dir_name)
    except FileExistsError:
        pass

    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°

    cmap = cm.get_cmap('inferno')
    cmap.set_under('w')
    levels = logspace(-13, 1, 15)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    save_name = list([])

    # PLOT
    for i in range(time.shape[0]):
        if not info_netcdf.data_target[i, :, :].mask.all():
            fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': orthographic}, figsize=figsize_1graph,
                                   facecolor='white')
            ax.set_title(title + f', sols {floor(time[i]):.0f} LT {time[i] * 24 % 24:.0f}')
            ax.set_facecolor('white')

            ctf = ax.pcolormesh(info_netcdf.data_dim.longitude[:], latitude, info_netcdf.data_target[i, :, :],
                                norm=norm,
                                transform=plate_carree, cmap=cmap)
            ax.set_global()
            workaround_gridlines(plate_carree, axes=ax, pole=pole)
            cbar = fig.colorbar(ctf, format=FuncFormatter(lambda x, levels: "%.0e" % x))
            cbar.ax.set_title('kg.m$^{-2}$')
            savename = f'{dir_name}co2_ice_density_column_evolution_sols_{floor(time[i]):.0f}_LT' \
                       f'_{time[i] * 24 % 24:.0f}.png'
            save_name.append(savename)
            plt.savefig(savename, bbox_inches='tight')
            plt.close(fig)

    # create the gif
    create_gif(save_name)
    return


def display_co2_ice_localtime_ls(info_netcdf, lat_min, lat_max, title, unit, norm, vmin, vmax, save_name):
    from matplotlib.colors import LogNorm, Normalize
    from numpy.ma import masked_inside

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=0)

    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
            data_time = data_ls[idx::len(data_local_time)]
            data, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
            ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
        else:
            data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
            data_time = data_ls[idx::len(data_local_time)]
            data, data_time = linearize_ls(data=info_netcdf.data_taget, data_ls=data_time)
            ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])
    else:
        data_time = info_netcdf.data_dim.time[idx::len(data_local_time)]
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
        data = info_netcdf.data_target

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    ctf = ax.pcolormesh(data_time[:], data_local_time[:], data, norm=norm, cmap="plasma", shading='auto')
    cbar = fig.colorbar(ctf)
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_facecolor('white')

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Solar longitude (°)', fontsize=fontsize)
    ax.set_ylabel('Local time (h)', fontsize=fontsize)
    ax.set_xticks(ndx)
    ax.set_xticklabels(axis_ls, fontsize=fontsize)
    ax.set_yticks(data_local_time)
    ax.set_yticklabels(data_local_time, fontsize=fontsize)

    # observation
    list_instrument = ['HRSC', 'OMEGAlimb', 'OMEGAnadir', 'SPICAM', 'THEMIS', 'NOMAD']
    list_marker = ['s', 'o', 'v', 'P', 'X', '1']
    list_colors = ['black', 'brown', 'green', 'aquamarine', 'red', 'blue']
    for i, value_i in enumerate(list_instrument):
        data_ls, data_lat, data_lon, data_lt, data_alt, data_alt_min, data_alt_max = \
            mesospheric_clouds_altitude_localtime_observed(instrument=value_i)

        mask = masked_inside(data_lat, lat_min, lat_max)  # mask inside but we want mask.mask = True
        ax.scatter(data_ls[mask.mask], data_lt[mask.mask], color=list_colors[i], marker=list_marker[i], label=value_i)

    ax.legend(loc=0)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_co2ice_cumulative_mass_polar_region(info_netcdf, data_co2_ice_north, data_co2_ice_south,
                                                data_precip_co2_ice_north, data_precip_co2_ice_south,
                                                data_direct_condco2_north, data_direct_condco2_south):
    data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=0)

    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        data_time = data_ls[idx::len(data_local_time)]
    else:
        data_time = info_netcdf.data_dim.time

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=figsize_1graph)
    ax[0].set_yscale('symlog')
    ax[1].set_yscale('symlog')
    ax[0].set_title(u'Northern polar region, diurnal mean, lat=[60:90]°N', fontsize=fontsize)
    ax[0].plot(data_time[1:], data_co2_ice_north, color='black', label='Total co2 ice')
    ax[0].plot(data_time[:], data_precip_co2_ice_north, color='blue', label='Precipitation')
    ax[0].plot(data_time[1:], data_direct_condco2_north, color='red', label='Direct condensation')
    ax[0].hlines(0, xmin=0, xmax=360, color='grey')
    ax[0].legend(loc='best')

    ax[1].set_title(u'Southern polar region, diurnal mean, lat=[60:90]°S', fontsize=fontsize)
    ax[1].plot(data_time[1:], data_co2_ice_south, color='black', label='Total co2 ice')
    ax[1].plot(data_time[:], data_precip_co2_ice_south * 10, color='blue', label='Precipitation')
    ax[1].plot(data_time[1:], data_direct_condco2_south, color='red', label='Direct condensation')
    ax[1].hlines(0, xmin=0, xmax=360, color='grey')
    ax[1].legend(loc='best')

    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].set_xlim(0, 360)
    #    ax[0].set_ylim(-1e-10, 1e6)

    fig.text(0.05, 0.5, 'Flux (kg/m2)', ha='center', va='center', rotation='vertical',
             fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)

    fig.savefig(f'co2ice_cumulative_mass_polar_region_diurnal_mean.png', bbox_inches='tight')
    return


def display_emis_polar_projection_garybicas2020_figs11_12(info_netcdf, time, levels, cmap, save_name):
    import cartopy.crs as crs
    from numpy import ndarray

    if isinstance(info_netcdf.data_target, ndarray):
        array_mask = True
    else:
        array_mask = False

    plate_carree = crs.PlateCarree(central_longitude=0)

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[60, 90])
    data_np, tmp = slice_data(data=info_netcdf.data_target,
                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                              dimension_slice=info_netcdf.data_dim.latitude,
                              value=[60, 90])

    latitude_sp, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[-90, -60])
    data_sp, tmp = slice_data(data=info_netcdf.data_target,
                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                              dimension_slice=info_netcdf.data_dim.latitude,
                              value=[-90, -60])

    # Slice data binned in 15°Ls during their winter period
    data_np = data_np[12:, :, :]
    data_sp = data_sp[0:12, :, :]

    # North polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic},
                           figsize=figsize_12graph_3rows_4cols)
    fig.suptitle('North polar region', fontsize=fontsize)
    ctf = None
    for i, axes in enumerate(ax.reshape(-1)):
        axes.set_title(f'{int(time[i] + 180)}° - {int(time[i + 1] + 180)}°')
        if array_mask:
            ctf = axes.contourf(info_netcdf.data_dim.longitude, latitude_np, data_np[i, :, :], levels=levels,
                                transform=plate_carree, cmap=cmap)
        else:
            if data_np[i, :, :].mask.all():
                continue
            else:
                ctf = axes.contourf(info_netcdf.data_dim.longitude, latitude_np, data_np[i, :, :], levels=levels,
                                    transform=plate_carree, cmap=cmap)
        axes.set_global()
        workaround_gridlines(plate_carree, axes=axes, pole='north')
        axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    plt.savefig(save_name + 'northern_polar_region_as_fig11_gary-bicas2020.png', bbox_inches='tight')

    # South polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic},
                           figsize=figsize_12graph_3rows_4cols)
    fig.suptitle('South polar region', fontsize=fontsize)
    for i, axes in enumerate(ax.reshape(-1)):
        if i < 12:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°')
            if array_mask:
                ctf = axes.contourf(info_netcdf.data_dim.longitude, latitude_sp, data_sp[i, :, :], levels=levels,
                                    transform=plate_carree, cmap=cmap)
            else:
                if data_sp[i, :, :].mask.all():
                    continue
                else:
                    ctf = axes.contourf(info_netcdf.data_dim.longitude, latitude_sp, data_sp[i, :, :], levels=levels,
                                        transform=plate_carree, cmap=cmap)
            workaround_gridlines(plate_carree, axes=axes, pole='south')
            axes.set_global()
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    plt.savefig(f'{save_name}southern_polar_region_as_fig12_gary-bicas2020.png', bbox_inches='tight')
    return


def display_riceco2_global_mean(info_netcdf, list_data):
    from numpy import mean
    list_data[0] = mean(list_data[0], axis=2)
    list_data[1] = mean(list_data[1], axis=2)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='col', figsize=figsize_1graph)
    fig.subplots_adjust(wspace=0, hspace=0)

    levels = arange(0, 200, 20)
    for axes in ax.reshape(-1):
        axes.set_yscale('log')
        axes.invert_yaxis()
        axes.set_ylim(1e3, 0.2)

    ax[0, 0].set_title('Zonal and latitudinal mean of riceco2', fontsize=fontsize)
    pc = ax[0, 0].contourf(info_netcdf.data_dim.time, info_netcdf.data_dim.altitude, list_data[0].T * 1e6,
                           levels=levels, cmap='inferno')
    ax[1, 0].contourf(info_netcdf.data_dim.time, info_netcdf.data_dim.altitude, list_data[1].T * 1e6, levels=levels,
                      cmap='inferno')
    ax[1, 0].set_xlabel('Solar longitude (°)', fontsize=fontsize)

    ax[0, 1].set_title('Global mean of riceco2', fontsize=fontsize)
    ax[0, 1].set_xscale('log')
    ax[0, 1].plot(mean(list_data[0], axis=0).T * 1e6, info_netcdf.data_dim.altitude)
    ax[0, 1].text(1.1, 0.5, 'North Pole (40°-90°N)', ha='center', va='center', rotation='vertical', fontsize=fontsize,
                  transform=ax[0, 1].transAxes)

    ax[1, 1].plot(mean(list_data[1], axis=0).T * 1e6, info_netcdf.data_dim.altitude)
    ax[1, 1].set_xscale('log')
    ax[1, 1].text(1.1, 0.5, 'South Pole (40°-90°S)', ha='center', va='center', rotation='vertical', fontsize=fontsize,
                  transform=ax[1, 1].transAxes)
    ax[1, 1].set_xlabel('Radius (µm)')

    fig.text(0.06, 0.5, 'Pressure (Pa)', ha='center', va='center', rotation='vertical', fontsize=fontsize)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pc, cax=cbar_ax)
    plt.savefig('riceco2_global_mean_winter_polar_regions.png', bbox_inches='tight')
    plt.show()
    return


def display_riceco2_local_time_evolution(info_netcdf, data, data_std, latitude):
    from matplotlib.cm import get_cmap

    cmap = get_cmap('hsv')
    fig, ax = plt.subplots(figsize=figsize_1graph)
    ax.set_yscale('log')
    ax.set_ylim(1e3, 1e-3)
    ax.set_xlim(1e-3, 1e2)
    ax.set_xscale('log')
    for i in range(data.shape[1]):
        ax.errorbar(data[:, i], info_netcdf.data_dim.altitude, xerr=[data[:, i] - data_std[:, i],
                                                                     data[:, i] + data_std[:, i]],
                    color=cmap(((i + 6) % data.shape[1]) / data.shape[1]),
                    label=f'{info_netcdf.local_time[i]:2.0f} h', capsize=2)

    ax.legend(loc=0)
    ax.set_title(f'Radius of CO$_2$ ice particles at {latitude:.0f}°N, zonal mean', fontsize=fontsize)
    ax.set_ylabel(f'Altitude ({info_netcdf.data_dim.altitude.units})', fontsize=fontsize)
    ax.set_xlabel('Radius particle (µm)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.savefig(f'riceco2_localtime_evolution_{latitude:.0f}N.png', bbox_inches='tight')
    return


def display_riceco2_mean_local_time_evolution(info_netcdf, data_min_radius, data_max_radius, data_mean_radius,
                                              data_std_radius, data_mean_alt, data_min_alt, data_max_alt, latitude):
    fig, ax = plt.subplots(figsize=figsize_1graph)

    ax.plot(info_netcdf.local_time, data_mean_radius, color='black', linestyle='-', label='mean')
    ax.fill_between(info_netcdf.local_time, data_mean_radius - data_std_radius, data_mean_radius + data_std_radius,
                    color='black', alpha=0.7, label='1-$\sigma$')
    ax.fill_between(info_netcdf.local_time, data_min_radius, data_max_radius, color='black', alpha=0.1, label='min-max')
    ax.set_yscale('log')
    ax.set_xlim(0, 24)
    ax.set_ylim(1e-3, 1e2)

    ax2 = ax.twinx()
    ax2.plot(info_netcdf.local_time, data_mean_alt, color='red', linestyle='-', label='alt: mean')
    ax2.fill_between(info_netcdf.local_time, data_min_alt, data_max_alt, color='red', alpha=0.3, label='alt: min-max')

    ax2.set_yscale('log')
    ax2.set_ylim(1e3, 1e-3)
    ax2.set_xlim(0, 24)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel(f'Altitude ({info_netcdf.data_dim.altitude.units})', fontsize=fontsize, color='red')
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='y', colors='red')

    ax.set_title(f'Radius of CO2 ice particles at {latitude}N\n with their location (red)', fontsize=fontsize)
    ax.set_ylabel(f'Radius particle (µm)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(loc=0)
    ax.set_xlabel('Local time (h)', fontsize=fontsize)
    ax.set_xticks(info_netcdf.local_time)
    ax.set_xticklabels(info_netcdf.local_time, fontsize=fontsize)
    plt.savefig(f'riceco2_max_localtime_evolution_{latitude}N.png', bbox_inches='tight')
    return


def display_riceco2_polar_latitudes(info_netcdf, data_north, data_stddev_north, data_south, data_stddev_south):
    from numpy import flip
    from matplotlib import cm

    latitude_north, idx_north = slice_data(data=info_netcdf.data_dim.latitude,
                                           idx_dim_slice=1,
                                           dimension_slice=info_netcdf.data_dim.latitude,
                                           value=[60, 90])
    latitude_south, idx_south = slice_data(data=info_netcdf.data_dim.latitude,
                                           idx_dim_slice=1,
                                           dimension_slice=info_netcdf.data_dim.latitude,
                                           value=[-60, -90])

    data_zareoid, list_var = get_data(filename=info_netcdf.filename, target='zareoid')
    data_surface_local = gcm_surface_local(data_zareoid[::12, :, :, :])
    data_surface_local, latitude = slice_data(data=data_surface_local[:, :, :, :],
                                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                                              dimension_slice=info_netcdf.data_dim.data_latitude,
                                              value=75)
    longitude, idx_longitude = slice_data(data=info_netcdf.data_dim.longitude,
                                          idx_dim_slice=1,
                                          dimension_slice=info_netcdf.data_dim.longitude,
                                          value=0)
    if info_netcdf.data_dim.altitude.units == 'Pa':
        index_10 = abs(data_surface_local[0, :, idx_longitude] - 10e3).argmin()
        index_40 = abs(data_surface_local[0, :, idx_longitude] - 40e3).argmin()
        index_80 = abs(data_surface_local[0, :, idx_longitude] - 80e3).argmin()
    else:
        index_10, index_40, index_80 = None, None, None

    cmap = cm.get_cmap('hsv')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_2graph_rows)
    fig.subplots_adjust(wspace=0.01)

    # northern polar region
    ax[0].set_title('Northern polar region', fontsize=fontsize)
    for i in range(latitude_north.shape[0]):
        part = (i % data_north.shape[1]) / data_north.shape[1]
        ax[0].plot(data_north[:, i], info_netcdf.data_dim.altitude, label=latitude_north[i], color=cmap(part))
        ax[0].errorbar(data_north[:, i], info_netcdf.data_dim.altitude,
                       xerr=[data_north[:, i] * (1 - 1 / data_stddev_north[:, i]),
                             data_north[:, i] * (1 + 1 * data_stddev_north[:, i])],
                       color=cmap(part))
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].hlines(info_netcdf.data_dim.altitude[index_10], 1e-4, 1e3, ls='--', color='black')
    ax[0].hlines(info_netcdf.data_dim.altitude[index_40], 1e-4, 1e3, ls='--', color='black')
    ax[0].hlines(info_netcdf.data_dim.altitude[index_80], 1e-4, 1e3, ls='--', color='black')
    ax[0].text(1e-4, info_netcdf.data_dim.altitude[index_10], '10 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)
    ax[0].text(1e-4, info_netcdf.data_dim.altitude[index_40], '40 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)
    ax[0].text(1e-4, info_netcdf.data_dim.altitude[index_80], '80 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)
    ax[0].set_ylim(1e3, 1e-2)
    ax[0].set_xlim(1e-4, 1e3)
    ax[0].grid()
    ax[0].legend(loc='best', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

    # southern polar region
    ax[1].set_title('Southern polar region', fontsize=fontsize)
    data_south = flip(data_south, axis=1)
    data_stddev_south = flip(data_stddev_south, axis=1)
    latitude_south = flip(latitude_south, axis=0)
    data_surface_local = gcm_surface_local(data_zareoid[::12, :, :, :])
    data_surface_local, latitude = slice_data(data=data_surface_local[:, :, :, :],
                                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                                              dimension_slice=info_netcdf.data_dim.latitude,
                                              value=-75)
    longitude, idx_longitude = slice_data(data=info_netcdf.data_dim.longitude,
                                          idx_dim_slice=1,
                                          dimension_slice=info_netcdf.data_dim.longitude,
                                          value=0)
    if info_netcdf.data_dim.altitude.units == 'Pa':
        index_10 = abs(data_surface_local[0, :, idx_longitude] - 10e3).argmin()
        index_40 = abs(data_surface_local[0, :, idx_longitude] - 40e3).argmin()
        index_80 = abs(data_surface_local[0, :, idx_longitude] - 80e3).argmin()

    for i in range(latitude_south.shape[0]):
        part = (i % data_south.shape[1]) / data_south.shape[1]
        ax[1].plot(data_south[:, i], info_netcdf.data_dim.altitude, label=latitude_south[i], color=cmap(part))
        ax[1].errorbar(data_south[:, i], info_netcdf.data_dim.altitude,
                       xerr=[data_south[:, i] * (1 - 1 / data_stddev_south[:, i]),
                             data_south[:, i] * (1 + 1 * data_stddev_south[:, i])],
                       color=cmap(part))
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].hlines(info_netcdf.data_dim.altitude[index_10], 1e-4, 1e3, ls='--', color='black')
    ax[1].hlines(info_netcdf.data_dim.altitude[index_40], 1e-4, 1e3, ls='--', color='black')
    ax[1].hlines(info_netcdf.data_dim.altitude[index_80], 1e-4, 1e3, ls='--', color='black')
    ax[1].text(1e-4, info_netcdf.data_dim.altitude[index_10], '10 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)
    ax[1].text(1e-4, info_netcdf.data_dim.altitude[index_40], '40 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)
    ax[1].text(1e-4, info_netcdf.data_dim.altitude[index_80], '80 km',
               verticalalignment='bottom',
               horizontalalignment='left', color='black', fontsize=10)

    ax[1].set_ylim(1e3, 1e-2)
    ax[1].set_xlim(1e-4, 1e3)
    ax[1].grid()
    ax[1].legend(loc='best', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

    fig.text(0.5, 0.06, 'Radius size (µm)', ha='center', va='center', fontsize=fontsize)
    fig.text(0.03, 0.5, 'Altitude (Pa)', ha='center', va='center', rotation='vertical', fontsize=fontsize)
    fig.savefig('riceco2_polar_latitudes_structure.png', bbox_inches='tight')
    return


def display_riceco2_top_cloud_altitude(info_netcdf, top_cloud, mola=False):
    from matplotlib.colors import Normalize, DivergingNorm

    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                            selected_time=info_netcdf.local_time)
        data_time = data_ls[idx::len(data_local_time)]
    else:
        data_time = info_netcdf.data_dim.time

    top_cloud, interp_time = linearize_ls(data=top_cloud, data_ls=data_time)
    idx, axis_ls, ls_lin = get_ls_index(interp_time)

    top_cloud = correction_value(data=top_cloud, operator='inf', value=0)

    if mola:
        cmap = 'Spectral'
        # norm = Normalize(vmin=0, vmax=40)
        norm = DivergingNorm(vmin=0, vcenter=10, vmax=40)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_2graph_rows)
        fig.subplots_adjust(right=0.8, hspace=0.05)
        cb = ax[0].pcolormesh(interp_time[:], info_netcdf.data_dim.latitude[:], top_cloud, norm=norm, cmap=cmap)
        ax[0].set_facecolor('white')
        ax[0].set_xticks(interp_time[idx])
        ax[0].set_xticklabels(axis_ls, fontsize=fontsize)
        ax[0].set_yticks(info_netcdf.data_dim.latitude[::8])
        ax[0].set_yticklabels([str(int(x)) for x in info_netcdf.data_dim.latitude[::8]], fontsize=fontsize)

        # MOLA observations
        mola_latitude, mola_ls, mola_altitude = observation_mola(only_location=False)
        ax[1].pcolormesh(mola_ls, mola_latitude, mola_altitude, norm=norm, cmap=cmap)
        ax[1].set_facecolor('white')
        ax[1].set_xticks(mola_ls[::90])
        ax[1].set_xticklabels([str(int(x)) for x in mola_ls[::90]], fontsize=fontsize)
        ax[1].set_yticks(mola_latitude[::30])
        ax[1].set_yticklabels([str(int(x)) for x in mola_latitude[::30]], fontsize=fontsize)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = plt.colorbar(cb, cax=cbar_ax)
        cbar.ax.set_title('km', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[0].set_title(f'Zonal mean of top cloud altitude, diurnal mean', fontsize=fontsize)
        fig.text(0.5, 0.06, 'Solar Longitude (°)', ha='center', va='center', fontsize=fontsize)
        fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=fontsize)

        if len(info_netcdf.local_time) == 1:
            ax[0].set_title(f'Zonal mean of top cloud altitude, at {info_netcdf.local_time:0.f}h', fontsize=fontsize)
            plt.savefig(f'top_cloud_altitude_compared_to_mola_{info_netcdf.local_time:0.f}h.png', bbox_inches='tight')
        else:
            ax[0].set_title(f'Zonal mean of top cloud altitude, diurnal mean', fontsize=fontsize)
            plt.savefig(f'top_cloud_altitude_diurnal_mean_compared_to_mola.png', bbox_inches='tight')
    else:
        cmap = colormap_idl_rainbow_plus_white()
        cmap.set_over("grey")
        norm = Normalize(vmin=0, vmax=10)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
        cb = ax.pcolormesh(interp_time[:], info_netcdf.data_dim.latitude[:], top_cloud, norm=norm, cmap=cmap)
        ax.set_facecolor('white')
        ax.set_xticks(interp_time[idx])
        ax.set_xticklabels(axis_ls)
        ax.set_yticks(info_netcdf.data_dim.latitude[::8])
        ax.set_yticklabels(info_netcdf.data_dim.latitude[::8])

        cbar = plt.colorbar(cb, extend='max')
        cbar.ax.set_title('km', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_title(f'Zonal mean of top cloud altitude, diurnal mean', fontsize=fontsize)
        ax.set_xlabel('Solar Longitude (°)', fontsize=fontsize)
        ax.set_ylabel('Latitude (°N)', fontsize=fontsize)

        if len(info_netcdf.local_time) == 1:
            ax.set_title(f'Zonal mean of top cloud altitude, at {info_netcdf.local_time:0.f}h', fontsize=fontsize)
            plt.savefig(f'top_cloud_altitude_comparable_to_mola_{info_netcdf.local_time:0.f}h.png', bbox_inches='tight')
        else:
            ax.set_title(f'Zonal mean of top cloud altitude, diurnal mean', fontsize=fontsize)
            plt.savefig(f'top_cloud_altitude_comparable_to_mola_diurnal_mean.png', bbox_inches='tight')
    plt.show()
    return


def display_satuco2_thickness_atm_layer(info_netcdf, data_std, save_name):
    from numpy import arange, ma, array
    data = ma.masked_where(info_netcdf.data_target == 0, info_netcdf.data_target)

    # data from Fig 9 in Hu et al., 2012

    north_pole_my28 = array(([0.5, 0.95, 1.4],
                             [0.6, 0.91, 1.3],
                             [0.9, 1.1, 1.38],
                             [0.2, 0.7, 1.15],
                             [0.46, 1.01, 1.58],
                             [0.45, 0.98, 1.50],
                             [0.52, 1.18, 1.79],
                             [0.63, 1.19, 1.63],
                             [0.63, 1.36, 2.07],
                             [0.59, 1.31, 2.06],
                             [0.75, 1.59, 2.4],
                             [0.82, 1.85, 2.85],
                             [0.42, 0.9, 1.3],
                             [1.2, 2.89, 4.5],
                             [1.52, 3.12, 4.7],
                             [0.85, 2.22, 3.63],
                             [0.52, 0.88, 1.20],
                             [0.7, 0.99, 1.28],
                             [0.8, 1.2, 1.6],
                             [0.82, 1.19, 1.50],
                             [0.86, 1.19, 1.46],
                             [0.7, 1.12, 1.59],
                             [0.65, 1.48, 2.28],
                             [0.73, 1.77, 2.78],
                             [0.89, 2.08, 3.25],
                             [1., 2.21, 3.45],
                             [1.09, 2.41, 3.76],
                             [1.04, 2.29, 3.5],
                             [1.08, 2.15, 3.21],
                             [1.20, 2.30, 3.40],
                             [1.18, 2.23, 3.30],
                             [0.81, 1.38, 1.90],
                             [0.71, 1.10, 1.47],
                             )) * 5 / 1.1  # 1.1 cm pour 5 km

    north_pole_my29 = array(([0.6, 1.1, 1.5],
                             [0.7, 1.3, 1.83],
                             [0.6, 1.5, 2.25],
                             [0.9, 1.4, 2.0],
                             [1, 1.35, 1.75],
                             [0.8, 1.4, 2.1],
                             [1.0, 1.5, 2.21],
                             [1.1, 1.6, 2.4],
                             [0.8, 1.5, 2.2],
                             [1.1, 1.95, 2.9],
                             [1.4, 2.3, 3.4],
                             [1.3, 2.15, 3.23],
                             [1.2, 2.35, 3.5],
                             [1.3, 2.5, 3.8],
                             [1.2, 2.6, 3.83],
                             [1.31, 2.7, 4.1],
                             [1.25, 2.7, 4.1],
                             [1.2, 2.5, 3.8],
                             [1.2, 2.4, 3.6],
                             [1.2, 2.45, 3.75],
                             [1.1, 2.4, 3.7],
                             [1, 2.25, 3.35],
                             [0.9, 2, 3.1],
                             [1, 2, 3],
                             [1.05, 2.05, 3.05],
                             [0.95, 1.95, 2.95],
                             [0.8, 1.8, 2.8],
                             )) * 5 / 1.1

    south_pole_my29 = array(([0.63, 0.98, 1.30],
                             [0.9, 1.42, 1.99],
                             [1.3, 1.95, 2.60],
                             [1.43, 2.20, 2.89],
                             [1.51, 2.40, 3.25],
                             [1.40, 2.35, 3.28],
                             [1.49, 2.40, 3.32],
                             [1.52, 2.58, 3.60],
                             [1.69, 2.70, 3.72],
                             [1.68, 2.70, 3.75],
                             [2.10, 3.40, 4.70],
                             [2.23, 3.56, 4.85],
                             [2.18, 3.47, 4.72],
                             [2.49, 3.76, 5.05],
                             [2.40, 3.70, 5.00],
                             [2.30, 3.55, 4.82],
                             [2.30, 3.53, 4.80],
                             [2.21, 3.50, 4.73],
                             [2.10, 3.30, 4.50],
                             [1.81, 2.95, 4.08],
                             [2.08, 3.23, 4.39],
                             [2.19, 3.35, 4.51],
                             [2.23, 3.39, 4.51],
                             [2.22, 3.38, 4.50],
                             [2.15, 3.28, 4.29],
                             [2.00, 3.08, 4.13],
                             [1.18, 2.89, 3.88],
                             [1.92, 2.94, 3.94],
                             [2.30, 3.32, 4.36],
                             [1.28, 2.11, 2.94],
                             [1.21, 2.10, 2.95],
                             [0.92, 1.70, 2.49],
                             [0.77, 1.50, 2.22],
                             [0.18, 0.68, 1.12],
                             [0.12, 0.22, 0.32],
                             )) * 5 / 0.95  # 0.95 cm pour 5 km

    north_pole_ls_my28 = 192.5 + arange(33) * 5
    north_pole_ls_my29 = 197.5 + arange(27) * 5
    south_pole_ls_my29 = 12.5 + arange(35) * 5

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_2graph_rows)
    ax[0].set_title('North pole above 60°N', fontsize=fontsize)
    ax[0].errorbar(arange(data.shape[1]) * 5, data[0, :] / 1e3,
                   yerr=data_std[0, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')  # 72 points binned in 5°
    ax[0].errorbar(north_pole_ls_my29, north_pole_my29[:, 1],
                   yerr=[north_pole_my29[:, 2] - north_pole_my29[:, 1], north_pole_my29[:, 1] - north_pole_my29[:, 0]],
                   color='black',
                   ls=' ', marker='+', label='MCS MY29')
    ax[0].set_xticks(ticks=arange(0, 405, 45))
    ax[0].set_xticklabels(labels=arange(0, 405, 45), fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].legend(loc=2)

    ax[1].set_title('South pole above 60°S', fontsize=fontsize)
    ax[1].errorbar(arange(data.shape[1]) * 5, data[1, :] / 1e3,
                   yerr=data_std[1, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')
    ax[1].errorbar(south_pole_ls_my29, south_pole_my29[:, 1],
                   yerr=[south_pole_my29[:, 2] - south_pole_my29[:, 1], south_pole_my29[:, 1] - south_pole_my29[:, 0]],
                   color='black',
                   ls=' ', marker='+', label='MCS MY29')

    ax[1].set_xticks(ticks=arange(0, 405, 45))
    ax[1].set_xticklabels(labels=arange(0, 405, 45), fontsize=fontsize)
    ax[1].legend(loc='best')
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

    ax[0].set_ylim(0, 40)
    ax[1].set_ylim(0, 40)
    fig.text(0.06, 0.5, 'Thickness (km)', ha='center', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)

    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    return


def display_satuco2_with_co2_ice_altitude_ls(info_netcdf, data_satuco2_north, data_satuco2_eq, data_satuco2_south,
                                             data_co2ice_north, data_co2ice_eq, data_co2ice_south, latitude_north,
                                             latitude_eq, latitude_south, binned):
    from numpy import array, round, ones

    # Info latitude
    list_latitudes = [latitude_north, latitude_eq, latitude_south]

    # Get latitude range between value-1 et value+1
    data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
    data_maven_limb, data_spicam, data_tesmoc, data_themis = mesospheric_clouds_observed()

    list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                data_maven_limb, data_spicam, data_tesmoc, data_themis]

    data_zareoid, altitude_unit, altitude_name, data_surface_local, ticks_altitude = None, None, None, None, None

    if info_netcdf.data_dim.altitude.units == 'm':
        altitude_unit = 'km'
        altitude_name = 'Altitude'
        info_netcdf.data_dim.altitude = info_netcdf.data_dim.altitude / 1e3
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif info_netcdf.data_dim.altitude.units == 'km':
        altitude_unit = info_netcdf.data_dim.altitude.units
        altitude_name = 'Altitude'
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif info_netcdf.data_dim.altitude.units == 'Pa':
        altitude_unit = info_netcdf.data_dim.altitude.units
        altitude_name = 'Pressure'
        data_zareoid, list_var = get_data(filename=info_netcdf.filename, target='zareoid')
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    data_time, list_var = get_data(filename=info_netcdf.filename, target='Time')
    if data_time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        data_local_time, idx, stats_file = check_local_time(data_time=data_time, selected_time=info_netcdf.local_time)
        data_time = data_ls[idx::len(data_local_time)]

    if binned.lower() == 'y' and data_zareoid is not None:
        data_time = data_time[::60]  # 5°Ls binned
        data_zareoid = data_zareoid[::12, :, :, :]
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    norm_satu = None  # TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100)
    levels_satu = array([1, 10, 20, 50, 100])
    levels_co2 = None
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize_3graph_rows)
    ax[0].set_title(f'{latitude_north}°N', fontsize=fontsize)
    ax[0].contourf(data_time[:], info_netcdf.data_dim.altitude, data_satuco2_north, norm=norm_satu, cmap='coolwarm',
                   levels=levels_satu, extend='max')
    ax[0].contour(data_time[:], info_netcdf.data_dim.altitude, data_co2ice_north, norm=None, levels=levels_co2,
                  colors='black')

    ax[1].set_title(f'{latitude_eq}°N', fontsize=fontsize)
    ax[1].contourf(data_time[:], info_netcdf.data_dim.altitude, data_satuco2_eq, norm=norm_satu, cmap='coolwarm',
                   levels=levels_satu, extend='max')
    ax[1].contour(data_time[:], info_netcdf.data_dim.altitude, data_co2ice_eq, norm=None, levels=levels_co2,
                  colors='black')

    ax[2].set_title(f'{abs(latitude_south)}°S', fontsize=fontsize)
    cb = ax[2].contourf(data_time[:], info_netcdf.data_dim.altitude, data_satuco2_south, norm=norm_satu,
                        cmap='coolwarm', levels=levels_satu, extend='max')
    ax[2].contour(data_time[:], info_netcdf.data_dim.altitude, data_co2ice_south, norm=None, levels=levels_co2,
                  colors='black')

    for i, axe in enumerate(ax):
        #        axe.set_xticks(ticks=axis_ls)
        #        axe.set_xticklabels(labels=axis_ls)
        axe.set_xlim(0, 360)
        axe.set_ylim(1e-3, 1e3)

        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(data_obs=value, dim='latitude',
                                                                         data_dim=info_netcdf.data_dim.altitude,
                                                                         value=list_latitudes[i])
            if data_obs_ls.shape[0] != 0:
                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data=data_surface_local,
                                                        idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                        dimension_slice=info_netcdf.data_dim.latitude,
                                                        value=list_latitudes[i])

            lines_altitudes_0km = get_mean_index_altitude(data_surface_local_sliced, value=0, dimension='Time')
            lines_altitudes_10km = get_mean_index_altitude(data_surface_local_sliced, value=1e4, dimension='Time')
            lines_altitudes_40km = get_mean_index_altitude(data_surface_local_sliced, value=4e4, dimension='Time')
            lines_altitudes_80km = get_mean_index_altitude(data_surface_local_sliced, value=8e4, dimension='Time')
            del data_surface_local_sliced

            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_0km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_10km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_40km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_80km], '-', color='grey', linewidth=0.5)

            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_0km[0]], '0 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_10km[0]], '10 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_40km[0]], '40 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_80km[0]], '80 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)

            axe.set_yscale('log')
            axe.invert_yaxis()
        else:
            axe.set_yticks(ticks=ticks_altitude)
            axe.set_yticklabels(labels=round(info_netcdf.data_dim.altitude))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cb, cax=cbar_ax)

    fig.text(0.02, 0.5, f'{altitude_name} ({altitude_unit})', ha='center', va='center', rotation='vertical',
             fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)

    save_name = f'satuco2_zonal_mean_with_co2ice_at_{latitude_north}N_{latitude_eq}N_{latitude_south}N'
    if binned.lower() == 'y':
        save_name = f'{save_name}_binned'

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_satuco2_with_co2_ice_altitude_longitude(info_netcdf, data_satuco2_north, data_satuco2_south,
                                                    data_co2ice_north,
                                                    data_co2ice_south, latitude_north, latitude_south, binned):
    from numpy import array, round, ones

    # Info latitude
    list_latitudes = [latitude_north, latitude_south]

    list_time_range = array(([270, 300], [0, 30]))

    # Info longitude

    # Get latitude range between value-1 et value+1
    data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
    data_maven_limb, data_spicam, data_tesmoc, data_themis = mesospheric_clouds_observed()

    list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                data_maven_limb, data_spicam, data_tesmoc, data_themis]

    data_surface_local, ticks_altitude = None, None

    if info_netcdf.data_dim.altitude.units == 'm':
        altitude_unit = 'km'
        altitude_name = 'Altitude'
        info_netcdf.data_dim.altitude = info_netcdf.data_dim.altitude / 1e3
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif info_netcdf.data_dim.altitude.units == 'km':
        altitude_unit = info_netcdf.data_dim.altitude.units
        altitude_name = 'Altitude'
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    else:
        altitude_unit = info_netcdf.data_dim.altitude.units
        altitude_name = 'Pressure'
        data_zareoid, list_var = get_data(filename=info_netcdf.filename, target='zareoid')
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    norm_satu, levels_co2 = None, None
    levels_satu = array([1, 10, 20, 50, 100])

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_1graph)
    ax[0].set_title(f'{latitude_north}°N', fontsize=fontsize)
    ax[0].contourf(info_netcdf.data_dim.longitude, info_netcdf.data_dim.altitude, data_satuco2_north, norm=norm_satu,
                   cmap='coolwarm', levels=levels_satu, extend='max')
    ax[0].contour(info_netcdf.data_dim.longitude, info_netcdf.data_dim.altitude, data_co2ice_north, norm=None,
                  levels=levels_co2, colors='black')

    ax[1].set_title(f'{abs(latitude_south)}°S', fontsize=fontsize)
    cb = ax[1].contourf(info_netcdf.data_dim.longitude, info_netcdf.data_dim.altitude, data_satuco2_south,
                        norm=norm_satu, cmap='coolwarm', levels=levels_satu, extend='max')
    ax[1].contour(info_netcdf.data_dim.longitude, info_netcdf.data_dim.altitude, data_co2ice_south, norm=None,
                  levels=levels_co2, colors='black')

    for i, axe in enumerate(ax):
        axe.set_ylim(1e-3, 1e3)

        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(data_obs=value, dim='latitude',
                                                                         data_dim=info_netcdf.data_dim.latitude,
                                                                         value=list_latitudes[i])
            if data_obs_ls.shape[0] != 0:
                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data=data_surface_local,
                                                        idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                        dimension_slice=info_netcdf.data_dim.latitude,
                                                        value=list_latitudes[i])

            data_surface_local_sliced, tmp = slice_data(data=data_surface_local_sliced,
                                                        idx_dim_slice=info_netcdf.idx_dim.time,
                                                        dimension_slice=info_netcdf.data_dim.time,
                                                        value=list_time_range[i])

            lines_altitudes_0km = get_mean_index_altitude(data_surface_local_sliced, value=0, dimension='longitude')
            lines_altitudes_10km = get_mean_index_altitude(data_surface_local_sliced, value=1e4, dimension='longitude')
            lines_altitudes_40km = get_mean_index_altitude(data_surface_local_sliced, value=4e4, dimension='longitude')
            lines_altitudes_80km = get_mean_index_altitude(data_surface_local_sliced, value=8e4, dimension='longitude')
            del data_surface_local_sliced

            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_0km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_10km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_40km], '-', color='grey', linewidth=0.5)
            axe.plot(info_netcdf.data_dim.altitude[lines_altitudes_80km], '-', color='grey', linewidth=0.5)

            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_0km[0]], '0 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_10km[0]], '10 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_40km[0]], '40 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, info_netcdf.data_dim.altitude[lines_altitudes_80km[0]], '80 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)

            axe.set_yscale('log')
            axe.invert_yaxis()
        else:
            axe.set_yticks(ticks=ticks_altitude)
            axe.set_yticklabels(labels=round(info_netcdf.data_dim.altitude))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cb, cax=cbar_ax)

    fig.text(0.02, 0.5, f'{altitude_name} ({altitude_unit})', ha='center', va='center', rotation='vertical',
             fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)

    save_name = f'satuco2_time_mean_with_co2ice_at_{latitude_north}N_{latitude_south}N'
    if binned.lower() == 'y':
        save_name = f'{save_name}_binned'

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_satuco2_zonal_mean_day_night(data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night,
                                         data_altitude, ndx, axis_ls, title, save_name):
    from numpy import array, round
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_1graph)
    fig.suptitle(title)
    ax[0].set_title('Day 6h - 18h', fontsize=fontsize)
    ax[0].contourf(data_satuco2_day.T, norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100), cmap='seismic',
                   levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[0].contour(data_co2ice_day.T, colors='black')

    ax[1].set_title('Night 18h - 6h', fontsize=fontsize)
    cb = ax[1].contourf(data_satuco2_night.T, norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100), cmap='seismic',
                        levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[1].contour(data_co2ice_night.T, colors='black')

    ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    for axe in ax:
        axe.set_xticks(ticks=ndx)
        axe.set_xticklabels(labels=axis_ls)

        axe.set_yticks(ticks=ticks_altitude)
        axe.set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cb, cax=cbar_ax)

    fig.text(0.02, 0.5, 'Altitude (km)', ha='center', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_satuco2_maxval_with_maxalt(info_netcdf, data_maxval, data_altval):
    from matplotlib.colors import LogNorm
    cmap = plt.get_cmap('inferno')
    cmap.set_under('white')
    cmap_revert = plt.get_cmap('inferno_r')
    cmap_revert.set_under('white')
    norm_val = LogNorm(vmin=1e0, vmax=1e6)
    norm_alt = LogNorm(vmin=1e-3, vmax=1e3)

    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                        selected_time=info_netcdf.local_time)

    data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
        if info_netcdf.data_dim.time.units != 'deg':
            data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = info_netcdf.data_dim.time[idx::len(data_local_time)]

        data_maxval, tmp = linearize_ls(data=data_maxval, data_ls=data_time)
        data_altval, data_time = linearize_ls(data=data_altval, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
    else:
        data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
        data_time = data_ls[idx::len(data_local_time)]
        data_maxval, tmp = linearize_ls(data=data_maxval, data_ls=data_time)
        data_altval, data_time = linearize_ls(data=data_altval, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize_2graph_cols)
    axes[0].set_title('Max value of CO$_2$ saturation\n(zonal and diurnal mean)')
    pc1 = axes[0].pcolormesh(data_time, info_netcdf.data_dim.latitude, data_maxval, norm=norm_val, cmap=cmap)
    cb1 = plt.colorbar(pc1, ax=axes[0])
    cb1.ax.tick_params(labelsize=fontsize)

    axes[1].set_title('Altitude corresponding')
    pc2 = axes[1].pcolormesh(data_time, info_netcdf.data_dim.latitude, data_altval, norm=norm_alt, cmap=cmap_revert)
    cb2 = plt.colorbar(pc2, ax=axes[1])
    cb2.ax.set_title('Pa', fontsize=fontsize)
    cb2.ax.tick_params(labelsize=fontsize)

    for ax in axes.reshape(-1):
        ax.set_xticks(ticks=ndx)
        ax.set_xticklabels(axis_ls, fontsize=fontsize)
        ax.set_yticks(ticks=info_netcdf.data_dim.latitude[::4])
        ax.set_yticklabels(labels=[str(int(x)) for x in info_netcdf.data_dim.latitude[::4]], fontsize=fontsize)
        ax.set_xlabel('Solar longitude (°)', fontsize=fontsize)
        ax.set_ylabel('Latitude (°N)', fontsize=fontsize)
    plt.savefig('satuco2_maxval_with_maxalt_zonalmean_diurnalmean.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig6(info_netcdf, data_localtime):
    from numpy import arange

    cmap = colormap_idl_rainbow_plus_white()

    fig = plt.figure(figsize=figsize_1graph)
    pc = plt.contourf(data_localtime, info_netcdf.data_dim.altitude / 1e3, info_netcdf.data_target.T,
                      levels=arange(0, 125, 5), cmap=cmap)
    plt.ylim(0, 120)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title('K')
    plt.title('Temperature - Tcond CO$_2$', fontsize=fontsize)
    plt.ylabel('Altitude above areoid (km)', fontsize=fontsize)
    plt.xlabel('Local time (h)', fontsize=fontsize)
    plt.savefig('temp_altitude_localtime_ls120-150_lat0N_lon0E_gg2011_fig6.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig7(info_netcdf, data_altitude):
    cmap = colormap_idl_rainbow_plus_white()

    # plot
    fig, ax = plt.subplots(figsize=figsize_1graph)
    ax.set_title('Temperature - Tcond CO$_2$', fontsize=fontsize)
    ctf = ax.contourf(info_netcdf.data_dim.latitude, data_altitude / 1e3, info_netcdf.data_target,
                      levels=arange(0, 130, 10), cmap=cmap)
    ax.set_ylim(40, 120)

    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('K', fontsize=fontsize)

    ax.set_xlabel('Latitude (°N)', fontsize=fontsize)
    ax.set_ylabel('Altitude above areoid (km)', fontsize=fontsize)

    plt.savefig('temp_zonal_mean_altitude_latitude_ls_0-30_gg2011_fig7.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig8(info_netcdf, data_thermal_tides):
    fig, ax = plt.subplots(nrows=2, figsize=figsize_1graph)

    cmap = colormap_idl_rainbow_plus_white()
    # Zonal mean at 16 H local time
    ctf1 = ax[0].contourf(info_netcdf.data_dim.latitude, info_netcdf.data_dim.altitude / 1e3, info_netcdf.data_target,
                          levels=arange(110, 240, 10), cmap=cmap)
    cbar = plt.colorbar(ctf1, ax=ax[0])
    cbar.ax.set_title('K', fontsize=fontsize)
    ax[0].set_ylim(40, 120)
    ax[0].set_xlabel('Latitude (°N)', fontsize=fontsize)
    ax[0].set_ylabel('Altitude above areoid (km)', fontsize=fontsize)

    # Thermal tides: 12h - 00h
    ctf2 = ax[1].contourf(info_netcdf.data_dim.latitude, info_netcdf.data_dim.altitude / 1e3, data_thermal_tides,
                          levels=arange(-20, 32, 4), cmap=cmap)
    cbar2 = plt.colorbar(ctf2, ax=ax[1])
    cbar2.ax.set_title('K', fontsize=fontsize)
    ax[1].set_ylim(40, 120)

    ax[1].set_xlabel('Latitude (°N)', fontsize=fontsize)
    ax[1].set_ylabel('Altitude above areoid (km)', fontsize=fontsize)
    plt.savefig('temp_gg2011_fig8.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig9(info_netcdf, data_altitude):
    cmap = colormap_idl_rainbow_plus_white()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    pc = ax.contourf(info_netcdf.data_dim.longitude, data_altitude / 1e3, info_netcdf.data_target,
                     levels=arange(0, 130, 10), cmap=cmap)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title('K', fontsize=fontsize)
    ax.set_ylim(40, 120)

    plt.title('Temperature -Tcond CO$_2$', fontsize=fontsize)
    plt.xlabel('Longitude (°E)', fontsize=fontsize)
    plt.ylabel('Altitude above areoid (km)', fontsize=fontsize)
    plt.savefig('temp_altitude_longitude_ls_0-30_LT_16H_lat_0N_gg2011_fig9.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_structure_polar_region(info_netcdf, data_north, data_south, norm, levels, unit, save_name):
    cmap = 'coolwarm'

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize_1graph)

    ax[0].set_title('North pole at 60°N', fontsize=fontsize)
    ctf = ax[0].contourf(info_netcdf.data_dim.time, info_netcdf.data_dim.altitude, data_north.T, norm=norm,
                         levels=levels, cmap=cmap)

    ax[1].set_title('South pole at 60°S', fontsize=fontsize)
    ax[1].contourf(info_netcdf.data_dim.time, info_netcdf.data_dim.altitude, data_south.T, norm=norm, levels=levels,
                   cmap=cmap)

    for axes in ax.reshape(-1):
        axes.set_yscale('log')
        axes.invert_yaxis()
        axes.set_ylim(info_netcdf.data_dim.altitude[0], 1e1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.ax.set_title(unit)

    fig.text(0.02, 0.5, f'{info_netcdf.data_dim.altitude.name} ({info_netcdf.data_dim.altitude.units})', ha='center',
             va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_cold_pocket_spicam(info_netcdf, title, save_name):
    if info_netcdf.data_dim.time.units != 'deg':
        data_time, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    else:
        data_time = info_netcdf.data_dim.time

    data_time, local_time = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_time)

    data, interp_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)

    fig, axes = plt.subplot(figsize=figsize_1graph, projection='polar')
    axes.set_title(title, fontsize=fontsize)
    # TODO: not finished !
    axes.polar(interp_time, )

    axes.set_xlabel('Solar longitude (°)', fontsize=fontsize)
    axes.set_ylabel(f'Pressure ({info_netcdf.data_dim.altitude.units})', fontsize=fontsize)
    axes.set_yscale('log')
    axes.invert_yaxis()

    fig.savefig(f'{save_name}.png')
    fig.show()
    return


def display_vars_altitude_variable(info_netcdf, data_latitude, data_pressure, title):
    plt.figure(figsize=figsize_1graph)
    plt.title(title, fontsize=fontsize)

    plt.semilogy(info_netcdf.data_target[:, :], data_pressure[0, :, :, 0], label=data_latitude)

    plt.xlabel('K', fontsize=fontsize)
    plt.ylabel('Pressure (Pa)', fontsize=fontsize)
    plt.legend(loc='best')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('vertical_profile_temperature_equator.png', bbox_inches='tight')
    plt.show()
    return


def display_vars_altitude_latitude(info_netcdf, unit, title, save_name):
    if unit == 'pr.µm':
        info_netcdf.data_target = info_netcdf.data_target * 1e3  # convert kg/m2 to pr.µm

    cmap = 'hot'

    # plot
    fig, ax = plt.subplots(figsize=figsize_1graph)
    ax.set_title(title, fontsize=fontsize)
    ax.set_yscale('log')
    ctf = ax.contourf(info_netcdf.data_dim.latitude, info_netcdf.data_dim.altitude, info_netcdf.data_target, cmap=cmap)
    ax.invert_yaxis()

    cbar = plt.colorbar(ctf)
    cbar.ax.set_title(unit, fontsize=fontsize)

    ax.set_xlabel('Latitude (°N)', fontsize=fontsize)
    ax.set_ylabel(f'{info_netcdf.data_dim.altitude.name} ({info_netcdf.data_dim.altitude.units})')

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_vars_altitude_localtime(info_netcdf, title, unit, save_name):
    from numpy import arange, zeros
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=figsize_1graph)
    if unit == '':
        scale = zeros(5)
        scale[1] = 1
        scale[2:] = 10 ** arange(1, 4)
        pc = plt.contourf(info_netcdf.local_time, info_netcdf.data_dim.altitude, info_netcdf.data_target.T,
                          norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=1000), levels=scale, cmap='seismic', extend='max')
    else:
        plt.yscale('log')
        pc = plt.contourf(info_netcdf.local_time, info_netcdf.data_dim.altitude, info_netcdf.data_target.T,
                          levels=arange(0, 140, 10), cmap='seismic')
        ax = plt.gca()
        ax.invert_yaxis()

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.ylabel(f'{info_netcdf.data_dim.altitude.name} ({info_netcdf.data_dim.altitude.units})', fontsize=fontsize)
    plt.xlabel('Local time (h)', fontsize=fontsize)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.show()
    return


def display_vars_altitude_longitude(info_netcdf, unit, norm, vmin, vcenter, vmax, title, save_name):
    from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm

    if info_netcdf.data_dim.altitude.units == 'Pa':
        yscale = 'log'
    else:
        yscale = 'linear'

    if norm == 'div':
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = 'coolwarm'
    elif norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'plasma'
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'plasma'

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    pc = ax.pcolormesh(info_netcdf.data_dim.longitude, info_netcdf.data_dim.altitude, info_netcdf.data_target,
                       norm=norm, cmap=cmap, shading='auto')

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_yscale(yscale)
    if info_netcdf.data_dim.altitude.units == 'Pa':
        ax.invert_yaxis()
    ax.set_xticks(info_netcdf.data_dim.longitude[::8])
    ax.set_xticklabels(labels=info_netcdf.data_dim.longitude[::8], fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Longitude (°E)', fontsize=fontsize)
    ax.set_ylabel(f'Altitude ({info_netcdf.data_dim.altitude.units})', fontsize=fontsize)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_altitude_ls(info_netcdf, varname_1, shortname_1, latitude, norm, unit,
                             altitude_min, altitude_max, vmin, vmax, title, save_name, data_2=None, norm_2=None,
                             vmin_2=None, vmax_2=None, varname_2=None, shortname_2=None, alti_line=None):
    from numpy import round
    from numpy.ma import masked_outside
    from matplotlib.colors import LogNorm, Normalize

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    if norm_2 == 'log':
        norm_2 = LogNorm(vmin=vmin_2, vmax=vmax_2)
    else:
        norm_2 = Normalize(vmin=vmin_2, vmax=vmax_2)

    longitude, idx_longitude = slice_data(data=info_netcdf.data_dim.longitude,
                                          idx_dim_slice=0,
                                          dimension_slice=info_netcdf.data_dim.longitude,
                                          value=0)

    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                        selected_time=info_netcdf.local_time)
    if info_netcdf.data_dim.altitude.units == 'm':
        unit_altitude = 'm'
        altitude_name = 'Altitude'
        data_surface_local = info_netcdf.data_dim.altitude[:] * 1e3
    elif info_netcdf.data_dim.altitude.units == 'km':
        unit_altitude = info_netcdf.data_dim.altitude.units
        altitude_name = 'Altitude'
        data_surface_local = info_netcdf.data_dim.altitude[:]
    else:
        unit_altitude = info_netcdf.data_dim.altitude.units
        altitude_name = 'Pressure'
        data_zareoid, list_var = get_data(filename=info_netcdf.filename, target='zareoid')
        data_zareoid = data_zareoid[idx::len(data_local_time), :, :, :]
        data_surface_local = gcm_surface_local(data_zareoid=data_zareoid[:, :, :, :])
        if len(latitude) > 1:
            data_surface_local, tmp = slice_data(data=data_surface_local[:, :, :, :],
                                                 dimension_slice=info_netcdf.data_dim.latitude,
                                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                 value=(latitude[0] + latitude[-1]) / 2.)
        else:
            data_surface_local, idx_latitude = slice_data(data=data_surface_local[:, :, :, :],
                                                          dimension_slice=info_netcdf.data_dim.latitude,
                                                          idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                          value=latitude)
            latitude = info_netcdf.data_dim.latitude[idx_latitude - 1: idx_latitude + 2]

    data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
        if info_netcdf.data_dim.time.units != 'deg':
            data_ls = data_ls[idx::len(data_local_time)]
        else:
            data_ls = info_netcdf.data_dim.time[idx::len(data_local_time)]
        data_1, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_ls)
        if isinstance(data_2, ndarray):
            data_2, data_time = linearize_ls(data=data_2, data_ls=data_ls)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
    else:
        data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
        data_time = data_ls[idx::len(data_local_time)]
        data_1, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
        if isinstance(data_2, ndarray):
            data_2, data_time = linearize_ls(data=data_2, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])

    if alti_line:
        if info_netcdf.data_dim.altitude.units == 'Pa':
            index_10 = abs(data_surface_local[0, :, idx_longitude] - 10e3).argmin()
            index_40 = abs(data_surface_local[0, :, idx_longitude] - 40e3).argmin()
            index_80 = abs(data_surface_local[0, :, idx_longitude] - 80e3).argmin()
        else:
            index_10 = abs(data_surface_local[idx_longitude] - 10e3).argmin()
            index_40 = abs(data_surface_local[idx_longitude] - 40e3).argmin()
            index_80 = abs(data_surface_local[idx_longitude] - 80e3).argmin()
    else:
        index_10, index_40, index_80 = None, None, None

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    cb = axes.pcolormesh(data_time[:], info_netcdf.data_dim.altitude[:], data_1, norm=norm, cmap='inferno',
                         shading='auto')  # autumn
    if isinstance(data_2, ndarray):
        cb2 = axes.pcolormesh(data_time[:], info_netcdf.data_dim.altitude[:], data_2, norm=norm_2, cmap='winter',
                              shading='auto')
        cbar2 = plt.colorbar(cb2, ax=axes)
        cbar2.ax.set_title(unit, fontsize=fontsize)
        cbar2.ax.tick_params(labelsize=fontsize)

    if unit_altitude == 'Pa':
        axes.set_yscale('log')
        axes.invert_yaxis()
        axes.set_ylim(altitude_min, altitude_max)
    else:
        axes.set_ylim(0, altitude_max)
        axes.set_yticklabels(labels=round(info_netcdf.data_dim.altitude[:], 0), fontsize=fontsize)

    cbar = plt.colorbar(cb, ax=axes)
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    axes.set_xticks(ticks=ndx)
    axes.set_xticklabels(axis_ls, fontsize=fontsize)

    axes.set_title(title, fontsize=fontsize)
    axes.set_xlabel('Solar longitude (°)', fontsize=fontsize)
    axes.set_ylabel(f'{altitude_name} ({unit_altitude})', fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    if alti_line:
        axes.hlines(info_netcdf.data_dim.altitude[index_10], data_time[0], data_time[-1], ls='--', color='black')
        axes.hlines(info_netcdf.data_dim.altitude[index_40], data_time[0], data_time[-1], ls='--', color='black')
        axes.hlines(info_netcdf.data_dim.altitude[index_80], data_time[0], data_time[-1], ls='--', color='black')
        axes.text(300, info_netcdf.data_dim.altitude[index_10], '10 km', verticalalignment='bottom',
                  horizontalalignment='left', color='black', fontsize=12, weight='bold')
        axes.text(300, info_netcdf.data_dim.altitude[index_40], '40 km', verticalalignment='bottom',
                  horizontalalignment='left', color='black', fontsize=12, weight='bold')
        axes.text(300, info_netcdf.data_dim.altitude[index_80], '80 km', verticalalignment='bottom',
                  horizontalalignment='left', color='black', fontsize=12, weight='bold')

    # observation
    list_instrument = ['HRSC', 'OMEGAlimb', 'OMEGAnadir', 'SPICAM', 'THEMIS', 'NOMAD']
    list_marker = ['s', 'o', 'v', 'P', 'X', '1']
    list_colors = ['red', 'red', 'red', 'red', 'red', 'red']
    for i, value_i in enumerate(list_instrument):
        data_ls, data_lat, data_lon, data_lt, data_alt, data_alt_min, data_alt_max = \
            mesospheric_clouds_altitude_localtime_observed(instrument=value_i)
        mask = masked_outside(data_lat, latitude[0], latitude[-1])
        if not all(mask.mask):
            for j in range(data_alt[mask.mask].shape[0]):
                if data_alt[mask.mask][j] != 0:
                    index = abs(data_surface_local[0, :, idx_longitude] - data_alt[mask.mask][j] * 1e3).argmin()
                    axes.scatter(data_ls[mask.mask][j], info_netcdf.data_dim.altitude[index], color=list_colors[i],
                                 marker=list_marker[i], label=value_i, s=64)

    fig.savefig(f'{save_name}.png', bbox_inches='tight')

    dict_var = [{'data': data_time[:], 'varname': 'Solar longitude', 'units': 'deg', 'shortname': 'TIME'},
                {'data': info_netcdf.data_dim.altitude[:], 'varname': f"{altitude_name}", 'units': f"{unit_altitude}",
                 'shortname': "ALTITUDE"},
                {'data': data_1, 'varname': f"{varname_1}", 'units': f"{unit}", 'shortname': f"{shortname_1}"},
                ]
    if alti_line:
        dict_var.append({'data': array([index_10, index_40, index_80]),
                         'varname': f"altitude index [10, 40, 80] km above local surface", 'units': "km",
                         'shortname': "idx_km"})

    if isinstance(data_2, ndarray):
        dict_var.append({'data': data_2, 'varname': f"{varname_2}", 'units': f"{unit}", 'shortname': f"{shortname_2}"})

    save_figure_data(list_dict_var=dict_var, savename=save_name)
    return


def display_vars_latitude_longitude(info_netcdf, unit, norm, vmin, vmax, title, save_name):
    from matplotlib.colors import LogNorm, Normalize

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph_xtend)
    ax.set_title(title, fontsize=fontsize)
    ctf = ax.pcolormesh(info_netcdf.data_dim.longitude, info_netcdf.data_dim.latitude, info_netcdf.data_target,
                        norm=norm, cmap='plasma', shading='auto')
    ax.set_xticks(info_netcdf.data_dim.longitude[::8])
    ax.set_xticklabels([str(int(x)) for x in info_netcdf.data_dim.longitude[::8]], fontsize=fontsize)
    ax.set_yticks(info_netcdf.data_dim.latitude[::4])
    ax.set_yticklabels([str(int(x)) for x in info_netcdf.data_dim.latitude[::4]], fontsize=fontsize)
    ax.set_xlabel('Longitude (°E)', fontsize=fontsize)
    ax.set_ylabel('Latitude (°N)', fontsize=fontsize)
    cbar = fig.colorbar(ctf)
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.grid()
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_latitude_ls(info_netcdf, unit, norm, vmin, vmax, cmap, observation=False,
                             latitude_selected=None, title=None, tes=None, mvals=None,
                             layer=None, save_name='test'):
    from matplotlib.colors import LogNorm, Normalize, DivergingNorm, BoundaryNorm
    from matplotlib import cm
    idx1, idx2 = None, None

    n_subplot = 1

    if tes:
        n_subplot += 1
    if mvals:
        n_subplot += 1

    cmap = cm.get_cmap(cmap)

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        extend = False
    elif norm == 'div':
        norm = DivergingNorm(vmin=vmin, vmax=vmax)
        extend = False
    elif norm == 'set':
        norm = BoundaryNorm([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2], ncolors=cmap.N, clip=False)
        cmap.set_over('red')
        extend = True
    else:
        extend = False
        norm = Normalize(vmin=vmin, vmax=vmax)

    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                        selected_time=info_netcdf.local_time)

    data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
    if info_netcdf.data_dim.time.shape[0] == data_ls.shape[0]:
        if info_netcdf.data_dim.time.units != 'deg':
            data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = info_netcdf.data_dim.time[idx::len(data_local_time)]

        info_netcdf.data_target, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)
    else:
        data_ls = data_ls[:info_netcdf.data_dim.time.shape[0]]
        data_time = data_ls[idx::len(data_local_time)]
        info_netcdf.data_target, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
        ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time, tab_ls=data_time[::20])

    if latitude_selected is not None:
        data_latitude, latitude_selected = slice_data(data=info_netcdf.data_dim.latitude,
                                                      dimension_slice=info_netcdf.data_dim.latitude,
                                                      idx_dim_slice=0,
                                                      value=[latitude_selected[0], latitude_selected[-1]])
    else:
        latitude_selected = [-90, 90]
        data_latitude = info_netcdf.data_dim.latitude[:]

    # PLOT
    fig, ax = plt.subplots(nrows=n_subplot, ncols=1, figsize=figsize_1graph)
    fig.subplots_adjust()

    i_subplot = 0
    if tes:
        # Extract TES data
        print('Takes TES data')
        if info_netcdf.name_target == 'tsurf':
            data_time_tes = observation_tes(target='time', year=None)  # None == climatology
            data_latitude_tes = observation_tes(target='latitude', year=None)

            # (time, latitude, longitude), also Tsurf_day/Tsurf_nit
            data_tes = observation_tes(target='Tsurf_day', year=None)
            ax[i_subplot].set_title('TES climatology', fontsize=fontsize)
            idx1 = (abs(data_time_tes[:] - 360 * 1)).argmin()
            idx2 = (abs(data_time_tes[:] - 360 * 2)).argmin()
            data_time_tes = data_time_tes[idx1:idx2] - 360
            data_tes = mean(data_tes, axis=2).T
        elif info_netcdf.name_target == 'temp':
            data_time_tes = observation_tes(target='time', year=25)
            data_latitude_tes = observation_tes(target='latitude', year=25)
            data_altitude_tes = observation_tes(target='altitude', year=25)  # Pa
            data_tes = tes(target='T_limb_day', year=25)
            year = 1

            # Select altitude for TES data close to the specified layer
            target_layer = info_netcdf.data_dim.altitude[::-1][layer]
            in_range = ma.masked_outside(data_altitude_tes[:], target_layer, target_layer)
            if not in_range.all():
                data_tes = zeros((data_tes.shape[2], data_tes.shape[0]))
                altitude_tes = 'out range'
            else:
                idx = abs(data_altitude_tes[:] - info_netcdf.data_dim.altitude[::-1][layer]).argmin()
                data_tes = data_tes[:, idx, :, :]
                data_tes, tmp = vars_zonal_mean(data_input=data_tes[:, :, :])
                altitude_tes = data_altitude_tes[idx]

            data_tes, tmp = vars_zonal_mean(data_input=data_tes[:, :, :])
            ax[i_subplot].set_title(f'TES Mars Year {24 + year:d} at {altitude_tes} {data_altitude_tes.units}',
                                    fontsize=fontsize)
        else:
            data_time_tes = observation_tes(target='time', year=None)
            data_latitude_tes = observation_tes(target='latitude', year=None)
            data_tes = None
            print('No case for TES, to be check!')

        ax[i_subplot].pcolormesh(data_time_tes, data_latitude_tes[:], data_tes[:, idx1:idx2], norm=norm,
                                 shading='flat', cmap=cmap)
        i_subplot += 1

    if mvals:
        # Extract mvals data
        print('Takes M. Vals data')
        data_mvals = simulation_mvals(target=info_netcdf.name_target, localtime=2)
        data_time_mvals = simulation_mvals(target='Time', localtime=2)
        data_latitude_mvals = simulation_mvals(target='latitude', localtime=2)
        data_altitude_mvals = simulation_mvals(target='altitude', localtime=2)
        if layer is not None:
            data_mvals = data_mvals[:, layer, :, :]

        if data_mvals.ndim == 3:
            data_mvals = correction_value(data_mvals[:, :, :], operator='inf', value=threshold)
        else:
            data_mvals = correction_value(data_mvals[:, :, :, :], operator='inf', value=threshold)

        # Compute zonal mean
        data_mvals, tmp = vars_zonal_mean(data_input=data_mvals, layer=layer)

        if info_netcdf.name_target == 'temp':
            ax[i_subplot].set_title(f'M. VALS at {data_altitude_mvals[layer]:.2e} {data_altitude_mvals.units}',
                                    fontsize=fontsize)
        else:
            ax[i_subplot].set_title('M. VALS', fontsize=fontsize)

        ax[i_subplot].pcolormesh(data_time_mvals[:], data_latitude_mvals[:], data_mvals, shading='auto', cmap=cmap)
        i_subplot += 1

    if i_subplot == 0:
        ctf = ax.pcolormesh(data_time[:], data_latitude[:], info_netcdf.data_target, norm=norm, cmap=cmap,
                            shading='flat', zorder=10)
    else:
        ctf = ax[i_subplot].pcolormesh(data_time[:], data_latitude[:], info_netcdf.data_target, norm=norm, cmap=cmap,
                                       shading='flat',
                                       zorder=10)
    ax.set_xlim(0, 360)
    # Seasonal boundaries caps
    north_cap_ls, north_cap_boundaries, north_cap_boundaries_error, south_cap_ls, south_cap_boundaries, \
    south_cap_boundaries_error = boundaries_seasonal_caps()
    if i_subplot == 0:
        ax.plot(north_cap_ls, north_cap_boundaries, color='black', zorder=11)
        ax.plot(south_cap_ls, south_cap_boundaries, color='black', zorder=11)

    if info_netcdf.target_name == 'temp':
        if i_subplot == 0:
            ax.set_title(title, fontsize=fontsize)
        else:
            ax[i_subplot].set_title(f'My work at {info_netcdf.data_dim.altitude[::-1][layer]:.2e}'
                                    f' {info_netcdf.data_dim.altitude.units}',
                                    fontsize=fontsize)
    else:
        if i_subplot == 0:
            ax.set_title(title, fontsize=fontsize)
        else:
            ax[i_subplot].set_title('My work', fontsize=fontsize)

    if observation:
        # Get latitude range between entre value-1 et value+1
        data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
        data_maven_limb, data_spicam, data_tesmoc, data_themis, data_nomad = mesospheric_clouds_observed()

        list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                    data_spicam, data_tesmoc, data_themis, data_nomad]  # data_maven_limb
        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         [latitude_selected[0], latitude_selected[-1]])
            if data_obs_ls.shape[0] != 0:
                plt.scatter(data_obs_ls, data_obs_latitude, color='black', marker='o', s=3, zorder=10000,
                            label='Meso')

    #        mola_latitude, mola_ls, mola_altitude = observation_mola(only_location=True)
    #        plt.scatter(mola_ls, mola_latitude, color='red', marker='o', zorder=10000, s=3, label='Tropo')

    if i_subplot != 0:
        for axes in ax.reshape(-1):
            axes.set_facecolor('white')
            axes.set_xticks(ticks=axis_ls)
        fig.text(0.02, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=fontsize)
        fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=fontsize)
        #        fig.suptitle(title, fontsize=fontsize)
        pos0 = ax[0].get_position()
        cb_axes = fig.add_axes([pos0.x0, 0.95, pos0.x1 - pos0.x0, 0.025])
        cbar = plt.colorbar(ctf, cax=cb_axes, orientation="horizontal")
        cbar.ax.set_title(unit)
    else:
        ax.set_xticks(ticks=ndx)
        ax.set_xticklabels(axis_ls, fontsize=fontsize)
        ax.set_yticks(ticks=data_latitude[::4])
        ax.set_yticklabels(labels=[str(int(x)) for x in data_latitude[::4]], fontsize=fontsize)
        ax.set_xlabel('Solar longitude (°)', fontsize=fontsize)
        ax.set_ylabel('Latitude (°N)', fontsize=fontsize)
        if extend:
            cbar = plt.colorbar(ctf, extend='max')
        else:
            cbar = plt.colorbar(ctf)
        cbar.ax.set_title(unit, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    dict_var = [{"data": data_time, "varname": "Solar longitude", "units": "deg", "shortname": "Ls"},
                {"data": data_latitude, "varname": "Latitude", "units": "deg N", "shortname": "Latitude"},
                {"data": info_netcdf.data_target,
                 "varname": f"Zonal mean of density column of {info_netcdf.target_name}",
                 "units": "kg.m-2", "shortname": f"{info_netcdf.target_name}"}
                ]

    save_figure_data(dict_var, savename=save_name)
    return


def display_vars_localtime_longitude(info_netcdf, norm, vmin, vmax, unit, title, save_name):
    from matplotlib.colors import Normalize, LogNorm

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    data_local_time, tmp, tmp = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)

    pcm = ax.pcolormesh(info_netcdf.data_dim.longitude[:], data_local_time[:], info_netcdf.data_target, norm=norm,
                        cmap='plasma', shading='auto')
    cbar = fig.colorbar(pcm)
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_yticks(data_local_time[:])
    ax.set_yticklabels([str(int(x)) for x in data_local_time[:]], fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel('Local time (h)', fontsize=fontsize)
    ax.set_xlabel('Longitude (°E)', fontsize=fontsize)
    fig.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_ls_longitude(info_netcdf, norm, vmin, vmax, unit, title, save_name):
    from matplotlib.colors import Normalize, LogNorm

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                   selected_time=info_netcdf.local_time)

    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        data_time = data_ls[idx::len(data_local_time)]
        print(info_netcdf.data_target.shape, data_time.shape, data_local_time.shape, idx)
        info_netcdf.data_target, data_time = linearize_ls(data=info_netcdf.data_target, data_ls=data_time)
    else:
        data_time = info_netcdf.data_dim.time

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)

    pcm = ax.pcolormesh(info_netcdf.data_dim.longitude, data_time[:], info_netcdf.data_target.T, norm=norm,
                        cmap='plasma', shading='auto')
    cbar = fig.colorbar(pcm)
    cbar.ax.set_title(unit, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_yticks(data_time[::45])
    ax.set_yticklabels(data_time[::45], fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel('Solar longitude (°)', fontsize=fontsize)
    ax.set_xlabel('Longitude (°E)', fontsize=fontsize)
    fig.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_ps_at_viking(data_pressure_at_viking1, latitude1, longitude1, data_pressure_at_viking2, latitude2,
                         longitude2):
    data_sols_1, data_pressure_viking1 = viking_lander(lander=1, mcd=False)
    data_sols_2, data_pressure_viking2 = viking_lander(lander=2, mcd=False)
    data_sols_mcd_1, data_pressure_viking1_mcd = viking_lander(lander=1, mcd=True)
    data_sols_mcd_2, data_pressure_viking2_mcd = viking_lander(lander=2, mcd=True)

    fig, ax = plt.subplots(ncols=2, figsize=figsize_2graph_cols)

    fig.suptitle('Annual mean and diurnal mean of surface pressure at', fontsize=fontsize)
    ax[0].set_title(f'Viking 1 ({latitude1:.0f}°N, {longitude1:.0f}°E)', fontsize=fontsize)
    ax[1].set_title(f'Viking 2 ({latitude2:.0f}°N, {longitude2:.0f}°E)', fontsize=fontsize)

    ax[0].scatter(data_sols_1, data_pressure_viking1, c='black', label='OBS')
    ax[1].scatter(data_sols_2, data_pressure_viking2, c='black', label='OBS')
    ax[0].plot(data_pressure_at_viking1[:], color='red', label='SIMU')
    ax[1].plot(data_pressure_at_viking2[:], color='red', label='SIMU')

    ax2 = plt.twiny(ax[0])
    ax2.scatter(data_sols_mcd_1, data_pressure_viking1_mcd, c='blue', label='MCD 5.3')
    ax2.set_xlim(0, 360)
    ax2 = plt.twiny(ax[1])
    ax2.scatter(data_sols_mcd_2, data_pressure_viking2_mcd, c='blue', label='MCD 5.3')
    ax2.set_xlim(0, 360)

    for axes in ax.reshape(-1):
        axes.legend(loc='best')
        axes.set_xlabel('Sols', fontsize=fontsize)
        axes.set_ylabel('Pressure (Pa)', fontsize=fontsize)
        axes.tick_params(axis='both', which='major', labelsize=fontsize)
        axes.set_xlim(0, 669)

    savename = 'ps_at_viking_land_site'
    fig.savefig(savename + '.png', bbox_inches='tight')

    dict_var = [{"data": data_sols_1, "varname": "Time Viking 1", "units": "sols", "shortname": "Time_VK1"},
                {"data": data_pressure_viking1, "varname": "Pressure at Viking 1", "units": "Pa",
                 "shortname": "Pres_VK1"},
                {"data": data_sols_2, "varname": "Time Viking 2", "units": "sols", "shortname": "Time_VK2"},
                {"data": data_pressure_viking2, "varname": "Pressure at Viking 2", "units": "Pa",
                 "shortname": "Pres_VK2"},
                {"data": arange(669), "varname": "Time", "units": "sols", "shortname": "Time_sim"},
                {"data": data_pressure_at_viking1, "varname": "Pressure simulated at Viking 1", "units": "Pa",
                 "shortname": "P_sim_VK1"},
                {"data": data_pressure_at_viking2, "varname": "Pressure simulated at Viking 2", "units": "Pa",
                 "shortname": "P_sim_VK2"},
                {"data": array([latitude1, longitude1]), "varname": "Viking lander 1 site coordinate in GCM",
                 "units": "deg N, deg E", "shortname": "VK1_XYsim"},
                {"data": array([latitude2, longitude2]), "varname": "Viking lander 2 site coordinate in GCM",
                 "units": "deg N, deg E", "shortname": "VK2_XYsim"},
                {"data": array([22.27, 312.05]), "varname": "Viking lander 1 site coordinate",
                 "units": "deg N, deg E", "shortname": "VK1_XY"},
                {"data": array([47.67, 134.28]), "varname": "Viking lander 2 site coordinate",
                 "units": "deg N, deg E", "shortname": "VK2_XY"}
                ]

    save_figure_data(dict_var, savename=savename)


def display_vars_1fig_profiles(info_netcdf, list_data, latitude_selected, x_min, x_max, x_label, x_scale='linear',
                               y_scale='linear', second_var=None, x_min2=None, x_max2=None, x_label2=None,
                               x_scale2=None, title='', save_name='profiles', title_option=None):
    from numpy import arange

    if info_netcdf.data_dim.altitude.units == 'm':
        units = 'km'
        altitude_name = 'Altitude'
        info_netcdf.data_dim.altitude = info_netcdf.data_dim.altitude / 1e3
        y_scale = 'linear'
    elif info_netcdf.data_dim.altitude.units == 'km':
        units = info_netcdf.data_dim.altitude.units
        altitude_name = 'Altitude'
        y_scale = 'linear'
    else:
        units = info_netcdf.data_dim.altitude.units
        altitude_name = 'Pressure'

    for i, d, s in zip(arange(len(save_name)), list_data, save_name):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
        # plot variable 1
        if d.ndim > 1:
            for j in range(d.shape[1]):
                ax.set_xscale(x_scale)
                ax.set_yscale(y_scale)
                ax.plot(d[:, j], info_netcdf.data_dim.altitude, label=f'{latitude_selected[j]:.2f}°N')
        else:
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.plot()
        # plot variable 2 if exists
        if second_var is not None:
            ax2 = ax.twiny()
            ax2.set_xscale(x_scale2)
            ax2.set_xlim(x_min2, x_max2)
            ax2.set_xlabel(x_label2)
            for j in range(second_var[0].shape[1]):
                ax2.plot(second_var[i][:, j], info_netcdf.data_dim.altitude, ls='--')
        ax.set_xlim(x_min, x_max)
        if altitude_name == 'Pressure':
            ax.invert_yaxis()
        ax.legend(loc='best')
        ax.set_xlabel(x_label)
        ax.set_ylabel(altitude_name + ' (' + units + ')')
        if title_option is not None:
            ax.set_title(f'{title}, and {title_option[i]}', fontsize=fontsize)
        fig.savefig(f'{s}.png', bbox_inches='tight')
        plt.close(fig)

    create_gif(save_name)
    return


def display_vars_stats_zonal_mean(info_netcdf):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    ax.set_title('Zonal mean of surface emissivity at 14h', fontsize=fontsize)
    ctf = ax.contourf(arange(12), info_netcdf.data_dim.latitude, info_netcdf.data_target.T,
                      levels=([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.]))
    ax.set_xlabel('Months', fontsize=fontsize)
    ax.set_ylabel('Latitude (°N)', fontsize=fontsize)
    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('W.m$^{-1}$', fontsize=fontsize)
    plt.savefig('emis_stats_zonal_mean_14h.png', bbox_inches='tight')
    plt.close(fig)
    return


def display_vars_histo(info_netcdf):
    labels = [f"{x:.2e}" for x in info_netcdf.data_dim.altitude]

    cmap = plt.get_cmap('inferno')
    tab_colors = cmap(arange(info_netcdf.data_target.shape[1]) / info_netcdf.data_target.shape[1])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
    ax.hist(info_netcdf.data_target, bins=250 - 60, range=(60, 250), density=True, histtype='barstacked',
            stacked=True, color=tab_colors, label=labels)
    ax.axvline(min(info_netcdf.data_target), ymin=0, ymax=100, ls='--', color='black')
    ax.axvline(max(info_netcdf.data_target), ymin=0, ymax=100, ls='--', color='black')

    ax.legend(loc=0)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Count normalized')
    ax.set_ylim(0, 0.018)
    ax.set_xlim(60, 250)
    plt.savefig(f'{info_netcdf.target_name}_histo.png', bbox_inches='tight')
    plt.show()
    return


def display_vars_polar_projection(info_netcdf, data_np, data_sp, levels, unit, cmap, sup_title, save_name):
    import cartopy.crs as crs

    latitude_np, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[60, 90])
    latitude_sp, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[-60, -90])

    plate_carree = crs.PlateCarree(central_longitude=0)

    orthographic_north = crs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic_north.y_limits
    orthographic_north._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic_north._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°

    orthographic_south = crs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic_south.y_limits
    orthographic_south._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic_south._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize_2graph_cols)
    fig.suptitle(sup_title, fontsize=fontsize)
    ax1 = plt.subplot(121, projection=orthographic_south)
    ax2 = plt.subplot(122, projection=orthographic_north)

    # South polar region
    ax1.set_title('South polar region', fontsize=fontsize)
    ctf = ax1.contourf(info_netcdf.data_dim.longitude, latitude_sp, data_sp, levels=levels, transform=plate_carree,
                       cmap=cmap)
    workaround_gridlines(plate_carree, axes=ax1, pole='south')
    ax1.set_global()

    # North polar region
    ax2.set_title('North polar region', fontsize=fontsize)
    ax2.contourf(info_netcdf.data_dim.longitude, latitude_np, data_np, levels=levels, transform=plate_carree,
                 cmap=cmap)
    workaround_gridlines(plate_carree, axes=ax2, pole='north')
    ax2.set_global()

    pos1 = ax2.get_position().x0 + ax2.get_position().width + 0.05
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([pos1, ax2.get_position().y0, 0.03, ax2.get_position().height])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.ax.set_title(unit, fontsize)

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_polar_projection_multi_plot(info_netcdf, time, vmin, vmax, norm, cmap, unit,
                                             title, save_name, levels=None, co2_ice_cover=None):
    import cartopy.crs as crs
    from numpy import unique, ma
    from matplotlib import cm
    from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
    import matplotlib.ticker as ticker

    tab_co2_ice_cover_np, tab_co2_ice_cover_sp, tab_lon, tab_lat_np, tab_lat_sp = None, None, None, None, None

    if isinstance(info_netcdf.data_target, ma.MaskedArray):
        array_mask = True
    else:
        array_mask = False

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif norm == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = cm.get_cmap(cmap)
        levels = levels
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    plate_carree = crs.PlateCarree(central_longitude=0)

    data_surface_local = gcm_surface_local(data_zareoid=None)

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[60, 90])
    data_np, tmp = slice_data(data=info_netcdf.data_target,
                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                              dimension_slice=info_netcdf.data_dim.latitude,
                              value=[60, 90])
    latitude_sp, tmp = slice_data(data=info_netcdf.data_dim.latitude,
                                  idx_dim_slice=1,
                                  dimension_slice=info_netcdf.data_dim.latitude,
                                  value=[-90, -60])
    data_sp, tmp = slice_data(data=info_netcdf.data_target,
                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                              dimension_slice=info_netcdf.data_dim.latitude,
                              value=[-90, -60])
    data_np_surface, tmp = slice_data(data=data_surface_local[:, :],
                                      idx_dim_slice=0,
                                      dimension_slice=info_netcdf.data_dim.latitude,
                                      value=[60, 90])
    data_sp_surface, tmp = slice_data(data=data_surface_local[:, :],
                                      idx_dim_slice=0,
                                      dimension_slice=info_netcdf.data_dim.latitude,
                                      value=[-90, -60])
    data_np = correction_value(data=data_np, operator='inf', value=threshold)
    data_sp = correction_value(data=data_sp, operator='inf', value=threshold)
    cmap = cm.get_cmap(cmap)
    cmap.set_under('w')

    if co2_ice_cover:
        tab_ls, tab_lat, tab_lon, tab_co2_ice_cover = get_polar_cap()
        tab_lat_np, tmp = slice_data(data=tab_lat,
                                     idx_dim_slice=1,
                                     dimension_slice=tab_lat,
                                     value=[60, 90])
        tab_co2_ice_cover_np, tmp = slice_data(data=tab_co2_ice_cover,
                                               idx_dim_slice=1,
                                               dimension_slice=tab_lat,
                                               value=[60, 90])
        tab_lat_sp, tmp = slice_data(data=tab_lat,
                                     idx_dim_slice=1,
                                     dimension_slice=tab_lat,
                                     value=[-90, -60])
        tab_co2_ice_cover_sp, tmp = slice_data(data=tab_co2_ice_cover,
                                               idx_dim_slice=1,
                                               dimension_slice=tab_lat,
                                               value=[-90, -60])

    # North polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=90)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=figsize_1graph_xtend)
    fig.suptitle(f'North polar region, {title} ({unit})', fontsize=fontsize, y=1.05)
    ctf = None

    for i, axes in enumerate(ax.reshape(-1)):
        if i < 24:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°', fontsize=fontsize)
            if array_mask and unique(data_np[i, :, :]).shape[0] != 1:
                # Need at least 1 row filled with values
                ctf = axes.pcolormesh(info_netcdf.data_dim.longitude[:], latitude_np, data_np[i, :, :], norm=norm,
                                      transform=plate_carree, cmap=cmap, shading='flat')
                axes.contour(info_netcdf.data_dim.longitude[:], latitude_np, data_np_surface[:, :],
                             transform=plate_carree, cmap='Blues')
                if co2_ice_cover:
                    tmp = mean(tab_co2_ice_cover_np[30 * i:30 * (i + 1), :, :], axis=0)
                    tmp = where(tmp == 1, 1.1, tmp)
                    axes.contour(tab_lon, tab_lat_np, tmp, transform=plate_carree,
                                 colors='black', linestyles='-', levels=[-1, 0, 1., 2.], linewidths=0.5)
                    axes.contourf(tab_lon, tab_lat_np, tmp, transform=plate_carree,
                                  hatches=[None, '//', '.', '-'], levels=[-1, 0, 1., 2.], alpha=0, colors='none')
                axes.set_global()
                workaround_gridlines(plate_carree, axes=axes, pole='north')
                axes.text(1, 1, '135°E', verticalalignment='top', horizontalalignment='right', rotation=45,
                          transform=axes.transAxes)
                axes.text(1, 0, '45°E', verticalalignment='bottom', horizontalalignment='right', rotation=-45,
                          transform=axes.transAxes)
                axes.text(0, 0, '315°E', verticalalignment='bottom', horizontalalignment='left', rotation=-315,
                          transform=axes.transAxes)

                axes.text(0, 1, '225°E', verticalalignment='top', horizontalalignment='left', rotation=315,
                          transform=axes.transAxes)
            else:
                axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    cbar = fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal',
                        format=ticker.FuncFormatter(lambda x, levels: "%.0e" % x))
    cbar.ax.tick_params(labelsize=fontsize, bottom=False, top=True, labeltop=True, labelbottom=False)

    if len(info_netcdf.local_time) == 1:
        plt.savefig(f'{save_name}_northern_polar_region_{int(info_netcdf.local_time[0])}h.png', bbox_inches='tight')
    else:
        plt.savefig(f'{save_name}_northern_polar_region_diurnal_mean.png', bbox_inches='tight')

    # South polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=-90)  # , globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=figsize_1graph_xtend)
    fig.suptitle(f'South polar region, {title} ({unit})', fontsize=fontsize, y=1.05)
    for i, axes in enumerate(ax.reshape(-1)):
        if i < 24:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°', fontsize=fontsize)
            if array_mask and unique(data_sp[i, :, :]).shape[0] != 1:
                ctf = axes.pcolormesh(info_netcdf.data_dim.longitude[:], latitude_sp, data_sp[i, :, :], norm=norm,
                                      transform=plate_carree, cmap=cmap, shading='flat')
                axes.contour(info_netcdf.data_dim.longitude[:], latitude_sp, data_sp_surface[:, :],
                             transform=plate_carree, cmap='Blues')
                if co2_ice_cover:
                    tmp = mean(tab_co2_ice_cover_sp[30 * i:30 * (1 + i), :, :], axis=0)
                    tmp = where(tmp == 1, 1.1, tmp)
                    axes.contour(tab_lon, tab_lat_sp, tmp, transform=plate_carree,
                                 colors='black', linestyles='-', levels=[-1, 0, 1, 2], linewidths=0.5)
                    axes.contourf(tab_lon, tab_lat_sp, tmp, transform=plate_carree,
                                  hatches=[None, '//', '.', '-'], levels=[-1, 0, 1, 2], alpha=0, colors='none',
                                  extend='lower')
                axes.set_global()
                workaround_gridlines(plate_carree, axes=axes, pole='south')
                axes.text(1, 1, '45°E', verticalalignment='top', horizontalalignment='right', rotation=45,
                          transform=axes.transAxes)
                axes.text(1, 0, '135°E', verticalalignment='bottom', horizontalalignment='right', rotation=-45,
                          transform=axes.transAxes)
                axes.text(0, 0, '225°E', verticalalignment='bottom', horizontalalignment='left', rotation=-315,
                          transform=axes.transAxes)

                axes.text(0, 1, '315°E', verticalalignment='top', horizontalalignment='left', rotation=315,
                          transform=axes.transAxes)
            else:
                axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    cbar = fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal',
                        format=ticker.FuncFormatter(lambda x, levels: "%.0e" % x))
    cbar.ax.tick_params(labelsize=fontsize, bottom=False, top=True, labeltop=True, labelbottom=False)

    if len(info_netcdf.local_time) == 1:
        plt.savefig(f'{save_name}_southern_polar_region_{int(info_netcdf.local_time[0])}h.png', bbox_inches='tight')
    else:
        plt.savefig(f'{save_name}_southern_polar_region_diurnal_mean.png', bbox_inches='tight')

    dict_var = [{"data": info_netcdf.data_dim.longitude[:], "varname": "Longitude", "units": "deg E", "shortname":
        "longitude"},
                {"data": latitude_np[:], "varname": "Northern latitude", "units": "deg N", "shortname": "latitude_np"},
                {"data": latitude_sp[:], "varname": "Southern latitude", "units": "deg N", "shortname": "latitude_sp"},
                {"data": arange(0, 360, 30), "varname": "Solar longitde  bin", "units": "deg", "shortname": "time"},
                {"data": data_np, "varname": "CO2 ice mass at the surface", "units": "kg", "shortname":
                    "co2_ice_north"},
                {"data": data_sp, "varname": "CO2 ice mass at the surface", "units": "kg", "shortname":
                    "co2_ice_south"},
                {"data": data_np_surface[:], "varname": "Northern topology", "units": "km", "shortname":
                    "topo_north"},
                {"data": data_sp_surface[:], "varname": "Southern topology", "units": "km", "shortname":
                    "topo_south"}
                ]

    save_figure_data(list_dict_var=dict_var, savename=f"{save_name}_polar_region")
    return


def workaround_gridlines(src_proj, axes, pole):
    from numpy import linspace, zeros
    # Workaround for plotting lines of constant latitude/longitude as gridlines
    # labels not supported for this projection.
    latitudes, levels = None, None
    longitudes = linspace(0, 360, num=360, endpoint=False)
    if pole == 'north':
        latitudes = linspace(59, 90, num=31, endpoint=True)
        levels = [60, 70, 80]
    elif pole == 'south':
        latitudes = linspace(-90, -59, num=31, endpoint=True)
        levels = [-80, -70, -60]
    else:
        print('Wrong input pole')
        exit()

    yn = zeros(len(latitudes))
    lona = longitudes + yn.reshape(len(latitudes), 1)
    cs2 = axes.contour(longitudes, latitudes, lona, 10, transform=src_proj, colors='grey', linestyles='--',
                       levels=arange(45, 495, 90), linewidths=1)
    #    axes.clabel(cs2, fontsize=12, inline=True, fmt='%1.0f', inline_spacing=10,  colors='blue',
    #                )

    yt = zeros(len(longitudes))
    contour_latitude = latitudes.reshape(len(latitudes), 1) + yt
    cs = axes.contour(longitudes, latitudes, contour_latitude, 10, transform=src_proj, colors='grey',
                      linestyles='--', levels=levels, linewidths=1)
    # axes.clabel(cs, fontsize=12, inline=True, fmt='%1.0f', inline_spacing=20, colors='grey')


def projection_3D_co2_ice():
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"

    for i, value_i in enumerate(info_netcdf.data_dim.time):
        data = info_netcdf.data_target[i, :, :, :]
        #        data = swapaxes(data, axis1=0, axis2=2)
        if not data.mask.all():
            fig = plt.figure(figsize=figsize_1graph)
            ax = fig.add_subplot(projection='3d')
            # Plot the surface
            X, Y, Z = meshgrid(log10(info_netcdf.data_dim.altitude[:]), latitude_selected,
                               info_netcdf.data_dim.longitude[:]
                               )
            ax.scatter(X, Y, Z, c=data.ravel())
            ax.set_zlim(info_netcdf.data_dim.longitude[0], info_netcdf.data_dim.longitude[-1])
            ax.set_ylim(latitude_selected[0], latitude_selected[-1])
            ax.set_xlim(-3, 3)
            ax.invert_xaxis()
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_zlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.set_xlabel('Altitude (Pa)')
            save_name = f'3d_cloud_evolution_latitude_sols_{value_i:.0f}_{value_i * 24 % 24:.0f}h.png'
            plt.savefig(dirsave + save_name, bbox_inches='tight')
            plt.close()

    altitude_limit, idx_altitude_min, idx_altitude_max = compute_column_density(info_netcdf=info_netcdf)
    for i, value_i in enumerate(info_netcdf.data_dim.time):
        if not info_netcdf.data_target[i, :, :].mask.all():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_1graph)
            fig.subplots_adjust(wspace=0.4)
            fig.suptitle(f'Sols: {value_i:.0f}, local time: {value_i * 24 % 24:.0f} h')
            ax.pcolormesh(info_netcdf.data_dim.longitude, latitude_selected, info_netcdf.data_target[i, :, :],
                          cmap=cmap)
            save_name = f'cloud_evolution_latitude_sols_{value_i:.0f}_{value_i * 24 % 24:.0f}h.png'
            plt.savefig(dirsave + save_name, bbox_inches='tight')
            plt.close()
