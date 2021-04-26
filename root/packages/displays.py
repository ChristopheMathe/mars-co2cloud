import matplotlib.pyplot as plt
from .DataObservation import *
from .DataProcessed import *


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


def display_co2_ice_mola(filename, data):
    from matplotlib.colors import LogNorm
    from numpy import logspace

    data_time = get_data(filename=filename, target='Time')
    data_latitude = get_data(filename=filename, target='latitude')
    mola_latitude, mola_ls, mola_altitude = mola()
    mola_altitude = correction_value(mola_altitude, operator='inf', threshold=0)
    mola_altitude = correction_value(mola_altitude, operator='sup', threshold=1e4)

    fig, ax = plt.subplots(ncols=2, figsize=(11, 8))
    ax[0].set_title('Zonal mean of column density \n of CO$_2$ ice (kg.m$^{-2}$)', loc='center')
    ctf = ax[0].contourf(data_time[:], data_latitude[:], data, norm=LogNorm(), levels=logspace(-9, 2, 12),
                         cmap='inferno')
    plt.colorbar(ctf, ax=ax[0])

    ax[1].set_title('Top altitude of the CO$_2$ cloud \nobserved from MOLA (km)')
    ctf2 = ax[1].contourf(mola_ls[:], mola_latitude[:], mola_altitude[:, :] / 1e3, levels=arange(11), cmap='inferno')
    plt.colorbar(ctf2, ax=ax[1])

    ax[0].set_xlim(0, 360)
    ax[1].set_xlim(0, 360)

    ax[0].set_ylabel('Latitude (°N)')
    fig.text(0.5, 0.03, u'Solar longitude (°)', ha='center', va='center', fontsize=12)
    plt.savefig('DARI_co2_ice_density_column_MOLA.png', bbox_inches='tight')
    plt.show()


def display_co2_ice_distribution_altitude_latitude_polar(filename, distribution_north, distribution_south,
                                                         north_latitude, south_latitude, save_name):
    data_altitude = get_data(filename=filename, target='altitude')

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))
    pc = ax[0].contourf(north_latitude, data_altitude[:], distribution_north.T, cmap='Greys')
    ax[0].set_ylim(0, 20000)

    ax[1].contourf(south_latitude, data_altitude[:], distribution_south.T, cmap='Greys')
    ax[1].set_ylim(0, 20000)

    plt.draw()
    p0 = ax[0].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0.95, p0[2] - p0[0], 0.025])  # left, bottom, width, height
    cbar = plt.colorbar(pc, cax=ax_cbar, orientation='horizontal')
    cbar.ax.set_title('count')

    fig.text(0.02, 0.5, f'{data_altitude.name} ({data_altitude.units})', ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=14)

    fig.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_co2_ice_cloud_evolution_latitude(filename, data, data_satuco2, data_temp, data_riceco2, idx_max, x_time,
                                             latitude_selected):
    from numpy import arange, round, logspace, concatenate, array
    from matplotlib.colors import DivergingNorm, LogNorm

    data_time = get_data(filename, target='Time')
    data_altitude = get_data(filename, target='altitude')
    data_latitude = get_data(filename, target='latitude')
    data_latitude, latitude_selected = slice_data(data_latitude, dimension_data=data_latitude,
                                                  value=[latitude_selected[0], latitude_selected[-1]])
    data = data[x_time, :, :]
    data_satuco2 = data_satuco2[x_time, :, :]
    data_temp = data_temp[x_time, :, :]
    data_riceco2 = data_riceco2[x_time, :, :]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 11))
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle('Sols: ' + str(int(data_time[:][idx_max[0] + x_time])) + ', local time: ' + str(
        int(round(data_time[:][idx_max[0] + x_time] * 24 % 24, 0))) + ' h')
    ax[0, 0].title.set_text('CO$_2$ ice mmr')
    pc0 = ax[0, 0].contourf(data_latitude[:], data_altitude[:], data, norm=LogNorm(vmin=1e-12, vmax=1e-4),
                            levels=logspace(-11, -1, 11), cmap='Greys')
    ax[0, 0].set_yscale('log')
    ax[0, 0].invert_yaxis()
    cbar0 = plt.colorbar(pc0, ax=ax[0, 0])
    cbar0.ax.set_title('kg/kg')
    cbar0.ax.set_yticklabels(["{:.2e}".format(i) for i in cbar0.get_ticks()])

    ax[0, 1].title.set_text('Temperature')
    pc1 = ax[0, 1].contourf(data_latitude[:], data_altitude[:], data_temp, vmin=80, vmax=240,
                            levels=arange(80, 260, 20), cmap='plasma')
    cbar1 = plt.colorbar(pc1, ax=ax[0, 1])
    ax[0, 1].set_yscale('log')
    ax[0, 1].invert_yaxis()
    cbar1.ax.set_title('K')

    ax[1, 0].title.set_text('Saturation of CO$_2$ ice')
    pc2 = ax[1, 0].contourf(data_latitude[:], data_altitude[:], data_satuco2,
                            norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=17),
                            levels=concatenate([array([0, 1]), arange(3, 19, 2)]), cmap='seismic')
    ax[1, 0].set_yscale('log')
    ax[1, 0].invert_yaxis()
    cbar2 = plt.colorbar(pc2, ax=ax[1, 0])
    cbar2.ax.set_title('')

    ax[1, 1].title.set_text('Radius of CO$_2$ ice particle')
    pc3 = ax[1, 1].contourf(data_latitude[:], data_altitude[:], data_riceco2 * 1e6, vmin=0, vmax=60,
                            levels=arange(0, 65, 5), cmap='Greys')
    ax[1, 1].set_yscale('log')
    ax[1, 1].invert_yaxis()
    cbar3 = plt.colorbar(pc3, ax=ax[1, 1])
    cbar3.ax.set_title('µm')

    fig.text(0.02, 0.5, 'Altitude (Pa)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=14)

    save_name = 'cloud_evolution_latitude_sols_' + str(int(data_time[:][idx_max[0] + x_time])) + '_' + str(
        round(data_time[:][idx_max[0] + x_time] * 24 % 24, 0)) + 'h.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

    return save_name


def display_co2_ice_max_longitude_altitude(filename, name, max_mmr, max_alt, max_temp, max_satu, max_radius,
                                           max_ccn_n, unit):
    from matplotlib.colors import LogNorm, DivergingNorm
    from numpy import arange, logspace

    data_latitude = get_data(filename=filename, target='latitude')

    # PLOT
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(11, 30))

    # plot 1
    ax[0].set_title(f'Max {name} in altitude/longitude')
    pc = ax[0].contourf(max_mmr, norm=LogNorm(), levels=logspace(-10, 1, 12), cmap='warm')
    ax[0].set_facecolor('white')
    ax[0].set_xticklabels(labels='')
    cbar = plt.colorbar(pc, ax=ax[0])
    cbar.ax.set_title(unit)
    ax[0].set_ylabel('Latitude (°N)')

    # plot 2
    ax[1].set_title('Altitude at co2_ice mmr max')
    pc2 = ax[1].contourf(max_alt, cmap='warm')
    ax[1].set_facecolor('white')
    ax[1].set_xticklabels(labels='')
    ax[1].set_ylabel('Latitude (°N)')
    cbar2 = plt.colorbar(pc2, ax=ax[1])
    cbar2.ax.set_title('km')

    # plot 3
    ax[2].set_title('Temperature at co2_ice mmr max')
    pc3 = ax[2].contourf(max_temp, cmap='warm')
    ax[2].set_facecolor('white')
    ax[2].set_xticklabels(labels='')
    ax[2].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[2])
    cbar3.ax.set_title('K')

    # plot 4
    divnorm = DivergingNorm(vmin=0, vcenter=1, vmax=4)
    ax[3].set_title('Saturation at co2_ice mmr max')
    pc4 = ax[3].contourf(max_satu, cmap='warm', norm=divnorm, levels=arange(0, 5))
    ax[3].set_facecolor('white')
    ax[3].set_xticklabels(labels='')
    ax[3].set_ylabel('Latitude (°N)')
    cbar4 = plt.colorbar(pc4, ax=ax[3])
    cbar4.ax.set_title(' ')

    # plot 5
    ax[4].set_title('Radius of co2_ice at co2_ice mmr max')
    pc5 = ax[4].contourf(max_radius * 1e3, cmap='warm')
    ax[4].set_facecolor('white')
    ax[4].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[4].set_yticklabels(labels=data_latitude[::6])
    ax[4].set_xticklabels(labels='')
    ax[4].set_ylabel('Latitude (°N)')
    cbar5 = plt.colorbar(pc5, ax=ax[4])
    cbar5.ax.set_title(u'µm')

    # plot 6
    ax[5].set_title('CCN number at co2_ice mmr max')
    pc3 = ax[5].contourf(max_ccn_n, norm=DivergingNorm(vmin=0, vcenter=1), levels=arange(0, 5), cmap='warm')
    ax[5].set_facecolor('white')
    ax[5].set_xlabel('Solar Longitude (°)')
    ax[5].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[5])
    cbar3.ax.set_title('nb/kg')

    fig.savefig(f'max_{name}_in_altitude_longitude.png', bbox_inches='tight')

    plt.show()


def display_co2_ice_density_column_evolution_polar_region(filename, data, time, latitude):
    from numpy import logspace
    from math import floor
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    import cartopy.crs as crs

    data_longitude = get_data(filename=filename, target='longitude')

    plate_carree = crs.PlateCarree(central_longitude=0)

    if latitude[0] > 0:
        orthographic = crs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
        title = 'North polar region'
        pole = 'north'
    else:
        orthographic = crs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
        title = 'South polar region'
        pole = 'south'

    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°

    levels = logspace(-13, 1, 15)
    cmap = cm.get_cmap('inferno')
    cmap.set_under('w')

    norm = LogNorm()
    save_name = list([])

    # PLOT
    for i in range(data.shape[0]):
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': orthographic}, figsize=(11, 11),
                               facecolor='white')
        ax.set_title(title + f', sols {floor(time[i]):.0f} LT {time[i] * 24 % 24:.0f}')
        ax.set_facecolor('white')
        ctf = ax.contourf(data_longitude[:], latitude, data[i, :, :], norm=norm, levels=levels,
                          transform=plate_carree,
                          cmap=cmap)
        workaround_gridlines(plate_carree, axes=ax, pole=pole)
        ax.set_global()
        cbar = fig.colorbar(ctf)
        cbar.ax.set_title('kg.m$^{-2}$')
        save_name.append(f'co2_ice_density_column_evolution_{i}.png')
        plt.savefig(save_name[i], bbox_inches='tight')
        plt.close(fig)

    # create the gif
    create_gif(save_name)
    return


def display_co2_ice_localtime_ls(filename, data):
    from matplotlib.colors import LogNorm

    data_time = get_data(filename=filename, target='Time')
    data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=0)

    if data_time.units != 'deg':
        data_ls = get_data(filename='../concat_Ls.nc', target='Ls')
        data_time = data_ls[idx::len(data_local_time)]
    else:
        data_time = data_time[idx::len(data_local_time)]

    data, data_time = linearize_ls(data=data, data_ls=data_time)

    data = correction_value(data=data, operator='inf', threshold=1e-13)
    data[data.mask] = 0
    ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
    ctf = ax.pcolormesh(data_time[:], data_local_time[:], data, norm=LogNorm(vmin=1e-4, vmax=1),
                        cmap="viridis", shading='auto')
    cbar = fig.colorbar(ctf)
    cbar.ax.set_title('kg.m$^{-2}$', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.set_facecolor('white')

    ax.set_title('CO2 ice density column at 0°N', fontsize=18)
    ax.set_xlabel('Solar longitude (°)', fontsize=18)
    ax.set_ylabel('Local time (h)', fontsize=18)
    ax.set_xticks(ndx)
    ax.set_xticklabels(axis_ls, fontsize=18)
    ax.set_yticks(data_local_time)
    ax.set_yticklabels(data_local_time, fontsize=18)
    plt.savefig('h2o_ice_zonal_mean_density_column_localtime_evolution_0N.png', bbox_inches='tight')
    return


def display_emis_polar_projection_garybicas2020_figs11_12(filename, data, time, levels, cmap, save_name):
    import cartopy.crs as crs
    from numpy import ndarray

    if isinstance(data, ndarray):
        array_mask = True
    else:
        array_mask = False

    plate_carree = crs.PlateCarree(central_longitude=0)

    data_longitude = get_data(filename=filename, target='longitude')
    data_latitude = get_data(filename=filename, target='latitude')

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[60, 90])
    data_np, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[60, 90])

    latitude_sp, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[-90, -60])
    data_sp, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[-90, -60])

    # Slice data binned in 15°Ls during their winter period
    data_np = data_np[12:, :, :]
    data_sp = data_sp[0:12, :, :]

    # North polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=(20, 15))
    fig.suptitle('North polar region', fontsize=20)
    ctf = None
    for i, axes in enumerate(ax.reshape(-1)):
        axes.set_title(f'{int(time[i] + 180)}° - {int(time[i + 1] + 180)}°')
        if array_mask:
            ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels,
                                transform=plate_carree, cmap=cmap)
        else:
            if data_np[i, :, :].mask.all():
                continue
            else:
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels,
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
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=(20, 15))
    fig.suptitle('South polar region', fontsize=20)
    for i, axes in enumerate(ax.reshape(-1)):
        if i < 12:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°')
            if array_mask:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels,
                                    transform=plate_carree, cmap=cmap)
            else:
                if data_sp[i, :, :].mask.all():
                    continue
                else:
                    ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels,
                                        transform=plate_carree, cmap=cmap)
            workaround_gridlines(plate_carree, axes=axes, pole='south')
            axes.set_global()
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    plt.savefig(save_name + 'southern_polar_region_as_fig12_gary-bicas2020.png', bbox_inches='tight')
    return


def display_riceco2_global_mean(filename, list_data):
    from numpy import mean
    list_data[0] = mean(list_data[0], axis=2)
    list_data[1] = mean(list_data[1], axis=2)

    data_altitude = get_data(filename=filename, target='altitude')
    data_time = get_data(filename, target='Time')

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='col', figsize=(8, 11))
    fig.subplots_adjust(wspace=0, hspace=0)

    levels = arange(0, 200, 20)
    for axes in ax.reshape(-1):
        axes.set_yscale('log')
        axes.invert_yaxis()
        axes.set_ylim(1e3, 0.2)

    ax[0, 0].set_title('Zonal and latitudinal mean of riceco2')
    pc = ax[0, 0].contourf(data_time[:], data_altitude[:], list_data[0].T * 1e6, levels=levels, cmap='inferno')
    ax[1, 0].contourf(data_time[:], data_altitude[:], list_data[1].T * 1e6, levels=levels, cmap='inferno')
    ax[1, 0].set_xlabel('Solar longitude (°)')

    ax[0, 1].set_title('Global mean of riceco2')
    ax[0, 1].set_xscale('log')
    ax[0, 1].plot(mean(list_data[0], axis=0).T * 1e6, data_altitude[:])
    ax[0, 1].text(1.1, 0.5, 'North Pole (40°-90°N)', ha='center', va='center', rotation='vertical', fontsize=8,
                  transform=ax[0, 1].transAxes)

    ax[1, 1].plot(mean(list_data[1], axis=0).T * 1e6, data_altitude[:])
    ax[1, 1].set_xscale('log')
    ax[1, 1].text(1.1, 0.5, 'South Pole (40°-90°S)', ha='center', va='center', rotation='vertical', fontsize=8,
                  transform=ax[1, 1].transAxes)
    ax[1, 1].set_xlabel('Radius (µm)')

    fig.text(0.06, 0.5, 'Pressure (Pa)', ha='center', va='center', rotation='vertical', fontsize=14)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pc, cax=cbar_ax)
    plt.savefig('riceco2_global_mean_winter_polar_regions.png', bbox_inches='tight')
    plt.show()


def display_riceco2_local_time_evolution(filename, data, local_time, latitude):
    from matplotlib.cm import get_cmap
    data_altitude = get_data(filename=filename, target='altitude')

    cmap = get_cmap('hsv')
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_yscale('log')
    ax.set_ylim(1e3, 1e-3)
    ax.set_xlim(1e-9, 1e-4)
    ax.set_xscale('log')
    for i in range(data.shape[1]):
        ax.plot(data[:, i], data_altitude[:], color=cmap(((i+6) % data.shape[1])/data.shape[1]), label=local_time[i])

    ax.legend(loc=0)
    ax.set_title(f'Radius of CO2 ice particles at {latitude}N, zonal mean')
    ax.set_ylabel(f'Altitude ({data_altitude.units})', fontsize=18)
    ax.set_xlabel('Radius particle (µm)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f'riceco2_localtime_evolution_{latitude}N.png', bbox_inches='tight')
    return


def display_riceco2_max_local_time_evolution(filename, data_max_radius, data_max_alt, local_time, latitude):

    data_altitude = get_data(filename=filename, target='altitude')

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.plot(local_time, data_max_radius, color='black', linestyle='-')
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1e-4)

    ax2 = ax.twinx()
    ax2.plot(local_time, data_max_alt, color='red', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_ylim(1e3, 1e-3)

    ax.set_title(f'Max radius of CO2 ice particles at {latitude}N\n with the location of the max (red)', fontsize=18)
    ax.set_ylabel(f'Radius particle (µm)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel(f'Altitude ({data_altitude.units})', fontsize=18, color='red')
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', colors='red')

    ax.set_xlabel('Local time (h)', fontsize=18)
    ax.set_xticks(local_time)
    ax.set_xticklabels(local_time, fontsize=18)
    plt.savefig(f'riceco2_max_localtime_evolution_{latitude}N.png', bbox_inches='tight')
    return


def display_riceco2_top_cloud_altitude(filename, top_cloud, local_time=None):
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0, vmax=10)
    data_latitude = get_data(filename=filename, target='latitude')
    data_time = get_data(filename=filename, target='Time')

    if data_time.units != 'deg':
        data_ls = get_data(filename='../concat_Ls.nc', target='Ls')
        data_local_time, idx, stats_file = check_local_time(data_time=data_time, selected_time=local_time)
        data_time = data_ls[idx::len(data_local_time)]

    top_cloud, interp_time = linearize_ls(data=top_cloud, data_ls=data_time)

    top_cloud = correction_value(data=top_cloud, operator='inf', threshold=0)

    cmap = colormap_idl_rainbow_plus_white()
    cmap.set_over("grey")
    idx, axis_ls, ls_lin = get_ls_index(interp_time)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
    cb = ax.pcolormesh(interp_time[:], data_latitude[:], top_cloud, norm=norm, cmap=cmap)
    ax.set_facecolor('white')
    ax.set_xticks(interp_time[idx])
    ax.set_xticklabels(axis_ls)
    ax.set_yticks(data_latitude[::8])
    ax.set_yticklabels(data_latitude[::8])

    cbar = plt.colorbar(cb, extend='max')
    cbar.ax.set_title('km', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(f'Zonal mean of top cloud altitude ({local_time}h)', fontsize=18)
    ax.set_xlabel('Solar Longitude (°)', fontsize=18)
    ax.set_ylabel('Latitude (°N)', fontsize=18)
    plt.savefig(f'top_cloud_altitude_comparable_to_mola_{local_time}h.png', bbox_inches='tight')
    plt.show()
    return


def display_satuco2_thickness_atm_layer(data, data_std, save_name):
    from numpy import arange, ma, array
    data = ma.masked_where(data == 0, data)

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

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))
    print(data.shape[1], data.shape, data_std.shape, north_pole_ls_my29.shape, north_pole_my29)
    ax[0].set_title('North pole above 60°N', fontsize=18)
    ax[0].errorbar(arange(data.shape[1]) * 5, data[0, :] / 1e3,
                   yerr=data_std[0, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')  # 72 points binned in 5°
    ax[0].errorbar(north_pole_ls_my29, north_pole_my29[:, 1],
                   yerr=[north_pole_my29[:, 2] - north_pole_my29[:, 1], north_pole_my29[:, 1] - north_pole_my29[:, 0]],
                   color='black',
                   ls=' ', marker='+', label='MCS MY29')
    ax[0].set_xticks(ticks=arange(0, 405, 45))
    ax[0].set_xticklabels(labels=arange(0, 405, 45), fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].legend(loc=2)

    ax[1].set_title('South pole above 60°S', fontsize=18)
    ax[1].errorbar(arange(data.shape[1]) * 5, data[1, :] / 1e3,
                   yerr=data_std[1, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')
    ax[1].errorbar(south_pole_ls_my29, south_pole_my29[:, 1],
                   yerr=[south_pole_my29[:, 2] - south_pole_my29[:, 1], south_pole_my29[:, 1] - south_pole_my29[:, 0]],
                   color='black',
                   ls=' ', marker='+', label='MCS MY29')

    ax[1].set_xticks(ticks=arange(0, 405, 45))
    ax[1].set_xticklabels(labels=arange(0, 405, 45), fontsize=18)
    ax[1].legend(loc='best')
    ax[1].tick_params(axis='both', which='major', labelsize=18)

    fig.text(0.06, 0.5, 'Thickness (km)', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=18)

    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    return


def display_satuco2_with_co2_ice_altitude_ls(filename, data_satuco2_north, data_satuco2_eq, data_satuco2_south,
                                             data_co2ice_north, data_co2ice_eq, data_co2ice_south, latitude_north,
                                             latitude_eq, latitude_south, binned):
    from numpy import array, round, ones

    # Info latitude
    data_latitude = get_data(filename, target='latitude')
    list_latitudes = [latitude_north, latitude_eq, latitude_south]

    # Get latitude range between value-1 et value+1
    data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
    data_maven_limb, data_spicam, data_tesmoc, data_themis = mesospheric_clouds_observed()

    list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                data_maven_limb, data_spicam, data_tesmoc, data_themis]

    data_altitude = get_data(filename, target='altitude')
    data_zareoid = None
    altitude_unit = None
    altitude_name = None
    data_surface_local = None
    ticks_altitude = None
    if data_altitude.units == 'm':
        altitude_unit = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif data_altitude.units == 'km':
        altitude_unit = data_altitude.units
        altitude_name = 'Altitude'
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif data_altitude.units == 'Pa':
        altitude_unit = data_altitude.units
        altitude_name = 'Pressure'
        data_zareoid = get_data(filename, target='zareoid')
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    data_time = get_data(filename=filename, target='Time')
    if binned.lower() == 'y' and data_zareoid is not None:
        data_time = data_time[::60]  # 5°Ls binned
        data_zareoid = data_zareoid[::12, :, :, :]
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])
    ndx, axis_ls = get_ls_index(data_time)

    norm_satu = None  # TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100)
    levels_satu = array([1, 10, 20, 50, 100])
    levels_co2 = None
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 11))
    ax[0].set_title('{}°N'.format(latitude_north))
    ax[0].contourf(data_time[:], data_altitude[:], data_satuco2_north, norm=norm_satu, cmap='coolwarm',
                   levels=levels_satu, extend='max')
    ax[0].contour(data_time[:], data_altitude[:], data_co2ice_north, norm=None, levels=levels_co2, colors='black')

    ax[1].set_title('{}°N'.format(latitude_eq))
    ax[1].contourf(data_time[:], data_altitude[:], data_satuco2_eq, norm=norm_satu, cmap='coolwarm',
                   levels=levels_satu, extend='max')
    ax[1].contour(data_time[:], data_altitude[:], data_co2ice_eq, norm=None, levels=levels_co2, colors='black')

    ax[2].set_title('{}°S'.format(abs(latitude_south)))
    cb = ax[2].contourf(data_time[:], data_altitude[:], data_satuco2_south, norm=norm_satu, cmap='coolwarm',
                        levels=levels_satu, extend='max')
    ax[2].contour(data_time[:], data_altitude[:], data_co2ice_south, norm=None, levels=levels_co2, colors='black')

    for i, axe in enumerate(ax):
        axe.set_xticks(ticks=axis_ls)
        axe.set_xticklabels(labels=axis_ls)
        axe.set_xlim(0, 360)
        axe.set_ylim(1e-3, 1e3)

        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         list_latitudes[i])
            if data_obs_ls.shape[0] != 0:
                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data_surface_local, dimension_data=data_latitude,
                                                        value=list_latitudes[i])

            lines_altitudes_0km = get_mean_index_altitude(data_surface_local_sliced, value=0, dimension='time')
            lines_altitudes_10km = get_mean_index_altitude(data_surface_local_sliced, value=1e4, dimension='time')
            lines_altitudes_40km = get_mean_index_altitude(data_surface_local_sliced, value=4e4, dimension='time')
            lines_altitudes_80km = get_mean_index_altitude(data_surface_local_sliced, value=8e4, dimension='time')
            del data_surface_local_sliced

            axe.plot(data_altitude[lines_altitudes_0km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_10km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_40km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_80km], '-', color='grey', linewidth=0.5)

            axe.text(0, data_altitude[lines_altitudes_0km[0]], '0 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_10km[0]], '10 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_40km[0]], '40 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_80km[0]], '80 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)

            axe.set_yscale('log')
            axe.invert_yaxis()
        else:
            axe.set_yticks(ticks=ticks_altitude)
            axe.set_yticklabels(labels=round(data_altitude))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cb, cax=cbar_ax)

    fig.text(0.02, 0.5, f'{altitude_name} ({altitude_unit})', ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    save_name = f'satuco2_zonal_mean_with_co2ice_at_{latitude_north}N_{latitude_eq}N_{latitude_south}N'
    if binned.lower() == 'y':
        save_name = save_name + '_binned'

    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_satuco2_with_co2_ice_altitude_longitude(filename, data_satuco2_north, data_satuco2_south, data_co2ice_north,
                                                    data_co2ice_south, latitude_north, latitude_south, binned):
    from numpy import array, round, ones

    # Info latitude
    data_latitude = get_data(filename, target='latitude')
    list_latitudes = [latitude_north, latitude_south]

    data_time = get_data(filename=filename, target='Time')
    list_time_range = array(([270, 300], [0, 30]))

    # Info longitude
    data_longitude = get_data(filename, target='longitude')

    # Get latitude range between value-1 et value+1
    data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
    data_maven_limb, data_spicam, data_tesmoc, data_themis = mesospheric_clouds_observed()

    list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                data_maven_limb, data_spicam, data_tesmoc, data_themis]

    data_altitude = get_data(filename, target='altitude')
    data_surface_local = None
    ticks_altitude = None
    if data_altitude.units == 'm':
        altitude_unit = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    elif data_altitude.units == 'km':
        altitude_unit = data_altitude.units
        altitude_name = 'Altitude'
        ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    else:
        altitude_unit = data_altitude.units
        altitude_name = 'Pressure'
        data_zareoid = get_data(filename, target='zareoid')
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    norm_satu = None  # TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100)
    levels_satu = array([1, 10, 20, 50, 100])
    levels_co2 = None
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))
    ax[0].set_title('{}°N'.format(latitude_north))
    ax[0].contourf(data_longitude[:], data_altitude[:], data_satuco2_north, norm=norm_satu, cmap='coolwarm',
                   levels=levels_satu, extend='max')
    ax[0].contour(data_longitude[:], data_altitude[:], data_co2ice_north, norm=None, levels=levels_co2, colors='black')

    ax[1].set_title('{}°S'.format(abs(latitude_south)))
    cb = ax[1].contourf(data_longitude[:], data_altitude[:], data_satuco2_south, norm=norm_satu, cmap='coolwarm',
                        levels=levels_satu, extend='max')
    ax[1].contour(data_longitude[:], data_altitude[:], data_co2ice_south, norm=None, levels=levels_co2, colors='black')

    for i, axe in enumerate(ax):
        axe.set_ylim(1e-3, 1e3)

        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         list_latitudes[i])
            if data_obs_ls.shape[0] != 0:
                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data_surface_local, dimension_data=data_latitude,
                                                        value=list_latitudes[i])

            data_surface_local_sliced, tmp = slice_data(data_surface_local_sliced, dimension_data=data_time,
                                                        value=list_time_range[i])

            lines_altitudes_0km = get_mean_index_altitude(data_surface_local_sliced, value=0, dimension='longitude')
            lines_altitudes_10km = get_mean_index_altitude(data_surface_local_sliced, value=1e4, dimension='longitude')
            lines_altitudes_40km = get_mean_index_altitude(data_surface_local_sliced, value=4e4, dimension='longitude')
            lines_altitudes_80km = get_mean_index_altitude(data_surface_local_sliced, value=8e4, dimension='longitude')
            del data_surface_local_sliced

            axe.plot(data_altitude[lines_altitudes_0km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_10km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_40km], '-', color='grey', linewidth=0.5)
            axe.plot(data_altitude[lines_altitudes_80km], '-', color='grey', linewidth=0.5)

            axe.text(0, data_altitude[lines_altitudes_0km[0]], '0 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_10km[0]], '10 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_40km[0]], '40 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)
            axe.text(0, data_altitude[lines_altitudes_80km[0]], '80 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=6)

            axe.set_yscale('log')
            axe.invert_yaxis()
        else:
            axe.set_yticks(ticks=ticks_altitude)
            axe.set_yticklabels(labels=round(data_altitude))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cb, cax=cbar_ax)

    fig.text(0.02, 0.5, f'{altitude_name} ({altitude_unit})', ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    save_name = f'satuco2_time_mean_with_co2ice_at_{latitude_north}N_{latitude_south}N'
    if binned.lower() == 'y':
        save_name = save_name + '_binned'

    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_satuco2_zonal_mean_day_night(data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night,
                                         data_altitude, ndx, axis_ls, title, save_name):
    from numpy import array, round
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))
    fig.suptitle(title)
    ax[0].set_title('Day 6h - 18h')
    ax[0].contourf(data_satuco2_day.T, norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=100), cmap='seismic',
                   levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[0].contour(data_co2ice_day.T, colors='black')

    ax[1].set_title('Night 18h - 6h')
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

    fig.text(0.02, 0.5, 'Altitude (km)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_satuco2_altitude_latitude(data, data_altitude, data_latitude):
    from matplotlib.colors import DivergingNorm
    from numpy import arange, searchsorted, max
    from scipy.interpolate import interp2d

    # ----------------------------------------------------------
    #    south pole
    # -------------
    #    idx_lat_1 = (abs(data_latitude[:] + 57)).argmin() + 1
    #    idx_lat_2 = (abs(data_latitude[:] + 90)).argmin() + 1
    #    idx_ls_1 = (abs(data_time[:] - 0)).argmin()
    #    idx_ls_2 = (abs(data_time[:] - 180)).argmin()
    #    save_name = 'altitude_latitude_saturation_south_pole'
    # ----------------------------------------------------------
    #    north pole
    # -------------
    idx_lat_1 = (abs(data_latitude[:] + 90)).argmin()
    idx_lat_2 = (abs(data_latitude[:] - 90)).argmin()
    #    idx_ls_1 = (abs(data_time[:] - 180)).argmin()
    #    idx_ls_2 = (abs(data_time[:] - 360)).argmin()
    save_name = 'altitude_latitude_saturation_max'
    # ----------------------------------------------------------

    #    idx_alt = (abs(data_altitude[:] - 20)).argmin() + 1

    zoomed_data = data[:, :, idx_lat_1:idx_lat_2, :]

    print(zoomed_data.shape)
    # zoom data: [ls, alt, lat, lon] => [alt, lat]
    zoomed_data = max(max(zoomed_data[:, :, :, :], axis=3), axis=0)  # zonal mean, then temporal mean

    zoomed_data = zoomed_data[:, ::-1]

    f = interp2d(x=arange(len(data_latitude[idx_lat_1:idx_lat_2])), y=arange(len(data_altitude[:])),
                 z=zoomed_data, kind='linear')

    axis_altitude = arange(0, 123)
    interp_altitude = searchsorted(data_altitude[:], axis_altitude)
    zoomed_data = f(arange(len(data_latitude[idx_lat_1:idx_lat_2])), interp_altitude)

    # plot
    divnorm = DivergingNorm(vmin=0, vcenter=1, vmax=4)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('CO$_2$ saturation max')
    plt.contourf(zoomed_data, cmap='seismic', norm=divnorm, levels=arange(0, 5))
    plt.xticks(ticks=arange(0, len(data_latitude[idx_lat_1:idx_lat_2]), 4),
               labels=data_latitude[idx_lat_1:idx_lat_2:4])
    plt.yticks(ticks=arange(0, len(interp_altitude)), labels=[f'{i:1d}' for i in data_altitude[:]])
    cbar = plt.colorbar()
    cbar.ax.set_title('')
    plt.xlabel('Latitude (°)')
    plt.ylabel('Altitude (km)')
    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_temp_gg2011_fig6(filename, data, data_localtime):
    from numpy import arange

    data_altitude = get_data(filename=filename, target='altitude')

    cmap = colormap_idl_rainbow_plus_white()

    fig = plt.figure(figsize=(8, 11))
    pc = plt.contourf(data_localtime, data_altitude[:] / 1e3, data.T, levels=arange(0, 125, 5), cmap=cmap)
    plt.ylim(0, 120)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title('K')
    plt.title('Temperature - Tcond CO$_2$')
    plt.ylabel('Altitude above areoid (km)')
    plt.xlabel('Local time (h)')
    plt.savefig('temp_altitude_localtime_ls120-150_lat0N_lon0E_gg2011_fig6.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig7(filename, data, data_altitude):
    data_latitude = get_data(filename=filename, target='latitude')

    cmap = colormap_idl_rainbow_plus_white()

    # plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('Temperature - Tcond CO$_2$')
    ctf = ax.contourf(data_latitude[:], data_altitude / 1e3, data, levels=arange(0, 130, 10), cmap=cmap)
    ax.set_ylim(40, 120)

    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('K')

    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Altitude above areoid (km)')

    plt.savefig('temp_zonal_mean_altitude_latitude_ls_0-30_gg2011_fig7.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig8(filename, data_zonal_mean, data_thermal_tides):
    data_latitude = get_data(filename=filename, target='latitude')
    data_altitude = get_data(filename=filename, target='altitude')[:]

    fig, ax = plt.subplots(nrows=2, figsize=(8, 11))

    cmap = colormap_idl_rainbow_plus_white()
    # Zonal mean at 16 H local time
    ctf1 = ax[0].contourf(data_latitude[:], data_altitude / 1e3, data_zonal_mean, levels=arange(110, 240, 10),
                          cmap=cmap)
    cbar = plt.colorbar(ctf1, ax=ax[0])
    cbar.ax.set_title('K')
    ax[0].set_ylim(40, 120)
    ax[0].set_xlabel('Latitude (°N)')
    ax[0].set_ylabel('Altitude above areoid (km)')

    # Thermal tides: 12h - 00h
    ctf2 = ax[1].contourf(data_latitude[:], data_altitude / 1e3, data_thermal_tides, levels=arange(-20, 32, 4),
                          cmap=cmap)
    cbar2 = plt.colorbar(ctf2, ax=ax[1])
    cbar2.ax.set_title('K')
    ax[1].set_ylim(40, 120)

    ax[1].set_xlabel('Latitude (°N)')
    ax[1].set_ylabel('Altitude above areoid (km)')
    plt.savefig('temp_gg2011_fig8.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig9(filename, data, data_altitude):
    data_longitude = get_data(filename=filename, target='longitude')

    cmap = colormap_idl_rainbow_plus_white()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
    pc = ax.contourf(data_longitude[:], data_altitude / 1e3, data, levels=arange(0, 130, 10), cmap=cmap)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title('K')
    ax.set_ylim(40, 120)

    plt.title('Temperature -Tcond CO$_2$')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Altitude above areoid (km)')
    plt.savefig('temp_altitude_longitude_ls_0-30_LT_16H_lat_0N_gg2011_fig9.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_structure_polar_region(filename, data_north, data_south, norm, levels, unit, save_name):
    cmap = 'coolwarm'

    data_altitude = get_data(filename=filename, target='altitude')
    data_time = get_data(filename=filename, target='Time')

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))

    ax[0].set_title('North pole at 60°N')
    ctf = ax[0].contourf(data_time[:], data_altitude[:], data_north.T, norm=norm, levels=levels, cmap=cmap)

    ax[1].set_title('South pole at 60°S')
    ax[1].contourf(data_time[:], data_altitude[:], data_south.T, norm=norm, levels=levels, cmap=cmap)

    for axes in ax.reshape(-1):
        axes.set_yscale('log')
        axes.invert_yaxis()
        axes.set_ylim(data_altitude[0], 1e1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.ax.set_title(unit)

    fig.text(0.02, 0.5, '{} ({})'.format(data_altitude.name, data_altitude.units), ha='center', va='center',
             rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_cold_pocket_spicam(filename, data, local_time, title, save_name):
    from matplotlib.colors import DivergingNorm

    data_altitude = get_data(filename, target='altitude')

    data_time = get_data(filename, target='Time')
    if data_time.units != 'deg':
        data_time = get_data(filename='../concat_Ls.nc', target='Ls')

    data_time, local_time = extract_at_a_local_time(filename=filename, data=data_time, local_time=local_time)

    data, interp_time = linearize_ls(data=data, data_ls=data_time)

    fig, axes = plt.subplot(figsize=(11, 8), projection='polar')
    axes.set_title(title)
    # TODO: not finished !
    axes.polar(interp_time, )

    axes.set_xlabel('Solar longitude (°)')
    axes.set_ylabel(f'Pressure ({data_altitude.units})')
    axes.set_yscale('log')
    axes.invert_yaxis()

    fig.savefig(f'{save_name}.png')
    fig.show()
    return


def display_vars_altitude_variable(data, data_latitude, data_pressure, title):
    plt.figure(figsize=(11, 8))
    plt.title(title)

    plt.semilogy(data[:, :], data_pressure[0, :, :, 0], label=data_latitude)

    plt.xlabel('K')
    plt.ylabel('Pressure (Pa)')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('vertical_profile_temperature_equator.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_latitude(filename, data, unit, title, save_name):
    data_altitude = get_data(filename=filename, target='altitude')
    data_latitude = get_data(filename=filename, target='latitude')

    if unit == 'pr.µm':
        data = data * 1e3  # convert kg/m2 to pr.µm

    cmap = 'hot'

    # plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title(title)
    ax.set_yscale('log')
    ctf = ax.contourf(data_latitude[:], data_altitude[:], data, cmap=cmap)
    ax.invert_yaxis()

    cbar = plt.colorbar(ctf)
    cbar.ax.set_title(unit)

    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('{} ({})'.format(data_altitude.name, data_altitude.units))

    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_localtime(filename, data, data_localtime, title, unit, save_name):
    from numpy import arange, zeros
    from matplotlib.colors import TwoSlopeNorm

    data_altitude = get_data(filename=filename, target='altitude')

    fig = plt.figure(figsize=(8, 11))
    if unit == '':
        scale = zeros(5)
        scale[1] = 1
        scale[2:] = 10 ** arange(1, 4)
        pc = plt.contourf(data_localtime, data_altitude[:], data.T, norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=1000),
                          levels=scale, cmap='seismic', extend='max')
    else:
        plt.yscale('log')
        pc = plt.contourf(data_localtime, data_altitude[:], data.T, levels=arange(0, 140, 10), cmap='seismic')
        ax = plt.gca()
        ax.invert_yaxis()

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit)
    plt.title(title)
    plt.ylabel('{} ({})'.format(data_altitude.name, data_altitude.units))
    plt.xlabel('Local time (h)')
    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_longitude(filename, data, unit, norm, vmin, vcenter, vmax, title, save_name):
    from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm

    data_altitude = get_data(filename=filename, target='altitude')
    data_longitude = get_data(filename=filename, target='longitude')

    if data_altitude.units == 'Pa':
        yscale = 'log'
    else:
        yscale = 'linear'

    if norm == 'div':
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = 'coolwarm'
    elif norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'inferno'
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'inferno'

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
    pc = ax.pcolormesh(data_longitude[:], data_altitude[:], data, norm=norm, cmap=cmap, shading='auto')

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit, fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.set_yscale(yscale)
    if data_altitude.units == 'Pa':
        ax.invert_yaxis()
    ax.set_xticks(data_longitude[::8])
    ax.set_xticklabels(labels=data_longitude[::8], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Longitude (°E)', fontsize=18)
    ax.set_ylabel(f'Altitude ({data_altitude.units})', fontsize=18)
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_altitude_ls(filename, data_1, data_2, local_time, title, save_name):
    from numpy import round
    from matplotlib.colors import LogNorm

    norm = LogNorm(vmin=1e-13, vmax=1e-3)

    data_altitude = get_data(filename, target='altitude')

    if data_altitude.units == 'm':
        units = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
    elif data_altitude.units == 'km':
        units = data_altitude.units
        altitude_name = 'Altitude'
    else:
        units = data_altitude.units
        altitude_name = 'Pressure'

    data_time = get_data(filename, target='Time')
    if data_time.units != 'deg':
        data_ls = get_data(filename='../concat_Ls.nc', target='Ls')
        data_local_time, idx, stats = check_local_time(data_time=data_time[:], selected_time=local_time)
        data_time = data_ls[idx::len(data_local_time)]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
    cb = axes.pcolormesh(data_time[:], data_altitude[:], data_1, norm=norm, cmap='plasma', shading='auto')
    axes.pcolormesh(data_time[:], data_altitude[:], data_2, norm=norm, cmap='binary', shading='auto')

    if units == 'Pa':
        axes.set_yscale('log')
        axes.invert_yaxis()
    else:
        axes.set_yticklabels(labels=round(data_altitude[:], 0))

    cbar = plt.colorbar(cb, ax=axes)
    cbar.ax.set_title('kg/kg', fontsize=18)

    axes.set_title(title, fontsize=18)
    axes.set_xlabel('Solar longitude (°)', fontsize=18)
    axes.set_ylabel(f'{altitude_name} ({units})', fontsize=18)
    axes.tick_params(axis='both', which='major', labelsize=18)
    fig.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_latitude_longitude(filename, data, unit, norm, vmin, vmax, title, save_name):
    from matplotlib.colors import LogNorm, Normalize

    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    data_longitude = get_data(filename=filename, target='longitude')
    data_latitude = get_data(filename=filename, target='latitude')

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    ax1 = plt.subplot(111)
    ax1.set_title(title, fontsize=18)
    ctf = ax1.pcolormesh(data_longitude[:], data_latitude[:], data, norm=norm, cmap='plasma', shading='auto')
    ax1.set_xticks(data_longitude[::8])
    ax1.set_xticklabels(data_longitude[::8], fontsize=18)
    ax1.set_yticks(data_latitude[::4])
    ax1.set_yticklabels(data_latitude[::4], fontsize=18)
    ax1.set_xlabel('Longitude (°E)', fontsize=18)
    ax1.set_ylabel('Latitude (°N)', fontsize=18)
    cbar = fig.colorbar(ctf)
    cbar.ax.set_title(unit, fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    ax1.grid()

    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    return


def display_vars_latitude_ls(filename, name_target, data, unit, norm, vmin, vmax, observation=False,
                             latitude_selected=None, localtime_selected=None, title=None, tes=None, mvals=None,
                             layer=None, save_name='test'):
    from matplotlib.colors import LogNorm, Normalize

    n_subplot = 1
    if tes:
        n_subplot += 1
    if mvals:
        n_subplot += 1

    cmap = 'inferno'  # 'coolwarm'
    if norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    data_time = get_data(filename=filename, target='Time')
    data_local_time, idx, stats_file = check_local_time(data_time=data_time[:], selected_time=localtime_selected)

    if data_time.units != 'deg':
        data_ls = get_data(filename='../concat_Ls.nc', target='Ls')
        data_time = data_ls[idx::len(data_local_time)]
    else:
        data_time = data_time[idx::len(data_local_time)]

    data, data_time = linearize_ls(data=data, data_ls=data_time)
    ndx, axis_ls, ls_lin = get_ls_index(data_time=data_time)

    data_altitude = get_data(filename, target='altitude')

    data_latitude = get_data(filename, target='latitude')
    if latitude_selected is not None:
        data_latitude, latitude_selected = slice_data(data_latitude, dimension_data=data_latitude,
                                                      value=[latitude_selected[0], latitude_selected[-1]])
    else:
        latitude_selected = [-90, 90]
    data_latitude = data_latitude[::-1]

    # PLOT
    fig, ax = plt.subplots(nrows=n_subplot, ncols=1, figsize=(11, 11))
    fig.subplots_adjust()

    i_subplot = 0
    if tes:
        # Extract TES data
        print('Takes TES data')
        if name_target == 'tsurf':
            data_time_tes = observation_tes(target='time', year=None)  # None == climatology
            data_latitude_tes = observation_tes(target='latitude', year=None)

            # (time, latitude, longitude), also Tsurf_day/Tsurf_nit
            data_tes = observation_tes(target='Tsurf_nit', year=None)
            ax[i_subplot].set_title('TES climatology')
            idx1 = (abs(data_time_tes[:] - 360 * 2)).argmin()
            idx2 = (abs(data_time_tes[:] - 360 * 3)).argmin()
            data_time_tes = data_time_tes[idx1:idx2] - 360 * 2
            data_tes, tmp = vars_zonal_mean(filename='', data=data_tes[idx1:idx2, :, :], layer=None, flip=True)
        elif name_target == 'temp':
            data_time_tes = observation_tes(target='time', year=25)
            data_latitude_tes = observation_tes(target='latitude', year=25)
            data_altitude_tes = observation_tes(target='altitude', year=25)  # Pa
            data_tes = tes(target='T_limb_day', year=25)
            year = 1

            # Select altitude for TES data close to the specified layer
            target_layer = data_altitude[::-1][layer]
            in_range = ma.masked_outside(data_altitude_tes[:], target_layer, target_layer)
            if not in_range.all():
                data_tes = zeros((data_tes.shape[2], data_tes.shape[0]))
                altitude_tes = 'out range'
            else:
                idx = abs(data_altitude_tes[:] - data_altitude[::-1][layer]).argmin()
                data_tes = data_tes[:, idx, :, :]
                data_tes, tmp = vars_zonal_mean('', data_tes[:, :, :], layer=None)
                altitude_tes = data_altitude_tes[idx]

            data_tes, tmp = vars_zonal_mean('', data_tes[:, :, :], layer=None)
            ax[i_subplot].set_title(f'TES Mars Year {24 + year:d} at {altitude_tes} {data_altitude_tes.units}')
        else:
            data_time_tes = observation_tes(target='time', year=None)
            data_latitude_tes = observation_tes(target='latitude', year=None)
            data_tes = None
            print('No case for TES, to be check!')

        ax[i_subplot].pcolormesh(data_time_tes[:], data_latitude_tes[:], data_tes, shading='auto', cmap=cmap)
        i_subplot += 1

    if mvals:
        # Extract mvals data
        print('Takes M. Vals data')
        data_mvals = simulation_mvals(target=name_target, localtime=2)
        data_time_mvals = simulation_mvals(target='Time', localtime=2)
        data_latitude_mvals = simulation_mvals(target='latitude', localtime=2)
        data_altitude_mvals = simulation_mvals(target='altitude', localtime=2)
        if layer is not None:
            data_mvals = data_mvals[:, layer, :, :]

        if data_mvals.ndim == 3:
            data_mvals = correction_value(data_mvals[:, :, :], operator='inf', threshold=1e-13)
        else:
            data_mvals = correction_value(data_mvals[:, :, :, :], operator='inf', threshold=1e-13)

        # Compute zonal mean
        data_mvals, tmp = vars_zonal_mean(filename='', data=data_mvals, layer=layer, flip=False)

        if name_target == 'temp':
            ax[i_subplot].set_title(f'M. VALS at {data_altitude_mvals[layer]:.2e} {data_altitude_mvals.units}')
        else:
            ax[i_subplot].set_title('M. VALS')

        ax[i_subplot].pcolormesh(data_time_mvals[:], data_latitude_mvals[:], data_mvals, shading='auto', cmap=cmap)
        i_subplot += 1

    if i_subplot == 0:
        ctf = ax.pcolormesh(data_time[:], data_latitude[:], data, norm=norm, cmap=cmap, shading='auto', zorder=10)
    else:
        ctf = ax[i_subplot].pcolormesh(data_time[:], data_latitude[:], data, norm=norm, cmap=cmap, shading='auto',
                                       zorder=10)

    if name_target == 'temp':
        if i_subplot == 0:
            ax.set_title(title, fontsize=18)
        else:
            ax[i_subplot].set_title(f'My work at {data_altitude[::-1][layer]:.2e} {data_altitude.units}', fontsize=18)
    else:
        if i_subplot == 0:
            ax.set_title(title, fontsize=18)
        else:
            ax[i_subplot].set_title('My work', fontsize=18)

    if observation:
        # Get latitude range between entre value-1 et value+1
        data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
        data_maven_limb, data_spicam, data_tesmoc, data_themis = mesospheric_clouds_observed()

        list_obs = [data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs,
                    data_maven_limb, data_spicam, data_tesmoc, data_themis]
        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         [latitude_selected[0], latitude_selected[-1]])
            if data_obs_ls.shape[0] != 0:
                plt.scatter(data_obs_ls, data_obs_latitude, color='black', marker='+', zorder=1)

    if i_subplot != 0:
        for axes in ax.reshape(-1):
            axes.set_facecolor('white')
            axes.set_xticks(ticks=axis_ls)
        fig.text(0.02, 0.5, 'Latitude (°N)'.format('g'), ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
        plt.title(title)
        pos0 = ax[0].get_position()
        cb_axes = fig.add_axes([pos0.x0, 0.95, pos0.x1 - pos0.x0, 0.025])
        cbar = plt.colorbar(ctf, cax=cb_axes, orientation="horizontal")
        cbar.ax.set_title(unit)
    else:
        ax.set_xticks(ticks=ndx)
        ax.set_xticklabels(axis_ls, fontsize=18)
        ax.set_yticks(ticks=data_latitude[::4])
        ax.set_yticklabels(labels=data_latitude[::4], fontsize=18)
        ax.set_xlabel('Solar longitude (°)', fontsize=18)
        ax.set_ylabel('Latitude (°N)', fontsize=18)
        cbar = plt.colorbar(ctf)
        cbar.ax.set_title(unit, fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    plt.savefig(save_name + '.png', bbox_inches='tight')
    return


def display_vars_1fig_profiles(filename, data, latitude_selected, x_min, x_max, x_label, x_scale='linear',
                               y_scale='linear', second_var=None, x_min2=None, x_max2=None, x_label2=None,
                               x_scale2=None, title='', save_name='profiles', title_option=None):
    from numpy import arange

    data_time = get_data(filename, target='Time')

    if data_time.units == 'degrees':
        time_unit = u'° Ls'
    else:
        time_unit = 'sols'

    data_altitude = get_data(filename, target='altitude')
    if data_altitude.units == 'm':
        units = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
        y_scale = 'linear'
    elif data_altitude.units == 'km':
        units = data_altitude.units
        altitude_name = 'Altitude'
        y_scale = 'linear'
    else:
        units = data_altitude.units
        altitude_name = 'Pressure'

    for i, d, s in zip(arange(len(save_name)), data, save_name):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
        # plot variable 1
        if d.ndim > 1:
            for j in range(d.shape[1]):
                ax.set_xscale(x_scale)
                ax.set_yscale(y_scale)
                ax.plot(d[:, j], data_altitude[:], label='%.2f°N' % (latitude_selected[j]))
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
                ax2.plot(second_var[i][:, j], data_altitude[:], ls='--')
        ax.set_xlim(x_min, x_max)
        if altitude_name == 'Pressure':
            ax.invert_yaxis()
        ax.legend(loc='best')
        ax.set_xlabel(x_label)
        ax.set_ylabel(altitude_name + ' (' + units + ')')
        if title_option is not None:
            ax.set_title(f'{title}, and {title_option[i][0]:.0f} - {title_option[i][1]:.0f} {time_unit}')
        fig.savefig(s + '.png', bbox_inches='tight')
        plt.close(fig)

    create_gif(save_name)
    return


def display_vars_2fig_profile(filename, data1, data2):
    data_time = get_data(filename, target='Time')

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='row', sharey='col', figsize=(11, 11))

    ax[0].plot(data_time[:], data1 * 1e3, color='black')  # kg to g

    ax[1].plot(data_time[:], data2 * 1e3, color='black')  # kg to g

    ax[0].set_xlim(data_time[0], data_time[-1])
    fig.text(0.02, 0.5, 'Cumulative masses ({})'.format('g'), ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    fig.savefig('co2ice_cumulative_mass_polar_region.png', bbox_inches='tight')


def display_vars_4figs_polar_projection(filename, data_riceco2):
    from cartopy import crs

    longitudes = get_data(filename, target='longitude')
    latitudes = get_data(filename, target='latitude')

    plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(1, 1, 1, projection=crs.Orthographic())
    ax1.contourf(longitudes[:], latitudes[:], data_riceco2[0, 0, :, :], 60, transform=crs.Orthographic())
    plt.show()


def display_vars_stats_zonal_mean(filename, data):
    data_latitude = get_data(filename, target='latitude')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
    ax.set_title('Zonal mean of surface emissivity at 14h')
    ctf = ax.contourf(arange(12), data_latitude[:], data.T, levels=([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.]))
    ax.set_xlabel('Months')
    ax.set_ylabel('Latitude (°N)')
    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('W.m$^{-1}$')
    plt.savefig('emis_stats_zonal_mean_14h.png', bbox_inches='tight')
    plt.close(fig)
    return


def display_vars_polar_projection(filename, data_np, data_sp, levels, unit, cmap, sup_title, save_name):
    import cartopy.crs as crs

    data_longitude = get_data(filename=filename, target='longitude')
    data_latitude = get_data(filename=filename, target='latitude')

    latitude_np, tmp = slice_data(data=data_latitude, dimension_data=data_latitude, value=[60, 90])
    latitude_sp, tmp = slice_data(data=data_latitude, dimension_data=data_latitude, value=[-60, -90])

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
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 11))
    fig.suptitle(sup_title, fontsize=20)
    ax1 = plt.subplot(121, projection=orthographic_south)
    ax2 = plt.subplot(122, projection=orthographic_north)

    # South polar region
    ax1.set_title('South polar region')
    ctf = ax1.contourf(data_longitude[:], latitude_sp, data_sp, levels=levels, transform=plate_carree, cmap=cmap)
    workaround_gridlines(plate_carree, axes=ax1, pole='south')
    ax1.set_global()

    # North polar region
    ax2.set_title('North polar region')
    ax2.contourf(data_longitude[:], latitude_np, data_np, levels=levels, transform=plate_carree, cmap=cmap)
    workaround_gridlines(plate_carree, axes=ax2, pole='north')
    ax2.set_global()

    pos1 = ax2.get_position().x0 + ax2.get_position().width + 0.05
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([pos1, ax2.get_position().y0, 0.03, ax2.get_position().height])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.ax.set_title(unit)

    plt.savefig(save_name + '.png', bbox_inches='tight')
    return


def display_vars_polar_projection_multi_plot(filename, data, time, localtime, levels, norm, cmap, unit, save_name):
    import cartopy.crs as crs
    from numpy import unique, ma
    from matplotlib import cm

    if isinstance(data, ma.MaskedArray):
        array_mask = True
    else:
        array_mask = False

    plate_carree = crs.PlateCarree(central_longitude=0)

    data_longitude = get_data(filename=filename, target='longitude')
    data_latitude = get_data(filename=filename, target='latitude')

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[60, 90])
    data_np, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[60, 90])
    latitude_sp, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[-90, -60])
    data_sp, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[-90, -60])

    if array_mask:
        data_np[data_np.mask] = 0
        data_sp[data_sp.mask] = 0

    cmap = cm.get_cmap(cmap)
    cmap.set_under('w')

    # North polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=5, ncols=5, subplot_kw={'projection': orthographic}, figsize=(20, 20))
    fig.suptitle(f'North polar region ({unit})', fontsize=20)

    ctf = None
    for i, axes in enumerate(ax.reshape(-1)):
        if i < 24:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°')
            if array_mask and unique(data_np[i, :, :]).shape[0] != 1:
                # Need at least 1 row filled with values
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels, norm=norm,
                                    transform=plate_carree, cmap=cmap, extend='max')
            else:
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels, norm=norm,
                                    transform=plate_carree, cmap=cmap, extend='max')

            axes.set_global()
            workaround_gridlines(plate_carree, axes=axes, pole='north')
            axes.set_facecolor('white')

    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 4].get_position().x0 + ax[0, 4].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    fig.tight_layout()
    plt.savefig(f'{save_name}_northern_polar_region_{localtime}h.png', bbox_inches='tight')

    # South polar region
    orthographic = crs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic.y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=5, ncols=5, subplot_kw={'projection': orthographic}, figsize=(20, 20))
    fig.suptitle(f'South polar region ({unit})', fontsize=20)

    for i, axes in enumerate(ax.reshape(-1)):
        if i < 24:
            axes.set_title(f'{int(time[i])}° - {int(time[i + 1])}°')
            if array_mask and unique(data_sp[i, :, :]).shape[0] != 1:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels, norm=norm,
                                    transform=plate_carree, cmap=cmap, extend='max')
            else:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels, norm=norm,
                                    transform=plate_carree, cmap=cmap, extend='max')

            axes.set_global()
            workaround_gridlines(plate_carree, axes=axes, pole='south')
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 4].get_position().x0 + ax[0, 4].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    fig.tight_layout()

    plt.savefig(f'{save_name}_southern_polar_region_{localtime}h.png', bbox_inches='tight')


def workaround_gridlines(src_proj, axes, pole):
    from numpy import linspace, zeros
    # Workaround for plotting lines of constant latitude/longitude as gridlines
    # labels not supported for this projection.
    latitudes = None
    longitudes = linspace(0, 360, num=360, endpoint=False)
    if pole == 'north':
        latitudes = linspace(60, 90, num=31, endpoint=True)
    elif pole == 'south':
        latitudes = linspace(-90, -59, num=31, endpoint=True)
    else:
        print('Wrong input pole')
        exit()

    yn = zeros(len(latitudes))
    lona = longitudes + yn.reshape(len(latitudes), 1)

    cs2 = axes.contour(longitudes, latitudes, lona, 10, transform=src_proj, colors='grey', linestyles='solid',
                       levels=arange(0, 450, 90), linewidths=2)
    axes.clabel(cs2, fontsize=10, inline=True, fmt='%1.0f', inline_spacing=30)

    yt = zeros(len(longitudes))
    contour_latitude = latitudes.reshape(len(latitudes), 1) + yt
    cs = axes.contour(longitudes, latitudes, contour_latitude, 10, transform=src_proj, colors='grey',
                      linestyles='solid', levels=2, linewidths=2)
    axes.clabel(cs, fontsize=10, inline=True, fmt='%1.0f', inline_spacing=20)
