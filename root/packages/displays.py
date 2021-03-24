import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from .lib_function import *
from .DataObservation import *
from .DataProcessed import *
from matplotlib import cm


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


def display_co2_ice_MOLA(filename, data):
    from numpy import logspace
    data_time = getdata(filename=filename, target='Time')
    data_latitude = getdata(filename=filename, target='latitude')
    mola_latitude, mola_ls, mola_altitude = MOLA()
    mola_altitude = correction_value(mola_altitude, operator='inf', threshold=0)
    mola_altitude = correction_value(mola_altitude, operator='sup', threshold=1e4)

    fig, ax = plt.subplots(ncols=2, figsize=(11, 8))
    ax[0].set_title('Moyenne zonale de la densité \n de colonne de glace de CO$_2$ (kg.m$^{-2}$)', loc='center')
    ctf = ax[0].contourf(data_time[:], data_latitude[:], data, norm=LogNorm(), levels=logspace(-9, 2, 12),
                         cmap='inferno')
    cb = plt.colorbar(ctf, ax=ax[0])

    ax[1].set_title('Altitude du dessus des nuages\nde CO$_2$ observées par MOLA (km)')
    ctf2 = ax[1].contourf(mola_ls[:], mola_latitude[:], mola_altitude[:, :] / 1e3, levels=arange(11), cmap='inferno')
    cb2 = plt.colorbar(ctf2, ax=ax[1])

    ax[0].set_xlim(0, 360)
    ax[1].set_xlim(0, 360)

    ax[0].set_ylabel('Latitude (°N)')
    fig.text(0.5, 0.03, 'Longitude solaire (°)', ha='center', va='center', fontsize=12)
    plt.savefig('DARI_co2_ice_density_column_MOLA.png', bbox_inches='tight')
    plt.show()

def display_co2_ice_distribution_altitude_latitude_polar(filename, distribution_north, distribution_south,
                                                         north_latitude, south_latitude, savename):

    data_altitude = getdata(filename=filename, target='altitude')

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

    fig.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_co2_ice_cloud_evolution_latitude(filename, data, data_satuco2, data_temp, data_riceco2, idx_max, xtime,
                                             latitude_selected):
    from numpy import arange, round, logspace, concatenate, array, max
    from matplotlib.colors import DivergingNorm, LogNorm

    data_time = getdata(filename, target='Time')
    data_altitude = getdata(filename, target='altitude')
    data_latitude = getdata(filename, target='latitude')
    data_latitude, latitude_selected = slice_data(data_latitude, dimension_data=data_latitude,
                                                  value=[latitude_selected[0], latitude_selected[-1]])
    print(data_latitude[:].shape, data.shape)
    data = data[xtime, :, :]
    data_satuco2 = data_satuco2[xtime, :, :]
    data_temp = data_temp[xtime, :, :]
    data_riceco2 = data_riceco2[xtime, :, :]

    #    ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 11))
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle('Sols: ' + str(int(data_time[:][idx_max[0] + xtime])) + ', local time: ' + str(
        int(round(data_time[:][idx_max[0] + xtime] * 24 % 24, 0))) + ' h')
    ax[0, 0].title.set_text('CO$_2$ ice mmr')
    pc0 = ax[0, 0].contourf(data_latitude[:], data_altitude[:], data, norm=LogNorm(vmin=1e-12, vmax=1e-4),
                            levels=logspace(-11, -1, 11), cmap='Greys')
    ax[0, 0].set_yscale('log')
    ax[0, 0].invert_yaxis()
    cbar0 = plt.colorbar(pc0, ax=ax[0, 0])
    cbar0.ax.set_title('kg/kg')
    cbar0.ax.set_yticklabels(["{:.2e}".format(i) for i in cbar0.get_ticks()])
    #  ax[0, 0].set_xticks(ticks=arange(0, data.shape[1], 2))
    #  ax[0, 0].set_xticklabels(labels=round(data_latitude[::2], 2))
    #    ax[0, 0].set_yticks(ticks=ticks_altitude)
    #   ax[0, 0].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[0, 1].title.set_text('Temperature')
    pc1 = ax[0, 1].contourf(data_latitude[:], data_altitude[:], data_temp, vmin=80, vmax=240,
                            levels=arange(80, 260, 20), cmap='plasma')
    cbar1 = plt.colorbar(pc1, ax=ax[0, 1])
    ax[0, 1].set_yscale('log')
    ax[0, 1].invert_yaxis()
    cbar1.ax.set_title('K')
    # ax[0, 1].set_xticks(ticks=arange(0, data.shape[1], 2))
    # ax[0, 1].set_xticklabels(labels=round(data_latitude[::2], 2))
    #  ax[0, 1].set_yticks(ticks=ticks_altitude)
    # ax[0, 1].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[1, 0].title.set_text('Saturation of CO$_2$ ice')
    print(max(data_satuco2))
    pc2 = ax[1, 0].contourf(data_latitude[:], data_altitude[:], data_satuco2,
                            norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=17),
                            levels=concatenate([array([0, 1]), arange(3, 19, 2)]), cmap='seismic')
    ax[1, 0].set_yscale('log')
    ax[1, 0].invert_yaxis()
    cbar2 = plt.colorbar(pc2, ax=ax[1, 0])
    cbar2.ax.set_title('')
    #  ax[1, 0].set_xticks(ticks=arange(0, data.shape[1], 2))
    #   ax[1, 0].set_xticklabels(labels=round(data_latitude[::2], 2))
    # ax[1, 0].set_yticks(ticks=ticks_altitude)
    # ax[1, 0].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[1, 1].title.set_text('Radius of CO$_2$ ice particle')
    print(max(data_riceco2 * 1e6))
    pc3 = ax[1, 1].contourf(data_latitude[:], data_altitude[:], data_riceco2 * 1e6, vmin=0, vmax=60,
                            levels=arange(0, 65, 5), cmap='Greys')
    ax[1, 1].set_yscale('log')
    ax[1, 1].invert_yaxis()
    cbar3 = plt.colorbar(pc3, ax=ax[1, 1])
    cbar3.ax.set_title('µm')
    #    ax[1, 1].set_xticks(ticks=arange(0, data.shape[1], 2))
    #    ax[1, 1].set_xticklabels(labels=round(data_latitude[::2], 2))
    # ax[1, 1].set_yticks(ticks=ticks_altitude)
    #    ax[1, 1].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    fig.text(0.02, 0.5, 'Altitude (Pa)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=14)

    savename = 'cloud_evolution_latitude_sols_' + str(int(data_time[:][idx_max[0] + xtime])) + '_' + str(
        round(data_time[:][idx_max[0] + xtime] * 24 % 24, 0)) + 'h.png'
    plt.savefig(savename, bbox_inches='tight')
    plt.close()

    return savename


def display_co2_ice_max_longitude_altitude(name, data_latitude, max_mmr, max_alt, max_temp, max_satu, max_radius,
                                           max_ccnN, axis_ls, ndx, unit):
    from matplotlib.colors import LogNorm, DivergingNorm
    from numpy import arange, int_, logspace

    print('----------------------------')
    print('Enter in display_max_lon_alt')

    # PLOT
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(11, 30))

    # plot 1
    ax[0].set_title('Max ' + name + ' in altitude/longitude')
    pc = ax[0].contourf(max_mmr, norm=LogNorm(), levels=logspace(-10, 1, 12), cmap='warm')
    ax[0].set_facecolor('white')
    ax[0].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[0].set_yticklabels(labels=data_latitude[::6])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_xticklabels(labels='')
    cbar = plt.colorbar(pc, ax=ax[0])
    cbar.ax.set_title(unit)
    ax[0].set_ylabel('Latitude (°N)')

    # plot 2
    ax[1].set_title('Altitude at co2_ice mmr max')
    pc2 = ax[1].contourf(max_alt, cmap='warm')
    ax[1].set_facecolor('white')
    ax[1].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[1].set_yticklabels(labels=data_latitude[::6])
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels='')
    ax[1].set_ylabel('Latitude (°N)')
    cbar2 = plt.colorbar(pc2, ax=ax[1])
    cbar2.ax.set_title('km')

    # plot 3
    ax[2].set_title('Temperature at co2_ice mmr max')
    pc3 = ax[2].contourf(max_temp, cmap='warm')
    ax[2].set_facecolor('white')
    ax[2].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[2].set_yticklabels(labels=data_latitude[::6])
    ax[2].set_xticks(ticks=ndx)
    ax[2].set_xticklabels(labels='')
    ax[2].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[2])
    cbar3.ax.set_title('K')

    # plot 4
    divnorm = DivergingNorm(vmin=0, vcenter=1, vmax=4)
    ax[3].set_title('Saturation at co2_ice mmr max')
    pc4 = ax[3].contourf(max_satu, cmap='warm', norm=divnorm, levels=arange(0, 5))
    ax[3].set_facecolor('white')
    ax[3].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[3].set_yticklabels(labels=data_latitude[::6])
    ax[3].set_xticks(ticks=ndx)
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
    ax[4].set_xticks(ticks=ndx)
    ax[4].set_xticklabels(labels='')
    ax[4].set_ylabel('Latitude (°N)')
    cbar5 = plt.colorbar(pc5, ax=ax[4])
    cbar5.ax.set_title(u'µm')

    # plot 6
    ax[5].set_title('CCN number at co2_ice mmr max')
    pc3 = ax[5].contourf(max_ccnN, norm=DivergingNorm(vmin=0, vcenter=1), levels=arange(0, 5), cmap='warm')
    ax[5].set_facecolor('white')
    ax[5].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[5].set_yticklabels(labels=data_latitude[::6])
    ax[5].set_xticks(ticks=ndx)
    ax[5].set_xticklabels(labels=int_(axis_ls))
    ax[5].set_xlabel('Solar Longitude (°)')
    ax[5].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[5])
    cbar3.ax.set_title('nb/kg')

    fig.savefig('max_' + name + '_in_altitude_longitude.png', bbox_inches='tight')

    plt.show()


def display_co2_ice_density_column_evolution_polar_region(filename, data, time, latitude):
    from numpy import logspace
    from math import floor
    import cartopy.crs as ccrs

    data_longitude = getdata(filename=filename, target='longitude')

    platecarree = ccrs.PlateCarree(central_longitude=0)

    if latitude[0] > 0:
        orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
        title = 'North polar region'
        pole = 'north'
    else:
        orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
        title = 'South polar region'
        pole = 'south'

    y_min, y_max = orthographic._y_limits
    orthographic._y_limits = (y_min * 0.5, y_max * 0.5)
    orthographic._x_limits = (y_min * 0.5, y_max * 0.5)  # Zoom de 60° à 90°

    levels = logspace(-13, 1, 15)
    cmap = cm.get_cmap('inferno')
    cmap.set_under('w')

    norm = LogNorm()
    savename = list([])

    # PLOT
    for i in range(data.shape[0]):
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': orthographic}, figsize=(11, 11),
                               facecolor='white')
        ax.set_title(title + f', sols {floor(time[i]):.0f} LT {time[i]*24%24:.0f}')
        ax.set_facecolor('white')
        ctf = ax.contourf(data_longitude[:], latitude, data[i, :, :].filled(), norm=norm, levels=levels, transform=platecarree,
                          cmap=cmap)
        workaround_gridlines(platecarree, axes=ax, pole=pole)
        ax.set_global()
        cbar = fig.colorbar(ctf)
        cbar.ax.set_title('kg.m$^{-2}$')
        savename.append(f'co2_ice_density_column_evolution_{i}.png')
        plt.savefig(savename[i], bbox_inches='tight')
        plt.close(fig)

    # create the gif
    create_gif(savename)
    return


def display_emis_polar_projection_garybicas2020_figs11_12(filename, data, time, levels, cmap, unit, savename):
    import cartopy.crs as ccrs

    if isinstance(data, ndarray):
        array_mask = True
    else:
        array_mask = False

    platecarree = ccrs.PlateCarree(central_longitude=0)

    data_longitude = getdata(filename=filename, target='longitude')
    data_latitude = getdata(filename=filename, target='latitude')

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[60, 90])
    data_np, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[60, 90])

    latitude_sp, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[-90, -60])
    data_sp, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[-90, -60])

    # Slice data binned in 15°Ls during their winter period
    data_np = data_np[12:, :, :]
    data_sp = data_sp[0:12, :, :]

    # North polar region
    orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic._y_limits
    orthographic._y_limits = (y_min*0.5, y_max*0.5)
    orthographic._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=(20, 15))
    fig.suptitle('North polar region', fontsize=20)
    for i, axes in enumerate(ax.reshape(-1)):
        if i <12:
            axes.set_title(f'{int(time[i] + 180)}° - {int(time[i+1] + 180)}°')
            if array_mask:
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels,
                                    transform=platecarree, cmap=cmap)
            else:
                if data_np[i, :, :].mask.all():
                    continue
                else:
                    ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels,
                                        transform=platecarree, cmap=cmap)
            axes.set_global()
            workaround_gridlines(platecarree, axes=axes, pole='north')
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    plt.savefig(savename+'northern_polar_region_as_fig11_gary-bicas2020.png', bbox_inches='tight')

    # South polar region
    orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic._y_limits
    orthographic._y_limits = (y_min*0.5, y_max*0.5)
    orthographic._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': orthographic}, figsize=(20, 15))
    fig.suptitle('South polar region', fontsize=20)
    for i, axes in enumerate(ax.reshape(-1)):
        if i <12:
            axes.set_title(f'{int(time[i])}° - {int(time[i+1])}°')
            if array_mask:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels,
                                    transform=platecarree, cmap=cmap)
            else:
                if data_sp[i, :, :].mask.all():
                    continue
                else:
                    ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels,
                                        transform=platecarree, cmap=cmap)
            workaround_gridlines(platecarree, axes=axes, pole='south')
            axes.set_global()
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 3].get_position().x0 + ax[0, 3].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    plt.savefig(savename+'southern_polar_region_as_fig12_gary-bicas2020.png', bbox_inches='tight')
    return


def display_riceco2_global_mean(filename, list_data):
    from numpy import mean, zeros
    list_data[0] = mean(list_data[0], axis=2)
    list_data[1] = mean(list_data[1], axis=2)

    data_altitude = getdata(filename=filename, target='altitude')
    data_time = getdata(filename, target='Time')

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8, 11))
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


def display_riceco2_topcloud_altitude(filename, top_cloud):
    top_cloud = correction_value(data=top_cloud, operator='inf', threshold=0)

    data_latitude = getdata(filename=filename, target='latitude')
    data_time = getdata(filename=filename, target='Time')

    plt.figure(figsize=(8, 11))
    plt.title('Zonal mean of top cloud altitude')
    print(max(top_cloud / 1e3))
    cb = plt.contourf(data_time[:], data_latitude[:], top_cloud / 1e3, cmap='cool')
    ax = plt.gca()
    ax.set_facecolor('white')

    cbar = plt.colorbar(cb)
    cbar.ax.set_title('km')

    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    plt.savefig('topcloud_altitude_comparable_to_mola.png', bbox_inches='tight')
    plt.show()


def display_satuco2_thickness_atm_layer(data, data_std, savename):
    from numpy import arange, ma, array
    data = ma.masked_where(data == 0, data)

    # data from Fig 9 in Hu et al., 2012
    norhtpole = array((33, 3))
    southpole = array((35, 3))

    northpole = array(([0.5, 0.95, 1.4],
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

    southpole = array(([0.63, 0.98, 1.30],
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

    northpole_ls = 192.5 + arange(33) * 5
    southpole_ls = 12.5 + arange(35) * 5

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))

    ax[0].set_title('North pole above 60°N')
    ax[0].errorbar(arange(data.shape[1]) * 5, data[0, :] / 1e3,
                   yerr=data_std[0, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')  # 72 points binned in 5°
    ax[0].errorbar(northpole_ls, northpole[:, 1],
                   yerr=[northpole[:, 2] - northpole[:, 1], northpole[:, 1] - northpole[:, 0]], color='black', ls=' ',
                   marker='+', label='MCS MY28')
    ax[0].set_xticks(ticks=arange(0, 405, 45))
    ax[0].set_xticklabels(labels=arange(0, 405, 45))
    ax[0].legend(loc='best')

    ax[1].set_title('South pole above 60°S')
    ax[1].errorbar(arange(data.shape[1]) * 5, data[1, :] / 1e3,
                   yerr=data_std[1, :] / 1e3,
                   ls=' ', marker='+', color='blue', label='GCM')
    ax[1].errorbar(southpole_ls, southpole[:, 1],
                   yerr=[southpole[:, 2] - southpole[:, 1], southpole[:, 1] - southpole[:, 0]], color='black', ls=' ',
                   marker='+', label='MCS MY29')

    ax[1].set_xticks(ticks=arange(0, 405, 45))
    ax[1].set_xticklabels(labels=arange(0, 405, 45))
    ax[1].legend(loc='best')

    fig.text(0.06, 0.5, 'Thickness (km)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    plt.savefig(savename, bbox_inches='tight')
    plt.show()


def display_satuco2_with_co2_ice_altitude_ls(filename, data_satuco2_north, data_satuco2_eq, data_satuco2_south,
                                             data_co2ice_north, data_co2ice_eq, data_co2ice_south, latitude_north,
                                             latitude_eq, latitude_south, binned):
    from numpy import array, round

    # Info latitude
    data_latitude = getdata(filename, target='latitude')
    list_latitudes = [latitude_north, latitude_eq, latitude_south]

    # chopper l'intervalle de latitude comprise entre value-1 et value+1
    data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS = mesoclouds_observed()

    list_obs = [data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS,
                data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS]

    data_altitude = getdata(filename, target='altitude')
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
        data_zareoid = getdata(filename, target='zareoid')
        data_surface_local = gcm_surface_local(data_zareoid[:, :, :, :])

    data_time = getdata(filename=filename, target='Time')
    if binned.lower() == 'y':
        data_time = data_time[::60]  # 5°Ls binned
        data_zareoid = data_zareoid[::12, :, :, :]
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

        #        for j, value in enumerate(list_obs):
        #            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
        #                                                                         list_latitudes[i])
        #            if data_obs_ls.shape[0] != 0:
        #                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
        #                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data_surface_local, dimension_data=data_latitude,
                                                        value=list_latitudes[i])

            lines_altitudes_0km = get_mean_index_alti(data_surface_local_sliced, value=0, dimension='time')
            lines_altitudes_10km = get_mean_index_alti(data_surface_local_sliced, value=1e4, dimension='time')
            lines_altitudes_40km = get_mean_index_alti(data_surface_local_sliced, value=4e4, dimension='time')
            lines_altitudes_80km = get_mean_index_alti(data_surface_local_sliced, value=8e4, dimension='time')
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

    fig.text(0.02, 0.5, '{} ({})'.format(altitude_name, altitude_unit), ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    savename = 'satuco2_zonal_mean_with_co2ice_at_{}N_{}N_{}N'.format(latitude_north, latitude_eq, latitude_south)
    if binned.lower() == 'y':
        savename = savename + '_binned'

    plt.savefig(savename+'.png', bbox_inches='tight')
    plt.show()


def display_satuco2_with_co2_ice_altitude_longitude(filename, data_satuco2_north, data_satuco2_south, data_co2ice_north,
                                                    data_co2ice_south, latitude_north, latitude_south, binned):
    from numpy import array, round

    # Info latitude
    data_latitude = getdata(filename, target='latitude')
    list_latitudes = [latitude_north, latitude_south]

    data_time = getdata(filename=filename, target='Time')
    list_time_range = array(([270, 300], [0, 30]))

    # Info longitude
    data_longitude = getdata(filename, target='longitude')

    # chopper l'intervalle de latitude comprise entre value-1 et value+1
    data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS = mesoclouds_observed()

    list_obs = [data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS,
                data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS]

    data_altitude = getdata(filename, target='altitude')
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
        data_zareoid = getdata(filename, target='zareoid')
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

        #        for j, value in enumerate(list_obs):
        #            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
        #                                                                         list_latitudes[i])
        #            if data_obs_ls.shape[0] != 0:
        #                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0]) * 1e-3, zeros(data_obs_ls.shape[0]),
        #                           -ones(data_obs_ls.shape[0]) * 3, color='black')

        if altitude_unit == 'Pa':
            data_surface_local_sliced, tmp = slice_data(data_surface_local, dimension_data=data_latitude,
                                                        value=list_latitudes[i])

            data_surface_local_sliced, tmp = slice_data(data_surface_local_sliced, dimension_data=data_time,
                                                        value=list_time_range[i])

            lines_altitudes_0km = get_mean_index_alti(data_surface_local_sliced, value=0, dimension='longitude')
            lines_altitudes_10km = get_mean_index_alti(data_surface_local_sliced, value=1e4, dimension='longitude')
            lines_altitudes_40km = get_mean_index_alti(data_surface_local_sliced, value=4e4, dimension='longitude')
            lines_altitudes_80km = get_mean_index_alti(data_surface_local_sliced, value=8e4, dimension='longitude')
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

    fig.text(0.02, 0.5, '{} ({})'.format(altitude_name, altitude_unit), ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    savename = 'satuco2_time_mean_with_co2ice_at_{}N_{}N'.format(latitude_north, latitude_south)
    if binned.lower() == 'y':
        savename = savename + '_binned'

    plt.savefig(savename+'.png', bbox_inches='tight')
    plt.show()


def display_satuco2_zonal_mean_day_night(data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night,
                                         data_altitude, ndx, axis_ls, title, savename):
    from numpy import array, round

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

    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_satuco2_altitude_latitude(data, data_altitude, data_latitude):
    from matplotlib.colors import DivergingNorm
    from numpy import mean, arange, where, searchsorted, max, argmax
    from scipy.interpolate import interp2d

    # ----------------------------------------------------------
    #    south pole
    # -------------
    #    idx_lat_1 = (abs(data_latitude[:] + 57)).argmin() + 1
    #    idx_lat_2 = (abs(data_latitude[:] + 90)).argmin() + 1
    #    idx_ls_1 = (abs(data_time[:] - 0)).argmin()
    #    idx_ls_2 = (abs(data_time[:] - 180)).argmin()
    #    savename = 'altitude_latitude_saturation_south_pole'
    # ----------------------------------------------------------
    #    north pole
    # -------------
    idx_lat_1 = (abs(data_latitude[:] + 90)).argmin()
    idx_lat_2 = (abs(data_latitude[:] - 90)).argmin()
    #    idx_ls_1 = (abs(data_time[:] - 180)).argmin()
    #    idx_ls_2 = (abs(data_time[:] - 360)).argmin()
    savename = 'altitude_latitude_saturation_max'
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

    fmt = lambda x: "{:1d}".format(int(x))

    # plot
    divnorm = DivergingNorm(vmin=0, vcenter=1, vmax=4)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('CO$_2$ saturation max')
    plt.contourf(zoomed_data, cmap='seismic', norm=divnorm, levels=arange(0, 5))
    plt.xticks(ticks=arange(0, len(data_latitude[idx_lat_1:idx_lat_2]), 4),
               labels=data_latitude[idx_lat_1:idx_lat_2:4])
    plt.yticks(ticks=arange(0, len(interp_altitude)), labels=[fmt(round(i)) for i in data_altitude[:]])
    cbar = plt.colorbar()
    cbar.ax.set_title('')
    plt.xlabel('Latitude (°)')
    plt.ylabel('Altitude (km)')
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_temp_gg2011_fig6(data, data_localtime, data_surface):
    from numpy import arange

    cmap = colormap_idl_rainbow_plus_white()
    data_surface = data_surface / 1e3

    fig = plt.figure(figsize=(8, 11))
    pc = plt.contourf(data_localtime, data_surface, data.T, levels=arange(0, 125, 5), cmap=cmap)
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
    data_latitude = getdata(filename=filename, target='latitude')

    cmap = colormap_idl_rainbow_plus_white()

    # plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('Temperature - Tcond CO$_2$')
    ctf = ax.contourf(data_latitude[:], data_altitude/1e3, data, levels=arange(0, 130, 10), cmap=cmap)
    ax.set_ylim(40, 120)

    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('K')

    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Altitude above areoid (km)')

    plt.savefig('temp_zonalmean_altitude_latitude_ls_0-30_gg2011_fig7.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig8(filename, data_zonal_mean, data_thermal_tides, data_altitude):
    data_latitude = getdata(filename=filename, target='latitude')

    fig, ax = plt.subplots(nrows=2, figsize=(11, 8))

    cmap = colormap_idl_rainbow_plus_white()
    # Zonal mean at 16 H local time
    ctf1 = ax[0].contourf(data_latitude[:], data_altitude/1e3, data_zonal_mean, levels=arange(110, 240, 10), cmap=cmap)
    cbar = plt.colorbar(ctf1, ax=ax[0])
    cbar.ax.set_title('K')
    ax[0].set_ylim(40, 120)
    ax[0].set_xlabel('Latitude (°N)')
    ax[0].set_ylabel('Altitude above areoid (km)')

    # Thermal tides: 12h - 00h
    ctf2 = ax[1].contourf(data_latitude[:], data_altitude/1e3, data_thermal_tides, levels=arange(-20, 32, 4), cmap=cmap)
    cbar2 = plt.colorbar(ctf2, ax=ax[1])
    cbar2.ax.set_title('K')
    ax[1].set_ylim(40, 120)

    ax[1].set_xlabel('Latitude (°N)')
    ax[1].set_ylabel('Altitude above areoid (km)')
    plt.savefig('temp_gg2011_fig8.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_gg2011_fig9(filename, data, data_altitude):
    data_longitude = getdata(filename=filename, target='longitude')

    cmap = colormap_idl_rainbow_plus_white()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
    pc = ax.contourf(data_longitude[:], data_altitude/1e3, data, levels=arange(0, 130, 10), cmap=cmap)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title('K')
    ax.set_ylim(40, 120)

    plt.title('Temperature -Tcond CO$_2$')
    plt.xlabel('Longituge (°E)')
    plt.ylabel('Altitude above areoid (km)')
    plt.savefig('temp_altitude_longitude_ls_0-30_LT_16H_lat_0N_gg2011_fig9.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_structure_polar_region(filename, data_north, data_south, norm, levels, unit, savename):
    cmap = 'coolwarm'

    data_altitude = getdata(filename=filename, target='altitude')
    data_time = getdata(filename=filename, target='Time')

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
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()
    return


def display_temp_cold_pocket_SPICAM(filename, data):
    fig, ax = plt.subplots('')

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


def display_vars_altitude_latitude(filename, data, unit, title, savename):
    data_altitude = getdata(filename=filename, target='altitude')
    data_latitude = getdata(filename=filename, target='latitude')

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

    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_localtime(filename, data, data_localtime, title, unit, savename):
    from numpy import arange, zeros, round

    data_altitude = getdata(filename=filename, target='altitude')

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
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_longitude(filename, data, unit, title, savename):
    from numpy import zeros, arange, round, int_, searchsorted

    data_altitude = getdata(filename=filename, target='altitude')
    data_longitude = getdata(filename=filename, target='longitude')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
    if unit == '':
        scale = zeros(5)
        scale[1] = 1
        scale[2:] = 10 ** arange(1, 4)
        pc = ax.contourf(data_longitude[:], data_altitude[:], data,
                         norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=1000), levels=scale,
                         cmap='seismic', extend='max')
    else:
        ax.set_yscale('log')
        ax.invert_yaxis()
        pc = ax.contourf(data_longitude[:], data_altitude[:], data, cmap='hot')

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit)

    plt.title(title)
    plt.xlabel('Longituge (°E)')
    plt.ylabel('{} ({})'.format(data_altitude.name, data_altitude.units))
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_vars_altitude_ls(filename, data_1, data_2, levels, title, savename, latitude_selected=None):
    from numpy import round

    data_altitude = getdata(filename, target='altitude')
    ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]

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

    data_time = getdata(filename, target='Time')
    ndx, axis_ls = get_ls_index(data_time)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))

    cb = axes.contourf(data_time[:] * 12, data_altitude[:], data_1, norm=LogNorm(vmin=1e-13, vmax=1e2), levels=levels,
                       cmap='coolwarm')
    axes.contour(data_time[:] * 12, data_altitude[:], data_2, norm=LogNorm(vmin=1e-13, vmax=1e2), levels=levels,
                 colors='black')

    cbar = plt.colorbar(cb, ax=axes)
    cbar.ax.set_title('kg/kg')

    axes.set_xlabel('Solar longitude (°)')

    axes.set_yticks(ticks=ticks_altitude)
    axes.set_ylabel(altitude_name + ' (' + units + ')')
    if units == 'Pa':
        try:
            data_zareoid = getdata(filename, target='zareoid')
        except:
            print('Zareoid is missing!')
            exit()
        if latitude_selected is not None:
            data_latitude = getdata(filename, target='latitude')
            data_zareoid, tmp = slice_data(data_zareoid, dimension_data=data_latitude, value=latitude_selected)

        lines_altitudes_0km = get_mean_index_alti(data_zareoid, 0)
        lines_altitudes_10km = get_mean_index_alti(data_zareoid, 1e4)
        lines_altitudes_40km = get_mean_index_alti(data_zareoid, 4e4)
        lines_altitudes_80km = get_mean_index_alti(data_zareoid, 8e4)

        axes.plot(data_altitude[lines_altitudes_0km], '--', color='grey')
        axes.plot(data_altitude[lines_altitudes_10km], '--', color='grey')
        axes.plot(data_altitude[lines_altitudes_40km], '--', color='grey')
        axes.plot(data_altitude[lines_altitudes_80km], '--', color='grey')

        axes.text(data_time[-1] * 12 + 1, data_altitude[lines_altitudes_0km[0]], '0 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1] * 12 + 1, data_altitude[lines_altitudes_10km[0]], '10 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1] * 12 + 1, data_altitude[lines_altitudes_40km[0]], '40 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1] * 12 + 1, data_altitude[lines_altitudes_80km[0]], '80 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)

        axes.set_yscale('log')
        axes.invert_yaxis()
    else:
        axes.set_yticklabels(labels=round(data_altitude[ticks_altitude], 0))

    axes.set_title(title)
    fig.savefig(savename + '.png', bbox_inches='tight')
    fig.show()


def display_vars_latitude_ls(filename, name_target, data, unit, norm, levels, observation=False, latitude_selected=None,
                             localtime_selected=None,title=None, TES=None, PFS=None, MVALS=None, layer=None,
                             savename='test'):
    nsubplot = 1
    if TES:
        nsubplot += 1
    if MVALS:
        nsubplot += 1

    cmap = 'coolwarm'

    data_time = getdata(filename, target='Time')
    if localtime_selected is not None:
        data_local_time, idx, statsfile = check_local_time(data_time=data_time[:], selected_time=localtime_selected)
        data_time = data_time[idx::len(data_local_time)]
    ndx, axis_ls = get_ls_index(data_time)

    data_altitude = getdata(filename, target='altitude')

    data_latitude = getdata(filename, target='latitude')
    if latitude_selected is not None:
        data_latitude, latitude_selected = slice_data(data_latitude, dimension_data=data_latitude,
                                                      value=[latitude_selected[0], latitude_selected[-1]])
    else:
        latitude_selected = [-90, 90]
    data_latitude = data_latitude[::-1]

    # PLOT
    fig, ax = plt.subplots(nrows=nsubplot, ncols=1, figsize=(11, 11))
    fig.subplots_adjust()

    isubplot = 0

    if TES:
        # Extract TES data
        print('Takes TES data')
        if name_target == 'tsurf':
            data_time_tes = ObsTES(target='time', year=None)  # None == climatology
            data_latitude_tes = ObsTES(target='latitude', year=None)
            data_tes = ObsTES(target='Tsurf_nit', year=None)  # (time, latitude, longitude), also Tsurf_day/Tsurf_nit
            ax[isubplot].set_title('TES climatology')
            idx1 = (abs(data_time_tes[:] - 360 * 2)).argmin()
            idx2 = (abs(data_time_tes[:] - 360 * 3)).argmin()
            data_time_tes = data_time_tes[idx1:idx2] - 360 * 2
            data_tes, tmp = vars_zonal_mean(filename='', data=data_tes[idx1:idx2, :, :], layer=None, flip=True)
        elif name_target == 'temp':
            data_time_tes = ObsTES(target='time', year=25)
            data_latitude_tes = ObsTES(target='latitude', year=25)
            data_altitude_tes = ObsTES(target='altitude', year=25)  # Pa
            data_tes = TES(target='T_limb_day', year=25)
            year = 1

            # Select altitude for TES data close to the specified layer
            target_layer = data_altitude[::-1][layer]
            in_range = (data_altitude_tes[:] >= target_layer) & (data_altitude_tes[:] <= target_layer)
            if not in_range.any():
                data_tes = zeros((data_tes.shape[2], data_tes.shape[0]))
                altitude_tes = 'out range'
            else:
                idx = abs(data_altitude_tes[:] - data_altitude[::-1][layer]).argmin()
                data_tes = data_tes[:, idx, :, :]
                data_tes, tmp = vars_zonal_mean('', data_tes[:, :, :], layer=None)
                altitude_tes = data_altitude_tes[idx]

            data_tes, tmp = vars_zonal_mean('', data_tes[:, :, :], layer=None)
            ax[isubplot].set_title('TES Mars Year {:d} at {} {}'.format(24 + year, altitude_tes,
                                                                        data_altitude_tes.units))
        ax[isubplot].contourf(data_time_tes[:], data_latitude_tes[:], data_tes, levels=levels, cmap=cmap)
        isubplot += 1

    if MVALS:
        # Extract mvals data
        print('Takes M. VALS data')
        data_mvals = SimuMV(target=name_target, localtime=2)
        data_time_mvals = SimuMV(target='Time', localtime=2)
        data_latitude_mvals = SimuMV(target='latitude', localtime=2)
        data_altitude_mvals = SimuMV(target='altitude', localtime=2)
        if layer is not None:
            data_mvals = data_mvals[:, layer, :, :]

        if data_mvals.ndim == 3:
            data_mvals = correction_value(data_mvals[:, :, :], operator='inf', threshold=1e-13)
        else:
            data_mvals = correction_value(data_mvals[:, :, :, :], operator='inf', threshold=1e-13)

        # Compute zonal mean
        data_mvals, tmp = vars_zonal_mean(filename='', data=data_mvals, layer=layer, flip=False)

        if name_target == 'temp':
            ax[isubplot].set_title('M. VALS at {:.2e} {}'.format(data_altitude_mvals[layer], data_altitude_mvals.units))
        else:
            ax[isubplot].set_title('M. VALS')

        ax[isubplot].contourf(data_time_mvals[:], data_latitude_mvals[:], data_mvals, levels=levels, cmap=cmap)
        isubplot += 1

    if isubplot == 0:
        ctf = ax.contourf(data_time[:], data_latitude[:], data, norm=norm, levels=levels, cmap=cmap, zorder=10)
    else:
        ctf = ax[isubplot].contourf(data_time[:], data_latitude[:], data, norm=norm, levels=levels, cmap=cmap,
                                    zorder=10)

    if name_target == 'temp':
        if isubplot == 0:
            ax.set_title(title)
        else:
            ax[isubplot].set_title('My work at {:.2e} {}'.format(data_altitude[::-1][layer], data_altitude.units))
    else:
        if isubplot == 0:
            ax.set_title(title)
        else:
            ax[isubplot].set_title('My work')

    if observation:
        # chopper l'intervalle de latitude comprise entre value-1 et value+1
        data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
        data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS = mesoclouds_observed()

        list_obs = [data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS,
                    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS]
        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         [latitude_selected[0], latitude_selected[-1]])
            if data_obs_ls.shape[0] != 0:
                plt.scatter(data_obs_ls, data_obs_latitude, color='black', marker='+', zorder=1)

    if isubplot != 0:
        for axes in ax.reshape(-1):
            axes.set_facecolor('white')
            axes.set_xticks(ticks=axis_ls)
        fig.text(0.02, 0.5, 'Latitude (°N)'.format('g'), ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)
        plt.title(title)
        pos0 = ax[0].get_position()
        cbaxes = fig.add_axes([pos0.x0, 0.95, pos0.x1 - pos0.x0, 0.025])
        cbar = plt.colorbar(ctf, cax=cbaxes, orientation="horizontal")
        cbar.ax.set_title(unit)
    else:
        ax.set_xticks(ticks=axis_ls)
        ax.set_xlim(axis_ls[0], axis_ls[-1])
        ax.set_xlabel('Solar longitude (°)')
        ax.set_ylabel('Latitude (°N)')
        cbar = plt.colorbar(ctf)
        cbar.ax.set_title(unit)

    plt.savefig(savename + '.png', bbox_inches='tight')
    return


def display_vars_1fig_profiles(filename, data, latitude_selected, xmin, xmax, xlabel, xscale='linear', yscale='linear',
                               second_var=None, xmin2=None, xmax2=None, xlabel2=None, xscale2=None, title='',
                               savename='profiles', title_option=None):
    from numpy import arange

    data_time = getdata(filename, target='Time')

    if data_time.units == 'degrees':
        time_unit = u'° Ls'
    else:
        time_unit = 'sols'

    data_altitude = getdata(filename, target='altitude')
    if data_altitude.units == 'm':
        units = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
        yscale = 'linear'
    elif data_altitude.units == 'km':
        units = data_altitude.units
        altitude_name = 'Altitude'
        yscale = 'linear'
    else:
        units = data_altitude.units
        altitude_name = 'Pressure'

    for i, d, s in zip(arange(len(savename)), data, savename):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
        # plot variable 1
        if d.ndim > 1:
            for j in range(d.shape[1]):
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.plot(d[:, j], data_altitude[:], label='%.2f°N' % (latitude_selected[j]))
        else:
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.plot()
        # plot variable 2 if exists
        if second_var is not None:
            ax2 = ax.twiny()
            ax2.set_xscale(xscale2)
            ax2.set_xlim(xmin2, xmax2)
            ax2.set_xlabel(xlabel2)
            for j in range(second_var[0].shape[1]):
                ax2.plot(second_var[i][:, j], data_altitude[:], ls='--')
        ax.set_xlim(xmin, xmax)
        if altitude_name == 'Pressure':
            ax.invert_yaxis()
        ax.legend(loc='best')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(altitude_name + ' (' + units + ')')
        if title_option is not None:
            ax.set_title(f'{title}, and {title_option[i][0]:.0f} - {title_option[i][1]:.0f} {time_unit}')
        fig.savefig(s + '.png', bbox_inches='tight')
        plt.close(fig)

    create_gif(savename)
    return


def display_vars_2fig_profile(filename, data1, data2):
    data_time = getdata(filename, target='Time')

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(11, 11))

    ax[0].plot(data_time[:], data1 * 1e3, color='black')  # kg to g

    ax[1].plot(data_time[:], data2 * 1e3, color='black')  # kg to g

    ax[0].set_xlim(data_time[0], data_time[-1])
    fig.text(0.02, 0.5, 'Cumulative masses ({})'.format('g'), ha='center', va='center', rotation='vertical',
             fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    fig.savefig('co2ice_cumulative_mass_polar_region.png', bbox_inches='tight')


def display_vars_4figs_polar_projection(filename, data_riceco2):
    from cartopy import crs

    lons = getdata(filename, target='longitude')
    lats = getdata(filename, target='latitude')

    plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(1, 1, 1, projection=crs.Orthographic())
    ax1.contourf(lons[:], lats[:], data_riceco2[0, 0, :, :], 60, transform=crs.Orthographic())
    plt.show()


def display_vars_stats_zonalmean(filename, data):
    data_latitude = getdata(filename, target='latitude')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
    ax.set_title('Zonal mean of surface emissivity at 14h')
    ctf = ax.contourf(arange(12), data_latitude[:], data.T, levels=([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.]))
    ax.set_xlabel('Months')
    ax.set_ylabel('Latitude (°N)')
    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('W.m$^{-1}$')
    plt.savefig('emis_stats_zonalmean_14h.png', bbox_inches='tight')
    plt.close(fig)
    return


def display_vars_polar_projection(filename, data_np, data_sp, levels, unit, cmap, suptitle, savename):
    import cartopy.crs as ccrs

    data_longitude = getdata(filename=filename, target='longitude')
    data_latitude = getdata(filename=filename, target='latitude')

    latitude_np, tmp = slice_data(data=data_latitude, dimension_data=data_latitude, value=[60, 90])
    latitude_sp, tmp = slice_data(data=data_latitude, dimension_data=data_latitude, value=[-60, -90])

    platecarree = ccrs.PlateCarree(central_longitude=0)

    orthographic_north = ccrs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic_north._y_limits
    orthographic_north._y_limits = (y_min*0.5, y_max*0.5)
    orthographic_north._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°

    orthographic_south = ccrs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic_south._y_limits
    orthographic_south._y_limits = (y_min*0.5, y_max*0.5)
    orthographic_south._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°

    # PLOT
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 11))
    fig.suptitle(suptitle, fontsize=20)
    ax1 = plt.subplot(121, projection=orthographic_south)
    ax2 = plt.subplot(122, projection=orthographic_north)

    # South polar region
    ax1.set_title('South polar region')
    ctf = ax1.contourf(data_longitude[:], latitude_sp, data_sp, levels=levels, transform=platecarree, cmap=cmap)
    workaround_gridlines(platecarree, axes=ax1, pole='south')
    ax1.set_global()

    # North polar region
    ax2.set_title('North polar region')
    ax2.contourf(data_longitude[:], latitude_np, data_np, levels=levels, transform=platecarree, cmap=cmap)
    workaround_gridlines(platecarree, axes=ax2, pole='north')
    ax2.set_global()

    pos1 = ax2.get_position().x0 + ax2.get_position().width + 0.05
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([pos1, ax2.get_position().y0, 0.03, ax2.get_position().height])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.ax.set_title(unit)

    plt.savefig(savename+'.png', bbox_inches='tight')
    return


def display_vars_polar_projection_multiplot(filename, data, time, localtime, levels, norm, cmap, unit, savename):
    import cartopy.crs as ccrs
    from numpy import max, unique, ma

    if isinstance(data, ma.MaskedArray):
        array_mask = True
    else:
        array_mask = False

    platecarree = ccrs.PlateCarree(central_longitude=0)

    data_longitude = getdata(filename=filename, target='longitude')
    data_latitude = getdata(filename=filename, target='latitude')

    # Slice data in polar regions
    latitude_np, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[60, 90])
    data_np, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[60, 90])
    latitude_sp, tmp = slice_data(data_latitude, dimension_data=data_latitude[:], value=[-90, -60])
    data_sp, tmp = slice_data(data[:, :, :], dimension_data=data_latitude[:], value=[-90, -60])
    if array_mask:
        data_np[data_np.mask] = -1
        data_sp[data_sp.mask] = -1

    cmap = cm.get_cmap(cmap)
    cmap.set_under('w')

    # North polar region
    orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=90, globe=False)
    y_min, y_max = orthographic._y_limits
    orthographic._y_limits = (y_min*0.5, y_max*0.5)
    orthographic._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=5, ncols=5, subplot_kw={'projection': orthographic}, figsize=(20, 20))
    fig.suptitle('North polar region ({})'.format(unit), fontsize=20)

    for i, axes in enumerate(ax.reshape(-1)):
        if i < 24:
            axes.set_title(f'{int(time[i])}° - {int(time[i+1])}°')
            print(unique(data_np[i, :, :]).shape[0])
            if array_mask and unique(data_np[i, :, :]).shape[0] != 1:
                # Cas special où il y a une ligne qui a des valeurs mais pas totalement
                #  => il faut au moins 1 ligne complète sans valeurs masqués pour que ça fonctionne
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels, norm=norm,
                                    transform=platecarree, cmap=cmap, extend='max')
            else:
                ctf = axes.contourf(data_longitude[:], latitude_np, data_np[i, :, :], levels=levels, norm=norm,
                                    transform=platecarree, cmap=cmap, extend='max')

            axes.set_global()
            workaround_gridlines(platecarree, axes=axes, pole='north')
            axes.set_facecolor('white')

    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 4].get_position().x0 + ax[0, 4].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    fig.tight_layout()
    plt.savefig(savename+'_northern_polar_region.png', bbox_inches='tight')

    # South polar region
    orthographic = ccrs.Orthographic(central_longitude=0, central_latitude=-90, globe=False)
    y_min, y_max = orthographic._y_limits
    orthographic._y_limits = (y_min*0.5, y_max*0.5)
    orthographic._x_limits = (y_min*0.5, y_max*0.5)  # Zoom de 60° à 90°
    fig, ax = plt.subplots(nrows=5, ncols=5, subplot_kw={'projection': orthographic}, figsize=(20, 20))
    fig.suptitle('South polar region ({})'.format(unit), fontsize=20)

    for i, axes in enumerate(ax.reshape(-1)):
        if i <24:
            axes.set_title(f'{int(time[i])}° - {int(time[i+1])}°')
            print(unique(data_sp[i, :, :]).shape[0])
            if array_mask and unique(data_sp[i, :, :]).shape[0] != 1:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels, norm=norm,
                                    transform=platecarree, cmap=cmap, extend='max')
            else:
                ctf = axes.contourf(data_longitude[:], latitude_sp, data_sp[i, :, :], levels=levels, norm=norm,
                                    transform=platecarree, cmap=cmap, extend='max')

            axes.set_global()
            workaround_gridlines(platecarree, axes=axes, pole='south')
            axes.set_facecolor('white')
    pos1 = ax[0, 0].get_position().x0
    pos2 = (ax[0, 4].get_position().x0 + ax[0, 4].get_position().width) - pos1
    cbar_ax = fig.add_axes([pos1, 0.925, pos2, 0.03])
    fig.colorbar(ctf, cax=cbar_ax, orientation='horizontal')
    fig.tight_layout()

    plt.savefig(savename+'_southern_polar_region.png', bbox_inches='tight')


def workaround_gridlines(src_proj, axes, pole):
    from numpy import linspace, zeros
    # Workaround for plotting lines of constant latitude/longitude as gridlines
    # labels not supported for this projection.
    if pole == 'north':
        lats = linspace(60, 90, num=31, endpoint=True)
    elif pole == 'south':
        lats = linspace(-90, -59, num=31, endpoint=True)
    else:
        print('Wrong input pole')
        exit()
    lons = linspace(0, 360, num=360, endpoint=False)

    yn = zeros(len(lats))
    lona = lons + yn.reshape(len(lats), 1)

    cs2 = axes.contour(lons, lats, lona, 10, transform=src_proj, colors='grey', linestyles='solid',
                       levels=arange(0, 450, 90), linewidths=2)
    axes.clabel(cs2, fontsize=10, inline=True, fmt='%1.0f', inline_spacing=30)

    yt = zeros(len(lons))
    lata = lats.reshape(len(lats), 1) + yt
    cs = axes.contour(lons, lats, lata, 10, transform=src_proj, colors='grey', linestyles='solid', levels=2,
                      linewidths=2)
    axes.clabel(cs, fontsize=10, inline=True, fmt='%1.0f', inline_spacing=20)
