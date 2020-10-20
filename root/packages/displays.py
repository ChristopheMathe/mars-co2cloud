import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm, LogNorm
from .lib_function import *
from .ncdump import *


# ==================================================================================================================== #
# ================================================== DISPLAY 1D ====================================================== #
# ==================================================================================================================== #
def display_1d(data):
    plt.figure()
    plt.plot(data)
    plt.xlim(0, len(data))
    plt.xlabel('Time')
    plt.ylabel(data.name + ' (' + data.units + ')')
    plt.show()


# ==================================================================================================================== #
# ================================================== DISPLAY 2D ====================================================== #
# ==================================================================================================================== #
def display_2d(data):
    from matplotlib.widgets import Slider
    from matplotlib.contour import QuadContourSet

    def compute_and_plot(ax, alpha):
        CS = QuadContourSet(ax, data[0, alpha, :, :])
        plt.clabel(CS, inline=1, fontsize=10)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Add Ls and Altitude')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

    compute_and_plot(ax, 0)

    # Define slider for alpha
    alpha_axis = plt.axes([0.12, 0.1, 0.65, 0.03])
    alpha_slider = Slider(alpha_axis, 'Alt', 0, len(data[0, :, 0, 0]) - 1, valinit=0, valstep=1)

    def update(ax, val):
        alpha = val
        ax.cla()
        compute_and_plot(ax, alpha)
        plt.draw()

    alpha_slider.on_changed(lambda val: update(ax, val))

    plt.show()


# ==================================================================================================================== #
# =========================================== DISPLAY 1 fig: lat - ls ================================================ #
# ==================================================================================================================== #
def display_colonne(filename, data, unit, norm, levels, latitude_selected=None, title=None, savename='test'):
    from matplotlib.colors import LogNorm
    from numpy import int_

    data_altitude = getdata(filename, target='altitude')
    data_time = getdata(filename, target='Time')
    ndx, axis_ls = get_ls_index(data_time)

    data_latitude = getdata(filename, target='latitude')
    if latitude_selected is not None:
        data_latitude, latitude_selected = slice_data(data_latitude, dimension_data=data_latitude,
                                                      value=[latitude_selected[0], latitude_selected[-1]])
    data_latitude = data_latitude[::-1]

    # chopper l'intervalle de latitude comprise entre value-1 et value+1
    data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS = mesoclouds_observed()

    list_obs = [data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS]

    # plot
    plt.figure(figsize=(11, 8))

    if unit is 'pr.µm':
        # convert kg/m2 to pr.µm
        plt.contourf(data_time[:], data_latitude[:], data * 1e3, levels=levels, cmap='coolwarm', zorder=1)
    else:
        if norm == 'log':
            ctf = plt.contourf(data_time[:], data_latitude[:], data, norm=LogNorm(), levels=levels, cmap='coolwarm',
                               zorder=1)
        else:
            ctf = plt.contourf(data_time[:], data_latitude[:], data, levels=levels, cmap='coolwarm', zorder=1)

    for j, value in enumerate(list_obs):
        data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                     [latitude_selected[0],latitude_selected[-1]])
        if data_obs_ls.shape[0] != 0:
            plt.scatter(data_obs_ls, data_obs_latitude, color='black', marker='+', zorder=10)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.xticks(ticks=axis_ls, labels=int_(axis_ls))
    plt.xlim(0, 360)
    if data_latitude.shape[0] > 18:
        plt.yticks(ticks=data_latitude[::6], labels=data_latitude[::6])
    plt.ylim(data_latitude[0], data_latitude[-1])
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    cbar = plt.colorbar(ctf)
    cbar.ax.set_title(unit)
    plt.title(title)
    plt.savefig(savename+'.png', bbox_inches='tight')


# =====================================================================================================================#
# ============================================ DISPLAY latitude-altitude ==============================================#
# =====================================================================================================================#
def display_altitude_latitude(data, unit, title, data_altitude, data_latitude, savename):
    from numpy import arange, round
    if unit is 'pr.µm':
        data = data * 1e3  # convert kg/m2 to pr.µm

    # plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title(title)
    plt.contourf(data, cmap='inferno')

    ax.set_yticks(ticks=arange(0, len(data_altitude), 3))
    ax.set_yticklabels(labels=round(data_altitude[::3] / 1e3, 2))
    ax.set_xticks(ticks=arange(0, len(data_latitude), 3))
    ax.set_xticklabels(labels=data_latitude[::3])

    cbar = plt.colorbar()
    cbar.ax.set_title(unit)

    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Altitude (km)')

    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


# ==================================================================================================================== #
# ============================================= DISPLAY latitude-altitude satuco2 ==================================== #
# ==================================================================================================================== #
def display_altitude_latitude_satuco2(data, data_time, data_altitude, data_latitude):
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


# ==================================================================================================================== #
# ================================ DISPLAY max value in altitude / longitude ========================================= #
# ==================================================================================================================== #
def display_max_lon_alt(name, data_latitude, max_mmr, max_alt, max_temp, max_satu, max_radius, max_ccnN, axis_ls,
                        ndx, unit):
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


# ==================================================================================================================== #
# ============================================ DISPLAY Zonal mean ==================================================== #
# ==================================================================================================================== #
def display_vertical_profile(data, data_latitude, data_pressure, title):
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


# ==================================================================================================================== #
# ============================================ DISPLAY Temperature difference ======================================== #
# ==================================================================================================================== #
def display_temperature_diff(data):
    plt.figure()
    plt.contourf(data)
    plt.savefig('temper')
    plt.show()


def display_max_lon_alt_satuco2(data, data_latitude, axis_ls, ndx, max_alt):
    from numpy import arange, int_
    from matplotlib.colors import DivergingNorm

    print(data.shape)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 8))
    ax[0].set_title('Max saturation along altitude/longitude')
    pc = ax[0].contourf(data, norm=DivergingNorm(vmin=0, vcenter=1, vmax=10), levels=arange(0, 10), cmap='seismic',
                        extend='max')
    ax[0].set_yticks(ticks=arange(0, len(data_latitude), 3))
    ax[0].set_yticklabels(labels=data_latitude[::3])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_ylabel('Latitude (°N)')
    plt.colorbar(pc, ax=ax[0])

    ax[1].set_title('Altitude at max saturation')
    pc2 = ax[1].contourf(max_alt, cmap='seismic')
    ax[1].set_yticks(ticks=arange(0, len(data_latitude), 3))
    ax[1].set_yticklabels(labels=data_latitude[::3])
    ax[1].set_ylabel('Latitude (°N)')
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels=int_(axis_ls[ndx]))
    ax[1].set_xlabel('Solar Longitude (°)')
    plt.colorbar(pc2, ax=ax[1])

    plt.savefig('max_satu.png', bbox_inches='tight')


def display_temperature_profile_evolution(data, data_latitude, data_pressure, T_sat):
    plt.figure(figsize=(11, 8))
    plt.title('Vertical profile of temperature at 0°N')

    for i in range(data.shape[0]):
        plt.semilogy(data[i, :], data_pressure[0, :, idx_lat, 0], label='+%.2i' % (2 * i) + ' h')

    plt.semilogy(T_sat, data_pressure[0, :, idx_lat, 0], '--', color='red')
    plt.xlabel('K')
    plt.ylabel('Pressure (Pa)')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('vertical_profile_temperature_equator_evolution.png', bbox_inches='tight')
    plt.show()


def display_lat_ls_maxsatuco2(data_day, data_night, data_alt_day, data_alt_night, data_latitude, ndx, axis_ls, unit,
                              title, savename):
    from matplotlib.colors import DivergingNorm, LogNorm
    from numpy import arange, zeros, int_, min, logspace, where, ones

    scale = zeros(7)
    scale[1] = 1
    scale[2:] = 10 * arange(1, 6)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11, 8))

    if unit == '':
        pc1 = ax[0, 0].contourf(data_day, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=50),
                                levels=scale, cmap='seismic', extend='max')
        pc2 = ax[1, 0].contourf(data_night, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=50),
                                levels=scale, cmap='seismic', extend='max')
    elif unit == 'µm':
        pc1 = ax[0, 0].contourf(data_day * 1e6, cmap='seismic')
        pc2 = ax[1, 0].contourf(data_night * 1e6, cmap='seismic')
    elif unit == '#/kg':
        levels = ones(4)
        levels[1] = 1e6
        levels[2] = 1e7
        levels[3] = 1e8
        pc1 = ax[0, 0].contourf(data_day, cmap='seismic', levels=levels, extend='max')
        pc2 = ax[1, 0].contourf(data_night, cmap='seismic', levels=levels, extend='max')
    else:
        pc1 = ax[0, 0].contourf(data_day, norm=LogNorm(), vmin=1e-10, levels=logspace(-10, 0, 11), cmap='seismic')
        pc2 = ax[1, 0].contourf(data_night, norm=LogNorm(), vmin=1e-10, levels=logspace(-10, 0, 11), cmap='seismic')

    pc3 = ax[0, 1].contourf(data_alt_day / 1e3, norm=DivergingNorm(vmin=0, vcenter=40, vmax=90), levels=arange(
        10) * 10, cmap='seismic')
    pc4 = ax[1, 1].contourf(data_alt_night / 1e3, norm=DivergingNorm(vmin=0, vcenter=40, vmax=90), levels=arange(
        10) * 10, cmap='seismic')
    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.2)
    pos1 = ax[0, 0].get_position()
    pos2 = ax[0, 1].get_position()
    cbar_ax1 = fig.add_axes([pos1.x0, 0.07, 0.38, 0.03])
    cbar_ax2 = fig.add_axes([pos2.x0, 0.07, 0.38, 0.03])
    cbar1 = fig.colorbar(pc1, cax=cbar_ax1, orientation="horizontal", )
    cbar1.ax.set_title(unit)
    cbar2 = fig.colorbar(pc3, cax=cbar_ax2, orientation="horizontal")
    cbar2.ax.set_title('km')

    for axe in ax.reshape(-1):
        axe.set_yticks(ticks=arange(0, len(data_latitude), 3))
        axe.set_yticklabels(labels=data_latitude[::3])
        axe.set_xticks(ticks=ndx)
        axe.set_xticklabels(labels=axis_ls)
        axe.set_facecolor('white')

    ax[0, 0].set_title(title)
    ax[0, 1].set_title('Corresponding altitude')
    ax[0, 1].set_ylabel('Day time: 6h - 18h', rotation='vertical')
    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].set_ylabel('Night time: 18h - 6h', rotation='vertical')
    ax[1, 1].yaxis.set_label_position("right")

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.15, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    plt.savefig(savename, bbox_inches='tight')
    plt.show()


def display_temperature_profile_day_night(data_day, data_night, T_sat, data_altitude, temperature_stats_day,
                                          temperature_stats_night):
    plt.figure(figsize=(8, 11))
    plt.plot(data_day, data_altitude / 1e3, color='red', label='diagfi day <16h>')
    plt.plot(data_night, data_altitude / 1e3, color='blue', label='diagfi night <2h>')
    plt.plot(temperature_stats_day, data_altitude / 1e3, linestyle='--', color='red', label='stats day 16h')
    plt.plot(temperature_stats_night, data_altitude / 1e3, linestyle='--', color='blue', label='stats night 2h')
    plt.plot(T_sat, data_altitude / 1e3, linestyle='--', color='black', label='CO2 condensation')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (km)')
    plt.legend(loc='best')
    # plt.ylim(40, data_altitude[-1]/1e3)
    # plt.xlim(80, 220)
    plt.savefig('temperature_day_night_equator_ls_0-30.png', bbox_inches='tight')
    plt.show()


def display_altitude_localtime(data_target, data_altitude, title, unit, savename):
    from numpy import arange, zeros, round
    from matplotlib.colors import DivergingNorm

    fig = plt.figure(figsize=(8, 11))
    if unit == '':
        scale = zeros(5)
        scale[1] = 1
        scale[2:] = 10 ** arange(1, 4)
        pc = plt.contourf(data_target.T, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=1000), levels=scale,
                          cmap='seismic', extend='max')
    else:
        pc = plt.contourf(data_target.T)
    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit)
    plt.title(title)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Local time (h)')
    plt.xticks(ticks=arange(data_target.shape[0]), labels=arange(12) * 2)
    plt.yticks(ticks=arange(data_target.shape[1]), labels=round(data_altitude[:] / 1e3, 2))
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_altitude_longitude(data_target, data_altitude, data_longitude, unit, title, savename):
    from numpy import zeros, arange, round, int_, searchsorted
    from matplotlib.colors import DivergingNorm

    fig = plt.figure(figsize=(8, 11))
    if unit == '':
        scale = zeros(5)
        scale[1] = 1
        scale[2:] = 10 ** arange(1, 4)
        pc = plt.contourf(data_target, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=1000), levels=scale,
                          cmap='seismic', extend='max')
    else:
        pc = plt.contourf(data_target)

    cbar = fig.colorbar(pc, orientation="vertical")
    cbar.ax.set_title(unit)
    ndx = searchsorted(data_longitude[:], [-180, -90, 0, 90, 180])
    plt.title(title)
    plt.xlabel('Longituge (°E)')
    plt.xticks(ticks=ndx, labels=[-180, -90, 0, 90, 180])
    plt.ylabel('Altitude (km)')
    plt.yticks(ticks=arange(data_target.shape[0]), labels=round(data_altitude[:] / 1e3, 2))
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_thickness_co2ice_atm_layer(data, data_std, savename):
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

    #    ax[0].text(0.05, 0.9, 'North pole above 60°N', transform=ax[0].transAxes, fontsize=14)
    ax[0].set_title('North pole above 60°N')
    print(data[1, :])
    ax[0].errorbar(arange(73) * 5, data[1, :] / 1e3,
                   yerr=data_std[1, :] / 1e3,
                   ls=' ', marker='+', color='black', label='GCM')  # 72 points binned in 5°
    ax[0].errorbar(northpole_ls, northpole[:, 1],
                   yerr=[northpole[:, 2] - northpole[:, 1], northpole[:, 1] - northpole[:, 0]], color='blue', ls=' ',
                   marker='+', label='MCS MY28')
    ax[0].set_xticks(ticks=arange(0, 405, 45))
    ax[0].set_xticklabels(labels=arange(0, 405, 45))
    ax[0].legend(loc='best')

    #    ax[1].text(0.6, 0.9, 'South pole above 60°S', transform=ax[1].transAxes, fontsize=14)
    ax[1].set_title('South pole above 60°S')
    ax[1].errorbar(arange(73) * 5, data[0, :] / 1e3,
                   yerr=data_std[0, :] / 1e3,
                   ls=' ', marker='+', color='black', label='GCM')
    ax[1].errorbar(southpole_ls, southpole[:, 1],
                   yerr=[southpole[:, 2] - southpole[:, 1], southpole[:, 1] - southpole[:, 0]], color='blue', ls=' ',
                   marker='+', label='MCS MY29')

    ax[1].set_xticks(ticks=arange(0, 405, 45))
    ax[1].set_xticklabels(labels=arange(0, 405, 45))
    ax[1].legend(loc='best')

    fig.text(0.06, 0.5, 'Thickness (km)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    plt.savefig(savename, bbox_inches='tight')
    plt.show()


def display_distribution_altitude_latitude_polar(distribution_north, distribution_south, data_altitude,
                                                 north_latitude, south_latitude, savename):
    from numpy import arange, round

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))

    pc = ax[0].imshow(distribution_north.T, origin='lower', aspect='auto')
    ax[0].set_xticks(ticks=arange(north_latitude.shape[0]))
    ax[0].set_xticklabels(labels=north_latitude)
    ax[0].set_yticks(ticks=arange(data_altitude[:].shape[0]))
    ax[0].set_yticklabels(labels=round(data_altitude[:] / 1e3, 2))

    ax[1].imshow(distribution_south.T, origin='lower', aspect='auto')
    ax[1].set_xticks(ticks=arange(north_latitude.shape[0]))
    ax[1].set_xticklabels(labels=south_latitude)
    ax[1].set_yticks(ticks=arange(data_altitude[:].shape[0]))
    ax[1].set_yticklabels(labels=round(data_altitude[:] / 1e3, 2))

    ax[0].set_ylim(0, 15)
    ax[1].set_ylim(0, 15)

    plt.draw()
    p0 = ax[0].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 1, p0[2] - p0[0], 0.05])
    cbar = plt.colorbar(pc, cax=ax_cbar, orientation='horizontal')
    cbar.ax.set_title('count')

    fig.text(0.02, 0.5, 'Altitude (km)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=14)

    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


def display_cloud_evolution_latitude(data, data_satuco2, data_temp, data_riceco2, idx_max, xtime,
                                     data_time, data_altitude, data_latitude):
    from numpy import arange, round, zeros, logspace, concatenate, array, max
    from matplotlib.colors import DivergingNorm, LogNorm

    filenames = []
    for i in range(-3, 3):
        filenames.append(displays.display_cloud_evolution_latitude(data_target, data_satuco2,
                                                                   data_temp, data_riceco2, idx_max, i,
                                                                   data_time[idx_max[0] + i], data_altitude,
                                                                   data_latitude))

    make_gif = input('Do you want create a gif (Y/n)?: ')
    if make_gif.lower() == 'y':
        libf.create_gif(filenames)

    if idx_max[2] - 3 < 0:
        data = data[xtime, :, 0:idx_max[2] + 3]
        data_satuco2 = data_satuco2[xtime, :, 0:idx_max[2] + 3]
        data_temp = data_temp[xtime, :, 0:idx_max[2] + 3]
        data_riceco2 = data_riceco2[xtime, :, 0:idx_max[2] + 3]
        data_latitude = data_latitude[0:idx_max[2] + 3]
    elif idx_max[2] + 3 > data_latitude.shape[0]:
        data = data[xtime, :, idx_max[2] - 3:-1]
        data_satuco2 = data_satuco2[xtime, :, idx_max[2] - 3:-1]
        data_temp = data_temp[xtime, :, idx_max[2] - 3:-1]
        data_riceco2 = data_riceco2[xtime, :, idx_max[2] - 3:-1]
        data_latitude = data_latitude[idx_max[2] - 3:-1]
    else:
        dlatitude = 5
        data = data[xtime, :, idx_max[2] - dlatitude:idx_max[2] + dlatitude + 1]
        data_satuco2 = data_satuco2[xtime, :, idx_max[2] - dlatitude:idx_max[2] + dlatitude + 1]
        data_temp = data_temp[xtime, :, idx_max[2] - dlatitude:idx_max[2] + dlatitude + 1]
        data_riceco2 = data_riceco2[xtime, :, idx_max[2] - dlatitude:idx_max[2] + dlatitude + 1]
        data_latitude = data_latitude[idx_max[2] - dlatitude:idx_max[2] + dlatitude + 1]

    ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 11))
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle('Sols: ' + str(int(data_time)) + ', local time: ' + str(int(round(data_time * 24 % 24, 0))) + ' h')
    ax[0, 0].title.set_text('CO$_2$ ice mmr')
    pc0 = ax[0, 0].contourf(data, norm=LogNorm(vmin=1e-12, vmax=1e-4), levels=logspace(-12, -3, 10), cmap='Greys')
    cbar0 = plt.colorbar(pc0, ax=ax[0, 0])
    cbar0.ax.set_title('kg/kg')
    cbar0.ax.set_yticklabels(["{:.2e}".format(i) for i in cbar0.get_ticks()])
    ax[0, 0].set_xticks(ticks=arange(0, data.shape[1], 2))
    ax[0, 0].set_xticklabels(labels=round(data_latitude[::2], 2))
    ax[0, 0].set_yticks(ticks=ticks_altitude)
    ax[0, 0].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[0, 1].title.set_text('Temperature')
    pc1 = ax[0, 1].contourf(data_temp, vmin=80, vmax=240, levels=arange(80, 260, 20), cmap='plasma')
    cbar1 = plt.colorbar(pc1, ax=ax[0, 1])
    cbar1.ax.set_title('K')
    ax[0, 1].set_xticks(ticks=arange(0, data.shape[1], 2))
    ax[0, 1].set_xticklabels(labels=round(data_latitude[::2], 2))
    ax[0, 1].set_yticks(ticks=ticks_altitude)
    ax[0, 1].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[1, 0].title.set_text('Saturation of CO$_2$ ice')
    print(max(data_satuco2))
    pc2 = ax[1, 0].contourf(data_satuco2, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=17),
                            levels=concatenate([array([0, 1]), arange(3, 19, 2)]), cmap='seismic')
    cbar2 = plt.colorbar(pc2, ax=ax[1, 0])
    cbar2.ax.set_title('')
    ax[1, 0].set_xticks(ticks=arange(0, data.shape[1], 2))
    ax[1, 0].set_xticklabels(labels=round(data_latitude[::2], 2))
    ax[1, 0].set_yticks(ticks=ticks_altitude)
    ax[1, 0].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    ax[1, 1].title.set_text('Radius of CO$_2$ ice particle')
    pc3 = ax[1, 1].contourf(data_riceco2 * 1e6, vmin=0, vmax=200, levels=arange(0, 220, 20), cmap='Greys')
    cbar3 = plt.colorbar(pc3, ax=ax[1, 1])
    cbar3.ax.set_title('µm')
    ax[1, 1].set_xticks(ticks=arange(0, data.shape[1], 2))
    ax[1, 1].set_xticklabels(labels=round(data_latitude[::2], 2))
    ax[1, 1].set_yticks(ticks=ticks_altitude)
    ax[1, 1].set_yticklabels(labels=round(data_altitude[ticks_altitude] / 1e3, 0))

    fig.text(0.02, 0.5, 'Altitude (km)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Latitude (°N)', ha='center', va='center', fontsize=14)

    savename = 'cloud_evolution_latitude_sols_' + str(int(data_time)) + '_' + str(
        round(data_time * 24 % 24, 0)) + 'h.png'
    plt.savefig(savename, bbox_inches='tight')
    plt.close()

    return savename


def display_satuco2_view_mode7(filename, data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north,
                               data_co2ice_eq, data_co2ice_south, latitude_north, latitude_eq, latitude_south, binned):
    from numpy import array, round, ones, zeros, logspace, max, argmax

    data_latitude = getdata(filename, target='latitude')
    list_latitudes = [latitude_north, latitude_eq, latitude_south]

    # chopper l'intervalle de latitude comprise entre value-1 et value+1
    data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS = mesoclouds_observed()

    list_obs = [data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
    data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS]

    data_altitude = getdata(filename, target='altitude')
    ticks_altitude = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    if data_altitude.units == 'm':
        altitude_unit = 'km'
        altitude_name = 'Altitude'
        data_altitude = data_altitude[:] / 1e3
    elif data_altitude.units == 'km':
        altitude_unit = data_altitude.units
        altitude_name = 'Altitude'
    else:
        altitude_unit = data_altitude.units
        altitude_name = 'Pressure'
        try:
            data_zareoid = getdata(filename, target='zareoid')
        except:
            print('Zareoid is missing!')
            exit()

    data_time = getdata(filename=filename, target='Time')
    if binned.lower() == 'y':
        data_time = data_time[::60] # 5°Ls binned
        data_zareoid = data_zareoid[::12,:,:,:]
    ndx, axis_ls = get_ls_index(data_time)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 11))
    ax[0].set_title('{}°N'.format(latitude_north))
    ax[0].contourf(data_time[:], data_altitude[:], data_satuco2_north,
                   norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=100), cmap='coolwarm',
                   levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[0].contour(data_time[:], data_altitude[:], data_co2ice_north, norm=LogNorm(), levels=logspace(-13, 1, 15),
    colors='black')

    ax[1].set_title('{}°N'.format(latitude_eq))
    ax[1].contourf(data_time[:], data_altitude[:], data_satuco2_eq,
                   norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=100), cmap='coolwarm',
                   levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[1].contour(data_time[:], data_altitude[:], data_co2ice_eq, norm=LogNorm(), levels=logspace(-13, 1, 15),
                  colors='black')

    ax[2].set_title('{}°S'.format(abs(latitude_south)))
    cb = ax[2].contourf(data_time[:], data_altitude[:], data_satuco2_south,
                        norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=100), cmap='coolwarm',
                        levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[2].contour(data_time[:], data_altitude[:], data_co2ice_south, norm=LogNorm(), levels=logspace(-13, 1, 15),
                  colors='black')

    for i, axe in enumerate(ax):
        axe.set_xticks(ticks=axis_ls)
        axe.set_xticklabels(labels=axis_ls)
        axe.set_xlim(0, 360)
        axe.set_ylim(1e-3, 1e3)

        for j, value in enumerate(list_obs):
            data_obs_ls, data_obs_latitude = get_nearest_clouds_observed(value, 'latitude', data_latitude,
                                                                         list_latitudes[i])
            if data_obs_ls.shape[0] != 0:
                axe.quiver(data_obs_ls, ones(data_obs_ls.shape[0])*1e-3, zeros(data_obs_ls.shape[0]),
                           -ones(data_obs_ls.shape[0])*3, color='black')

        if altitude_unit == 'Pa':
            data_zareoid_sliced, tmp = slice_data(data_zareoid, dimension_data=data_latitude, value=list_latitudes[i])

            lines_altitudes_0km = get_mean_index_alti(data_zareoid_sliced, 0)
            lines_altitudes_10km = get_mean_index_alti(data_zareoid_sliced, 1e4)
            lines_altitudes_40km = get_mean_index_alti(data_zareoid_sliced, 4e4)
            lines_altitudes_80km = get_mean_index_alti(data_zareoid_sliced, 8e4)
            del data_zareoid_sliced

            axe.plot(data_altitude[lines_altitudes_0km], '--', color='grey')
            axe.plot(data_altitude[lines_altitudes_10km], '--', color='grey')
            axe.plot(data_altitude[lines_altitudes_40km], '--', color='grey')
            axe.plot(data_altitude[lines_altitudes_80km], '--', color='grey')

            axe.text(0, data_altitude[lines_altitudes_0km[0]], '0 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=10)
            axe.text(0, data_altitude[lines_altitudes_10km[0]], '10 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=10)
            axe.text(0, data_altitude[lines_altitudes_40km[0]], '40 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=10)
            axe.text(0, data_altitude[lines_altitudes_80km[0]], '80 km',
                     verticalalignment='bottom',
                     horizontalalignment='left', color='grey', fontsize=10)

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

    if binned.lower() == 'y':
        plt.savefig('satuco2_with_co2ice_altitude-ls_max_along_longitude_at_{}N_{}N_{}N_binned.png'.format(
            latitude_north, latitude_eq, latitude_south), bbox_inches='tight')
    else:
        plt.savefig('satuco2_with_co2ice_altitude-ls_max_along_longitude_at_{}N_{}N_{}N.png'.format(latitude_north,
                    latitude_eq, latitude_south), bbox_inches='tight')
    plt.show()


def saturation_zonal_mean_day_night(data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night,
                                    data_altitude, ndx, axis_ls, title, savename):
    from numpy import array, round

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 11))
    fig.suptitle(title)
    ax[0].set_title('Day 6h - 18h')
    ax[0].contourf(data_satuco2_day.T, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=100), cmap='seismic',
                   levels=array([0, 1, 10, 20, 50, 100]), extend='max')
    ax[0].contour(data_co2ice_day.T, colors='black')

    ax[1].set_title('Night 18h - 6h')
    cb = ax[1].contourf(data_satuco2_night.T, norm=DivergingNorm(vmin=0, vcenter=1.0, vmax=100), cmap='seismic',
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


def topcloud_altitude(top_cloud, data_latitude, data_altitude, ndx, axis_ls):
    from numpy import arange, round, where, flip, concatenate, array

    for i in range(top_cloud.shape[0]):
        dim2 = where(top_cloud[i, :] == 0)
        top_cloud[i, dim2] = None

    plt.figure(figsize=(8, 11))
    plt.title('Zonal mean of top cloud altitude')
    cb = plt.contourf(flip(top_cloud.T, axis=0), norm=DivergingNorm(vmin=data_altitude[1], vcenter=40, vmax=50),
                      levels=concatenate([array([data_altitude[1]]), arange(1, 6) * 10]), cmap='seismic')
    ax = plt.gca()
    ax.set_facecolor('white')

    cbar = plt.colorbar(cb)
    cbar.ax.set_title('km')

    plt.xticks(ticks=ndx, labels=axis_ls)
    plt.yticks(ticks=arange(0, data_latitude.shape[0], 2), labels=round(data_latitude[::2], 2))
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    plt.savefig('topcloud_altitude_comparable_to_mola.png', bbox_inches='tight')
    plt.show()


# figure in altitude - X
def display_1fig_profiles(filename, data, latitude_selected, xmin, xmax, xlabel, xscale='linear', yscale='linear',
                          title='', savename='profiles'):
    data_altitude = getdata(filename, target='altitude')
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

    for i, s in zip(data, savename):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 11))
        if i.ndim > 1:
            for j in range(i.shape[1]):
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.plot(i[:, j], data_altitude[:], label='%.2f°N' % (latitude_selected[j]))
        else:
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.plot()

        plt.xlim(xmin, xmax)
        plt.ylim(data_altitude[0], data_altitude[-1])
        if altitude_name == 'Pressure':
            ax.invert_yaxis()
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(altitude_name + ' (' + units + ')')
        plt.title(title)
        plt.savefig(s + '.png', bbox_inches='tight')
        plt.close(fig)

    create_gif(savename)


def display_alt_ls(filename, data_1, data_2, levels, title, savename, latitude_selected=None):
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

    axes.set_xlim(data_time[0], data_time[-1]*12)
    axes.set_xticks(ticks=ndx)
    axes.set_xticklabels(labels=axis_ls)
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

        axes.text(data_time[-1]*12 + 1, data_altitude[lines_altitudes_0km[0]], '0 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1]*12 + 1, data_altitude[lines_altitudes_10km[0]], '10 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1]*12 + 1, data_altitude[lines_altitudes_40km[0]], '40 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)
        axes.text(data_time[-1]*12 + 1, data_altitude[lines_altitudes_80km[0]], '80 km', verticalalignment='bottom',
                  horizontalalignment='right', color='grey', fontsize=14)

        axes.set_yscale('log')
        axes.invert_yaxis()
    else:
        axes.set_yticklabels(labels=round(data_altitude[ticks_altitude], 0))

    axes.set_title(title)
    fig.savefig(savename + '.png', bbox_inches='tight')
    fig.show()


def display_alt_lat(filename, data):
    data_altitude = getdata(filename, target='altitude')
    data_latitude = getdata(filename, target='latitude')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,11))
    print(data_altitude.shape, data_latitude.shape, data.shape)
    ctf = ax.contourf(data_latitude[:], data_altitude[:], data, cmap='coolwarm')
    cbar = plt.colorbar(ctf)
    cbar.ax.set_title('K')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlabel('Latitude (°N)')
    ax.set_title('T - Tcondco2, 12h - 00h, between 0-30° Ls, zonal mean')
    fig.savefig('deltaT_thermal_tides_0-30Ls_zonalmean.png', bbox_inches='tight')
    plt.show()


def display_4figs_polar_projection(data_riceco2, data_co2ice, data_temp, data_satuco2):

    from mpl_toolkits.basemap import Basemap

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='polar')
    axes.contourf(data_riceco2[0, 0, :, :])
    plt.show()
