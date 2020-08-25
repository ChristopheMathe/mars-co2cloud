import matplotlib.pyplot as plt
from .lib_function import *
import matplotlib as mpl

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
# ================================================== DISPLAY 3D ====================================================== #
# ==================================================================================================================== #
def display_3D(data):
    print('To be build')


# ==================================================================================================================== #
# ================================================== DISPLAY Hovmöller diagram ======================================= #
# ==================================================================================================================== #
def hovmoller_diagram(data, data_latitude, data_altitude, data_longitude, data_time):
    from numpy import arange
    from scipy.interpolate import interp2d

    idx_lat = (abs(data_latitude[:] - 80)).argmin()
    idx_alt = (abs(data_altitude[:] - 25)).argmin()

    shape_data = data.shape
    zoom_data = data[:, idx_alt, idx_lat, :]

    zoom_data = correction_value_co2_ice(zoom_data)
    interp_time, axis_ls, ndx = linear_grid_ls(data_time)

    zoom_data = linearize_ls(zoom_data, shape_data[0], shape_data[3], interp_time)

    mask = (axis_ls >= 255) & (axis_ls <= 315)

    plt.figure(figsize=(11, 8))
    plt.title(
        u'Hovmöller diagram of ' + data.title + ' at ' + str(int(data_latitude[idx_lat])) + '°N, altitude %3.2e km' % (
            data_altitude[idx_alt]))
    plt.contourf(zoom_data[:, mask], cmap='inferno')
    plt.yticks(ticks=arange(0, len(data_longitude), 8), labels=data_longitude[::8])
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Longitude (°W)')
    plt.xlim(axis_ls[mask][0], axis_ls[mask][-1])
    cbar = plt.colorbar()
    cbar.ax.set_title('kg/kg')
    plt.savefig('hovmoller_diagram_co2_at_' + str(int(data_latitude[idx_lat])) + 'N_altitude %3.2e km.png' % (
        data_altitude[idx_alt]), bbox_inches='tight')
    plt.show()


# ==================================================================================================================== #
# =========================================== DISPLAY Column density ================================================= #
# ==================================================================================================================== #
def display_colonne(data, data_altitude, data_latitude, ndx, axis_ls, interp_time, unit):
    from matplotlib.colors import LogNorm
    from numpy import sum, mean, arange, int_, logspace, flip

    altitude_limit = input('Do you want perform the computation on the entire column(Y/n)? ')
    if altitude_limit.lower() == 'n':
        print('Altitude range (km): {:.3f} {:.3f}'.format(data_altitude[0], data_altitude[-1]))
        z_min = float(input('Start altitude (km): '))
        z_max = float(input('End altitude (km): '))
        idx_z_min = (abs(data_altitude[:] - z_min)).argmin()
        idx_z_max = (abs(data_altitude[:] - z_max)).argmin()
        data = data[:, idx_z_min:idx_z_max, :, :]

    shape_data = data.shape

    # compute zonal mean column density
    zonal_mean_column_density = sum(mean(data, axis=3), axis=1)  # Ls function of lat

    zonal_mean_column_density = flip(zonal_mean_column_density.T, axis=0)
    print(zonal_mean_column_density.shape)
    zonal_mean_column_density = linearize_ls(zonal_mean_column_density, shape_data[0], shape_data[2], interp_time)
    if unit is 'pr.µm':
        zonal_mean_column_density = zonal_mean_column_density * 1e3  # convert kg/m2 to pr.µm

    # plot
    plt.figure(figsize=(11, 8))
    plt.contourf(zonal_mean_column_density, norm=LogNorm(), levels=logspace(-10, 1, 12), cmap='seismic')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.yticks(ticks=arange(0, len(data_latitude), 6), labels=data_latitude[::6])
    plt.xticks(ticks=ndx, labels=int_(axis_ls[ndx]))
    cbar = plt.colorbar()
    cbar.ax.set_title(unit)
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')

    if altitude_limit.lower() == 'y':
        plt.title('Zonal mean column density of ' + data.title)
        plt.savefig('zonal_mean_density_column_' + data.title + '.png', bbox_inches='tight')
    else:
        plt.title(
            'Zonal mean column density of ' + data.title + ' between ' + str(z_min) + ' and ' + str(z_max) + ' km')
        plt.savefig('zonal_mean_density_column_' + data.title + '_' + str(z_min) + '_' + str(z_max) + '.png',
                    bbox_inches='tight')

    plt.show()


# =====================================================================================================================#
# ============================================ DISPLAY latitude-altitude ==============================================#
# =====================================================================================================================#
def display_altitude_latitude(data, data_time, data_altitude, data_latitude, unit):
    from numpy import mean, arange, where, searchsorted
    from scipy.interpolate import interp2d

    # ---------------------------------------------------------
    # South pole:
    # -----------
    idx_lat_1 = (abs(data_latitude[:] + 57)).argmin() + 1
    idx_lat_2 = (abs(data_latitude[:] + 90)).argmin() + 1
    idx_ls_1 = (abs(data_time[:] - 0)).argmin()
    idx_ls_2 = (abs(data_time[:] - 180)).argmin()
    pole = 'south_pole'
    # ---------------------------------------------------------
    # North pole:
    # -----------
    # idx_lat_1 = (abs(data_latitude[:] - 90)).argmin()
    # idx_lat_2 = (abs(data_latitude[:] - 57)).argmin()
    # idx_ls_1 = (abs(data_time[:] - 180)).argmin()
    # idx_ls_2 = (abs(data_time[:] - 360)).argmin()
    # pole = 'north_pole'
    # ---------------------------------------------------------

    idx_alt = (abs(data_altitude[:] - 20)).argmin() + 1

    zoomed_data = data[idx_ls_1:idx_ls_2, :idx_alt, idx_lat_1:idx_lat_2, :]

    # zoom data => [alt, lat]
    zoomed_data = mean(mean(zoomed_data[:, :, :, :], axis=3), axis=0)  # zonal mean, then temporal mean
    rows, cols = where(zoomed_data < 1e-10)
    zoomed_data[rows, cols] = 1e-10

    zoomed_data = zoomed_data[:, ::-1]

    f = interp2d(x=arange(len(data_latitude[idx_lat_1:idx_lat_2])), y=arange(len(data_altitude[:idx_alt])),
                 z=zoomed_data, kind='cubic')

    axis_altitude = arange(0, 21)  # , len(data_altitude[:idx_alt]))
    interp_altitude = searchsorted(data_altitude[:idx_alt], axis_altitude)
    zoomed_data = f(arange(len(data_latitude[idx_lat_1:idx_lat_2])), interp_altitude)

    if unit is 'pr.µm':
        zoomed_data = zoomed_data * 1e3  # convert kg/m2 to pr.µm

    fmt = lambda x: "{:1d}".format(int(x))

    # plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('CO$_2$ ice mass mixing ratio')
    plt.contourf(zoomed_data, cmap='inferno')
    plt.xticks(ticks=arange(0, len(data_latitude[idx_lat_1:idx_lat_2])),
               labels=data_latitude[idx_lat_1:idx_lat_2][::-1])
    plt.yticks(ticks=arange(0, len(interp_altitude)), labels=[fmt(round(i)) for i in axis_altitude[:]])
    cbar = plt.colorbar()
    cbar.ax.set_title(unit)
    plt.xlabel('Latitude (°N)')
    plt.ylabel('Altitude (km)')
    plt.savefig('altitude_latitude_' + data.title + '_' + pole + '.png', bbox_inches='tight')
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
    pc = ax[0].contourf(max_mmr, norm=LogNorm(), levels=logspace(-10, 1, 12), cmap='seismic')
    ax[0].set_facecolor('black')
    ax[0].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[0].set_yticklabels(labels=data_latitude[::6])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_xticklabels(labels='')
    cbar = plt.colorbar(pc, ax=ax[0])
    cbar.ax.set_title(unit)
    ax[0].set_ylabel('Latitude (°N)')

    # plot 2
    ax[1].set_title('Altitude at co2_ice mmr max')
    pc2 = ax[1].contourf(max_alt, cmap='seismic')
    ax[1].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[1].set_yticklabels(labels=data_latitude[::6])
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels='')
    ax[1].set_ylabel('Latitude (°N)')
    cbar2 = plt.colorbar(pc2, ax=ax[1])
    cbar2.ax.set_title('km')

    # plot 3
    ax[2].set_title('Temperature at co2_ice mmr max')
    pc3 = ax[2].contourf(max_temp, cmap='seismic')
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
    pc4 = ax[3].contourf(max_satu, cmap='seismic', norm=divnorm, levels=arange(0, 5))
    ax[3].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[3].set_yticklabels(labels=data_latitude[::6])
    ax[3].set_xticks(ticks=ndx)
    ax[3].set_xticklabels(labels='')
    ax[3].set_ylabel('Latitude (°N)')
    cbar4 = plt.colorbar(pc4, ax=ax[3])
    cbar4.ax.set_title(' ')

    # plot 5
    ax[4].set_title('Radius of co2_ice at co2_ice mmr max')
    pc5 = ax[4].contourf(max_radius * 1e3, cmap='seismic')
    ax[4].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[4].set_yticklabels(labels=data_latitude[::6])
    ax[4].set_xticks(ticks=ndx)
    ax[4].set_xticklabels(labels='')
    ax[4].set_ylabel('Latitude (°N)')
    cbar5 = plt.colorbar(pc5, ax=ax[4])
    cbar5.ax.set_title(u'µm')

    # plot 6
    ax[5].set_title('CCN number at co2_ice mmr max')
    pc3 = ax[5].contourf(max_ccnN, norm=DivergingNorm(vmin=0, vcenter=1), levels=arange(0, 5), cmap='seismic')
    ax[5].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[5].set_yticklabels(labels=data_latitude[::6])
    ax[5].set_xticks(ticks=ndx)
    ax[5].set_xticklabels(labels=int_(axis_ls[ndx]))
    ax[5].set_xlabel('Solar Longitude (°)')
    ax[5].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[5])
    cbar3.ax.set_title('nb/kg')

    fig.savefig('max_' + name + '_in_altitude_longitude.png', bbox_inches='tight')

    plt.show()


# ==================================================================================================================== #
# ============================================ DISPLAY Equateur profile ============================================== #
# ==================================================================================================================== #
def display_equa_profile(data, data_time, data_latitude, data_altitude, data_temperature, data_saturation, unit):
    from numpy import abs, arange, linspace, searchsorted, int_, logspace, asarray, reshape, where, \
        unravel_index
    from numpy.ma import masked_where
    from matplotlib.colors import LogNorm, DivergingNorm
    from scipy.interpolate import interp2d

    idx_eq = (abs(data_latitude[:] - 0)).argmin()
    data = data[:, :, idx_eq, :]
    data_temperature = data_temperature[:, :, idx_eq, :]
    data_saturation = data_saturation[:, :, idx_eq, :]

    dim1 = data.shape[0]
    dim2 = data.shape[1]

    # correct very low values of co2 mmr
    data = masked_where(data < 1e-10, data)
    value_1, value_2, value_3 = where(data < 1e-10)
    data[value_1, value_2, value_3] = 0

    # find the co2_ice max value along longitude
    idx_max = data.reshape((data.shape[0], -1)).argmax(axis=1)
    idx_alt, idx_lon = unravel_index(idx_max, (data.shape[1], data.shape[2]))
    B = [data[i, :, idx_lon[i]] for i in range(dim1)]
    B = asarray(B)
    data = reshape(B, (dim1, dim2))

    B = [data_temperature[i, :, idx_lon[i]] for i in range(dim1)]
    B = asarray(B)
    data_temperature = reshape(B, (dim1, dim2))

    B = [data_saturation[i, :, idx_lon[i]] for i in range(dim1)]
    B = asarray(B)
    data_saturation = reshape(B, (dim1, dim2))

    data = data.T
    data_temperature = data_temperature.T
    data_saturation = data_saturation.T

    f = interp2d(x=arange(len(data_time)), y=arange(len(data_altitude)), z=data, kind='linear')
    f2 = interp2d(x=arange(len(data_time)), y=arange(len(data_altitude)), z=data_temperature, kind='linear')
    f3 = interp2d(x=arange(len(data_time)), y=arange(len(data_altitude)), z=data_saturation, kind='linear')

    axis_ls = linspace(0, 360, len(data_time[:]))
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data_time, axis_ls)

    data = f(interp_time, arange(len(data_altitude)))
    data_temperature = f2(interp_time, arange(len(data_altitude)))
    data_saturation = f3(interp_time, arange(len(data_altitude)))

    # plot 1: co2_ice mmr
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11, 20))
    ax[0].set_title('co2_ice mmr max longitudinal profile at ' + str(data_latitude[idx_eq]) + '°N')
    pc = ax[0].contourf(data, norm=LogNorm(), vmin=1e-10, levels=logspace(-10, 1, 12), cmap='seismic')
    ax[0].set_facecolor('black')
    ax[0].set_yticks(ticks=arange(0, len(data_altitude), 3))
    ax[0].set_yticklabels(labels=data_altitude[::3])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_xticklabels(labels='')
    ax[0].set_ylabel('Altitude (km)')
    cbar = plt.colorbar(pc, ax=ax[0])
    cbar.ax.set_title(unit)

    # plot 2: temperature
    ax[1].set_title('Temperature')
    pc2 = ax[1].contourf(data_temperature, cmap='seismic')
    ax[1].set_facecolor('black')
    ax[1].set_yticks(ticks=arange(0, len(data_altitude), 3))
    ax[1].set_yticklabels(labels=data_altitude[::3])
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels='')
    ax[1].set_ylabel('Altitude (km)')
    cbar2 = plt.colorbar(pc2, ax=ax[1])
    cbar2.ax.set_title('K')

    # plot 3: saturation rate
    ax[2].set_title('Saturation')
    pc3 = ax[2].contourf(data_saturation, norm=DivergingNorm(vmin=0, vcenter=1, vmax=4), levels=arange(0, 5),
                         cmap='seismic')
    ax[2].set_facecolor('black')
    ax[2].set_yticks(ticks=arange(0, len(data_altitude), 3))
    ax[2].set_yticklabels(labels=data_altitude[::3])
    ax[2].set_xticks(ticks=ndx)
    ax[2].set_xticklabels(labels=int_(axis_ls[ndx]))
    ax[2].set_xlabel('Solar Longitude (°)')
    ax[2].set_ylabel('Altitude (km)')
    plt.colorbar(pc3, ax=ax[2])

    plt.savefig('co2_ice_profile_' + str(data_latitude[idx_eq]) + 'N.png', bbox_inches='tight')
    plt.show()


# =====================================================================================================================#
# ============================================ DISPLAY Zonal mean =====================================================#
# =====================================================================================================================#
def display_zonal_mean(data):
    from numpy import sum

    print(data)
    zonal_mean = sum(data[:, :, :], axis=2)
    zonal_mean = zonal_mean.T

    plt.figure()
    plt.title('Zonal mean of ' + data.title)
    plt.contourf(zonal_mean)
    cbar = plt.colorbar()
    cbar.ax.set_title(data.units)
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    plt.savefig('zonal_mean_' + data.title + '.png', bbox_inches='tight')
    plt.show()


# ==================================================================================================================== #
# ============================================ DISPLAY Zonal mean ==================================================== #
# ==================================================================================================================== #
def display_vertical_profile(data, data_latitude, data_pressure):
    from numpy import argmin, mean, abs, min, log10, asarray, reshape

    idx_lat = (abs(data_latitude[:] - 15)).argmin() + 1
    idx_lat2 = (abs(data_latitude[:] + 15)).argmin()
    data = data[:, :, idx_lat2:idx_lat, :]

    idx_lon = data[:, :, :, :].argmin(axis=3)
    dim = data.shape
    data = [data[i, j, k, idx_lon[i, j, k]] for i in range(dim[0]) for j in range(dim[1]) for k in range(dim[2])]
    data = asarray(data)
    data = reshape(data, (dim[0], dim[1], dim[2]))

    idx_ls = data[:, :, :].argmin(axis=0)
    data = [data[idx_ls[i, j], i, j] for i in range(dim[1]) for j in range(dim[2])]
    data = asarray(data)
    data = reshape(data, (dim[1], dim[2]))

    A = 6.81228
    B = 1301.679
    C = -3.494
    T_sat = B / (A - log10((data_pressure[0, :, idx_lat, 0] + 0.0001) / 10 ** 5)) - C  # +0.0001 to avoid log10(0)

    plt.figure(figsize=(11, 8))
    plt.title('Vertical profile of temperature minimal between 15°N and 15°S, along longitude and time')
    print(data.shape)
    for i in range(data.shape[1]):
        plt.semilogy(data[:, i], data_pressure[0, :, idx_lat2 + i, 0], label=data_latitude[idx_lat2 + i])

    plt.semilogy(T_sat, data_pressure[0, :, idx_lat, 0], '--', color='red')
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


def display_riceco2_profile_pole(data, data_latitude, data_time, data_pressure):
    from numpy import argmin, max

    idx_lat = (abs(data_latitude[:] - 15)).argmin()
    idx_lat2 = (abs(data_latitude[:] + 15)).argmin() + 1
    #    idx_ls = (abs(data_time[:] - 180)).argmin()
    #    idx_ls2 = (abs(data_time[:] - 180)).argmin()

    data = data[:, :, idx_lat2:idx_lat, :]

    data = max(max(data, axis=3), axis=0)
    nb_lat = idx_lat - idx_lat2

    plt.figure()
    for i in range(nb_lat):
        plt.semilogy(data[:, i] * 1e6, data_pressure[0, :, idx_lat2 + i, 0],
                     label='%.2f°N' % (data_latitude[idx_lat2 + i]))
    plt.legend(loc='best')
    plt.xlabel('Radius of ice particle (µm)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Maximum radius ice particle between Ls=0°-360° in the equatorial region')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig('radius_co2_ice_particle_equator.png', bbox_inches='tight')
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
        pc1 = ax[0, 0].contourf(data_day*1e6, cmap='seismic')
        pc2 = ax[1, 0].contourf(data_night*1e6, cmap='seismic')
    elif unit == '#/kg':
#        levels = 10**arange(0,10,2)
        levels = ones(4)
        levels[1] = 1e6
        levels[2] = 1e7
        levels[3] = 1e8
#        cmap = mpl.colors.ListedColormap(['blue', 'cyan', 'orange', 'red'])
#        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
        pc1 = ax[0, 0].contourf(data_day, cmap='seismic', levels=levels, extend='max')
        pc2 = ax[1, 0].contourf(data_night, cmap='seismic', levels=levels, extend='max')
    else:
        pc1 = ax[0, 0].contourf(data_day, norm=LogNorm(), vmin=1e-10, levels=logspace(-10, 0, 11), cmap='seismic')
        pc2 = ax[1, 0].contourf(data_night, norm=LogNorm(), vmin=1e-10, levels=logspace(-10, 0, 11), cmap='seismic')

    pc3 = ax[0, 1].contourf(data_alt_day/1e3, norm=DivergingNorm(vmin=0, vcenter=40, vmax=120), levels=arange(
                            12)*10, cmap='seismic')
    pc4 = ax[1, 1].contourf(data_alt_night/1e3, norm=DivergingNorm(vmin=0, vcenter=40, vmax=120), levels=arange(
                            12)*10, cmap='seismic')
    ax[0,0].set_facecolor('black')
    ax[1,0].set_facecolor('black')
    ax[0,1].set_facecolor('black')
    ax[1,1].set_facecolor('black')
    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.2)
    pos1 = ax[0,0].get_position()
    pos2 = ax[0,1].get_position()
    cbar_ax1 = fig.add_axes([pos1.x0, 0.07, 0.38, 0.03])
    cbar_ax2 = fig.add_axes([pos2.x0, 0.07, 0.38, 0.03])
    cbar1 = fig.colorbar(pc1, cax=cbar_ax1, orientation="horizontal", )
    cbar1.ax.set_title(unit)
#    cbar1.ax.set_xticklabels(['1', '10$^2$', '10$^4$', '10$^6$', '10$^8$'])
    cbar2 = fig.colorbar(pc3, cax=cbar_ax2, orientation="horizontal")
    cbar2.ax.set_title('km')

    ax[0, 0].set_title(title)
    ax[0, 1].set_title('Corresponding altitude')
    ax[0, 1].set_ylabel('Day time: 6h - 18h', rotation='vertical')
    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].set_ylabel('Night time: 18h - 6h', rotation='vertical')
    ax[1, 1].yaxis.set_label_position("right")

    ax[0, 0].set_yticks(ticks=arange(0, len(data_latitude), 3))
    ax[0, 0].set_yticklabels(labels=data_latitude[::3])
    ax[1, 0].set_yticks(ticks=arange(0, len(data_latitude), 3))
    ax[1, 0].set_yticklabels(labels=data_latitude[::3])
    ax[1, 0].set_xticks(ticks=ndx)
    ax[1, 0].set_xticklabels(labels=int_(axis_ls[ndx]))

    fig.text(0.06, 0.5, 'Latitude (°N)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.15, 'Solar longitude (°)', ha='center', va='center', fontsize=14)

    plt.savefig(savename, bbox_inches='tight')
    plt.show()


def display_temperature_profile_day_night(data_day, data_night, T_sat, data_altitude):
    plt.figure(figsize=(8, 11))
    plt.plot(data_day, data_altitude, color='red')
    plt.plot(data_night, data_altitude, color='blue')
    plt.plot(T_sat, data_altitude, linestyle='--', color='black')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (km)')
    plt.ylim(40, data_altitude[-1])
    plt.xlim(80, 170)
    plt.savefig('temperature_day_night_equator_ls_0-30.png', bbox_inches='tight')
    plt.show()
