import matplotlib.pyplot as plt
from lib_function import *
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
def hovmoller_diagram(data, data_latitude, data_altitude):
    from numpy import arange
    from scipy.interpolate import interp2d

    idx_lat = (abs(data_latitude[:] - 80)).argmin()
    idx_alt = (abs(data_altitude[:] - 25)).argmin()
    zoom_data = data[:, idx_alt, idx_lat, :]
    zoom_data = zoom_data.T
    f2 = interp2d(x=arange(len(data_time)), y=arange(len(data_longitude)), z=zoom_data, kind='linear')
    zoom_data = f2(interp_time, arange(len(data_longitude)))
    rows, cols = where(zoom_data < 1e-10)
    zoom_data[rows, cols] = 1e-10

    mask = (axis_ls >= 255) & (axis_ls <= 315)
    plt.figure(figsize=(11, 8))
    #    plt.title(u'Hovmöller diagram of temperature at '+str(int(data_latitude[-idx_lat]))+'°N, altitude %3.2e km'%(
    plt.title(u'Hovmöller diagram of CO$_2$ at ' + str(int(data_latitude[-idx_lat])) + '°N, altitude %3.2e km' % (
        data_altitude[idx_alt]))
    plt.contourf(zoom_data[:, mask], cmap='inferno')
    plt.yticks(ticks=arange(0, len(data_longitude), 8), labels=data_longitude[::8])
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Longitude (°W)')
    plt.xlim(axis_ls[mask][0], axis_ls[mask][-1])
    cbar = plt.colorbar()
    cbar.ax.set_title('kg/m$^2$')
    plt.savefig('hovmoller_diagram_co2_at_' + str(int(data_latitude[-idx_lat])) + 'N_altitude %3.2e km.png' % (
        data_altitude[idx_alt]), bbox_inches='tight')
    plt.show()


# ==================================================================================================================== #
# =========================================== DISPLAY Column density ================================================= #
# ==================================================================================================================== #
# TODO: display => gas non-condensable, la température, distribution de colonne de co2_ice, h2o_ice, h2o_vap, tau
def display_colonne(data, data_time, data_latitude, unit):
    from matplotlib.colors import LogNorm
    from numpy import sum, mean, arange, searchsorted, int_, flip, linspace, logspace, where
    from scipy.interpolate import interp2d

    # compute zonal mean column density
    zonal_mean_column_density = sum(mean(data[:, :, :, :], axis=3), axis=1)  # Ls function of lat
    zonal_mean_column_density = zonal_mean_column_density.T  # lat function of Ls
    zonal_mean_column_density = flip(zonal_mean_column_density, axis=0)  # reverse to get North pole on top of the fig
    data_latitude = data_latitude[::-1]  # And so the labels

    # interpolation to get linear Ls
    f = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=zonal_mean_column_density, kind='linear')
    axis_ls = linspace(0, 360, len(data_time[:]))
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data_time, axis_ls)
    zonal_mean_column_density = f(interp_time, arange(len(data_latitude)))

    rows, cols = where(zonal_mean_column_density < 1e-10)
    zonal_mean_column_density[rows, cols] = 1e-10

    if unit is 'pr.µm':
        zonal_mean_column_density = zonal_mean_column_density * 1e3  # convert kg/m2 to pr.µm

    # plot
    plt.figure(figsize=(11, 8))
    plt.title('Zonal mean column density of ' + data.title)
    plt.contourf(zonal_mean_column_density, norm=LogNorm(), levels=logspace(-10, 0, 11), cmap='inferno')
    plt.yticks(ticks=arange(0, len(data_latitude), 6), labels=data_latitude[::6])
    plt.xticks(ticks=ndx, labels=int_(axis_ls[ndx]))
    cbar = plt.colorbar()
    cbar.ax.set_title(unit)
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    plt.savefig('zonal_mean_density_column_' + data.title + '.png', bbox_inches='tight')


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
    from numpy import mean, arange, where, searchsorted
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
    idx_lat_1 = (abs(data_latitude[:] - 90)).argmin()
    idx_lat_2 = (abs(data_latitude[:] - 57)).argmin()
    idx_ls_1 = (abs(data_time[:] - 180)).argmin()
    idx_ls_2 = (abs(data_time[:] - 360)).argmin()
    savename = 'altitude_latitude_saturation_north_pole'
    # ----------------------------------------------------------

    idx_alt = (abs(data_altitude[:] - 20)).argmin() + 1

    zoomed_data = data[idx_ls_1:idx_ls_2, :idx_alt, idx_lat_1:idx_lat_2, :]

    # zoom data: [ls, alt, lat, lon] => [alt, lat]
    zoomed_data = mean(mean(zoomed_data[:, :, :, :], axis=3), axis=0)  # zonal mean, then temporal mean

    # correction value below 1e-10
    rows, cols = where(zoomed_data < 1e-10)
    zoomed_data[rows, cols] = 1e-10

    zoomed_data = zoomed_data[:, ::-1]

    f = interp2d(x=arange(len(data_latitude[idx_lat_1:idx_lat_2])), y=arange(len(data_altitude[:idx_alt])),
                 z=zoomed_data, kind='linear')

    axis_altitude = arange(0, 21)
    interp_altitude = searchsorted(data_altitude[:idx_alt], axis_altitude)
    zoomed_data = f(arange(len(data_latitude[idx_lat_1:idx_lat_2])), interp_altitude)

    fmt = lambda x: "{:1d}".format(int(x))

    # plot
    divnorm = DivergingNorm(vmin=0, vcenter=1, vmax=2)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('CO$_2$ saturation')
    plt.contourf(zoomed_data, cmap='seismic', norm=divnorm, levels=arange(0, 3))
    plt.xticks(ticks=arange(0, len(data_latitude[idx_lat_1:idx_lat_2])),
               labels=data_latitude[idx_lat_1:idx_lat_2][::-1])
    plt.yticks(ticks=arange(0, len(interp_altitude)), labels=[fmt(round(i)) for i in axis_altitude[:]])
    cbar = plt.colorbar()
    cbar.ax.set_title('')
    plt.xlabel('Latitude (°)')
    plt.ylabel('Altitude (km)')
    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.show()


# =====================================================================================================================#
# ================================ DISPLAY max value in altitude / longitude ==========================================#
# =====================================================================================================================#
def display_max_lon_alt(data, data_time, data_latitude, data_altitude, data_temp, data_satu, data_riceco2, data_ccnNco2,
                        unit):
    from matplotlib.colors import LogNorm, DivergingNorm
    from numpy import arange, int_, logspace

    print('----------------------------')
    print('Enter in display_max_lon_alt')
    print('----------------------------')
    data_y = data[:, :, :, :]
    shape_data_y = data_y.shape

    # correct very low values of co2 mmr
    print('Correction value...')
    datay_y = correction_value_co2_ice(data_y)

    print('Get max co2_ice...')
    max_mmr, x, y = get_max_co2_ice_in_alt_lon(data_y)

    print('Extract other variable at co2_ice max value...')
    max_temp = extract_at_max_co2_ice(data_temp, x, y)
    max_satu = extract_at_max_co2_ice(data_satu, x, y)
    max_riceco2 = extract_at_max_co2_ice(data_riceco2, x, y)
    max_ccnNco2 = extract_at_max_co2_ice(data_ccnNco2, x, y)
    max_alt = extract_at_max_co2_ice(data_altitude, x, y, shape_data_y)

    print('Create linear grid time...')
    interp_time, axis_ls, ndx = linear_grid_ls(data_time)

    print('Reshape and linearized data...')
    max_mmr = reshape_and_linearize_data(max_mmr, data_time, data_latitude)
    max_satu = reshape_and_linearize_data(max_satu, data_time, data_latitude)
    max_riceco2 = reshape_and_linearize_data(max_riceco2, data_time, data_latitude)
    max_ccnNco2 = reshape_and_linearize_data(max_ccnNco2, data_time, data_latitude)
    max_alt = reshape_and_linearize_data(max_alt, data_time, data_latitude)

    data_latitude = data_latitude[::-1]  # And so the labels

    # PLOT
    print('Plotting...')
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(11, 30))

    # plot 1
    ax[0].set_title('Max ' + data.title + ' in altitude/longitude')
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
    pc5 = ax[4].contourf(max_radius*1e3, cmap='seismic')
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

    fig.savefig('max_' + data.title + '_in_altitude_longitude.png', bbox_inches='tight')

    plt.show()


# ==================================================================================================================== #
# ============================================ DISPLAY Equateur profile ============================================== #
# ==================================================================================================================== #
def display_equa_profile(data, data_time, data_latitude, data_altitude, data_temperature, data_saturation, unit):
    from numpy import abs, argmax, arange, linspace, searchsorted, int_, logspace, asarray, reshape, where, \
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
    print('To be build')
