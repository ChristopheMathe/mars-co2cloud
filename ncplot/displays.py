from sys import exit


# ======================================================================================================================#
# ================================================== DISPLAY 1D ========================================================#
# ======================================================================================================================#
def display_1D(data):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data)
    plt.xlim(0, len(data))
    plt.xlabel('Time')
    plt.ylabel(data.name + ' (' + data.units + ')')
    plt.show()


# ======================================================================================================================#
# ================================================== DISPLAY 2D ========================================================#
# ======================================================================================================================#

def display_2D(data):
    import matplotlib.pyplot as plt
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


# ======================================================================================================================#
# ================================================== DISPLAY 3D ========================================================#
# ======================================================================================================================#


def display_3D(data):
    print('To be build')


# ======================================================================================================================#
# ============================================ DISPLAY Densité de colonne ==============================================#
# ======================================================================================================================#

# TODO: display => gas non-condensable, la température, distribution de colonne de co2_ice, h2o_ice, h2o_vap, tau
def display_colonne(data, data_time, data_latitude, unit):
    import matplotlib.pyplot as plt
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
    plt.show()


# ======================================================================================================================#
# ================================= DISPLAY max value in altitude / longitude ==========================================#
# ======================================================================================================================#


def display_max_lon_alt(data, data_time, data_latitude, data_altitude, data_temp, data_satu, unit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, DivergingNorm
    from numpy import arange, searchsorted, int_, flip, linspace, where, logspace, unravel_index, swapaxes, reshape, \
        asarray
    from scipy.interpolate import interp2d

    data_y = data[:, :, :, :]

    # correct very low values of co2 mmr
    for i in range(data_y.shape[0]):
        dim2, dim3, dim4 = where(data_y[i, :, :, :] < 1e-10)
        data_y[i, dim2, dim3, dim4] = 0

    # get max value along altitude and longitude
    # max_mmr = amax(data_y, axis=(1,3)) # get the max mmr value in longitude/altitude

    B = swapaxes(data_y, 1, 2)
    max_idx = B.reshape((B.shape[0], B.shape[1], -1)).argmax(2)
    x, y = unravel_index(max_idx, B[0, 0, :].shape)

    max_mmr = [B[i, j, x[i, j], y[i, j]] for i in range(B.shape[0]) for j in range(B.shape[1])]
    max_mmr = asarray(max_mmr)
    max_mmr = reshape(max_mmr, (data_y.shape[0], data_y.shape[2]))

    max_temp = swapaxes(data_temp, 1, 2)
    max_temp = [max_temp[i, j, x[i, j], y[i, j]] for i in range(max_temp.shape[0]) for j in range(max_temp.shape[1])]
    max_temp = asarray(max_temp)
    max_temp = reshape(max_temp, (data_y.shape[0], data_y.shape[2]))

    max_satu = swapaxes(data_satu, 1, 2)
    max_satu = [max_satu[i, j, x[i, j], y[i, j]] for i in range(max_satu.shape[0]) for j in range(max_satu.shape[1])]
    max_satu = asarray(max_satu)
    max_satu = reshape(max_satu, (data_y.shape[0], data_y.shape[2]))

    max_alt = [data_altitude[x[i, j]] for i in range(B.shape[0]) for j in range(B.shape[1])]
    max_alt = asarray(max_alt)
    max_alt = reshape(max_alt, (data_y.shape[0], data_y.shape[2]))

    # reshape data
    max_mmr = max_mmr.T  # lat function of Ls
    max_mmr = flip(max_mmr, axis=0)  # reverse to get North pole on top of the fig

    max_temp = max_temp.T
    max_temp = flip(max_temp, axis=0)

    max_satu = max_satu.T
    max_satu = flip(max_satu, axis=0)

    max_alt = max_alt.T
    max_alt = flip(max_alt, axis=0)

    data_latitude = data_latitude[::-1]  # And so the labels

    # interpolation to get linear Ls
    f = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=max_mmr, kind='linear')
    f2 = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=max_alt, kind='linear')
    f3 = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=max_temp, kind='linear')
    f4 = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=max_satu, kind='linear')

    axis_ls = linspace(0, 360, len(data_time[:]))
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data_time, axis_ls)

    max_mmr = f(interp_time, arange(len(data_latitude)))
    max_alt = f2(interp_time, arange(len(data_latitude)))
    max_temp = f3(interp_time, arange(len(data_latitude)))
    max_satu = f4(interp_time, arange(len(data_latitude)))

    # mask above/below 40 km
    #    mask = (data_altitude[:] <= 40)
    #    mask_40km = (max_alt >= 40)
    #    max_mmr = array(max_mmr, mask=mask_40km, fill_value=None)
    #    max_alt = array(max_alt, mask=mask_40km, fill_value=None)
    #    max_temp = array(max_temp, mask=mask_40km, fill_value=None)
    #    max_satu = array(max_satu, mask=mask_40km, fill_value=None)

    # plot 1
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(11, 20))
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
    pc3 = ax[1].contourf(max_alt, cmap='seismic')
    ax[1].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[1].set_yticklabels(labels=data_latitude[::6])
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels='')
    ax[1].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[1])
    cbar3.ax.set_title('km')

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
    pc3 = ax[3].contourf(max_satu, cmap='seismic', norm=divnorm, levels=arange(0, 5))
    ax[3].set_yticks(ticks=arange(0, len(data_latitude), 6))
    ax[3].set_yticklabels(labels=data_latitude[::6])
    ax[3].set_xticks(ticks=ndx)
    ax[3].set_xticklabels(labels=int_(axis_ls[ndx]))
    ax[3].set_xlabel('Solar Longitude (°)')
    ax[3].set_ylabel('Latitude (°N)')
    cbar3 = plt.colorbar(pc3, ax=ax[3])
    cbar3.ax.set_title(' ')

    fig.savefig('max_' + data.title + '_in_altitude_longitude.png', bbox_inches='tight')

    plt.show()


# ======================================================================================================================#
# ============================================ DISPLAY XXXXXX ==========================================================#
# ======================================================================================================================#


def display_equa_profile(data, data_time, data_latitude, data_altitude, data_temperature, data_saturation, unit):
    import matplotlib.pyplot as plt
    from numpy import abs, argmax, arange, linspace, searchsorted, int_, logspace, asarray, reshape, where, unravel_index
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
    #idx_max = argmax(data, axis=2)
    idx_max = data.reshape((data.shape[0], -1)).argmax(axis=1)
    idx_alt, idx_lon = unravel_index(idx_max, (data.shape[1], data.shape[2]))
#    B = [data[i, j, idx_max[i, j]] for i in range(dim1) for j in range(dim2)]
    B = [data[i, :, idx_lon[i]] for i in range(dim1)]
    B = asarray(B)
    data = reshape(B, (dim1, dim2))

    #    B = [data_temperature[i, j, idx_max[i, j]] for i in range(dim1) for j in range(dim2)]
    B = [data_temperature[i, :, idx_lon[i]] for i in range(dim1)]
    B = asarray(B)
    data_temperature = reshape(B, (dim1, dim2))

    #    B = [data_saturation[i, j, idx_max[i, j]] for i in range(dim1) for j in range(dim2)]
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
    import matplotlib.pyplot as plt
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
