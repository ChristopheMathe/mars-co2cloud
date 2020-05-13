#======================================================================================================================#
#================================================== DISPLAY 1D ========================================================#
#======================================================================================================================#
def display_1D(data):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data)
    plt.xlim(0, len(data))
    plt.xlabel('Time')
    plt.ylabel(data.name + ' (' + data.units + ')')
    plt.show()
#======================================================================================================================#
#================================================== DISPLAY 2D ========================================================#
#======================================================================================================================#

def display_2D(data):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.contour import QuadContourSet

    def compute_and_plot(ax, alpha):
        CS = QuadContourSet(ax, data[0,alpha,:,:])
        plt.clabel(CS, inline=1, fontsize=10)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Add Ls and Altitude')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

    compute_and_plot(ax, 0)

    # Define slider for alpha
    alpha_axis = plt.axes([0.12, 0.1, 0.65, 0.03])
    alpha_slider = Slider(alpha_axis, 'Alt', 0, len(data[0,:,0,0])-1, valinit=0, valstep=1)

    def update(ax, val):
        alpha = val
        ax.cla()
        compute_and_plot(ax, alpha)
        plt.draw()

    alpha_slider.on_changed(lambda val: update(ax, val))

    plt.show()
#======================================================================================================================#
#================================================== DISPLAY 3D ========================================================#
#======================================================================================================================#


def display_3D(data):
    print('To be build')

#======================================================================================================================#
#============================================ DISPLAY Densité de colonne ==============================================#
#======================================================================================================================#

#TODO: display => gas non-condensable, la température, distribution de colonne de co2_ice, h2o_ice, h2o_vap, tau
def display_colonne(data, data_time, data_latitude, unit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from numpy import sum, mean, arange, searchsorted, int_, flip, linspace, logspace, where
    from scipy.interpolate import interp2d

    # compute zonal mean column density
    zonal_mean_column_density = sum(mean(data[:,:,:,:], axis=3), axis=1) # Ls function of lat
    zonal_mean_column_density = zonal_mean_column_density.T # lat function of Ls
    zonal_mean_column_density = flip(zonal_mean_column_density, axis=0) # reverse to get North pole on top of the fig
    data_latitude = data_latitude[::-1] # And so the labels

    # interpolation to get linear Ls
    f = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=zonal_mean_column_density,kind='linear')
    axis_ls = linspace(0, 360, len(data_time[:]))
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data_time,axis_ls)
    zonal_mean_column_density = f(interp_time,arange(len(data_latitude)))

    rows, cols = where(zonal_mean_column_density < 1e-10)
    zonal_mean_column_density[rows, cols] = 1e-10

    if unit is 'pr.µm':
        zonal_mean_column_density = zonal_mean_column_density * 1e3 # convert kg/m2 to pr.µm

    # plot
    plt.figure(figsize=(11,8))
    plt.title('Zonal mean column density of '+data.title)
    plt.contourf(zonal_mean_column_density, norm=LogNorm(), levels=logspace(-10,0,11), cmap='inferno')
    plt.yticks(ticks=arange(0,len(data_latitude), 6), labels=data_latitude[::6])
    plt.xticks(ticks=ndx, labels=int_(axis_ls[ndx]))
    cbar = plt.colorbar()
    cbar.ax.set_title(unit)
    plt.xlabel('Solar Longitude (°)')
    plt.ylabel('Latitude (°N)')
    plt.savefig('zonal_mean_density_column_'+data.title+'.png', bbox_inches='tight')
    plt.show()


#======================================================================================================================#
#================================= DISPLAY max value in altitude / longitude ==========================================#
#======================================================================================================================#


def display_max_lon_alt(data, data_time, data_latitude, data_altitude, unit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from numpy import amax, arange, searchsorted, int_, flip, linspace, where, logspace, argmax, zeros
    from scipy.interpolate import interp2d

    # get max value along altitude and longitude
    max_in_longitude = amax(data[:,:,:,:], axis=3) # get the max value in longitude
    max_in_longitude_altitude = amax(max_in_longitude, axis=1) # get the max value in altitude

    # get the altitude index of the max value
    index_max = argmax(max_in_longitude, axis=1)
    altitude_max = zeros((index_max.shape[0], index_max.shape[1]))
    for i in range(index_max.shape[0]):
        for j in range(index_max.shape[1]):
            altitude_max[i,j] = data_altitude[index_max[i,j]]

    # reshape data
    max_in_longitude_altitude = max_in_longitude_altitude.T # lat function of Ls
    max_in_longitude_altitude = flip(max_in_longitude_altitude, axis=0) # reverse to get North pole on top of the fig
    altitude_max = altitude_max.T
    altitude_max = flip(altitude_max, axis=0)
    data_latitude = data_latitude[::-1] # And so the labels


    # interpolation to get linear Ls
    f = interp2d(x=arange(len(data_time)), y=arange(len(data_latitude)), z=max_in_longitude_altitude,kind='linear')
    axis_ls = linspace(0, 360, len(data_time[:]))
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data_time,axis_ls)
    max_in_longitude_altitude = f(interp_time,arange(len(data_latitude)))

    rows, cols = where(max_in_longitude_altitude < 1e-10)
    max_in_longitude_altitude[rows, cols] = 1e-10

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,8))
    ax[0].set_title('Max '+data.title+' in altitude/longitude')
    pc = ax[0].contourf(max_in_longitude_altitude, norm=LogNorm(), levels=logspace(-10,0,11), cmap='inferno')
    ax[0].set_yticks(ticks=arange(0,len(data_latitude), 6))
    ax[0].set_yticklabels(labels=data_latitude[::6])
    ax[0].set_xticks(ticks=ndx)
    ax[0].set_xticklabels(labels=int_(axis_ls[ndx]))
    cbar = plt.colorbar(pc, ax=ax[0])
    cbar.ax.set_title(unit)
    ax[0].set_ylabel('Latitude (°N)')

    # plot 2
    print(altitude_max.shape)
    ax[1].set_title('Altitude of max mmr co2_ice')
    pc2 = ax[1].contourf(altitude_max, cmap='inferno')
    ax[1].set_yticks(ticks=arange(0,len(data_latitude), 6))
    ax[1].set_yticklabels(labels=data_latitude[::6])
    ax[1].set_xticks(ticks=ndx)
    ax[1].set_xticklabels(labels=int_(axis_ls[ndx]))
    ax[1].set_xlabel('Solar Longitude (°)')
    ax[1].set_ylabel('Latitude (°N)')
    cbar2 = plt.colorbar(pc2, ax=ax[1])
    cbar2.ax.set_title('km')

    fig.savefig('max_'+data.title+'_in_altitude_longitude.png', bbox_inches='tight')
    plt.show()


#======================================================================================================================#
#============================================ DISPLAY XXXXXX ==============================================#
#======================================================================================================================#


def display_zonal_mean(data, data_time, data_latitude):
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