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
def display_colonne(data):
    import matplotlib.pyplot as plt
    from numpy import sum, mean

    zonal_mean_density_column = sum(mean(data[:,:,:,:], axis=3), axis=1)
    zonal_mean_density_column = zonal_mean_density_column.T

    plt.figure()
    plt.contour(zonal_mean_density_column)
    plt.show()


def display_zonal_mean(data):
    print('To be build')