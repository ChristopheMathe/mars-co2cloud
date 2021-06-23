#!/bin/bash python3
from packages.lib_function import *
from packages.ncdump import *
from packages.DataObservation import viking_lander
from sys import argv, exit
import matplotlib.pyplot as plt
from numpy import ceil, floor, min, max, mean, zeros, sum, arange, abs, concatenate, append
from matplotlib.colors import DivergingNorm, Normalize


def number_year(data_time):
    # Define the number of years simulated
    if (data_time[-1]/669.) % 0:
        nb_year = int(round(data_time[-1]/669.))
    elif (data_time[-1]/360.) % 0:
        nb_year = int(round(data_time[-1]/360.))
    else:
        nb_year = 0
        print(f'Time: from {data_time[0]} {data_time.units} to {data_time[-1]} {data_time.units}')
        print('Did you finish the year?')
        exit()
    return nb_year


def plot_global_mean(data_ps, data_tsurf, data_co2ice, data_h2o_ice_s, data_mtot, data_icetot, nb_year):
    def plot(axes, data, title, y_label):
        axes.set_title(title)
        for i in range(nb_year):
            axes.plot(data[i * 669:(i + 1) * 669], label=f'year {i + 1}')
        axes.legend(loc='best')
        axes.set_xlabel('Sols')
        axes.set_ylabel(y_label)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize_1graph_ytend)
    plot(axes=ax[0, 0], data=data_ps, title='Surface pressure, global mean', y_label=f'Pressure (Pa)')
    plot(axes=ax[0, 1], data=data_tsurf, title='Surface temperature, global mean', y_label=f'Temperature (K)')
    plot(axes=ax[1, 0], data=data_co2ice, title='Total surface CO2 ice', y_label=f'Mass (kg)')
    plot(axes=ax[1, 1], data=data_h2o_ice_s, title='Total surface H2O ice', y_label=f'Mass (kg)')
    plot(axes=ax[2, 0], data=data_mtot, title='Total H2O vapor in atm', y_label=f'Mass (kg)')
    plot(axes=ax[2, 1], data=data_icetot, title='Total H2O ice in atm', y_label=f'Mass (kg)')

    fig.savefig(f'check_convergence_global_mean.png', bbox_inches='tight')


def plot_pressure_viking(data_pressure, data_latitude, nb_year):
    data_sols_1, data_pressure_viking1 = viking_lander(lander=1)
    data_sols_2, data_pressure_viking2 = viking_lander(lander=2)

    # Viking 1: Chryse Planitia (26° 42′ N, 320° 00′ E)
    data_pressure_at_viking1, latitude1 = slice_data(data=data_pressure, dimension_data=data_latitude, value=26)

    # Viking 2: Utopia Planitia (49° 42′ N, 118° 00′ E)
    data_pressure_at_viking2, latitude2 = slice_data(data=data_pressure, dimension_data=data_latitude, value=49)

    fig, ax = plt.subplots(ncols=2, figsize=figsize_2graph_cols)
    fig.suptitle('Zonal mean and diurnal mean of surface pressure at', fontsize=fontsize)
    ax[0].set_title(f'Viking 1 ({data_latitude[latitude1]:.0f}°N)', fontsize=fontsize)
    ax[1].set_title(f'Viking 2 ({data_latitude[latitude2]:.0f}°N)', fontsize=fontsize)

    ax[0].scatter(data_sols_1, data_pressure_viking1, c='black')
    ax[1].scatter(data_sols_2, data_pressure_viking2, c='black')

    for i in range(nb_year):
        ax[0].plot(data_pressure_at_viking1[i*669:(i+1)*669], label=f'year {i+1}')
        ax[1].plot(data_pressure_at_viking2[i*669:(i+1)*669], label=f'year {i+1}')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[0].set_xlabel('Sols', fontsize=fontsize)
    ax[0].set_ylabel('Pressure (Pa)', fontsize=fontsize)
    ax[1].set_xlabel('Sols', fontsize=fontsize)
    ax[1].set_ylabel('Pressure (Pa)', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig('check_convergence_pressure_viking_land_site.png')


def list_filename():
    directory_store = [x for x in listdir('.') if 'occigen' in x][0] + '/'
    files = listdir(directory_store)

    filename = list([])
    filename.append(directory_store+getfilename(files, selection=None))
    test = 1
    while test >= 0:
        test = int(input('Another (-1: exit): '))
        print(test)
        if test >= 0:
            print(test)
            filename.append(directory_store+getfilename(files, selection=test))
    return filename


def concatenation(filenames, target):
    list_tmp = []
    for i, value_i in enumerate(filenames):
        print(value_i)
        data_target, list_var = get_data(filename=value_i, target=target)
        data_time, list_var = get_data(filename=value_i, target='Time')
        nb_year = number_year(data_time=data_time[:])
        if data_target.name == 'watercap':
            data_tmp = data_target[:, :, :]
        else:
            data_tmp = correction_value(data=data_target[:, :, :], operator='inf', threshold=threshold)
        data_tmp = data_tmp.reshape(669*nb_year, 12, data_tmp.shape[1], data_tmp.shape[2])
        data_tmp = mean(data_tmp, axis=1)
        area = gcm_area()
        if data_target.name in ['co2ice', 'h2o_ice_s', 'watercap', 'mtot','icetot']:
            data_tmp = data_tmp * area
            data_tmp = sum(data_tmp, axis=(1, 2))
        else:
            data_tmp = sum(data_tmp*area/sum(area), axis=(1,2))

        list_tmp.append(data_tmp)

    if len(list_tmp) == 2:
        data = concatenate((list_tmp[0], list_tmp[1]), axis=0)
    else:
        data = list_tmp[0]

    return data


def main():
    filenames = list_filename()

    # Deals with variables
    data_ps = concatenation(filenames=filenames, target='ps')
    data_tsurf = concatenation(filenames=filenames, target='tsurf')
    data_co2ice = concatenation(filenames=filenames, target='co2ice')
    data_h2o_ice_s = concatenation(filenames=filenames, target='h2o_ice_s')
    data_watercap = concatenation(filenames=filenames, target='watercap')
    data_mtot = concatenation(filenames=filenames, target='mtot')
    data_icetot = concatenation(filenames=filenames, target='icetot')

    # Total number of year
    total_nb_year = 0
    for i, value_i in enumerate(filenames):
        data_time, list_var = get_data(filename=value_i, target='Time')
        total_nb_year += number_year(data_time=data_time[:])

    # main plot
    plot_global_mean(data_ps, data_tsurf, data_co2ice, data_h2o_ice_s + data_watercap, data_mtot, data_icetot,
                     nb_year=total_nb_year)

    plt.figure()
    for i in range(total_nb_year):
        tmp = data_h2o_ice_s[i*669:(i+1)*669] + data_mtot[i*669:(i+1)*669] + data_icetot[i*669:(i+1)*669] + \
              data_watercap[i*669:(i+1)*669]
        plt.plot(tmp, label=f'year {i+1}')
    plt.savefig('check_converge_total_mass_h2o.png')
    # Check pressure at Viking land site
#    data_latitude, list_var = get_data(filename=filenames[0], target='latitude')
#    plot_pressure_viking(data_pressure=data_ps, data_latitude=data_latitude, nb_year=total_nb_year)


if '__main__' == __name__:
    main()


# OLDEST

#    norm_relative = DivergingNorm(vmin=-1, vcenter=0, vmax=1)
#    fig, ax = plt.subplots(ncols=2, figsize=(22, 11))
#    fig.suptitle(f'Zonal mean of {name_target} (diurnal mean)')
#    ax[0].set_title(f'year 1')
#    pcm = ax[0].pcolormesh(data_time[:669], data_latitude[:], data_target[:, :669], norm=norm_1styear, cmap='plasma')
#    cbar = plt.colorbar(pcm, ax=ax[0])
#    cbar.ax.set_title(unit_target, fontsize=fontsize)

#    mean_relative = zeros((data_target.shape[0], nb_year-1))
#    for i in range(1, nb_year):
#        relative = 100 * (data_target[:, (i-1)*669:i*669] - data_target[:, i*669:(i+1)*669]) /\
#                   data_target[:, i*669:(i+1)*669]
#        mean_relative[:, i-1] = mean(relative, axis=1)
#    pcm2 = ax[1].pcolormesh(arange(nb_year), data_latitude[:], mean_relative, norm=norm_relative, cmap='seismic')
#    ax[1].set_xlabel('Year')
#    ax[1].set_ylabel(f'Latitude (°N)')
#    ax[1].set_xticks(arange(nb_year-1)+0.5)
#    ax[1].set_xticklabels(['2', '3', '4', '5'], fontsize=fontsize)
#    cbar2 = plt.colorbar(pcm2, ax=ax[1])
#    cbar2.ax.set_title('%')
#    fig.savefig(f'check_convergence_{name_target}_zonal_mean_diurnal_mean.png', bbox_inches='tight')
#    fig.show()


#    norm_1styear = Normalize(vmin=floor(min(data_target[:, 669])), vmax=ceil(max(data_target[:, :669])))
#    fig, ax = plt.subplots(ncols=2, figsize=(22, 11))
#    fig.suptitle(f'Zonal mean of {name_target} (diurnal mean)')
#    ax[0].set_title(f'year 1')
#    tmp = data_time[::12]

#    pcm = ax[0].pcolormesh(tmp[:669], data_latitude[:], data_target[:, :669], norm=norm_1styear, cmap='plasma')
#    cbar = plt.colorbar(pcm, ax=ax[0])
#    cbar.ax.set_title(unit_target, fontsize=fontsize)
#    norm = zeros(nb_year-1)
#    for i in range(1, nb_year):
#        a = data_target[:, (i-1)*669:i*669]
#        b = data_target[:, i*669:(i+1)*669]
#        norm[i-1] = mean(abs(100.*(b - a)/a))
#    ax[1].plot(norm)
#    ax[1].set_xlabel('Year')
#    ax[1].set_ylabel('Mean relative error from previous year (%)')
#    ax[1].set_xticks(arange(nb_year-1))
#    ax[1].set_xticklabels(['2', '3', '4', '5', '6', '7'])
#    fig.savefig(f'check_convergence_{name_target}_zonal_mean_diurnal_mean.png', bbox_inches='tight')
