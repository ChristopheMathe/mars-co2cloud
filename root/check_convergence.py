#!/bin/bash python3
from packages.lib_function import *
from packages.ncdump import *
from packages.DataObservation import viking_lander
from sys import argv, exit
import matplotlib.pyplot as plt
from numpy import ceil, floor, min, max, mean, zeros, sum, arange, abs, concatenate, append, linspace
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

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
    print(f'There is {nb_year} years simulated.')
    return nb_year


def plot_global_mean(data_ps, data_tsurf, data_co2ice, data_h2o_ice_s, data_mtot, data_icetot, nb_year):
    def plot(axes, data, title, y_label):
        cmap = plt.get_cmap('jet')
        colors = [cmap(i) for i in linspace(0, 1, nb_year)]
        axes.set_title(title)
        for i in range(nb_year):
            axes.plot(data[i * 669:(i + 1) * 669], color=colors[i], label=f'year {i + 1}')
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


def list_filename():
    directory_store = [x for x in listdir('.') if 'occigen' in x][0] + '/'
    files = listdir(directory_store)

    filename = list([])
    filename.append(directory_store+getfilename(files, selection=None))
    test = 1
    while test >= 0:
        test = int(input('Another (-1: exit): '))
        if test >= 0:
            filename.append(directory_store+getfilename(files, selection=test))
    return filename


def concatenation(filenames, target):
    list_tmp = []
    for i, value_i in enumerate(filenames):
        data_target, list_var = get_data(filename=value_i, target=target)
        data_time, list_var = get_data(filename=value_i, target='Time')
        nb_year = number_year(data_time=data_time[:])
        if data_target.name == 'watercap':
            data_tmp = data_target[:, :, :]
        else:
            data_tmp = correction_value(data=data_target[:, :, :], operator='inf', value=threshold)
        data_tmp = data_tmp.reshape(669*nb_year, 12, data_tmp.shape[1], data_tmp.shape[2])
        data_tmp = mean(data_tmp, axis=1)
        area = gcm_area()
        if data_target.name in ['co2ice', 'h2o_ice_s', 'watercap', 'mtot', 'icetot']:
            data_tmp = data_tmp * area
            data_tmp = sum(data_tmp, axis=(1, 2))
        else:
            data_tmp = sum(data_tmp*area/sum(area), axis=(1, 2))

        list_tmp.append(data_tmp)

    if len(list_tmp) > 1:
        data = concatenate(([list_tmp[x] for x in range(len(list_tmp))]), axis=0)
    else:
        data = list_tmp[0]

    return data


def main():
    filenames = list_filename()

    # Total number of year
    total_nb_year = 0
    for i, value_i in enumerate(filenames):
        data_time, list_var = get_data(filename=value_i, target='Time')
        total_nb_year += number_year(data_time=data_time[:])

    test = input(f'There is {total_nb_year} years. Do you want select a year limit? (y/N)')
    if test.lower() == 'y':
        total_nb_year = int(input("Enter the year: "))

    # Deals with variables
    data_ps = concatenation(filenames=filenames, target='ps')
    data_tsurf = concatenation(filenames=filenames, target='tsurf')
    try:
        data_co2ice = concatenation(filenames=filenames, target='co2ice')
    except:
        data_co2ice = zeros(data_ps.shape)
    data_h2o_ice_s = concatenation(filenames=filenames, target='h2o_ice_s')
    data_watercap = concatenation(filenames=filenames, target='watercap')
    data_mtot = concatenation(filenames=filenames, target='mtot')
    data_icetot = concatenation(filenames=filenames, target='icetot')

    # main plot
    plot_global_mean(data_ps, data_tsurf, data_co2ice, data_h2o_ice_s + data_watercap, data_mtot, data_icetot,
                     nb_year=total_nb_year)

    plt.figure()
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in linspace(0, 1, total_nb_year)]
    for i in range(total_nb_year):
        tmp = data_h2o_ice_s[i*669:(i+1)*669] + data_mtot[i*669:(i+1)*669] + data_icetot[i*669:(i+1)*669] + \
              data_watercap[i*669:(i+1)*669]
        plt.plot(tmp, label=f'year {i+1}', color=colors[i])
    plt.xlabel('Time (sols)')
    plt.ylabel('Total H2O mass (kg)')
    plt.legend(loc=0)
    plt.savefig('check_converge_total_mass_h2o.png')

    # save data
    dict_var = [{'data': arange(669*total_nb_year), 'varname': 'Time', 'units': 'sols', 'shortname': 'TIME',
                 'dimension': True},
                {'data': data_ps[:669*total_nb_year], 'varname': 'Surface pressure, global mean', 'units': 'Pa', 'shortname': 'ps',
                 'dimension': False},
                {'data': data_tsurf[:669*total_nb_year], 'varname': 'Surface temperature, global mean', 'units': 'K', 'shortname':
                    'tsurf', 'dimension': False},
                {'data': data_co2ice[:669*total_nb_year], 'varname': 'Total surface CO2 ice', 'units': '(kg)', 'shortname': 'co2ice',
                 'dimension': False}
                ]
    save_figure_data(list_dict_var=dict_var, savename=f"convergence_of_{total_nb_year:.0f}_year")


if '__main__' == __name__:
    main()