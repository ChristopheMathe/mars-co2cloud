#!/bin/bash python3
from netCDF4 import Dataset
from os import listdir, path
from sys import exit, stdout
import importlib
from numpy import mean, abs, min, max, zeros, where, ones, concatenate, flip, arange, unravel_index, argmax, array, \
    count_nonzero, diff, std, savetxt, c_, append, loadtxt, asarray, power, linspace, logspace, sum

displays = importlib.import_module('packages.displays')
libf = importlib.import_module('packages.lib_function')
ncdump = importlib.import_module('packages.ncdump')


def getfilename(files):
    if any(".nc" in s for s in files):
        list_files = sorted([x for x in files if '.nc' in x])
        if len(list_files) > 1:
            print('Netcdf files available: (0) {}'.format(list_files[0]))
            for i, value_i in enumerate(list_files[1:]):
                print('                        ({}) {}'.format(i + 1, value_i))
            filename = int(input("Select the file number: "))
            filename = list_files[filename]
            print('')
        else:
            filename = list_files[0]
    else:
        print('There is no Netcdf file in this directory !')
        filename = ''
        exit()
    return filename


def getdata(filename, target=None):
    # name_dimension_target = data.variables[variable_target].dimensions  # get access to all dimensions names
    # dimension = array(len(name_dimension_target))

    data = Dataset(filename, "r", format="NETCDF4")
    if target == None:
        tmp, tmp, tmp, tmp, variable_target = ncdump.ncextract(filename, data, verb=True)
        if variable_target == None:
            variable_target = input('Select the variable: ')  # TODO faire les tests si la variable existe
        else:
            print("Variable selected is {}".format(variable_target))
    else:
        variable_target = target

    data_target = data.variables[variable_target]

    return data_target


def main():
    directory_store = 'occigen_test_64x48x32_1years_Tµphy_para_check_co2_ice/'

    try:
        files = listdir(directory_store)
    except:
        files = listdir('.')
        directory_store = ''

    filename = getfilename(files)

    data_target = getdata(directory_store+filename)
    variable_target = data_target.name

    data_latitude = getdata(directory_store+filename, target="latitude")
    data_latitude = data_latitude[::-1]  # reverse the labels

    print('Create linear grid time...')
    data_time = getdata(directory_store+filename, target="Time")

    print(max(data_time))
    if max(data_time) <= 360:
        interp_time, axis_ls, ndx = libf.linear_grid_ls(data_time)
        print('Check if you correctly linearile data on ls')
        lslin = True
    else:
        ndx, axis_ls = libf.get_ls_index(data_time[:])
        interp_time = 0
        lslin = False

    if variable_target in ['co2_ice', 'h2o_ice', 'q01']: # q01 = h2o_ice
        # correct very low values of co2/h2o mmr
        print('Correction value...')
        data_target = libf.correction_value_co2_ice(data_target[:, :, :, :])
        shape_data_target = data_target.shape

        view_mode = int(input('View (max=1, column=2, equa_prof=3, alt-lat=4, hovmoller=5): '))

        if view_mode == 1:
            print('Get max co2_ice...')
            max_mmr, x, y = libf.get_max_co2_ice_in_alt_lon(data_target)

            print('Extract other variable at co2_ice max value...')
            max_temp = libf.extract_at_max_co2_ice(data_temperature, x, y)
            max_satu = libf.extract_at_max_co2_ice(data_satuco2, x, y)
            max_radius = libf.extract_at_max_co2_ice(data_riceco2, x, y)
            max_ccnN = libf.extract_at_max_co2_ice(data_ccnNco2, x, y)
            max_alt = libf.extract_at_max_co2_ice(data_altitude, x, y, shape_data_target)

            print('Reshape and linearized data...')
            max_mmr = libf.linearize_ls(max_mmr, shape_data_target[0], shape_data_target[2], interp_time)
            max_temp = libf.linearize_ls(max_temp, shape_data_target[0], shape_data_target[2], interp_time)
            max_satu = libf.linearize_ls(max_satu, shape_data_target[0], shape_data_target[2], interp_time)
            max_radius = libf.linearize_ls(max_radius, shape_data_target[0], shape_data_target[2], interp_time)
            max_ccnN = libf.linearize_ls(max_ccnN, shape_data_target[0], shape_data_target[2], interp_time)
            max_alt = libf.linearize_ls(max_alt, shape_data_target[0], shape_data_target[2], interp_time)

            displays.display_max_lon_alt(data_target.title,
                                         data_latitude,
                                         max_mmr,
                                         max_alt,
                                         max_temp,
                                         max_satu,
                                         max_radius,
                                         max_ccnN,
                                         axis_ls,
                                         ndx,
                                         unit='kg/kg')

        elif view_mode == 2:
            data_altitude = getdata(directory_store+filename, target='altitude')

            try:
                data_pressure = getdata(directory_store+filename, target='pressure')
            except:
                data_pressure = getdata(directory_store+'concat_sols_vars_S.nc', target='pressure')

            data_target, altitude_limit, zmin, zmax = libf.zonal_mean_column_density(data_target, data_pressure,
                                                                                     data_altitude, interp_time)

            displays.display_colonne(data_target, data_latitude, altitude_limit, zmin, zmax, ndx, axis_ls, interp_time,
                                     levels=logspace(-8, -1, 8), unit='kg/m$^2$', name=variable_target)

        elif view_mode == 3:
            displays.display_equa_profile(data_target, data_time, data_latitude, data_altitude, data_temperature,
                                          data_satuco2, unit='kg/kg')

        elif view_mode == 4:
            displays.display_altitude_latitude(data_target, unit='kg/kg',
                                               savename='altitude_latitude_co2_ice_northpole.png')

        elif view_mode == 5:
            displays.hovmoller_diagram(data_target, data_latitude, data_altitude, data_longitude, data_time)

        elif view_mode == 6:
            displays.display_vertical_profile()

        elif view_mode == 7:
            filename_2 = getfilename(files)
            data_target_2 = getdata(directory_store+filename_2, variable_target)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')
            print('Correction value...')
            data_y_2 = libf.correction_value_co2_ice(data_target_2[:, :, :, :])
            data_altitude = getdata(directory_store+filename_2, 'altitude')
            print('Get max of {}...'.format(data_target.name))
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_y, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_y_2, extrema='max')

            max_alt_day = ones(idx_altitude_day.shape)
            max_alt_night = ones(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
                    max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
                    if max_alt_day[i, j] == 0 and max_satu_day[i, j] < 1e-10:
                        max_alt_day[i, j] = None
                        max_satu_day[i, j] = None
                    if max_alt_night[i, j] == 0 and max_satu_night[i, j] < 1e-10:
                        max_alt_night[i, j] = None
                        max_satu_night[i, j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='kg/kg', title='Max vmr of CO$_2$ ice',
                                               savename='max_co2_ice_day_night.png')
        if view_mode == 8:
            data_longitude = getdata(directory_store+filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lon]

            idx_ls_1 = (abs(data_time[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time[:] - 318)).argmin()
            data_target = data_target[idx_ls_1:idx_ls_2, :]
            shape_data_target = data_target.shape
            data_target = mean(data_target.reshape(-1, 59), axis=1)
            data_target = data_target.reshape(int(shape_data_target[0] / 59), shape_data_target[1])

            filename_2 = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')

            data_target_2 = getdata(directory_store+filename_2, variable_target)
            print('Correction value...')
            data_target_2 = libf.correction_value_co2_ice(data_target_2[:, :, :, :])
            data_altitude = getdata(directory_store+filename_2, 'altitude')
            data_time_2 = getdata(directory_store+filename_2, 'Time')

            idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
            data_target_2 = data_target_2[idx_ls_1:idx_ls_2, :, idx_lat, idx_lon]
            shape_data_target_2 = data_target_2.shape
            data_target_2 = mean(data_target_2.reshape(-1, 59), axis=1)
            data_target_2 = data_target_2.reshape(int(shape_data_target_2[0] / 59), shape_data_target_2[1])

            # 00-4h + 6-18h + 20-22h
            data_target = concatenate((data_target_2[3:-1, :], data_target, data_target_2[1:3, :]))

            displays.display_altitude_localtime(data_target, data_altitude, unit='kg/kg',
                                                savename='co2_ice_altitude_localtime')
        if view_mode == 9:
            # la plus grande épaisseur de neige de co2 pendant l'hiver polaire
            data_altitude = getdata(directory_store+filename, 'altitude')

            idx_lat_N = (abs(data_latitude[:] - 60)).argmin()
            idx_lat_S = (abs(data_latitude[:] + 60)).argmin()
            print(data_target.shape)

            data_target_north = data_target[:, :, idx_lat_N:, :]  # north pole
            data_target_south = data_target[:, :, :idx_lat_S + 1, :]  # south pole
            print(data_latitude[idx_lat_N:])
            print(data_latitude[:idx_lat_S + 1])

            # bin ls to 5°
            if max(data_time) < 360:
                ls_bin = arange(0, 365, 5)
            else:
                # sols binned at 5° Ls
                ls_bin = libf.convert_sols_tols()

            nbin = ls_bin.shape[0]
            data_icelayer = zeros((2, nbin))

            for bin in range(nbin - 1):
                idx = (abs(data_time[:] - ls_bin[bin])).argmin()
                idx2 = (abs(data_time[:] - ls_bin[bin + 1])).argmin()
                data_binned_north = data_target_north[idx:idx2, :, :, :]
                data_binned_south = data_target_south[idx:idx2, :, :, :]
                if (max(data_binned_north) >= 1e-10):
                    ind = unravel_index(argmax(data_binned_north, axis=None), data_binned_north.shape)
                    for i in range(ind[1], data_altitude.shape[0]):
                        if data_binned_north[ind[0], i, ind[2], ind[3]] >= 1e-10:
                            idx_max = i
                    data_icelayer[0, bin] = data_altitude[idx_max] - data_altitude[ind[1]]
                if (max(data_binned_south) >= 1e-10):
                    ind = unravel_index(argmax(data_binned_south, axis=None), data_binned_south.shape)
                    for i in range(ind[1], data_altitude.shape[0]):
                        if data_binned_south[ind[0], i, ind[2], ind[3]] >= 1e-10:
                            idx_max = i
                    data_icelayer[1, bin] = data_altitude[idx_max] - data_altitude[ind[1]]
            displays.display_thickness_co2ice_atm_layer(data_icelayer)

        if view_mode == 10:
            data_altitude = getdata(directory_store+filename, 'altitude')

            idx_lat_N = (abs(data_latitude[:] - 60)).argmin()
            idx_lat_S = (abs(data_latitude[:] + 60)).argmin()

            data_target_north = data_target[:, :, idx_lat_N:, :]  # north pole
            data_target_south = data_target[:, :, :idx_lat_S + 1, :]  # south pole

            idx_ls_1 = (abs(data_time[:] - 225)).argmin()  # ls = 104°
            idx_ls_2 = (abs(data_time[:] - 669)).argmin()
            data_target_north = data_target_north[idx_ls_1:idx_ls_2 + 1, :, :, :]
            data_target_south = data_target_south[idx_ls_1:idx_ls_2 + 1, :, :, :]

            distribution_north = zeros((data_target_north.shape[2], data_target_north.shape[1]))
            distribution_south = zeros((data_target_north.shape[2], data_target_north.shape[1]))

            for latitude in range(data_target_north.shape[2]):
                for altitude in range(data_target_north.shape[1]):
                    distribution_north[latitude, altitude] = count_nonzero(
                        data_target_north[:, altitude, latitude, :] >= 1e-10)
                    distribution_south[latitude, altitude] = count_nonzero(
                        data_target_south[:, altitude, latitude, :] >= 1e-10)

            displays.display_distribution_altitude_latitude_polar(distribution_north, distribution_south, data_altitude,
                                                                  data_latitude[idx_lat_N:], data_latitude[
                                                                                             :idx_lat_S + 1],
                                                                  savename='distribution_polar_clouds')

        if view_mode == 11:
            data_altitude = getdata(directory_store+filename, 'altitude')
            data_altitude = data_altitude[:]

            # between 15°S and 15°N
            idx_lat_1 = (abs(data_latitude[:] + 15)).argmin()
            idx_lat_2 = (abs(data_latitude[:] - 15)).argmin()

            data_target = data_target[:, :, idx_lat_1:idx_lat_2 + 1, :]
            idx_max = unravel_index(argmax(data_target, axis=None), data_target.shape)
            print(idx_max)

            data_satuco2 = getdata(directory_store+filename, 'satuco2')
            data_satuco2 = data_satuco2[:, :, idx_lat_1:idx_lat_2 + 1, :]

            data_temp = getdata(directory_store+filename, 'temp')
            data_temp = data_temp[:, :, idx_lat_1:idx_lat_2 + 1, :]

            data_riceco2 = getdata(directory_store+filename, 'riceco2')
            data_riceco2 = data_riceco2[:, :, idx_lat_1:idx_lat_2 + 1, :]

            displays.display_cloud_evolution_localtime(data_target, data_satuco2, data_temp, data_riceco2, idx_max,
                                                       data_altitude, data_time)
        if view_mode == 12:
            data_altitude = getdata(directory_store+filename, 'altitude')
            data_longitude = getdata(directory_store+filename, 'longitude')

            # between 15°S and 15°N
            idx_lat_1 = (abs(data_latitude[:] + 15)).argmin()
            idx_lat_2 = (abs(data_latitude[:] - 15)).argmin()

            data_target = data_target[:, :, idx_lat_1:idx_lat_2 + 1, :]
            idx_max = unravel_index(argmax(data_target, axis=None), data_target.shape)
            print(idx_max)

            data_satuco2 = getdata(directory_store+filename, 'satuco2')
            data_satuco2 = data_satuco2[:, :, idx_lat_1:idx_lat_2 + 1, :]

            data_temp = getdata(directory_store+filename, 'temp')
            data_temp = data_temp[:, :, idx_lat_1:idx_lat_2 + 1, :]

            data_riceco2 = getdata(directory_store+filename, 'riceco2')
            data_riceco2 = data_riceco2[:, :, idx_lat_1:idx_lat_2 + 1, :]

            filenames = []
            for i in [-2, -1, 0, 1, 2]:
                filenames.append(displays.display_cloud_evolution_longitude(data_target, data_satuco2,
                                                                            data_temp, data_riceco2, idx_max, i,
                                                                            data_time, data_altitude,
                                                                            data_longitude))
            make_gif = input('Do you want create a gif (Y/n)?: ')
            if make_gif.lower() == 'y':
                print(repr(filenames))
                libf.create_gif(filenames)

        if view_mode == 13:
            filenames = []
            data_altitude = getdata(directory_store+filename, target='altitude')
            data_longitude = getdata(directory_store+filename, target='longitude')

            print('Select the latitude region (°N):')
            input_latitude_1 = int(input('   latitude 1: '))
            input_latitude_2 = int(input('   latitude 2: '))
            idx_lat_1 = (abs(data_latitude[:] - input_latitude_1)).argmin()
            idx_lat_2 = (abs(data_latitude[:] - input_latitude_2)).argmin()

            if idx_lat_1 > idx_lat_2:
                tmp = idx_lat_2
                idx_lat_2 = idx_lat_1
                idx_lat_1 = tmp

            data_target = flip(data_target, axis=2)
            idx_max = unravel_index(argmax(data_target[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=None),
                                    data_target[:, :, idx_lat_1:idx_lat_2 + 1, :].shape)
            idx_max = asarray(idx_max)
            idx_max[2] += idx_lat_1

            data_target = data_target[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]

            data_satuco2 = getdata(directory_store+filename, 'satuco2')
            data_satuco2 = data_satuco2[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
            data_satuco2 = flip(data_satuco2, axis=2)

            data_temp = getdata(directory_store+filename, 'temp')

            print('Temperature profile save in progres...')
            savetxt('temperature_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                                             data_latitude[idx_max[2]],
                                                                             data_longitude[idx_max[3]]),
                    c_[data_temp[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
            print('Done.')

            print('Pressure profile save in progres...')
            savetxt('pressure_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                                          data_latitude[idx_max[2]],
                                                                          data_longitude[idx_max[3]]),
                    c_[data_pressure[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
            print('Done.')

            print('Dust profile save in progres...')
            data_dust = getdata(directory_store+'concat_sols_dustq_S.nc', target='dustq')
            savetxt('dust_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                                      data_latitude[idx_max[2]],
                                                                      data_longitude[idx_max[3]]),
                    c_[data_dust[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
            print('Done.')

            data_temp = data_temp[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
            data_temp = flip(data_temp, axis=2)

            data_riceco2 = getdata(directory_store+filename, 'riceco2')
            data_riceco2 = data_riceco2[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
            data_riceco2 = flip(data_riceco2, axis=2)

            print('the maximum is at :' + str(data_time[idx_max[0]] * 24 % 24) + 'h local time.')
            for i in range(-3, 3):
                filenames.append(displays.display_cloud_evolution_latitude(data_target, data_satuco2,
                                                                           data_temp, data_riceco2, idx_max, i,
                                                                           data_time[idx_max[0] + i], data_altitude,
                                                                           data_latitude))

            make_gif = input('Do you want create a gif (Y/n)?: ')
            if make_gif.lower() == 'y':
                libf.create_gif(filenames)

    elif variable_target in ['temp']:
        view_mode = int(input('View (profile_evolution=1, compare_day_night=2): '))
        if view_mode == 1:
            data_pressure = getdata(directory_store+filename, target='pressure')
            data_longitude = getdata(directory_store+filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lon]

            idx_ls = data_target[:, :].argmin(axis=0)
            data_target = data_target[idx_ls[20] - 5:idx_ls[20] + 5, :]

            T_sat = libf.tcondco2(data_pressure, idx_ls=None, idx_lat=None, idx_lon=None)
            displays.display_temperature_profile_evolution(data_target, data_latitude, data_pressure, T_sat)

        if view_mode == 2:
            data_longitude = getdata(directory_store+filename, target='longitude')
            data_altitude = getdata(directory_store+filename, target='altitude')
            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target_day = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target_day = data_target_day[:, :, idx_lon]

            idx_ls = (abs(data_time[:] - 61)).argmin()
            data_target_day = mean(data_target_day[5:idx_ls:7, :], axis=0)

            filename_night = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_night))
            print('')

            data_target_night = getdata(directory_store+filename_night, 'temp')
            data_target_night = mean(data_target_night[4:idx_ls:7, :, idx_lat, idx_lon], axis=0)
            data_time_night = getdata(filename_night, target='Time')

            data_pressure = getdata(directory_store+'concat_sols_vars_S.nc', target='pressure')
            T_sat = libf.tcondco2(data_pressure, idx_ls=idx_ls, idx_lat=idx_lat, idx_lon=idx_lon)

            # local time: 2 4 6 8 ... 24
            temperature_stats = getdata(directory_store+'stats1_S_temp.nc', target='temp')
            temperature_stats_night = temperature_stats[0, :, idx_lat, idx_lon]
            temperature_stats_day = temperature_stats[7, :, idx_lat, idx_lon]
            print(temperature_stats_day.shape[0], data_target_day.shape[0])
            print(temperature_stats_night.shape[0], data_target_night.shape[0])
            displays.display_temperature_profile_day_night(data_target_day, data_target_night, T_sat, data_altitude[:],
                                                           temperature_stats_day, temperature_stats_night)

        if view_mode == 3:
            data_longitude = getdata(directory_store+filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lon]

            idx_ls_1 = (abs(data_time[:] - 259)).argmin()  # here time is in sols eqv to ls=120-150°
            idx_ls_2 = (abs(data_time[:] - 318)).argmin()
            data_target = data_target[idx_ls_1:idx_ls_2, :]
            shape_data_target = data_target.shape
            data_target = mean(data_target.reshape(-1, 59), axis=1)
            data_target = data_target.reshape(int(shape_data_target[0] / 59), shape_data_target[1])

            filename_2 = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')

            data_target_2 = getdata(directory_store+filename_2, variable_target)
            data_altitude = getdata(directory_store+filename_2, 'altitude')
            data_time_2 = getdata(directory_store+filename_2, 'Time')

            idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
            data_target_2 = data_target_2[idx_ls_1:idx_ls_2, :, idx_lat, idx_lon]
            shape_data_target_2 = data_target_2.shape
            data_target_2 = mean(data_target_2.reshape(-1, 59), axis=1)
            data_target_2 = data_target_2.reshape(int(shape_data_target_2[0] / 59), shape_data_target_2[1])

            # 00-4h + 6-18h + 20-22h
            data_target = concatenate((data_target_2[3:-1, :], data_target, data_target_2[1:3, :]))

            displays.display_altitude_localtime(data_target, data_altitude, unit='K',
                                                savename='temperature_altitude_localtime')

    elif variable_target in ['deltaT']:
        view_mode = int(input('View (altitude_vs_LT=1 (Fig 6, G-G2011), fig 7 = 2, fig 9 = 3):'))
        if view_mode == 1:
            data_longitude = getdata(filename, target='longitude')
            data_altitude = getdata(filename, target='altitude')

            idx_alt_min = (abs(data_altitude[:] / 1e3 - 40)).argmin() + 1
            idx_alt_max = (abs(data_altitude[:] / 1e3 - 100)).argmin() + 1

            data_altitude = data_altitude[idx_alt_min:idx_alt_max]

            idx_lat = (abs(data_latitude[:] - 0)).argmin() + 1
            data_target = data_target[:, idx_alt_min:idx_alt_max, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin() + 1
            data_target = data_target[:, :, idx_lon]

            idx_ls_1 = (abs(data_time[:] - 259)).argmin()  # here time is in sols eqv ls=120-150°
            idx_ls_2 = (abs(data_time[:] - 318)).argmin()
            data_target = data_target[idx_ls_1:idx_ls_2, :]
            shape_data_target = data_target.shape
            mean_data_target = zeros((int(shape_data_target[0]/59), shape_data_target[1]))
            print(mean_data_target.shape)
            for i in range(mean_data_target.shape[0]):
                mean_data_target[i,:] = mean(data_target[i::7,:], axis=0)

            filename_2 = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')

            data_target_2 = getdata(filename_2, variable_target)
            data_time_2 = getdata(filename_2, 'Time')

            idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
            data_target_2 = data_target_2[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, idx_lat, idx_lon]
            shape_data_target_2 = data_target_2.shape
            mean_data_target_2 = zeros((int(shape_data_target_2[0]/59), shape_data_target_2[1]))
            for i in range(mean_data_target_2.shape[0]):
                mean_data_target_2[i,:] = mean(data_target_2[i::7,:], axis=0)

            # 00-4h + 6-18h + 20-22h
            data_target = concatenate((mean_data_target_2[3:-1, :], mean_data_target, mean_data_target_2[1:3, :]))

            displays.display_altitude_localtime(data_target, data_altitude,
                                                title='Temperature - Tcond CO$_2$',
                                                unit='K',
                                                savename='difftemperature_altitude_localtime')

        if view_mode == 2:
            data_longitude = getdata(filename, target='longitude')
            data_altitude = getdata(filename, target='altitude')


            idx_alt_min = (abs(data_altitude[:]/1e3 - 40)).argmin() + 1
            idx_alt_max = (abs(data_altitude[:]/1e3 - 100)).argmin() + 1

            data_altitude = data_altitude[idx_alt_min:idx_alt_max]
            idx_ls_1 = (abs(data_time[:] - 0)).argmin()  # here time is in sols eqv ls=0-30°
            idx_ls_2 = (abs(data_time[:] - 51)).argmin()

            data_target = data_target[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, :, :]
            data_target = mean(data_target, axis=3)
            data_target = mean(data_target[5::7, :, :], axis=0) # LT 16 si LT_6_18_2
            displays.display_altitude_latitude(data_target, unit='K', title='Temperature -Tcond CO$_2$',
                                               data_altitude=data_altitude,
                                               data_latitude=data_latitude,
                                               savename='temperature_zonalmean_altitude_latitude_ls_0-30')

        if view_mode == 3:
            data_longitude = getdata(filename, target='longitude')
            data_altitude = getdata(filename, target='altitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin() + 1
            idx_alt_min = (abs(data_altitude[:]/1e3 - 40)).argmin() + 1
            idx_alt_max = (abs(data_altitude[:]/1e3 - 100)).argmin() + 1

            data_altitude = data_altitude[idx_alt_min:idx_alt_max]
            idx_ls_1 = (abs(data_time[:] - 0)).argmin()  # here time is in sols eqv ls=0-30°
            idx_ls_2 = (abs(data_time[:] - 51)).argmin()

            data_target = data_target[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, idx_lat, :]
            data_target = mean(data_target[5::7, :, :], axis=0) # LT 16 si LT_6_18_2
            displays.display_altitude_longitude(data_target, data_altitude, data_longitude,
                                                unit='K',
                                                title='Temperature -Tcond CO$_2$',
                                                savename='difftemperature_altitude_longitude_ls_0-30_LT_16H_lat_0N')

    elif variable_target in ['h2o_vap', 'q02']:
        data_altitude = getdata(directory_store + filename, target='altitude')
        try:
            data_pressure = getdata(directory_store + filename, target='pressure')
        except:
            data_pressure = getdata(directory_store + 'concat_sols_vars_S.nc', target='pressure')
        data_target, altitude_limit, zmin, zmax = libf.zonal_mean_column_density(data_target, data_pressure,
                                                                                 data_altitude, interp_time)

        displays.display_colonne(data_target, data_latitude, altitude_limit, zmin, zmax, ndx, axis_ls, interp_time,
                                 levels=logspace(-7,0,8), unit='kg/m$^2$', name=variable_target)

    elif variable_target in ['co2_conservation', 'Sols', 'Ls']:
        displays.display_1d(data_target)

    elif variable_target in ['satuco2']:
        shape_data_y = data_target[:, :, :, :].shape
        view_mode = int(input('View (alt-lat=1, max=2, max_day_night=3): '))

        if view_mode == 1:
            displays.display_altitude_latitude_satuco2(data_target, data_time, data_altitude, data_latitude)

        if view_mode == 2:
            print('Get max satuco2...')
            max_satu, x, y = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_alt = libf.extract_at_max_co2_ice(data_altitude, x, y, shape_data_y)

            print('Reshape and linearized data...')
            max_satu = libf.linearize_ls(max_satu, shape_data_y[0], shape_data_y[2], interp_time)
            max_alt = libf.linearize_ls(max_alt, shape_data_y[0], shape_data_y[2], interp_time)
            displays.display_max_lon_alt_satuco2(max_satu, data_latitude, axis_ls, ndx, max_alt)

        if view_mode == 3:
            filename_night = getfilename(files)
            data_target_night = getdata(directory_store+filename_night, 'satuco2')
            data_altitude = getdata(directory_store+filename_night, 'altitude')

            idx_altitude_max = (abs(data_altitude[:]/1e3 - 90)).argmin() + 1
            data_target = data_target[:, :idx_altitude_max, :, :]
            data_target_night = data_target_night[:, :idx_altitude_max, :, :]

            print('Get max satuco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_night, extrema='max')

            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
                    max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
                    if max_satu_day[i, j] < 1.0:
                        max_alt_day[i, j] = None
                        max_satu_day[i, j] = None

                    if max_satu_night[i, j] < 1.0:
                        max_alt_night[i, j] = None
                        max_satu_night[i, j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='',
                                               title='Max saturation of CO$_2$ ice',
                                               savename='max_satuco2_day_night.png')

        if view_mode == 4:
            data_longitude = getdata(directory_store+filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lon]

            idx_ls_1 = (abs(data_time[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time[:] - 318)).argmin()
            data_target = data_target[idx_ls_1:idx_ls_2, :]
            shape_data_target = data_target.shape
            data_target = mean(data_target.reshape(-1, 59), axis=1)
            data_target = data_target.reshape(int(shape_data_target[0] / 59), shape_data_target[1])

            filename_2 = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')

            data_target_2 = getdata(directory_store+filename_2, variable_target)
            data_altitude = getdata(directory_store+filename_2, 'altitude')
            data_time_2 = getdata(directory_store+filename_2, 'Time')

            idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
            data_target_2 = data_target_2[idx_ls_1:idx_ls_2, :, idx_lat, idx_lon]
            shape_data_target_2 = data_target_2.shape
            data_target_2 = mean(data_target_2.reshape(-1, 59), axis=1)
            data_target_2 = data_target_2.reshape(int(shape_data_target_2[0] / 59), shape_data_target_2[1])

            # 00-4h + 6-18h + 20-22h
            data_target = concatenate((data_target_2[3:-1, :], data_target, data_target_2[1:3, :]))

            displays.display_altitude_localtime(data_target, data_altitude, unit='',
                                                savename='saturation_altitude_localtime')

        if view_mode == 5:
            data_altitude = getdata(directory_store+filename, target='altitude')
            data_longitude = getdata(directory_store+filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_ls_1 = (abs(data_time[:] - 0)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time[:] - 15)).argmin()
            data_target = data_target[idx_ls_1:idx_ls_2, :, :]

            idx_local_time = (abs(data_time[:] * 24 - 16)).argmin()  # to get all value at this local time
            data_target = data_target[idx_local_time::7, :, :]

            data_target = mean(data_target, axis=0)

            displays.display_altitude_longitude(data_target, data_altitude, data_longitude, unit='',
                                                savename='saturation_altitude_longitude')

        if view_mode == 6:

            if path.exists('thickness_layer_satuco2_polar_region.dat'):
                nbin = arange(0, 365, 5).shape[0]
                data_icelayer = zeros((2, nbin))
                data_icelayer_std = zeros((2, nbin))

                data = loadtxt('thickness_layer_satuco2_polar_region.dat', usecols=[1, 2, 3, 4])
                data_icelayer[0, :] = data[:, 0]
                data_icelayer_std[0, :] = data[:, 1]
                data_icelayer[1, :] = data[:, 2]
                data_icelayer_std[1, :] = data[:, 3]
            else:
                data_altitude = getdata(directory_store+filename, 'altitude')

                idx_lat_N = (abs(data_latitude[:] - 60)).argmin()
                idx_lat_S = (abs(data_latitude[:] + 60)).argmin()

                data_target_north = data_target[:, :, idx_lat_N:, :]  # north pole
                data_target_south = data_target[:, :, :idx_lat_S + 1, :]  # south pole
                print(data_latitude[idx_lat_N:])
                print(data_latitude[:idx_lat_S + 1])

                # bin ls to 5°
                if max(data_time) <= 360:
                    ls_bin = arange(0, 365, 5)
                else:
                    # sols binned at 5° Ls
                    ls_bin = array(
                        [0, 10, 20, 30, 41, 51, 61, 73, 83, 94, 105, 116, 126, 139, 150, 160, 171, 183, 193.47,
                         205, 215, 226, 236, 248, 259, 269, 279, 289, 299, 309, 317, 327, 337, 347, 355, 364,
                         371.99, 381, 390, 397, 406, 415, 422, 430, 437, 447, 457, 467, 470, 477, 485, 493, 500,
                         507, 514.76, 523, 533, 539, 547, 555, 563, 571, 580, 587, 597, 605, 613, 623, 632, 641,
                         650, 660, 669])
                nbin = ls_bin.shape[0]
                data_icelayer = zeros((2, nbin))
                data_icelayer_std = zeros((2, nbin))

                for bin in range(nbin - 1):
                    print(bin, nbin - 1)
                    idx = (abs(data_time[:] - ls_bin[bin])).argmin()
                    idx2 = (abs(data_time[:] - ls_bin[bin + 1])).argmin()
                    data_binned_north = data_target_north[idx:idx2, :, :, :]
                    data_binned_south = data_target_south[idx:idx2, :, :, :]

                    tmp_north = array([])
                    tmp_south = array([])

                    for i in range(data_binned_north.shape[0]):
                        for longitude in range(data_target_north.shape[3]):
                            for latitude_north in range(data_target_north.shape[2]):
                                for l in range(data_target_north.shape[1]):
                                    if data_target_north[i, l, latitude_north, longitude] >= 1:
                                        idx_min_north = l
                                        for l2 in range(l + 1, data_target_north.shape[1]):
                                            if data_target_north[i, l2, latitude_north, longitude] < 1:
                                                idx_max_north = l2 - 1
                                                tmp_north = append(tmp_north, data_altitude[idx_max_north] -
                                                                   data_altitude[idx_min_north])
                                                break
                                        break

                            for latitude_south in range(data_target_south.shape[2]):
                                for l in range(data_target_south.shape[1]):
                                    if data_target_south[i, l, latitude_south, longitude] >= 1:
                                        idx_min_south = l
                                        for l2 in range(l + 1, data_target_south.shape[1]):
                                            if data_target_south[i, l2, latitude_south, longitude] < 1:
                                                idx_max_south = l2 - 1
                                                tmp_south = append(tmp_south, data_altitude[idx_max_south] -
                                                                   data_altitude[idx_min_south])
                                                break
                                        break
                    if tmp_north.size != 0:
                        data_icelayer[0, bin] = mean(tmp_north)
                        data_icelayer_std[0, bin] = std(tmp_north)
                    else:
                        data_icelayer[0, bin] = 0
                        data_icelayer_std[0, bin] = 0

                    if tmp_south.size != 0:
                        data_icelayer[1, bin] = mean(tmp_south)
                        data_icelayer_std[1, bin] = std(tmp_south)
                    else:
                        data_icelayer[1, bin] = 0
                        data_icelayer_std[1, bin] = 0

                    nbp_north = data_binned_north.shape[0] * data_binned_north.shape[2] * data_binned_north.shape[3]
                    nbp_south = data_binned_south.shape[0] * data_binned_south.shape[2] * data_binned_south.shape[3]
                    print(tmp_north.size, tmp_north.size / nbp_north, data_icelayer[0, bin], data_icelayer_std[0, bin])
                    print(tmp_south.size, tmp_south.size / nbp_south, data_icelayer[1, bin], data_icelayer_std[1, bin])

                savetxt('thickness_layer_satuco2_polar_region.dat', c_[arange(0, 365, 5),
                                                                       data_icelayer[0, :], data_icelayer_std[0, :],
                                                                       data_icelayer[1, :], data_icelayer_std[1, :]])
            displays.display_thickness_co2ice_atm_layer(data_icelayer,
                                                        data_icelayer_std,
                                                        savename='satuco2_thickness_polar_region.png')

        if view_mode == 7:
            idx_lat_north = (abs(data_latitude[:] - 50)).argmin() + 1
            idx_lat_eq = (abs(data_latitude[:] - 0)).argmin() + 1
            idx_lat_south = (abs(data_latitude[:] + 60)).argmin() + 1

            data_satuco2_north = data_target[:, :, -idx_lat_north, :]
            ind_north = unravel_index(argmax(data_satuco2_north.reshape(data_satuco2_north.shape[0], -1), axis=1),
                                      data_satuco2_north.shape[1:3])
            data_satuco2_north = [data_satuco2_north[i, :, ind_north[1][i]] for i in range(data_satuco2_north.shape[0])]
            data_satuco2_north = asarray(data_satuco2_north)

            data_satuco2_eq = data_target[:, :, -idx_lat_eq, :]
            ind_eq = unravel_index(argmax(data_satuco2_eq.reshape(data_satuco2_eq.shape[0], -1), axis=1),
                                   data_satuco2_eq.shape[1:3])
            data_satuco2_eq = [data_satuco2_eq[i, :, ind_eq[1][i]] for i in range(data_satuco2_eq.shape[0])]
            data_satuco2_eq = asarray(data_satuco2_eq)

            data_satuco2_south = data_target[:, :, -idx_lat_south, :]
            ind_south = unravel_index(argmax(data_satuco2_south.reshape(data_satuco2_south.shape[0], -1), axis=1),
                                      data_satuco2_south.shape[1:3])
            data_satuco2_south = [data_satuco2_south[i, :, ind_south[1][i]] for i in range(data_satuco2_south.shape[0])]
            data_satuco2_south = asarray(data_satuco2_south)

            del data_target

            data_co2ice = getdata(directory_store+filename, target='co2_ice')
            data_co2ice = libf.correction_value_co2_ice(data_co2ice[:])

            data_co2ice_north = data_co2ice[:, :, -idx_lat_north, :]
            data_co2ice_north = [data_co2ice_north[i, :, ind_north[1][i]] for i in range(data_co2ice_north.shape[0])]
            data_co2ice_north = asarray(data_co2ice_north)

            data_co2ice_eq = data_co2ice[:, :, -idx_lat_eq, :]
            data_co2ice_eq = [data_co2ice_eq[i, :, ind_eq[1][i]] for i in range(data_co2ice_eq.shape[0])]
            data_co2ice_eq = asarray(data_co2ice_eq)

            data_co2ice_south = data_co2ice[:, :, -idx_lat_south, :]
            data_co2ice_south = [data_co2ice_south[i, :, ind_south[1][i]] for i in range(data_co2ice_south.shape[0])]
            data_co2ice_south = asarray(data_co2ice_south)

            del data_co2ice, ind_north, ind_eq, ind_south

            data_altitude = getdata(directory_store+filename, target='altitude')[:]

            time_grid_ls = libf.convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]
            data_satuco2_north_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_eq_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_south_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_north_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_eq_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_south_binned = zeros((nb_bin, data_altitude.shape[0]))

            for i in range(nb_bin-1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i+1])).argmin() + 1

                data_satuco2_north_binned[i,:] = mean(data_satuco2_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_eq_binned[i,:] = mean(data_satuco2_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_south_binned[i,:] = mean(data_satuco2_south[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_north_binned[i,:] = mean(data_co2ice_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_eq_binned[i,:] = mean(data_co2ice_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_south_binned[i,:] = mean(data_co2ice_south[idx_ls_1:idx_ls_2, :], axis=0)

            del data_satuco2_north, data_satuco2_eq, data_satuco2_south, \
                data_co2ice_north, data_co2ice_eq, data_co2ice_south

            ndx, axis_ls = libf.get_ls_index(time_grid_ls)
            displays.display_saturation_and_co2_ice(data_satuco2_north_binned, data_satuco2_eq_binned,
                                                    data_satuco2_south_binned, data_co2ice_north_binned,
                                                    data_co2ice_eq_binned, data_co2ice_south_binned,
                                                    data_altitude, ndx, axis_ls)
        if view_mode == 8:
            data_altitude = getdata(directory_store+filename, target='altitude')[:]
            idx_lat_1 = (abs(data_latitude[:] + 50)).argmin()
            idx_lat_2 = (abs(data_latitude[:] + 60)).argmin()

#            idx_lat_1 = (abs(data_latitude[:] - 15)).argmin()
#            idx_lat_2 = (abs(data_latitude[:] + 15)).argmin()

            if idx_lat_1 > idx_lat_2:
                tmp = idx_lat_1
                idx_lat_1 = idx_lat_2
                idx_lat_2 = tmp

            data_satuco2_day = mean(data_target[:, :, idx_lat_1:idx_lat_2+1, :], axis=3)
            data_satuco2_day = mean(data_satuco2_day, axis=2)

            print('-----------')
            print('Select satuco2 night file')
            filename_2 = getfilename(files)
            data_satuco2_night = getdata(directory_store+filename_2, target='satuco2')
            data_satuco2_night = mean(data_satuco2_night[:, :, idx_lat_1:idx_lat_2+1, :], axis=3)
            data_satuco2_night = mean(data_satuco2_night, axis=2)

            print('-----------')
            print('Select co2_ice day file')
            filename_3 = getfilename(files)
            data_co2ice_day = getdata(directory_store+filename_3, target='co2_ice')
            data_co2ice_day = libf.correction_value_co2_ice(data_co2ice_day[:])
            data_co2ice_day = mean(data_co2ice_day[:, :, idx_lat_1:idx_lat_2+1, :], axis=3)
            data_co2ice_day = mean(data_co2ice_day, axis=2)

            print('-----------')
            print('Select co2_ice night file')
            filename_4 = getfilename(files)
            data_co2ice_night = getdata(directory_store+filename_4, target='co2_ice')
            data_co2ice_night = libf.correction_value_co2_ice(data_co2ice_night[:])
            data_co2ice_night = mean(data_co2ice_night[:, :, idx_lat_1:idx_lat_2+1, :], axis=3)
            data_co2ice_night = mean(data_co2ice_night, axis=2)

            time_grid_ls = libf.convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]
            data_satuco2_day_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_night_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_day_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_night_binned = zeros((nb_bin, data_altitude.shape[0]))

            for i in range(nb_bin-1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i+1])).argmin() + 1

                data_satuco2_day_binned[i,:] = mean(data_satuco2_day[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_night_binned[i,:] = mean(data_satuco2_night[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_day_binned[i,:] = mean(data_co2ice_day[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_night_binned[i,:] = mean(data_co2ice_night[idx_ls_1:idx_ls_2, :], axis=0)

            del data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night

            ndx, axis_ls = libf.get_ls_index(time_grid_ls)

            displays.saturation_zonal_mean_day_night(data_satuco2_day_binned, data_satuco2_night_binned,
                                                     data_co2ice_day_binned, data_co2ice_night_binned, data_altitude,
                                                     ndx, axis_ls,
                                                     title='Zonal mean of CO2 saturation/mmr ['+
                                                     str(data_latitude[::-1][idx_lat_1])+':'+
                                                     str(data_latitude[::-1][idx_lat_2])+']°N',
                                                     savename='saturationco2_co2ice_zonalmean_'+
                                                              str(data_latitude[::-1][idx_lat_1])+'N_'+
                                                              str(data_latitude[::-1][idx_lat_2])+'N_day_night')

    elif variable_target in ['riceco2']:
        view_mode = int(input('View (profile=1, max_day_night=2): '))

        if view_mode == 1:
            data_altitude = getdata(directory_store+filename, target='altitude')
            displays.display_riceco2_profile_pole(data_target, data_latitude, data_time, data_altitude)

        if view_mode == 2:
            filename_2 = getfilename(files)
            data_target_2 = getdata(directory_store+filename_2, 'riceco2')
            data_altitude = getdata(directory_store+filename_2, 'altitude')

            print('Get max riceco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')

            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
                    max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
                    if max_satu_day[i, j] == 0:
                        max_alt_day[i, j] = None
                        max_satu_day[i, j] = None

                    if max_satu_night[i, j] == 0:
                        max_alt_night[i, j] = None
                        max_satu_night[i, j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='µm',
                                               title='Max radius of CO$_2$ ice',
                                               savename='max_riceco2_day_night.png')

        if view_mode == 3:
            data_altitude = getdata(directory_store+filename, target='altitude')
            data_altitude = data_altitude[:] / 1e3
            data_ccnNco2 = getdata(directory_store+filename, target='ccnNco2')
            filename_2 = getfilename(files)
            data_rho = getdata(directory_store+filename_2, target='rho')

            data_target = mean(data_target[:, :, :, :], axis=3)
            data_ccnNco2 = mean(data_ccnNco2[:, :, :, :], axis=3)
            data_rho = mean(data_rho[:, :, :, :], axis=3)

            N_reflect = 2e-8 * power(data_target * 1e6, -2)
            N_part = data_rho * data_ccnNco2
            nb_time = data_target.shape[0]
            nb_lat = data_target.shape[2]
            nb_alt = data_target.shape[1]
            del [data_target, data_ccnNco2, data_rho]

            top_cloud = zeros((nb_time, nb_lat))
            for t in range(nb_time):
                for lat in range(nb_lat):
                    for alt in range(nb_alt - 1, -1, -1):
                        if N_part[t, alt, lat] >= N_reflect[t, alt, lat]:
                            top_cloud[t, lat] = data_altitude[alt]
                            break

            idx_lat_1 = (abs(data_latitude[:] - 40)).argmin()
            idx_lat_2 = (abs(data_latitude[:] + 40)).argmin()

            if idx_lat_1 > idx_lat_2:
                tmp = idx_lat_1
                idx_lat_1 = idx_lat_2
                idx_lat_2 = tmp

            top_cloud[:, idx_lat_1:idx_lat_2 + 1] = None

            displays.topcloud_altitude(top_cloud, data_latitude, data_altitude, ndx, axis_ls)

    elif variable_target in ['ccnNco2']:
        view_mode = int(input('View (max_day_night=1): '))
        if view_mode == 1:

            filename_2 = getfilename(files)
            data_target_2 = getdata(directory_store+filename_2, 'ccnNco2')
            data_altitude = getdata(directory_store+filename_2, 'altitude')

            print('Get max ccnNco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')
            print(min(max_satu_day), max(max_satu_day))
            print(min(max_satu_night), max(max_satu_night))
            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
                    max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
                    if max_satu_day[i, j] < 1:
                        max_alt_day[i, j] = None
                        max_satu_day[i, j] = None

                    if max_satu_night[i, j] < 1:
                        max_alt_night[i, j] = None
                        max_satu_night[i, j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='#/kg',
                                               title='Max CCN number',
                                               savename='max_ccnNco2_day_night.png')

    elif variable_target in ['tau1mic']:
        view_mode = int(input('View (zonal_mean lat-ls=1): '))
        if view_mode == 1:
            zonal_mean = mean(data_target[:, :, :], axis=2)
            zonal_mean = flip(zonal_mean.T, axis=0)

            del data_target

            for i in range(zonal_mean.shape[0]):
                dim2 = where(zonal_mean[i, :] == 0)
                zonal_mean[i, dim2] = None
            if lslin:
                print('Perform linearization in progress')
                zonal_mean = libf.linearize_ls(zonal_mean, data_time.shape[0], data_latitude.shape[0], interp_time)

            displays.display_zonal_mean(zonal_mean, data_latitude, ndx, axis_ls, levels=arange(0,0.014,0.002),
                                        title=variable_target, units='')

    elif variable_target in ['tau']: #Time, latitude, longitude
        zonal_mean = mean(data_target[:, :, :], axis=2)
        zonal_mean = flip(zonal_mean.T, axis=0)

        del data_target

        for i in range(zonal_mean.shape[0]):
            dim2 = where(zonal_mean[i, :] == 0)
            zonal_mean[i, dim2] = None

        if lslin:
            print('Perform linearization in progress')
            zonal_mean = libf.linearize_ls(zonal_mean, data_time.shape[0], data_latitude.shape[0], interp_time)
        displays.display_zonal_mean(zonal_mean, data_latitude, ndx, axis_ls, levels=None, title=variable_target,
                                    units='SI')

    else:
        print('Variable not used for the moment')


if '__main__' == __name__:
    main()
