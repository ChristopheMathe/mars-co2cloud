#!/bin/bash python3
from packages.ncdump import *
from packages.lib_function import *
from packages.DataProcessed import *
from packages.displays import *
from os import listdir, path

from numpy import mean, abs, min, max, zeros, where, concatenate, flip, arange, unravel_index, argmax, array, \
    std, savetxt, c_, append, loadtxt, asarray, power, logspace, random


def main():
    # TODO: si présence d'argument alors allez directement au processing et displays

    files = listdir('.')

    directory_store = [x for x in files if 'occigen' in x][0] + '/'

    if directory_store is None:
        directory_store = ''
    else:
        files = listdir(directory_store)

    filename = getfilename(files)
    filename = directory_store + filename

    data_target = getdata(filename)
    name_target = data_target.name

    #    print('Create linear grid time...')
    #    data_time = ncdump.getdata(directory_store+filename, target="Time")
    #
    #    if max(data_time) <= 360:
    #        interp_time, axis_ls, ndx = libf.linear_grid_ls(data_time)
    #        print('Check if you correctly linearize data on ls')
    #        lslin = True
    #    else:
    #        ndx, axis_ls = libf.get_ls_index(data_time[:])
    #        interp_time = 0
    #        lslin = False

    if name_target in ['co2_ice', 'h2o_ice', 'q01']:  # q01 = h2o_ice
        # correct very low values of co2/h2o mmr < 1e-13
        print('Correction value...')
        data_target = correction_value_co2_ice(data_target[:, :, :, :], threshold=1e-13)

        print('What do you wanna do?')
        print('     1: maximum in altitude and longitude, with others variables at these places (fig: lat-ls)')
        print('     2: zonal mean column density (fig: lat-ls)')
        print('     3: extract profile (fig: alt-X)')
        print('     4: maximum during day and night with altitude corresponding (fig: lat-ls)')
        print('     5: profile in function of localtime (fig: alt-localtime)')
        print('     6: layer ice thickness (fig: thickness-lat)')
        print('     7: polar cloud distribution (fig: #clouds-lat)')
        print('     8: cloud evolution with satuco2/temperature/radius (fig: alt-lat, gif)')
        if  name_target in ['h2o_ice', 'q01']:
            print('     9: h2o_ice profile with co2_ice presence (fig: alt-ls)')
        print('')
        view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            max_mmr, max_temp, max_satu, max_radius, max_ccnN, max_alt = \
                vars_max_value_with_others(data_target)

            print('Display:')
            display_max_lon_alt(name_target, max_mmr, max_alt, max_temp, max_satu, max_radius, max_ccnN,
                                         unit='kg/kg')

        elif view_mode == 2:
            print('Processing data:')
            data_target, altitude_limit, zmin, zmax = vars_zonal_mean_column_density(filename, data_target)

            print('Display:')
            display_colonne(filename, data_target, altitude_limit, zmin, zmax, levels=logspace(-13, 2, 16),
                                     unit='kg/m$^2$', name=name_target)

        elif view_mode == 3:
            print('Processing data:')
            data_profile = vars_select_profile(data_target)

            print('Display:')
            display_vertical_profile(data_profile, unit='kg/kg',
                                              savename='altitude_latitude_co2_ice_northpole.png')

        elif view_mode == 4:
            print('Processing data:')
            max_day, max_night, max_alt_day, max_alt_night = max_value_day_night_with_altitude(data_target)

            print('Display:')
            display_lat_ls_maxsatuco2(max_day, max_night, max_alt_day, max_alt_night, unit='kg/kg',
                                               title='Max vmr of CO$_2$ ice', savename='max_co2_ice_day_night.png')

        if view_mode == 5:
            print('Processing data:')
            data, data_altitude = concate_localtime(data_target)

            print('Display:')
            display_altitude_localtime(data_target, data_altitude, unit='kg/kg',
                                                savename='co2_ice_altitude_localtime')

        if view_mode == 6:
            print('Processing data:')
            data_icelayer = thickness_co2ice_atm_layer(data_target)

            print('Display:')
            display_thickness_co2ice_atm_layer(data_icelayer)

        if view_mode == 7:
            print('Processing data:')
            distribution_north, distribution_south = polar_cloud_distribution()

            print('Display:')
            display_distribution_altitude_latitude_polar(distribution_north, distribution_south,
                                                                  savename='distribution_polar_clouds')

        if view_mode == 8:
            print('Processing data:')
            data_target, data_satuco2, data_temp, data_riceco2, idx_max = cloud_evolution(data_target)

            print('Display:')
            display_cloud_evolution_latitude(data_target, data_satuco2, data_temp, data_riceco2, idx_max)

        if view_mode == 9:
            print('Processing data:')
            data_target, data_co2_ice, latitude_selected = h2o_ice_alt_ls_with_co2_ice(filename, data_target)

            print('Display:')
            display_alt_ls(filename, data_target, data_co2_ice, levels=logspace(-13, 2, 16),
                           title='Zonal mean of H2O ice mmr and CO2 ice mmr (black), at '+str(int(
                               latitude_selected))+'°N',
                           savename='h2o_ice_zonalmean_with_co2_ice_'+str(int(latitude_selected))+'N')

    elif name_target in ['temp']:
        print('What do you wanna do?')
        print('     1: extract profile (fig: alt-K)')
        print('     2: mean profile during day and night (fig: alt-K)')
        print('')
        view_mode = int(input('Select number:'))

        view_mode = int(input('View (profile_evolution=1, compare_day_night=2): '))
        if view_mode == 1:
            data_pressure = getdata(directory_store + filename, target='pressure')
            data_longitude = getdata(directory_store + filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lon]

            idx_ls = data_target[:, :].argmin(axis=0)
            data_target = data_target[idx_ls[20] - 5:idx_ls[20] + 5, :]

            T_sat = libf.tcondco2(data_pressure, idx_ls=None, idx_lat=None, idx_lon=None)
            displays.display_temperature_profile_evolution(data_target, data_latitude, data_pressure, T_sat)

        if view_mode == 2:
            data_longitude = getdata(directory_store + filename, target='longitude')
            data_altitude = getdata(directory_store + filename, target='altitude')
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

            data_target_night = getdata(directory_store + filename_night, 'temp')
            data_target_night = mean(data_target_night[4:idx_ls:7, :, idx_lat, idx_lon], axis=0)
            data_time_night = getdata(filename_night, target='Time')

            data_pressure = getdata(directory_store + 'concat_sols_vars_S.nc', target='pressure')
            T_sat = libf.tcondco2(data_pressure, idx_ls=idx_ls, idx_lat=idx_lat, idx_lon=idx_lon)

            # local time: 2 4 6 8 ... 24
            temperature_stats = getdata(directory_store + 'stats1_S_temp.nc', target='temp')
            temperature_stats_night = temperature_stats[0, :, idx_lat, idx_lon]
            temperature_stats_day = temperature_stats[7, :, idx_lat, idx_lon]
            displays.display_temperature_profile_day_night(data_target_day, data_target_night, T_sat, data_altitude[:],
                                                           temperature_stats_day, temperature_stats_night)

        if view_mode == 3:
            data_longitude = getdata(directory_store + filename, target='longitude')

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

            data_target_2 = getdata(directory_store + filename_2, name_target)
            data_altitude = getdata(directory_store + filename_2, 'altitude')
            data_time_2 = getdata(directory_store + filename_2, 'Time')

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

    elif name_target in ['deltaT']:
        view_mode = int(input('View (altitude_vs_LT=1 (Fig 6, G-G2011), fig 7 = 2, fig 9 = 3):'))
        if view_mode == 1:
            data_longitude = getdata(directory_store + filename, target='longitude')
            data_altitude = getdata(directory_store + filename, target='altitude')

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
            mean_data_target = zeros((int(shape_data_target[0] / 59), shape_data_target[1]))

            for i in range(mean_data_target.shape[0]):
                mean_data_target[i, :] = mean(data_target[i::7, :], axis=0)

            filename_2 = getfilename(files)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')

            data_target_2 = getdata(directory_store + filename_2, name_target)
            data_time_2 = getdata(directory_store + filename_2, 'Time')

            idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
            idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
            data_target_2 = data_target_2[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, idx_lat, idx_lon]
            shape_data_target_2 = data_target_2.shape
            mean_data_target_2 = zeros((int(shape_data_target_2[0] / 59), shape_data_target_2[1]))
            for i in range(mean_data_target_2.shape[0]):
                mean_data_target_2[i, :] = mean(data_target_2[i::7, :], axis=0)

            # 00-4h + 6-18h + 20-22h
            data_target = concatenate((mean_data_target_2[3:-1, :], mean_data_target, mean_data_target_2[1:3, :]))

            displays.display_altitude_localtime(data_target, data_altitude,
                                                title='Temperature - Tcond CO$_2$',
                                                unit='K',
                                                savename='difftemperature_altitude_localtime')

        if view_mode == 2:
            data_longitude = getdata(directory_store + filename, target='longitude')
            data_altitude = getdata(directory_store + filename, target='altitude')

            idx_alt_min = (abs(data_altitude[:] / 1e3 - 40)).argmin() + 1
            idx_alt_max = (abs(data_altitude[:] / 1e3 - 100)).argmin() + 1

            data_altitude = data_altitude[idx_alt_min:idx_alt_max]
            idx_ls_1 = (abs(data_time[:] - 0)).argmin()  # here time is in sols eqv ls=0-30°
            idx_ls_2 = (abs(data_time[:] - 51)).argmin()

            data_target = data_target[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, :, :]
            data_target = mean(data_target, axis=3)
            data_target = mean(data_target[5::7, :, :], axis=0)  # LT 16 si LT_6_18_2
            displays.display_altitude_latitude(data_target, unit='K', title='Temperature -Tcond CO$_2$',
                                               data_altitude=data_altitude,
                                               data_latitude=data_latitude,
                                               savename='temperature_zonalmean_altitude_latitude_ls_0-30')

        if view_mode == 3:
            data_longitude = getdata(directory_store + filename, target='longitude')
            data_altitude = getdata(directory_store + filename, target='altitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin() + 1
            idx_alt_min = (abs(data_altitude[:] / 1e3 - 40)).argmin() + 1
            idx_alt_max = (abs(data_altitude[:] / 1e3 - 100)).argmin() + 1

            data_altitude = data_altitude[idx_alt_min:idx_alt_max]
            idx_ls_1 = (abs(data_time[:] - 0)).argmin()  # here time is in sols eqv ls=0-30°
            idx_ls_2 = (abs(data_time[:] - 51)).argmin()

            data_target = data_target[idx_ls_1:idx_ls_2, idx_alt_min:idx_alt_max, idx_lat, :]
            data_target = mean(data_target[5::7, :, :], axis=0)  # LT 16 si LT_6_18_2
            displays.display_altitude_longitude(data_target, data_altitude, data_longitude,
                                                unit='K',
                                                title='Temperature -Tcond CO$_2$',
                                                savename='difftemperature_altitude_longitude_ls_0-30_LT_16H_lat_0N')

    elif name_target in ['h2o_vap', 'q02']:
        data_altitude = getdata(directory_store + filename, target='altitude')
        try:
            data_pressure = getdata(directory_store + filename, target='pressure')
        except:
            data_pressure = getdata(directory_store + 'concat_sols_vars_S.nc', target='pressure')
        data_target, altitude_limit, zmin, zmax = libf.zonal_mean_column_density(data_target, data_pressure,
                                                                                 data_altitude, interp_time)

        displays.display_colonne(data_target, data_time, data_latitude, altitude_limit, zmin, zmax, ndx, axis_ls,
                                 interp_time,
                                 # levels=logspace(-7,0,8),
                                 # unit='kg/m$^2$',
                                 levels=arange(13) * 10,
                                 unit='pr.µm',
                                 name=name_target)

    elif name_target in ['satuco2']:
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
            data_target_night = getdata(directory_store + filename_night, 'satuco2')
            data_altitude = getdata(directory_store + filename_night, 'altitude')

            idx_altitude_max = (abs(data_altitude[:] / 1e3 - 90)).argmin() + 1
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
            data_longitude = getdata(directory_store + filename, target='longitude')

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

            data_target_2 = getdata(directory_store + filename_2, name_target)
            data_altitude = getdata(directory_store + filename_2, 'altitude')
            data_time_2 = getdata(directory_store + filename_2, 'Time')

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
            data_altitude = getdata(directory_store + filename, target='altitude')
            data_longitude = getdata(directory_store + filename, target='longitude')

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
                data_altitude = getdata(directory_store + filename, 'altitude')

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

            data_co2ice = getdata(directory_store + filename, target='co2_ice')
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

            data_altitude = getdata(directory_store + filename, target='altitude')[:]

            time_grid_ls = libf.convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]
            data_satuco2_north_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_eq_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_south_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_north_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_eq_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_south_binned = zeros((nb_bin, data_altitude.shape[0]))

            for i in range(nb_bin - 1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i + 1])).argmin() + 1

                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_eq_binned[i, :] = mean(data_satuco2_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_eq_binned[i, :] = mean(data_co2ice_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[idx_ls_1:idx_ls_2, :], axis=0)

            del data_satuco2_north, data_satuco2_eq, data_satuco2_south, \
                data_co2ice_north, data_co2ice_eq, data_co2ice_south

            ndx, axis_ls = libf.get_ls_index(time_grid_ls)
            displays.display_saturation_and_co2_ice(data_satuco2_north_binned, data_satuco2_eq_binned,
                                                    data_satuco2_south_binned, data_co2ice_north_binned,
                                                    data_co2ice_eq_binned, data_co2ice_south_binned,
                                                    data_altitude, ndx, axis_ls)
        if view_mode == 8:
            data_altitude = getdata(directory_store + filename, target='altitude')[:]
            idx_lat_1 = (abs(data_latitude[:] + 50)).argmin()
            idx_lat_2 = (abs(data_latitude[:] + 60)).argmin()

            #            idx_lat_1 = (abs(data_latitude[:] - 15)).argmin()
            #            idx_lat_2 = (abs(data_latitude[:] + 15)).argmin()

            if idx_lat_1 > idx_lat_2:
                tmp = idx_lat_1
                idx_lat_1 = idx_lat_2
                idx_lat_2 = tmp

            data_satuco2_day = mean(data_target[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=3)
            data_satuco2_day = mean(data_satuco2_day, axis=2)

            print('-----------')
            print('Select satuco2 night file')
            filename_2 = getfilename(files)
            data_satuco2_night = getdata(directory_store + filename_2, target='satuco2')
            data_satuco2_night = mean(data_satuco2_night[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=3)
            data_satuco2_night = mean(data_satuco2_night, axis=2)

            print('-----------')
            print('Select co2_ice day file')
            filename_3 = getfilename(files)
            data_co2ice_day = getdata(directory_store + filename_3, target='co2_ice')
            data_co2ice_day = libf.correction_value_co2_ice(data_co2ice_day[:])
            data_co2ice_day = mean(data_co2ice_day[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=3)
            data_co2ice_day = mean(data_co2ice_day, axis=2)

            print('-----------')
            print('Select co2_ice night file')
            filename_4 = getfilename(files)
            data_co2ice_night = getdata(directory_store + filename_4, target='co2_ice')
            data_co2ice_night = libf.correction_value_co2_ice(data_co2ice_night[:])
            data_co2ice_night = mean(data_co2ice_night[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=3)
            data_co2ice_night = mean(data_co2ice_night, axis=2)

            time_grid_ls = libf.convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]
            data_satuco2_day_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_satuco2_night_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_day_binned = zeros((nb_bin, data_altitude.shape[0]))
            data_co2ice_night_binned = zeros((nb_bin, data_altitude.shape[0]))

            for i in range(nb_bin - 1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i + 1])).argmin() + 1

                data_satuco2_day_binned[i, :] = mean(data_satuco2_day[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_night_binned[i, :] = mean(data_satuco2_night[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_day_binned[i, :] = mean(data_co2ice_day[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_night_binned[i, :] = mean(data_co2ice_night[idx_ls_1:idx_ls_2, :], axis=0)

            del data_satuco2_day, data_satuco2_night, data_co2ice_day, data_co2ice_night

            ndx, axis_ls = libf.get_ls_index(time_grid_ls)

            displays.saturation_zonal_mean_day_night(data_satuco2_day_binned, data_satuco2_night_binned,
                                                     data_co2ice_day_binned, data_co2ice_night_binned, data_altitude,
                                                     ndx, axis_ls,
                                                     title='Zonal mean of CO2 saturation/mmr [' +
                                                           str(data_latitude[::-1][idx_lat_1]) + ':' +
                                                           str(data_latitude[::-1][idx_lat_2]) + ']°N',
                                                     savename='saturationco2_co2ice_zonalmean_' +
                                                              str(data_latitude[::-1][idx_lat_1]) + 'N_' +
                                                              str(data_latitude[::-1][idx_lat_2]) + 'N_day_night')

    elif name_target in ['riceco2']:
        print('What do you wanna do?')
        print('     1: mean radius at a latitude where co2_ice exists (fig: alt-µm)')
        print('     2: max radius day-night (fig: lat-ls)')
        print('     3: altitude of top clouds (fig: lat-ls)')
        print('     4: radius/co2ice/temp/satu in polar projection')
        print('')
        view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_target, latitude_selected = vars_zonal_mean_in_time_co2ice_exists(filename, data_target)

            print('Display:')
            display_1fig_profiles(filename, data_target*1e6, latitude_selected, xmin=1e-9, xmax=500,
                            xlabel='Radius of ice particle (µm)',
                            xscale='log', yscale='linear',
                            title='Mean radius of ice particle between Ls=0-360° and '+str(latitude_selected[0])+'-'+
                                  str(latitude_selected[-1])+'°N',
                            savename='riceco2_mean_'+str(int(latitude_selected[0]))+'N_'+
                                     str(int(latitude_selected[-1]))+'N')

        if view_mode == 2:
            print('Processing data:')
            processing.riceco2_max_day_night()

            print('Display:')
            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='µm',
                                               title='Max radius of CO$_2$ ice',
                                               savename='max_riceco2_day_night.png')

        if view_mode == 3:
            print('Processing data:')
            top_cloud = processing.riceco2_topcloud_altitude(data_target)

            print('Display:')
            displays.topcloud_altitude(top_cloud)

        if view_mode == 4:
            print('Processing data:')
            data_riceco2 = random.rand(10,32,48,64)
            data_co2ice = random.rand(10,32,48,64)
            data_temp = random.rand(10,32,48,64)
            data_satuco2 = random.rand(10,32,48,64)
            print('Display:')
            display_4figs_polar_projection(data_riceco2, data_co2ice, data_temp, data_satuco2)

    elif name_target in ['ccnNco2']:
        view_mode = int(input('View (max_day_night=1): '))
        if view_mode == 1:

            filename_2 = getfilename(files)
            data_target_2 = getdata(directory_store + filename_2, 'ccnNco2')
            data_altitude = getdata(directory_store + filename_2, 'altitude')

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

    elif name_target in ['tau1mic']:
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

            displays.display_zonal_mean(zonal_mean, data_latitude, ndx, axis_ls, levels=arange(0, 0.014, 0.002),
                                        title=name_target, units='')

    elif name_target in ['tau']:  # Time, latitude, longitude
        zonal_mean = mean(data_target[:, :, :], axis=2)
        zonal_mean = flip(zonal_mean.T, axis=0)

        del data_target

        for i in range(zonal_mean.shape[0]):
            dim2 = where(zonal_mean[i, :] == 0)
            zonal_mean[i, dim2] = None

        if lslin:
            print('Perform linearization in progress')
            zonal_mean = libf.linearize_ls(zonal_mean, data_time.shape[0], data_latitude.shape[0], interp_time)
        displays.display_zonal_mean(zonal_mean, data_latitude, ndx, axis_ls, levels=None, title=name_target,
                                    units='SI')

    elif name_target in ['co2_conservation', 'Sols', 'Ls']:
        display_1d(data_target)

    else:
        print('Variable not used for the moment')


if '__main__' == __name__:
    main()
