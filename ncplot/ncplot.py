#!/bin/bash python3
from netCDF4 import Dataset
from os import listdir
from sys import exit
import importlib
from numpy import mean, abs, min, max, zeros, flip, where, ones

displays = importlib.import_module('packages.displays')
libf = importlib.import_module('packages.lib_function')
ncdump = importlib.import_module('packages.ncdump')

def getfilename(files):
    if any(".nc" in s for s in files):
        list_files = sorted([x for x in files if '.nc' in x])
        if len(list_files) > 1:
            print('Netcdf files available: (0) {}'.format(list_files[0]))
            for i, value_i in enumerate(list_files[1:]):
                print('                        ({}) {}'.format(i+1, value_i))
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


def get_alt_at_max(data,index):
    return data[index]


def main():
    files = listdir('.')

    filename = getfilename(files)

    data_target = getdata(filename)

    data_latitude = getdata(filename, target="latitude")
    data_latitude = data_latitude[::-1]  # reverse the labels

    print('Create linear grid time...')
    data_time = getdata(filename, target="Time")
    interp_time, axis_ls, ndx = libf.linear_grid_ls(data_time)

    if data_target.name in ['co2_ice', 'h2o_ice']:
        # correct very low values of co2 mmr
        print('Correction value...')
        data_y = libf.correction_value_co2_ice(data_target[:, :, :, :])
        shape_data_y = data_y.shape

        view_mode = int(input('View (max=1, column=2, equa_prof=3, alt-lat=4, hovmoller=5): '))

        if view_mode == 1:
            print('Get max co2_ice...')
            max_mmr, x, y = libf.get_max_co2_ice_in_alt_lon(data_y)

            print('Extract other variable at co2_ice max value...')
            max_temp = libf.extract_at_max_co2_ice(data_temperature, x, y)
            max_satu = libf.extract_at_max_co2_ice(data_satuco2, x, y)
            max_radius = libf.extract_at_max_co2_ice(data_riceco2, x, y)
            max_ccnN = libf.extract_at_max_co2_ice(data_ccnNco2, x, y)
            max_alt = libf.extract_at_max_co2_ice(data_altitude, x, y, shape_data_y)

            print('Reshape and linearized data...')
            max_mmr = libf.linearize_ls(max_mmr, shape_data_y[0], shape_data_y[2], interp_time)
            max_temp = libf.linearize_ls(max_temp, shape_data_y[0], shape_data_y[2], interp_time)
            max_satu = libf.linearize_ls(max_satu, shape_data_y[0], shape_data_y[2], interp_time)
            max_radius = libf.linearize_ls(max_radius, shape_data_y[0], shape_data_y[2], interp_time)
            max_ccnN = libf.linearize_ls(max_ccnN, shape_data_y[0], shape_data_y[2], interp_time)
            max_alt = libf.linearize_ls(max_alt, shape_data_y[0], shape_data_y[2], interp_time)

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
            data_altitude = getdata(filename, target='altitude')
            displays.display_colonne(data_target, data_altitude, data_latitude, ndx, axis_ls, interp_time,
                                     unit='kg/m$^2$')

        elif view_mode == 3:
            displays.display_equa_profile(data_target, data_time, data_latitude, data_altitude, data_temperature,
                                          data_satuco2, unit='kg/kg')

        elif view_mode == 4:
            displays.display_altitude_latitude(data_target, data_time, data_altitude, data_latitude, unit='kg/kg')

        elif view_mode == 5:
            displays.hovmoller_diagram(data_target, data_latitude, data_altitude, data_longitude, data_time)

        elif view_mode == 6:
            displays.display_vertical_profile()

        elif view_mode == 7:
            filename_2 = getfilename(files)
            data_target_2 = getdata(filename_2, data_target.name)
            print('Day file is {}'.format(filename))
            print('Night file is {}'.format(filename_2))
            print('')
            print('Correction value...')
            data_y_2 = libf.correction_value_co2_ice(data_target_2[:, :, :, :])
            data_altitude = getdata(filename_2, 'altitude')
            print('Get max of {}...'.format(data_target.name))
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_y, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_y_2, extrema='max')

            max_alt_day = ones(idx_altitude_day.shape)
            max_alt_night = ones(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i,j] = data_altitude[idx_altitude_night[i,j]]
                    max_alt_day[i,j] = data_altitude[idx_altitude_day[i,j]]
                    if max_alt_day[i,j] == 0 and max_satu_day[i,j] < 1e-10:
                        max_alt_day[i,j] = None
                        max_satu_day[i,j] = None
                    if max_alt_night[i,j] == 0 and max_satu_night[i,j] < 1e-10:
                        max_alt_night[i,j] = None
                        max_satu_night[i,j] = None


            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='kg/kg', title='Max vmr of CO$_2$ ice',
                                               savename='max_co2_ice_day_night.png')

        else:
            print('Wrong number')

    elif data_target.name in ['temp']:
        view_mode = int(input('View (profile_evolution=1, compare_day_night=2): '))
        if view_mode == 1:
            data_pressure = getdata(filename, target='pressure')
            data_longitude = getdata(filename, target='longitude')

            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:,:,idx_lon]

            idx_ls = data_target[:, :].argmin(axis=0)
            data_target = data_target[idx_ls[20] - 5:idx_ls[20] + 5, :]

            T_sat = libf.tcondco2(data_pressure, idx_ls=None, idx_lat=None, idx_lon=None)
            displays.display_temperature_profile_evolution(data_target,  data_latitude, data_pressure, T_sat)

        elif view_mode == 2:
            data_longitude = getdata(filename, target='longitude')
            data_altitude = getdata(filename, target='altitude')
            idx_lat = (abs(data_latitude[:] - 0)).argmin()
            data_target = data_target[:, :, idx_lat, :]

            idx_lon = (abs(data_longitude[:] - 0)).argmin()
            data_target = data_target[:,:,idx_lon]

            idx_ls = (abs(data_time[:] - 30)).argmin()
            data_target = mean(data_target[5:idx_ls+5:7, :], axis=0)

            filename_2 = getfilename(files)
            data_target_2 = getdata(filename_2, 'temp')
            data_target_2 = mean(data_target_2[4:idx_ls+4:7,:,idx_lat,idx_lon], axis=0)

            filename_3 = getfilename(files)
            data_pressure = getdata(filename_3, target='pressure')
            T_sat = libf.tcondco2(data_pressure, idx_ls=idx_ls, idx_lat=idx_lat, idx_lon=idx_lon)

            displays.display_temperature_profile_day_night(data_target, data_target_2, T_sat, data_altitude)

    elif data_target.name in ['h2o_vap']:
        displays.display_colonne(data_target, data_time, data_latitude, unit='pr.µm')

    elif data_target.name in ['co2_conservation', 'Sols', 'Ls']:
        displays.display_1d(data_target)

    elif data_target.name in ['satuco2']:
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
            displays.display_max_lon_alt_satuco2(max_satu, data_latitude,axis_ls, ndx, max_alt)

        if view_mode == 3:
            filename_2 = getfilename(files)
            data_target_2 = getdata(filename_2, 'satuco2')
            data_altitude = getdata(filename_2, 'altitude')

            print('Get max satuco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')

            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i,j] = data_altitude[idx_altitude_night[i,j]]
                    max_alt_day[i,j] = data_altitude[idx_altitude_day[i,j]]
                    if max_satu_day[i,j] <= 1.0:
                        max_alt_day[i,j] = None
                        max_satu_day[i,j] = 0

                    if max_satu_night[i,j] <= 1.0:
                        max_alt_night[i,j] = None
                        max_satu_night[i,j] = 0

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='',
                                               title='Max saturation of CO$_2$ ice',
                                               savename='max_satuco2_day_night.png')

    elif data_target.name in ['riceco2']:
        view_mode = int(input('View (profile=1, max_day_night=2): '))

        if view_mode == 1:
            displays.display_riceco2_profile_pole(data_target, data_latitude, data_time, data_pressure)

        if view_mode == 2:
            filename_2 = getfilename(files)
            data_target_2 = getdata(filename_2, 'riceco2')
            data_altitude = getdata(filename_2, 'altitude')

            print('Get max riceco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')

            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i,j] = data_altitude[idx_altitude_night[i,j]]
                    max_alt_day[i,j] = data_altitude[idx_altitude_day[i,j]]
                    if max_satu_day[i,j] == 0:
                        max_alt_day[i,j] = None
                        max_satu_day[i,j] = None

                    if max_satu_night[i,j] == 0:
                        max_alt_night[i,j] = None
                        max_satu_night[i,j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='µm',
                                               title='Max radius of CO$_2$ ice',
                                               savename='max_riceco2_day_night.png')

    elif data_target.name in ['ccnNco2']:
        view_mode = int(input('View (max_day_night=1): '))
        if view_mode == 1:

            filename_2 = getfilename(files)
            data_target_2 = getdata(filename_2, 'ccnNco2')
            data_altitude = getdata(filename_2, 'altitude')

            print('Get max ccnNco2...')
            max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
            max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')
            print(min(max_satu_day), max(max_satu_day))
            print(min(max_satu_night), max(max_satu_night))
            max_alt_day = zeros(idx_altitude_day.shape)
            max_alt_night = zeros(idx_altitude_night.shape)
            for i in range(idx_altitude_night.shape[0]):
                for j in range(idx_altitude_night.shape[1]):
                    max_alt_night[i,j] = data_altitude[idx_altitude_night[i,j]]
                    max_alt_day[i,j] = data_altitude[idx_altitude_day[i,j]]
                    if max_satu_day[i,j] < 1:
                        max_alt_day[i,j] = None
                        max_satu_day[i,j] = None

                    if max_satu_night[i,j] < 1:
                        max_alt_night[i,j] = None
                        max_satu_night[i,j] = None

            displays.display_lat_ls_maxsatuco2(max_satu_day, max_satu_night, max_alt_day, max_alt_night,
                                               data_latitude, ndx, axis_ls, unit='#/kg',
                                               title='Max CCN number',
                                               savename='max_ccnNco2_day_night.png')
    else:
        print('Variable not used for the moment')


if '__main__' == __name__:
    main()
