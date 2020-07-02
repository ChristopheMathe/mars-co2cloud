#!/bin/bash python3
from netCDF4 import Dataset
from ncdump import ncextract
from os import listdir
from sys import exit
from displays import *


def getfilename(files):
    if any(".nc" in s for s in files):
        list_files = [x for x in files if '.nc' in x]
        if len(list_files) > 1:
            print('Netcdf files available: {} '.format(list_files[:3]))
            print('                        {} '.format(list_files[3:6]))
            filename = input("Select the file :")
            print('')
        else:
            filename = list_files[0]
    else:
        print('There is no Netcdf file in this directory !')
        filename = ''
        exit()
    return filename


def getdata(filename):
    # name_dimension_target = data.variables[variable_target].dimensions  # get access to all dimensions names
    # dimension = array(len(name_dimension_target))

    data = Dataset(filename, "r", format="NETCDF4")
    ncextract(filename, data, verb=True)
    variable_target = input('Select the variable: ')  # TODO faire les tests si la variable existe

    data_target = data.variables[variable_target]
    data_latitude = data.variables['latitude']
    data_longitude = data.variables['longitude']
    data_altitude = data.variables['altitude']
    data_time = data.variables['Ls']
    data_temperature = data.variables['temp']
    data_satuco2 = data.variables['satuco2']
    data_riceco2 = data.variables['riceco2']
    data_ccnNco2 = data.variables['ccnNco2']

    return data_target, data_time, data_altitude, data_latitude, data_longitude, data_temperature, data_satuco2, \
           data_riceco2, data_ccnNco2

def main():
    files = listdir('.')

    filename = getfilename(files)

    data_target, data_time, data_altitude, data_latitude, data_longitude, data_temperature, data_satuco2, \
    data_riceco2, data_ccnNco2 = getdata(filename)

    if data_target.name in ['co2_ice', 'h2o_ice', 'temp']:
        view_mode = int(input('View (max=1, column=2, equa_prof=3, alt-lat=4): '))
        if view_mode == 1:
            display_max_lon_alt(data_target, data_time, data_latitude, data_altitude,
                                         data_temperature, data_satuco2, data_riceco2, data_ccnNco2, unit='kg/kg')
        elif view_mode == 2:
            display_colonne(data_target, data_time, data_altitude, data_latitude, data_longitude,
                                     unit='kg/m$^2$')
        elif view_mode == 3:
            display_equa_profile(data_target, data_time, data_latitude, data_altitude, data_temperature,
                                          data_satuco2, unit='kg/kg')
        elif view_mode == 4:
            display_altitude_latitude(data_target, data_time, data_altitude, data_latitude, unit='kg/kg')
        else:
            print('Wrong number')

    elif data_target.name in ['h2o_vap']:
        display_colonne(data_target, data_time, data_latitude, unit='pr.µm')

    elif data.variables[variable_target].name in ['co2_conservation']:
        display_1D(data.variables[variable_target])

    elif data.variables[variable_target].name in ['satuco2']:
        display_altitude_latitude_satuco2(data_target, data_time, data_altitude, data_latitude)
    else:
        print('Variable not used for the moment')


if '__main__' == __name__:
    main()
