#!/bin/bash python3

from netCDF4 import Dataset
from ncdump import ncextract
from displays import display_1D, display_2D, display_3D, display_colonne, display_zonal_mean, display_max_lon_alt
from os import listdir
from sys import exit
import numpy as np

def main():
    files = listdir('.')

    if any(".nc" in s for s in files):
        list_files = [x for x in files if '.nc' in x]
        filename = input("Choose between these files "+repr(list_files)+" ::")
    else:
        print('There is no Netcdf file in this directory !')
        exit()

    data = Dataset(filename, "r", format="NETCDF4")

    ncextract(filename, data, verb=True)
    #    dimension_target = data.variables[variable_target].dimensions # get access to all dimensions names
    #    dimension1 = len(data.dimensions[dimension_target[0]]) # get access to the value of a dimension

    variable_target = input('Select the variable: ') #TODO faire les tests si la variable existe
    data_time = data.variables['Ls']
    data_latitude = data.variables['latitude']
    data_altitude = data.variables['altitude']
    data_temperature = data.variables['temp']
    data_satuco2 = data.variables['satuco2']

    if data.variables[variable_target].name in ['co2_ice', 'h2o_ice']:
        view_mode = int(input('View (max=1, column=2): '))
        if view_mode == 1:
            display_max_lon_alt(data.variables[variable_target], data_time, data_latitude, data_altitude,
                                data_temperature, data_satuco2, unit='kg/kg')
        elif view_mode == 2:
            display_colonne(data.variables[variable_target], data_time, data_latitude, unit='kg/m$^2$')
        else:
            print('Wrong number')
    elif data.variables[variable_target].name in ['h2o_vap']:
        display_colonne(data.variables[variable_target], data_time, data_latitude, unit='pr.µm')
    elif data.variables[variable_target].name in ['co2_conservation']:
        display_1D(data.variables[variable_target])
    elif data.variables[variable_target].name in ['rsedcloudco2']:
        display_rsedco2(data.variables[variable_target], data_time, data_latitude, unit='m')
    else:
        print('Variable not used for the moment')


if '__main__' == __name__:
    main()