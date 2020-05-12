#!/bin/bash python3

from netCDF4 import Dataset
from ncdump import ncextract
from displays import display_1D, display_2D, display_3D, display_colonne, display_zonal_mean, display_rsedco2
from os import listdir
from sys import exit


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

    variable_target = input('Select the variable: ') #TODO faire les tests si la variable existe
    data_time = data.variables['Ls']
    data_latitude = data.variables['latitude']


    if data.variables[variable_target].name in ['co2_ice', 'h2o_ice']:
        display_colonne(data.variables[variable_target], data_time, data_latitude, unit='kg/m$^2$')
    elif data.variables[variable_target].name in ['h2o_vap']:
        display_colonne(data.variables[variable_target], data_time, data_latitude, unit='pr.µm')
    elif data.variables[variable_target].name in ['co2_conservation']:
        display_1D(data.variables[variable_target])
    elif data.variables[variable_target].name in ['rsedcloudco2']:
        display_rsedco2(data.variables[variable_target], data_time, data_latitude, unit='m')
    else:
        print('Variable not used for the moment')

 #   if len(data.variables[variable_target].shape) > 1:
 #       display_mode = input('View mode (2D/3D): ')
 #       if display_mode == '2D':
 #           purpose = input('What do you want (view/column/zonal)? ')
 #           if purpose == 'view':
 #               display_2D(data.variables[variable_target])
 #           else:
 #               print("Wrong input.")
 #       elif display_mode == '3D':
 #           display_3D(data.variables[variable_target])
 #       else:
 #           print('Choose between 2D and 3D')
 #   else:
 #       display_1D(data.variables[variable_target])


#    dimension_target = data.variables[variable_target].dimensions # get access to all dimensions names
#    dimension1 = len(data.dimensions[dimension_target[0]]) # get access to the value of a dimension


if '__main__' == __name__:
    main()