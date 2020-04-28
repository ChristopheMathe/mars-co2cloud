#!/bin/bash python3

from netCDF4 import Dataset
from ncdump import ncdump
from displays import display_1D, display_2D, display_3D, display_colonne
from os import listdir
from sys import exit

def main():
    files = listdir('.')

    if any(".nc" in s for s in files):
        print(files)
        filename = [x for x in files if '.nc' in x][0]
    else:
        print('There is no Netcdf file in this directory !')
        exit()

    data = Dataset(filename, "r", format="NETCDF4")

    ncdump(filename, data, verb=True)

    variable_target = input('Select the variable: ') #TODO faire les tests si la variable existe
    display_colonne(data.variables[variable_target])

#    if len(data.variables[variable_target].shape) > 1:
#        display_mode = input('View mode (2D/3D): ')
#        if display_mode == '2D':
#            display_2D(data.variables[variable_target])
#        elif display_mode == '3D':
#            display_3D(data.variables[variable_target])
#        else:
#            print('Choose between 2D and 3D')
#    else:
#        display_1D(data.variables[variable_target])


#    dimension_target = data.variables[variable_target].dimensions # get access to all dimensions names
#    dimension1 = len(data.dimensions[dimension_target[0]]) # get access to the value of a dimension


if '__main__' == __name__:
    main()