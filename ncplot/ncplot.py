#!/bin/bash python3

from netCDF4 import Dataset
from sys import argv
from ncdump import ncdump
from displays import display_1D, display_2D, display_3D

def main():
    filename = argv[1]

    data = Dataset(filename, "r", format="NETCDF4")

    ncdump(filename, data, verb=True)

    variable_target = input('Select the variable: ') #TODO faire les tests si la variable existe

    if len(data.variables[variable_target].shape) > 1:
        display_mode = input('View mode (2D/3D): ')
        if display_mode == '2D':
            display_2D(data.variables[variable_target])
        elif display_mode == '3D':
            display_3D(data.variables[variable_target])
        else:
            print('Choose between 2D and 3D')
    else:
        display_1D(data.variables[variable_target])

    display_2D(data.variables[variable_target])
    display_3D(data.variables[variable_target])


#    dimension_target = data.variables[variable_target].dimensions # get access to all dimensions names
#    dimension1 = len(data.dimensions[dimension_target[0]]) # get access to the value of a dimension
#
#    mask = (data[variable_target][:,:,:,:] >= 1.)
#    print(data[variable_target][:,:,:,:][mask])


if '__main__' == __name__:
    main()