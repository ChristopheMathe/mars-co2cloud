#!/bin/bash python3
from packages.DataProcessed import *
from packages.displays import *
from os import listdir


def main():
    files = listdir('.')

    directory_store = [x for x in files if 'occigen' in x][0] + '/'

    if directory_store is None:
        directory_store = ''
    else:
        files = listdir(directory_store)

    filenames = ['stats1.nc', 'stats2.nc', 'stats3.nc', 'stats4.nc', 'stats5.nc', 'stats6.nc',
                 'stats7.nc', 'stats8.nc', 'stats9.nc', 'stats10.nc', 'stats11.nc', 'stats12.nc']

    data_target = getdata(directory_store + filenames[0])
    name_target = data_target.name
    unit_target = data_target.units

    data_time = getdata(directory_store + filenames[0], target='Time')
    if name_target in ['emis']:
        print('What do you wanna do?')
        print('     1: zonal mean month by month (stats.nc)')
        print('')
        view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed = zeros((12, data_target.shape[1])) # 12 months

            for f, file in enumerate(filenames):
                data_target = getdata(directory_store + file, target=name_target)
                data_target, time_selected = slice_data(data_target, dimension_data=data_time[:], value=14)
                data_target = correction_value(data_target, threshold=1e-13)
                data_processed[f, :] = mean(data_target, axis=1)

            print('Display:')
            display_stats_vars_zonalmean(directory_store + filenames[0], data_processed)

if '__main__' == __name__:
    main()
