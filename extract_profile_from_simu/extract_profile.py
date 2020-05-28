#!/bin/bash python3

from numpy import abs, argmin, savetxt, c_
from netCDF4 import Dataset
from os import listdir


def main():
    list_var = ['temp',    # temperature
                'co2',     # co2 vap mmr
                'co2_ice', # co2 ice mmr
                'h2o_vap', # h2o vap mmr
                'h2o_ice', # h2o ice mmr
                'ccnqco2', # ccn mass for co2 (mmr)      => ccnco2_mass
                'ccnNco2', # ccn number for co2 (#/kg)   => ccnco2_number
                'dustq',   # dust mass (mmr)             => dust_mass
                'dustN'    # dust number (#/kg)          => dust_number
                ]

    files = listdir('.')

    if any(".nc" in s for s in files):
        list_files = [x for x in files if '.nc' in x]
        filename = input("Choose between these files " + repr(list_files) + " ::")
    else:
        print('There is no Netcdf file in this directory !')
        exit()

    bigdata = Dataset(filename, "r", format="NETCDF4")
    data_ls = bigdata.variables['Ls']
    data_latitude = bigdata.variables['latitude']

    # choose the solar longitude and latitude for the extraction
    target_ls = float(input('Select the ls for the extraction: '))
    target_latitude = float(input('Select the latitude for the extraction: '))

    idx_latitude = (abs(data_latitude[:] - target_latitude)).argmin()
    idx_ls = (abs(data_ls[:] - target_ls)).argmin()

    # add/remove a variable for the extraction
    #TODO: add/remove a variable for the extraction
    print('Variable for the extraction: ',repr(list_var))

    # extract data at longitde 0°E
    for i, value_i in enumerate(list_var):
        print(i, value_i)
        data = bigdata.variables[list_var[i]][idx_ls, :, idx_latitude, 0]

        # write the extracted data in file
        if value_i == 'temp':
            savetxt('profile', c_[data], fmt='%.3f')

        elif value_i == 'ccnqco2':
            savetxt('profile_ccnco2_mass', c_[data], fmt='%.3f')

        elif value_i == 'ccnNco2':
            savetxt('profile_ccnco2_number', c_[data], fmt='%.3f')

        elif value_i == 'dustq':
            savetxt('profile_dust_mass', c_[data], fmt='%.3f')

        elif value_i == 'dustN':
            savetxt('profile_dust_number', c_[data], fmt='%.3f')

        else:
            savetxt('profile_'+str(value_i), c_[data], fmt='%.3f')

if '__main__' == __name__:
    main()
