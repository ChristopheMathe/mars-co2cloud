#!/bin/bash python3
from numpy import abs, amax, argmin, argmax, savetxt, c_, append, unravel_index
from netCDF4 import Dataset
from os import listdir, mkdir
from packages.ncdump import getfilename


def main():
    list_var = ['temp',    # temperature
                'co2',     # co2 vap mmr
                'co2_ice', # co2 ice mmr
                'h2o_vap', # h2o vap mmr
                'h2o_ice', # h2o ice mmr
                'ccnq',    # ccn mass for h2o (mmr)      => ccn_mass
                'ccnN',    # ccn number for h2o (#/kg)   => ccn_number
                'ccnqco2', # ccn mass for co2 (mmr)      => ccnco2_mass
                'ccnNco2', # ccn number for co2 (#/kg)   => ccnco2_number
                'dustq',   # dust mass (mmr)             => dust_mass
                'dustN'    # dust number (#/kg)          => dust_number
                ]

    format_output ='%.8e'
    files = listdir('.')

    directory_store = [x for x in files if 'occigen' in x][0] + '/'
    print(directory_store)

    files = listdir(directory_store)

    filename = getfilename(files)

    bigdata = Dataset(directory_store + filename, "r", format="NETCDF4")
    data_time = bigdata.variables['Time']
    data_latitude = bigdata.variables['latitude']

    # choose the solar longitude and latitude for the extraction
    target_ls = float(input('Select the time for the extraction (max='+str(amax(data_time))+'): '))
    target_latitude = float(input('Select the latitude for the extraction: '))
    target_mmr_max = input('Select the longitude where the mmr is max? (y/n): ')

    idx_latitude = (abs(data_latitude[:] - target_latitude)).argmin()
    idx_ls = (abs(data_time[:] - target_ls)).argmin()

    # add/remove a variable for the exxtraction
    #TODO: add/remove a variable for the extraction
    print('Variable for the extraction: ', repr(list_var))

    # extract data at longitude where co2_ice mmr is max
    if target_mmr_max == 'y':
        data = bigdata.variables['co2_ice'][idx_ls, :, idx_latitude, :]
        idx_max = argmax(data)  # along longitude
        tmp, idx_longitude = unravel_index(idx_max, data.shape)

    directory_output = 'profiles_'+filename[:-3]+'_sols'+str(int(data_time[idx_ls]))+'_lat'+str(int(data_latitude[
                                                                                                  idx_latitude]))+'N/'
    print(directory_output)
    try:
        mkdir(directory_output)
    except:
        print('Be careful, the folder already exist')

    # extract data at longitude 0°E
    for i, value_i in enumerate(list_var):
        print(i, value_i)
        if target_mmr_max == 'y':
            data = bigdata.variables[list_var[i]][idx_ls, :, idx_latitude, idx_longitude]
        else:
            data = bigdata.variables[list_var[i]][idx_ls, :, idx_latitude, 0]

        # add a value
        data = append(data, data[-1])

        # write the extracted data in file
        if value_i == 'temp':
            savetxt(directory_output+'profile', c_[data], fmt=format_output)

        elif value_i == 'ccnq':
            savetxt(directory_output+'profile_ccn_mass', c_[data], fmt=format_output)

        elif value_i == 'ccnN':
            savetxt(directory_output+'profile_ccn_number', c_[data], fmt=format_output)

        elif value_i == 'ccnqco2':
            savetxt(directory_output+'profile_ccnco2_mass', c_[data], fmt=format_output)

        elif value_i == 'ccnNco2':
            savetxt(directory_output+'profile_ccnco2_number', c_[data], fmt=format_output)

        elif value_i == 'dustq':
            savetxt(directory_output+'profile_dust_mass', c_[data], fmt=format_output)

        elif value_i == 'dustN':
            savetxt(directory_output+'profile_dust_number', c_[data], fmt=format_output)

        else:
            savetxt(directory_output+'profile_'+str(value_i), c_[data], fmt=format_output)

if '__main__' == __name__:
    main()
