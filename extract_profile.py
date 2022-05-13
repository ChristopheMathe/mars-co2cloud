from numpy import abs, amax, argmax, savetxt, c_, append, unravel_index
from netCDF4 import Dataset
from os import listdir, mkdir, path
from packages.ncdump import getfilename


def main():
    list_var = ['temp',               # temperature
                'co2',                # co2 vap mmr
                'co2_ice',            # co2 ice mmr
                'h2o_vap',            # h2o vap mmr
                'h2o_ice',            # h2o ice mmr
                'ccnq',               # ccn mass for h2o (mmr)                           => ccn_mass
                'ccnN',               # ccn number for h2o (#/kg)                        => ccn_number
                'ccnqco2',            # ccn mass for co2 (mmr)                           => ccnco2_mass
                'ccnNco2',            # ccn number for co2 (#/kg)                        => ccnco2_number
                'ccnqco2_h2o_m_ice',  # h2o ccn mass of h2o ice for co2 (mmr)            => ccnco2_h2o_mass_ice
                'ccnqco2_h2o_m_ccn',  # h2o ccn mass of dust for co2 (mmr)               => ccnco2_h2o_mass_ccn
                'ccnNco2_h2o',        # ccn number of h2o ice for co2 (#/kg)             => ccnco2_h2o_mass_ice
                'ccnqco2_meteor',     # ccn mass of meteoric particle for co2 (mmr)      => ccnco2_meteor_mass
                'ccnNco2_meteor',     # ccn number of meteoric particle for co2 (#/kg)   => ccnco2_meteor_number
                'dustq',              # dust mass (mmr)                                  => dust_mass
                'dustN'               # dust number (#/kg)                               => dust_number
                ]

    format_output = '%.8e'
    files = listdir('.')

    directory_store = [x for x in files if 'occigen' in x][0] + '/'

    files = listdir(directory_store)

    filename = getfilename(files)

    bigdata = Dataset(directory_store + filename, "r", format="NETCDF4")
    data_time = bigdata.variables['Time']
    data_latitude = bigdata.variables['latitude']

    # choose the solar longitude and latitude for the extraction
    target_ls = float(input(f'Select the time for the extraction (max={amax(data_time):.2f} sols): '))
    target_latitude = float(input('Select the latitude for the extraction: '))
    target_mmr_max = input('Select the longitude where the mmr is max? (y/n): ')

    idx_latitude = (abs(data_latitude[:] - target_latitude)).argmin()
    idx_ls = (abs(data_time[:] - target_ls)).argmin()

    # add/remove a variable for the extraction
    print('Variable for the extraction: ', repr(list_var))

    # extract data at longitude where co2_ice mmr is max
    if target_mmr_max == 'y':
        data = bigdata.variables['co2_ice'][idx_ls, :, idx_latitude, :]
        idx_max = argmax(data)  # along longitude
        tmp, idx_longitude = unravel_index(idx_max, data.shape)
    else:
        idx_longitude = 0

    directory_output = f'profiles_{filename[:-3]}_sols{data_time[idx_ls]:.0f}_lat{data_latitude[idx_latitude]:.0f}+N/'

    if path.isdir(directory_output):
        print('Be careful, the folder already exist')
    else:
        mkdir(directory_output)

    # extract data
    for i, value_i in enumerate(list_var):
        print(i, value_i)
        try:
            data = bigdata.variables[list_var[i]][idx_ls, :, idx_latitude, idx_longitude]
        except KeyError:
            print(f'{value_i} is not found in the file, the script continue')
            continue
        # add a value
        data = append(data, data[-1])

        # write the extracted data in file
        if value_i == 'temp':
            savetxt(f'{directory_output}profile', c_[data], fmt=format_output)

        elif value_i == 'ccnq':
            savetxt(f'{directory_output}profile_ccn_mass', c_[data], fmt=format_output)

        elif value_i == 'ccnN':
            savetxt(f'{directory_output}profile_ccn_number', c_[data], fmt=format_output)

        elif value_i == 'ccnqco2':
            savetxt(f'{directory_output}profile_ccnco2_mass', c_[data], fmt=format_output)

        elif value_i == 'ccnNco2':
            savetxt(f'{directory_output}profile_ccnco2_number', c_[data], fmt=format_output)

        elif value_i == 'ccnqco2_h2o_m_ice':
            savetxt(f'{directory_output}profile_ccnco2_h2o_mass_ice', c_[data], fmt=format_output)

        elif value_i == 'ccnqco2_h2o_m_ccn':
            savetxt(f'{directory_output}profile_ccnco2_h2o_mass_ccn', c_[data], fmt=format_output)

        elif value_i == 'ccnNco2_h2o':
            savetxt(f'{directory_output}profile_ccnco2_h2o_number', c_[data], fmt=format_output)

        elif value_i == 'ccnqco2_meteor':
            savetxt(f'{directory_output}profile_ccnco2_meteor_mass', c_[data], fmt=format_output)

        elif value_i == 'ccnNco2_meteor':
            savetxt(f'{directory_output}profile_ccnco2_meteor_number', c_[data], fmt=format_output)

        elif value_i == 'dustq':
            savetxt(f'{directory_output}profile_dust_mass', c_[data], fmt=format_output)

        elif value_i == 'dustN':
            savetxt(f'{directory_output}profile_dust_number', c_[data], fmt=format_output)

        else:
            savetxt(f'{directory_output}profile_{value_i}', c_[data], fmt=format_output)


if '__main__' == __name__:
    main()
