from numpy import mean, abs, min, max, zeros, where, ones, concatenate, arange, unravel_index, argmax, array, \
    count_nonzero, std, append, asarray, power, ma, ndarray
from .lib_function import *
from .ncdump import getdata, getfilename
from os import mkdir
from sys import exit


def co2ice_thickness_atm_layer(filename, data):
    data_altitude = getdata(filename=filename, target='altitude')
    if data_altitude.units in ['Pa']:
        data_altitude = getdata(filename=filename, target='zaeroid')

    data_latitude = getdata(filename=filename, target='latitude')
    data_north = slice_data(data, dimension_data=data_latitude[:], value=[60, 90])
    data_south = slice_data(data, dimension_data=data_latitude[:], value=[-60, -90])

    data_time = getdata(filename=filename, target='Time')
    # bin ls to 5°
    if max(data_time[:]) < 360:
        ls_bin = arange(0, 365, 5)
    else:
        # sols binned at 5° Ls
        ls_bin = convert_sols_to_ls()

    nbin = ls_bin.shape[0]
    data_icelayer = zeros((2, nbin))

    for bin in range(nbin - 1):
        idx = (abs(data_time[:] - ls_bin[bin])).argmin()
        idx2 = (abs(data_time[:] - ls_bin[bin + 1])).argmin()
        data_binned_north = data_north[idx:idx2, :, :, :]
        data_binned_south = data_south[idx:idx2, :, :, :]
        if max(data_binned_north) >= 1e-10:
            ind = unravel_index(argmax(data_binned_north, axis=None), data_binned_north.shape)
            for i in range(int(ind[1]), data_altitude.shape[0]):
                if data_binned_north[ind[0], i, ind[2], ind[3]] >= 1e-10:
                    idx_max = i
            data_icelayer[0, bin] = data_altitude[idx_max] - data_altitude[ind[1]]
        if max(data_binned_south) >= 1e-10:
            ind = unravel_index(argmax(data_binned_south, axis=None), data_binned_south.shape)
            for i in range(int(ind[1]), data_altitude.shape[0]):
                if data_binned_south[ind[0], i, ind[2], ind[3]] >= 1e-10:
                    idx_max = i
            data_icelayer[1, bin] = data_altitude[idx_max] - data_altitude[ind[1]]

    return data_icelayer


def co2ice_polar_cloud_distribution(filename, data, normalization):
    data_altitude = getdata(filename=filename, target='altitude')
    if data_altitude.long_name != 'Altitude above areoid':
        print('Data did not zrecasted above the aroid')
        print(f'\tCurrent: {data_altitude.long_name}')
        exit()

    # sliced data on latitude region
    data_latitude = getdata(filename, target='latitude')
    data_north, latitude_north = slice_data(data, dimension_data=data_latitude, value=[60, 90])
    data_south, latitude_south = slice_data(data, dimension_data=data_latitude, value=[-60, -90])
    del data

    # sliced data between 104 - 360°Ls time to compare with Fig8 of Neumann et al. 2003
    data_time = getdata(filename, target='Time')
    data_north, time_selected = slice_data(data_north, dimension_data=data_time, value=[104, 360])
    data_south, time_selected = slice_data(data_south, dimension_data=data_time, value=[104, 360])

    distribution_north = zeros((data_north.shape[2], data_north.shape[1]))
    distribution_south = zeros((data_south.shape[2], data_south.shape[1]))

    for latitude in range(data_north.shape[2]):
        for altitude in range(data_north.shape[1]):
            distribution_north[latitude, altitude] = count_nonzero(data_north[:, altitude, latitude, :] >= 1e-13)
            distribution_south[latitude, altitude] = count_nonzero(data_south[:, altitude, latitude, :] >= 1e-13)

    # normalisation
    if normalization:
        distribution_north = distribution_north / max(distribution_north) * 2000  # To compare with Fig.8 of Neumann2003
        distribution_south = distribution_south / max(distribution_south) * 2000

    return distribution_north, distribution_south, latitude_north, latitude_south


def co2ice_cloud_evolution(filename, data):
    data_latitude = getdata(filename, target='latitude')

    print('Select the latitude region (°N):')
    lat_1 = float(input('   latitude 1: '))
    lat_2 = float(input('   latitude 2: '))

    data, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[lat_1, lat_2])

    idx_max = unravel_index(argmax(data[:, :, :, :], axis=None), data[:, :, :, :].shape)
    idx_max = asarray(idx_max)
    data = data[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_satuco2 = getdata(filename, 'satuco2')
    data_satuco2, latitude_selected = slice_data(data_satuco2, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_satuco2 = data_satuco2[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_temp = getdata(filename, 'temp')
    data_temp, latitude_selected = slice_data(data_temp, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_temp = data_temp[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_riceco2 = getdata(filename, 'riceco2')
    data_riceco2, latitude_selected = slice_data(data_riceco2, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_riceco2 = data_riceco2[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_time = getdata(filename, target='Time')
    print('the maximum is at :' + str(data_time[idx_max[0]] * 24 % 24) + 'h local time.')

    return data, data_satuco2, data_temp, data_riceco2, idx_max, latitude_selected


def co2ice_cumulative_masses_polar_cap(filename, data):
    from numpy import sum

    # TODO: improved by getting polar cap
    data_latitude = getdata(filename, target='latitude')
    data_north, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[60, 90])
    data_south, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[-60, -90])
    del data

    # extract the area of grid
    data_aire = gcm_aire()

    data_aire_north, latitude_selected = slice_data(data_aire, dimension_data=data_latitude[:], value=[60, 90])
    data_aire_south, latitude_selected = slice_data(data_aire, dimension_data=data_latitude[:], value=[-60, -90])

    cumul_co2ice_north = zeros(data_north.shape[0])
    cumul_co2ice_south = zeros(data_south.shape[0])

    for ls in range(data_north.shape[0]):
        cumul_co2ice_north[ls] = sum(data_north[ls, :, :] * data_aire_north[:, :])
        cumul_co2ice_south[ls] = sum(data_south[ls, :, :] * data_aire_south[:, :])

    return cumul_co2ice_north, cumul_co2ice_south


def co2ice_time_mean(filename, data, duration, localtime):
    data, time = vars_time_mean(filename=filename, data=data, duration=duration, localtime=localtime)

    data = correction_value(data, operator='eq', threshold=0)
    data_aire = gcm_aire()
    data = data * data_aire[:, :]

    return data, time


def co2ice_density_column_evolution(filename, data):
    from math import floor
    from numpy import zeros
    # Show the evolution of density column at winter polar region
    data_time = getdata(filename=filename, target='Time')
    if data_time.units == 'degrees':
        print('The netcdf file is in ls !')
        print(f'Time[0] = {data_time[0]}, Time[-1] = {data_time[-1]}')
    #        exit()

    # Slice in time:
    print(f'Select the time range in sols from {floor(data_time[0])} to {int(data_time[-1])}')
    time_begin = float(input('Start time: '))
    time_end = float(input('End time: '))
    data, time_range = slice_data(data=data, dimension_data=data_time, value=[time_begin, time_end])

    # Slice data in polar region:
    data_latitude = getdata(filename=filename, target='latitude')
    pole = input('Select the polar region (N/S):')
    if pole.lower() == 'n':
        data, latitude = slice_data(data=data, dimension_data=data_latitude, value=[60, 90])
    elif pole.lower() == 's':
        data, latitude = slice_data(data=data, dimension_data=data_latitude, value=[-60, -90])
    else:
        latitude = None
        print('Wrong selection')
        exit()
    data, altitude_limit, zmin, zmax, altitude_unit = compute_column_density(filename=filename, data=data)

    print(data[0, :, :].filled())
    return data, time_range, latitude


def emis_polar_winter_gg2020_fig13(filename, data):
    # Slice in time
    data_time = getdata(filename=filename, target='Time')
    #       NP: 180°-360°
    data_np, time = slice_data(data=data, dimension_data=data_time, value=[180, 360])

    #       SP: 0°-180°
    data_sp, time = slice_data(data=data, dimension_data=data_time, value=[0, 180])

    # Slice in latitude > 60°
    data_latitude = getdata(filename=filename, target='latitude')

    data_np, latitude = slice_data(data=data_np, dimension_data=data_latitude[:], value=[60, 90])
    data_sp, latitude = slice_data(data=data_sp, dimension_data=data_latitude[:], value=[-60, -90])

    # Mean in time
    data_mean_np = mean(data_np, axis=0)
    data_mean_sp = mean(data_sp, axis=0)

    return data_mean_np, data_mean_sp


def h2o_ice_alt_ls_with_co2_ice(filename, data):
    data_latitude = getdata(filename=filename, target='latitude')

    data, latitude_selected = slice_data(data, dimension_data=data_latitude, value=20)
    zonal_mean = mean(data, axis=2)  # zonal mean

    data_co2_ice = getdata(filename=filename, target='co2_ice')
    data_co2_ice, latitude_selected = slice_data(data_co2_ice, dimension_data=data_latitude, value=20)

    zonal_mean_co2_ice = ma.masked_where(data_co2_ice < 1e-13, data_co2_ice)
    zonal_mean_co2_ice = mean(zonal_mean_co2_ice, axis=2)

    zonal_mean, zonal_mean_co2_ice = rotate_data(zonal_mean, zonal_mean_co2_ice, doflip=False)

    del data, data_co2_ice
    return zonal_mean, zonal_mean_co2_ice, latitude_selected


def riceco2_zonal_mean_co2ice_exists(filename, data):
    data_latitude = getdata(filename, target='latitude')

    # extract co2_ice data
    data_co2_ice = getdata(filename, target='co2_ice')

    data_slice_lat, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[-15, 15])
    data_co2_ice_slice_lat, latitude_selected = slice_data(data_co2_ice, dimension_data=data_latitude[:],
                                                           value=[-15, 15])

    data = ma.masked_where(data_co2_ice_slice_lat < 1e-13, data_slice_lat)

    zonal_mean = mean(data, axis=3)  # zonal mean
    zonal_mean = mean(zonal_mean, axis=1)  # altitude mean
    zonal_mean = rotate_data(zonal_mean, doflip=True)

    zonal_mean = correction_value(zonal_mean[0], threshold=1e-13)
    zonal_mean = zonal_mean * 1e6  # m to µm

    return zonal_mean, latitude_selected


def riceco2_topcloud_altitude(filename, data_target):
    #TODO: il faut que cela soit au-dessus de la surface local!
    data_altitude = getdata(filename, target='altitude')

    if data_altitude.long_name != 'Altitude above local surface':
        print(f'{data_altitude.long_name}')
        exit()

    data_ccnNco2 = getdata(filename, target='ccnNco2')
    data_rho = getdata(filename, target='rho')

    data_ccnNco2 = correction_value(data_ccnNco2[:, :, :, :], operator='inf', threshold=1e-13)
    data_rho = correction_value(data_rho[:, :, :, :], operator='inf', threshold=1e-13)

    data_target = mean(data_target[:, :, :, :], axis=3)
    data_ccnNco2 = mean(data_ccnNco2[:, :, :, :], axis=3)
    data_rho = mean(data_rho[:, :, :, :], axis=3)

    N_reflect = 2e-8 * power(data_target * 1e6, -2)  # valid latitude range > 60°
    N_part = data_rho * data_ccnNco2

    nb_time = data_target.shape[0]
    nb_alt = data_target.shape[1]
    nb_lat = data_target.shape[2]
    del [data_target, data_ccnNco2, data_rho]

    data_latitude = getdata(filename=filename, target='latitude')
    a = (abs(data_latitude[:] - 60)).argmin() + 1
    polar_latitude = concatenate((arange(a), arange(nb_lat - a, nb_lat)))

    top_cloud = zeros((nb_time, nb_lat))
    for t in range(nb_time):
        for lat in polar_latitude:
            for alt in range(nb_alt - 1, -1, -1):
                if N_part[t, alt, lat] >= N_reflect[t, alt, lat] and alt > 1:
                    top_cloud[t, lat] = data_altitude[alt]
                    break

    top_cloud = rotate_data(top_cloud, doflip=False)[0]
    return top_cloud


def riceco2_max_day_night(filename, data):

    print('Compute max in progress...')
    max_satu_day, idx_altitude_day, y_day = get_extrema_in_alt_lon(data, extrema='max')
    max_satu_night, idx_altitude_night, y_night = get_extrema_in_alt_lon(data_target_2, extrema='max')

    max_alt_day = zeros(idx_altitude_day.shape)
    max_alt_night = zeros(idx_altitude_night.shape)
    for i in range(idx_altitude_night.shape[0]):
        for j in range(idx_altitude_night.shape[1]):
            max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
            max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
            if max_satu_day[i, j] == 0:
                max_alt_day[i, j] = None
                max_satu_day[i, j] = None

            if max_satu_night[i, j] == 0:
                max_alt_night[i, j] = None
                max_satu_night[i, j] = None

    return


def satuco2_zonal_mean_with_co2_ice(filename, data):
    data_latitude = getdata(filename=filename, target='latitude')
    # Select the three latitudes
    north = 80
    eq = 0
    south = -80
    print('Latitude selected:')
    print('\tNorth =   {}°N'.format(north))
    print('\tEquator = {}°N'.format(eq))
    print('\tSouth =   {}°S'.format(abs(south)))

    # Slice data for the three latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data, dimension_data=data_latitude, value=north)
    data_satuco2_eq, eq_latitude_selected = slice_data(data, dimension_data=data_latitude, value=eq)
    data_satuco2_south, south_latitude_selected = slice_data(data, dimension_data=data_latitude, value=south)

    # Compute zonal mean
    data_satuco2_north = mean(data_satuco2_north, axis=2)
    data_satuco2_eq = mean(data_satuco2_eq, axis=2)
    data_satuco2_south = mean(data_satuco2_south, axis=2)

    del data

    # Get co2 ice mmr
    data_co2ice = getdata(filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:, :, :, :], operator='inf', threshold=1e-13)

    # Slice co2 ice mmr at these 3 latitudes
    data_co2ice_north, north_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=north)
    data_co2ice_eq, eq_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=eq)
    data_co2ice_south, south_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=south)

    # Compute zonal mean
    data_co2ice_north = mean(data_co2ice_north, axis=2)
    data_co2ice_eq = mean(data_co2ice_eq, axis=2)
    data_co2ice_south = mean(data_co2ice_south, axis=2)
    del data_co2ice

    #    data_satuco2_north = correction_value(data_satuco2_north, operator='inf', threshold=1e-13)
    #    data_satuco2_eq = correction_value(data_satuco2_eq, operator='inf', threshold=1e-13)
    #    data_satuco2_south = correction_value(data_satuco2_south, operator='inf', threshold=1e-13)
    #    data_co2ice_north = correction_value(data_co2ice_north, operator='inf', threshold=1e-13)
    #    data_co2ice_eq = correction_value(data_co2ice_eq, operator='inf', threshold=1e-13)
    #    data_co2ice_south = correction_value(data_co2ice_south, operator='inf', threshold=1e-13)

    binned = input('Do you want bin data (Y/n)? ')

    if binned.lower() == 'y':
        # Bin time in 5° Ls
        data_time = getdata(filename=filename, target='Time')
        if max(data_time) > 360:
            time_grid_ls = convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]

            data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
            data_satuco2_eq_binned = zeros((nb_bin, data_satuco2_eq.shape[1]))
            data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
            data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
            data_co2ice_eq_binned = zeros((nb_bin, data_co2ice_eq.shape[1]))
            data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))

            for i in range(nb_bin - 1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i + 1])).argmin() + 1

                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_eq_binned[i, :] = mean(data_satuco2_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_eq_binned[i, :] = mean(data_co2ice_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[idx_ls_1:idx_ls_2, :], axis=0)
        else:
            if data_time.shape[0] % 60 == 0:
                print('Test 5°Ls binning: {} - {}'.format(data_time[0], data_time[60]))
            else:
                print('The data will not be binned in 5°Ls, need to work here')

            nb_bin = int(data_time.shape[0] / 60)
            data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
            data_satuco2_eq_binned = zeros((nb_bin, data_satuco2_eq.shape[1]))
            data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
            data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
            data_co2ice_eq_binned = zeros((nb_bin, data_co2ice_eq.shape[1]))
            data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))

            for i in range(nb_bin):
                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[i * 60:(i + 1) * 60, :], axis=0)
                data_satuco2_eq_binned[i, :] = mean(data_satuco2_eq[i * 60:(i + 1) * 60, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[i * 60:(i + 1) * 60, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[i * 60:(i + 1) * 60, :], axis=0)
                data_co2ice_eq_binned[i, :] = mean(data_co2ice_eq[i * 60:(i + 1) * 60, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[i * 60:(i + 1) * 60, :], axis=0)
            print(min(data_satuco2_north_binned))

        del data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, data_co2ice_south

        data_satuco2_north = correction_value(data_satuco2_north_binned, threshold=1e-13)
        data_satuco2_eq = correction_value(data_satuco2_eq_binned, threshold=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, threshold=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, threshold=1e-13)
        data_co2ice_eq = correction_value(data_co2ice_eq_binned, threshold=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, threshold=1e-13)

        del data_satuco2_north_binned, data_satuco2_eq_binned, data_satuco2_south_binned, data_co2ice_north_binned, \
            data_co2ice_eq_binned, data_co2ice_south_binned
    # No binning
    else:
        pass

    data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, data_co2ice_south = \
        rotate_data(data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq,
                    data_co2ice_south, doflip=False)

    return [data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq,
            data_co2ice_south, north_latitude_selected, eq_latitude_selected, south_latitude_selected, binned]


def satuco2_time_mean_with_co2_ice(filename, data):
    data_latitude = getdata(filename=filename, target='latitude')
    # Select the three latitudes
    north = 80
    south = -80
    print('Latitude selected:')
    print('\tNorth = {}°N'.format(north))
    print('\tSouth = {}°S'.format(abs(south)))

    # Slice data for the three latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data, dimension_data=data_latitude, value=north)
    data_satuco2_south, south_latitude_selected = slice_data(data, dimension_data=data_latitude, value=south)

    data_time = getdata(filename=filename, target='Time')
    north_winter = [270, 300]
    south_winter = [0, 30]
    print('Time selected:')
    print('\tNorth = {}°Ls'.format(north_winter))
    print('\tSouth = {}°Ls'.format(abs(south_winter)))

    # Slice data in time
    data_satuco2_north, north_winter_time = slice_data(data_satuco2_north, dimension_data=data_time, value=north_winter)
    data_satuco2_south, south_winter_time = slice_data(data_satuco2_south, dimension_data=data_time, value=south_winter)

    # Compute time mean
    data_satuco2_north = mean(data_satuco2_north, axis=0)
    data_satuco2_south = mean(data_satuco2_south, axis=0)

    del data

    # Get co2 ice mmr
    data_co2ice = getdata(filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:, :, :, :], operator='inf', threshold=1e-13)

    # Slice co2 ice mmr at these 3 latitudes
    data_co2ice_north, north_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=north)
    data_co2ice_south, south_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=south)

    # Slice data in time
    data_co2ice_north, north_winter_time = slice_data(data_co2ice_north, dimension_data=data_time, value=north_winter)
    data_co2ice_south, south_winter_time = slice_data(data_co2ice_south, dimension_data=data_time, value=south_winter)

    # Compute Time mean
    data_co2ice_north = mean(data_co2ice_north, axis=0)
    data_co2ice_south = mean(data_co2ice_south, axis=0)
    del data_co2ice

    binned = input('Do you want bin data (Y/n)? ')
    if binned.lower() == 'y':
        # Bin time in 5° Ls
        data_time = getdata(filename=filename, target='Time')
        if max(data_time) > 360:
            time_grid_ls = convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]

            data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
            data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
            data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
            data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))

            for i in range(nb_bin - 1):
                idx_ls_1 = (abs(data_time[:] - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(data_time[:] - time_grid_ls[i + 1])).argmin() + 1

                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[idx_ls_1:idx_ls_2, :], axis=0)
        else:
            if data_time.shape[0] % 60 == 0:
                print('Test 5°Ls binning: {} - {}'.format(data_time[0], data_time[60]))
            else:
                print('The data will not be binned in 5°Ls, need to work here')

            nb_bin = int(data_time.shape[0] / 60)
            data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
            data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
            data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
            data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))

            for i in range(nb_bin):
                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[i * 60:(i + 1) * 60, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[i * 60:(i + 1) * 60, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[i * 60:(i + 1) * 60, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[i * 60:(i + 1) * 60, :], axis=0)
            print(min(data_satuco2_north_binned))

        del data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south

        data_satuco2_north = correction_value(data_satuco2_north_binned, threshold=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, threshold=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, threshold=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, threshold=1e-13)

        del data_satuco2_north_binned, data_satuco2_south_binned, data_co2ice_north_binned, data_co2ice_south_binned
    # No binning
    else:
        pass

    return [data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south, north_latitude_selected,
            south_latitude_selected, binned]


def satuco2_hu2012_fig9(filename, data):
    data_latitude = getdata(filename, target='latitude')
    data_altitude = getdata(filename=filename, target='altitude')

    if data_altitude.long_name != 'Altitude above local surface':
        print('The netCDF file did not zrecasted above the local surface')
        exit()

    data_north, latitude_selected = slice_data(data, dimension_data=data_latitude, value=[60, 90])
    data_south, latitude_selected = slice_data(data, dimension_data=data_latitude, value=[-60, -90])
    del data

    # Bin time in 5° Ls
    data_time = getdata(filename=filename, target='Time')
    if data_time.shape[0] % 60 == 0:
        print(f'5°Ls binning: {data_time[0]} - {data_time[60]}')
        nb_bin = int(data_time.shape[0] / 60) + 1
    else:
        print(f'5°Ls binning: from {data_time[0]} to {data_time[-1]}')
        nb_bin = 72

    data_icelayer = zeros((2, nb_bin))
    data_icelayer_std = zeros((2, nb_bin))

    for bin in range(nb_bin - 1):
        data_binned_north, time_selected = slice_data(data_north, dimension_data=data_time[:],
                                                      value=[bin * 5, (bin + 1) * 5])
        data_binned_south, time_selected = slice_data(data_south, dimension_data=data_time[:],
                                                      value=[bin * 5, (bin + 1) * 5])
        print(f'Time: {time_selected[0]:.0f} / {time_selected[-1]:.0f}°Ls')
        tmp_north = array([])
        tmp_south = array([])

        # Find each super-saturation of co2 thickness
        for ls in range(data_binned_north.shape[0]):
            for longitude in range(data_binned_north.shape[3]):

                # For northern polar region
                for latitude_north in range(data_binned_north.shape[2]):
                    for alt in range(data_binned_north.shape[1]):
                        if data_binned_north[ls, alt, latitude_north, longitude] >= 1:
                            idx_min_north = alt
                            for alt2 in range(alt + 1, data_binned_north.shape[1]):
                                if data_binned_north[ls, alt2, latitude_north, longitude] < 1:
                                    idx_max_north = alt2 - 1
                                    tmp_north = append(tmp_north, abs(data_altitude[idx_max_north] -
                                                                      data_altitude[idx_min_north]))
                                    break
                            break

                # For southern polar region
                for latitude_south in range(data_binned_south.shape[2]):
                    for alt in range(data_binned_south.shape[1]):
                        if data_binned_south[ls, alt, latitude_south, longitude] >= 1:
                            idx_min_south = alt
                            for alt2 in range(alt + 1, data_binned_south.shape[1]):
                                if data_binned_south[ls, alt2, latitude_south, longitude] < 1:
                                    idx_max_south = alt2 - 1
                                    tmp_south = append(tmp_south, abs(data_altitude[idx_max_south] -
                                                                      data_altitude[idx_min_south]))
                                    break
                            break
        tmp_north = correction_value(tmp_north, 'inf', threshold=0)
        tmp_south = correction_value(tmp_south, 'inf', threshold=0)

        if tmp_north.size != 0:
            data_icelayer[0, bin] = mean(tmp_north)
            data_icelayer_std[0, bin] = std(tmp_north)
        else:
            data_icelayer[0, bin] = 0
            data_icelayer_std[0, bin] = 0

        if tmp_south.size != 0:
            data_icelayer[1, bin] = mean(tmp_south)
            data_icelayer_std[1, bin] = std(tmp_south)
        else:
            data_icelayer[1, bin] = 0
            data_icelayer_std[1, bin] = 0

        del tmp_north, tmp_south

    return data_icelayer, data_icelayer_std


def temp_gg2011_fig6(filename, data):
    # GG2011 worked with stats file
    data_latitude = getdata(filename=filename, target='latitude')
    data_longitude = getdata(filename=filename, target='longitude')
    data_time = getdata(filename=filename, target='Time')
    data_altitude = getdata(filename=filename, target='altitude')
    data_rho = getdata(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areroid (A) must be performed
    if data_altitude.long_name != 'Altitude above areoid':
        print(data_altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(data_time[:])
    if not stats_file or 'stats5' not in filename:
        print('This file is not a stats file required to compare with GG2011')
        exit()


    # Slice data: lon = 0°, lat = 0° [Fig 6, GG2011]
    ## For temperature
    data_tmp, longitude = slice_data(data=data[:, :, :, :], dimension_data=data_longitude[:], value=0)
    data_tmp, latitude = slice_data(data=data_tmp, dimension_data=data_latitude[:], value=0)

    ## For density
    data_rho_tmp, longitude = slice_data(data=data_rho[:, :, :, :], dimension_data=data_longitude[:], value=0)
    data_rho_tmp, latitude = slice_data(data=data_rho_tmp, dimension_data=data_latitude[:], value=0)

    # Compute Tcond CO2 from pressure
    data_tcondco2 = tcondco2(data_pressure=None, data_temperature=data_tmp, data_rho=data_rho_tmp)

    # Mean for each local time and subtract tcondco2
    data_final = zeros((len(data_local_time), data_tmp.shape[1]))
    data_tcondco2_final = zeros((len(data_local_time), data_tmp.shape[1]))

    for i in range(len(data_local_time)):
        data_final[i, :] = mean(data_tmp[i::len(data_local_time), :], axis=0)
        data_tcondco2_final[i, :] = mean(data_tcondco2[i::len(data_local_time), :], axis=0)

    data_p = ma.masked_values(data_final, 0.)
    data_tcondco2_p = ma.masked_values(data_tcondco2_final, 0.)
    for i in range(len(data_local_time)):
        data_p[i, :] = data_p[i, :] - data_tcondco2_p[i, :]

    del data, data_tmp

    print(f'T-Tcondco2: min = {min(data_p):.2f}, max = {max(data_p):.2f}')
    return data_p, data_local_time


def temp_gg2011_fig7(filename, data):
    data_time = getdata(filename=filename, target='Time')
    data_altitude = getdata(filename=filename, target='altitude')
    data_rho = getdata(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areroid (A) must be performed
    if data_altitude.long_name != 'Altitude above areoid':
        print(data_altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()
    data_surface_local = data_altitude[:]

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(data_time[:], selected_time=16)
    if not stats_file:
        print('This file is not a stats file required to compare with GG2011')
        exit()
    data = data[idx, :, :, :]
    data_rho = data_rho[idx, :, :, :]

    # Compute Tcond CO2 from pressure, ensure that was zrecasted
    data_tcondco2 = tcondco2(data_pressure=None, data_temperature=data, data_rho=data_rho)

    # Mean for each local time and subtract tcondco2
    data = mean(data, axis=2)
    data_tcondco2 = mean(data_tcondco2, axis=2)

    data_final = zeros((data.shape[0], data.shape[1]))
    data_final = ma.masked_values(data_final, 0.)
    for i in range(data_final.shape[1]):
        data_final[:, i] = data[:, i] - data_tcondco2[:, i]

    del data

    print('T-Tcondco2: min = {:.2f}, max = {:.2f}'.format(min(data_final), max(data_final)))
    return data_final, data_surface_local


def temp_gg2011_fig8(filename, data):
    data_time = getdata(filename=filename, target='Time')
    data_altitude = getdata(filename=filename, target='altitude')

    # Slice data: Ls=0-30°
    data, ls = slice_data(data=data, dimension_data=data_time[:], value=[0, 30])
    data_zonal_mean = mean(data, axis=3)

    # Check the kind of zrecast have been performed : above areroid (A) must be performed
    if data_altitude.long_name != 'Altitude above areoid':
        print(data_altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()

    # Check local time available and slice data at 0, 12, 16 H
    data_local_time, idx_0, statsfile = check_local_time(data_time=data_time[:], selected_time=0)
    data_zonal_mean_0h = data_zonal_mean[idx_0::len(data_local_time), :, :]
    data_local_time, idx_12, statsfile = check_local_time(data_time=data_time[:], selected_time=12)
    data_zonal_mean_12h = data_zonal_mean[idx_12::len(data_local_time), :, :]
    data_local_time, idx_16, statsfile = check_local_time(data_time=data_time[:], selected_time=16)
    data_zonal_mean_16h = data_zonal_mean[idx_16::len(data_local_time), :, :]

    # Mean
    data_zonal_mean_0h = mean(data_zonal_mean_0h, axis=0)
    data_zonal_mean_12h = mean(data_zonal_mean_12h, axis=0)
    data_zonal_mean_16h = mean(data_zonal_mean_16h, axis=0)

    # 12h - 00h
    data_thermal_tide = data_zonal_mean_12h - data_zonal_mean_0h

    return data_zonal_mean_16h, data_thermal_tide


def temp_gg2011_fig9(filename, data):
    from numpy import mean

    data_time = getdata(filename=filename, target='Time')
    data_latitude = getdata(filename=filename, target='latitude')
    data_altitude = getdata(filename=filename, target='altitude')
    data_rho = getdata(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areroid (A) must be performed
    if data_altitude.long_name != 'Altitude above areoid':
        print(data_altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()
    data_surface_local = data_altitude[:]

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(data_time[:], selected_time=16)
    if not stats_file:
        print('This file is not a stats file required to compare with GG2011')
        exit()
    data = data[idx, :, :, :]
    data_rho = data_rho[idx, :, :, :]

    # Slice data at 0°N latitude
    data, latitude = slice_data(data=data, dimension_data=data_latitude[:], value=0)
    data_rho, latitude = slice_data(data=data_rho, dimension_data=data_latitude[:], value=0)

    # Compute Tcond CO2 from pressure, ensure that was zrecasted
    data_tcondco2 = tcondco2(data_pressure=None, data_temperature=data, data_rho=data_rho)

    # Mean for each local time and subtract tcondco2
    data_final = zeros(data.shape)
    for i in range(data_final.shape[1]):
        data_final[:, i] = data[:, i] - data_tcondco2[:, i]
    data_final = ma.masked_values(data_final, 0.)

    del data

    print('T-Tcondco2: min = {:.2f}, max = {:.2f}'.format(min(data_final), max(data_final)))
    return data_final, data_surface_local


def temp_thermal_structure_polar_region(filename, data):
    data_latitude = getdata(filename=filename, target='latitude')

    data_north, latitude_north = slice_data(data=data, dimension_data=data_latitude[:], value=60)
    data_south, latitude_south = slice_data(data=data, dimension_data=data_latitude[:], value=-60)

    data_north = mean(data_north, axis=2)
    data_south = mean(data_south, axis=2)

    return data_north, data_south


def temp_cold_pocket(filename, data):
    data_altitude = getdata(filename=filename, target='altitude')
    if data_altitude.units != 'Pa':
        print('Stop ! File did not zrecasted')
        exit()

    zonal_mean = ma.masked_values(data, 0.)
    data_tcondco2 = tcondco2(data_pressure=data_altitude[:])
    delta_temp = zeros(zonal_mean.shape)
    for ls in range(zonal_mean.shape[0]):
        for lat in range(zonal_mean.shape[2]):
            for lon in range(zonal_mean.shape[3]):
                delta_temp[ls, :, lat, lon] = zonal_mean[ls, :, lat, lon] - data_tcondco2[:]
    delta_temp = ma.masked_values(delta_temp, 0.)
    print('Cold pocket?', delta_temp[where(delta_temp < 0)])
    print(min(delta_temp), max(delta_temp))
    return


def vars_max_value_with_others(filename, data_target):
    shape_data_target = data_target.shape

    print('Get max value of {} in progress...'.format(data_target.name))
    max_mmr, x, y = get_extrema_in_alt_lon(data_target, extrema='max')
    del data_target
    print('Extract other variable at co2_ice max value:')

    print(' (1) Temperature')
    data_temperature = getdata(filename, target='temp')[:, :, :, :]
    max_temp = extract_at_max_co2_ice(data_temperature, x, y, shape_data_target)
    del data_temperature

    print(' (2) Saturation')
    data_satuco2 = getdata(filename, target='satuco2')[:, :, :, :]
    max_satu = extract_at_max_co2_ice(data_satuco2, x, y, shape_data_target)
    del data_satuco2

    print(' (3) CCN radius')
    data_riceco2 = getdata(filename, target='riceco2')[:, :, :, :]
    max_radius = extract_at_max_co2_ice(data_riceco2, x, y, shape_data_target)
    del data_riceco2

    print(' (4) CCN number')
    data_ccnNco2 = getdata(filename, target='ccnNco2')[:, :, :, :]
    max_ccnN = extract_at_max_co2_ice(data_ccnNco2, x, y, shape_data_target)
    del data_ccnNco2

    print(' (5) Altitude')
    data_altitude = getdata(filename, target='altitude')
    max_alt = extract_at_max_co2_ice(data_altitude, x, y, shape_data_target)

    print('Reshape data in progress...')
    max_mmr, max_temp, max_satu, max_radius, max_ccnN, max_alt = rotate_data(max_mmr, max_temp, max_satu,
                                                                             max_radius, max_ccnN, max_alt,
                                                                             doflip=True)

    return max_mmr, max_temp, max_satu, max_radius, max_ccnN, max_alt


def vars_time_mean(filename, data, duration, localtime=None):
    from math import ceil

    data_time = getdata(filename=filename, target='Time')

    if localtime is not None:
        data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=localtime)
        if data_time[-1] <= 360.: # Ensure we are in ls time coordinate
            data_time = data_time[idx::len(data_local_time)]
        else:
            data_ls = getdata(filename='../concat_Ls.nc', target='Ls')
            data_time = data_ls[idx::len(data_local_time)]

    nbin = ceil(data_time[-1] / duration)
    data_mean = zeros((nbin, data.shape[1], data.shape[2]))
    time_bin = arange(0, data_time[-1] + duration, duration)

    for i in range(nbin):
        data_sliced, time = slice_data(data=data, dimension_data=data_time[:], value=[duration * i, duration * (i + 1)])
        print(f'{time[0]:.2f} - {time[-1]:.2f}')
        data_mean[i, :, :] = mean(data_sliced, axis=0)

    return data_mean, time_bin


def vars_zonal_mean(filename, data, layer=None, flip=None):
    if layer is not None:
        if filename != '':
            data_altitude = getdata(filename=filename, target='altitude')
            if data_altitude.units in ['Pa']:
                data_altitude = data_altitude[::-1]  # in pressure coordinate, the direction is reversed
                data = data[:, ::-1, :, :]
            data, layer_selected = slice_data(data, dimension_data=data_altitude[:], value=float(data_altitude[layer]))
        else:
            # for observational data
            data = data[:, layer, :, :]
            layer_selected = None
    else:
        layer_selected = None

    zonal_mean = mean(data[:, :, :], axis=2)
    if flip is None:
        zonal_mean = rotate_data(zonal_mean, doflip=True)[0]
    else:
        zonal_mean = rotate_data(zonal_mean, doflip=False)[0]
    del data

    return zonal_mean, layer_selected


def vars_zonal_mean_column_density(filename, data_target):
    data_target, altitude_limit, zmin, zmax, altitude_unit = compute_column_density(filename=filename, data=data_target)

    # compute zonal mean column density
    data_target = mean(data_target, axis=2)  # Ls function of lat

    data_target = rotate_data(data_target, doflip=True)[0]

    return data_target, altitude_limit, zmin, zmax, altitude_unit


def vars_zonal_mean_where_co2ice_exists(filename, data, polar_region):
    data_where_co2_ice = extract_where_co2_ice(filename, data)

    if polar_region:
        # Slice data in north and south polar regions
        data_latitude = getdata(filename=filename, target='latitude')
        data_where_co2_ice_np, north_latitude = slice_data(data=data_where_co2_ice, dimension_data=data_latitude[:],
                                                           value=[45, 90])
        data_where_co2_ice_sp, south_latitude = slice_data(data=data_where_co2_ice, dimension_data=data_latitude[:],
                                                           value=[-45, -90])

        data_where_co2_ice_np_mean = mean(data_where_co2_ice_np, axis=3)
        data_where_co2_ice_sp_mean = mean(data_where_co2_ice_sp, axis=3)
        list_data = ([data_where_co2_ice_np_mean, data_where_co2_ice_sp_mean])
        del data_where_co2_ice, data_where_co2_ice_np, data_where_co2_ice_sp

    else:
        data_where_co2_ice = mean(data_where_co2_ice, axis=3)
        list_data = ([data_where_co2_ice])

    return list_data


def vars_zonal_mean_in_time_co2ice_exists(filename, data, data_name, density=False):
    lat1 = int(input('\t Latitude range 1 (°N): '))
    lat2 = int(input('\t Latitude range 2 (°N): '))

    # extract co2_ice data
    data_co2_ice = getdata(filename, target='co2_ice')

    # select the latitude range
    data_latitude = getdata(filename, target='latitude')
    data_sliced_lat, latitude_selected = slice_data(data[:, :, :, :], data_latitude, value=[lat1, lat2])
    data_co2_ice_sliced_lat, latitude_selected = slice_data(data_co2_ice[:, :, :, :], data_latitude,
                                                            value=[lat1, lat2])
    if density:
        data_rho = getdata(filename, target='rho')
        data_rho_sliced_lat, latitude_selected = slice_data(data_rho[:, :, :, :], data_latitude, value=[lat1, lat2])
        data_tau = getdata(filename, target='Tau3D1mic')
        data_tau_sliced_lat, latitude_selected = slice_data(data_tau[:, :, :, :], data_latitude, value=[lat1, lat2])
        del data_rho, data_tau
    del data, data_co2_ice

    # select the time range
    data_time = getdata(filename=filename, target='Time')
    print('')
    print(f'Time range: {data_time[0]:.2f} - {data_time[-1]:.2f} {data_time.units}')
    breakdown = input('Do you want compute mean radius over all the time (Y/n)?')

    if breakdown.lower() in ['y', 'yes']:
        # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
        data_final = ma.masked_where(data_co2_ice_sliced_lat < 1e-13, data_sliced_lat)
        if density:
            data_rho_final = ma.masked_where(data_co2_ice_sliced_lat < 1e-13, data_rho_sliced_lat)
            data_final = data_final * data_rho_final
            data_tau_final = ma.masked_where(data_co2_ice_sliced_lat < 1e-13, data_tau_sliced_lat)
            del data_rho_sliced_lat, data_tau_sliced_lat
            data_tau_final = mean(mean(data_tau_final, axis=3), axis=0)
            list_tau = list([data_tau_final])
        del data_co2_ice_sliced_lat, data_sliced_lat

        data_final = mean(mean(data_final, axis=3), axis=0) * 1e6  # zonal mean and temporal mean, and m to µm
        list_data = list([data_final])
        filenames = list([f'{data_name}_mean_{latitude_selected[0]:.0f}N_{latitude_selected[-1]:.0f}N_0-360Ls'])
        list_time_selected = list([data_time[0], data_time[-1]])
    else:
        directory_output = f'{data_name}_mean_radius_{latitude_selected[0]:.0f}N_{latitude_selected[-1]:.0f}N'
        try:
            mkdir(directory_output)
        except:
            pass

        timestep = float(input(f'Select the time step range ({data_time.units}): '))
        nb_step = int(data_time[-1] / timestep) + 1
        print(f'nb_step: {nb_step}')
        if data_time[-1] % timestep != 0:
            print(f'data_time[-1]%timestep = {data_time[-1] % timestep}')

        list_data = list([])
        filenames = list([])
        list_time_selected = list([])
        list_tau = list([])
        for i in range(nb_step):
            data_sliced_lat_ls, time_selected = slice_data(data_sliced_lat, dimension_data=data_time[:],
                                                           value=[i * timestep, (i + 1) * timestep])
            data_co2_ice_sliced_lat_ls, time_selected = slice_data(data_co2_ice_sliced_lat, dimension_data=data_time[:],
                                                                   value=[i * timestep, (i + 1) * timestep])

            print(f'\t\tselected: {time_selected[0]:.0f} {time_selected[-1]:.0f}')
            list_time_selected.append([time_selected[0], time_selected[-1]])
            # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
            data_final = ma.masked_where(data_co2_ice_sliced_lat_ls < 1e-13, data_sliced_lat_ls)
            if density:
                data_rho_sliced_lat_ls, time_selected = slice_data(data_rho_sliced_lat, dimension_data=data_time[:],
                                                                   value=[i * timestep, (i + 1) * timestep])
                data_rho_final = ma.masked_where(data_co2_ice_sliced_lat_ls < 1e-13, data_rho_sliced_lat_ls)
                data_final = data_final * data_rho_final

                data_tau_sliced_lat_ls, time_selected = slice_data(data_tau_sliced_lat, dimension_data=data_time[:],
                                                                   value=[i * timestep, (i + 1) * timestep])
                data_tau_final = ma.masked_where(data_co2_ice_sliced_lat_ls < 1e-13, data_tau_sliced_lat_ls)
                data_tau_final = mean(mean(data_tau_final, axis=3), axis=0)
                list_tau.append(data_tau_final)
                del data_rho_sliced_lat_ls, data_tau_sliced_lat_ls

            del data_co2_ice_sliced_lat_ls, data_sliced_lat_ls

            data_final = mean(mean(data_final, axis=3), axis=0) * 1e6  # zonal mean and temporal mean, and m to µm
            list_data.append(data_final)
            filenames.append(directory_output + '/{}_mean_{:.0f}N_{:.0f}N_Ls_{:.0f}-{:.0f}'.format(data_name,
                                                                                                   latitude_selected[0],
                                                                                                   latitude_selected[
                                                                                                       -1],
                                                                                                   time_selected[0],
                                                                                                   time_selected[-1]))

        del data_sliced_lat, data_co2_ice_sliced_lat

    return list_data, filenames, latitude_selected, list_time_selected, list_tau


def vars_select_profile(data_target):
    print('To be done')
    print('Select latitude, longitude, altitude, time to extract profile')
    print('Perform a list of extracted profile')
