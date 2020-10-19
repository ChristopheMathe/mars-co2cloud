from numpy import mean, abs, min, max, zeros, where, ones, concatenate, flip, arange, unravel_index, argmax, array, \
    count_nonzero, std, savetxt, c_, append, loadtxt, asarray, power, logspace, ma
from .lib_function import *
from .ncdump import getdata, getfilename
from sys import exit

def vars_max_value_with_others(data_target):

    shape_data_target = data_target.shape

    print('Get max value of {} in progress...'.format(data_target.name))
    max_mmr, x, y = get_extrema_in_alt_lon(data_target, extrema='max')
    del data_target
    print('Extract other variable at co2_ice max value:')

    print(' (1) Temperature')
    data_temperature = getdata(directory_store + filename, target='temp')[:, :, :, :]
    max_temp = extract_at_max_co2_ice(data_temperature, x, y, shape_data_target)
    del data_temperature

    print(' (2) Saturation')
    data_satuco2 = getdata(directory_store + filename, target='satuco2')[:, :, :, :]
    max_satu = extract_at_max_co2_ice(data_satuco2, x, y, shape_data_target)
    del data_satuco2

    print(' (3) CCN radius')
    data_riceco2 = getdata(directory_store + filename, target='riceco2')[:, :, :, :]
    max_radius = extract_at_max_co2_ice(data_riceco2, x, y, shape_data_target)
    del data_riceco2

    print(' (4) CCN number')
    data_ccnNco2 = getdata(directory_store + filename, target='ccnNco2')[:, :, :, :]
    max_ccnN = extract_at_max_co2_ice(data_ccnNco2, x, y, shape_data_target)
    del data_ccnNco2

    print(' (5) Altitude')
    data_altitude = getdata(directory_store + filename, target='altitude')
    max_alt = extract_at_max_co2_ice(data_altitude, x, y, shape_data_target)

    print('Reshape data in progress...')
    max_mmr, max_temp, max_satu, max_radius, max_ccnN, max_alt = rotate_data(max_mmr, max_temp, max_satu,
                                                                                  max_radius, max_ccnN, max_alt,
                                                                             doflip=True)

    print('Linearized data in progress...')
    if lslin:
        max_mmr = linearize_ls(max_mmr, shape_data_target[0], shape_data_target[2], interp_time)
        max_temp = linearize_ls(max_temp, shape_data_target[0], shape_data_target[2], interp_time)
        max_satu = linearize_ls(max_satu, shape_data_target[0], shape_data_target[2], interp_time)
        max_radius = linearize_ls(max_radius, shape_data_target[0], shape_data_target[2], interp_time)
        max_ccnN = linearize_ls(max_ccnN, shape_data_target[0], shape_data_target[2], interp_time)
        max_alt = linearize_ls(max_alt, shape_data_target[0], shape_data_target[2], interp_time)

    return max_mmr, max_temp, max_satu, max_radius, max_ccnN, max_alt


def vars_zonal_mean(data):
    zonal_mean = mean(data[:, :, :], axis=2)
    zonal_mean = rotate_data(zonal_mean, doflip=True)

    del data

    zonal_mean = correction_value(zonal_mean[0], threshold=1e-13)

    return zonal_mean


def vars_zonal_mean_column_density(filename, data_target):
    data_altitude = getdata(filename, target='altitude')

    if data_altitude.units in ['m', 'km']:
        try:
            data_pressure = getdata(filename, target='pressure')
        except:
            data_pressure = getdata('concat_sols_vars_S.nc', target='pressure')
    else:
        data_pressure = data_altitude

    data_target, altitude_limit, zmin, zmax = compute_zonal_mean_column_density(data_target, data_pressure,
                                                                                data_altitude)

    data_target = rotate_data(data_target, doflip=True)
    data_target = asarray(data_target[0])

    del data_pressure

    return data_target, altitude_limit, zmin, zmax


def vars_zonal_mean_in_time_co2ice_exists(data, data_co2ice):

    # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
    data = ma.masked_where(data_co2ice < 1e-13, data)
    del data_co2ice

    data = mean(mean(data, axis=3), axis=0)  # zonal mean and temporal mean

    return data


def vars_select_profile(data_target):
    print('To be done')
    print('Select latitude, longitude, altitude, time to extract profile')
    print('Perform a list of extracted profile')


def vars_max_value_day_night_with_altitude(files, directory_store, filename, variable_target, data):
    print('Select the night file:')
    filename_2 = getfilename(files)
    data_2 = getdata(directory_store + filename_2, variable_target)

    print('Day file is {}'.format(filename))
    print('Night file is {}'.format(filename_2))
    print('')

    print('Correction value in progress...')
    data_2 = correction_value(data_2[:, :, :, :])

    data_altitude = getdata(directory_store + filename_2, 'altitude')

    print('Get max of {}...'.format(variable_target))
    max_day, idx_altitude_day, idx_longitude_day = get_extrema_in_alt_lon(data, extrema='max')
    max_night, idx_altitude_night, idx_longitude_night = get_extrema_in_alt_lon(data_2, extrema='max')

    max_alt_day = ones(idx_altitude_day.shape)
    max_alt_night = ones(idx_altitude_night.shape)

    for i in range(idx_altitude_night.shape[0]):
        for j in range(idx_altitude_night.shape[1]):
            max_alt_night[i, j] = data_altitude[idx_altitude_night[i, j]]
            max_alt_day[i, j] = data_altitude[idx_altitude_day[i, j]]
            if max_alt_day[i, j] == 0 and max_day[i, j] < 1e-13:
                max_alt_day[i, j] = None
                max_day[i, j] = None
            if max_alt_night[i, j] == 0 and max_night[i, j] < 1e-13:
                max_alt_night[i, j] = None
                max_night[i, j] = None

    return max_day, max_night, max_alt_day, max_alt_night


def vars_concat_localtime(directory_store, filename, data):
    data_longitude = getdata(directory_store + filename, target='longitude')

    data_target = slice_data(data_target, data_latitude, value=0)

    data_target = slice_data(data_target, data_longitude, value=0)

    data_target = slice_data(data_target, data_time, value=[259,318])  # here time is in sols

    shape_data_target = data_target.shape
    data_target = mean(data_target.reshape(-1, 59), axis=1)
    data_target = data_target.reshape(int(shape_data_target[0] / 59), shape_data_target[1])

    filename_2 = getfilename(files)
    print('Day file is {}'.format(filename))
    print('Night file is {}'.format(filename_2))
    print('')

    data_target_2 = getdata(directory_store + filename_2, variable_target)
    print('Correction value...')
    data_target_2 = libf.correction_value(data_target_2[:, :, :, :])
    data_altitude = getdata(directory_store + filename_2, 'altitude')
    data_time_2 = getdata(directory_store + filename_2, 'Time')

    idx_ls_1 = (abs(data_time_2[:] - 259)).argmin()  # here time is in sols
    idx_ls_2 = (abs(data_time_2[:] - 318)).argmin()
    data_target_2 = data_target_2[idx_ls_1:idx_ls_2, :, idx_lat, idx_lon]
    shape_data_target_2 = data_target_2.shape
    data_target_2 = mean(data_target_2.reshape(-1, 59), axis=1)
    data_target_2 = data_target_2.reshape(int(shape_data_target_2[0] / 59), shape_data_target_2[1])

    # 00-4h + 6-18h + 20-22h
    data_target = concatenate((data_target_2[3:-1, :], data_target, data_target_2[1:3, :]))

    return data, data_altitude


def co2ice_thickness_atm_layer(data_target):
    # la plus grande épaisseur de neige de co2 pendant l'hiver polaire
    data_altitude = getdata(directory_store + filename, 'altitude')

    idx_lat_N = (abs(data_latitude[:] - 60)).argmin()
    idx_lat_S = (abs(data_latitude[:] + 60)).argmin()
    print(data_target.shape)

    data_target_north = data_target[:, :, idx_lat_N:, :]  # north pole
    data_target_south = data_target[:, :, :idx_lat_S + 1, :]  # south pole
    print(data_latitude[idx_lat_N:])
    print(data_latitude[:idx_lat_S + 1])

    # bin ls to 5°
    if max(data_time) < 360:
        ls_bin = arange(0, 365, 5)
    else:
        # sols binned at 5° Ls
        ls_bin = libf.convert_sols_tols()

    nbin = ls_bin.shape[0]
    data_icelayer = zeros((2, nbin))

    for bin in range(nbin - 1):
        idx = (abs(data_time[:] - ls_bin[bin])).argmin()
        idx2 = (abs(data_time[:] - ls_bin[bin + 1])).argmin()
        data_binned_north = data_target_north[idx:idx2, :, :, :]
        data_binned_south = data_target_south[idx:idx2, :, :, :]
        if (max(data_binned_north) >= 1e-10):
            ind = unravel_index(argmax(data_binned_north, axis=None), data_binned_north.shape)
            for i in range(ind[1], data_altitude.shape[0]):
                if data_binned_north[ind[0], i, ind[2], ind[3]] >= 1e-10:
                    idx_max = i
            data_icelayer[0, bin] = data_altitude[idx_max] - data_altitude[ind[1]]
        if (max(data_binned_south) >= 1e-10):
            ind = unravel_index(argmax(data_binned_south, axis=None), data_binned_south.shape)
            for i in range(ind[1], data_altitude.shape[0]):
                if data_binned_south[ind[0], i, ind[2], ind[3]] >= 1e-10:
                    idx_max = i
            data_icelayer[1, bin] = data_altitude[idx_max] - data_altitude[ind[1]]

    return data_icelayer


def co2ice_polar_cloud_distribution():
    data_altitude = getdata(directory_store + filename, 'altitude')

    idx_lat_N = (abs(data_latitude[:] - 60)).argmin()
    idx_lat_S = (abs(data_latitude[:] + 60)).argmin()

    data_target_north = data_target[:, :, idx_lat_N:, :]  # north pole
    data_target_south = data_target[:, :, :idx_lat_S + 1, :]  # south pole

    idx_ls_1 = (abs(data_time[:] - 225)).argmin()  # ls = 104°
    idx_ls_2 = (abs(data_time[:] - 669)).argmin()
    data_target_north = data_target_north[idx_ls_1:idx_ls_2 + 1, :, :, :]
    data_target_south = data_target_south[idx_ls_1:idx_ls_2 + 1, :, :, :]

    distribution_north = zeros((data_target_north.shape[2], data_target_north.shape[1]))
    distribution_south = zeros((data_target_north.shape[2], data_target_north.shape[1]))

    for latitude in range(data_target_north.shape[2]):
        for altitude in range(data_target_north.shape[1]):
            distribution_north[latitude, altitude] = count_nonzero(
                data_target_north[:, altitude, latitude, :] >= 1e-10)
            distribution_south[latitude, altitude] = count_nonzero(
                data_target_south[:, altitude, latitude, :] >= 1e-10)

    return distribution_north, distribution_south


def co2ice_cloud_evolution(data):
    data_altitude = getdata(directory_store + filename, target='altitude')
    data_longitude = getdata(directory_store + filename, target='longitude')

    print('Select the latitude region (°N):')
    input_latitude_1 = int(input('   latitude 1: '))
    input_latitude_2 = int(input('   latitude 2: '))
    idx_lat_1 = (abs(data_latitude[:] - input_latitude_1)).argmin()
    idx_lat_2 = (abs(data_latitude[:] - input_latitude_2)).argmin()

    if idx_lat_1 > idx_lat_2:
        tmp = idx_lat_2
        idx_lat_2 = idx_lat_1
        idx_lat_1 = tmp

    data_target = flip(data_target, axis=2)
    idx_max = unravel_index(argmax(data_target[:, :, idx_lat_1:idx_lat_2 + 1, :], axis=None),
                            data_target[:, :, idx_lat_1:idx_lat_2 + 1, :].shape)
    idx_max = asarray(idx_max)
    idx_max[2] += idx_lat_1

    data_target = data_target[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]

    data_satuco2 = getdata(directory_store + filename, 'satuco2')
    data_satuco2 = data_satuco2[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
    data_satuco2 = flip(data_satuco2, axis=2)

    data_temp = getdata(directory_store + filename, 'temp')

    print('Temperature profile save in progres...')
    savetxt('temperature_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                                     data_latitude[idx_max[2]],
                                                                     data_longitude[idx_max[3]]),
            c_[data_temp[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
    print('Done.')

    print('Pressure profile save in progres...')
    savetxt('pressure_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                                  data_latitude[idx_max[2]],
                                                                  data_longitude[idx_max[3]]),
            c_[data_pressure[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
    print('Done.')

    print('Dust profile save in progres...')
    data_dust = getdata(directory_store + 'concat_sols_dustq_S.nc', target='dustq')
    savetxt('dust_profile_sols_{}_lat_{}N_lon_{}E.txt'.format(int(data_time[idx_max[0] - 25]),
                                                              data_latitude[idx_max[2]],
                                                              data_longitude[idx_max[3]]),
            c_[data_dust[idx_max[0] - 25:idx_max[0] + 25, :, idx_max[2], idx_max[3]]])
    print('Done.')

    data_temp = data_temp[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
    data_temp = flip(data_temp, axis=2)

    data_riceco2 = getdata(directory_store + filename, 'riceco2')
    data_riceco2 = data_riceco2[idx_max[0] - 3:idx_max[0] + 3, :, :, idx_max[3]]
    data_riceco2 = flip(data_riceco2, axis=2)

    print('the maximum is at :' + str(data_time[idx_max[0]] * 24 % 24) + 'h local time.')

    return


def riceco2_topcloud_altitude():
    data_altitude = getdata(directory_store + filename, target='altitude')
    data_altitude = data_altitude[:] / 1e3
    data_ccnNco2 = getdata(directory_store + filename, target='ccnNco2')
    filename_2 = getfilename(files)
    data_rho = getdata(directory_store + filename_2, target='rho')

    data_target = mean(data_target[:, :, :, :], axis=3)
    data_ccnNco2 = mean(data_ccnNco2[:, :, :, :], axis=3)
    data_rho = mean(data_rho[:, :, :, :], axis=3)

    N_reflect = 2e-8 * power(data_target * 1e6, -2)
    N_part = data_rho * data_ccnNco2
    nb_time = data_target.shape[0]
    nb_lat = data_target.shape[2]
    nb_alt = data_target.shape[1]
    del [data_target, data_ccnNco2, data_rho]

    top_cloud = zeros((nb_time, nb_lat))
    for t in range(nb_time):
        for lat in range(nb_lat):
            for alt in range(nb_alt - 1, -1, -1):
                if N_part[t, alt, lat] >= N_reflect[t, alt, lat]:
                    top_cloud[t, lat] = data_altitude[alt]
                    break

    idx_lat_1 = (abs(data_latitude[:] - 40)).argmin()
    idx_lat_2 = (abs(data_latitude[:] + 40)).argmin()

    if idx_lat_1 > idx_lat_2:
        tmp = idx_lat_1
        idx_lat_1 = idx_lat_2
        idx_lat_2 = tmp

    top_cloud[:, idx_lat_1:idx_lat_2 + 1] = None

    return top_cloud


def riceco2_max_day_night():
    filename_2 = getfilename(files)
    data_target_2 = getdata(directory_store + filename_2, 'riceco2')
    data_altitude = getdata(directory_store + filename_2, 'altitude')

    print('Compute max in progress...')
    max_satu_day, idx_altitude_day, y_day = libf.get_extrema_in_alt_lon(data_target, extrema='max')
    max_satu_night, idx_altitude_night, y_night = libf.get_extrema_in_alt_lon(data_target_2, extrema='max')

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


def satuco2_with_co2_ice(filename, data):
    data_latitude = getdata(filename=filename, target='latitude')
    data = correction_value(data[:,:,:,:], threshold=1e-13)

    # Slice data for the 3 latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data, dimension_data=data_latitude, value=50)
    data_satuco2_eq, eq_latitude_selected = slice_data(data, dimension_data=data_latitude, value=0)
    data_satuco2_south, south_latitude_selected = slice_data(data, dimension_data=data_latitude, value=-60)

    # Extract profile of satuco2 where the value is maximal along longitude for each time Ls
    data_satuco2_north, idx_lon_north = extract_vars_max_along_lon(data_satuco2_north)
    data_satuco2_eq, idx_lon_eq = extract_vars_max_along_lon(data_satuco2_eq)
    data_satuco2_south, idx_lon_south = extract_vars_max_along_lon(data_satuco2_south)

    del data

    # Get co2 ice mmr
    data_co2ice = getdata(filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:,:,:,:], threshold=1e-13)

    # Slice co2 ice mmr at these 3 latitudes
    data_co2ice_north, north_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=50)
    data_co2ice_eq, eq_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=0)
    data_co2ice_south, south_latitude_selected = slice_data(data_co2ice, dimension_data=data_latitude, value=-60)

    # Extract profile of co2 ice mmr at longitudes where the satuco2 was observed for each time Ls
    data_co2ice_north, idx_lon_north = extract_vars_max_along_lon(data_co2ice_north, idx_lon_north)
    data_co2ice_eq, idx_lon_eq = extract_vars_max_along_lon(data_co2ice_eq, idx_lon_eq)
    data_co2ice_south, idx_lon_south = extract_vars_max_along_lon(data_co2ice_south, idx_lon_south)

    del data_co2ice, idx_lon_north, idx_lon_eq, idx_lon_south

    data_satuco2_north = correction_value(data_satuco2_north, threshold=1e-13)
    data_satuco2_eq = correction_value(data_satuco2_eq, threshold=1e-13)
    data_satuco2_south = correction_value(data_satuco2_south, threshold=1e-13)
    data_co2ice_north = correction_value(data_co2ice_north, threshold=1e-13)
    data_co2ice_eq = correction_value(data_co2ice_eq, threshold=1e-13)
    data_co2ice_south = correction_value(data_co2ice_south, threshold=1e-13)

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
        if data_time.shape[0]%60 == 0:
            print('Test 5°Ls binning: {} - {}'.format(data_time[0], data_time[60]))
        else:
            print('The data will not be binned in 5°Ls, need to work here')

        nb_bin = int(data_time.shape[0]/60)
        data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
        data_satuco2_eq_binned = zeros((nb_bin, data_satuco2_eq.shape[1]))
        data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
        data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
        data_co2ice_eq_binned = zeros((nb_bin, data_co2ice_eq.shape[1]))
        data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))
        print(min(data_satuco2_north))
        print(min(data_satuco2_north_binned))
        for i in range(nb_bin):
            data_satuco2_north_binned[i, :] = mean(data_satuco2_north[i*60:(i+1)*60, :], axis=0)
            data_satuco2_eq_binned[i, :] = mean(data_satuco2_eq[i*60:(i+1)*60, :], axis=0)
            data_satuco2_south_binned[i, :] = mean(data_satuco2_south[i*60:(i+1)*60, :], axis=0)
            data_co2ice_north_binned[i, :] = mean(data_co2ice_north[i*60:(i+1)*60, :], axis=0)
            data_co2ice_eq_binned[i, :] = mean(data_co2ice_eq[i*60:(i+1)*60, :], axis=0)
            data_co2ice_south_binned[i, :] = mean(data_co2ice_south[i*60:(i+1)*60, :], axis=0)
        print(min(data_satuco2_north_binned))

    del data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, data_co2ice_south

    data_satuco2_north_binned, data_satuco2_eq_binned, data_satuco2_south_binned, data_co2ice_north_binned, \
    data_co2ice_eq_binned, data_co2ice_south_binned = rotate_data(data_satuco2_north_binned, data_satuco2_eq_binned,
                                                                  data_satuco2_south_binned, data_co2ice_north_binned,
                                                                  data_co2ice_eq_binned, data_co2ice_south_binned,
                                                                  doflip=False)

    data_satuco2_north_binned = correction_value(data_satuco2_north_binned, threshold=1e-13)
    data_satuco2_eq_binned = correction_value(data_satuco2_eq_binned, threshold=1e-13)
    data_satuco2_south_binned = correction_value(data_satuco2_south_binned, threshold=1e-13)
    data_co2ice_north_binned = correction_value(data_co2ice_north_binned, threshold=1e-13)
    data_co2ice_eq_binned = correction_value(data_co2ice_eq_binned, threshold=1e-13)
    data_co2ice_south_binned = correction_value(data_co2ice_south_binned, threshold=1e-13)

    return data_satuco2_north_binned, data_satuco2_eq_binned, data_satuco2_south_binned, data_co2ice_north_binned, \
           data_co2ice_eq_binned, data_co2ice_south_binned, north_latitude_selected, eq_latitude_selected, \
           south_latitude_selected
