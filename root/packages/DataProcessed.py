from numpy import mean, abs, min, max, zeros, where, arange, unravel_index, argmin, argmax, array, \
    count_nonzero, std, append, asarray, power, ma, reshape, swapaxes, log, exp, concatenate, amin, amax, diff
from .lib_function import *
from .ncdump import get_data, getfilename
from os import mkdir, path
from sys import exit
from .constant_parameter import cst_stefan, threshold


def co2ice_at_viking_lander_site(filename, data):
    data_time, list_var = get_data(filename=filename, target='Time')
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_longitude, list_var = get_data(filename=filename, target='longitude')
    data_area = gcm_area()

    # Viking 1: (22.27°N, 312.05°E so -48°E) near Chryse Planitia
    # https://nssdc.gsfc.nasa.gov/planetary/viking.html
    data_at_viking1, idx_latitude1 = slice_data(data=data, dimension_data=data_latitude[:], value=22)
    data_at_viking1, idx_longitude1 = slice_data(data=data_at_viking1, dimension_data=data_longitude[:], value=-48)
    data_area_at_viking1, idx_latitude1 = slice_data(data=data_area, dimension_data=data_latitude[:], value=22)
    data_area_at_viking1, idx_longitude1 = slice_data(data=data_area_at_viking1, dimension_data=data_longitude[:],
                                                      value=-48)

    # Viking 2:  (47.67°N, 134.28°E) near Utopia Planitia
    data_at_viking2, idx_latitude2 = slice_data(data=data, dimension_data=data_latitude[:], value=48)
    data_at_viking2, idx_longitude2 = slice_data(data=data_at_viking2, dimension_data=data_longitude[:], value=134)
    data_area_at_viking2, idx_latitude2 = slice_data(data=data_area, dimension_data=data_latitude[:], value=48)
    data_area_at_viking2, idx_longitude2 = slice_data(data=data_area_at_viking2, dimension_data=data_longitude[:],
                                                      value=134)

    data_at_viking1 = data_at_viking1 * data_area_at_viking1
    data_at_viking2 = data_at_viking2 * data_area_at_viking2

    return data_at_viking1, data_at_viking2, data_time


def co2ice_polar_cloud_distribution(filename, data, normalization, local_time):
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    if data_altitude.long_name != 'Altitude above areoid':
        print('Data did not zrecasted above the areoid')
        print(f'\tCurrent: {data_altitude.long_name}')
        exit()

    # sliced data on latitude region
    data_latitude, list_var = get_data(filename, target='latitude')
    data_north, latitude_north = slice_data(data, dimension_data=data_latitude, value=[60, 90])
    data_south, latitude_south = slice_data(data, dimension_data=data_latitude, value=[-60, -90])
    del data

    latitude_north = data_latitude[latitude_north[0]: latitude_north[1]]
    latitude_south = data_latitude[latitude_south[0]: latitude_south[1]]

    # sliced data between 104 - 360°Ls time to compare with Fig8 of Neumann et al. 2003
    data_time, list_var = get_data(filename, target='Time')
    if data_time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if len(local_time) == 1:
            data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=local_time)
            data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = data_ls[:]
    data_north, time_selected = slice_data(data_north, dimension_data=data_time, value=[104, 360])
    data_south, time_selected = slice_data(data_south, dimension_data=data_time, value=[104, 360])

    distribution_north = zeros((data_north.shape[1], data_north.shape[2]))
    distribution_south = zeros((data_south.shape[1], data_south.shape[2]))

    for latitude in range(data_north.shape[2]):
        for altitude in range(data_north.shape[1]):
            distribution_north[altitude, latitude] = count_nonzero(data_north[:, altitude, latitude, :] >= threshold)
            distribution_south[altitude, latitude] = count_nonzero(data_south[:, altitude, latitude, :] >= threshold)

    # normalisation
    if normalization:
        distribution_north = distribution_north / max(distribution_north) * 2000  # To compare with Fig.8 of Neumann2003
        distribution_south = distribution_south / max(distribution_south) * 2000

    return distribution_north, distribution_south, latitude_north, latitude_south


def co2ice_cloud_evolution(filename, data):
    data_latitude, list_var = get_data(filename, target='latitude')

    print('Select the latitude region (°N):')
    lat_1 = float(input('   latitude 1: '))
    lat_2 = float(input('   latitude 2: '))

    data, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[lat_1, lat_2])

    idx_max = unravel_index(argmax(data[:, :, :, :], axis=None), data[:, :, :, :].shape)
    idx_max = asarray(idx_max)
    data = data[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_satuco2, list_var = get_data(filename, 'satuco2')
    data_satuco2, latitude_selected = slice_data(data_satuco2, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_satuco2 = data_satuco2[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_temp, list_var = get_data(filename, 'temp')
    data_temp, latitude_selected = slice_data(data_temp, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_temp = data_temp[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_riceco2, list_var = get_data(filename, 'riceco2')
    data_riceco2, latitude_selected = slice_data(data_riceco2, dimension_data=data_latitude[:], value=[lat_1, lat_2])
    data_riceco2 = data_riceco2[idx_max[0] - 9:idx_max[0] + 3, :, :, idx_max[3]]

    data_time, list_var = get_data(filename, target='Time')
    print(f'the maximum is at: {data_time[idx_max[0]] * 24 % 24}h local time.')

    return data, data_satuco2, data_temp, data_riceco2, idx_max, latitude_selected


def co2ice_cloud_localtime_along_ls(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    latitude_min = float(input("Enter minimum latitude: "))
    latitude_max = float(input("Enter maximum latitude: "))
    data, latitude = slice_data(data=data, dimension_data=data_latitude[:], value=[latitude_min, latitude_max])

    data, altitude_limit, altitude_min, altitude_max, altitude_units = compute_column_density(filename=filename,
                                                                                              data=data)
    data = mean(data, axis=2)
    data = mean(data, axis=1)
    # Reshape every localtime for one year!
    if data.shape[0] % 12 != 0:
        nb_sol = 0
        print('Stop, there is no 12 localtime')
        exit()
    else:
        nb_sol = int(data.shape[0] / 12)  # if there is 12 local time!
    data = reshape(data, (nb_sol, 12)).T
    return data, altitude_min, latitude_min, latitude_max


def co2ice_cumulative_masses_polar_cap(filename, data):
    from numpy import sum

    ptimestep = 924.739583 * 8
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_north, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[60, 90])
    data_south, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[-60, -90])
    del data

    # get precip_co2_ice
    data_precip_co2_ice, list_var = get_data(filename=filename, target='precip_co2_ice_rate')
    data_precip_co2_ice = data_precip_co2_ice[:,:,:] * ptimestep
    data_precip_co2_ice_north, tmp = slice_data(data_precip_co2_ice[:, :, :], dimension_data=data_latitude[:],
                                                value=[60, 90])
    data_precip_co2_ice_south, tmp = slice_data(data_precip_co2_ice[:, :, :], dimension_data=data_latitude[:],
                                                value=[-60, -90])

    # Diurnal mean
    nb_lat = data_north.shape[1]
    nb_lon = data_north.shape[2]

    data_north = mean(data_north.reshape(669, 12, nb_lat, nb_lon), axis=1)
    data_south = mean(data_south.reshape(669, 12, nb_lat, nb_lon), axis=1)

    data_precip_co2_ice_north = mean(data_precip_co2_ice_north.reshape(669, 12, nb_lat, nb_lon), axis=1)
    data_precip_co2_ice_south = mean(data_precip_co2_ice_south.reshape(669, 12, nb_lat, nb_lon), axis=1)


    # diff co2ice car accumulé
    data_north = diff(data_north, axis=0)
    data_south = diff(data_south, axis=0)

    # sum over polar regions
    accumulation_precip_co2_ice_north = zeros(data_precip_co2_ice_north.shape[0])
    accumulation_precip_co2_ice_south = zeros(data_precip_co2_ice_north.shape[0])
    for ls in range(data_precip_co2_ice_north.shape[0]):
        accumulation_precip_co2_ice_north[ls] = sum(data_precip_co2_ice_north[ls, :, :])
        accumulation_precip_co2_ice_south[ls] = sum(data_precip_co2_ice_south[ls, :, :])

    accumulation_co2ice_north = zeros(data_north.shape[0])
    accumulation_co2ice_south = zeros(data_south.shape[0])
    for ls in range(data_north.shape[0]):
        accumulation_co2ice_north[ls] = sum(data_north[ls, :, :])
        accumulation_co2ice_south[ls] = sum(data_south[ls, :, :])

    accumulation_direct_condco2_north = accumulation_co2ice_north - accumulation_precip_co2_ice_north[1:]
    accumulation_direct_condco2_south = accumulation_co2ice_south - accumulation_precip_co2_ice_south[1:]

    return accumulation_co2ice_north, accumulation_co2ice_south, accumulation_precip_co2_ice_north, \
           accumulation_precip_co2_ice_south, accumulation_direct_condco2_north, accumulation_direct_condco2_south


def co2ice_time_mean(filename, data, duration, localtime, column=None):

    data, time = vars_time_mean(filename=filename, data=data, duration=duration, localtime=localtime)
    data = correction_value(data, operator='inf', threshold=threshold)
    if not column:
        data_area = gcm_area()
        data = data * data_area[:, :]
    else:
        data, altitude_limit, altitude_min, altitude_max, altitude_units = compute_column_density(filename=filename,
                                                                                                  data=data)

    return data, time


def co2ice_density_column_evolution(filename, data, localtime):
    from math import floor

    # Show the evolution of density column at winter polar region
    data_time, list_var = get_data(filename=filename, target='Time')
    if data_time.units == 'degrees':
        print('The netcdf file is in ls !')
        print(f'Time[0] = {data_time[0]}, Time[-1] = {data_time[-1]}')
        exit()

    # Slice in time:
    print(f'Select the time range in sols ({floor(data_time[0])} : {int(data_time[-1])})')
    time_begin = float(input('Start time: '))
    time_end = float(input('End time: '))
    data_time, localtime = extract_at_a_local_time(filename=filename, data=data_time, local_time=localtime)
    data, time_range = slice_data(data=data, dimension_data=data_time, value=[time_begin, time_end])

    # Slice data in polar region:
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    pole = input('Select the polar region (N/S):')
    if pole.lower() == 'n':
        data, latitude = slice_data(data=data, dimension_data=data_latitude, value=[60, 90])
    elif pole.lower() == 's':
        data, latitude = slice_data(data=data, dimension_data=data_latitude, value=[-60, -90])
    else:
        latitude = None
        print('Wrong selection')
        exit()
    data, altitude_limit, altitude_min, altitude_max, altitude_unit = compute_column_density(filename=filename,
                                                                                             data=data)

    return data, time_range, latitude


def co2ice_coverage(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_longitude, list_var = get_data(filename=filename, target='longitude')

    ntime = data.shape[0]
    nlat = data_latitude.shape[0]
    nlon = data_longitude.shape[0]

    data_altitude, list_var = get_data(filename=filename, target='altitude')
    idx_10pa = (abs(data_altitude[:] - 10)).argmin()

    data_co2ice_coverage = zeros((nlat, nlon))
    data_co2ice_coverage_meso = zeros((nlat, nlon))

    for lat in range(nlat):
        for lon in range(nlon):
            for ls in range(ntime):  # time
                if any(data[ls, :, lat, lon] > threshold):  # There at least one cell with co2_ice
                    data_co2ice_coverage[lat, lon] += 1
                    if any(data[ls, idx_10pa:, lat,
                           lon] > threshold):  # There at least one cell with co2_ice in mesosphere
                        data_co2ice_coverage_meso[lat, lon] = 1

    data_co2ice_coverage = correction_value(data=data_co2ice_coverage, operator='eq', threshold=0)
    data_co2ice_coverage_meso = correction_value(data=data_co2ice_coverage_meso, operator='eq', threshold=0)

    #  Normalization
    data_co2ice_coverage = (data_co2ice_coverage / ntime) * 100
    print(min(data_co2ice_coverage_meso), max(data_co2ice_coverage_meso))
    return data_co2ice_coverage, data_co2ice_coverage_meso


def emis_polar_winter_gg2020_fig13(filename, data, local_time):
    # Slice in time
    data_time, list_var = get_data(filename=filename, target='Time')
    if data_time.units != 'deg':
        data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=local_time)
        data_time, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if idx:
            data_time = data_time[idx::len(data_local_time)]

    #       NP: 180°-360°
    data_np, time = slice_data(data=data, dimension_data=data_time, value=[180, 360])

    #       SP: 0°-180°
    data_sp, time = slice_data(data=data, dimension_data=data_time, value=[0, 180])

    # Slice in latitude > 60°
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data_np, latitude = slice_data(data=data_np, dimension_data=data_latitude[:], value=[60, 90])
    data_sp, latitude = slice_data(data=data_sp, dimension_data=data_latitude[:], value=[-60, -90])

    # Mean in time
    data_mean_np = mean(data_np, axis=0)
    data_mean_sp = mean(data_sp, axis=0)

    return data_mean_np, data_mean_sp


def flux_lw_apparent_temperature_zonal_mean(data):
    # Flux = sigma T^4
    temperature_apparent = power(data / cst_stefan, 1 / 4)
    temperature_apparent = mean(temperature_apparent, axis=2).T
    return temperature_apparent


def h2o_ice_alt_ls_with_co2_ice(filename, data, local_time, directory, files):
    latitude = float(input('Which latitude (°N)? '))

    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data, idx_latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=latitude)

    if 'co2_ice' in list_var:
        data_co2_ice, list_var = get_data(filename=filename, target='co2_ice')
    else:
        filename_co2 = getfilename(files=files, selection=None)
        data_co2_ice, list_var = get_data(filename=directory + filename_co2, target='co2_ice')

    if len(local_time) == 1:
        data_co2_ice, tmp = extract_at_a_local_time(filename=filename, data=data_co2_ice, local_time=local_time)

    data_co2_ice, idx_latitude_selected = slice_data(data_co2_ice, dimension_data=data_latitude[:], value=latitude)
    data_co2_ice = correction_value(data_co2_ice, operator='inf', threshold=threshold)

    # zonal mean
    zonal_mean = mean(data, axis=2)  # zonal mean
    zonal_mean_co2_ice = mean(data_co2_ice, axis=2)
    if len(local_time) != 1:
        zonal_mean = mean(zonal_mean.reshape((669, 12, zonal_mean.shape[1])), axis=1)  # => sols, lon
        zonal_mean_co2_ice = mean(zonal_mean_co2_ice.reshape((669, 12, zonal_mean_co2_ice.shape[1])),
                                  axis=1)  # => sols,
    # lon

    zonal_mean, zonal_mean_co2_ice = rotate_data(zonal_mean, zonal_mean_co2_ice, do_flip=False)

    del data, data_co2_ice
    return zonal_mean, zonal_mean_co2_ice, data_latitude[idx_latitude_selected]


def ps_at_viking(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_longitude, list_var = get_data(filename=filename, target='longitude')

    # Viking 1: (22.27°N, 312.05°E so -48°E) near Chryse Planitia
    # https://nssdc.gsfc.nasa.gov/planetary/viking.html
    data_pressure_at_viking1, idx_latitude1 = slice_data(data=data, dimension_data=data_latitude[:], value=22)
    data_pressure_at_viking1, idx_longitude1 = slice_data(data=data_pressure_at_viking1,
                                                          dimension_data=data_longitude[:], value=-48)
    latitude1 = data_latitude[idx_latitude1]
    longitude1 = data_longitude[idx_longitude1]

    # Viking 2:  (47.67°N, 134.28°E) near Utopia Planitia
    data_pressure_at_viking2, idx_latitude2 = slice_data(data=data, dimension_data=data_latitude[:], value=48)
    data_pressure_at_viking2, idx_longitude2 = slice_data(data=data_pressure_at_viking2,
                                                          dimension_data=data_longitude[:], value=134)
    latitude2 = data_latitude[idx_latitude2]
    longitude2 = data_longitude[idx_longitude2]

    # Diurnal mean
    data_pressure_at_viking1 = mean(data_pressure_at_viking1.reshape(669, 12), axis=1)
    data_pressure_at_viking2 = mean(data_pressure_at_viking2.reshape(669, 12), axis=1)

    return data_pressure_at_viking1, latitude1, longitude1, data_pressure_at_viking2, latitude2, longitude2


def riceco2_local_time_evolution(filename, data, latitude):
    data = extract_where_co2_ice(filename=filename, data=data)

    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data, idx_latitudes = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    latitude = data_latitude[idx_latitudes]

    data = mean(data, axis=2)  # zonal mean

    # check if there are 12 local times
    if data.shape[0] % 12 != 0:
        print('Stop, there is no 12 localtime')
        exit()

    data_mean = zeros((data.shape[1], 12))  # altitude, local time
    data_std = zeros((data.shape[1], 12))  # altitude, local time
    for i in range(12):
        data_mean[:, i] = mean(data[i::12, :], axis=0)
        data_std[:, i] = std(data[i::12, :], axis=0)

    return data_mean*1e6, data_std, latitude


def riceco2_mean_local_time_evolution(filename, data):
    from scipy.stats import tmean, tsem
    data = extract_where_co2_ice(filename=filename, data=data)
    data = data * 1e6
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data, idx_latitudes = slice_data(data=data, dimension_data=data_latitude[:], value=0)
    latitudes = data_latitude[idx_latitudes]

    data = mean(data, axis=2)  # zonal mean

    # Reshape every localtime for one year!
    if data.shape[0] % 12 != 0:
        nb_sol = 0
        print('Stop, there is no 12 localtime')
        exit()
    else:
        nb_sol = int(data.shape[0] / 12)  # if there is 12 local time!

    data = reshape(data, (nb_sol, 12, data.shape[1]))
    data = mean(data, axis=0)  # mean over the year
    data = data.T

    data_min_radius = zeros(data.shape[1])
    data_max_radius = zeros(data.shape[1])
    data_mean_radius = zeros(data.shape[1])
    data_mean_alt = zeros(data.shape[1])
    data_std_radius = zeros(data.shape[1])
    data_min_alt = zeros(data.shape[1])
    data_max_alt = zeros(data.shape[1])

    data_altitude, list_var = get_data(filename=filename, target='altitude')

    for lt in range(data.shape[1]):

        data_min_radius[lt] = min(data[:, lt])
        data_max_radius[lt] = max(data[:, lt])
        data_mean_radius[lt] = tmean(data[:, lt][~data[:, lt].mask])
        data_std_radius[lt] = tsem(data[:, lt][~data[:, lt].mask])
        print(data_min_radius[lt], data_max_radius[lt], data_mean_radius[lt], data_std_radius[lt])
        print(data[:, lt])
        data_mean_alt[lt] = (int(argmin(data[:, lt])) + int(argmax(data[:, lt]))) / 2.
        data_min_alt[lt] = amin([int(argmin(data[:, lt])), int(argmax(data[:, lt]))])
        data_max_alt[lt] = amax([int(argmin(data[:, lt])), int(argmax(data[:, lt]))])

        if data_max_alt[lt] == 0:
            data_max_alt[lt] = -99999
        else:
            data_max_alt[lt] = data_altitude[data_max_alt[lt]]

        if data_min_alt[lt] == 0:
            data_min_alt[lt] = -99999
        else:
            data_min_alt[lt] = data_altitude[data_min_alt[lt]]

        if data_mean_alt[lt] == 0:
            data_mean_alt[lt] = -99999
        else:
            data_mean_alt[lt] = data_altitude[data_mean_alt[lt]]
    data_min_alt = correction_value(data=data_min_alt, operator='eq', threshold=-99999)
    data_max_alt = correction_value(data=data_max_alt, operator='eq', threshold=-99999)
    data_mean_alt = correction_value(data=data_mean_alt, operator='eq', threshold=-99999)
#    data_mean_radius = correction_value(data=data_mean_radius, operator='eq', threshold=0)
#    data_std_radius = correction_value(data=data_std_radius, operator='eq', threshold=0)

    return data_min_radius, data_max_radius, data_mean_radius, data_mean_alt, data_std_radius, data_min_alt, \
            data_max_alt, latitudes


def riceco2_max_day_night(filename, data):
    data_altitude, list_var = get_data(filename=filename, target='altitude')

    data_time, list_var = get_data(filename=filename, target='Time')
    data_local_time, idx_2, stats = check_local_time(data_time=data_time, selected_time=2)
    data_local_time, idx_14, stats = check_local_time(data_time=data_time, selected_time=14)

    data_day = data[idx_2::len(data_local_time), :, :, :]
    data_night = data[idx_14::len(data_local_time), :, :, :]

    print('Compute max in progress...')
    max_satu_day, idx_altitude_day, y_day = get_extrema_in_alt_lon(data=data_day, extrema='max')
    max_satu_night, idx_altitude_night, y_night = get_extrema_in_alt_lon(data=data_night, extrema='max')

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


def riceco2_top_cloud_altitude(filename, data_target, local_time):
    data_altitude, list_var = get_data(filename, target='altitude')

    if data_altitude.long_name != 'Altitude above local surface':
        print(f'{data_altitude.long_name}')
        exit()

    data_ccn_nco2, list_var = get_data(filename, target='ccnNco2')
    data_rho, list_var = get_data(filename, target='rho')

    data_ccn_nco2 = correction_value(data_ccn_nco2[:, :, :, :], operator='inf', threshold=threshold)
    data_rho = correction_value(data_rho[:, :, :, :], operator='inf', threshold=threshold)

    data_target = mean(data_target[:, :, :, :], axis=3)
    data_ccn_nco2 = mean(data_ccn_nco2[:, :, :, :], axis=3)
    data_rho = mean(data_rho[:, :, :, :], axis=3)

    data_target, tmp = slice_data(data=data_target, dimension_data=data_altitude[:], value=[0, 4e4])
    data_ccn_nco2, tmp = slice_data(data=data_ccn_nco2, dimension_data=data_altitude[:], value=[0, 4e4])
    data_rho, tmp = slice_data(data=data_rho, dimension_data=data_altitude[:], value=[0, 4e4])

    if len(local_time) == 1:
        data_ccn_nco2, local_time = extract_at_a_local_time(filename=filename, data=data_ccn_nco2,
                                                            local_time=local_time)
        data_rho, local_time = extract_at_a_local_time(filename=filename, data=data_rho, local_time=local_time)
        nb_time = data_target.shape[0]
        nb_alt = data_target.shape[1]
        nb_lat = data_target.shape[2]
    else:
        # diurnal mean: data (8028 => 669,12)
        nb_alt = data_target.shape[1]
        nb_lat = data_target.shape[2]
        data_target = mean(data_target.reshape((669, 12, nb_alt, nb_lat)), axis=1)
        data_ccn_nco2 = mean(data_ccn_nco2.reshape((669, 12, nb_alt, nb_lat)), axis=1)
        data_rho = mean(data_rho.reshape((669, 12, nb_alt, nb_lat)), axis=1)
        nb_time = data_target.shape[0]

    n_reflect = 2e-8 * power(data_target * 1e6, -2)  # from Tobie et al. 2003
    n_part = data_rho * data_ccn_nco2
    del [data_target, data_ccn_nco2, data_rho]

    data_latitude, list_var = get_data(filename=filename, target='latitude')
    a = (abs(data_latitude[:] - 45)).argmin() + 1
    polar_latitude = concatenate((arange(a), arange(nb_lat - a, nb_lat)))

    top_cloud = zeros((nb_time, nb_lat))
    for t in range(nb_time):
        for lat in polar_latitude:
            for alt in range(nb_alt - 1, -1, -1):
                if n_part[t, alt, lat] >= n_reflect[t, alt, lat] and alt > 1:
                    top_cloud[t, lat] = data_altitude[alt]
                    break
    top_cloud = top_cloud / 1e3
    top_cloud = rotate_data(top_cloud, do_flip=False)[0]
    print(f'Max top altitude is {max(top_cloud)} km.')
    return top_cloud


def riceco2_zonal_mean_co2ice_exists(filename, data, local_time):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_time, list_var = get_data(filename=filename, target='Time')

    # extract co2_ice data
    data_co2_ice, list_var = get_data(filename=filename, target='co2_ice')
    data_local_time, idx, stats = check_local_time(data_time=data_time[:], selected_time=local_time)
    if idx:
        data_co2_ice = data_co2_ice[idx::len(data_local_time), :, :, :]

    data_slice_lat, latitude_selected = slice_data(data, dimension_data=data_latitude[:], value=[-15, 15])
    data_co2_ice_slice_lat, latitude_selected = slice_data(data_co2_ice, dimension_data=data_latitude[:],
                                                           value=[-15, 15])

    data = ma.masked_where(data_co2_ice_slice_lat < 1e-13, data_slice_lat)

    zonal_mean = mean(data, axis=3)  # zonal mean
    zonal_mean = mean(zonal_mean, axis=1)  # altitude mean
    zonal_mean = rotate_data(zonal_mean, do_flip=True)

    zonal_mean = correction_value(zonal_mean[0], operator='inf', threshold=threshold)
    zonal_mean = zonal_mean * 1e6  # m to µm

    return zonal_mean, latitude_selected


def riceco2_polar_latitudes(filename, data):
    data = extract_where_co2_ice(filename=filename, data=data)

    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_north, idx_latitude_north = slice_data(data=data, dimension_data=data_latitude[:], value=[60, 90])
    data_south, idx_latitude_south = slice_data(data=data, dimension_data=data_latitude[:], value=[-60, -90])
    del data

    data_north = swapaxes(data_north, axis1=0, axis2=2)
    data_north = data_north.reshape(data_north.shape[0], data_north.shape[1], -1)
    data_zonal_mean_north = exp(mean(log(data_north), axis=2))
    stddev_north = exp(std(log(data_north), axis=2))

    data_south = swapaxes(data_south, axis1=0, axis2=2)
    data_south = data_south.reshape(data_south.shape[0], data_south.shape[1], -1)
    data_zonal_mean_south = exp(mean(log(data_south), axis=2))
    stddev_south = exp(std(log(data_south), axis=2))

    return data_zonal_mean_north.T*1e6, data_zonal_mean_south.T*1e6, stddev_north.T, stddev_south.T


def satuco2_zonal_mean_with_co2_ice(filename, data, local_time):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    # Select the three latitudes
    north = 80
    eq = 0
    south = -80
    print('Latitude selected:')
    print(f'\tNorth =   {north}°N')
    print(f'\tEquator = {eq}°N')
    print(f'\tSouth =   {south}°S')

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
    data_co2ice, list_var = get_data(filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:, :, :, :], operator='inf', threshold=threshold)

    if len(local_time) == 1:
        data_co2ice, tmp = extract_at_a_local_time(filename=filename, data=data_co2ice, local_time=local_time)

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
        data_time, list_var = get_data(filename=filename, target='Time')
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
                print(f'Test 5°Ls binning: {data_time[0]} - {data_time[60]}')
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

        del data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, \
            data_co2ice_south

        data_satuco2_north = correction_value(data_satuco2_north_binned, operator='inf', threshold=1e-13)
        data_satuco2_eq = correction_value(data_satuco2_eq_binned, operator='inf', threshold=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, operator='inf', threshold=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, operator='inf', threshold=1e-13)
        data_co2ice_eq = correction_value(data_co2ice_eq_binned, operator='inf', threshold=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, operator='inf', threshold=1e-13)

        del data_satuco2_north_binned, data_satuco2_eq_binned, data_satuco2_south_binned, data_co2ice_north_binned, \
            data_co2ice_eq_binned, data_co2ice_south_binned
    # No binning
    else:
        pass

    data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, data_co2ice_south = \
        rotate_data(data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq,
                    data_co2ice_south, do_flip=False)

    return [data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq,
            data_co2ice_south, north_latitude_selected, eq_latitude_selected, south_latitude_selected, binned]


def satuco2_time_mean_with_co2_ice(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    # Select the three latitudes
    north = 80
    south = -80
    print('Latitude selected:')
    print(f'\tNorth = {north}°N')
    print(f'\tSouth = {abs(south)}°S')

    # Slice data for the three latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data, dimension_data=data_latitude, value=north)
    data_satuco2_south, south_latitude_selected = slice_data(data, dimension_data=data_latitude, value=south)

    data_time, list_var = get_data(filename=filename, target='Time')
    north_winter = [270, 300]
    south_winter = [0, 30]
    print('Time selected:')
    print(f'\tNorth = {north_winter}°Ls')
    print(f'\tSouth = {south_winter}°Ls')

    # Slice data in time
    data_satuco2_north, north_winter_time = slice_data(data_satuco2_north, dimension_data=data_time, value=north_winter)
    data_satuco2_south, south_winter_time = slice_data(data_satuco2_south, dimension_data=data_time, value=south_winter)

    # Compute time mean
    data_satuco2_north = mean(data_satuco2_north, axis=0)
    data_satuco2_south = mean(data_satuco2_south, axis=0)

    del data

    # Get co2 ice mmr
    data_co2ice, list_var = get_data(filename, target='co2_ice')
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
        data_time, list_var = get_data(filename=filename, target='Time')
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
                print(f'Test 5°Ls binning: {data_time[0]} - {data_time[60]}')
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

        data_satuco2_north = correction_value(data_satuco2_north_binned, operator='inf', threshold=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, operator='inf', threshold=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, operator='inf', threshold=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, operator='inf', threshold=1e-13)

        del data_satuco2_north_binned, data_satuco2_south_binned, data_co2ice_north_binned, data_co2ice_south_binned
    # No binning
    else:
        pass

    return [data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south, north_latitude_selected,
            south_latitude_selected, binned]


def satuco2_hu2012_fig9(filename, data, local_time):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    if data_altitude.long_name != 'Altitude above local surface':
        print('The netCDF file did not zrecasted above the local surface')
        exit()

    data_north, latitude_selected = slice_data(data=data, dimension_data=data_latitude, value=[60, 90])
    data_south, latitude_selected = slice_data(data=data, dimension_data=data_latitude, value=[-60, -90])
    del data

    # Bin time in 5° Ls
    data_time, list_var = get_data(filename=filename, target='Time')
    if data_time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if len(local_time) == 1:
            data_local_time, idx, sats = check_local_time(data_time=data_time, selected_time=local_time)
            if idx is not None:
                data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = data_ls

    if data_time.shape[0] % 60 == 0:
        print(f'5°Ls binning: {data_time[0]} - {data_time[60]}')
        nb_bin = int(data_time.shape[0] / 60) + 1
    else:
        print(f'5°Ls binning: from {data_time[0]} to {data_time[-1]}')
        nb_bin = 72
        print(data_time.shape, data_time[-1] / nb_bin, nb_bin * 5)

    data_icelayer = zeros((2, nb_bin))
    data_icelayer_std = zeros((2, nb_bin))

    for BIN in range(nb_bin - 1):
        data_binned_north, time_selected = slice_data(data_north, dimension_data=data_time[:],
                                                      value=[BIN * 5, (BIN + 1) * 5])
        data_binned_south, time_selected = slice_data(data_south, dimension_data=data_time[:],
                                                      value=[BIN * 5, (BIN + 1) * 5])
        print(f'Time: {data_time[time_selected[0]]:.0f} / {data_time[time_selected[-1]]:.0f}°Ls')
        tmp_north = array([])
        tmp_south = array([])
        # Find each super-saturation of co2 thickness
        for ls in range(data_binned_north.shape[0]):
            for longitude in range(data_binned_north.shape[3]):

                # For northern polar region
                for latitude_north in range(data_binned_north.shape[2]):
                    a = data_altitude[data_binned_north[ls, :, latitude_north, longitude].mask == False]
                    if len(a) != 0:
                        tmp_north = append(tmp_north, abs(a[-1] - a[0]))

                # For southern polar region
                for latitude_south in range(data_binned_south.shape[2]):
                    a = data_altitude[data_binned_south[ls, :, latitude_south, longitude].mask == False]
                    if len(a) != 0:
                        tmp_south = append(tmp_south, abs(a[-1] - a[0]))
        tmp_north = correction_value(tmp_north, 'inf', threshold=0)
        tmp_south = correction_value(tmp_south, 'inf', threshold=0)

        if tmp_north.size != 0:
            data_icelayer[0, BIN] = mean(tmp_north)
            data_icelayer_std[0, BIN] = std(tmp_north)
        else:
            data_icelayer[0, BIN] = 0
            data_icelayer_std[0, BIN] = 0

        if tmp_south.size != 0:
            data_icelayer[1, BIN] = mean(tmp_south)
            data_icelayer_std[1, BIN] = std(tmp_south)
        else:
            data_icelayer[1, BIN] = 0
            data_icelayer_std[1, BIN] = 0

        del tmp_north, tmp_south
    return data_icelayer, data_icelayer_std


def satuco2_altitude_longitude(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data, latitude = slice_data(data=data, dimension_data=data_latitude[:], value=0)

    data = mean(data, axis=0)
    data = correction_value(data=data, operator='inf', threshold=0.9)
    return data


def temp_gg2011_fig6(filename, data):
    # GG2011 worked with stats file
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_longitude, list_var = get_data(filename=filename, target='longitude')
    data_time, list_var = get_data(filename=filename, target='Time')
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_rho, list_var = get_data(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
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
    #  For temperature
    data_tmp, longitude = slice_data(data=data[:, :, :, :], dimension_data=data_longitude[:], value=0)
    data_tmp, latitude = slice_data(data=data_tmp, dimension_data=data_latitude[:], value=0)

    #  For density
    data_rho_tmp, longitude = slice_data(data=data_rho[:, :, :, :], dimension_data=data_longitude[:], value=0)
    data_rho_tmp, latitude = slice_data(data=data_rho_tmp, dimension_data=data_latitude[:], value=0)

    # Compute condensation temperature of CO2
    data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=data_tmp, data_rho=data_rho_tmp)

    # Mean for each local time and subtract condensation temperature of CO2
    data_final = zeros((len(data_local_time), data_tmp.shape[1]))
    data_temp_cond_co2_final = zeros((len(data_local_time), data_tmp.shape[1]))

    for i in range(len(data_local_time)):
        data_final[i, :] = mean(data_tmp[i::len(data_local_time), :], axis=0)
        data_temp_cond_co2_final[i, :] = mean(data_temp_cond_co2[i::len(data_local_time), :], axis=0)

    data_p = ma.masked_values(data_final, 0.)
    data_temp_cond_co2_p = ma.masked_values(data_temp_cond_co2_final, 0.)
    for i in range(len(data_local_time)):
        data_p[i, :] = data_p[i, :] - data_temp_cond_co2_p[i, :]

    del data, data_tmp

    print(f'T - T(condensation): min = {min(data_p):.2f}, max = {max(data_p):.2f}')
    return data_p, data_local_time


def temp_gg2011_fig7(filename, data):
    data_time, list_var = get_data(filename=filename, target='Time')
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_rho, list_var = get_data(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
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

    # Compute condensation temperature of CO2 from pressure, ensure that was zrecasted
    data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=data, data_rho=data_rho)

    # Mean for each local time and subtract condensation temperature of CO2
    data = mean(data, axis=2)
    data_temp_cond_co2 = mean(data_temp_cond_co2, axis=2)

    data_final = zeros((data.shape[0], data.shape[1]))
    data_final = ma.masked_values(data_final, 0.)
    for i in range(data_final.shape[1]):
        data_final[:, i] = data[:, i] - data_temp_cond_co2[:, i]

    del data

    print(f'T - T(condensation): min = {min(data_final):.2f}, max = {max(data_final):.2f}')
    return data_final, data_surface_local


def temp_gg2011_fig8(filename, data):
    data_time, list_var = get_data(filename=filename, target='Time')
    data_altitude, list_var = get_data(filename=filename, target='altitude')

    # Slice data: Ls=0-30°
    data, ls = slice_data(data=data, dimension_data=data_time[:], value=[0, 30])
    data_zonal_mean = mean(data, axis=3)

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
    if data_altitude.long_name != 'Altitude above areoid':
        print(data_altitude.long_name)
        print('The netcdf did not zrecasted above the areoid.')
        exit()

    # Check local time available and slice data at 0, 12, 16 H
    data_local_time, idx_0, stats_file = check_local_time(data_time=data_time[:], selected_time=0)
    data_zonal_mean_0h = data_zonal_mean[idx_0::len(data_local_time), :, :]
    data_local_time, idx_12, stats_file = check_local_time(data_time=data_time[:], selected_time=12)
    data_zonal_mean_12h = data_zonal_mean[idx_12::len(data_local_time), :, :]
    data_local_time, idx_16, stats_file = check_local_time(data_time=data_time[:], selected_time=16)
    data_zonal_mean_16h = data_zonal_mean[idx_16::len(data_local_time), :, :]

    # Mean
    data_zonal_mean_0h = mean(data_zonal_mean_0h, axis=0)
    data_zonal_mean_12h = mean(data_zonal_mean_12h, axis=0)
    data_zonal_mean_16h = mean(data_zonal_mean_16h, axis=0)

    # 12h - 00h
    data_thermal_tide = data_zonal_mean_12h - data_zonal_mean_0h

    return data_zonal_mean_16h, data_thermal_tide


def temp_gg2011_fig9(filename, data):
    data_time, list_var = get_data(filename=filename, target='Time')
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_rho, list_var = get_data(filename=filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
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

    # Compute condensation temperature of CO2 from pressure, ensure that was zrecasted
    data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=data, data_rho=data_rho)

    # Mean for each local time and subtract condensation temperature of CO2
    data_final = zeros(data.shape)
    for i in range(data_final.shape[1]):
        data_final[:, i] = data[:, i] - data_temp_cond_co2[:, i]
    data_final = ma.masked_values(data_final, 0.)

    del data

    print(f'T - T(condensation): min = {min(data_final):.2f}, max = {max(data_final):.2f}')
    return data_final, data_surface_local


def temp_stationary_wave(filename, data, local_time):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data, latitudes = slice_data(data=data, dimension_data=data_latitude[:], value=0)

    test = input('Do you want performed T-Tcondco2(y/N)?')
    if test.lower() == 'y':
        diff_temp = True
        data_rho, list_var = get_data(filename=filename, target='rho')
        if local_time is not None:
            data_rho, local_time = extract_at_a_local_time(filename=filename, data=data_rho, local_time=local_time)
        data_rho, latitudes = slice_data(data=data_rho, dimension_data=data_latitude[:], value=0)

        # Compute condensation temperature of CO2 from pressure, ensure that was zrecasted
        data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=data, data_rho=data_rho)
        data_temp_cond_co2 = mean(data_temp_cond_co2, axis=0)
    else:
        data_temp_cond_co2 = 0
        diff_temp = False

    # average over the year at the same localtime
    data = mean(data, axis=0)

    if diff_temp:
        # subtract condensation temperature of CO2
        data_final = zeros(data.shape)
        for lon in range(data_final.shape[1]):
            data_final[:, lon] = data[:, lon] - data_temp_cond_co2[:, lon]
        data_final = ma.masked_values(data_final, 0.)
    else:
        data_final = data

    return data_final, diff_temp


def temp_thermal_structure_polar_region(filename, data):
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data_north, latitude_north = slice_data(data=data, dimension_data=data_latitude[:], value=60)
    data_south, latitude_south = slice_data(data=data, dimension_data=data_latitude[:], value=-60)

    data_north = mean(data_north, axis=2)
    data_south = mean(data_south, axis=2)

    return data_north, data_south


def temp_cold_pocket(filename, data, local_time):
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    if data_altitude.units != 'Pa':
        print('Stop ! File did not zrecasted in Pressure')
        exit()

    data_rho, list_var = get_data(filename=filename, target='rho')
    data_rho, local_time = extract_at_a_local_time(filename=filename, data=data_rho, local_time=local_time)
    data_rho = correction_value(data=data_rho, operator='inf', threshold=1e-13)

    data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=data, data_rho=data_rho)

    zonal_mean = ma.masked_values(data, 0.)

    delta_temp = zeros(zonal_mean.shape)
    for ls in range(zonal_mean.shape[0]):
        for lat in range(zonal_mean.shape[2]):
            for lon in range(zonal_mean.shape[3]):
                delta_temp[ls, :, lat, lon] = zonal_mean[ls, :, lat, lon] - data_temp_cond_co2[ls, :, lat, lon]
    delta_temp = ma.masked_values(delta_temp, 0.)

    # TODO: compter le numbre de cold pockets au-dessus de la tropopause, pour toutes les longitudes
    print('Cold pocket?', len(delta_temp[where(delta_temp < 0)]))
    print(f'min = {min(delta_temp)}, max = {max(delta_temp)}')

    return


def vars_altitude_ls(filename, data, latitude, local_time):
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data, idx_lat = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    data = mean(data, axis=2).T
    return data, data_latitude[idx_lat]


def vars_extract_at_grid_point(filename, data, latitude, longitude):
    data_latitude, list_var = get_data(filename=filename, target='latitude')
    data_longitude, list_var = get_data(filename=filename, target='longitude')

    data, latitudes = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    data, longitudes = slice_data(data=data, dimension_data=data_longitude[:], value=longitude)

    return data.T


def vars_max_value_with_others(filename, data_target):
    shape_data_target = data_target.shape

    print(f'Get max value of {data_target.name} in progress...')
    max_mmr, x, y = get_extrema_in_alt_lon(data_target, extrema='max')
    del data_target
    print('Extract other variable at co2_ice max value:')

    print(' (1) Temperature')
    data_temperature, list_var = get_data(filename, target='temp')[:, :, :, :]
    max_temp = extract_at_max_co2_ice(data_temperature, x, y, shape_data_target)
    del data_temperature

    print(' (2) Saturation')
    data_satuco2, list_var = get_data(filename, target='satuco2')[:, :, :, :]
    max_satu = extract_at_max_co2_ice(data_satuco2, x, y, shape_data_target)
    del data_satuco2

    print(' (3) CCN radius')
    data_riceco2, list_var = get_data(filename, target='riceco2')[:, :, :, :]
    max_radius = extract_at_max_co2_ice(data_riceco2, x, y, shape_data_target)
    del data_riceco2

    print(' (4) CCN number')
    data_ccn_nco2, list_var = get_data(filename, target='ccnNco2')[:, :, :, :]
    max_ccn_n = extract_at_max_co2_ice(data_ccn_nco2, x, y, shape_data_target)
    del data_ccn_nco2

    print(' (5) Altitude')
    data_altitude, list_var = get_data(filename, target='altitude')
    max_alt = extract_at_max_co2_ice(data_altitude, x, y, shape_data_target)

    print('Reshape data in progress...')
    max_mmr, max_temp, max_satu, max_radius, max_ccn_n, max_alt = rotate_data(max_mmr, max_temp, max_satu,
                                                                              max_radius, max_ccn_n, max_alt,
                                                                              do_flip=True)

    return max_mmr, max_temp, max_satu, max_radius, max_ccn_n, max_alt


def vars_time_mean(filename, data, duration, localtime=None):
    from math import ceil

    data_time, list_var = get_data(filename=filename, target='Time')
    if len(localtime) == 1:
        data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=localtime)
        if data_time[-1] <= 360.:  # Ensure we are in ls time coordinate
            data_time = data_time[idx::len(data_local_time)]
        else:
            data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
            data_time = data_ls[idx::len(data_local_time)]
    else:
        if data_time.units != 'deg':
            data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
            data_time = data_ls[:]

    if duration:
        nbin = ceil(data_time[-1] / duration)
        time_bin = arange(0, data_time[-1] + duration, duration)
        if data.ndim == 3:
            data_mean = zeros((nbin, data.shape[1], data.shape[2]))
            for i in range(nbin):
                data_sliced, time = slice_data(data=data, dimension_data=data_time[:],
                                               value=[duration * i, duration * (i + 1)])
                data_mean[i, :, :] = mean(data_sliced, axis=0)
        elif data.ndim == 4:
            data_mean = zeros((nbin, data.shape[1], data.shape[2], data.shape[3]))
            for i in range(nbin):
                data_sliced, time = slice_data(data=data, dimension_data=data_time[:],
                                               value=[duration * i, duration * (i + 1)])
                data_mean[i, :, :, :] = mean(data_sliced, axis=0)

    else:
        data_mean = mean(data, axis=0)
        time_bin = None

    return data_mean, time_bin


def vars_zonal_mean(filename, data, layer=None, flip=None, local_time=None):
    if layer is not None:
        if filename != '':
            data_altitude, list_var = get_data(filename=filename, target='altitude')
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

    if data.ndim == 3:
        zonal_mean = mean(data[:, :, :], axis=2)
    elif data.ndim == 2:
        zonal_mean = mean(data[:, :], axis=1)
    else:
        print('wrong ndim')
        exit()

    # Diurnal mean
    if len(local_time) > 1:
        zonal_mean = zonal_mean.reshape(669, 12, zonal_mean.shape[1])
        zonal_mean = mean(zonal_mean, axis=1)

    if flip is None:
        zonal_mean = rotate_data(zonal_mean, do_flip=True)[0]
    else:
        zonal_mean = rotate_data(zonal_mean, do_flip=False)[0]
    del data

    return zonal_mean, layer_selected


def vars_zonal_mean_column_density(filename, data, local_time):
    # diurnal mean
    if len(local_time) > 1:
        data = data.reshape(669, 12, data.shape[1], data.shape[2], data.shape[3])
        print(data.shape)
        data = mean(data, axis=1)

    data, altitude_limit, altitude_min, altitude_max, altitude_unit = compute_column_density(filename=filename,
                                                                                             data=data)

    # compute zonal mean column density
    print(max(data))
    if data.ndim == 3:
        data = mean(data, axis=2)  # Ls function of lat
    else:
        print('Stop wrong ndim !')
        exit()
    print(max(data))
    data = rotate_data(data, do_flip=True)[0]

    return data, altitude_limit, altitude_min, altitude_max, altitude_unit


def vars_zonal_mean_where_co2ice_exists(filename, data, polar_region):
    data_where_co2_ice = extract_where_co2_ice(filename, data)

    if polar_region:
        # Slice data in north and south polar regions
        data_latitude, list_var = get_data(filename=filename, target='latitude')
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


def vars_zonal_mean_in_time_co2ice_exists(filename, data, data_name, local_time):
    lat1 = int(input('\t Latitude range 1 (°N): '))
    lat2 = int(input('\t Latitude range 2 (°N): '))

    data_time, list_var = get_data(filename=filename, target='Time')
    if data_time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if local_time is not None:
            data_local_time, idx, stats = check_local_time(data_time=data_time, selected_time=local_time)
            data_time = data_ls[idx::len(data_local_time)]
        else:
            idx = None
            data_local_time = None
            data_time = data_ls
    else:
        idx = None
        data_local_time = None

    # extract co2_ice data
    data_co2_ice, list_var = get_data(filename, target='co2_ice')

    # select the latitude range
    data_latitude, list_var = get_data(filename, target='latitude')
    data_sliced_lat, idx_latitude = slice_data(data[:, :, :, :], data_latitude, value=[lat1, lat2])
    data_co2_ice_sliced_lat, idx_latitude = slice_data(data_co2_ice[:, :, :, :], data_latitude,
                                                       value=[lat1, lat2])
    latitude_selected = data_latitude[idx_latitude[0]:idx_latitude[1]]
    del data, data_co2_ice

    # extract at local time
    if local_time is not None:
        data_co2_ice_sliced_lat = data_co2_ice_sliced_lat[idx::len(data_local_time), :, :, :]

    # select the time range
    print('')
    print(f'Time range: {data_time[0]:.2f} - {data_time[-1]:.2f} (°)')
    breakdown = input('Do you want compute mean radius over all the time (Y/n)?')

    if breakdown.lower() in ['y', 'yes']:
        # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
        data_final = ma.masked_where(data_co2_ice_sliced_lat < 1e-13, data_sliced_lat)
        del data_co2_ice_sliced_lat, data_sliced_lat

        data_final = mean(mean(data_final, axis=3), axis=0)  # zonal mean and temporal mean, and m to µm
        list_data = list([data_final])
        filenames = list([f'{data_name}_mean_{latitude_selected[0]:.0f}N_'
                          f'{latitude_selected[-1]:.0f}N_0-360Ls'])
        list_time_selected = list(f'{data_time[0]} - {data_time[1]}')
    else:
        directory_output = f'{data_name}_mean_radius_{latitude_selected[0]:.0f}N_' \
                           f'{latitude_selected[-1]:.0f}N'
        if not path.exists(directory_output):
            mkdir(directory_output)

        time_step = float(input(f'Select the time step range (°): '))
        nb_step = int(data_time[-1] / time_step) + 1

        list_data = list([])
        filenames = list([])
        list_time_selected = list([])
        for i in range(nb_step):
            data_sliced_lat_ls, idx_time = slice_data(data_sliced_lat, dimension_data=data_time[:],
                                                      value=[i * time_step, (i + 1) * time_step])
            data_co2_ice_sliced_lat_ls, idx_time = slice_data(data_co2_ice_sliced_lat, dimension_data=data_time[:],
                                                              value=[i * time_step, (i + 1) * time_step])

            time_selected = data_time[idx_time[0]:idx_time[1]]

            print(f'\t\tselected: {time_selected[0]:.0f} {time_selected[-1]:.0f}')
            list_time_selected.append(f'{time_selected[0]:.0f} - {time_selected[-1]:.0f} °Ls')
            # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
            data_final = ma.masked_where(data_co2_ice_sliced_lat_ls < 1e-13, data_sliced_lat_ls)

            del data_co2_ice_sliced_lat_ls, data_sliced_lat_ls

            data_final = mean(mean(data_final, axis=3), axis=0)  # zonal mean and temporal mean
            list_data.append(data_final)
            filenames.append(f'{directory_output}/{data_name}_mean_{latitude_selected[0]:.0f}N_'
                             f'{latitude_selected[-1]:.0f}N_Ls_{time_selected[0]:.0f}-'
                             f'{time_selected[-1]:.0f}_{local_time:.0f}h')

        del data_sliced_lat, data_co2_ice_sliced_lat

    return list_data, filenames, latitude_selected, list_time_selected


def vars_localtime_longitude(filename, data, latitude, altitude):
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data, idx_latitude = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    data, idx_altitude = slice_data(data=data, dimension_data=data_altitude[:], value=altitude)
    data_mean = zeros((12, data.shape[1]))
    for i in range(12):
        data_mean[i, :] = mean(data[i::12, :], axis=0)
    return data_mean


def vars_ls_longitude(filename, data, latitude, altitude):
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data, idx_latitude = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    data, idx_altitude = slice_data(data=data, dimension_data=data_altitude[:], value=altitude)

    return data.T


def vars_localtime_ls(filename, data, latitude, altitude):
    data_altitude, list_var = get_data(filename=filename, target='altitude')
    data_latitude, list_var = get_data(filename=filename, target='latitude')

    data, idx_latitude = slice_data(data=data, dimension_data=data_latitude[:], value=latitude)
    data, idx_altitude = slice_data(data=data, dimension_data=data_altitude[:], value=altitude)

    data = mean(data, axis=1)

    data = data.reshape(669, 12)  # => hl, lon

    return data.T


def vars_select_profile(data_target):
    print('To be done')
    print('Select latitude, longitude, altitude, time to extract profile')
    print('Perform a list of extracted profile')
    print(data_target)
    exit()
