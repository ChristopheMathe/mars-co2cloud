from numpy import mean, abs, min, max, zeros, where, arange, unravel_index, argmin, argmax, array, ndarray, \
    count_nonzero, std, append, asarray, power, ma, reshape, swapaxes, log, exp, concatenate, amin, amax, diff
from .lib_function import *
from .ncdump import get_data, getfilename
from os import mkdir, path
from sys import exit
from .constant_parameter import cst_stefan, threshold


def co2ice_at_viking_lander_site(info_netcdf):
    data_area = gcm_area()

    # Viking 1: (22.27°N, 312.05°E so -48°E) near Chryse Planitia
    # https://nssdc.gsfc.nasa.gov/planetary/viking.html
    data_at_viking1, idx_latitude1 = slice_data(data=info_netcdf.data_target,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=22)
    data_at_viking1, idx_longitude1 = slice_data(data=data_at_viking1,
                                                 idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                 dimension_slice=info_netcdf.data_dim.longitude,
                                                 value=-48)
    data_area_at_viking1, idx_latitude1 = slice_data(data=data_area,
                                                     idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                     dimension_slice=info_netcdf.data_dim.latitude,
                                                     value=22)
    data_area_at_viking1, idx_longitude1 = slice_data(data=data_area_at_viking1,
                                                      idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                      dimension_slice=info_netcdf.data_dim.longitude,
                                                      value=-48)

    # Viking 2:  (47.67°N, 134.28°E) near Utopia Planitia
    data_at_viking2, idx_latitude2 = slice_data(data=info_netcdf.data_target,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=48)
    data_at_viking2, idx_longitude2 = slice_data(data=data_at_viking2,
                                                 idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                 dimension_slice=info_netcdf.data_dim.longitude,
                                                 value=134)
    data_area_at_viking2, idx_latitude2 = slice_data(data=data_area,
                                                     idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                     dimension_slice=info_netcdf.data_dim.latitude,
                                                     value=48)
    data_area_at_viking2, idx_longitude2 = slice_data(data=data_area_at_viking2,
                                                      idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                      dimension_slice=info_netcdf.data_dim.longitude,
                                                      value=134)

    data_at_viking1 = data_at_viking1 * data_area_at_viking1
    data_at_viking2 = data_at_viking2 * data_area_at_viking2

    return data_at_viking1, data_at_viking2


def co2ice_polar_cloud_distribution(info_netcdf, normalization):
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above areoid':
        print('Data did not zrecasted above the areoid')
        print(f'\tCurrent: {info_netcdf.data_dim.altitude.long_name}')
        exit()

    # sliced data on latitude region
    data_north, latitude_north = slice_data(data=info_netcdf.data_target,
                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                            dimension_slice=info_netcdf.data_dim.latitude,
                                            value=[60, 90])
    data_south, latitude_south = slice_data(data=info_netcdf.data_target,
                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                            dimension_slice=info_netcdf.data_dim.latitude,
                                            value=[-60, -90])

    latitude_north = info_netcdf.data_dim.latitude[latitude_north[0]: latitude_north[1]]
    latitude_south = info_netcdf.data_dim.latitude[latitude_south[0]: latitude_south[1]]

    # sliced data between 104 - 360°Ls time to compare with Fig8 of Neumann et al. 2003
    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if len(info_netcdf.local_time) == 1:
            data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                           selected_time=info_netcdf.local_time)
            info_netcdf.data_dim.time = data_ls[idx::len(data_local_time)]
        else:
            info_netcdf.data_dim.time = data_ls[:]
    data_north, time_selected = slice_data(data=data_north,
                                           idx_dim_slice=info_netcdf.idx_dim.time,
                                           dimension_slice=info_netcdf.data_dim.time,
                                           value=[104, 360])
    data_south, time_selected = slice_data(data=data_south,
                                           idx_dim_slice=info_netcdf.idx_dim.time,
                                           dimension_slice=info_netcdf.data_dim.time,
                                           value=[104, 360])

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


def co2ice_cloud_evolution(info_netcdf):
    from numpy import ma
    lat_1 = -15
    lat_2 = 15

    info_netcdf.data_target, latitude_selected = slice_data(data=info_netcdf.data_target,
                                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                            dimension_slice=info_netcdf.data_dim.latitude,
                                                            value=[lat_1, lat_2])

    # Saturation
    data_satuco2, list_var = get_data(filename=info_netcdf.filename, target='satuco2')
    if len(info_netcdf.local_time) == 1:
        data_satuco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_satuco2[:, :, :, :])
    data_satuco2, latitude_selected = slice_data(data=data_satuco2,
                                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                 dimension_slice=info_netcdf.data_dim.latitude,
                                                 value=[lat_1, lat_2])
    data_satuco2 = correction_value(data=data_satuco2, operator='inf', value=1)

    # Temperature
    data_temp, list_var = get_data(filename=info_netcdf.filename, target='temp')

    if len(info_netcdf.local_time) == 1:
        data_temp, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_temp[:, :, :, :])
    data_temp, latitude_selected = slice_data(data=data_temp,
                                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                                              dimension_slice=info_netcdf.data_dim.latitude,
                                              value=[lat_1, lat_2])
    data_temp = correction_value(data=data_temp, operator='inf', value=threshold)

    # Radius
    data_riceco2, list_var = get_data(filename=info_netcdf.filename, target='riceco2')
    if len(info_netcdf.local_time) == 1:
        data_riceco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_riceco2[:, :, :, :])
    data_riceco2, latitude_selected = slice_data(data=data_riceco2,
                                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                 dimension_slice=info_netcdf.data_dim.latitude,
                                                 value=[lat_1, lat_2])
    data_riceco2 = correction_value(data=data_riceco2, operator='inf', value=threshold)

    # CCNCO2
    data_ccnco2, list_var = get_data(filename=info_netcdf.filename, target='ccnNco2')
    if len(info_netcdf.local_time) == 1:
        data_ccnco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_ccnco2[:, :, :, :])
    data_ccnco2, latitude_selected = slice_data(data=data_ccnco2,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=[lat_1, lat_2])
    data_ccnco2 = correction_value(data=data_ccnco2, operator='inf', value=threshold)

    # h2o ice
    data_h2o_ice, list_var = get_data(filename=info_netcdf.filename, target='h2o_ice')
    if len(info_netcdf.local_time) == 1:
        data_h2o_ice, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_h2o_ice[:, :, :, :])
    data_h2o_ice, latitude_selected = slice_data(data=data_h2o_ice,
                                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                 dimension_slice=info_netcdf.data_dim.latitude,
                                                 value=[lat_1, lat_2])
    data_h2o_ice = correction_value(data=data_h2o_ice, operator='inf', value=threshold)

    latitudes = info_netcdf.data_dim.latitude[latitude_selected[0]: latitude_selected[1] + 1]

    # zonal mean
    info_netcdf.data_target = ma.mean(info_netcdf.data_target, axis=3)
    data_satuco2 = ma.mean(data_satuco2, axis=3)
    data_temp = ma.mean(data_temp, axis=3)
    data_riceco2 = ma.mean(data_riceco2, axis=3)
    data_ccnco2 = ma.mean(data_ccnco2, axis=3)
    data_h2o_ice = ma.mean(data_h2o_ice, axis=3)

    return data_satuco2, data_temp, data_riceco2, data_ccnco2, data_h2o_ice, latitudes


def co2ice_cloud_localtime_along_ls(info_netcdf):
    latitude_min = float(input("Enter minimum latitude: "))
    latitude_max = float(input("Enter maximum latitude: "))
    info_netcdf.data_target, latitude = slice_data(data=info_netcdf.data_target,
                                                   idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                   dimension_slice=info_netcdf.data_dim.latitude,
                                                   value=[latitude_min, latitude_max])

    altitude_limit, altitude_min, altitude_max = compute_column_density(info_netcdf=info_netcdf)
    info_netcdf.data_target = mean(info_netcdf.data_target, axis=2)
    info_netcdf.data_target = mean(info_netcdf.data_target, axis=1)

    # Reshape every localtime for one year!
    if len(info_netcdf.local_time) > 12:
        nb_sol = 0
        print('Stop, there is no 12 localtime')
        exit()
    else:
        nb_sol = int(info_netcdf.data_target.shape[0] / 12)  # if there is 12 local time!
    info_netcdf.data_target = reshape(info_netcdf.data_target, (nb_sol, 12)).T

    return altitude_min, latitude_min, latitude_max


def co2ice_cumulative_masses_polar_cap(info_netcdf):
    from numpy import sum

    ptimestep = 924.739583 * 8
    data_north, latitude_selected = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.latitude,
                                               dimension_slice=info_netcdf.data_dim.latitude,
                                               value=[60, 90])

    data_south, latitude_selected = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.latitude,
                                               dimension_slice=info_netcdf.data_dim.latitude,
                                               value=[-60, -90])

    # get precip_co2_ice
    data_precip_co2_ice, list_var = get_data(filename=info_netcdf.filename, target='precip_co2_ice_rate')
    data_precip_co2_ice = data_precip_co2_ice[:, :, :] * ptimestep
    data_precip_co2_ice_north, tmp = slice_data(data=data_precip_co2_ice,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=[60, 90])
    data_precip_co2_ice_south, tmp = slice_data(data=data_precip_co2_ice,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
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

    return [accumulation_co2ice_north, accumulation_co2ice_south, accumulation_precip_co2_ice_north,
            accumulation_precip_co2_ice_south, accumulation_direct_condco2_north, accumulation_direct_condco2_south]


def co2ice_time_mean(info_netcdf, duration, column=None):
    info_netcdf.data_target, time = vars_time_mean(info_netcdf=info_netcdf, duration=duration)

    info_netcdf.data_target = correction_value(data=info_netcdf.data_target, operator='inf', value=threshold)

    if column:
        altitude_limit, altitude_min, altitude_max = compute_column_density(info_netcdf=info_netcdf)
        info_netcdf.idx_dim.latitude -= 1
        info_netcdf.idx_dim.longitude -= 1
    return time


def co2ice_density_column_evolution(info_netcdf):
    from math import floor

    # Show the evolution of density column at winter polar region
    if info_netcdf.data_dim.time.units == 'degrees':
        print('The netcdf file is in ls !')
        print(f'Time[0] = {info_netcdf.data_dim.time[0]}, Time[-1] = {info_netcdf.data_dim.time[-1]}')
        exit()

    # Slice in time:
    if len(info_netcdf.local_time) == 1:
        data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time,
                                                            selected_time=info_netcdf.local_time)
        times = info_netcdf.data_dim.time[idx::len(data_local_time)]
        print(f'Select the time range in sols ({floor(times[0])} : {int(times[-1])})')
    else:
        print(f'Select the time range in sols ({floor(info_netcdf.data_dim.time[0])} : '
              f'{int(info_netcdf.data_dim.time[-1])})')
        times = info_netcdf.data_dim.time

    time_begin = float(input('Start time: '))
    time_end = float(input('End time: '))
    info_netcdf.data_target, time_range = slice_data(data=info_netcdf.data_target,
                                                     idx_dim_slice=info_netcdf.idx_dim.time,
                                                     dimension_slice=times,
                                                     value=[time_begin, time_end])
    time_range = times[time_range[0]:time_range[-1]+1]

    # Slice data in polar region:
    pole = input('Select the polar region (N/S):')
    if pole.lower() == 'n':
        info_netcdf.data_target, latitude = slice_data(data=info_netcdf.data_target,
                                                       idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                       dimension_slice=info_netcdf.data_dim.latitude,
                                                       value=[60, 90])
        latitudes = info_netcdf.data_dim.latitude[latitude[0]:latitude[-1]+1]
    elif pole.lower() == 's':
        info_netcdf.data_target, latitude = slice_data(data=info_netcdf.data_target,
                                                       idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                       dimension_slice=info_netcdf.data_dim.latitude,
                                                       value=[-60, -90])
        latitudes = info_netcdf.data_dim.latitude[latitude[0]:latitude[-1]+1]
    else:
        latitudes = None
        print('Wrong selection')
        exit()

    altitude_limit, idx_altitude_min, idx_altitude_max = compute_column_density(info_netcdf=info_netcdf)
    return time_range, latitudes


def co2ice_coverage(info_netcdf):
    if len(info_netcdf.local_time) == 1:
        data_local_time, idx, tmp = check_local_time(data_time=info_netcdf.data_dim.time,
                                                     selected_time=info_netcdf.local_time)
        ntime = info_netcdf.data_dim.time[idx::len(data_local_time)].shape[0]

    else:
        ntime = info_netcdf.data_dim.time.shape[0]

    nlat = info_netcdf.data_dim.latitude.shape[0]
    nlon = info_netcdf.data_dim.longitude.shape[0]

    idx_10pa = (abs(info_netcdf.data_dim.altitude[:] - 10)).argmin()

    data_co2ice_coverage = zeros((nlat, nlon))
    data_co2ice_coverage_meso = zeros((nlat, nlon))

    for lat in range(nlat):
        for lon in range(nlon):
            for ls in range(ntime):  # time
                if not all(info_netcdf.data_target[ls, :, lat, lon].mask):  # There at least one cell with co2_ice
                    data_co2ice_coverage[lat, lon] += 1
                    if not all(info_netcdf.data_target[ls, idx_10pa:, lat, lon].mask):  # There at least one cell
                        data_co2ice_coverage_meso[lat, lon] = 1  # with co2_ice in mesosphere

    data_co2ice_coverage = correction_value(data=data_co2ice_coverage, operator='eq', value=0)
    data_co2ice_coverage_meso = correction_value(data=data_co2ice_coverage_meso, operator='eq', value=0)

    #  Normalization
    data_co2ice_coverage = (data_co2ice_coverage / ntime) * 100
    return data_co2ice_coverage, data_co2ice_coverage_meso


def emis_polar_winter_gg2020_fig13(info_netcdf):
    # Slice in time
    if info_netcdf.data_dim.time.units != 'deg':
        data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                       selected_time=info_netcdf.local_time)
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if idx:
            data_ls = data_ls[idx::len(data_local_time)]
    else:
        data_ls = info_netcdf.data_dim.time

    #       NP: 180°-360°
    data_np, time = slice_data(data=info_netcdf.data_target,
                               idx_dim_slice=info_netcdf.idx_dim.time,
                               dimension_slice=data_ls,
                               value=[180, 360])

    #       SP: 0°-180°
    data_sp, time = slice_data(data=info_netcdf.data_target,
                               idx_dim_slice=info_netcdf.idx_dim,
                               dimension_slice=data_ls,
                               value=[0, 180])

    # Slice in latitude > 60°
    data_np, latitude = slice_data(data=data_np,
                                   idx_dim_slice=info_netcdf.idx_dim.latitude - 1,
                                   dimension_slice=info_netcdf.data_dim.latitude,
                                   value=[60, 90])
    data_sp, latitude = slice_data(data=data_sp,
                                   idx_dim_slice=info_netcdf.idx_dim.latitude - 1,
                                   dimension_slice=info_netcdf.data_dim.latitude,
                                   value=[-60, -90])

    # Mean in time
    data_mean_np = mean(data_np, axis=0)
    data_mean_sp = mean(data_sp, axis=0)

    return data_mean_np, data_mean_sp


def flux_lw_apparent_temperature_zonal_mean(info_netcdf):
    # Flux = sigma T^4
    temperature_apparent = power(info_netcdf.data_target / cst_stefan, 1 / 4)
    temperature_apparent = mean(temperature_apparent, axis=2).T
    return temperature_apparent


def h2o_ice_alt_ls_with_co2_ice(info_netcdf, directory, files):
    latitude = input('Which latitude (°N)? ')
    if len(latitude.split(',')) > 1:
        latitude = array(latitude.split(','), dtype=float)
    else:
        latitude = [float(latitude)]

    data, idx_latitude_selected = slice_data(data=info_netcdf.data_target,
                                             dimension_slice=info_netcdf.data_dim.latitude,
                                             idx_dim_slice=info_netcdf.idx_dim.latitude, value=latitude)
    # Deal now for CO2 ice
    try:
        data_co2_ice, list_var = get_data(filename=info_netcdf.filename, target='co2_ice')
        print('CO2 ice found.')
    except IOError:
        filename_co2 = getfilename(files=files, selection=None)
        data_co2_ice, list_var = get_data(filename=directory + filename_co2, target='co2_ice')

    if len(info_netcdf.local_time) == 1:
        data_co2_ice, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_co2_ice)

    data_co2_ice, idx_latitude_selected = slice_data(data=data_co2_ice,
                                                     dimension_slice=info_netcdf.data_dim.latitude,
                                                     idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                     value=latitude)

    data_co2_ice = correction_value(data_co2_ice, operator='inf', value=threshold)

    # latitude mean
    if len(latitude) > 1:
        data = mean(data, axis=2)
        data_co2_ice = mean(data_co2_ice, axis=2)

    # zonal mean
    zonal_mean = mean(data, axis=2)  # zonal mean
    zonal_mean_co2_ice = mean(data_co2_ice, axis=2)
    if len(info_netcdf.local_time) != 1:
        nb_sols = int(zonal_mean.shape[0] / 12)
        zonal_mean = mean(zonal_mean.reshape((nb_sols, 12, zonal_mean.shape[1])), axis=1)  # => sols, lon
        zonal_mean_co2_ice = mean(zonal_mean_co2_ice.reshape((nb_sols, 12, zonal_mean_co2_ice.shape[1])),
                                  axis=1)  # => sols, lon

    zonal_mean, zonal_mean_co2_ice = rotate_data(zonal_mean, zonal_mean_co2_ice, do_flip=False)

    info_netcdf.data_target = 0
    del data, data_co2_ice
    if len(latitude) == 1:
        latitude_out = array([info_netcdf.data_dim.latitude[idx_latitude_selected]])
    else:
        latitude_out = info_netcdf.data_dim.latitude[idx_latitude_selected]

    return zonal_mean, zonal_mean_co2_ice, latitude_out


def ps_at_viking(info_netcdf):
    # Viking 1: (22.27°N, 312.05°E so -48°E) near Chryse Planitia
    # https://nssdc.gsfc.nasa.gov/planetary/viking.html
    data_pressure_at_viking1, idx_latitude1 = slice_data(data=info_netcdf.data_target,
                                                         idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                         dimension_slice=info_netcdf.data_dim.latitude,
                                                         value=22)
    data_pressure_at_viking1, idx_longitude1 = slice_data(data=data_pressure_at_viking1,
                                                          idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                          dimension_slice=info_netcdf.data_dim.longitude,
                                                          value=-48)

    latitude1 = info_netcdf.data_dim.latitude[idx_latitude1]
    longitude1 = info_netcdf.data_dim.longitude[idx_longitude1]

    # Viking 2:  (47.67°N, 134.28°E) near Utopia Planitia
    data_pressure_at_viking2, idx_latitude2 = slice_data(data=info_netcdf.data_target,
                                                         idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                         dimension_slice=info_netcdf.data_dim.latitude,
                                                         value=48)
    data_pressure_at_viking2, idx_longitude2 = slice_data(data=data_pressure_at_viking2,
                                                          idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                                          dimension_slice=info_netcdf.data_dim.longitude,
                                                          value=134)
    latitude2 = info_netcdf.data_dim.latitude[idx_latitude2]
    longitude2 = info_netcdf.data_dim.longitude[idx_longitude2]

    # Diurnal mean
    data_pressure_at_viking1 = mean(data_pressure_at_viking1.reshape(669, 12), axis=1)
    data_pressure_at_viking2 = mean(data_pressure_at_viking2.reshape(669, 12), axis=1)

    return data_pressure_at_viking1, latitude1, longitude1, data_pressure_at_viking2, latitude2, longitude2


def riceco2_local_time_evolution(info_netcdf, latitude):
    info_netcdf.data_target = extract_where_co2_ice(info_netcdf=info_netcdf)

    data, idx_latitudes = slice_data(data=info_netcdf.data_target,
                                     idx_dim_slice=info_netcdf.idx_dim.latitude,
                                     dimension_slice=info_netcdf.data_dim.latitude,
                                     value=latitude)

    latitude = info_netcdf.data_dim.latitude[idx_latitudes]

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

    return data_mean * 1e6, data_std, latitude


def riceco2_mean_local_time_evolution(info_netcdf):
    from scipy.stats import tmean, tsem
    data = extract_where_co2_ice(info_netcdf=info_netcdf)
    data = data * 1e6
    latitude_input = float(input('Select a latitude (°N):'))
    data, idx_latitudes = slice_data(data=data,
                                     idx_dim_slice=info_netcdf.idx_dim.latitude,
                                     dimension_slice=info_netcdf.data_dim.latitude,
                                     value=latitude_input)
    latitudes = info_netcdf.data_dim.latitude[idx_latitudes]

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

    for lt in range(data.shape[1]):

        data_min_radius[lt] = min(data[:, lt])
        data_max_radius[lt] = max(data[:, lt])
        data_mean_radius[lt] = tmean(data[:, lt][~data[:, lt].mask])
        data_std_radius[lt] = tsem(data[:, lt][~data[:, lt].mask])
        data_mean_alt[lt] = (int(argmin(data[:, lt])) + int(argmax(data[:, lt]))) / 2.
        data_min_alt[lt] = amin([int(argmin(data[:, lt])), int(argmax(data[:, lt]))])
        data_max_alt[lt] = amax([int(argmin(data[:, lt])), int(argmax(data[:, lt]))])

        if data_max_alt[lt] == 0:
            data_max_alt[lt] = -99999
        else:
            data_max_alt[lt] = info_netcdf.data_dim.altitude[data_max_alt[lt]]

        if data_min_alt[lt] == 0:
            data_min_alt[lt] = -99999
        else:
            data_min_alt[lt] = info_netcdf.data_dim.altitude[data_min_alt[lt]]

        if data_mean_alt[lt] == 0:
            data_mean_alt[lt] = -99999
        else:
            data_mean_alt[lt] = info_netcdf.data_dim.altitude[data_mean_alt[lt]]
    data_min_alt = correction_value(data=data_min_alt, operator='eq', value=-99999)
    data_max_alt = correction_value(data=data_max_alt, operator='eq', value=-99999)
    data_mean_alt = correction_value(data=data_mean_alt, operator='eq', value=-99999)

    return data_min_radius, data_max_radius, data_mean_radius, data_mean_alt, data_std_radius, data_min_alt, \
           data_max_alt, latitudes


def riceco2_max_day_night(info_netcdf):
    data_local_time, idx_2, stats = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=2)
    data_local_time, idx_14, stats = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=14)

    data_day = info_netcdf.data_target[idx_2::len(data_local_time), :, :, :]
    data_night = info_netcdf.data_target[idx_14::len(data_local_time), :, :, :]

    print('Compute max in progress...')
    max_satu_day, idx_altitude_day, y_day = get_extrema_in_alt_lon(data=data_day, extrema='max')
    max_satu_night, idx_altitude_night, y_night = get_extrema_in_alt_lon(data=data_night, extrema='max')

    max_alt_day = zeros(idx_altitude_day.shape)
    max_alt_night = zeros(idx_altitude_night.shape)
    for i in range(idx_altitude_night.shape[0]):
        for j in range(idx_altitude_night.shape[1]):
            max_alt_night[i, j] = info_netcdf.data_dim.altitude[idx_altitude_night[i, j]]
            max_alt_day[i, j] = info_netcdf.data_dim.altitude[idx_altitude_day[i, j]]
            if max_satu_day[i, j] == 0:
                max_alt_day[i, j] = None
                max_satu_day[i, j] = None

            if max_satu_night[i, j] == 0:
                max_alt_night[i, j] = None
                max_satu_night[i, j] = None

    return


def riceco2_top_cloud_altitude(info_netcdf):
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above local surface':
        print(f'{info_netcdf.data_dim.altitude.long_name}')
        exit()

    data_ccn_nco2, list_var = get_data(filename=info_netcdf.filename, target='ccnNco2')
    data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')

    data_ccn_nco2 = correction_value(data_ccn_nco2[:, :, :, :], operator='inf', value=threshold)
    data_rho = correction_value(data_rho[:, :, :, :], operator='inf', value=threshold)

    data_target = mean(info_netcdf.data_target[:, :, :, :], axis=3)
    data_ccn_nco2 = mean(data_ccn_nco2[:, :, :, :], axis=3)
    data_rho = mean(data_rho[:, :, :, :], axis=3)

    data_target, tmp = slice_data(data=data_target,
                                  idx_dim_slice=info_netcdf.idx_dim.altitude,
                                  dimension_slice=info_netcdf.data_dim.altitude,
                                  value=[0, 4e4])
    data_ccn_nco2, tmp = slice_data(data=data_ccn_nco2,
                                    idx_dim_slice=info_netcdf.idx_dim.altitude,
                                    dimension_slice=info_netcdf.data_dim.altitude,
                                    value=[0, 4e4])
    data_rho, tmp = slice_data(data=data_rho,
                               idx_dim_slice=info_netcdf.idx_dim.altitude,
                               dimension_slice=info_netcdf.data_dim.altitude,
                               value=[0, 4e4])

    if len(info_netcdf.local_time) == 1:
        data_ccn_nco2, local_time = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_ccn_nco2)
        data_rho, local_time = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_rho)
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

    a = (abs(info_netcdf.data_dim.latitude[:] - 45)).argmin() + 1
    polar_latitude = concatenate((arange(a), arange(nb_lat - a, nb_lat)))

    top_cloud = zeros((nb_time, nb_lat))
    for t in range(nb_time):
        for lat in polar_latitude:
            for alt in range(nb_alt - 1, -1, -1):
                if n_part[t, alt, lat] >= n_reflect[t, alt, lat] and alt > 1:
                    top_cloud[t, lat] = info_netcdf.data_dim.altitude[alt]
                    break
    top_cloud = top_cloud / 1e3
    top_cloud = rotate_data(top_cloud, do_flip=False)[0]
    print(f'Max top altitude is {max(top_cloud)} km.')
    return top_cloud


def riceco2_zonal_mean_co2ice_exists(info_netcdf):
    # extract co2_ice data
    data_co2_ice, list_var = get_data(filename=info_netcdf.filename, target='co2_ice')
    data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                   selected_time=info_netcdf.local_time)
    if idx:
        data_co2_ice = data_co2_ice[idx::len(data_local_time), :, :, :]

    data_slice_lat, latitude_selected = slice_data(data=info_netcdf.data_target,
                                                   idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                   dimension_slice=info_netcdf.data_dim.latitude,
                                                   value=[-15, 15])
    data_co2_ice_slice_lat, latitude_selected = slice_data(data=data_co2_ice,
                                                           idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                           dimension_slice=info_netcdf.data_dim.latitude,
                                                           value=[-15, 15])

    data = ma.masked_where(data_co2_ice_slice_lat < 1e-13, data_slice_lat)

    zonal_mean = mean(data, axis=3)  # zonal mean
    zonal_mean = mean(zonal_mean, axis=1)  # altitude mean
    zonal_mean = rotate_data(zonal_mean, do_flip=True)

    zonal_mean = correction_value(zonal_mean[0], operator='inf', value=threshold)
    zonal_mean = zonal_mean * 1e6  # m to µm

    return zonal_mean, latitude_selected


def riceco2_polar_latitudes(info_netcdf):
    from scipy.stats import tmean, tsem
    data = extract_where_co2_ice(info_netcdf=info_netcdf)
    data = data * 1e6
    data_north, idx_latitude_north = slice_data(data=data,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=[60, 90])
    data_south, idx_latitude_south = slice_data(data=data,
                                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                dimension_slice=info_netcdf.data_dim.latitude,
                                                value=[-60, -90])
    del data

    data_north = swapaxes(data_north, axis1=0, axis2=2)
    data_north = data_north.reshape(data_north.shape[0], data_north.shape[1], -1)

    data_zonal_mean_north = zeros(shape=(data_north.shape[0:2]))
    stddev_north = zeros(shape=(data_north.shape[0:2]))
    for i in range(data_north.shape[0]):
        for j in range(data_north.shape[1]):
            data_zonal_mean_north[i,j] = tmean(data_north[i,j,:][~data_north[i,j,:].mask])
            stddev_north[i,j] = tsem(data_north[i,j,:][~data_north[i,j,:].mask])

    data_south = swapaxes(data_south, axis1=0, axis2=2)
    data_south = data_south.reshape(data_south.shape[0], data_south.shape[1], -1)
    data_zonal_mean_south = zeros(shape=(data_south.shape[0:2]))
    stddev_south = zeros(shape=(data_south.shape[0:2]))
    for i in range(data_south.shape[0]):
        for j in range(data_south.shape[1]):
            data_zonal_mean_south[i,j] = tmean(data_south[i,j,:][~data_south[i,j,:].mask])
            stddev_south[i,j] = tsem(data_south[i,j,:][~data_south[i,j,:].mask])

    return data_zonal_mean_north.T, data_zonal_mean_south.T, stddev_north.T, stddev_south.T


def satuco2_zonal_mean_with_co2_ice(info_netcdf):
    # Select the three latitudes
    north = 80
    eq = 0
    south = -80
    print('Latitude selected:')
    print(f'\tNorth =   {north}°N')
    print(f'\tEquator = {eq}°N')
    print(f'\tSouth =   {south}°S')

    # Slice data for the three latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data=info_netcdf.data_target,
                                                             idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                             dimension_slice=info_netcdf.data_dim.latitude,
                                                             value=north)
    data_satuco2_eq, eq_latitude_selected = slice_data(data=info_netcdf.data_target,
                                                       idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                       dimension_slice=info_netcdf.data_dim.latitude,
                                                       value=eq)
    data_satuco2_south, south_latitude_selected = slice_data(data=info_netcdf.data_target,
                                                             idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                             dimension_slice=info_netcdf.data_dim.latitude,
                                                             value=south)

    # Compute zonal mean
    data_satuco2_north = mean(data_satuco2_north, axis=2)
    data_satuco2_eq = mean(data_satuco2_eq, axis=2)
    data_satuco2_south = mean(data_satuco2_south, axis=2)

    # Get co2 ice mmr
    data_co2ice, list_var = get_data(filename=info_netcdf.filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:, :, :, :], operator='inf', value=threshold)

    if len(info_netcdf.local_time) == 1:
        data_co2ice, tmp = extract_at_a_local_time(info_netcdf=info_netcdf.filename, data=data_co2ice)

    # Slice co2 ice mmr at these 3 latitudes
    data_co2ice_north, north_latitude_selected = slice_data(data_co2ice,
                                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                            dimension_slice=info_netcdf.data_dim.latitude,
                                                            value=north)
    data_co2ice_eq, eq_latitude_selected = slice_data(data_co2ice,
                                                      idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                      dimension_slice=info_netcdf.data_dim.latitude,
                                                      value=eq)
    data_co2ice_south, south_latitude_selected = slice_data(data_co2ice,
                                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                            dimension_slice=info_netcdf.data_dim.latitude,
                                                            value=south)

    # Compute zonal mean
    data_co2ice_north = mean(data_co2ice_north, axis=2)
    data_co2ice_eq = mean(data_co2ice_eq, axis=2)
    data_co2ice_south = mean(data_co2ice_south, axis=2)
    del data_co2ice

    binned = input('Do you want bin data (Y/n)? ')

    if binned.lower() == 'y':
        # Bin time in 5° Ls
        if max(info_netcdf.data_dim.time) > 360:
            time_grid_ls = convert_sols_to_ls()
            nb_bin = time_grid_ls.shape[0]

            data_satuco2_north_binned = zeros((nb_bin, data_satuco2_north.shape[1]))
            data_satuco2_eq_binned = zeros((nb_bin, data_satuco2_eq.shape[1]))
            data_satuco2_south_binned = zeros((nb_bin, data_satuco2_south.shape[1]))
            data_co2ice_north_binned = zeros((nb_bin, data_co2ice_north.shape[1]))
            data_co2ice_eq_binned = zeros((nb_bin, data_co2ice_eq.shape[1]))
            data_co2ice_south_binned = zeros((nb_bin, data_co2ice_south.shape[1]))

            for i in range(nb_bin - 1):
                idx_ls_1 = (abs(info_netcdf.data_dim.time - time_grid_ls[i])).argmin()
                idx_ls_2 = (abs(info_netcdf.data_dim.time - time_grid_ls[i + 1])).argmin() + 1

                data_satuco2_north_binned[i, :] = mean(data_satuco2_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_eq_binned[i, :] = mean(data_satuco2_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_satuco2_south_binned[i, :] = mean(data_satuco2_south[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_north_binned[i, :] = mean(data_co2ice_north[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_eq_binned[i, :] = mean(data_co2ice_eq[idx_ls_1:idx_ls_2, :], axis=0)
                data_co2ice_south_binned[i, :] = mean(data_co2ice_south[idx_ls_1:idx_ls_2, :], axis=0)
        else:
            if info_netcdf.data_dim.time.shape[0] % 60 == 0:
                print(f'Test 5°Ls binning: {info_netcdf.data_dim.time[0]} - {info_netcdf.data_dim.time[60]}')
            else:
                print('The data will not be binned in 5°Ls, need to work here')

            nb_bin = int(info_netcdf.data_dim.time.shape[0] / 60)
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

        data_satuco2_north = correction_value(data_satuco2_north_binned, operator='inf', value=1e-13)
        data_satuco2_eq = correction_value(data_satuco2_eq_binned, operator='inf', value=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, operator='inf', value=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, operator='inf', value=1e-13)
        data_co2ice_eq = correction_value(data_co2ice_eq_binned, operator='inf', value=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, operator='inf', value=1e-13)

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


def satuco2_time_mean_with_co2_ice(info_netcdf):
    # Select the three latitudes
    north = 80
    south = -80
    print('Latitude selected:')
    print(f'\tNorth = {north}°N')
    print(f'\tSouth = {abs(south)}°S')

    # Slice data for the three latitudes
    data_satuco2_north, north_latitude_selected = slice_data(data=info_netcdf.data_target,
                                                             idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                             dimension_slice=info_netcdf.data_dim.latitude,
                                                             value=north)
    data_satuco2_south, south_latitude_selected = slice_data(data=info_netcdf.data_target,
                                                             idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                             dimension_slice=info_netcdf.data_dim.latitude,
                                                             value=south)

    north_winter = [270, 300]
    south_winter = [0, 30]
    print('Time selected:')
    print(f'\tNorth = {north_winter}°Ls')
    print(f'\tSouth = {south_winter}°Ls')
    if info_netcdf.data_dim.time[-1] > 360:
        data_time, tmp = get_data(filename='../concat_Ls.nc', target='Time')
        # TODO: check if many local time used => mean
    else:
        data_time = info_netcdf.data_dim.time
    # Slice data in time
    data_satuco2_north, north_winter_time = slice_data(data_satuco2_north,
                                                       idx_dim_slice=info_netcdf.idx_dim.time,
                                                       dimension_slice=data_time,
                                                       value=north_winter)
    data_satuco2_south, south_winter_time = slice_data(data_satuco2_south,
                                                       idx_dim_slice=info_netcdf.idx_dim.time,
                                                       dimension_slice=data_time,
                                                       value=south_winter)

    # Compute time mean
    data_satuco2_north = mean(data_satuco2_north, axis=0)
    data_satuco2_south = mean(data_satuco2_south, axis=0)

    # Get co2 ice mmr
    data_co2ice, list_var = get_data(filename=info_netcdf.filename, target='co2_ice')
    data_co2ice = correction_value(data_co2ice[:, :, :, :], operator='inf', value=1e-13)

    # Slice co2 ice mmr at these 3 latitudes
    data_co2ice_north, north_latitude_selected = slice_data(data=data_co2ice,
                                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                            dimension_slice=info_netcdf.data_dim.latitude,
                                                            value=north)
    data_co2ice_south, south_latitude_selected = slice_data(data=data_co2ice,
                                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                            dimension_slice=info_netcdf.data_dim.latitude,
                                                            value=south)

    # Slice data in time
    data_co2ice_north, north_winter_time = slice_data(data=data_co2ice_north,
                                                      idx_dim_slice=info_netcdf.idx_dim.time,
                                                      dimension_slice=data_time,
                                                      value=north_winter)
    data_co2ice_south, south_winter_time = slice_data(data=data_co2ice_south,
                                                      idx_dim_slice=info_netcdf.idx_dim.time,
                                                      dimension_slice=data_time,
                                                      value=south_winter)

    # Compute Time mean
    data_co2ice_north = mean(data_co2ice_north, axis=0)
    data_co2ice_south = mean(data_co2ice_south, axis=0)
    del data_co2ice

    binned = input('Do you want bin data (Y/n)? ')
    if binned.lower() == 'y':
        # Bin time in 5° Ls
        data_time, list_var = get_data(filename=info_netcdf.filename, target='Time')
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

        data_satuco2_north = correction_value(data_satuco2_north_binned, operator='inf', value=1e-13)
        data_satuco2_south = correction_value(data_satuco2_south_binned, operator='inf', value=1e-13)
        data_co2ice_north = correction_value(data_co2ice_north_binned, operator='inf', value=1e-13)
        data_co2ice_south = correction_value(data_co2ice_south_binned, operator='inf', value=1e-13)

        del data_satuco2_north_binned, data_satuco2_south_binned, data_co2ice_north_binned, data_co2ice_south_binned
    # No binning
    else:
        pass

    return [data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south, north_latitude_selected,
            south_latitude_selected, binned]


def satuco2_hu2012_fig9(info_netcdf):
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above local surface':
        print('The netCDF file did not zrecasted above the local surface')
        exit()

    data_north, latitude_selected = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.latitude,
                                               dimension_slice=info_netcdf.data_dim.latitude,
                                               value=[60, 90])
    data_south, latitude_selected = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.latitude,
                                               dimension_slice=info_netcdf.data_dim.latitude,
                                               value=[-60, -90])
    data_north, altitude_selected = slice_data(data=data_north,
                                               idx_dim_slice=info_netcdf.idx_dim.altitude,
                                               dimension_slice=info_netcdf.data_dim.altitude,
                                               value=[0, 70e3])
    data_south, altitude_selected = slice_data(data=data_south,
                                               idx_dim_slice=info_netcdf.idx_dim.altitude,
                                               dimension_slice=info_netcdf.data_dim.altitude,
                                               value=[0, 70e3])
    data_altitude, altitude_selected = slice_data(data=info_netcdf.data_dim.altitude,
                                                  idx_dim_slice=info_netcdf.idx_dim.altitude,
                                                  dimension_slice=info_netcdf.data_dim.altitude,
                                                  value=[0, 70e3])

    # Bin time in 5° Ls
    if info_netcdf.data_dim.time.units != 'deg':
        data_time = info_netcdf.data_dim.time
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if len(info_netcdf.local_time) == 1:
            data_local_time, idx, sats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                          selected_time=info_netcdf.local_time)
            if idx is not None:
                data_time = data_ls[idx::len(data_local_time)]
        else:
            data_time = data_ls
    else:
        data_time = info_netcdf.data_dim.time

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
        data_binned_north, time_selected = slice_data(data=data_north,
                                                      idx_dim_slice=info_netcdf.idx_dim.time,
                                                      dimension_slice=data_time[:],
                                                      value=[BIN * 5, (BIN + 1) * 5])
        data_binned_south, time_selected = slice_data(data=data_south,
                                                      idx_dim_slice=info_netcdf.idx_dim.time,
                                                      dimension_slice=data_time[:],
                                                      value=[BIN * 5, (BIN + 1) * 5])
        print(f'Time range {BIN}: {data_time[time_selected[0]]:.0f} / {data_time[time_selected[-1]]:.0f}°Ls')
        tmp_north = array([])
        tmp_south = array([])

        # Find each super-saturation of co2 thickness
        for ls in range(data_binned_north.shape[0]):
            for longitude in range(data_binned_north.shape[3]):

                # For northern polar region
                for latitude_north in range(data_binned_north.shape[2]):
                    a = data_altitude[data_binned_north[ls, :, latitude_north, longitude].mask is False]
                    if len(a) != 0:
                        tmp_north = append(tmp_north, abs(a[-1] - a[0]))

                # For southern polar region
                for latitude_south in range(data_binned_south.shape[2]):
                    a = data_altitude[data_binned_south[ls, :, latitude_south, longitude].mask is False]
                    if len(a) != 0:
                        tmp_south = append(tmp_south, abs(a[-1] - a[0]))
        tmp_north = correction_value(tmp_north, 'inf', value=0)
        tmp_south = correction_value(tmp_south, 'inf', value=0)

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


def satuco2_altitude_longitude(info_netcdf):
    data, latitude = slice_data(data=info_netcdf.data_target,
                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                dimension_slice=info_netcdf.data_dim.latitude,
                                value=0)

    data = mean(data, axis=0)
    data = correction_value(data=data, operator='inf', value=0.9)
    return data


def satuco2_maxvalue_with_maxalt(info_netcdf):
    # Diurnal mean
    nb_lt = len(info_netcdf.local_time)
    nb_sols = int(info_netcdf.data_dim.time.shape[0] / nb_lt)
    if nb_lt > 1:
        info_netcdf.data_target = info_netcdf.data_target.reshape(nb_sols, nb_lt,
                                                                  info_netcdf.data_dim.altitude.shape[0],
                                                                  info_netcdf.data_dim.latitude.shape[0],
                                                                  info_netcdf.data_dim.longitude.shape[0])
        info_netcdf.data_target = mean(info_netcdf.data_target, axis=1)

    info_netcdf.data_target = mean(info_netcdf.data_target, axis=info_netcdf.idx_dim.longitude)

    data_maxval = zeros(shape=(nb_sols, info_netcdf.data_dim.latitude.shape[0]))
    data_altval = zeros(shape=(nb_sols, info_netcdf.data_dim.latitude.shape[0]))
    data_altitude = info_netcdf.data_dim.altitude
    for time in range(nb_sols):
        for latitude in range(info_netcdf.data_dim.latitude.shape[0]):
            if info_netcdf.data_target[time, :, latitude].mask.all():
                data_maxval[time, latitude] = -1
                data_altval[time, latitude] = -1
            else:
                data_maxval[time, latitude] = max(info_netcdf.data_target[time, :, latitude])
                data_altval[time, latitude] = data_altitude[argmax(info_netcdf.data_target[time, :, latitude])]
    data_maxval = correction_value(data=data_maxval, operator='inf', value=threshold)
    data_altval = correction_value(data=data_altval, operator='inf', value=threshold)
    return data_maxval.T, data_altval.T


def temp_gg2011_fig6(info_netcdf):
    # GG2011 worked with stats file
    data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above areoid':
        print(info_netcdf.data_dim.altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(info_netcdf.data_dim.time)
    if not stats_file or 'stats5' not in info_netcdf.filename:
        print('This file is not a stats file required to compare with GG2011')
        exit()

    # Slice data: lon = 0°, lat = 0° [Fig 6, GG2011]
    #  For temperature
    data_tmp, longitude = slice_data(data=info_netcdf.data_target,
                                     idx_dim_slice=info_netcdf.idx_dim.longitude,
                                     dimension_slice=info_netcdf.data_dim.longitude,
                                     value=0)
    data_tmp, latitude = slice_data(data=data_tmp,
                                    idx_dim_slice=info_netcdf.idx_dim.latitude,
                                    dimension_slice=info_netcdf.data_dim.latitude,
                                    value=0)

    #  For density
    data_rho_tmp, longitude = slice_data(data=data_rho[:, :, :, :],
                                         idx_dim_slice=info_netcdf.idx_dim.longitude,
                                         dimension_slice=info_netcdf.data_dim.longitude,
                                         value=0)
    data_rho_tmp, latitude = slice_data(data=data_rho_tmp,
                                        idx_dim_slice=info_netcdf.idx_dim.latitude,
                                        dimension_slice=info_netcdf.data_dim.latitude,
                                        value=0)

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

    del data_tmp

    print(f'T - T(condensation): min = {min(data_p):.2f}, max = {max(data_p):.2f}')
    return data_p, data_local_time


def temp_gg2011_fig7(info_netcdf):
    data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above areoid':
        print(info_netcdf.data_dim.altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()
    data_surface_local = info_netcdf.data_dim.altitude

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(info_netcdf.data_dim.time, selected_time=16)
    if not stats_file:
        print('This file is not a stats file required to compare with GG2011')
        exit()
    data = info_netcdf.data_target[idx, :, :, :]
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


def temp_gg2011_fig8(info_netcdf):
    # Slice data: Ls=0-30°
    data, ls = slice_data(data=info_netcdf.data_target,
                          idx_dim_slice=info_netcdf.idx_dim.time,
                          dimension_slice=info_netcdf.data_dim.time,
                          value=[0, 30])
    data_zonal_mean = mean(data, axis=3)

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above areoid':
        print(info_netcdf.data_dim.altitude.long_name)
        print('The netcdf did not zrecasted above the areoid.')
        exit()

    # Check local time available and slice data at 0, 12, 16 H
    data_local_time, idx_0, stats_file = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=0)
    data_zonal_mean_0h = data_zonal_mean[idx_0::len(data_local_time), :, :]
    data_local_time, idx_12, stats_file = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=12)
    data_zonal_mean_12h = data_zonal_mean[idx_12::len(data_local_time), :, :]
    data_local_time, idx_16, stats_file = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=16)
    data_zonal_mean_16h = data_zonal_mean[idx_16::len(data_local_time), :, :]

    # Mean
    data_zonal_mean_0h = mean(data_zonal_mean_0h, axis=0)
    data_zonal_mean_12h = mean(data_zonal_mean_12h, axis=0)
    data_zonal_mean_16h = mean(data_zonal_mean_16h, axis=0)

    # 12h - 00h
    data_thermal_tide = data_zonal_mean_12h - data_zonal_mean_0h

    return data_zonal_mean_16h, data_thermal_tide


def temp_gg2011_fig9(info_netcdf):
    data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')

    # Check the kind of zrecast have been performed : above areoid (A) must be performed
    if info_netcdf.data_dim.altitude.long_name != 'Altitude above areoid':
        print(info_netcdf.data_dim.altitude.long_name)
        print('The netcdf did not zrecasted above the aeroid.')
        exit()
    data_surface_local = info_netcdf.data_dim.altitude

    # Check if we have a stats file with the local time
    data_local_time, idx, stats_file = check_local_time(data_time=info_netcdf.data_dim.time, selected_time=16)
    if not stats_file:
        print('This file is not a stats file required to compare with GG2011')
        exit()
    data = info_netcdf.data_target[idx, :, :, :]
    data_rho = data_rho[idx, :, :, :]

    # Slice data at 0°N latitude
    data, latitude = slice_data(data=data,
                                idx_dim_slice=info_netcdf.idx_dim.latitude,
                                dimension_slice=info_netcdf.data_dim.latitude,
                                value=0)
    data_rho, latitude = slice_data(data=data_rho,
                                    idx_dim_slice=info_netcdf.idx_dim.latitude,
                                    dimension_slice=info_netcdf.data_dim.latitude,
                                    value=0)

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


def temp_stationary_wave(info_netcdf):
    data, latitudes = slice_data(data=info_netcdf.data_target,
                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                 dimension_slice=info_netcdf.data_dim.latitude,
                                 value=0)

    test = input('Do you want performed T-Tcondco2(y/N)?')
    if test.lower() == 'y':
        diff_temp = True
        data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')
        if info_netcdf.local_time is not None:
            data_rho, local_time = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_rho)
        data_rho, latitudes = slice_data(data=data_rho,
                                         idx_dim_slice=info_netcdf.idx_dim.latitude,
                                         dimension_slice=info_netcdf.data_dim.latitude,
                                         value=0)

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


def temp_thermal_structure_polar_region(info_netcdf):
    data_north, latitude_north = slice_data(data=info_netcdf.data_target,
                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                            dimension_slice=info_netcdf.data_dim.latitude,
                                            value=60)
    data_south, latitude_south = slice_data(data=info_netcdf.data_target,
                                            idx_dim_slice=info_netcdf.idx_dim.latitude,
                                            dimension_slice=info_netcdf.data_dim.latitude,
                                            value=-60)

    data_north = mean(data_north, axis=2)
    data_south = mean(data_south, axis=2)

    return data_north, data_south


def temp_cold_pocket(info_netcdf):
    if info_netcdf.data_dim.altitude.units != 'Pa':
        print('Stop ! File did not zrecasted in Pressure')
        exit()

    data_rho, list_var = get_data(filename=info_netcdf.filename, target='rho')
    data_rho, local_time = extract_at_a_local_time(info_netcdf=info_netcdf.filename, data=data_rho)
    data_rho = correction_value(data=data_rho, operator='inf', value=1e-13)

    data_temp_cond_co2 = tcond_co2(data_pressure=None, data_temperature=info_netcdf.data_target, data_rho=data_rho)

    zonal_mean = ma.masked_values(data_temp_cond_co2, 0.)

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


def vars_altitude_ls(info_netcdf, latitude):
    # Diurnal mean
    nb_lt = len(info_netcdf.local_time)
    nb_sols = int(info_netcdf.data_dim.time.shape[0] / nb_lt)
    if nb_lt > 1:
        info_netcdf.data_target = info_netcdf.data_target.reshape(nb_sols, nb_lt,
                                                                  info_netcdf.data_dim.altitude.shape[0],
                                                                  info_netcdf.data_dim.latitude.shape[0],
                                                                  info_netcdf.data_dim.longitude.shape[0])
        info_netcdf.data_target = mean(info_netcdf.data_target, axis=1)

    # zonal mean
    info_netcdf.data_target = mean(info_netcdf.data_target, axis=info_netcdf.idx_dim.longitude)

    # latitude slice
    info_netcdf.data_target, idx_lat = slice_data(data=info_netcdf.data_target,
                                                  dimension_slice=info_netcdf.data_dim.latitude,
                                                  idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                  value=latitude)
    # latitudinal mean
    if not isinstance(latitude, float):
        info_netcdf.data_target = mean(info_netcdf.data_target, axis=2)
    info_netcdf.data_target = info_netcdf.data_target.T
    return


def vars_extract_at_grid_point(info_netcdf, latitude, longitude):
    if len(info_netcdf.local_time) > 1:
        info_netcdf.data_target = compute_diurnal_mean(info_netcdf=info_netcdf, data=info_netcdf.data_target)

    data, latitudes = slice_data(data=info_netcdf.data_target,
                                 idx_dim_slice=info_netcdf.idx_dim.latitude,
                                 dimension_slice=info_netcdf.data_dim.latitude,
                                 value=latitude)

    data, longitudes = slice_data(data=data,
                                  idx_dim_slice=info_netcdf.idx_dim.longitude - 1,
                                  dimension_slice=info_netcdf.data_dim.longitude,
                                  value=longitude)
    return data.T


def vars_max_value_with_others(info_netcdf):
    diurnal_mean = False
    if len(info_netcdf.local_time) > 1:
        diurnal_mean = True
        info_netcdf.data_target = compute_diurnal_mean(info_netcdf=info_netcdf, data=info_netcdf.data_target)

    shape_data_target = info_netcdf.data_target.shape

    print(f'Get max value of {info_netcdf.target_name} in progress...')
    max_mmr, x, y = get_extrema_in_alt_lon(info_netcdf.data_target, extrema='max')
    print('Extract other variable at co2_ice max value:')

    print(' (1) Temperature')
    data_temperature, list_var = get_data(filename=info_netcdf.filename, target='temp')
    if diurnal_mean:
        data_temperature = compute_diurnal_mean(info_netcdf=info_netcdf, data=data_temperature[:, :, :, :])
    else:
        data_temperature, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_temperature)
    max_temp = extract_at_max_co2_ice(data_temperature, x, y, shape_data_target)
    del data_temperature

    print(' (2) Saturation')
    data_satuco2, list_var = get_data(filename=info_netcdf.filename, target='satuco2')
    if diurnal_mean:
        data_satuco2 = compute_diurnal_mean(info_netcdf=info_netcdf, data=data_satuco2[:, :, :, :])
    else:
        data_satuco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_satuco2)
    max_satu = extract_at_max_co2_ice(data_satuco2, x, y, shape_data_target)
    del data_satuco2

    print(' (3) CCN radius')
    data_riceco2, list_var = get_data(filename=info_netcdf.filename, target='riceco2')
    if diurnal_mean:
        data_riceco2 = compute_diurnal_mean(info_netcdf=info_netcdf, data=data_riceco2[:, :, :, :])
    else:
        data_riceco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_riceco2)
    max_radius = extract_at_max_co2_ice(data_riceco2, x, y, shape_data_target)
    del data_riceco2

    print(' (4) CCN number')
    data_ccn_nco2, list_var = get_data(filename=info_netcdf.filename, target='ccnNco2')
    if diurnal_mean:
        data_ccn_nco2 = compute_diurnal_mean(info_netcdf=info_netcdf, data=data_ccn_nco2[:, :, :, :])
    else:
        data_ccn_nco2, tmp = extract_at_a_local_time(info_netcdf=info_netcdf, data=data_ccn_nco2)
    max_ccn_n = extract_at_max_co2_ice(data_ccn_nco2, x, y, shape_data_target)
    del data_ccn_nco2

    print(' (5) Altitude')  # 1 dim !
    max_alt = extract_at_max_co2_ice(info_netcdf.data_dim.altitude, x, y, shape_data_target)

    print('Reshape data in progress...')
    max_mmr, max_temp, max_satu, max_radius, max_ccn_n, max_alt = rotate_data(max_mmr, max_temp, max_satu,
                                                                              max_radius, max_ccn_n, max_alt,
                                                                              do_flip=True)

    max_mmr = correction_value(data=max_mmr, operator='invalide')
    max_temp = correction_value(data=max_temp, operator='invalide')
    max_satu = correction_value(data=max_satu, operator='invalide')
    max_radius = correction_value(data=max_radius, operator='invalide')
    max_ccn_n = correction_value(data=max_ccn_n, operator='invalide')
    max_alt = correction_value(data=max_alt, operator='invalide')

    return max_mmr, max_temp, max_satu, max_radius*1e6, max_ccn_n, max_alt


def vars_time_mean(info_netcdf, duration):
    from math import ceil

    if info_netcdf.data_dim.time.units != 'degrees':
        if len(info_netcdf.local_time) == 1:
            data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                           selected_time=info_netcdf.local_time)
            if info_netcdf.data_dim.time[-1] <= 360.:  # Ensure we are in ls time coordinate
                data_time = info_netcdf.data_dim.time[idx::len(data_local_time)]
            else:
                data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
                data_time = data_ls[idx::len(data_local_time)]
        else:
            data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
            data_time = data_ls[:]
    else:
        data_time = info_netcdf.data_dim.time

    if duration:
        nbin = ceil(data_time[-1] / duration)
        time_bin = arange(0, data_time[-1] + duration, duration)
        if info_netcdf.data_target.ndim == 3:
            data_mean = zeros((nbin, info_netcdf.data_dim.latitude.shape[0], info_netcdf.data_dim.longitude.shape[0]))
            for i in range(nbin):
                data_sliced, time = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.time,
                                               dimension_slice=data_time,
                                               value=[duration * i, duration * (i + 1)])
                data_mean[i, :, :] = mean(data_sliced, axis=0)
        elif info_netcdf.data_target.ndim == 4:
            data_mean = zeros((nbin, info_netcdf.data_dim.altitude.shape[0], info_netcdf.data_dim.latitude.shape[0],
                               info_netcdf.data_dim.longitude.shape[0]))
            for i in range(nbin):
                data_sliced, time = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.time,
                                               dimension_slice=data_time,
                                               value=[duration * i, duration * (i + 1)])
                data_mean[i, :, :, :] = mean(data_sliced, axis=0)
        else:
            data_mean = 0
            print('Wrong dim!')
            exit()
    else:
        data_mean = mean(info_netcdf.data_target, axis=0)
        time_bin = None
    data_mean = correction_value(data=data_mean, operator='inf', value=threshold)

    return data_mean, time_bin


def vars_zonal_mean(data_input, layer=None, flip=None):
    from .create_infofile import InfoFile
    data, layer_selected = None, None

    if isinstance(data_input, InfoFile):
        if layer is not None:
            if data_input.filename != '':
                data_altitude = data_input.data_dim.altitude
                if data_input.data_dim.altitude.units in ['Pa']:
                    data_altitude = data_altitude[::-1]  # in pressure coordinate, the direction is reversed
                    data_input.data_target = data_input.data_target[:, ::-1, :, :]
                data, layer_selected = slice_data(data=data_input.data_target,
                                                  idx_dim_slice=data_input.idx_dim.altitude,
                                                  dimension_slice=data_input.data_dim.altitude,
                                                  value=float(data_altitude[layer]))
            else:
                # for observational data
                data = data_input.data_target[:, layer, :, :]
                layer_selected = None
        else:
            layer_selected = None
            data = data_input.data_target
    elif isinstance(data_input, ndarray):
        data = data_input
    else:
        print("Wrong object/value info_netcdf.")
        exit()

    if data.ndim == 3:
        zonal_mean = mean(data[:, :, :], axis=2)
    elif data.ndim == 2:
        zonal_mean = mean(data[:, :], axis=1)
    elif data.ndim == 4:
        zonal_mean = mean(data[:, :, :, :], axis=3)
        zonal_mean = mean(zonal_mean, axis=1)
    else:
        zonal_mean = 0
        print('wrong ndim')
        exit()

    # Diurnal mean
    if len(data_input.local_time) > 1:
        print(zonal_mean.shape, zonal_mean)
        zonal_mean = zonal_mean.reshape((669, 12, zonal_mean.shape[1]))
        zonal_mean = mean(zonal_mean, axis=1)

    if flip is None:
        zonal_mean = rotate_data(zonal_mean, do_flip=True)[0]
    else:
        zonal_mean = rotate_data(zonal_mean, do_flip=False)[0]
    del data

    return zonal_mean, layer_selected


def vars_zonal_mean_column_density(info_netcdf):
    # diurnal mean
    nb_lt = len(info_netcdf.local_time)
    if nb_lt > 1:
        nb_sols = int(info_netcdf.data_dim.time.shape[0] / nb_lt)
        info_netcdf.data_target = info_netcdf.data_target.reshape(nb_sols, nb_lt,
                                                                  info_netcdf.data_dim.altitude.shape[0],
                                                                  info_netcdf.data_dim.latitude.shape[0],
                                                                  info_netcdf.data_dim.longitude.shape[0])
        info_netcdf.data_target = mean(info_netcdf.data_target, axis=1)

    altitude_limit, idx_altitude_min, idx_altitude_max = compute_column_density(info_netcdf=info_netcdf)

    # compute zonal mean column density
    if info_netcdf.data_target.ndim == 3:
        info_netcdf.data_target = mean(info_netcdf.data_target, axis=2)  # Ls function of lat
    else:
        print('Stop wrong ndim !')
        exit()
    info_netcdf.data_target = rotate_data(info_netcdf.data_target, do_flip=False)[0]

    return altitude_limit, idx_altitude_min, idx_altitude_max


def vars_zonal_mean_where_co2ice_exists(info_netcdf, polar_region):
    data_where_co2_ice = extract_where_co2_ice(info_netcdf=info_netcdf)

    if polar_region:
        # Slice data in north and south polar regions
        data_where_co2_ice_np, north_latitude = slice_data(data=data_where_co2_ice,
                                                           idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                           dimension_slice=info_netcdf.data_dim.latitude,
                                                           value=[45, 90])
        data_where_co2_ice_sp, south_latitude = slice_data(data=data_where_co2_ice,
                                                           idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                           dimension_slice=info_netcdf.data_dim.latitude,
                                                           value=[-45, -90])

        data_where_co2_ice_np_mean = mean(data_where_co2_ice_np, axis=3)
        data_where_co2_ice_sp_mean = mean(data_where_co2_ice_sp, axis=3)
        list_data = ([data_where_co2_ice_np_mean, data_where_co2_ice_sp_mean])
        del data_where_co2_ice, data_where_co2_ice_np, data_where_co2_ice_sp

    else:
        data_where_co2_ice = mean(data_where_co2_ice, axis=3)
        list_data = ([data_where_co2_ice])

    return list_data


def vars_zonal_mean_in_time_co2ice_exists(info_netcdf):
    lat1 = int(input('\t Latitude range 1 (°N): '))
    lat2 = int(input('\t Latitude range 2 (°N): '))

    if info_netcdf.data_dim.time.units != 'deg':
        data_ls, list_var = get_data(filename='../concat_Ls.nc', target='Ls')
        if info_netcdf.local_time is not None:
            data_local_time, idx, stats = check_local_time(data_time=info_netcdf.data_dim.time,
                                                           selected_time=info_netcdf.local_time)
            data_time = data_ls[idx::len(data_local_time)]
        else:
            idx = None
            data_local_time = None
            data_time = data_ls
    else:
        idx = None
        data_local_time = None
        data_time = info_netcdf.data_dim.time

    # extract co2_ice data
    data_co2_ice, list_var = get_data(filename=info_netcdf.filename, target='co2_ice')

    # select the latitude range
    data_sliced_lat, idx_latitude = slice_data(data=info_netcdf.data_target,
                                               idx_dim_slice=info_netcdf.idx_dim.latitude,
                                               dimension_slice=info_netcdf.data_dim.latitude,
                                               value=[lat1, lat2])
    data_co2_ice_sliced_lat, idx_latitude = slice_data(data=data_co2_ice[:, :, :, :],
                                                       idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                       dimension_slice=info_netcdf.data_dim.latitude,
                                                       value=[lat1, lat2])

    latitude_selected = info_netcdf.data_dim.latitude[idx_latitude[0]:idx_latitude[1]]
    del data_co2_ice

    # extract at local time
    if info_netcdf.local_time is not None:
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
        filenames = list([f'{info_netcdf.target_name}_mean_{latitude_selected[0]:.0f}N_'
                          f'{latitude_selected[-1]:.0f}N_0-360Ls'])
        list_time_selected = list(f'{data_time[0]} - {data_time[1]}')
    else:
        directory_output = f'{info_netcdf.target_name}_mean_radius_{latitude_selected[0]:.0f}N_' \
                           f'{latitude_selected[-1]:.0f}N'
        if not path.exists(directory_output):
            mkdir(directory_output)

        time_step = float(input(f'Select the time step range (°): '))
        nb_step = int(data_time[-1] / time_step) + 1

        list_data = list([])
        filenames = list([])
        list_time_selected = list([])
        for i in range(nb_step):
            data_sliced_lat_ls, idx_time = slice_data(data=data_sliced_lat,
                                                      idx_dim_slice=info_netcdf.idx_dim.time,
                                                      dimension_slice=data_time[:],
                                                      value=[i * time_step, (i + 1) * time_step])
            data_co2_ice_sliced_lat_ls, idx_time = slice_data(data=data_co2_ice_sliced_lat,
                                                              idx_dim_slice=info_netcdf.idx_dim.time,
                                                              dimension_slice=data_time[:],
                                                              value=[i * time_step, (i + 1) * time_step])

            time_selected = data_time[idx_time[0]:idx_time[1]]

            print(f'\t\tselected: {time_selected[0]:.0f} {time_selected[-1]:.0f}')
            list_time_selected.append(f'{time_selected[0]:.0f} - {time_selected[-1]:.0f} °Ls')
            # Mask data where co2ice is inferior to 1e-13, so where co2ice exists
            data_final = ma.masked_where(data_co2_ice_sliced_lat_ls < 1e-13, data_sliced_lat_ls)

            del data_co2_ice_sliced_lat_ls, data_sliced_lat_ls

            data_final = mean(mean(data_final, axis=3), axis=0)  # zonal mean and temporal mean
            list_data.append(data_final)
            filenames.append(f'{directory_output}/{info_netcdf.target_name}_mean_{latitude_selected[0]:.0f}N_'
                             f'{latitude_selected[-1]:.0f}N_Ls_{time_selected[0]:.0f}-'
                             f'{time_selected[-1]:.0f}_{info_netcdf.local_time:.0f}h')

        del data_sliced_lat, data_co2_ice_sliced_lat

    return list_data, filenames, latitude_selected, list_time_selected


def vars_localtime_longitude(info_netcdf, latitude, altitude):
    data, idx_latitude = slice_data(data=info_netcdf.data_target,
                                    idx_dim_slice=info_netcdf.idx_dim.latitude,
                                    dimension_slice=info_netcdf.data_dim.latitude,
                                    value=latitude)
    data, idx_altitude = slice_data(data=data,
                                    idx_dim_slice=info_netcdf.idx_dim.altitude,
                                    dimension_slice=info_netcdf.data_dim.altitude,
                                    value=altitude)

    data_mean = zeros((12, data.shape[1]))
    for i in range(12):
        data_mean[i, :] = mean(data[i::12, :], axis=0)
    return data_mean


def vars_ls_longitude(info_netcdf, latitude, altitude):
    info_netcdf.data_target = compute_diurnal_mean(info_netcdf=info_netcdf, data=info_netcdf.data_target)

    info_netcdf.data_target, idx_latitude = slice_data(data=info_netcdf.data_target,
                                                       dimension_slice=info_netcdf.data_dim.latitude,
                                                       idx_dim_slice=info_netcdf.idx_dim.latitude,
                                                       value=latitude)

    info_netcdf.data_target, idx_altitude = slice_data(data=info_netcdf.data_target,
                                                       dimension_slice=info_netcdf.data_dim.altitude,
                                                       idx_dim_slice=info_netcdf.idx_dim.altitude,
                                                       value=altitude)

    info_netcdf.data_target = info_netcdf.data_target.T
    return


def vars_localtime_ls(info_netcdf, latitude, altitude):
    data, idx_latitude = slice_data(data=info_netcdf.data_target,
                                    idx_dim_slice=info_netcdf.idx_dim.latitude,
                                    dimension_slice=info_netcdf.data_dim.latitude,
                                    value=latitude)
    data, idx_altitude = slice_data(data=data,
                                    idx_dim_slice=info_netcdf.idx_dim.altitude,
                                    dimension_slice=info_netcdf.data_dim.altitude,
                                    value=altitude)

    data = mean(data, axis=1)

    nb_sol = int(data.shape[0] / 12)
    data = data.reshape(nb_sol, 12)  # => hl, lon

    return data.T


def vars_min_mean_max(info_netcdf, latitude, altitude):
    info_netcdf.data_target, tmp = slice_data(data=info_netcdf.data_target,
                                              dimension_slice=info_netcdf.data_dim.latitude,
                                              idx_dim_slice=info_netcdf.idx_dim.latitude,
                                              value=latitude)

    info_netcdf.data_target, idx = slice_data(data=info_netcdf.data_target,
                                              dimension_slice=info_netcdf.data_dim.altitude,
                                              idx_dim_slice=info_netcdf.idx_dim.altitude,
                                              value=altitude)
    info_netcdf.data_dim.altitude = info_netcdf.data_dim.altitude[idx[0]:idx[1] + 1]

    with open(file=f"{info_netcdf.target_name}_min_mean_max.dat", mode='w') as fin:
        fin.write(f"{min(info_netcdf.data_target)}, {mean(info_netcdf.data_target)}, {max(info_netcdf.data_target)}\n")

    info_netcdf.data_target = swapaxes(info_netcdf.data_target, axis1=1, axis2=3)
    info_netcdf.data_target = info_netcdf.data_target.reshape(info_netcdf.data_target.shape[0] *
                                                              info_netcdf.data_target.shape[1] *
                                                              info_netcdf.data_target.shape[2],
                                                              info_netcdf.data_target.shape[-1]
                                                              )
    return


def vars_select_profile(info_netcdf):
    print('To be done')
    print('Select latitude, longitude, altitude, time to extract profile')
    print('Perform a list of extracted profile')
    print(info_netcdf.data_target)
    exit()
