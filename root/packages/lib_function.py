from sys import exit
from .ncdump import get_data


# correct very low values of co2/h2o mmr
def correction_value(data, operator, threshold):
    from numpy import ma

    if operator == 'inf':
        data = ma.masked_where(data <= threshold, data, False)
    elif operator == 'sup':
        data = ma.masked_where(data >= threshold, data, False)
    elif operator == 'eq':
        data = ma.masked_values(data, threshold)

    return data


def check_local_time(data_time, selected_time=None):
    from numpy import unique, round, delete

    # Deals with stats file
    if data_time.shape[0] == 12:
        data_local_time = data_time
        stats_file = True
    else:
        data_local_time = unique(round(data_time[:] * 24 % 24, 0))
        if 0 in data_local_time and 24 in data_local_time:
            data_local_time = delete(data_local_time, -1)
        stats_file = False

    print(f'Local time available: {data_local_time}')

    if selected_time is not None:
        idx = (abs(data_local_time[:] - selected_time)).argmin()
        print(f'\tSelected: {data_local_time[idx]}')
    else:
        test = input('Do you want extract at a local time (y/N)? ')
        if test.lower() == 'y':
            selected_time = int(input('\tEnter the local time: '))
            idx = (abs(data_local_time[:] - selected_time)).argmin()
        else:
            idx = None
            data_local_time = [0]  # ! otherwise it takes we have data[0::len(data_local_time)] !

    return data_local_time, idx, stats_file


def create_gif(filenames):
    import numpy as np
    import imageio

    make_gif = input('Do you want create a gif (Y/n)?: ')
    if make_gif.lower() == 'y':
        if all(".png" in s for s in filenames):
            filenames = [x for x in filenames]
        else:
            filenames = [x + '.png' for x in filenames]

        images = []
        idx = np.array([], dtype=np.int)

        print("Select files using number (one number per line/all): ")
        for i, value_i in enumerate(filenames):
            print('({}) {:}'.format(i, value_i))

        add_file = True
        while add_file:
            value = input('')
            if value == '':
                add_file = False
            elif value == 'all':
                idx = np.arange(len(filenames))
                break
            else:
                idx = np.append(idx, int(value))
        filenames = [filenames[i] for i in idx]

        save_name = input('Enter the gif name: ')
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{save_name}.gif', images, fps=1)
    else:
        pass

    return


def convert_sols_to_ls():
    from numpy import array

    # sols to ls, step 5°ls
    time_grid_ls = array([0, 10, 20, 30, 41, 51, 61, 73, 83, 94, 105, 116, 126, 139, 150, 160, 171, 183, 193.47,
                          205, 215, 226, 236, 248, 259, 269, 279, 289, 299, 309, 317, 327, 337, 347, 355, 364,
                          371.99, 381, 390, 397, 406, 415, 422, 430, 437, 447, 457, 467, 470, 477, 485, 493, 500,
                          507, 514.76, 523, 533, 539, 547, 555, 563, 571, 580, 587, 597, 605, 613, 623, 632, 641,
                          650, 660, 669])

    return time_grid_ls


def compute_column_density(filename, data):
    from numpy import zeros, sum

    data_altitude = get_data(filename, target='altitude')

    if data_altitude.units in ['m', 'km']:
        data_pressure = get_data(filename, target='pressure')
    else:
        data_pressure = data_altitude

    altitude_limit = input('Do you want perform the computation on the entire column(Y/n)? ')

    if altitude_limit.lower() == 'n':
        if data_altitude.units in ['m', 'km']:
            print(f'Altitude range (km): {data_altitude[0]:.3f} {data_altitude[-1]:.3f}')
            altitude_min = float(input('Start altitude (km): '))
            altitude_max = float(input('End altitude (km): '))
        else:
            print(f'Pressure range (Pa): {data_altitude[0]:.3e} {data_altitude[-1]:.3e}')
            altitude_min = float(input('Start altitude (Pa): '))
            altitude_max = float(input('End altitude (Pa): '))
    else:
        altitude_min = data_altitude[0]
        altitude_max = data_altitude[-1]

    idx_z_min = (abs(data_altitude[:] - altitude_min)).argmin()
    idx_z_max = (abs(data_altitude[:] - altitude_max)).argmin()

    if idx_z_min > idx_z_max:
        tmp = idx_z_min
        idx_z_min = idx_z_max
        idx_z_max = tmp + 1
    else:
        idx_z_max += 1
    data = data[:, idx_z_min:idx_z_max, :, :]

    shape_data = data.shape
    data_column = zeros(shape_data)

    alt = 0
    if data_altitude.units in ['m', 'km']:
        for alt in range(data.shape[1] - 1):
            data_column[:, alt, :, :] = data[:, alt, :, :] * \
                                        (data_pressure[:, alt, :, :] - data_pressure[:, alt + 1, :, :]) / 3.711  # g
        data_column[:, -1, :, :] = data[:, -1, :, :] * data_pressure[:, -1, :, :] / 3.711
    else:
        for alt in range(data.shape[1] - 1):
            data_column[:, alt, :, :] = data[:, alt, :, :] * \
                                        (data_altitude[idx_z_min + alt] - data_altitude[
                                            idx_z_min + alt + 1]) / 3.711  # g
        data_column[:, -1, :, :] = data[:, -1, :, :] * data_altitude[idx_z_min + alt + 1] / 3.711

    data_column = correction_value(data_column, 'inf', threshold=1e-13)
    data_column = sum(data_column, axis=1)
    data_column = correction_value(data_column, 'inf', threshold=1e-13)

    return data_column, altitude_limit, altitude_min, altitude_max, data_altitude.units


def extract_at_a_local_time(filename, data, local_time=None):
    data_time = get_data(filename=filename, target='Time')

    data_local_time, idx, stats_file = check_local_time(data_time=data_time, selected_time=local_time)

    if idx is not None:
        local_time = data_local_time[idx]
        if data.ndim == 4:
            data_processed = data[idx::len(data_local_time), :, :, :]
        elif data.ndim == 3:
            data_processed = data[idx::len(data_local_time), :, :]
        else:
            data_processed = data[idx::len(data_local_time)]
    else:
        local_time = None
        data_processed = data

    return data_processed, local_time


def extract_at_max_co2_ice(data, x, y, shape_big_data):
    from numpy import asarray, reshape, swapaxes

    # 1-D
    if data.ndim == 1:
        data_max = [data[x[i, j]] for i in range(shape_big_data[0]) for j in range(shape_big_data[2])]
        data_max = asarray(data_max)
        data_max = reshape(data_max, (shape_big_data[0], shape_big_data[2]))
    # 4-D
    elif data.ndim == 4:
        data_max = swapaxes(data, 1, 2)
        data_max = [data_max[i, j, x[i, j], y[i, j]] for i in range(data_max.shape[0]) for j in
                    range(data_max.shape[1])]
        data_max = asarray(data_max)
        data_max = reshape(data_max, (data.shape[0], data.shape[2]))
    else:
        print('Wrong dimension or taken into account')
        data_max = 0
        exit()

    return data_max


def extract_where_co2_ice(filename, data):
    from numpy import nan, ma

    # extract co2_ice data
    data_co2_ice = get_data(filename, target='co2_ice')
    data_where_co2_ice = ma.masked_where(data_co2_ice[:, :, :, :] < 1e-13, data, nan)
    del data_co2_ice

    return data_where_co2_ice


def extract_vars_max_along_lon(data, idx_lon=None):
    from numpy import unravel_index, argmax, asarray

    # Find the max value along longitude
    if idx_lon is None:
        if data.ndim == 3:
            tmp, idx_lon = unravel_index(argmax(data.reshape(data.shape[0], -1), axis=1), data.shape[1:3])
        else:
            print('Dimension is not equal to 3, you can do a mistake, have you slice the data for one latitude?')

    # Extract data at the longitude where max is observed, for each time ls
    data = [data[i, :, idx_lon[i]] for i in range(data.shape[0])]
    data = asarray(data)

    return data, idx_lon


def gcm_area():
    filename = '/home/mathe/Documents/owncloud/GCM/gcm_aire_phisinit.nc'
    return get_data(filename=filename, target='aire')


def gcm_surface_local(data_zaeroid=None):
    filename = '/home/mathe/Documents/owncloud/GCM/gcm_aire_phisinit.nc'
    data_phisinit = get_data(filename=filename, target='phisinit')

    if data_zaeroid is None:
       data_surface_local = data_phisinit[:, :] / 3.711
    else:
        data_surface_local = data_zaeroid - data_phisinit[:, :] / 3.711

    return data_surface_local


def get_extrema_in_alt_lon(data, extrema):
    from numpy import swapaxes, unravel_index, asarray, reshape
    # get max value along altitude and longitude
    # axis: 0 = Time, 1 = altitude, 2 = latitude, 3 = longitude
    print('In get_extrema')
    data_swap_axes = swapaxes(data, 1, 2)
    max_idx = None
    if extrema == 'max':
        max_idx = data_swap_axes.reshape((data_swap_axes.shape[0], data_swap_axes.shape[1], -1)).argmax(2)
    elif extrema == 'min':
        max_idx = data_swap_axes.reshape((data_swap_axes.shape[0], data_swap_axes.shape[1], -1)).argmin(2)

    x, y = unravel_index(max_idx, data_swap_axes[0, 0, :].shape)
    data_max = [data_swap_axes[i, j, x[i, j], y[i, j]] for i in range(data_swap_axes.shape[0]) for j in
                range(data_swap_axes.shape[1])]
    data_max = asarray(data_max)
    data_max = reshape(data_max, (data.shape[0], data.shape[2]))

    return data_max, x, y


def get_nearest_clouds_observed(data_obs, dim, data_dim, value):
    from numpy import abs

    if dim is 'latitude':

        # From the dimension, get the index(es) of the slice
        latitude_range = None
        if (isinstance(value, float) is True) or (isinstance(value, int) is True):
            if value > 0:
                idx = abs(data_dim[:] - value).argmin()
            else:
                idx = abs(data_dim[:] - value).argmin() + 1
            latitude_range = data_dim[idx - 1:idx + 1]
        elif len(value) == 2:
            idx1 = (abs(data_dim[:] - value[0])).argmin()
            idx2 = (abs(data_dim[:] - value[1])).argmin()

            if idx1 > idx2:
                tmp = idx1
                idx1 = idx2
                idx2 = tmp + 1
            else:
                idx2 += 1
            latitude_range = data_dim[idx1:idx2]
        else:
            print('Error in value given, exceed 2 values')
            print(value)
            exit()

        # Case for southern latitudes
        if (latitude_range[0] < 0) and (latitude_range[-1] < 0):
            mask = (data_obs[:, 1] <= latitude_range[0]) & (data_obs[:, 1] >= latitude_range[-1])
        # Case for northern latitudes
        elif (latitude_range[0] > 0) and (latitude_range[-1] > 0):
            mask = (data_obs[:, 1] >= latitude_range[0]) & (data_obs[:, 1] <= latitude_range[-1])
        # Case for both hemisphere latitudes
        else:
            if latitude_range[0] > latitude_range[-1]:
                tmp = latitude_range[0]
                latitude_range[0] = latitude_range[-1]
                latitude_range[-1] = tmp
            mask = (data_obs[:, 1] >= latitude_range[0]) & (data_obs[:, 1] <= latitude_range[-1])

        data_ls = data_obs[:, 0][mask]
        data_latitude = data_obs[:, 1][mask]

    else:
        data_ls = None
        data_latitude = None
        print('You do not search the nearest cloud along latitude')

    return data_ls, data_latitude


def get_ls_index(data_time):
    from numpy import array, searchsorted, max

    axis_ls = array([0, 90, 180, 270, 360])
    if max(data_time) > 361:
        # ls = 0, 90, 180, 270, 360
        idx = searchsorted(data_time[:], [0, 193.47, 371.99, 514.76, 669])
        ls_lin = True
    else:
        idx = searchsorted(data_time[:], axis_ls)
        ls_lin = False

    return idx, axis_ls, ls_lin


def get_mean_index_altitude(data_altitude, value, dimension):
    from numpy import abs, zeros, mean

    if dimension == 'time':
        mean_idx = zeros(data_altitude.shape[0], dtype=int)
        idx = zeros(data_altitude.shape[2], dtype=int)

        # (1) compute for each time (ls),
        # (2) search the index where _value_ km is reached for all longitude
        # (3) compute the mean of the index
        for ls in range(data_altitude.shape[0]):
            for longitude in range(data_altitude.shape[2]):
                idx[longitude] = (abs(data_altitude[ls, :, longitude] - value)).argmin()
            mean_idx[ls] = mean(idx)
            idx[:] = 0

    elif dimension == 'latitude':
        # data_altitude = [ls, alt, lat, lon]
        data_altitude = mean(data_altitude, axis=3)
        mean_idx = zeros(data_altitude.shape[2], dtype=int)
        idx = zeros((data_altitude.shape[0]), dtype=int)

        for latitude in range(data_altitude.shape[2]):
            for ls in range(data_altitude.shape[0]):
                idx[ls] = (abs(data_altitude[ls, :, latitude] - value)).argmin()
            mean_idx[latitude] = mean(idx)
            idx[:] = 0

    elif dimension == 'longitude':
        mean_idx = zeros(data_altitude.shape[2], dtype=int)
        idx = zeros(data_altitude.shape[0], dtype=int)
        for longitude in range(data_altitude.shape[2]):
            for ls in range(data_altitude.shape[0]):
                idx[ls] = (abs(data_altitude[ls, :, longitude] - value)).argmin()
            mean_idx[longitude] = mean(idx)
            idx[:] = 0
    else:
        print(f'Dimension {dimension} is not supported')
        mean_idx = None

    return mean_idx


def linearize_ls(data, data_ls):
    from numpy import arange
    from scipy.interpolate import interp2d

    # interpolation to get linear Ls
    f = interp2d(x=data_ls, y=arange(data.shape[0]), z=data, kind='linear')

    interp_time = arange(361)
    data = f(interp_time, arange(data.shape[0]))
    return data, interp_time


def linear_grid_ls(data):
    from numpy import linspace, searchsorted

    axis_ls = linspace(0, 360, data.shape[0])
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data, axis_ls)
    axis_ls = axis_ls[ndx]

    return interp_time, axis_ls, ndx


def rotate_data(*list_data, do_flip):
    from numpy import flip
    list_data = list(list_data)

    for i, value in enumerate(list_data):
        list_data[i] = list_data[i].T  # get Ls on x-axis
        if do_flip:
            list_data[i] = flip(list_data[i], axis=0)  # reverse to get North pole on top of the fig

    return list_data


def slice_data(data, dimension_data, value):
    idx, idx1, idx2, idx_dim = None, None, None, None

    # Select the dimension where the slice will be done
    # TODO: check if 2 dim have the same length ....
    for i in range(data.ndim):
        if data.shape[i] == dimension_data.shape[0]:
            idx_dim = dimension_data.shape[0]

    # Ensure we know what we do
    if idx_dim is None:
        print('Issue with the data and dimension')
        print(f'data.shape: {data.shape}')
        print(f'dimension.shape: {dimension_data.shape}')
        exit()

    # From the dimension, get the index(es) of the slice
    if (isinstance(value, float) is True) or (isinstance(value, int) is True):
        idx = (abs(dimension_data[:] - value)).argmin()
        selected_idx = float(dimension_data[idx])

    elif len(value) == 2:
        idx1 = (abs(dimension_data[:] - value[0])).argmin()
        idx2 = (abs(dimension_data[:] - value[1])).argmin()

        if idx1 > idx2:
            tmp = idx1
            idx1 = idx2
            idx2 = tmp + 1
        else:
            idx2 += 1
        selected_idx = dimension_data[idx1:idx2]

    else:
        print('Error in value given, exceed 2 values')
        print(value)
        selected_idx = None
        exit()

    if data.ndim == 1:
        if idx is not None:
            data = data[idx]
        else:
            data = data[idx1:idx2]

    elif data.ndim == 2:
        # 1st dimension
        if dimension_data.shape[0] == data.shape[0]:
            if idx is not None:
                data = data[idx, :]
            else:
                data = data[idx1:idx2, :]

        # 2nd dimension
        elif dimension_data.shape[0] == data.shape[1]:
            if idx is not None:
                data = data[:, idx]
            else:
                data = data[:, idx1:idx2]

    elif data.ndim == 3:
        # 1st dimension
        if dimension_data.shape[0] == data.shape[0]:
            if idx is not None:
                data = data[idx, :, :]
            else:
                data = data[idx1:idx2, :, :]

        # 2nd dimension
        elif dimension_data.shape[0] == data.shape[1]:
            if idx is not None:
                data = data[:, idx, :]
            else:
                data = data[:, idx1:idx2, :]

        # 3rd dimension
        elif dimension_data.shape[0] == data.shape[2]:
            if idx is not None:
                data = data[:, :, idx]
            else:
                data = data[:, :, idx1:idx2]

    elif data.ndim == 4:
        # 1st dimension
        if dimension_data.shape[0] == data.shape[0]:
            if idx is not None:
                data = data[idx, :, :, :]
            else:
                data = data[idx1:idx2, :, :, :]

        # 2nd dimension
        elif dimension_data.shape[0] == data.shape[1]:
            if idx is not None:
                data = data[:, idx, :, :]
            else:
                data = data[:, idx1:idx2, :, :]

        # 3rd dimension
        elif dimension_data.shape[0] == data.shape[2]:
            if idx is not None:
                data = data[:, :, idx, :]
            else:
                data = data[:, :, idx1:idx2, :]

        # 4th dimension
        elif dimension_data.shape[0] == data.shape[3]:
            if idx is not None:
                data = data[:, :, :, idx]
            else:
                data = data[:, :, :, idx1:idx2]
        else:
            print('The dimension of data exceed dimension 4 !')
            exit()

    else:
        print(f'Data has {data.ndim} dimension!')
        exit()

    return data, selected_idx


def tcond_co2(data_pressure=None, data_temperature=None, data_rho=None):
    from numpy import log10, log

    cst_r = 8.31

    if data_pressure is not None:
        a = 6.81228
        b = 1301.679
        c = -3.494

        t_sat = b / (a - log10((data_pressure + 1e-13) / 10 ** 5)) - c
    elif (data_temperature is not None) and (data_rho is not None):
        # Equation from Washburn (1948) ; tcond in G-G2011
        t_sat = -3148. / (log(0.01 * data_temperature * data_rho * cst_r) - 23.102)  # rho en kg/m3
    else:
        t_sat = None

    return t_sat
