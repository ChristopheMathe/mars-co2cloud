from sys import exit
from .ncdump import getdata


# correct very low values of co2/h2o mmr
def correction_value(data, threshold):
    from numpy import ma

    data = ma.masked_where(data <= threshold, data)

    return data


def get_extrema_in_alt_lon(data, extrema):
    from numpy import swapaxes, unravel_index, asarray, reshape, flip
    # get max value along altitude and longitude
    # max_mmr = amax(data_y, axis=(1,3)) # get the max mmr value in longitude/altitude
    # axis : 0 = Time, 1 = altitude, 2 = latitude, 3 = longitude
    print('In get_extrema')
    B = swapaxes(data, 1, 2)
    if extrema == 'max':
        max_idx = B.reshape((B.shape[0], B.shape[1], -1)).argmax(2)
    elif extrema == 'min':
        max_idx = B.reshape((B.shape[0], B.shape[1], -1)).argmin(2)

    x, y = unravel_index(max_idx, B[0, 0, :].shape)
    data_max = [B[i, j, x[i, j], y[i, j]] for i in range(B.shape[0]) for j in range(B.shape[1])]
    data_max = asarray(data_max)
    data_max = reshape(data_max, (data.shape[0], data.shape[2]))

    return data_max, x, y


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


def extract_vars_max_along_lon(data, idx_lon=None):
    from numpy import unravel_index, argmax, asarray

    # Find the max value along longitude
    if idx_lon is None:
        if data.ndim == 3:
            tmp, idx_lon = unravel_index(argmax(data.reshape(data.shape[0], -1), axis=1), data.shape[1:3])
        else:
            print('Ndim is not equal to 3, you can do a mistake, have you slice the data for one latitude?')

    # Extract data at the longitude where max is observed, for each time ls
    data = [data[i, :, idx_lon[i]] for i in range(data.shape[0])]
    data = asarray(data)

    return  data, idx_lon


def linearize_ls(data, dim_time, dim_latitude, interp_time):
    from numpy import arange
    from scipy.interpolate import interp2d

    # interpolation to get linear Ls
    print('In linearize_ls')
    f = interp2d(x=arange(dim_time), y=arange(dim_latitude), z=data, kind='linear')

    data = f(interp_time, arange(dim_latitude))
    return data


def linear_grid_ls(data):
    from numpy import linspace, searchsorted

    axis_ls = linspace(0, 360, data.shape[0])
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    interp_time = searchsorted(data, axis_ls)
    axis_ls = axis_ls[ndx]

    return interp_time, axis_ls, ndx


def tcondco2(data_pressure, idx_ls=None, idx_lat=None, idx_lon=None):
    from numpy import log10

    A = 6.81228
    B = 1301.679
    C = -3.494
    if (idx_ls is not None) and (idx_lat is not None) and (idx_lon is not None):
        T_sat = B / (A - log10((data_pressure[idx_ls, :, idx_lat, idx_lon] + 0.0001) / 10 ** 5)) - C
    else:
        T_sat = B / (A - log10((data_pressure[:, :, :, :] + 0.0001) / 10 ** 5)) - C

    return T_sat


def compute_zonal_mean_column_density(data_target, data_pressure, data_altitude):
    from numpy import mean, sum, zeros, max, min

    altitude_limit = input('Do you want perform the computation on the entire column(Y/n)? ')

    if altitude_limit.lower() == 'n':
        if data_altitude.units in ['m', 'km']:
            print('Altitude range (km): {:.3f} {:.3f}'.format(data_altitude[0], data_altitude[-1]))
            zmin = float(input('Start altitude (km): '))
            zmax = float(input('End altitude (km): '))
        else:
            print('Pressure range (Pa): {:.3e} {:.3e}'.format(data_altitude[0], data_altitude[-1]))
            zmin = float(input('Start altitude (Pa): '))
            zmax = float(input('End altitude (Pa): '))

        idx_z_min = (abs(data_altitude[:] - zmin)).argmin()
        idx_z_max = (abs(data_altitude[:] - zmax)).argmin()

        if idx_z_min > idx_z_max:
            tmp = idx_z_min
            idx_z_min = idx_z_max
            idx_z_max = tmp + 1
        else:
            idx_z_max += 1
        data_target = data_target[:, idx_z_min:idx_z_max, :, :]
    else:
        zmin = 0
        zmax = 0

    print(min(data_target), max(data_target))
    shape_data_target = data_target.shape
    data = zeros((shape_data_target))

    if data_altitude.units in ['m', 'km']:
        for alt in range(data.shape[1] - 1):
            data[:, alt, :, :] = data_target[:, alt, :, :] * (
                        data_pressure[:, alt, :, :] - data_pressure[:, alt + 1, :, :]) / 3.711  # g
        data[:, -1, :, :] = data_target[:, -1, :, :] * data_pressure[:, -1, :, :] / 3.711
    else:
        for alt in range(data.shape[1] - 1):
            data[:, alt, :, :] = data_target[:, alt, :, :] * (
                        data_altitude[idx_z_min + alt + 1] - data_altitude[idx_z_min + alt]) \
                                 / 3.711  # g
        data[:, -1, :, :] = data_target[:, -1, :, :] * data_altitude[idx_z_min + alt + 1] / 3.711

    data = correction_value(data, threshold=1e-13)
    # compute zonal mean column density
    data = sum(mean(data, axis=3), axis=1)  # Ls function of lat

    return data, altitude_limit, zmin, zmax


def convert_sols_to_ls():
    from numpy import array

    # sols to ls, step 5°ls
    time_grid_ls = array([0, 10, 20, 30, 41, 51, 61, 73, 83, 94, 105, 116, 126, 139, 150, 160, 171, 183, 193.47,
                          205, 215, 226, 236, 248, 259, 269, 279, 289, 299, 309, 317, 327, 337, 347, 355, 364,
                          371.99, 381, 390, 397, 406, 415, 422, 430, 437, 447, 457, 467, 470, 477, 485, 493, 500,
                          507, 514.76, 523, 533, 539, 547, 555, 563, 571, 580, 587, 597, 605, 613, 623, 632, 641,
                          650, 660, 669])

    return time_grid_ls


def get_ls_index(data_time):
    from numpy import array, searchsorted, max

    axis_ls = array([0, 90, 180, 270, 360])
    if max(data_time) > 361:
        # ls = 0, 90, 180, 270, 360
        idx = searchsorted(data_time[:], [0, 193.47, 371.99, 514.76, 669])
    else:
        idx = searchsorted(data_time[:], axis_ls)

    return idx, axis_ls


def ObsCoordConvert2GcmGrid(data, data_time, data_latitude):
    from numpy import zeros

    data_converted = zeros((data.shape))
    print(data_converted.shape, data.shape, data_time.shape, data_latitude.shape)

    for nbp in range(data.shape[0]):
        idx_ls = (abs(data_time - data[nbp, 0])).argmin()
        data_converted[nbp, 0] = idx_ls

        idx_lat = (abs(data_latitude - data[nbp, 1])).argmin()
        data_converted[nbp, 1] = idx_lat

    return data_converted


def mesoclouds_observed():
    from numpy import loadtxt

    directory = '/home/mathe/Documents/owncloud/observation_mesocloud/'
    filenames = ['Mesocloud_obs_CO2_CRISMlimb.txt',
                 'Mesocloud_obs_CO2_CRISMnadir.txt',
                 'Mesocloud_obs_CO2_OMEGA.txt',
                 'Mesocloud_obs_CO2_PFSeye.txt',
                 'Mesocloud_obs_CO2_PFSstats.txt',
                 'Mesocloud_obs_HRSC.txt',
                 'Mesocloud_obs_IUVS.txt',
                 'Mesocloud_obs_MAVENlimb.txt',
                 'Mesocloud_obs_SPICAM.txt',
                 'Mesocloud_obs_TES-MOC.txt',
                 'Mesocloud_obs_THEMIS.txt']

    # column:  1 = ls, 2 = lat (°N), 3 = lon (°E)
    data_CRISMlimb = loadtxt(directory + filenames[0], skiprows=1)
    data_CRISMnadir = loadtxt(directory + filenames[1], skiprows=1)
    data_OMEGA = loadtxt(directory + filenames[2], skiprows=1)
    data_PFSeye = loadtxt(directory + filenames[3], skiprows=1)
    data_PFSstats = loadtxt(directory + filenames[4], skiprows=1)
    data_HRSC = loadtxt(directory + filenames[5], skiprows=1)
    data_IUVS = loadtxt(directory + filenames[6], skiprows=1)
    data_MAVENlimb = loadtxt(directory + filenames[7], skiprows=1)
    data_SPICAM = loadtxt(directory + filenames[8], skiprows=1)
    data_TESMOC = loadtxt(directory + filenames[9], skiprows=1)
    data_THEMIS = loadtxt(directory + filenames[10], skiprows=1)

    return data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS,\
           data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS


def get_nearest_clouds_observed(data_obs, dim, data_dim, value):
    from numpy import abs

    if dim is 'latitude':

        # From the dimension, get the index(es) of the slice
        if (isinstance(value, float) is True) or (isinstance(value, int) is True):
            if value > 0:
                idx = abs(data_dim[:] - value).argmin()
            else:
                idx = abs(data_dim[:] - value).argmin() + 1
            latitude_range = data_dim[idx-1:idx+1]
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

        # Cas pour les latitudes sud
        if (latitude_range[0] < 0) and (latitude_range[-1] < 0):
            mask = (data_obs[:,1] <= latitude_range[0]) & (data_obs[:,1] >= latitude_range[-1])
        # Cas pour les latitudes nord
        elif (latitude_range[0] > 0) and (latitude_range[-1] > 0):
            mask = (data_obs[:,1] >= latitude_range[0]) & (data_obs[:,1] <= latitude_range[-1])
        # Cas pour un mélange des deux
        else:
            if latitude_range[0] > latitude_range[-1]:
                tmp = latitude_range[0]
                latitude_range[0] = latitude_range[-1]
                latitude_range[-1] = tmp
            mask = (data_obs[:,1] >= latitude_range[0]) & (data_obs[:,1] <= latitude_range[-1])

        data_ls = data_obs[:,0][mask]
        data_latitude = data_obs[:,1][mask]

    return data_ls, data_latitude


def rotate_data(*list_data, doflip):
    from numpy import flip
    list_data = list(list_data)

    for i, value in enumerate(list_data):
        list_data[i] = list_data[i].T  # get Ls on x-axis
        if doflip:
            list_data[i] = flip(list_data[i], axis=0)  # reverse to get North pole on top of the fig

    return list_data


def slice_data(data, dimension_data, value):
    idx, idx1, idx2, idx_dim = None, None, None, None

    # Select the dimension where the slice will be done
    for i in range(data.ndim):
        if data.shape[i] == dimension_data.shape[0]:
            idx_dim = dimension_data.shape[0]

    # Ensure we know what we do
    if idx_dim is None:
        print('Issue with the data and dimension')
        print('data.shape: {}'.format(data.shape))
        print('dimension.shape: {}'.format(dimension_data.shape))
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
        exit()

    if data.ndim == 1:
        if idx is not None:
            data = data[idx]
        else:
            data = data[idx1:idx2]
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
    return data, selected_idx


def get_mean_index_alti(data_altitude, value, dimension):
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

        # trop long => passé en zonal mean!
        for latitude in range(data_altitude.shape[2]):
            for ls in range(data_altitude.shape[0]):
                idx[ls] = (abs(data_altitude[ls, :, latitude] - value)).argmin()
            mean_idx[latitude] = mean(idx)
            idx[:] = 0

    return mean_idx


def create_gif(filenames):
    import numpy as np
    import imageio

    make_gif = input('Do you want create a gif (Y/n)?: ')
    if make_gif.lower() == 'y':
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

        savename = input('Enter the gif name: ')
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(savename + '.gif', images, fps=1)
    else:
        pass

    return
