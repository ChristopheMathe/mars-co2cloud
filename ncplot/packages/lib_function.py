from sys import exit


# correct very low values of co2 mmr
def correction_value_co2_ice(data):
    from numpy import where
    ndim = data.ndim

    if ndim == 1:
        dim1 = where(data[:] <= 1e-10)
        data[dim1] = 0

    elif ndim == 2:
        for i in range(data.shape[0]):
            dim2 = where(data[i, :] <= 1e-10)
            data[i, dim2] = 0

    elif ndim == 3:
        for i in range(data.shape[0]):
            dim2, dim3 = where(data[i, :, :] <= 1e-10)
            data[i, dim2, dim3] = 0

    elif ndim == 4:
        for i in range(data.shape[0]):
            dim2, dim3, dim4 = where(data[i, :, :, :] < 1e-10)
            data[i, dim2, dim3, dim4] = 1e-11

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

    data_max = data_max.T  # lat function of Ls
    data_max = flip(data_max, axis=0)  # reverse to get North pole on top of the fig
    x, y = x.T, y.T
    x, y = flip(x, axis=0), flip(y, axis=0)

    return data_max, x, y


def extract_at_max_co2_ice(data, x, y, shape_big_data=None):
    from numpy import asarray, reshape, swapaxes, flip

    # 1-D
    if data.ndim == 1:
        data_max = [data[x[i, j]] for i in range(shape_big_data[0]) for j in range(shape_big_data[2])]
        data_max = asarray(data_max)
        print(data_max.shape, shape_big_data[0], shape_big_data[2])
        data_max = reshape(data_max, (shape_big_data[0], shape_big_data[2]))
    # 4-D
    elif data.ndim == 4:
        data_max = swapaxes(data, 1, 2)
        data_max = [data_max[i, j, x[i, j], y[i, j]] for i in range(data_max.shape[0]) for j in range(data_max.shape[1])]
        data_max = asarray(data_max)
        data_max = reshape(data_max, (data.shape[0], data.shape[2]))
    else:
        print('Wrong dimension or taken into account')
        data_max = 0
        exit()

    data_max = data_max.T  # lat function of Ls
    data_max = flip(data_max, axis=0)  # reverse to get North pole on top of the fig

    return data_max


def linearize_ls(data, dim_time, dim_latitude, interp_time):
    from numpy import flip, arange
    from scipy.interpolate import interp2d

        # interpolation to get linear Ls
    f = interp2d(x=arange(dim_time), y=arange(dim_latitude), z=data, kind='linear')

    data = f(interp_time, arange(dim_latitude))

    return data


def linear_grid_ls(data):
    from numpy import linspace, searchsorted

    axis_ls = linspace(0, 360, data.shape[0])
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    #    ndx = searchsorted(axis_ls, [0, data_time[len(data_time[:])/2], data_time[-1]])
    interp_time = searchsorted(data, axis_ls)

    return interp_time, axis_ls, ndx


def tcondco2(data_pressure, idx_ls=None, idx_lat=None, idx_lon=None):
    from numpy import log10

    A = 6.81228
    B = 1301.679
    C = -3.494
    if (idx_ls is not None) and (idx_lat is not None) and (idx_lon is not None):
        T_sat = B / (A - log10((data_pressure[idx_ls, :, idx_lat, idx_lon] + 0.0001) / 10 ** 5)) - C
    else :
        T_sat = B / (A - log10((data_pressure[:, :, :, :] + 0.0001) / 10 ** 5)) - C

    return T_sat
