from sys import exit


# correct very low values of co2 mmr
def correction_value_co2_ice(data):
    from numpy import where
    for i in range(data.shape[0]):
        dim2, dim3, dim4 = where(data[i, :, :, :] < 1e-10)
        data[i, dim2, dim3, dim4] = 0

    return data


def get_max_co2_ice_in_alt_lon(data):
    from numpy import swapaxes, unravel_index, asarray, reshape
    # get max value along altitude and longitude
    # max_mmr = amax(data_y, axis=(1,3)) # get the max mmr value in longitude/altitude

    B = swapaxes(data, 1, 2)
    max_idx = B.reshape((B.shape[0], B.shape[1], -1)).argmax(2)
    x, y = unravel_index(max_idx, B[0, 0, :].shape)
    data_max = [B[i, j, x[i, j], y[i, j]] for i in range(B.shape[0]) for j in range(B.shape[1])]
    data_max = asarray(data_max)
    data_max = reshape(data_max, (data.shape[0], data.shape[2]))

    return data_max, x, y


def extract_at_max_co2_ice(data, x, y, shape_big_data=None):
    from numpy import asarray, reshape, swapaxes

    # 1-D
    if data.ndim == 1:
        data_max = [data[x[i, j]] for i in range(shape_big_data[0]) for j in range(shape_big_data[1])]
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

    return data_max


def reshape_and_linearize_data(data, data_time, data_latitude, interp_time):
    from numpy import flip, arange
    from scipy.interpolate import interp2d

    # reshape data
    data = data.T  # lat function of Ls
    data = flip(data, axis=0)  # reverse to get North pole on top of the fig

    # interpolation to get linear Ls
    f = interp2d(x=arange(data_time.shape[0]), y=arange(data_latitude.shape[0]), z=data, kind='linear')

    data = f(interp_time, arange(len(data_latitude)))

    return data

def linear_grid_ls(data):
    from numpy import linspace, searchsorted

    axis_ls = linspace(0, 360, data.shape[0])
    ndx = searchsorted(axis_ls, [0, 90, 180, 270, 360])
    #    ndx = searchsorted(axis_ls, [0, data_time[len(data_time[:])/2], data_time[-1]])
    interp_time = searchsorted(data, axis_ls)

    return interp_time, axis_ls, ndx
