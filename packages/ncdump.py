from numpy import isin, delete, array, append, int
from math import ceil
from netCDF4 import Dataset
from os import listdir


def get_argument(*argv, info_netcdf):
    arg_file = None
    arg_target = None
    arg_view_mode = None

    if len(argv) > 2:
        arg_file = int(argv[1])
        arg_target = argv[2]
        if len(argv) == 4:
            arg_view_mode = int(argv[3])

    files = listdir('.')
    try:
        directory_store = [x for x in files if 'occigen' in x or 'simu' in x][0] + '/'
    except IndexError:
        directory_store = None

    if directory_store is None:
        directory_store = ''
    else:
        files = listdir(directory_store)

    filename = getfilename(files, selection=arg_file)
    filename = directory_store + filename
    info_netcdf.filename = filename

    data_target, list_var = get_data(filename, target=arg_target)
    if data_target.ndim == 4:
        info_netcdf.data_target = data_target[:, :, :, :]
    elif data_target.ndim == 3:
        info_netcdf.data_target = data_target[:, :, :]
    elif data_target.ndim == 2:
        info_netcdf.data_target = data_target[:, :]
    elif data_target.ndim == 1:
        info_netcdf.data_target = data_target[:]
    else:
        print('Wrong dimension!')
        exit()

    info_netcdf.target_name = data_target.name

    print(f'You have selected the variable: {data_target.name}')
    dim_3d = False

    info_netcdf.idx_dim.time = data_target.dimensions.index('Time')
    try:
        info_netcdf.idx_dim.altitude = data_target.dimensions.index('altitude')
    except ValueError:
        dim_3d = True

    info_netcdf.idx_dim.latitude = data_target.dimensions.index('latitude')
    info_netcdf.idx_dim.longitude = data_target.dimensions.index('longitude')

    info_netcdf.data_dim.time, list_var = get_data(filename, target='Time')
    if not dim_3d:
        info_netcdf.data_dim.altitude, list_var = get_data(filename, target='altitude')
    info_netcdf.data_dim.latitude, list_var = get_data(filename, target='latitude')
    info_netcdf.data_dim.longitude, list_var = get_data(filename, target='longitude')

    return files, directory_store, arg_view_mode


def getfilename(files, selection=None):
    if any(".nc" in s for s in files):
        list_files = sorted([x for x in files if '.nc' in x])
        if len(list_files) > 1:
            if selection is None:
                print(f'Netcdf files available: \t(0) {list_files[0]}')
                for i, value_i in enumerate(list_files[1:]):
                    print(f'\t\t\t\t({i + 1}) {value_i}')
                filename = int(input("Select the file number: "))
            else:
                filename = selection
            filename = list_files[filename]
            print('')
        else:
            filename = list_files[0]
    else:
        print('There is no netCDF file in this directory !')
        filename = ''
        exit()
    return filename


def get_data(filename, target=None):
    data = Dataset(filename, "r", format="NETCDF4")
    if target is None:
        list_var = nc_extract(filename, data, verb=True)
        variable_target = input('Select the variable: ')  # TODO check if variable exists
    else:
        list_var = nc_extract(filename, data, verb=False)
        variable_target = target

    data_target = data.variables[variable_target]
    return data_target, list_var


def nc_extract(filename, nc_fid, verb=True):
    """
    ncdump outputs dimensions, variables and their attribute information.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
        :param verb:
        :param nc_fid:
        :param filename:
    """

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()

    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    nc_size = [nc_fid.dimensions[x].size for x in nc_dims]
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables

    max_width = 82
    if verb:
        print(f'|{"":{"="}<{max_width}}|')
        print(f'|{" NetCDF Global Attributes":<{max_width}}|')
        print(f'|{"":{"="}<{max_width}}|')
        print(f'|{" File name:":<{max_width}}|')
        if len(filename) >= max_width:
            n_line = ceil(len(filename)/max_width)
            for i in range(n_line):
                print(f'|{" "+filename[i*(max_width-2):(i+1)*(max_width-2)]:<{max_width}}|')
        else:
            print(f'|{" "+filename:<{max_width}}|')

        for nc_attr in nc_attrs:
            print(f'|{nc_attr:{""}<{max_width}}|')
            width = len(nc_fid.getncattr(nc_attr)) - max_width - 1
            if width < 0:
                print(f'|   {nc_fid.getncattr(nc_attr):{""}<{max_width}} |')
            else:
                for i in range(ceil(width / max_width) + 1):
                    name = nc_fid.getncattr(nc_attr)[(max_width - 1) * i: (max_width - 1) * (1 + i)]
                    print(f'|   {name:{""}<{width}}|')

        idx = array([], dtype=int)
        for i, value_i in enumerate(nc_dims):
            if value_i not in ['Time', 'longitude', 'latitude', 'altitude']:
                idx = append(idx, i)
        if idx.shape[0] != 0:
            nc_dims = delete(nc_dims, idx)
            nc_size = delete(nc_size, idx)
        nc_dims = nc_dims[::-1]
        nc_size = nc_size[::-1]
        print(f'|{"":{"="}<{max_width}}|')
        print(f'|{" NetCDF  information":<{max_width}}|')
        print(f'|{"":{"="}<{max_width}}|')
        print(f'|{" Dimension":{30}s} | {nc_dims[0]:<10s} | {nc_dims[1]:<10s} | {nc_dims[2]:<10s} | {nc_dims[3]:<10s}|')
        print(f'|{"":{"-"}<{31}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{11}}|')
        print(f'|{" Size":30s} | {nc_size[0]:<10d} | {nc_size[1]:<10d} | {nc_size[2]:<10d} | {nc_size[3]:<10d}|')
        print(f'|{"":{"="}<{31}}+{"":{"="}<{12}}+{"":{"="}<{12}}+{"":{"="}<{12}}+{"":{"="}<{11}}|')
        test = [x for x in nc_vars if x not in nc_dims]
        test.remove('aps')
        test.remove('bps')
        if 'controle' in test:
            test.remove('controle')
        if 'phisinit' in test:
            test.remove('phisinit')
        for var in test:
            #  if var not in nc_dims: otherwise we duplicate information of nc_dims
            a = isin(nc_size, nc_fid.variables[var].shape)
            print(f'|{var:30s} | {a[0]:<10d} | {a[1]:<10d} | {a[2]:<10d} | {a[3]:<10d}|')
            if var is not test[-1]:
                print(f'|{"":{"-"}<{31}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{11}}|')
        print(f'|{"":{"="}<{max_width}}|')

    return nc_vars
