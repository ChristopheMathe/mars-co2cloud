from numpy import isin, delete, array, append
from math import ceil
from netCDF4 import Dataset


def getfilename(files, selection=None):
    if any(".nc" in s for s in files):
        list_files = sorted([x for x in files if '.nc' in x])
        if len(list_files) > 1:
            print(f'Netcdf files available: \t(0) {list_files[0]}')
            for i, value_i in enumerate(list_files[1:]):
                print(f'\t\t\t\t({i + 1}) {value_i}')
            if selection is None:
                filename = int(input("Select the file number: "))
            else:
                filename = selection
            filename = list_files[filename]
            print('')
        else:
            filename = list_files[0]
    else:
        print('There is no Netcdf file in this directory !')
        filename = ''
        exit()
    print(f'You have selected the file: {filename}')
    return filename


def get_data(filename, target=None):
    # name_dimension_target = data.variables[variable_target].dimensions  # get access to all dimensions names
    # dimension = array(len(name_dimension_target))

    data = Dataset(filename, "r", format="NETCDF4")
    if target is None:
        tmp, tmp, tmp, list_var, variable_target = nc_extract(filename, data, verb=True)
        if variable_target is None:
            variable_target = input('Select the variable: ')  # TODO check if variable exists
    else:
        tmp, tmp, tmp, list_var, tmp = nc_extract(filename, data, verb=False)
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
        if len(test) == 2:
            target = test[0]
        else:
            for var in test:
                #  if var not in nc_dims: otherwise we duplicate information of nc_dims
                a = isin(nc_size, nc_fid.variables[var].shape)
                print(f'|{var:30s} | {a[0]:<10d} | {a[1]:<10d} | {a[2]:<10d} | {a[3]:<10d}|')
                if var is not test[-1]:
                    print(f'|{"":{"-"}<{31}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{12}}+{"":{"-"}<{11}}|')
            print(f'|{"":{"="}<{max_width}}|')
            target = None
    else:
        target = None

    return nc_attrs, nc_dims, nc_size, nc_vars, target
