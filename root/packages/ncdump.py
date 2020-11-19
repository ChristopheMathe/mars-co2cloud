from numpy import isin, delete, array, append
from math import ceil
from netCDF4 import Dataset


def getfilename(files):
    if any(".nc" in s for s in files):
        list_files = sorted([x for x in files if '.nc' in x])
        if len(list_files) > 1:
            print('Netcdf files available: (0) {}'.format(list_files[0]))
            for i, value_i in enumerate(list_files[1:]):
                print('                        ({}) {}'.format(i + 1, value_i))
            filename = int(input("Select the file number: "))
            filename = list_files[filename]
            print('')
        else:
            filename = list_files[0]
    else:
        print('There is no Netcdf file in this directory !')
        filename = ''
        exit()
    return filename


def getdata(filename, target=None):
    # name_dimension_target = data.variables[variable_target].dimensions  # get access to all dimensions names
    # dimension = array(len(name_dimension_target))

    data = Dataset(filename, "r", format="NETCDF4")
    if target is None:
        tmp, tmp, tmp, tmp, variable_target = ncextract(filename, data, verb=True)
        if variable_target is None:
            variable_target = input('Select the variable: ')  # TODO faire les tests si la variable existe
        else:
            print("Variable selected is {}".format(variable_target))
    else:
        variable_target = target

    data_target = data.variables[variable_target]
    return data_target


def ncextract(filename, nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
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
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
              a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()

    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    nc_size = [nc_fid.dimensions[x].size for x in nc_dims]
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables

    max_width = 82
    if len(filename) >= max_width:
        max_width = len(filename) + 10

    if verb:
        text= ''.rjust(max_width, '=')
        print('|{}|'.format(text, '', width=max_width))
        print("|   NetCDF Global Attributes                                           |")
        print("|======================================================================|")
        print("|File name:                                                            |")
        print("|   {} {:{fill}{width}}|".format(filename, '', fill='', width=max_width-len(filename)-1))
        for nc_attr in nc_attrs:
            print('|{}: {:{fill}{width}} |'.format(nc_attr, '', fill='', width=max_width-len(nc_attr)))

            width = len(nc_fid.getncattr(nc_attr)) - max_width - 1
            if width < 0:
                print('|   {}{:{fill}{width}}|'.format(nc_fid.getncattr(nc_attr), '', fill='',
                                                       width=max_width-len(nc_fid.getncattr(nc_attr))))
            else:
                for i in range(ceil(width/max_width)+1):
                    lenght = len(nc_fid.getncattr(nc_attr)[(max_width - 1)*i: (max_width - 1) * (1+i)])
                    print('|   {}{:{fill}{width}}|'.format(nc_fid.getncattr(nc_attr)[(max_width - 1)*i: (max_width - 1)
                                                                                      * (1+i)], '', fill='' ,
                                                           width=max_width - lenght ))
        idx = array([], dtype=int)
        for i, value_i in enumerate(nc_dims):
                if not value_i in ['Time','longitude','latitude','altitude']:
                    idx = append(idx, i)
        if idx.shape[0] != 0:
            nc_dims = delete(nc_dims, idx)
            nc_size = delete(nc_size, idx)
        nc_dims = nc_dims[::-1]
        nc_size = nc_size[::-1]
        print("|======================================================================|")
        print("|   NetCDF  information                                                |")
        print("|======================================================================|")
        print("|Dimension                    | {} | {} | {} | {} |".format(nc_dims[0], nc_dims[1], nc_dims[2],
                                                                            nc_dims[3]))
        print("|-----------------------------+------+----------+----------+-----------|")
        print("|Size                         | {:<4d} | {:<8d} | {:<8d} | {:<9d} |".format(nc_size[0], nc_size[1],
                                                                                            nc_size[2], nc_size[3]))
        print("|=============================+======+==========+==========+===========|")
        test = [x for x in nc_vars if x not in nc_dims]
        if len(test) == 2:
            test.remove('controle')
            target = test[0]
        else:
            for var in test:
    #            if var not in nc_dims: #sinon on duplique les infos de nc_dims
                a = isin(nc_size, nc_fid.variables[var].shape)
                print("|{:29}| {:<4} | {:<8} | {:<8} | {:<9} |".format(var, a[0], a[1], a[2], a[3]))
                if var is not test[-1]:
                    print("|-----------------------------+------+----------+----------+-----------|")
            print("========================================================================")
            target = None

    return nc_attrs, nc_dims, nc_size, nc_vars, target
