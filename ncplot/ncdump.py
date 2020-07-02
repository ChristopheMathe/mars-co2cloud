from numpy import isin
from math import ceil

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
    max_width=82
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

        print("|======================================================================|")
        print("|   NetCDF  information                                                |")
        print("|======================================================================|")
        print("|Dimension                    | {} | {} | {} | {} | {}".format(nc_dims[0], nc_dims[1], nc_dims[2],
                                                                            nc_dims[3], nc_dims[4]))
        print("|-----------------------------+------+----------+----------+-----------|")
        print("|Size                         | {:<4d} | {:<8d} | {:<8d} | {:<9d} | {}".format(nc_size[0], nc_size[1],
                                                                                            nc_size[2], nc_size[3],
                                                                                              nc_size[4]))
        print("|=============================+======+==========+==========+===========|")
        test = [x for x in nc_vars if x not in nc_dims]

        for var in test:
#            if var not in nc_dims: #sinon on duplique les infos de nc_dims
            a = isin(nc_size, nc_fid.variables[var].shape)
            print("|{:29}| {:<4} | {:<8} | {:<8} | {:<9} |".format(var, a[0], a[1], a[2], a[3]))
            if var is not test[-1]:
                print("|-----------------------------+------+----------+----------+-----------|")


        print("========================================================================")

    return nc_attrs, nc_dims, nc_size, nc_vars
