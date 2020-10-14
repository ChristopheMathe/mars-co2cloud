#!/bin/bash python3
from subprocess import run
from os import listdir, mkdir
from os.path import isdir

def main():
    list_var = ['Time', 'aire', 'phisinit', 'controle', 'aps', 'bps', 'ap', 'bp', # obligatoire
                'Ls', 'Sols', 'temp', 'ps', 'tsurf', 'pressure', 'rho', # basics
                'satuco2', 'co2_ice', 'co2', 'ccnqco2', 'ccnNco2', 'co2ice', 'ccnq', 'ccnN', 'precip_co2_ice',
                'h2o_vap', 'h2o_ice', 'tau', 'dustq', 'dustN', 'h2o_ice_s', 'riceco2', 'co2conservation', 'Tau3D1mic']

    list_files = listdir('.')

    list_diagfi = [name for b, name in enumerate(list_files) if 'diagfi' in name]
    list_diagfi.sort()

    directory_output = 'extraction/'

    if isdir(directory_output):
        run(["rm", "-Rf", directory_output+"/*"])
    else:
        mkdir(directory_output)


    # Extraction: ncks -v var1, var2, diagfi1.nc extract.nc
    for i, name in enumerate(list_diagfi):
        print(i, name)
        run(["ncks", "-v", ','.join(list_var), name, directory_output+"/extracted_from_"+name[:-3]+".nc"])


    # concat all files from the extraction




if '__main__' == __name__:
    main()