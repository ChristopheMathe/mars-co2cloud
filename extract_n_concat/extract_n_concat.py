#!/bin/bash python3
import subprocess
from os import listdir


def main():
    list_var = ['Time', 'aire', 'phisinit', 'controle', 'aps', 'bps', 'ap', 'bp']  # obligatoire
#                'Ls', 'Sols', 'temp', 'ps', 'tsurf', 'pressure',# basics
 #               'satuco2', 'co2_ice', 'co2', 'ccnqco2', 'ccnNco2', 'co2_ice', 'ccnq', 'ccnN', 'precip_co2_ice',
  #              'h2o_vap', 'h2o_ice', 'tau', 'dustq', 'dustN', 'h2o_ice_s', 'co2conservation'] # riceco2

    list_files = listdir('.')
    list_diagfi = [name for b, name in enumerate(list_files) if 'diagfi' in name]
    print(list_diagfi)

    list_diagfi.sort()
    # Extraction: ncks -v var1, var2, diagfi1.nc extract.nc
    for i, name in enumerate(list_diagfi):
        print(i, name)
        subprocess.run(["ncks", "-v", ','.join(list_var), name, "extraction/extracted_from_"+name[:-3]+".nc"])


    # concat all files from the extraction




if '__main__' == __name__:
    main()
