#!/bin/bash python3
import subprocess

def main():
    list_var = ['Time', 'aire', 'phisinit', 'controle', 'aps', 'bps', 'ap', 'bp',  # obligatoire
                'Ls', 'Sols', 'temp', 'ps', 'tsurf', 'pressure',# basics
                'satuco2', 'co2_ice', 'co2', 'ccnqco2', 'ccnNco2', 'co2_ice', 'ccnq', 'ccnN', 'precip_co2_ice',
                'h2o_vap', 'h2o_ice', 'tau', 'dustq', 'dustN', 'h2o_ice_s', 'co2_conservation'] # riceco2

    # Extraction: ncks -v var1, var2, diagfi1.nc extract.nc
    subprocess.run(["ncks", "-v", ','.join(list_var), "diagfi1.nc", "extraction/extracted_from_diagfi1.nc"])


    # concat all files from the extraction




if '__main__' == __name__:
    main()
