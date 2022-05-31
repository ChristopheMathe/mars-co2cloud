I did this repository during my work at LATMOS and LMD on the modelling of CO<sub>2</sub> mesospheric clouds in the
Martian atmosphere from October 2019 to May 2022.

The readme describe in a first part all Python scripts (how to install, how to run, what is needed). The second part 
describe the `packages` directory which contains Python routines used by latter scripts. The third part describe the
`miscellaneous` directory which contains some Fortran code or Python scripts I wrote during my work. The last part
gives information on the prerequisites to run Python scripts (excepts those on `miscellaneous`).

# 1. Scripts
This directory contains the main python script called `ncplot.pt`. It also contains secondaries python scripts used to
check: the water cycle of the simulation (`check_watercycle.py`), the convergence of basics variables over several
years simulated (`check_convergence.py`).

## 1.1 Install
On a terminal, run the following command line:  
`git clone https://github.com/ChristopheMathe/mars-co2cloud.git`

The new directory `mars-co2cloud` is created which contains all python scripts. Ensure you have the prerequisites
described in the part fourth to run these Python scripts.

On Ciclad, you can simply run the following command line:  
`module load python/meso-3.8`

You need to download observational data at the address on the CICLAD server: `/data/cmathe/observational_data/data/`. Place the directory `data/` in the same location of `ncplot.py`. Since the data are too big to be store on gitHub, we choose this solution. If you do not have access to CICLAD server, feel free to contact me at
 <christophe.mathe@obspm.fr>.

## 1.2 ncplot.py
Designed to plot netCDF file from the Mars PCM. This python script is especially designed to CO<sub>2</sub> clouds
simulation (co2clouds flag in the Mars PCM).

You can find more details about this script, and especially about figures you can done with it on the wiki: 
https://github.com/ChristopheMathe/mars-co2cloud/wiki


### 1.2.1 Run the script
On a terminal, run the script:  
`python3 ncplot.py`  

The script will ask you to select a netCDF file (if more than one file).  
    <pre><code>
    Netcdf files available: (0) concat_3D_year3_LT_all.nc
                            (1) concat_4D_year3_P_LT_all.nc
                            (2) concat_4D_year3_S_64km_riceco2_rho_ccnNco2_LT_all.nc
                            (3) concat_4D_year3_S_satuco2_LT_all.nc
                            (4) concat_convergence_year1-3_LT_all.nc
    Select the file number:
    </code></pre>

For example, I select the file `(1)`. The script open the file and list all variables availables with theirs dimension
information, as follow:  
`Select the file number: 1`

<pre><code>
  |==================================================================================|
  | NetCDF Global Attributes                                                         |
  |==================================================================================|
  | File name:                                                                       |
  | occigen_test_64x48x32_start_mph2ocn_year7_meteoflux_clim_correction/concat_4D_ye |
  | ar3_P_LT_all.nc                                                                  |
  |==================================================================================|
  | NetCDF  information                                                              |
  |==================================================================================|
  | Dimension                     | Time       | altitude   | longitude  | latitude  |
  |-------------------------------+------------+------------+------------+-----------|
  | Size                          | 8028       | 32         | 65         | 49        |
  |===============================+============+============+============+===========|
  |ps                             | 1          | 0          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |zareoid                        | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |co2_ice                        | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |temp                           | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |satuco2                        | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |riceco2                        | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |ccnNco2                        | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |rho                            | 1          | 1          | 1          | 1         |
  |-------------------------------+------------+------------+------------+-----------|
  |h2o_ice                        | 1          | 1          | 1          | 1         |
  |==================================================================================|
  Select the variable:
</code></pre>

Then, the script asks you if you want to extract at a local time or not. Here for example, there are 12 local times:
<pre><code>
  Local time available: [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22.]
  Do you want extract at a local time (y/N)?
</code></pre>

Finally, the script asks you what plot you want to do. Here an example for`co2_ice` variable:
<pre><code>
  Selection of display mode
  What do you wanna do?
       1: maximum in altitude and longitude, with others variables at these places (fig: lat-ls)
       2: zonal mean column density (fig: lat-ls)
          201: special features for DARI report (compare to MOLA obs)
          202: adapted for Anni paper
       3: co2_ice coverage (fig: lat-lon)
       4: polar cloud distribution to compare with Fig.8 of Neumann+2003 (fig: #clouds-lat)
       5: cloud evolution with satuco2/temperature/radius/CCNCO2/h2o_ice for each time, lat[-15:15]°N (fig: alt-lat)
       6: mmr structure at a given latitude (fig: alt-ls)
       7: Density column evolution in polar region, polar projection (fig: lon-lat)
       9: localtime co2_ice column density, zonal mean, [XX-YY]°N (fig: hl-ls)
         901: localtime co2_ice at 0.5 Pa, 0°N, zonal mean (hl-ls)
      10: co2_ice mmr/column density along longitude and localtime (fig: hl-lon)
         101: co2_ice mmr at 0.5 Pa and 0°N (fig: hl-lon)
         102: co2_ice column density between 15°S and 15°N, above 10 Pa (mean) (fig: hl-lon)
       11: co2_ice mmr/column density along longitude and solar longitude (fig: ls-lon)
         111: co2_ice mmr at 0.5 Pa and 0°N (fig: hl-lon)
         112: co2_ice column density between 15°S and 15°N, above 10 Pa (mean) (fig: hl-lon)
      12: co2_ice structure along longitude at 0°N (year mean) (fig: alt-lon)
      13: stationary wave, at 0°N and -45°E (fig: alt-ls)
      14: Polar plot every 30° ls mean, column density, lat=60°-90° (fig: lat-ls)

  Select number:
</code></pre>

Once you selected a `display mode`, the script will processed data and then display the results. In some display modes,
the script can ask to the user some complementarities information like which latitude you want or do we perform the
mean over the year instead of binning.

The script saves the figure on the local directory, and saves figures data on a netCDF4 file in the new local folder 
`figure_data/`

### 1.2.2 Best practices
I recommand to you to make a symbol link of these python scripts in the directory where your data are located.

The `ncplot.py` script takes arguments on the command line:  
`python3 ncplot.py file_id variable display mode`  
For example:  
`python3 ncplot.py 1 co2_ice 202`  


## 1.3 check_watercycle.py
This python script plots and compares with TES data and/or M. Vals simulation (who worked on H2O cycle on Mars) the
following variables: atmospheric water ice, water vapor, water ice at the surface, surface pressure, surface
temperature, and also CO<sub>2</sub> ice at the surface.

Need 3D-variables from Mars PCM outputs:
* h2o_ice_s
* watercap
* mtot
* icetot
* co2ice
* tsurf
* ps

Results:
* check_water_cycle_relative_error_co2ice.png
* check_water_cycle_relative_error_h2o_ice_s.png
* check_water_cycle_relative_error_icetot.png
* check_water_cycle_relative_error_mtot.png
* check_water_cycle_relative_error_ps.png
* check_water_cycle_relative_error_tauTES.png
* check_water_cycle_relative_error_tsurf.png
* check_water_cycle_tes_mvals_me_h2o_vap.png
* check_water_cycle_tes_mvals_me_tauice.png
* check_water_cycle_tes_mvals_me_tsurf_day.png



## 1.4 check_convergence.py
This Python script plots for each year the global mean of basics variables: surface temperature, surface pressure, 
CO<sub>2</sub> ice at the surface, H2O ice at the surface, the total amount of water ice in the atmosphere, and the
total amount of water vapor in the atmosphere. The script also plots the total amount of water along the year for
each year, in order to see the water conservation.

This script works with several input files as long as all variables are present.


Need 3D-variables from Mars PCM outputs:
* ps
* tsurf
* co2ice
* h2o_ice_s
* watercap
* mtot
* icetot

Results:
* check_convergence_global_mean.png
* check_converge_total_mass_h2o.png



## 1.5 extract_profile.py
This Python scrips is used to extract profiles needed to run the Mars PCM in 1-D. Your input netCDF4 file must contains
 following 4-D variables:
* temp
* co2
* co2_ice
* h2o_vap
* h2o_ice
* ccnq
* ccnN
* dustq
* dustN
* ccnqco2
* ccnNco2
* ccnqco2_h2o_m_ice  # if co2useh2o option was True in the Mars PCM simulation
* ccnqco2_h2o_m_ccn  # if co2useh2o option was True in the Mars PCM simulation
* ccnNco2_h2o        # if co2useh2o option was True in the Mars PCM simulation
* ccnqco2_meteor     # if meteo_flux option was True in the Mars PCM simulation 
* ccnNco2_meteor     # if meteo_flux option was True in the Mars PCM simulation



# 2. Packages
In this directory, all files are used as modules. 

* constant_parameter.py
* create_infofile.py
* data_processing.py
* displays.py
* lib_functions.py
* lib_observation.py (this file list all observational data available. Need the `data/` directory)
* ncdump.py (this file open the netCDF file and extract all information needed)


# 3. Miscellaneous


# 4. Prerequisites
The script was developed using:
* Python,     version = 3.8
* netCDF,     version = 1.5.3
* scipy,      version = 
* numpy,      version = 1.19.2
* matplotlib, version = 3.4.0
* cartopy,    version = 0.18.0

