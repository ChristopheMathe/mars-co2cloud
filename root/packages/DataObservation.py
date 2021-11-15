from .ncdump import get_data
from numpy import loadtxt


def observation_mola(only_location=None):
    """
    dimensions:
        x = 361 ;
        y = 181 ;
    variables:
        float Latitude(y) ;
            Latitude:long_name = "Latitude" ;
            Latitude:units = "Degrees" ;
        float Ls(x) ;
            Ls:long_name = "Ls" ;
            Ls:units = "Degrees" ;
        float Altitude(y, x) ;
            Altitude:units = "km" ;
            Altitude:long_name = "Cloud top altitude above surface" ;

    // global attributes:
            :Title = "MOLA cloud top altitude above surface data (binned)" ;
    """
    path = '/home/mathe/Documents/owncloud/GCM/MOLA_cloudaltis_1x1.nc'
    mola_latitude, list_var = get_data(filename=path, target='Latitude')
    mola_ls, list_var = get_data(filename=path, target='Ls')
    mola_altitude, list_var = get_data(filename=path, target='Altitude')
    mola_altitude = mola_altitude[:, :] / 1e3  # m to km
    from numpy import max, min, append, array, nan

    if only_location:
        tmp_lat = array([])
        tmp_ls = array([])
        for i in range(mola_altitude.shape[0]):
            for j in range(mola_altitude.shape[1]):
                if mola_altitude[i, j] != mola_altitude[i, j]:
                    mola_altitude[i, j] = None
                else:
                    tmp_lat = append(tmp_lat, mola_latitude[i])
                    tmp_ls = append(tmp_ls, mola_ls[j])
        mola_ls = tmp_ls
        mola_latitude = tmp_lat
    else:
        for i in range(mola_altitude.shape[0]):
            for j in range(mola_altitude.shape[1]):
                if mola_altitude[i, j] != mola_altitude[i, j]:
                    mola_altitude[i, j] = nan
        print(max(mola_altitude), min(mola_altitude))

    return mola_latitude, mola_ls, mola_altitude


'''
========================================================================================================================
                                        TES OBSERVATIONS INFORMATION
========================================================================================================================
- TES.MappedClimatology.limb.MY(24-25-26-27).nc => temperature at 2h and 14h (ls, alt, lat, lon)
        T_limb_day
        T_limb_nit
        altitude [Pa]

- TES.MappedClimatology.nadir.MY(24-25-26-27).nc:
        tau_dust(time, latitude, longitude)              ; "Dust optical depth at 1075 cm-1"
        tau_ice(time, latitude, longitude)               ; "Water ice optical depth at 825 cm-1"
        water_vapor(time, latitude, longitude)           ; "Water vapor column" ; "precip-microns"
        Psurf_day(time, latitude, longitude)             ; "Daytime (~2 pm) surface pressure"
        Psurf_nit(time, latitude, longitude)             ; "Nighttime (~2 am) surface pressure"
        Tsurf_day(time, latitude, longitude)             ; "Daytime (~2 pm) surface temperature"
        Tsurf_nit(time, latitude, longitude)             ; "Nighttime (~2 am) surface temperature"
        T_nadir_day(time, altitude, latitude, longitude) ; "Daytime (~2 pm) atmospheric temperature"
        T_nadir_nit(time, altitude, latitude, longitude) ; "Nighttime (~2 am) atmospheric temperature"

- TES.SeasonalClimatology.nc:
        taudust(time, latitude, longitude) ; "Dust optical depth at 1075 cm-1 (scaled to a 610 Pa surface)"
        tauice(time, latitude, longitude) ; tauice:long_name = "Water ice optical depth at 825 cm-1" ;
        water(time, latitude, longitude) ; water:long_name = "Water vapor column" ; water:units = "precip-microns" ; 
        Tsurf_day(time, latitude, longitude) ; Tsurf_day:long_name = "Daytime (~2 pm) surface temperature" ;
        T_50Pa_day(time, latitude, longitude) ; T_50Pa_day:long_name = "Daytime (~2 pm) temperature at 50 Pa" ;
        Tsurf_nit(time, latitude, longitude) ; Tsurf_nit:long_name = "Nighttime (~2 pm) surface temperature" ;
        T_50Pa_nit(time, latitude, longitude) ; T_50Pa_nit:long_name = "Nighttime (~2 am) temperature at 50 Pa" ;
'''


def observation_tes(target, year=None):
    directory_tes = '/home/mathe/Documents/owncloud/GCM/TES/'

    data = None
    if year is not None:
        filename = f'TES.MappedClimatology.limb.MY{year:d}.nc'
        if target in ['T_limb_day', 'T_limb_nit']:
            data, tmp = get_data(directory_tes + filename, target=target)
        else:
            print(f'Wrong target for {filename}!')
            exit()
    else:
        filename = 'TES.SeasonalClimatology.nc'
        if target in ['time', 'latitude', 'taudust', 'tauice', 'water', 'Tsurf_day', 'T_50Pa_day', 'Tsurf_nit',
                      'T_50Pa_nit']:
            data, tmp = get_data(directory_tes + filename, target=target)
        else:
            print('Wrong target for TES.SeasonalClimatology.nc !')
            exit()

    print(f'TES file is: {directory_tes + filename}')
    return data


'''
========================================================================================================================
                                              PFS OBSERVATIONS INFORMATION                                                 
========================================================================================================================
28/10/2020: This file will be constantly updated as new PFS observations will be available and new retrievals will be
            performed.

The current release contains retrievals from Ls=331.18° of MY 26 to Ls=218.66° of MY 35  (from MEx orbit 00010 to 20793)

Estimated uncertainty for retrieved quantities:
    Atmospheric temperatures:   +/- 1 K for mid heights (10-30 km)
                                +/- 1-4 K for lower and higher heights
    Dust opacity:
    "warm regime" (Tsurf >= 220 K): +/- 0.02-0.06
    "cold regime" (Tsurf  < 220 K): +/- 0.11
    Ice opacity:
    "warm regime" (Tsurf >= 210 K): +/- 0.01
    "cold regime" (Tsurf  < 210 K): +/- 0.06

Dimensions
    0  Name: P_layer  Size: 61
    1  Name: N_Meas  Size: 1590000
 
Global Attributes
    0  Title: PFS MEX Retrievals (QF=1)
    1  Version: v0.1
    2  Mission: MARS-EXPRESS
    3  Content: PFS unbinned retrievals from Ls=331.18° of MY 26 to Ls=218.66° of MY 35 (from MEx orbit 00010 to 20793)
    4  Description: Please, read the accompanying file "PFS_Dataset_Description.txt" for a description of this dataset
    5  EULA: The End-User License Agreement (EULA) is provided in the accompanying file "PFS_Dataset_EULA.txt"
    6  Author: Marco Giuranna
    7  email: marco.giuranna@inaf.it
    8  Institution: INAF-IAPS

Variables and attributes
    0  Temperature:     FLOAT(61,1590000) = FLOAT(P_layer,N_Meas)
         0  long_name: Retrieved atmospheric temperature profiles
         1  short_name: T
         2  notes: The atmospheric temperatures are retrieved on a fix Pressure grid. NaN values are used when pressure
                   values correspond to altitudes "below" the surface
         3  units: K
    1  Pressures:       FLOAT(61) = FLOAT(P_layer)
         0  long_name: Pressure grid used in all retrievals
         1  short_name: P
         2  units: mbar
    2  Altitude:        FLOAT(61,1590000) = FLOAT(P_layer,N_Meas)
         0  long_name: Altitude profiles of retrieved atmospheric temperatures
         1  short_name: H
         2  units: km
         3  notes: From local surface -> min value = 0. Values of "-100" are used for altitudes "below" the surface
    3  orbit_number:    FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: MEx orbit Number
         1  short_name: Otbit
    4  p_surf:          FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Surface pressure
         1  short_name: Psurf
         2  units: mbar
         3  notes: From MCD v5.2
    5  T_surf:          FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved Surface Temperature
         1  short_name: Tsurf
         2  units: K
    6  h_surf:          FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Surface altitude wrt Mars aeroid
         1  short_name: Hsurf
         2  units: m
         3  notes: MOLA data uses an aeroid, or sea level measurement as a zero point, and the altitude data is
                   measured up and down from that level
    7  dust:            FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved dust opacity (column)
         1  short_name: tau_dust
         2  notes: Column dust optical depth at 1075 cm-1
    8  dust_norm:       FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved NORMALIZED dust opacity (column)
         1  short_name: tau_dust_norm
         2  notes: Column dust optical depth at 1075 cm-1, normalised to Psurf = 6.1 mbar
    9  ice:             FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved ice opacity (column)
         1  short_name: tau_ice
         2  notes: Column ice optical depth at 825 c
    10  lat:             FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Latitude
         1  short_name: Lat
         2  units: degrees
         3  valid_range:      -90.0000
         3  valid_range:       90.0000
         4  notes: [-90°, +90°]
    11  lon:             FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: East Longitude
         1  short_name: Lon
         2  units: degrees
         3  valid_range:      -180.000
         3  valid_range:       180.000
         4  notes: East Longitude, [-180°, +180°]
    12  loct:            FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Local Solar Time
         1  short_name: LST
         2  units: hours
         3  valid_range:      0.000000
         3  valid_range:       24.0000
         4  notes: Local Solar Time, [0h, 24h], at observed longitude and ephemeris epoch calculated through SPICE
                   (CSPICE_ET2LST)
    13  ls:              FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Solar Longitude
         1  short_name: Ls
         2  units: degrees
         3  valid_range:      0.000000
         3  valid_range:       360.000
         4  notes: Mars Solar Longitude (season), [0°, 360°]
    14  ls_MY:           FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Incremental Solar Longitude
         1  short_name: Ls_MY
         2  units: degrees
         3  notes: 0-360 are from MY 26; 360-360*2 are from MY 27; ... ; 360*8-360*9 are from MY 34
    15  incidence_angle: FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Solar Incidence Angle
         1  short_name: IncAng
         2  units: degree
         3  valid_range:      0.000000
         3  valid_range:       180.000
         4  notes: This is the angle between the Sun and a "normal" drawn perpendicular to the surface of the planet
    16  emission_angle:  FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Emission Angle
         1  short_name: EmAng
         2  units: degree
         3  valid_range:      0.000000
         3  valid_range:       90.0000
         4  notes: This is the angle between PFS boresight and a "normal" drawn perpendicular to surface of the planet.
                   When PFS is looking straight down ("nadir"), the emission angle is 0°
    17  SCET:            FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Unique ID to identify a specific PFS measurement
         1  short_name: SCET
    18  quality_factor:  FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Quality Factor
         1  short_name: QF
         2  notes: Quality Factor: 1 = good; 2 = warning (check manually); 3 = bad. Only retrievals with QF = 1 are
                   provided in this initial release. The full dataset including retrievals with QF =2 and QF =3 might be
                   obtained by request. Please contact the PFS PI at the following address: marco.giuranna@inaf.it
'''


def observation_pfs(target):
    data = get_data('/home/mathe/Documents/owncloud/GCM/PFS/PFS_dataset_20793/PFS_data/PFS_data.nc', target=target)

    return data


def mesospheric_clouds_observed():
    from numpy import loadtxt

    directory = '/home/mathe/Documents/owncloud/GCM/observation_mesocloud/'
    filenames = ['Mesocloud_obs_CO2_CRISMlimb.txt',
                 'Mesocloud_obs_CO2_CRISMnadir.txt',
                 'Mesocloud_obs_CO2_OMEGA.txt',
                 'Mesocloud_obs_CO2_PFSeye.txt',
                 'Mesocloud_obs_CO2_PFSstats.txt',
                 'Mesocloud_obs_HRSC.txt',
                 'Mesocloud_obs_IUVS.txt',
                 'Mesocloud_obs_MAVENlimb.txt',
                 'Mesocloud_obs_SPICAM.txt',
                 'Mesocloud_obs_TES-MOC.txt',
                 'Mesocloud_obs_THEMIS.txt',
                 'CO2clouds_altitude_localtime_files/nomad_liuzzi2021.txt']

    # column:  1 = ls, 2 = lat (°N), 3 = lon (°E)
    data_crism_limb = loadtxt(directory + filenames[0], skiprows=1)
    data_crism_nadir = loadtxt(directory + filenames[1], skiprows=1)
    data_omega = loadtxt(directory + filenames[2], skiprows=1)
    data_pfs_eye = loadtxt(directory + filenames[3], skiprows=1)
    data_pfs_stats = loadtxt(directory + filenames[4], skiprows=1)
    data_hrsc = loadtxt(directory + filenames[5], skiprows=1)
    data_iuvs = loadtxt(directory + filenames[6], skiprows=1)
    data_maven_limb = loadtxt(directory + filenames[7], skiprows=1)
    data_spicam = loadtxt(directory + filenames[8], skiprows=1)
    data_tesmoc = loadtxt(directory + filenames[9], skiprows=1)
    data_themis = loadtxt(directory + filenames[10], skiprows=1)
    data_nomad = loadtxt(directory + filenames[11], usecols=(1, 2, 3), skiprows=2, delimiter=',')

    return data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
           data_maven_limb, data_spicam, data_tesmoc, data_themis, data_nomad


def mesospheric_clouds_altitude_localtime_observed(instrument):
    from numpy import zeros
    folder = '/home/mathe/Documents/owncloud/GCM/observation_mesocloud/CO2clouds_altitude_localtime_files/'
    if instrument == 'HRSC':
        # nro2, alti2, velo2, lat2, long2, loctime2, ls2
        data = loadtxt(folder + 'altilista_HRSC.txt')
        data_lat = data[:, 3]
        data_lon = data[:, 4]
        data_lt = data[:, 5]
        data_ls = data[:, 6]
        data_alt = data[:, 1]
        data_alt_min = zeros(data_alt.shape[0])
        data_alt_max = zeros(data_alt.shape[0])
    elif instrument == 'OMEGAlimb':
        # nro4, lat4, long4, ls4, loctime4, alti4
        data = loadtxt(folder + 'altilista_OMEGAlimb.txt')
        data_lat = data[1]
        data_lon = data[2]
        data_lt = data[4]
        data_ls = data[3]
        data_alt = data[5]
        data_alt_min = 0
        data_alt_max = 0
    elif instrument == 'OMEGAnadir':
        # nro1, lat1, long1, ls1, loctime1, alti1
        data = loadtxt(folder + 'altilista_OMEGA_nadir.txt')
        data_lat = data[:, 1]
        data_lon = data[:, 2]
        data_lt = data[:, 4]
        data_ls = data[:, 3]
        data_alt = data[:, 5]
        data_alt_min = zeros(data_alt.shape[0])
        data_alt_max = zeros(data_alt.shape[0])
    elif instrument == 'SPICAM':
        # nro5, lat5, long5, ls5, loctime5, alti5
        data = loadtxt(folder + 'altilista_SPICAM_stelocc.txt')
        data_lat = data[:, 1]
        data_lon = data[:, 2]
        data_lt = data[:, 4]
        data_ls = data[:, 3]
        data_alt = data[:, 5]
        data_alt_min = zeros(data_alt.shape[0])
        data_alt_max = zeros(data_alt.shape[0])
    elif instrument == 'THEMIS':
        # nro3, ls3, lat3, long3, loctime3, inci, alti3, minalti3,maxalti3, velo3, minvelo3, maxvelo3
        data = loadtxt(folder + 'altilista_THEMIS.txt', usecols=(1, 2, 3, 4, 6))
        data_lat = data[:, 1]
        data_lon = data[:, 2]
        data_lt = data[:, 3]
        data_ls = data[:, 0]
        data_alt = data[:, 4]
        data_alt_min = zeros(data_alt.shape[0])
        data_alt_max = zeros(data_alt.shape[0])
    elif instrument == 'NOMAD':
        # UT time, LS, Lat, E Lon, Local_time, Altitude (min, highest CO2 ice, max), CO2 ice max [ppmv], Radius [um], Criteria (T=temp. profile, H=highres)
        data = loadtxt(folder + 'nomad_liuzzi2021.txt', usecols=(1, 2, 3, 4, 5, 6, 7), skiprows=2, delimiter=',')
        data_ls = data[:, 0]
        data_lat = data[:, 1]
        data_lon = data[:, 2]
        data_lt = data[:, 3]
        data_alt = data[:, 5]
        data_alt_min = data[:, 4]
        data_alt_max = data[:, 6]
    else:
        print(f'Wrong instrument: {instrument}')
        exit()

    return data_ls, data_lat, data_lon, data_lt, data_alt, data_alt_min, data_alt_max


def simulation_mvals(target, localtime):
    filename = None
    if target in ['tsurf', 'Time', 'latitude', 'longitude', 'altitude']:
        if localtime == 2:
            filename = '../simu_ref_cycle_eau_mvals/simu_ref_cycle_eau_mvals/concat_vars_3D_LT_2h_Ls.nc'
        if localtime == 14:
            filename = '../simu_ref_cycle_eau_mvals/simu_ref_cycle_eau_mvals/concat_vars_3D_LT_14h_Ls.nc'
    elif target in ['temp']:
        filename = '../simu_ref_cycle_eau_mvals/simu_ref_cycle_eau_mvals/concat_vars_4D_P_LT_14h_Ls.nc'

    print(f'\tFile name: {filename}')

    return get_data(filename=filename, target=target)


'''
    Data extracted from: https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=VL1%2FVL2-M-MET-3-P-V1.0
    Information from: https://atmos.nmsu.edu/PDS/data/vl_1001/catalog/vl_p.cat
    Other information from: https://nssdc.gsfc.nasa.gov/planetary/viking.html
    Data Set Overview                                                         
    =================                                                         
      This data set contains the martian surface atmospheric pressure         
      readings obtained through much of the duration of the Viking            
      Lander 1 and 2 missions (data are included for Viking Lander 1          
      sols 0 - 2245 and Viking Lander 2 sols 0 - 1050).  The data are         
      derived from the ambient pressure sensor carried onboard the            
      Landers and values are presented on a point by point basis.             
      The sampling rate was variable throughout the mission; the              
      maximum sampling rate was one measurement per second, but               
      ranged up to 65 minutes for Lander 1 and 105 minutes for Lander         
      2.  For further background information on the Viking                    
      Meteorology Instrument System (VMIS) and results from this              
      experiment, see [CHAMBERLAIN_ETAL1976; HESS_ETAL_1976A;                 
      HESS_ETAL_1976B; and HESS_ETAL_1977].  For discussions of               
      analyses of experiment data, see also [TILLMAN_1977;                    
      HESS_ETAL_1980; and TILLMAN_1988].  An earlier version of this          
      data set is archived at the NSSDC (NSSDC ID 75-075C-07D and             
      75-083C-07D).                                                           
                                                                              
      This data set is composed of the following fields (listed as            
      the field name followed by a description):                              
                                                                              
      SC_ID                                                                   
        Spacecraft id   /!\ Je l'ai supprime pour utiliser facilement loadtxt 
                                                                              
      SOL_LON                                                                 
        Areocentric longitude of the Sun (Ls), derived for the                
        local time of each measurement                                        
                                                                              
      VIKING_YEAR                                                             
        Viking mission year, starting at 1 when the Viking                    
        spacecraft reached Mars, and incremented at Ls = 0                    
        every martian year                                                    
                                                                              
      MARTIAN_DAY                                                             
        The martian solar day (sol), starting at day 0 when                   
        each Lander touched down                                              
                                                                              
      LOCAL_HOUR                                                              
        Local hour (Earth hour), beginning at midnight (hr 0)                 
                                                                              
      LOCAL_MINUTE                                                            
        Local minute (Earth minute)                                           
                                                                              
      LOCAL_SECOND                                                            
        Local second (Earth second)                                           
                                                                              
      PRESSURE                                                                
        Surface atmospheric pressure     
'''


def viking_lander(lander, mcd):
    from numpy import append, array, mean, where, unique

    path = ''
    if mcd:
        if lander ==1:
            path = '/home/mathe/Documents/owncloud/GCM/Viking_lander/viking_lander1_pression_mcd.dat'
        elif lander ==2:
            path = '/home/mathe/Documents/owncloud/GCM/Viking_lander/viking_lander2_pression_mcd.dat'
        else:
            print('wrong lander number')
            exit()
        data_sols_unique = loadtxt(path)[:,0]
        data_pressure_annual = loadtxt(path)[:,1]
    else:
        sols_0 = 0
        if lander == 1:
            path = '/home/mathe/Documents/owncloud/GCM/Viking_lander/viking_lander1_pression.dat'
            sols_0 = 209  # http://www-mars.lmd.jussieu.fr/mars/time/martian_time.html , ls=97°
        elif lander == 2:
            path = '/home/mathe/Documents/owncloud/GCM/Viking_lander/viking_lander2_pression.dat'
            sols_0 = 253  # http://www-mars.lmd.jussieu.fr/mars/time/martian_time.html , ls=117°
        else:
            print('wrong lander number')
            exit()
        dataset = loadtxt(path)
        data_sols = sols_0 + dataset[:, 2]
        data_pressure = dataset[:, 6] * 1e2  # mbar to Pa

        # Diurnal mean
        data_sols_diurnal = array([])
        data_pressure_diurnal = array([])
        main_cpt = 0
        while main_cpt < data_sols.shape[0]:
            cpt = 0
            while data_sols[main_cpt] == data_sols[main_cpt + cpt]:
                cpt += 1
                if main_cpt + cpt == data_sols.shape[0]:
                    break
            data_sols_diurnal = append(data_sols_diurnal, mean(data_sols[main_cpt:main_cpt + cpt]))
            data_pressure_diurnal = append(data_pressure_diurnal, mean(data_pressure[main_cpt:main_cpt + cpt]))
            main_cpt += cpt

        # Annual mean
        data_sols_diurnal = data_sols_diurnal % 669
        data_sols_unique = unique(data_sols_diurnal)
        data_pressure_annual = array([])
        for i in range(data_sols_unique.shape[0]):
            indexes = where(data_sols_diurnal == data_sols_unique[i])
            data_pressure_annual = append(data_pressure_annual, mean(data_pressure_diurnal[indexes]))

    return data_sols_unique, data_pressure_annual


def boundaries_seasonal_caps():
    '''
    Data from Hu et al. (2012) paper. Inferred from MCS data
    '''
    from numpy import arange, array
    north_ls = arange(185, 360, 5)
    north_boundaries = array([90.00, 75.51, 74.07, 81.85, 78.17, 74.53, 74.52, 71.84, 68.74, 73.28, 71.54, 71.80, 69.71,
                              71.08, 69.22, 68.02, 68.66, 74.85, 78.88, 78.46, 76.33, 76.48, 77.91, 81.00, 76.85, 75.21,
                              73.79, 74.49, 71.59, 60.55, 66.37, 59.93, 80.90, 79.34, 90.00])

    north_boundaries_error = array([0, 1.94, 1.22, 0.54, 1.35, 1.21, 0.53, 0.90, 1.34, 0.87, 0.75, 0.62, 0.98, 7.45,
                                    3.62, 5.52, 3.59, 1.89, 2.59, 2.12, 1.89, 1.02, 1.02, 0.50, 0.49, 1.41, 1.17, 0.65,
                                    3.33, 3.66, 6.88, 7.88, 1.09, 1.91, 0])

    south_ls = arange(5, 190, 5)
    south_boundaries = -1 * array([90.00, 83.31, 78.62, 76.10, 74.51, 71.22, 67.94, 66.37, 62.20, 64.20, 59.66, 60.99,
                                   59.38, 58.21, 56.07, 55.47, 55.65, 54.68, 54.44, 53.27, 53.26, 54.19, 54.60, 54.25,
                                   55.27, 56.19, 60.44, 64.06, 64.38, 68.48, 72.32, 73.62, 72.97, 75.37, 78.50,
                                   85.39, 90.00])
    south_boundaries_error = array([0, 1.27, 2.96, 3.89, 1.16, 2.97, 5.10, 5.76, 7.41, 3.90, 6.74, 2.23, 1.80, 2.20,
                                    1.10, 1.35, 1.51, 1.28, 0.77, 2.59, 2.21, 2.33, 2.43, 1.69, 1.75, 2.09, 1.59,
                                    4.72, 3.65, 2.78, 1.78, 1.90, 2.82, 2.25, 1.39, 1.29, 0])

    return north_ls, north_boundaries, north_boundaries_error, south_ls, south_boundaries, south_boundaries_error
