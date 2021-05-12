from .ncdump import get_data


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
            data = get_data(directory_tes + filename, target=target)
        else:
            print(f'Wrong target for {filename}!')
            exit()
    else:
        filename = 'TES.SeasonalClimatology.nc'
        if target in ['taudust', 'tauice', 'water', 'Tsurf_day', 'T_50Pa_day', 'Tsurf_nit', 'T_50Pa_nit']:
            data = get_data(directory_tes + filename, target=target)
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
                 'Mesocloud_obs_THEMIS.txt']

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

    return data_crism_limb, data_crism_nadir, data_omega, data_pfs_eye, data_pfs_stats, data_hrsc, data_iuvs, \
        data_maven_limb, data_spicam, data_tesmoc, data_themis


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
