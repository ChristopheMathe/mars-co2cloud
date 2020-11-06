from .ncdump import getdata

'''
========================================================================================================================
                                        TES OBSERVATIONS INFORMATION
========================================================================================================================
avec TES.MappedClimatology.limb.MY(24-25-26-27).nc => on a accès à la température à 2h et 14h (ls, alt, lat, lon)
        T_limb_day
        T_limb_nit
        altitude [Pa]

avec TES.MappedClimatology.nadir.MY(24-25-26-27).nc => on a accès à :
        tau_dust(time, latitude, longitude)              ; "Dust optical depth at 1075 cm-1"
        tau_ice(time, latitude, longitude)               ; "Water ice optical depth at 825 cm-1"
        water_vapor(time, latitude, longitude)           ; "Water vapor column" ; "precip-microns"
        Psurf_day(time, latitude, longitude)             ; "Daytime (~2 pm) surface pressure"
        Psurf_nit(time, latitude, longitude)             ; "Nighttime (~2 am) surface pressure"
        Tsurf_day(time, latitude, longitude)             ; "Daytime (~2 pm) surface temperature"
        Tsurf_nit(time, latitude, longitude)             ; "Nighttime (~2 am) surface temperature"
        T_nadir_day(time, altitude, latitude, longitude) ; "Daytime (~2 pm) atmospheric temperature"
        T_nadir_nit(time, altitude, latitude, longitude) ; "Nighttime (~2 am) atmospheric temperature"

avec TES.SeasonalClimatology.nc => on a accès à:
        taudust(time, latitude, longitude) ; "Dust optical depth at 1075 cm-1 (scaled to a 610 Pa surface)"
        tauice(time, latitude, longitude) ; tauice:long_name = "Water ice optical depth at 825 cm-1" ;
        water(time, latitude, longitude) ; water:long_name = "Water vapor column" ; water:units = "precip-microns" ; 
        Tsurf_day(time, latitude, longitude) ; Tsurf_day:long_name = "Daytime (~2 pm) surface temperature" ;
        T_50Pa_day(time, latitude, longitude) ; T_50Pa_day:long_name = "Daytime (~2 pm) temperature at 50 Pa" ;
        Tsurf_nit(time, latitude, longitude) ; Tsurf_nit:long_name = "Nighttime (~2 pm) surface temperature" ;
        T_50Pa_nit(time, latitude, longitude) ; T_50Pa_nit:long_name = "Nighttime (~2 am) temperature at 50 Pa" ;
'''


def TES(target, year=None):
    directory_tes = '/home/mathe/Documents/owncloud/GCM/TES/'

    if year is not None:
        filename = 'TES.MappedClimatology.limb.MY{:d}.nc'.format(year)
        try:
            data = getdata(directory_tes + filename, target=target)
        except:
            print('Wrong target for {} !'.format(filename))
            exit()
    else:
        try:
            data = getdata(directory_tes + 'TES.SeasonalClimatology.nc', target=target)
        except:
            print('Wrong target for TES.SeasonalClimatology.nc !')
            exit()

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
         2  notes: Coulmn dust optical depth at 1075 cm-1
    8  dust_norm:       FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved NORMALIZED dust opacity (column)
         1  short_name: tau_dust_norm
         2  notes: Coulmn dust optical depth at 1075 cm-1, normalised to Psurf = 6.1 mbar
    9  ice:             FLOAT(1590000) = FLOAT(N_Meas)
         0  long_name: Retrieved ice opacity (column)
         1  short_name: tau_ice
         2  notes: Coulmn ice optical depth at 825 c
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


def PFS(target):
    from numpy import arange

    try:
        data = getdata('/home/mathe/Documents/owncloud/GCM/PFS/PFS_dataset_20793/PFS_data/PFS_data.nc',
                       target=target)
    except:
        print('Wrong target for PFS_data.nc !')
        exit()

    return data


def mesoclouds_observed():
    from numpy import loadtxt

    directory = '/home/mathe/Documents/owncloud/observation_mesocloud/'
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
    data_CRISMlimb = loadtxt(directory + filenames[0], skiprows=1)
    data_CRISMnadir = loadtxt(directory + filenames[1], skiprows=1)
    data_OMEGA = loadtxt(directory + filenames[2], skiprows=1)
    data_PFSeye = loadtxt(directory + filenames[3], skiprows=1)
    data_PFSstats = loadtxt(directory + filenames[4], skiprows=1)
    data_HRSC = loadtxt(directory + filenames[5], skiprows=1)
    data_IUVS = loadtxt(directory + filenames[6], skiprows=1)
    data_MAVENlimb = loadtxt(directory + filenames[7], skiprows=1)
    data_SPICAM = loadtxt(directory + filenames[8], skiprows=1)
    data_TESMOC = loadtxt(directory + filenames[9], skiprows=1)
    data_THEMIS = loadtxt(directory + filenames[10], skiprows=1)

    return data_CRISMlimb, data_CRISMnadir, data_OMEGA, data_PFSeye, data_PFSstats, data_HRSC, data_IUVS, \
           data_MAVENlimb, data_SPICAM, data_TESMOC, data_THEMIS
