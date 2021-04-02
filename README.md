# Main
---


## 1. Root
----------


### 1.1 ncplot.py
-----------------
Designed to plot NETCDF4 file from the LMD-MGCM. This python script is especially designed to CO<sub>2</sub> clouds simulation (co2clouds flag in the MGCM).

As to date, you can plot the following figures:

#### 1.1.1 The CO<sub>2</sub> ice mass mixing ratio (var=co2_ice)
This variable is a 4-D field, corresponding the mass mixing ratio of carbon dioxide ice (eqv to cloud).


view_mode | axis-x    | axis-y   | information                                                              | requirement
:-        | :-        | :-       | :-                                                                       | :-
1         | ls        | latitude | maximum in altitude and longitude, with others variables at these places | 
2         | ls        | latitude | display the zonal mean                                                   |
201       |           |          | special display for DARI report (compare to MOLA obs)                    |
4         | latitude  | #clouds  | polar cloud distribution to compare with Fig.8 of Neumann+2003           |
5         | latitude  | altitude | cloud evolution with satuco2/temperature/radius                          | satuco2, temp, riceco2
6         | ls        | altitude | mmr structure in winter polar regions at 60°N/S                          |
7         | longitude | latitude | Density column evolution in polar region, polar projection               |
8         | ls        | atitude  | TODO: h2o_ice profile with co2_ice presence                              | h2o_ice


#### 1.1.2 The CO<sub>2</sub> saturation (var=satuco2)
This variable is a 4-D field, corresponding to the saturation of carbon dioxide in the Martian atmosphere.

view_mode | axis-x    | axis-y    | information                                                                 | requirement
:-        | :-        | :-        | :-                                                                          | :-
1         | ls        | altitude  | zonal mean of saturation, for 3 latitudes, with co2 ice mmr                 | co2_ice 
2         | lt        | altitude  | [TODO] saturation in localtime                                              |
3         | longitude | altitude  | saturation with co2ice mmr in polar regions, [0:30]°ls SP, [270-300]°ls NP  |
4         | ls        | thickness | Thickness atmosphere layer in polar regions [Fig 9, Hu2019] |


#### 1.1.3 The CO<sub>2</sub> ice particle radius (var=riceco2)
This variable is a 4-D field, corresponding to the radius of carbon dioxide particle.

view_mode | axis-x    | axis-y    | information                                                          | requirement
:-        | :-        | :-        | :-                                                                   | :-
1         | ls        | µm        | mean radius at a latitude where co2_ice exists                       | co2_ice 
2         | ls        | lat       | [TODO] max radius day-night 
3         | ls        | lat       | altitude of top clouds
4         | longitude | latitude  | radius/co2ice/temp/satu in polar projection (not working)            | co2_ice, temp, satuco2
5         | ls        | latitude  | zonal mean of mean radius where co2_ice exists in the 15°N-15°S
6         | ls        | altitude  | mean radius profile along year, with global mean radius (x=alt, y=µm)
7         |           |           | radius structure


#### 1.1.4 The CCN CO2 (var=ccnNco2)
This variable is a 4-D field, corresponding to the number of condensation nuclei used to form carbon dioxide clouds.

view_mode | axis-x           | axis-y    | information                                                          | requirement
:-        | :-               | :-        | :-                                                                   | :-
1         | #.m<sup>-3</sup> | µm        | mean radius at a latitude where co2_ice exists                       | co2_ice 


#### 1.1.5 The temperature (var=temp)
This variable is a 4-D field, corresponding to the number of condensation nuclei used to form carbon dioxide clouds.

view_mode | axis-x           | axis-y    | information                                                          | requirement
:-        | :-               | :-        | :-                                                                   | :-
1         | localtime        | altitude  | delta T, Ls=120–150°, lon=0°, lat=0° [Fig 6, G-G2011]
2         | latitude         | altitude  | delta T, Ls=0–30°, LT=16 [fig 7, G-G2011]
3         | latitude         | altitude  | fig1=zonal mean at LT=16h, fig2=12h - 00h, both: Ls=0-30° [fig.8, G-G2011]
4         | longitude        | altitude  | delta T, Ls=0–30°, LT=16, lat=0° [fig 9, G-G2011]
5         | ls               | latitude  | [TBD] zonal mean for the X layer 
6         | ls               | altitude  | thermal structure in winter polar regions at 60°N/S
601       |                  |           | compare with another run
7         | xx               |xx         | [TBD] delta T zonal mean and altitude of cold pocket, compared to SPICAM data


#### 1.1.6 The water ice mass mixing ratio (var=h2o_ice)
This variable is a 4-D field, corresponding to the mass mixing ratio of water ice (eqv. clouds). This variable is the same as co2_ice (cf. 1.1.1).


#### 1.1.7 the saturation of water ice (var=saturation)
This variable is a 4-D field, corresponding to the saturation of water ice clouds.

view_mode | axis-x           | axis-y    | information                                                          | requirement
:-        | :-               | :-        | :-                                                                   | :-
1         | saturation       | altitude  | profile of saturation zonal-mean in a latitude region


### 1.2 extract_profile.py
--------------------------


## 2. Check_watercycle
----------------------


## 3. extract_n_concat
----------------------


## 4. spicam_cloud
------------------


