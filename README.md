# Main
---


## 1. Root
----------


### 1.1 ncplot.py
-----------------
Designed to plot NETCDF4 file from the LMD-MGCM.
As to date, you can plot the following figures:

#### 1.1.1 The CO<sub>2</sub> ice mass mixing ratio (vars=co2_ice)
This variable is a 4-D field.

view_mode | axis-x / axis-y | information  | requirement
:-        | :-              | :-           | :-
1         | ls / latitude   |   To delete! | 
2         | ls / latitude   | display the zonal mean
201       |                 | special display for DARI report (compare to MOLA obs)
3         | xxx             | xxxx
4         | xxxx            | xxx
5         | xxx             | xxx
6         | thickness / latitude | layer ice thickness in polar region
7         | latitude / #clouds | polar cloud distribution to compare with Fig.8 of Neumann+2003
8         | latitude / altitude | cloud evolution with satuco2/temperature/radius | satuco2, temp, riceco2
9         | ls / altitude | mmr structure in winter polar regions at 60°N/S
10        | longitude / latitude | Density column evolution in polar region, polar projection


#### 1.1.2. The CO<sub>2</sub> saturation


### 1.2 extract_profile.py
--------------------------


## 2. Check_watercycle
----------------------


## 3. extract_n_concat
----------------------


## 4. spicam_cloud
------------------


