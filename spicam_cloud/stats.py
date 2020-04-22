#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  stats.py
# -------------------
#  Author: Christophe Mathé
# -------------------
# Note:
#    * CO2 profile are easier to extract than temperature profile
#             => less temperature profile
#    * According to Loïc Verdier, we should use the CO2 extinction from
#      temperature file:
#        '''
#           La différence fondamentale entre les deux types de profils,
#           c’est que pour « CO2_LocalDensities », les couches
#           (discrètes, qui dépendent des points de mesure lors de
#           l’occultation) de l’atmosphère sont supposées avoir la même
#           densité alors que pour « CO2_temperature »  la densité
#           évolue dans la couche et cette évolution dépend de la
#           température, qui reste constante pour une couche donnée
#           (hypothèse d’iso-température).
#           La deuxième hypothèse est plus proche de la réalité que
#           la première, c’est pourquoi je te conseille plutôt
#           d’utiliser « CO2_temperature ».
#        '''
# -------------------
# Obs-descriptions-Delivered.txt:
#  col1 = name_orbite
#  col2 = longitude (tangent point), degree W ? (ask Anni, Franck)
#  col3 = latitude
#  col4 = solar local time
#  col5 = solar longitude
#  col6 = distance between satellite and (Mars? Surface? tangent point?)
# -------------------
# Temperature files
# ----
#  htfit:    altitude (km), corrected by an ellipsoïde of Mars and
#            aeroid.topography (MOLA model)
#
#  tempalt:  altitude (km) between two htift altitudes
#
#  temp:     temperature (K) at tempalt altitude
#
#  err_temp: 1-sigma temperature error (K), sometimes the error is not
#            calculated associated to hardly retrievable profile
#
#  press:    pression (Pa), deduced from the temperature and the
#            hydrostatic equilibrium
#
#  ext_co2:  CO2 extinction (molecules/m3) retrieved simultaneously with
#            the temperature
# -------------------
# CO2_LocaDensities files
# ----
# htfit:   altitude (km), corrected by an ellipsoïde of Mars and
#          aeroid.topography (MOLA model)
#
# ext_o3:  O3 extinction (molecules/m3) retrieved
#
# ext_co2: CO2 extinction (molecules/m3) retrieved
#
# ppm:     O3 parts per million
#
# err_o3:  1-sigma O3 extinction error
#
# err_co2: 1-sigma CO2 extinction error
#
# err_ppm: 1-sigma ppm error
# -------------------
# Aerosols_LocalDensities_250nm files
# ----
# htfit:   altitude (km), corrected by an ellipsoïde of Mars and
#          aeroid.topography (MOLA model)
# ext_aer:
# err_aer:
# -------------------
# Libraries
# -------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.io import readsav
from matplotlib.lines import Line2D

'''
    -------------------
    Class: cloud
    -------------------
'''

class Observation:
    def __init__(self, name=None, latitude=None, longitude=None, solarlon=None, localtime=None, coldpocket=None):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.solarlon = solarlon
        self.localtime = localtime
        self.coldpocket = []

    def add_coldpocket(self, Coldpocket):
        self.coldpocket.append(Coldpocket)


class Coldpocket(Observation):
    def __init__(self, number, altitude_bot=None, altitude_top=None, dT=None):
        self.number = number
        self.index_bot = altitude_bot
        self.index_top = altitude_top
        self.dT = dT

'''
    -------------------
    Read file 'Obs-descriptions-Delivered.txt'
    -------------------
'''

def ReadInfObs(filename):
    with open(filename, 'r') as fin:
        data = fin.readlines()
    nb_line = len(data)
    tab_name_obs = np.chararray(nb_line, itemsize=9)
    tab_longitude = np.zeros(nb_line)
    tab_latitude = np.zeros(nb_line)
    tab_localtime = np.zeros(nb_line)
    tab_solarlon = np.zeros(nb_line)
    tab_distance = np.zeros(nb_line)
    for l, line in enumerate(data):
        items = line.split()
        tab_name_obs[l] = str(items[0])
        tab_longitude[l] = float(items[1])
        tab_latitude[l] = float(items[2])
        tab_localtime[l] = float(items[3])
        tab_solarlon[l] = float(items[4])
        tab_distance[l] = float(items[5])

    return tab_name_obs, tab_longitude, tab_latitude, tab_localtime, \
           tab_solarlon, tab_distance

'''
    -------------------
    Compute the CO2 saturation profile
    -------------------
        Note:
            Antoine equation: log10(P_sat(bar)) = A - B/(T+C)
            Based on NIST database:
            Temperature range validity: 154.26 - 195.89 (K)
'''

def Co2Saturation(pression):
    A = 6.81228
    B = 1301.679
    C = -3.494
    T_sat = B / (A - np.log10((pression + 0.0001) / 10 ** 5)) - C  # +0.0001 to avoid log10(0)
    return T_sat

'''
    -------------------
    Open IDL file
    -------------------
'''
def open_idlfile(filename):

    # try to open filename
    try:
        data = readsav(filename)
        # if not, exit this orbit
    except:
        data = False

    return data


'''
    -------------------
    Main
    -------------------
'''

def main(directory_profT, directory_profq_200, directory_profq_250, directory_profq_300, file_info_obs):
    list_good_obs = []

    # call ReadInfObs
    tab_name_obs, tab_longitude, tab_latitude, tab_localtime, \
    tab_solarlon, tab_distance = ReadInfObs(file_info_obs)

    # loop on all observations
    for o, name_obs in enumerate(tab_name_obs):
        filename = 'Orb_' + name_obs.decode('ascii')

        data_T = open_idlfile(directory_profT + filename + '_temp.idl')
        # if not, exit this orbit
        if data_T is False:
            continue

        data_aerosol_200 = open_idlfile(directory_profq_200 + filename + '_200nm.idl')
        data_aerosol_250 = open_idlfile(directory_profq_250 + filename + '_250nm.idl')
        data_aerosol_300 = open_idlfile(directory_profq_300 + filename + '_300nm.idl')


        # compute co2 saturation temperature
        T_sat = Co2Saturation(data_T['press'])

        # mask above 40 km: to get mesospheric cloud
        mask_40km = (data_T['tempalt'] >= 40.)

        # compute T_obs - T_sat
        delta_T = data_T['temp'][mask_40km] - T_sat[mask_40km]

        # determine caracteristics of all cold pockets
        if any(delta_T < 0) is True:
            obs = Observation(name=filename, latitude=tab_latitude[o], longitude=tab_longitude[o],
                              solarlon=tab_solarlon[o], localtime=tab_localtime[o])
            mask_cold = (delta_T < 0)
            cptt = 0
            nbcoldpocket1obs = 0

            print('---')
            #print(data_T['tempalt'][mask_40km][mask_cold])

            # determine the number of cold pockets for one observation
            for t in range(mask_cold.shape[0]):
                if mask_cold[t + cptt]:
                    cpt = 0
                    while (mask_cold[t + cptt + cpt]):
                        cpt += 1
                        if t + cptt + cpt == mask_cold.shape[0]:
                            break
                    #print(data_T['tempalt'][mask_40km][t + cptt:t + cptt + cpt])

                    nbcoldpocket1obs += 1
                    obs.add_coldpocket(Coldpocket(number=nbcoldpocket1obs,
                                                  altitude_bot=data_T['tempalt'][mask_40km][t + cptt],
                                                  altitude_top=data_T['tempalt'][mask_40km][t + cptt + cpt -1],
                                                  dT = np.min(delta_T)))
                    cptt += cpt
                #print(t + cptt + 1)
                if t + cptt + 1 >= mask_cold.shape[0]:
                    break

            list_good_obs.append(obs)
            print('number of cold pocket for %s: %i'%(filename, nbcoldpocket1obs))
            for i in range(nbcoldpocket1obs):
                print('         altitude of the cold pocket %i: %9.5f - %9.5f km' % (obs.coldpocket[i].number,
                                                                                     obs.coldpocket[i].index_bot,
                                                                                     obs.coldpocket[i].index_top))

    cpt_day=0
    cpt_night=0
    # list_thickness = np.array([])
    # list_temperature = np.array([])
    # # plot
    # fig, ax = plt.subplots(figsize=(8,11))
    # ax.set_title('Cold pockets above 50 km altitude, observed by SPICAM', fontsize=16)
    # ax.set_xlabel(u'Solar longitude (°)', fontsize=14)
    # ax.set_ylabel(u'Latitude (°N)', fontsize=14)
    #
    # for z in range(len(list_good_obs)):
    #     for l in range(len(list_good_obs[z].coldpocket[:])):
    #         test=list_good_obs[z].coldpocket[l].index_bot
    #         if test >= 50:
    #             diurne = list_good_obs[z].localtime
    #             thickness = list_good_obs[z].coldpocket[l].index_top - test
    #             list_thickness = np.append(list_thickness, thickness)
    #
    #             if diurne >= 6 and diurne <=18:
    #                 cpt_day+=1
    #                 if thickness < 5:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=5)
    #                 elif thickness < 15:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=15)
    #                 else:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=25)
    #             else:
    #                 cpt_night+=1
    #                 if thickness < 5:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=5)
    #                 elif thickness < 15:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=15)
    #                 else:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=25)
    #
    # ax.scatter(-100, 0, color='white', s=1, label='Thickness')
    # ax.scatter(-100, 0, color='blue', s=5, label='< 5 km')
    # ax.scatter(-100, 0, color='blue', s=15, label='< 15 km')
    # ax.scatter(-100, 0, color='blue', s=25, label='< 25 km')
    # ax.legend(bbox_to_anchor=(0.80, 0.35), loc='upper left', fontsize=14)
    # ax.text(230,-78, 'Day (6:00 to 18:00)', color='red', fontsize=14)
    # ax.text(230,-85, 'Night (18:00 to 6:00)', color='blue', fontsize=14)
    # ax.set_xticks([0,90,180,270,360])
    # ax.xaxis.label.set_fontsize(14)
    # ax.yaxis.label.set_fontsize(14)
    # ax.set_xlim(0,360)
    # ax.set_ylim(-90,90)
    # ax.grid()
    # plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets.ps', bbox_inches='tight')
    # plt.show()
    #
    # # 2nd plot
    # fig, ax = plt.subplots(figsize=(8,11))
    # ax.set_title('Cold pockets above 50 km altitude, observed by SPICAM', fontsize=16)
    # ax.set_xlabel(u'Solar longitude (°)', fontsize=14)
    # ax.set_ylabel(u'Latitude (°N)', fontsize=14)
    #
    # for z in range(len(list_good_obs)):
    #     for l in range(len(list_good_obs[z].coldpocket[:])):
    #         test=list_good_obs[z].coldpocket[l].index_bot
    #         if test >= 50:
    #             diurne = list_good_obs[z].localtime
    #             temperature = list_good_obs[z].coldpocket[l].dT
    #             list_temperature = np.append(list_temperature, temperature)
    #
    #             if diurne >= 6 and diurne <=18:
    #                 if -temperature < 5:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=5)
    #                 elif -temperature < 15:
    #                     ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=15)
    # #                elif -temperature < 25:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=25)
    #                else:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=50)
    #            else:
    #                if -temperature < 5:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=5)
    #                elif -temperature < 15:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=15)
    #                elif -temperature < 25:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=25)
    #                else:
    #                    ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=50)


    #ax.scatter(-100, 0, color='white', s=1, label='min(T-T$_c$)')
    #ax.scatter(-100, 0, color='blue', s=5, label='< 5 K')
    #ax.scatter(-100, 0, color='blue', s=15, label='< 15 K')
    #ax.scatter(-100, 0, color='blue', s=25, label='< 25 K')
    #ax.scatter(-100, 0, color='blue', s=50, label='> 25 K')

    #ax.legend(bbox_to_anchor=(0.85, 0.37), loc='upper left', fontsize=14)
    #ax.text(230,-78, 'Day (6:00 to 18:00)', color='red', fontsize=14)
    #ax.text(230,-85, 'Night (18:00 to 6:00)', color='blue', fontsize=14)
    #ax.set_xticks([0,90,180,270,360])
    #ax.xaxis.label.set_fontsize(14)
    #ax.yaxis.label.set_fontsize(14)
    #ax.set_xlim(0,360)
    #ax.set_ylim(-90,90)
    #ax.grid()
    #plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets_temperature.ps', bbox_inches='tight')
    #plt.show()

    # 3e plot
    fig, ax = plt.subplots(figsize=(8,11))
    fig.subplots_adjust(right=0.8)
    ax.set_title('Cold pockets above 50 km altitude, observed by SPICAM', fontsize=16)
    ax.set_xlabel(u'Solar longitude (°)', fontsize=14)
    ax.set_ylabel(u'Latitude (°N)', fontsize=14)

    for z in range(len(list_good_obs)):
        for l in range(len(list_good_obs[z].coldpocket[:])):
            test=list_good_obs[z].coldpocket[l].index_bot
            if test >= 50:
                diurne = list_good_obs[z].localtime
                thickness = list_good_obs[z].coldpocket[l].index_top - test
                temperature = list_good_obs[z].coldpocket[l].dT

                if diurne >= 6 and diurne <=18:
                    if -temperature < 5:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='orange', s=5, zorder=5,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=5, zorder=5,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='firebrick', s=5, zorder=5,edgecolors='black')

                    elif -temperature < 15:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='orange', s=50, zorder=4,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=50, zorder=4,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='firebrick', s=50, zorder=4,edgecolors='black')

                    elif -temperature < 25:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='orange', s=150, zorder=3,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=150, zorder=3,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='firebrick', s=150, zorder=3,edgecolors='black')

                    else:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='orange', s=300, zorder=2,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='red', s=300, zorder=2,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='firebrick', s=300, zorder=2,edgecolors='black')

                else:
                    if -temperature < 5:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='cyan', s=5, zorder=5,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=5, zorder=5,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='navy', s=5, zorder=5,edgecolors='black')

                    elif -temperature < 15:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='cyan', s=50, zorder=4,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=50, zorder=4,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='navy', s=50, zorder=4,edgecolors='black')

                    elif -temperature < 25:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='cyan', s=150, zorder=3,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=150, zorder=3,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='navy', s=150, zorder=3,edgecolors='black')

                    else:
                        if thickness < 5:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='cyan', s=300, zorder=2,edgecolors='black')
                        elif thickness <15:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='blue', s=300, zorder=2,edgecolors='black')
                        else:
                            ax.scatter(list_good_obs[z].solarlon, list_good_obs[z].latitude, c='navy', s=300, zorder=2,edgecolors='black')


    ax.scatter(-100, 0, color='white', s=1,edgecolors='white', label='min(T-T$_c$)')
    ax.scatter(-100, 0, color='blue', s=5,edgecolors='black', label='< 5 K')
    ax.scatter(-100, 0, color='blue', s=50,edgecolors='black', label='< 15 K')
    ax.scatter(-100, 0, color='blue', s=150,edgecolors='black', label='< 25 K')
    ax.scatter(-100, 0, color='blue', s=300,edgecolors='black', label='> 25 K')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14)


    ax.text(1.01,0.7,'Day / Nigth / Thickness', transform=ax.transAxes)

    circle1 = plt.Circle((1.04, 0.655), 0.01, color='orange',edgecolor='black',transform=ax.transAxes)
    fig.add_artist(circle1)
    circle2 = plt.Circle((1.12, 0.655), 0.01, color='cyan',edgecolor='black',transform=ax.transAxes)
    fig.add_artist(circle2)
    ax.text(1.18,0.65,'< 5 km', transform=ax.transAxes)

    circle1 = plt.Circle((1.04, 0.605), 0.01, color='red',edgecolor='black',transform=ax.transAxes)
    fig.add_artist(circle1)
    circle2 = plt.Circle((1.12, 0.605), 0.01, color='blue',edgecolor='black',transform=ax.transAxes)
    fig.add_artist(circle2)
    ax.text(1.18,0.60,'< 15 km', transform=ax.transAxes)

    circle1 = plt.Circle((1.12, 0.555), 0.01, color='navy',edgecolor='black',transform=ax.transAxes)
    fig.add_artist(circle1)
    ax.text(1.18,0.55,'> 15 km', transform=ax.transAxes)

    ax.set_xticks([0,90,180,270,360])
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.set_xlim(0,360)
    ax.set_ylim(-90,90)
    ax.grid()
    plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets2.ps', bbox_inches='tight')
    plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets2.eps', bbox_inches='tight')
    plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets2.png', bbox_inches='tight')
    plt.savefig('/home/mathe/Documents/owncloud/SPICAM/coldpockets2.pdf', bbox_inches='tight')
    plt.show()


    print('number of observations: ', tab_name_obs.shape[0])
    print('number of good obs: ', len(list_good_obs))
    print('%', 100*len(list_good_obs)/tab_name_obs.shape[0])

            #-----------------------------------------------------------------------------------------------------------
            #
            # compute dln(q) / dP
            #dxdy = np.diff(np.log(data_aerosol_250['ext_aer'][:,0])) / np.diff(data_aerosol_250['htfit'])

            #plt.figure(0)
            #plt.plot(dxdy, data_aerosol_250['htfit'][:-1])
            #plt.figure(1)
            #plt.xscale('log')
            #plt.errorbar(data_aerosol_200['ext_aer'][:,0], data_aerosol_200['htfit'], xerr=data_aerosol_200['err_aer'], color='blue')
            #plt.errorbar(data_aerosol_250['ext_aer'][:,0], data_aerosol_250['htfit'], xerr=data_aerosol_250['err_aer'], color='orange')
            #plt.errorbar(data_aerosol_300['ext_aer'][:,0], data_aerosol_300['htfit'], xerr=data_aerosol_300['err_aer'], color='red')
            #plt.show()

            # match cold pockets with local enrichment in aerosols: if yes = cloud !
            #-----------------------------------------------------------------------------------------------------------





if __name__ == '__main__':
    directory_profT = '/home/mathe/Documents/owncloud/SPICAM/Temperatures/'
    directory_profq_200 = '/home/mathe/Documents/owncloud/SPICAM/Aerosols_LocalDensities_200nm/'
    directory_profq_250 = '/home/mathe/Documents/owncloud/SPICAM/Aerosols_LocalDensities_250nm/'
    directory_profq_300 = '/home/mathe/Documents/owncloud/SPICAM/Aerosols_LocalDensities_300nm/'
    file_info_obs = '/home/mathe/Documents/owncloud/SPICAM/Obs-descriptions-Delivered.txt'
    main(directory_profT, directory_profq_200, directory_profq_250, directory_profq_300, file_info_obs)
