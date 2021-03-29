#!/bin/bash python3
from packages.displays import *
from os import listdir

from numpy import mean, min, max, logspace, arange, linspace

from sys import argv


def plot_sim_3d(filename, data_target, name_target, view_mode=None):

    data_target, local_time = extract_at_a_local_time(filename=filename, data=data_target)

    print('Correction value...')
    if data_target.ndim == 4:
        data_target = correction_value(data_target[:, :, :, :], 'inf', threshold=1e-13)
    elif data_target.ndim == 3:
        data_target = correction_value(data_target[:, :, :], 'inf', threshold=1e-13)

    # ================================================================================================================ #

    # 4-Dimension variable

    # ================================================================================================================ #
    if name_target in ['co2_ice', 'h2o_ice', 'q01']:  # q01 = h2o_ice
        print('What do you wanna do?')
        print('     1: maximum in altitude and longitude, with others variables at these places (fig: lat-ls)')
        print('     2: zonal mean column density (fig: lat-ls)')
        print('       201: special features for DARI report (compare to MOLA obs)')
        print('     3: ')
        print('     4: ')
        print('     5: ')
        print('     6: layer ice thickness (fig: thickness-lat)')
        print('     7: polar cloud distribution to compare with Fig.8 of Neumann+2003 (fig: #clouds-lat)')
        print('     8: cloud evolution with satuco2/temperature/radius (fig: alt-lat, gif)')
        print('     9: mmr structure in winter polar regions at 60°N/S (fig: alt-ls)')
        print('    10: Density column evolution in polar region, polar projection (fig: lon-lat)')
        if name_target in ['h2o_ice', 'q01']:
            print('     11: h2o_ice profile with co2_ice presence (fig: alt-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            max_mmr, max_temp, max_satu, max_radius, max_ccn_n, max_alt = \
                vars_max_value_with_others(filename=filename, data_target=data_target)

            print('Display:')
            display_max_lon_alt(name_target, max_mmr, max_alt, max_temp, max_satu, max_radius, max_ccnN,
                                unit='kg/kg')

        elif view_mode == 2 or view_mode == 201:
            print('Processing data:')
            data_processed, altitude_limit, altitude_min, altitude_max, altitude_unit = \
                vars_zonal_mean_column_density(filename, data_target)
            del data_target
            print('Display:')
            if view_mode == 2:
                display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed,
                                         unit='kg/m$^2$', norm=LogNorm(), levels=logspace(-13, 2, 16),
                                         observation=True, latitude_selected=None, localtime_selected=local_time,
                                         title=f'Zonal mean column density of {name_target} between {altitude_min:.1e}'
                                               f' and {altitude_max:.1e} {altitude_unit}, {local_time} h',
                                         tes=None, pfs=None, mvals=None, layer=None,
                                         save_name=f'zonal_mean_density_column_{name_target}_{altitude_min:.1e}_'
                                                  f'{altitude_max:.1e}_{altitude_unit}_{local_time}h')

            else:
                display_co2_ice_mola(filename, data_processed)

        elif view_mode == 6:
            print('Processing data:')
            data_icelayer = co2ice_thickness_atm_layer(filename=filename, data=data_target)

            print('Display:')
            display_thickness_co2ice_atm_layer(data_icelayer)

        elif view_mode == 7:
            print('Processing data:')
            distribution_north, distribution_south, latitude_north, latitude_south = co2ice_polar_cloud_distribution(
                filename, data_target, normalization='True')

            print('Display:')
            display_co2_ice_distribution_altitude_latitude_polar(filename, distribution_north, distribution_south,
                                                                 latitude_north, latitude_south,
                                                                 save_name='distribution_polar_clouds')

        elif view_mode == 8:
            print('Processing data:')
            data_target, data_satuco2, data_temp, data_riceco2, idx_max, latitude_selected = co2ice_cloud_evolution(
                filename, data_target)

            print('Display:')
            filenames = []
            for i in range(-9, 3):
                filenames.append(display_cloud_evolution_latitude(filename, data_target, data_satuco2,
                                                                  data_temp, data_riceco2, idx_max, i,
                                                                  latitude_selected))

            make_gif = input('Do you want create a gif (Y/n)?: ')
            if make_gif.lower() == 'y':
                create_gif(filenames)

        elif view_mode == 9 or view_mode == 901:
            print('Processing data:')
            data_north, data_south = temp_thermal_structure_polar_region(filename=filename, data=data_target)

            if view_mode == 901:
                path_2 = '../occigen_test_64x48x32_1years_Tµphy_para_start_simu_ref_Margaux_co2clouds_Radiatif_actif/'
                files = listdir(path_2)
                directory = []
                try:
                    directory = [x for x in files if 'occigen' in x][0] + '/'
                except not directory:
                    directory = ''

                if directory is not None:
                    files = listdir(path_2 + directory)

                filename_2 = getfilename(files)
                filename_2 = path_2 + directory + filename_2
                data_target_2 = getdata(filename_2, target=name_target)
                data_north_2, data_south_2 = temp_thermal_structure_polar_region(filename=filename_2,
                                                                                 data=data_target_2)

                print('Display:')
                display_temp_structure_polar_region(filename=filename,
                                                    data_north=data_north - data_north_2,
                                                    data_south=data_south - data_south_2,
                                                    norm=LogNorm(),
                                                    levels=None,
                                                    unit=unit_target,
                                                    save_name='diff_co2_ice_zonal_mean_60NS_' + directory[:-1])
            else:
                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north, data_south=data_south,
                                                    norm=LogNorm(),
                                                    levels=logspace(-13, 0, 14),
                                                    unit=unit_target,
                                                    save_name='co2_ice_zonal_mean_60NS')

        elif view_mode == 10:
            print('Processing data:')
            data_processed, time, latitude = co2ice_density_column_evolution(filename=filename, data=data_target)

            print('Display:')
            display_co2_ice_density_column_evolution_polar_region(filename=filename, data=data_processed, time=time,
                                                                  latitude=latitude)

        elif view_mode == 11:
            print('Processing data:')
            data_target, data_co2_ice, latitude_selected = h2o_ice_alt_ls_with_co2_ice(filename, data_target)

            print('Display:')
            display_alt_ls(filename, data_target, data_co2_ice, levels=logspace(-13, 2, 16),
                           title='Zonal mean of H2O ice mmr and CO2 ice mmr (black), at ' + str(int(
                               latitude_selected)) + '°N',
                           savename='h2o_ice_zonal_mean_with_co2_ice_' + str(int(latitude_selected)) + 'N',
                           latitude_selected=latitude_selected)

        else:
            print('Wrong value')
            exit()

    elif name_target in ['ccnNco2']:
        print('What do you wanna do?')
        print('     1: Zonal mean of mean density where co2_ice exists (fig: alt-#/m3)')
        print('')
        view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            list_data, list_filename, latitude_selected, list_time_selected, list_tau = \
                vars_zonal_mean_in_time_co2ice_exists(filename, data_target, name_target, density=True)

            print('Display:')
            display_1fig_profiles(filename, list_data, latitude_selected, xmin=1e0, xmax=1e10,
                                  xlabel='density of CCN (#.m$^{-3}$)',
                                  xscale='log', yscale='log',
                                  second_var=list_tau,
                                  xmin2=1e-9, xmax2=1e-1,
                                  xlabel2='opacity at 1 micron',
                                  xscale2='log',
                                  title='Zonal mean density of CCN and Opacity where co2 ice exists between {} - {} '
                                        '°N'.format(latitude_selected[0], latitude_selected[-1]),
                                  savename=list_filename,
                                  title_option=list_time_selected)

    elif name_target in ['riceco2']:
        print('What do you wanna do?')
        print('     1: mean radius at a latitude where co2_ice exists (fig: alt-µm)')
        print('     2: max radius day-night [To be done] (fig: lat-ls)')
        print('     3: altitude of top clouds (fig: lat-ls)')
        print('     4: radius/co2ice/temp/satu in polar projection (not working)')
        print('     5: zonal mean of mean radius where co2_ice exists in the 15°N-15°S (fig: lat-ls)')
        print('     6: mean radius profile along year, with global mean radius (fig: alt-ls + alt+µm)')
        print('     7: radius structure ')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            list_data, list_filename, latitude_selected, time_selected, list_tau = \
                vars_zonal_mean_in_time_co2ice_exists(filename, data_target, name_target, density=False)

            print('Display:')
            display_vars_1fig_profiles(filename, list_data, latitude_selected, x_min=1e-2, x_max=2000,
                                       x_label='radius of ice particle (µm)',
                                       x_scale='log', y_scale='log',
                                       second_var=None, x_min2=None, x_max2=None, x_label2=None, x_scale2=None,
                                       title='Mean radius of ice particle between {:.2f} - {:.2f} °N'
                                       .format(latitude_selected[0], latitude_selected[-1]),
                                       save_name=list_filename, title_option=time_selected)

        if view_mode == 2:
            print('Processing data:')
            riceco2_max_day_night(filename=filename, data=data_target)

            print('Display:')

        if view_mode == 3:
            print('Processing data:')
            top_cloud = riceco2_top_cloud_altitude(filename, data_target)

            print('Display:')
            display_riceco2_topcloud_altitude(filename, top_cloud)

        if view_mode == 4:
            print('Processing data:')

            print('Display:')
            display_4figs_polar_projection(filename, data_target[:, :, :, :])

        if view_mode == 5:
            print('Processing data:')
            data, latitude_selected = riceco2_zonal_mean_co2ice_exists(filename, data_target)

            print('Display:')
            display_colonne(filename, data, unit='µm', norm='log', levels=logspace(-2, 2, 5),
                            observation=True,
                            latitude_selected=latitude_selected,
                            title='Zonal mean of mean radius of co2 ice',
                            savename='riceco2_zonal_mean_altitude_mean_equatorial_region')

        if view_mode == 6:
            print('Processing data:')
            list_data = vars_zonal_mean_where_co2ice_exists(filename=filename, data=data_target, polar_region=True)

            print('Display:')
            display_riceco2_global_mean(filename, list_data)

        if view_mode == 7 or view_mode == 701:
            print('Processing data:')
            data_north, data_south = temp_thermal_structure_polar_region(filename=filename, data=data_target)

            if view_mode == 701:
                path_2 = '../occigen_test_64x48x32_1years_Tµphy_para_start_simu_ref_Margaux_co2clouds_Radiatif_actif/'
                files = listdir(path_2)
                print(files)
                directory = []
                try:
                    directory = [x for x in files if 'occigen' in x][0] + '/'
                except not directory:
                    directory = None
                if directory is None:
                    directory = ''
                else:
                    files = listdir(path_2 + directory)
                filename_2 = getfilename(files)
                filename_2 = path_2 + directory + filename_2
                data_target_2 = getdata(filename_2, target=name_target)
                data_north_2, data_south_2 = temp_thermal_structure_polar_region(filename=filename_2,
                                                                                 data=data_target_2)

                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north - data_north_2,
                                                    data_south=data_south - data_south_2, norm=None,
                                                    levels=None,
                                                    unit=unit_target,
                                                    save_name='diff_riceoc2_zonal_mean_60NS_' + directory[:-1])
            else:
                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north, data_south=data_south,
                                                    norm=None,
                                                    levels=None,
                                                    unit=unit_target,
                                                    save_name='riceco2_zonal_mean_60NS')

    elif name_target in ['satuco2']:
        print('What do you wanna do?')
        print('     1: zonal mean of saturation, for 3 latitudes, with co2ice mmr (fig: alt-ls)')
        print('     2: [TBD] saturation in localtime (fig: lt-alt')
        print('     3: saturation with co2ice mmr in polar regions, [0:30]°ls SP, [270-300]°ls NP (fig: alt-lon)')
        print('     4: Thickness atmosphere layer in polar regions, to compare with Fig.9 of Hu2019 (fig: thick-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, \
                data_co2ice_south, latitude_north, latitude_eq, latitude_south, binned = \
                satuco2_zonal_mean_with_co2_ice(filename=filename, data=data_target)

            print('Display:')
            display_satuco2_with_co2_ice_altitude_ls(filename, data_satuco2_north, data_satuco2_eq, data_satuco2_south,
                                                     data_co2ice_north, data_co2ice_eq, data_co2ice_south,
                                                     latitude_north, latitude_eq, latitude_south, binned)

        if view_mode == 2:
            print('TO be done.')

        if view_mode == 3:
            print('Processing data:')
            data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south, latitude_north, \
                latitude_south, binned = satuco2_time_mean_with_co2_ice(filename, data_target)

            print('Display:')
            display_satuco2_with_co2_ice_altitude_longitude(filename, data_satuco2_north, data_satuco2_south,
                                                            data_co2ice_north, data_co2ice_south, latitude_north,
                                                            latitude_south, binned)

        if view_mode == 4:
            print('Processing data:')
            data_icelayer, data_icelayer_std = satuco2_hu2012_fig9(filename, data_target)

            print('Display:')
            display_satuco2_thickness_atm_layer(data_icelayer, data_icelayer_std,
                                                save_name='satuco2_thickness_polar_region.png')

    elif name_target in ['saturation']:
        print('What do you wanna do?')
        print('     1: profile of saturation zonal-mean in a latitude region (fig: satu-alt)')
        print('')
        view_mode = int(input('Select number:'))

        if view_mode == 1:
            lat1 = float(input('Select the first latitude (°N): '))
            lat2 = float(input('Select the second latitude (°N): '))
            data_latitude = getdata(filename, target='latitude')
            data_target, tmp = slice_data(data_target, dimension_data=data_latitude[:], value=[lat1, lat2])

            data_target = correction_value(data_target[:, :, :, :], operator='inf', threshold=1e-13)
            data_target = mean(data_target[:, :, :, :], axis=3)

            data_target = mean(data_target[::72, :, :], axis=0)
            print(data_target.shape)

            display_saturation_profiles(filename, data_target)

    elif name_target in ['temp']:
        print('What do you wanna do?')
        print('     1: dT altitude - LT                  // Ls=120–150°, lon=0°, lat=0° [Fig 6, G-G2011]')
        print('     2: dT altitude - latitude            // Ls=0–30°, LT=16             [fig 7, G-G2011]')
        print('     3: zonal mean at LT=16h,  12h - 00h  // Ls=0-30°, (2 figs: alt-lat) [fig.8, G-G2011]')
        print('     4: dT altitude - longitude           // Ls=0–30°, LT=16, lat=0°     [fig 9, G-G2011]')
        print('     5: Zonal mean for the X layer [to be fixed]')
        print('     6: Thermal structure in winter polar regions at 60°N/S (fig. alt-ls)')
        print('     \t 601: compare with another run')
        print('     7: dT zonal mean and altitude of cold pocket, compared to SPICAM data')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))
        else:
            print('You have selected view_mode number: {}'.format(view_mode))
        print('')

        if view_mode == 1:
            print('Processing data:')
            data_processed, data_localtime = temp_gg2011_fig6(filename=filename, data=data_target)

            print('Display:')
            display_temp_gg2011_fig6(filename=filename, data=data_processed, data_localtime=data_localtime)

        if view_mode == 2:
            print('Processing data:')
            data_processed, data_altitude = temp_gg2011_fig7(filename=filename, data=data_target)

            print('Display:')
            display_temp_gg2011_fig7(filename=filename, data=data_processed, data_altitude=data_altitude)

        if view_mode == 3:
            print('Processing data:')
            data_zonal_mean, data_thermal_tides = temp_gg2011_fig8(filename=filename, data=data_target)

            print('Display:')
            display_temp_gg2011_fig8(filename, data_zonal_mean, data_thermal_tides)

        if view_mode == 4:
            print('Processing data:')
            data_processed, data_altitude = temp_gg2011_fig9(filename=filename, data=data_target)

            print('Display:')
            display_temp_gg2011_fig9(filename=filename, data=data_processed, data_altitude=data_altitude)

        if view_mode == 5:
            print('Parameters:')
            layer = int(input('\t layer (from 1 to {}): '.format(data_target.shape[1]))) - 1

            print('Processing data:')
            data_processed, layer_selected = vars_zonal_mean(filename, data_target, layer=layer)

            print('Display:')
            display_vars_lat_ls_compare_pfs_tes_mvals(filename, data_processed, name_target, layer,
                                                      savename='temp_zonal_mean_layer{}_{:.0e}_Pa_comparison_tes_mvals'.
                                                      format(layer, layer_selected))

        if view_mode == 6 or view_mode == 601:
            print('Processing data:')
            data_north, data_south = temp_thermal_structure_polar_region(filename=filename, data=data_target)

            if view_mode == 601:
                path_2 = '../occigen_test_64x48x32_1years_Tµphy_para_start_simu_ref_Margaux_co2clouds_Radiatif_actif/'
                files = listdir(path_2)
                print(files)
                directory = []
                try:
                    directory = [x for x in files if 'occigen' in x][0] + '/'
                except not directory:
                    directory = None

                if directory is not None:
                    files = listdir(path_2 + directory)

                filename_2 = getfilename(files)
                filename_2 = path_2 + directory_store + filename_2
                data_target_2 = getdata(filename_2, target=name_target)
                data_north_2, data_south_2 = temp_thermal_structure_polar_region(filename=filename_2,
                                                                                 data=data_target_2)

                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north - data_north_2,
                                                    data_south=data_south - data_south_2, norm=None,
                                                    levels=arange(-20, 20, 1),
                                                    unit=unit_target,
                                                    save_name='difftemp_zonal_mean_60NS_' + directory_store[:-1])
            else:
                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north, data_south=data_south,
                                                    norm=None,
                                                    levels=arange(80, 320, 20),
                                                    unit=unit_target,
                                                    save_name='temp_zonal_mean_60NS')

        if view_mode == 7:
            data_zonal_mean = temp_cold_pocket(filename=filename, data=data_target)
            print('Need to be done')
            print(data_zonal_mean)

    # ================================================================================================================ #

    # 3-Dimension variable

    # ================================================================================================================ #
    elif name_target in ['co2ice']:
        print('What do you wanna do?')
        print('     1: cumulative masses in polar cap region, to compare with fig.10 of Hu+2012 (fig: g-ls)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lat-ls)')
        print('')

        if view_mode is None:
            view_mode = int(input('Select number:'))
        if view_mode == 1:
            print('Processing data:')
            accumulation_north, accumulation_south = co2ice_cumulative_masses_polar_cap(filename, data_target)

            print('Display:')
            display_vars_2fig_profile(filename, accumulation_north, accumulation_south)

        if view_mode == 2:
            print('Processing data:')
            data_mean, time_bin = co2ice_time_mean(filename=filename, data=data_target, duration=15,
                                                   localtime=local_time)

            print('Display:')
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     localtime=local_time, levels=linspace(0, 1e13, 100), norm=None,
                                                     cmap='inferno',
                                                     unit='kg', save_name=f'co2ice_15ls_mean_{local_time}h')

    elif name_target in ['emis']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: ls-lat)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lon-lat)')
        print('        201: display for direct comparison with figs. 11  and 12 [Gary-Bicas2020], (fig: lon-lat)')
        print('     3: Polar plot time mean during winter [Gary-Bicas2020, fig. 13] (fig: lon-lat)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing:')
            zonal_mean, layer_selected = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            display_colonne(filename=filename, data=zonal_mean, unit=unit_target, norm=None,
                            levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            observation=False,
                            latitude_selected=layer_selected,
                            title='Zonal mean of {}'.format(name_target),
                            savename='{}_zonal_mean'.format(name_target))

        if view_mode == 2 or view_mode == 201:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15)

            print('Display:')
            if view_mode == 2:
                display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                         localtime=local_time,
                                                         levels=linspace(0.55, 1., 100), norm=None, cmap='inferno',
                                                         unit='',
                                                         save_name='emis_15ls_mean')
            if view_mode == 201:
                display_emis_polar_projection_garybicas2020_figs11_12(filename=filename, data=data_mean, time=time_bin,
                                                                      levels=linspace(0.55, 1., 100), cmap='inferno',
                                                                      unit='',
                                                                      save_name='emis_15ls_mean_')

        if view_mode == 3:
            print('Processing data:')
            data_mean_np, data_mean_sp = emis_polar_winter_gg2020_fig13(filename=filename, data=data_target)

            print('Display:')
            display_vars_polar_projection(filename=filename, data_np=data_mean_np, data_sp=data_mean_sp,
                                          levels=linspace(0.55, 1., 100), unit='', cmap='inferno',
                                          sup_title='Surface emissivity mean in time during polar winter',
                                          save_name='emis_time_mean_gary-bicas2020')

    elif name_target in ['fluxtop_lw', 'fluxtop_sw', 'fluxsurf_lw', 'fluxsurf_sw']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: ls-lat)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing:')
            zonal_mean, layer_selected = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit=unit_target,
                                     norm=None, levels=arange(0, 160, 20), observation=False,
                                     latitude_selected=layer_selected, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target}',
                                     tes=None, pfs=None, mvals=None, layer=None,
                                     save_name=f'{name_target}_zonal_mean')

        if view_mode == 2:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15)

            print('Display:')
            print(max(data_mean))
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     levels=arange(0, 450, 25), norm=None, cmap='inferno',
                                                     localtime=local_time, unit=unit_target,
                                                     save_name=f'{name_target}_15ls_mean_')

    elif name_target in ['tau1mic']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed, tmp = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit=unit_target,
                                     norm=LogNorm(), levels=logspace(-13, -1, 13), observation=False,
                                     latitude_selected=None, title=name_target, tes=None, pfs=None, mvals=None,
                                     save_name='tau1mic_zonal_mean')

    elif name_target in ['tau']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed, tmp = vars_zonal_mean(filename, data_target, layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit=unit_target,
                                     norm=None, levels=arange(0, 1.1, 0.1), observation=False,
                                     latitude_selected=None, title=name_target, tes=None, pfs=None, mvals=None,
                                     save_name='tau_zonal_mean')

    elif name_target in ['tauTES']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed, tmp = vars_zonal_mean(filename, data_target, layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit=unit_target,
                                     norm=None, levels=[0, 0.5, 1., 3., 5., 7., 10.], observation=False,
                                     latitude_selected=None, title=name_target, tes=None, pfs=None, mvals=None,
                                     save_name='tauTES_zonal_mean')

    elif name_target in ['tsurf']:
        print('What do you wanna do?')
        print('     1: Zonal mean (fig: lat-ls)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            zonal_mean, tmp = vars_zonal_mean(filename, data_target[:, :, :], layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='K', norm=None,
                                     levels=arange(100, 375, 25), observation=False, latitude_selected=None,
                                     title=None, tes=True, pfs=None, mvals=True, layer=None,
                                     save_name='tsurf_zonal_mean')

        if view_mode == 2:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15)

            print('Display:')
            print(min(data_mean))
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     levels=arange(140, 310, 10), norm=None, cmap='seismic',
                                                     unit=unit_target, localtime=local_time,
                                                     save_name='tsurf_15ls_mean_')

    # ================================================================================================================ #

    # 1-Dimension variable

    # ================================================================================================================ #
    elif name_target in ['co2_conservation', 'Sols', 'Ls']:
        display_1d(data_target)

    else:
        print('Variable not used for the moment')


def plot_sim_1d(data_target, name_target, view_mode=None):

    if name_target in ['riceco2']:
        print('What do you wanna do?')
        print('     1: mean radius at a latitude where co2_ice exists (fig: alt-µm)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            plt.figure(0)
            ctf = plt.contourf(data_target[:, :, 0, 0].T * 1e6, levels=arange(0, 650, 50))
            cbar = plt.colorbar(ctf)
            cbar.ax.set_title('µm')
            plt.xlabel('Time')
            plt.ylabel('Level')
            plt.show()

    if name_target in ['emis']:
        print('What do you wanna do?')
        print('     1: basic plot ls - altitude')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            plt.figure()
            plt.plot(data_target[:, 0, 0], color='black')
            plt.show()

    if name_target in ['co2conservation']:
        plt.figure()
        plt.plot(data_target[:], color='black')
        plt.show()


def main():
    arg_file = None
    arg_target = None
    arg_view_mode = None
    if len(argv) > 2:
        arg_file = int(argv[1])
        arg_target = argv[2]
        if len(argv) == 4:
            arg_view_mode = int(argv[3])

    files = listdir('.')
    directory_store = []
    try:
        directory_store = [x for x in files if 'occigen' in x][0] + '/'
    except not directory_store:
        directory_store = None

    if directory_store is None:
        directory_store = ''
    else:
        files = listdir(directory_store)

    filename = getfilename(files, selection=arg_file)
    filename = directory_store + filename

    data_target = getdata(filename, target=arg_target)
    print(f'You have selected the variable: {data_target.name}')

    if data_target.ndim <= 2:
        plot_sim_1d(data_target=data_target, name_target=data_target.name, view_mode=arg_view_mode)
    else:
        plot_sim_3d(filename=filename, data_target=data_target, name_target=data_target.name, view_mode=arg_view_mode)

    return


if '__main__' == __name__:
    main()
