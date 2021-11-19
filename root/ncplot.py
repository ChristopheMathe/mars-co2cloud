#!/bin/bash python3
from packages.displays import *
from packages.ncdump import *
from os import listdir
from numpy import mean, logspace, arange, linspace, min, max, log
from sys import argv
from packages.constant_parameter import threshold


def plot_sim_3d(filename, data_target, name_target, directory, files, view_mode=None):
    # With h2o_ice_s we must add watercap before anything we attempt to
#    if name_target == "h2o_ice_s":
#        data_watercap, list_var = get_data(filename=filename, target='watercap')
#        data_target = data_target[:, :, :] #- data_watercap[:, :, :]

    data_target, local_time = extract_at_a_local_time(filename=filename, data=data_target, local_time=None)


    print('Correction value...')
    if data_target.ndim == 4:
        if name_target == 'satuco2':
            data_target = correction_value(data_target[:, :, :, :], operator='inf_strict', threshold=1)
        else:
            data_target = correction_value(data_target[:, :, :, :], operator='inf', threshold=threshold)
    elif data_target.ndim == 3:
        data_target = correction_value(data_target[:, :, :], operator='inf', threshold=threshold)

    # ================================================================================================================ #

    # 4-Dimension variable

    # ================================================================================================================ #
    if name_target in ['co2_ice', 'h2o_ice', 'q01', 'dustq']:  # q01 = h2o_ice
        print('What do you wanna do?')
        print('     1: maximum in altitude and longitude, with others variables at these places (fig: lat-ls)')
        print('     2: zonal mean column density (fig: lat-ls)')
        print('        201: special features for DARI report (compare to MOLA obs)')
        print('        202: adapted for Anni paper')
        print('     3: co2_ice coverage (fig: lat-lon)')
        print('     4: polar cloud distribution to compare with Fig.8 of Neumann+2003 (fig: #clouds-lat)')
        print('     5: cloud evolution with satuco2/temperature/radius (fig: alt-lat, gif)')
        print('     6: mmr structure at a given latitude (fig: alt-ls)')
        print('     7: Density column evolution in polar region, polar projection (fig: lon-lat)')
        print('     8: h2o_ice profile with co2_ice presence (fig: alt-ls)')
        print('     9: localtime co2_ice column density, zonal mean, [XX-YY]°N (fig: loc-ls)')
        print('        901: localtime co2_ice at 0.5 Pa, 0°N, zonal mean (loc-ls)')
        print('    10: co2_ice column density along longitude and localtime at 0.5 Pa and 0°N (fig: hl-lon)')
        print('    11: co2_ice column density along longitude and solar longitude at 0.5 Pa and 0°N (fig: ls-lon)')
        print('    12: co2_ice structure along longitude at 0°N (year mean) (fig: alt-lon)')
        print('    13: stationary wave, at 0°N and -45°E (fig: alt-ls)')
        print('    14: Polar plot every 30° ls mean, column density, lat=60°-90° (fig: lat-ls)')
        print('')

        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            max_mmr, max_temp, max_satu, max_radius, max_ccn_n, max_alt = \
                vars_max_value_with_others(filename=filename, data_target=data_target)

            print('Display:')
            display_co2_ice_max_longitude_altitude(filename=filename, name=name_target, max_mmr=max_mmr,
                                                   max_alt=max_alt, max_temp=max_temp, max_satu=max_satu,
                                                   max_radius=max_radius, max_ccn_n=max_ccn_n, unit='kg/kg')

        elif view_mode in [2, 201, 202]:
            print('Processing data:')
            data_processed, altitude_limit, altitude_min, altitude_max, altitude_unit = \
                vars_zonal_mean_column_density(filename, data_target, local_time=local_time)

            print('Display:')
            if view_mode == 2:
                if name_target == 'co2_ice':
                    vmin, vmax = 1e-13, 10
                else:
                    vmin, vmax = 1e-7, 1e-1
                display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed,
                                         unit='kg/m$^2$', norm='log', vmin=vmin, vmax=vmax, cmap='coolwarm',
                                         observation=True, latitude_selected=None, localtime_selected=local_time,
                                         title=f'Zonal mean column density of {name_target}\n between'
                                               f' {altitude_min:.1e}'
                                               f' and {altitude_max:.1e} {altitude_unit}, {local_time} h',
                                         tes=None, mvals=None, layer=None,
                                         save_name=f'zonal_mean_density_column_{name_target}_{altitude_min:.1e}_'
                                                   f'{altitude_max:.1e}_{altitude_unit}_{local_time}h')

            elif view_mode == 201:
                display_co2_ice_mola(filename, data_processed)

            elif view_mode == 202:
                if name_target == 'co2_ice':
                    vmin, vmax = 1e-13, 10
                elif name_target == 'dustq':
                    vmin, vmax = 1e-5, 1e-2
                else:
                    vmin, vmax = 1e-7, 1e-1

                if len(local_time) == 1:
                    display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed,
                                             unit='kg/m$^2$', norm='log', vmin=vmin, vmax=vmax, cmap='coolwarm',
                                             observation=False, latitude_selected=None, localtime_selected=local_time,
                                             title=f'Zonal mean column density of {name_target}'
                                                   f' ({local_time[0]:.0f}h)',
                                             tes=None, mvals=None, layer=None,
                                             save_name=f'zonal_mean_density_column_{name_target}_{altitude_min:.1e}_'
                                                       f'{altitude_max:.1e}_{altitude_unit}_{local_time[0]:.0f}h')
                else:
                    display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed,
                                             unit='kg/m$^2$', norm='log', vmin=vmin, vmax=vmax, cmap='coolwarm',
                                             observation=True, latitude_selected=None, localtime_selected=local_time,
                                             title=f'Zonal and diurnal mean of column density of CO$_2$ ice',
                                             tes=None, mvals=None, layer=None,
                                             save_name=f'zonal_mean_density_column_{name_target}_diurnal_mean')

        elif view_mode == 3:
            print('Processing data:')
            data_processed, data_co2ice_coverage_meso = co2ice_coverage(filename=filename, data=data_target)

            print('Display:')
            if len(local_time) == 1:
                print(max(data_processed))
                display_vars_latitude_longitude(filename=filename, data=data_processed,
                                                unit='%', norm=None, vmin=0, vmax=70,
                                                title=f'Percentage of Martian year with presence of CO2 clouds '
                                                      f'({local_time} h)',
                                                save_name=f'co2_ice_coverage_{local_time}h')
            else:
                print(max(data_processed))
                display_vars_latitude_longitude(filename=filename, data=data_processed,
                                                unit='%', norm=None, vmin=0, vmax=70,
                                                title=f'Percentage of Martian year with presence of CO2 clouds '
                                                      f'(diurnal mean)',
                                                save_name=f'co2_ice_coverage_diurnal_mean')

        elif view_mode == 4:
            print('Processing data:')
            distribution_north, distribution_south, latitude_north, latitude_south = co2ice_polar_cloud_distribution(
                filename, data_target, normalization=True, local_time=local_time)

            print('Display:')
            if len(local_time) == 1:
                display_co2_ice_distribution_altitude_latitude_polar(filename, distribution_north, distribution_south,
                                                                     latitude_north, latitude_south,
                                                                     save_name=f'distribution_polar_clouds_'
                                                                               f'{local_time}h')
            else:
                display_co2_ice_distribution_altitude_latitude_polar(filename, distribution_north, distribution_south,
                                                                     latitude_north, latitude_south,
                                                                     save_name='distribution_polar_clouds_diurnal_mean')

        elif view_mode == 5:
            print('Processing data:')
            data_target, data_satuco2, data_temp, data_riceco2, idx_max, latitude_selected = co2ice_cloud_evolution(
                filename, data_target)

            print('Display:')
            filenames = []
            for nb in range(-9, 3):
                filenames.append(display_co2_ice_cloud_evolution_latitude(filename, data_target, data_satuco2,
                                                                          data_temp, data_riceco2, idx_max, nb,
                                                                          latitude_selected))

            make_gif = input('Do you want create a gif (Y/n)?: ')
            if make_gif.lower() == 'y':
                create_gif(filenames)

        elif view_mode == 6:
            latitude = float(input('Enter a latitude (°N): '))

            print('Processing data:')
            data_processed, latitude = vars_altitude_ls(filename=filename, data=data_target, latitude=latitude,
                                                        local_time=local_time)

            print('Display:')
            display_vars_altitude_ls(filename=filename, data_1=data_processed, data_2=None, altitude_max=None,
                                     norm='log', unit='kg/kg', vmin=1e-13, vmax=1e-4,
                                     title=f'{name_target} mmr at {latitude} ({local_time[0]:.0f} h), DS: MY28',
                                     local_time=local_time,
                                     save_name=f'{name_target}_zonal_mean_{latitude}N_{local_time[0]:.0f}h')

        elif view_mode == 7:
            print('Processing data:')
            data_processed, time, latitude = co2ice_density_column_evolution(filename=filename, data=data_target,
                                                                             localtime=local_time)

            print('Display:')
            display_co2_ice_density_column_evolution_polar_region(filename=filename, data=data_processed, time=time,
                                                                  latitude=latitude)

        elif view_mode == 8 and name_target == 'h2o_ice':
            print('Processing data:')
            data_target, data_co2_ice, latitude_selected = h2o_ice_alt_ls_with_co2_ice(filename=filename,
                                                                                       data=data_target,
                                                                                       local_time=local_time,
                                                                                       files=files,
                                                                                       directory=directory)

            print('Display:')
            if len(local_time) == 1:
                vmax_2 = 1e-6
                if len(latitude_selected) == 1:
                    title = f'Zonal mean of H$_2$O ice mmr (yellowish) and\n CO$_2$ ice mmr (greenish), ' \
                            f'at {int(latitude_selected)}°N ({int(local_time[0])} h)'
                    save_name = f'h2o_ice_zonal_mean_with_co2_ice_{int(latitude_selected)}N_{int(local_time[0])}h'
                else:
                    title = f'Zonal mean of H$_2$O ice mmr (yellowish) and CO$_2$ ice\nmmr (greenish), ' \
                            f'between [{int(latitude_selected[0])}:{int(latitude_selected[-1])}]°N ' \
                            f'({int(local_time[0])}h)'
                    save_name = f'h2o_ice_zonal_mean_with_co2_ice_{int(latitude_selected[0])}_' \
                                f'{int(latitude_selected[-1])}N_{int(local_time[0])}h'
            else:
                vmax_2 = 1e-6
                if len(latitude_selected) == 1:
                    title = f'Zonal and diurnal mean of H$_2$O ice (yellowish) and \n CO$_2$ ice (greenish) mmr, ' \
                            f'at {int(latitude_selected)}°N'
                    save_name = f'h2o_ice_zonal_mean_with_co2_ice_{int(latitude_selected)}N_diurnal_mean'
                else:
                    title = f'Zonal and diurnal mean of H$_2$O ice (yellowish) and\nCO$_2$ ice (greenish) mmr, ' \
                            f'between [{int(latitude_selected[0])}:{int(latitude_selected[-1])}]°N'
                    save_name = f'h2o_ice_zonal_mean_with_co2_ice_{int(latitude_selected[0])}_' \
                                f'{int(latitude_selected[-1])}N_diurnal_mean'

            display_vars_altitude_ls(filename=filename, data_1=data_target,
                                     varname_1='H2O ice mmr in atmospheric layer', shortname_1='h2o_ice',
                                     local_time=local_time, latitude=latitude_selected,
                                     norm='log', vmin=1e-13, vmax=1e-3, unit='kg/kg', altitude_max=1e-3,
                                     title=title, save_name=save_name, data_2=data_co2_ice, norm_2='log',
                                     vmin_2=1e-13, vmax_2=vmax_2, varname_2='CO2 ice mmr in atmospheric layer',
                                     shortname_2='co2_ice')

        elif view_mode in [9, 901]:
            if view_mode == 9:
                print('Processing data:')
                data_processed, altitude_min, latitude_min, latitude_max = co2ice_cloud_localtime_along_ls(
                    filename=filename, data=data_target)
                print('Display:')
                display_co2_ice_localtime_ls(filename=filename, data=data_processed, lat_min=latitude_min,
                                             lat_max=latitude_max, unit='kg/m2', norm='log', vmin=1e-13, vmax=1e-6,
                                             title=f'Zonal mean of CO2 ice column density\n'
                                                   f' above {altitude_min:.0e}Pa, between [{latitude_min:.0f}:'
                                                   f'{latitude_max:.0f}]°N',
                                             save_name=f'co2_ice_zonal_mean_localtime_ls_{latitude_min:.0f}_'
                                                       f'{latitude_max:.0f}N_above{altitude_min:.0e}Pa')

            else:
                data_processed = vars_localtime_ls(filename=filename, data=data_target, latitude=0, altitude=0.5)
                print('Display:')
                display_co2_ice_localtime_ls(filename=filename, data=data_processed, unit='kg/kg', norm='log',
                                             vmin=1e-13, vmax=1e-9, title='CO2 ice mmr at 0°N and 0.5 Pa',
                                             save_name='co2_ice_zonal_mean_localtime_ls_0N_0p5Pa')

        elif view_mode == 10:
            print('Processing data:')
            data_processed = vars_localtime_longitude(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            display_vars_localtime_longitude(filename=filename, data=data_processed, norm='log', vmin=1e-13, vmax=1e-7,
                                             unit='kg/kg', title=f'CO$_2$ ice mmr at 0°N and 0.5 Pa',
                                             save_name=f'co2_ice_local_time_longitude_0N_0p5Pa')

        elif view_mode == 11:
            print('Processing data:')
            data_processed = vars_ls_longitude(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            display_vars_ls_longitude(filename=filename, data=data_processed, norm='log', vmin=1e-13, vmax=1e-7,
                                      local_time=local_time,
                                      unit='kg/kg', title=f'CO$_2$ ice mmr at 0°N and 0.5 Pa ('
                                                          f'{local_time[0]:.0f}h)',
                                      save_name=f'co2_ice_ls_longitude_0N_0p5Pa_{local_time[0]:.0f}h')
        elif view_mode == 12:
            print('Processing data:')
            data_latitude, list_var = get_data(filename=filename, target='latitude')
            data, latitudes = slice_data(data=data_target, dimension_data=data_latitude[:], value=0)
            # average over the year at the same localtime
            data_processed = mean(data, axis=0)

            print('Display:')
            vmin = 1e-13
            vmax = 1e-6
            title = f'CO2 ice at 0°N, averaged over 1 year ({local_time}h) '
            save_name = f'co2_ice_altitude_longitude_0N_{local_time}h'
            display_vars_altitude_longitude(filename=filename, data=data_processed, unit='kg.kg-1', norm='log',
                                            vmin=vmin, vcenter=None, vmax=vmax, title=title, save_name=save_name)

        elif view_mode == 13:
            print('Processing data:')
            data_processed = vars_extract_at_grid_point(filename=filename, data=data_target, latitude=0, longitude=-45)

            print('Display:')
            vmin = 1e-13
            vmax = 1e-6
            print(min(data_processed), max(data_processed))
            title = f'CO2 ice at [0°N, -45°E] ({local_time}h) '
            save_name = f'co2_ice_altitude_ls_0N_-45E_{local_time}h'
            display_vars_altitude_ls(filename=filename, data_1=data_processed, local_time=local_time, altitude_max=None,
                                     norm='log', unit='kg.kg-1', vmin=vmin, vmax=vmax, title=title, save_name=save_name)

        elif view_mode == 14:
            print('Processing data:')
            data_mean, time_bin = co2ice_time_mean(filename=filename, data=data_target, duration=30,
                                                   localtime=local_time, column=True)

            print('Display:')
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     localtime=local_time, vmin=None, vmax=None, norm='nonlinear',
                                                     cmap='inferno', unit='kg/m$^2$',
                                                     title='density column of CO$_2$',
                                                     save_name=f'co2_ice_density_column_15ls_mean',
                                                     levels=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1])
        else:
            print('Wrong value')
            exit()

    elif name_target in ['ccnNco2']:
        print('What do you wanna do?')
        print('     1: Zonal mean of CCNCO2 density where co2_ice exists (fig: alt-#/m3)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            list_data, list_filename, latitude_selected, list_time_selected = \
                vars_zonal_mean_in_time_co2ice_exists(filename=filename, data=data_target, data_name=name_target,
                                                      local_time=local_time)

            print('Display:')
            display_vars_1fig_profiles(filename, list_data, latitude_selected, x_min=1e0, x_max=1e10,
                                       x_label='density of CCN (#.m$^{-3}$)',
                                       x_scale='log', y_scale='log',
                                       second_var=None,
                                       x_min2=None, x_max2=None,
                                       x_label2='None',
                                       x_scale2='None',
                                       title=f'Zonal mean density of CCN where co2 ice exists between'
                                             f' {latitude_selected[0]:.0f} - {latitude_selected[-1]:.0f}°N '
                                             f'({local_time}h)',
                                       save_name=list_filename,
                                       title_option=list_time_selected)

    elif name_target in ['riceco2']:
        print('What do you wanna do?')
        print('     1: mean radius at a latitude where co2_ice exists (fig: alt-µm)')
        print('     2: max radius day-night [To be done] (fig: lat-ls)')
        print('     3: altitude of top clouds (fig: lat-ls)')
        print('        301: same but adapted for paper, limited to 40 km altitude')
        print('     4: radius/co2ice/temp/satu in polar projection (not working)')
        print('     5: zonal mean of mean radius where co2_ice exists in the 15°N-15°S (fig: lat-ls)')
        print('     6: mean radius profile along year, with global mean radius (fig: alt-ls + alt+µm)')
        print('     7: radius structure in polar regions for each latitudes, diurnal mean (fig: alt-µm)')
        print('     8: radius local time evolution at a latitude (fig: alt-µm)')
        print('     9: max radius local time evolution at 0°N (fig: µm-lt)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            list_data, list_filename, latitude_selected, time_selected = \
                vars_zonal_mean_in_time_co2ice_exists(filename=filename, data=data_target, data_name=name_target,
                                                      local_time=local_time)

            print('Display:')
            display_vars_1fig_profiles(filename, list_data, latitude_selected, x_min=1e-2, x_max=2000,
                                       x_label='radius of ice particle (µm)',
                                       x_scale='log', y_scale='log',
                                       second_var=None, x_min2=None, x_max2=None, x_label2=None, x_scale2=None,
                                       title=f'Mean radius of ice particle between {latitude_selected[0]:.2f} - '
                                             f'{latitude_selected[-1]:.2f} °N',
                                       save_name=list_filename, title_option=time_selected)

        if view_mode == 2:
            print('Processing data:')
            riceco2_max_day_night(filename=filename, data=data_target)

            print('Display:')
            print('To be Done.')  # TODO

        if view_mode in [3, 301]:
            print('Processing data:')
            top_cloud = riceco2_top_cloud_altitude(filename=filename, data_target=data_target, local_time=local_time)

            print('Display:')
            if view_mode == 3:
                display_riceco2_top_cloud_altitude(filename=filename, top_cloud=top_cloud, local_time=local_time,
                                                   mola=False)
            if view_mode == 301:
                display_riceco2_top_cloud_altitude(filename=filename, top_cloud=top_cloud, local_time=local_time,
                                                   mola=True)

        if view_mode == 4:
            print('Processing data:')
            print('Nothing')
            print('Display:')
            display_vars_4figs_polar_projection(filename=filename, data_riceco2=data_target[:, :, :, :])

        if view_mode == 5:
            print('Processing data:')
            data, latitude_selected = vars_zonal_mean(filename=filename, data=data_target, flip=None, layer=None,
                                                                       local_time=local_time)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data*1e6, unit='µm', norm='log',
                                     vmin=1e-13, vmax=1e6, observation=False, cmap='inferno',
                                     latitude_selected=latitude_selected,
                                     localtime_selected=local_time,
                                     title=f'Zonal mean of mean radius of co2 ice, at {local_time}h',
                                     tes=None, mvals=None, layer=None,
                                     save_name=f'riceco2_zonal_mean_altitude_mean_equatorial_region_{local_time}h')

        if view_mode == 6:
            print('Processing data:')
            list_data = vars_zonal_mean_where_co2ice_exists(filename=filename, data=data_target, polar_region=True)

            print('Display:')
            display_riceco2_global_mean(filename, list_data)

        if view_mode == 7:
            print('Processing data:')
            data_zonal_n_time_mean_north, data_zonal_n_time_mean_south, stddev_north, stddev_south = \
                riceco2_polar_latitudes(filename=filename, data=data_target)

            print('Display:')
            display_riceco2_polar_latitudes(filename=filename, data_north=data_zonal_n_time_mean_north,
                                            data_stddev_north=stddev_north, data_south=data_zonal_n_time_mean_south,
                                            data_stddev_south=stddev_south)

        if view_mode == 8:
            latitude_selected = float(input('Select a latitude (°N): '))
            print('Processing data:')
            data_processed, data_std, latitude = riceco2_local_time_evolution(filename=filename, data=data_target,
                                                                              latitude=latitude_selected)

            print('Display:')
            display_riceco2_local_time_evolution(filename=filename, data=data_processed, data_std=data_std,
                                                 local_time=local_time, latitude=latitude)

        if view_mode == 9:
            print('Processing data:')
            data_min_radius, data_max_radius, data_mean_radius, data_mean_alt, data_std_radius, data_min_alt, \
                data_max_alt, latitude = riceco2_mean_local_time_evolution(filename=filename, data=data_target)

            print('Display:')
            display_riceco2_mean_local_time_evolution(filename=filename, data_min_radius=data_min_radius,
                                                      data_max_radius=data_max_radius, data_mean_radius=data_mean_radius,
                                                      data_std_radius=data_std_radius, data_mean_alt=data_mean_alt,
                                                      data_min_alt=data_min_alt, data_max_alt=data_max_alt,
                                                      local_time=local_time, latitude=latitude)

    elif name_target in ['satuco2']:
        print('What do you wanna do?')
        print('     1: zonal mean of saturation, for 3 latitudes, with co2ice mmr (fig: alt-ls)')
        print('     2: [TBD] saturation in localtime (fig: lt-alt')
        print('     3: saturation with co2ice mmr in polar regions, [0:30]°ls SP, [270-300]°ls NP (fig: alt-lon)')
        print('     4: Thickness atmosphere layer in polar regions (fig: thick-ls) Hu et al. 2012')
        print('     5: Saturation at 0°N along longitude (fig: alt-lon)')
        print('     6: Saturation local time - ls, at 0°N, 0.5 Pa')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_satuco2_north, data_satuco2_eq, data_satuco2_south, data_co2ice_north, data_co2ice_eq, \
            data_co2ice_south, latitude_north, latitude_eq, latitude_south, binned = \
                satuco2_zonal_mean_with_co2_ice(filename=filename, data=data_target, local_time=local_time)

            print('Display:')
            display_satuco2_with_co2_ice_altitude_ls(filename, data_satuco2_north, data_satuco2_eq, data_satuco2_south,
                                                     data_co2ice_north, data_co2ice_eq, data_co2ice_south,
                                                     latitude_north, latitude_eq, latitude_south, binned, local_time)

        elif view_mode == 2:
            print('TO be done.')

        elif view_mode == 3:
            print('Processing data:')
            data_satuco2_north, data_satuco2_south, data_co2ice_north, data_co2ice_south, latitude_north, \
            latitude_south, binned = satuco2_time_mean_with_co2_ice(filename, data_target)

            print('Display:')
            display_satuco2_with_co2_ice_altitude_longitude(filename, data_satuco2_north, data_satuco2_south,
                                                            data_co2ice_north, data_co2ice_south, latitude_north,
                                                            latitude_south, binned, local_time)

        elif view_mode == 4:
            print('Processing data:')
            data_ice_layer, data_ice_layer_std = satuco2_hu2012_fig9(filename=filename, data=data_target,
                                                                     local_time=local_time)

            print('Display:')
            if len(local_time) == 1:
                display_satuco2_thickness_atm_layer(data_ice_layer, data_ice_layer_std,
                                                    save_name=f'satuco2_thickness_polar_region_{local_time}h.png')
            else:
                display_satuco2_thickness_atm_layer(data_ice_layer, data_ice_layer_std,
                                                    save_name=f'satuco2_thickness_polar_region_diurnal_mean.png')

        elif view_mode == 5:
            print('Processing data:')
            data_processed = satuco2_altitude_longitude(filename=filename, data=data_target)

            print('Display:')
            display_vars_altitude_longitude(filename=filename, data=data_processed, unit='', norm='log', vmin=1,
                                            vcenter=1, vmax=1000,
                                            title=f'Saturation of CO2 at 0°N, mean over the year ({local_time}h)',
                                            save_name=f'satuco2_altitude_longitude_0N_{local_time}h')
        elif view_mode == 6:
            data_processed = vars_localtime_ls(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            display_co2_ice_localtime_ls(filename=filename, data=data_processed, unit='', norm='linear', vmin=1,
                                         vmax=10, title='CO2 saturation at 0°N and 0.5 Pa',
                                         save_name='satuco2_zonal_mean_localtime_ls_0N_0p5Pa')

    elif name_target in ['saturation']:
        print('What do you wanna do?')
        print('     1: profile of saturation zonal-mean in a latitude region (fig: satu-alt)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            # TODO: put this in DataProcessed
            lat1 = float(input('Select the first latitude (°N): '))
            lat2 = float(input('Select the second latitude (°N): '))
            data_latitude, list_var = get_data(filename, target='latitude')
            data_target, tmp = slice_data(data_target, dimension_data=data_latitude[:], value=[lat1, lat2])

            data_target = correction_value(data_target[:, :, :, :], operator='inf', threshold=1e-13)
            data_target = mean(data_target[:, :, :, :], axis=3)

            data_target = mean(data_target[::72, :, :], axis=0)
            print('Display:')
            print('To be done.', data_target)  # TODO

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
        print('     8: stationary wave, year mean at 0°N (fig: alt-lon)')
        print('     9: stationary wave, at 0°N and -45°E (fig: alt-ls)')
        print('    10: temperature along longitude and localtime at 0.5 Pa and 0°N (fig: hl-lon)')
        print('    11: temperature along longitude and solar longitude at 0.5 Pa and 0°N (fig: ls-lon)')
        print('    12: temperature local time - ls, at 0°N, 0.5 Pa')
        print('')

        if view_mode is None:
            view_mode = int(input('Select number:'))

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
            layer = int(input(f'\t layer (from 1 to {data_target.shape[1]}): ')) - 1

            print('Processing data:')
            data_processed, layer_selected = vars_zonal_mean(filename, data_target, layer=layer)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit='K',
                                     norm=None, vmin=100, vmax=300, observation=False, latitude_selected=None,
                                     localtime_selected=local_time, title=None, tes=None, mvals=None, layer=layer,
                                     save_name=f'temp_zonal_mean_layer{layer:d}_{layer_selected:.0e}_Pa_comparison'
                                               f'_tes_mvals')

        if view_mode == 6 or view_mode == 601:
            print('Processing data:')
            data_north, data_south = temp_thermal_structure_polar_region(filename=filename, data=data_target)

            if view_mode == 601:
                # TODO: clean this to DataProcessed
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
                filename_2 = path_2 + directory + filename_2
                data_target_2, list_var = get_data(filename_2, target=name_target)
                data_north_2, data_south_2 = temp_thermal_structure_polar_region(filename=filename_2,
                                                                                 data=data_target_2)

                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north - data_north_2,
                                                    data_south=data_south - data_south_2, norm=None,
                                                    levels=arange(-20, 20, 1),
                                                    unit='K',
                                                    save_name=f'diff_temp_zonal_mean_60NS_{directory[:-1]}')
            else:
                print('Display:')
                display_temp_structure_polar_region(filename=filename, data_north=data_north, data_south=data_south,
                                                    norm=None,
                                                    levels=arange(80, 320, 20),
                                                    unit='K',
                                                    save_name='temp_zonal_mean_60NS')

        if view_mode == 7:
            print('Processing data:')
            data_zonal_mean = temp_cold_pocket(filename=filename, data=data_target, local_time=local_time)

            print('Display:')
            display_temp_cold_pocket_spicam(filename=filename, data=data_zonal_mean, local_time=local_time,
                                            title=f'T - Tcondco2, mean between 15°S and 15°N, zonal mean, '
                                                  f'at {local_time}h',
                                            save_name=f'temp_delta_equator_zonal_mean_at_{local_time}h')

        if view_mode == 8:
            print('Processing data:')
            data_processed, diff_temp = temp_stationary_wave(filename=filename, data=data_target, local_time=local_time)

            print('Display:')
            if diff_temp:
                vmin = 0
                vmax = 120
                title = f'T - Tcondco2 at 0°N, averaged over 1 year ({local_time}h) '
                save_name = f'temp_tcondco2_altitude_longitude_0N_{local_time}h'
            else:
                print(min(data_processed), max(data_processed))
                vmin = 100
                vmax = 220
                title = f'Temperature at 0°N, averaged over 1 year ({local_time}h) '
                save_name = f'temp_altitude_longitude_0N_{local_time}h'
            display_vars_altitude_longitude(filename=filename, data=data_processed, unit='K', norm='linear', vmin=vmin,
                                            vcenter=None, vmax=vmax, title=title, save_name=save_name)

        if view_mode == 9:
            print('Processing data:')
            data_processed = vars_extract_at_grid_point(filename=filename, data=data_target, latitude=0, longitude=-45)

            print('Display:')
            vmin = 80
            vmax = 240
            print(min(data_processed), max(data_processed))
            title = f'Temperature at [0°N, -45°E] ({local_time}h) '
            save_name = f'temp_altitude_ls_0N_-45E_{local_time}h'
            display_vars_altitude_ls(filename=filename, data_1=data_processed, local_time=local_time, altitude_max=None,
                                     norm='linear', unit='K', vmin=vmin, vmax=vmax, title=title, save_name=save_name)
        elif view_mode == 10:
            print('Processing data:')
            data_processed = vars_localtime_longitude(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            print(min(data_processed), max(data_processed))
            display_vars_localtime_longitude(filename=filename, data=data_processed, norm='linear', vmin=120, vmax=165,
                                             unit='K', title=f'Temperature at 0°N and 0.5 Pa',
                                             save_name=f'temp_local_time_longitude_0N_0p5Pa')
        elif view_mode == 11:
            print('Processing data:')
            data_processed = vars_ls_longitude(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            print(min(data_processed), max(data_processed))
            display_vars_ls_longitude(filename=filename, data=data_processed, norm='linear', vmin=90, vmax=180,
                                      local_time=local_time,
                                      unit='K', title=f'Temperature at 0°N and 0.5 Pa ('
                                                      f'{local_time[0]:.0f}h)',
                                      save_name=f'temp_ls_longitude_0N_0p5Pa_{local_time[0]:.0f}h')
        elif view_mode == 12:
            data_processed = vars_localtime_ls(filename=filename, data=data_target, latitude=0, altitude=0.5)

            print('Display:')
            display_co2_ice_localtime_ls(filename=filename, data=data_processed, unit='K', norm='linear', vmin=120,
                                         vmax=160, title='Temperature at 0°N and 0.5 Pa',
                                         save_name='temp_zonal_mean_localtime_ls_0N_0p5Pa')

    # ================================================================================================================ #

    # 3-Dimension variable

    # ================================================================================================================ #
    elif name_target in ['co2ice']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig:lat-ls)')
        print('     2: cumulative masses in polar cap region, to compare with fig.10 of Hu+2012 (fig: g-ls)')
        print('     3: Polar plot every 30° ls mean, lat=60°-90° (fig: lat-ls)')
        print('     4: CO2 ice at Viking lander site along the year (fig: g-sols)')
        print('')

        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing:')
            print(data_target)
            zonal_mean, layer_selected = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='', norm=None,
                                     vmin=None, vmax=None, observation=False,
                                     latitude_selected=layer_selected, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target}, at {local_time}h', tes=None, mvals=None,
                                     layer=None,
                                     save_name=f'{name_target}_zonal_mean_{local_time}')

        if view_mode == 2:
            print('Processing data:')
            accumulation_north, accumulation_south, accumulation_precip_co2_ice_north, \
            accumulation_precip_co2_ice_south, accumulation_direct_condco2_north, \
            accumulation_direct_condco2_south = co2ice_cumulative_masses_polar_cap(filename=filename,
                                                                                   data=data_target)

            print('Display:')
            display_co2ice_cumulative_mass_polar_region(filename=filename,
                                                        data_co2_ice_north=accumulation_north,
                                                        data_co2_ice_south=accumulation_south,
                                                        data_precip_co2_ice_north=accumulation_precip_co2_ice_north,
                                                        data_precip_co2_ice_south=accumulation_precip_co2_ice_south,
                                                        data_direct_condco2_north=accumulation_direct_condco2_north,
                                                        data_direct_condco2_south=accumulation_direct_condco2_south)

        if view_mode == 3:
            print('Processing data:')
            data_mean, time_bin = co2ice_time_mean(filename=filename, data=data_target, duration=30,
                                                   localtime=local_time)

            print('Display:')
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     localtime=local_time, vmin=1e-5, vmax=1e4, norm='nonlinear',
                                                     cmap='inferno',
                                                     unit='kg/m$^2$',
                                                     title='CO$_2$ ice at the surface',
                                                     save_name=f'co2ice_15ls_mean',
                                                     levels=[1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                                                     co2_ice_cover=True)
        if view_mode == 4:
            print('Processing data:')
            data_at_vk1, data_at_vk2, data_time = co2ice_at_viking_lander_site(filename=filename, data=data_target)

            print('Display:')
            display_co2ice_at_viking_lander_site(data_at_vk1=data_at_vk1, data_at_vk2=data_at_vk2, data_time=data_time)

    elif name_target in ['h2o_ice_s']:
        print('What do you wanna do?')
        print('     1: Polar plot every 30° ls mean, lat=60°-90° (fig: lat-ls)')

        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=30, localtime=local_time)

            print('Display:')
            print(min(data_mean), max(data_mean))
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     localtime=local_time, vmin=1e-9, vmax=1e2, norm='nonlinear',
                                                     cmap='inferno',
                                                     unit='kg/m$^2$', save_name=f'h2o_ice_s_30ls_mean',
                                                     levels=[1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1,
                                                             1e2])

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
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='', norm=None,
                                     vmin=0.4, vmax=1.0, observation=False,
                                     latitude_selected=layer_selected, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target}, at {local_time}h', tes=None, mvals=None,
                                     layer=None,
                                     save_name=f'{name_target}_zonal_mean_{local_time}h')

        if view_mode == 2 or view_mode == 201:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15, localtime=local_time)

            print('Display:')
            if view_mode == 2:
                display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                         localtime=local_time,
                                                         vmin=0.55, vmax=1., norm=None, cmap='inferno', unit='',
                                                         save_name=f'emis_15ls_mean_{local_time}h')
            if view_mode == 201:
                display_emis_polar_projection_garybicas2020_figs11_12(filename=filename, data=data_mean, time=time_bin,
                                                                      levels=linspace(0.55, 1., 100), cmap='inferno',
                                                                      save_name=f'emis_15ls_mean_{local_time}h')

        if view_mode == 3:
            print('Processing data:')
            data_mean_np, data_mean_sp = emis_polar_winter_gg2020_fig13(filename=filename, data=data_target,
                                                                        local_time=local_time)

            print('Display:')
            display_vars_polar_projection(filename=filename, data_np=data_mean_np, data_sp=data_mean_sp,
                                          levels=linspace(0.55, 1., 100), unit='', cmap='inferno',
                                          sup_title='Surface emissivity mean in time during polar winter',
                                          save_name=f'emis_time_mean_gary-bicas2020_{local_time}h')

    elif name_target in ['fluxtop_lw', 'fluxtop_sw', 'fluxsurf_lw', 'fluxsurf_sw']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: ls-lat)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lat-ls)')
        print('     3: Apparent temperature (fig:ls-lat)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing:')
            zonal_mean, layer_selected = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            if name_target == 'fluxtop_lw':
                vmax = 100
            elif name_target == 'fluxsurf_lw':
                vmax = 160
            else:
                vmax = 160
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='W.m$^{-2}$',
                                     norm=None, vmin=0, vmax=vmax, observation=False,
                                     latitude_selected=layer_selected, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target}, at {local_time}h',
                                     tes=None, mvals=None, layer=None,
                                     save_name=f'{name_target}_zonal_mean_{local_time}h')

        elif view_mode == 2:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15)

            print('Display:')
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     levels=arange(0, 450, 25), norm=None, cmap='inferno',
                                                     localtime=local_time, unit='W.m$^{-2}$',
                                                     save_name=f'{name_target}_15ls_mean_')

        elif view_mode == 3:
            print('Processing:')
            zonal_mean = flux_lw_apparent_temperature_zonal_mean(data=data_target)

            print('Display:')
            print(min(zonal_mean), max(zonal_mean))
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='K',
                                     norm=None, vmin=130, vmax=220, observation=False, cmap='inferno',
                                     latitude_selected=None, localtime_selected=local_time,
                                     title=f'Zonal mean of apparent temperature, at {local_time}h',
                                     tes=None, mvals=None, layer=None,
                                     save_name=f'apparent_temperature_zonal_mean_{local_time}h')

    elif name_target in ['tau1mic']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: lat-ls)')
        print('')

        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed, tmp = vars_zonal_mean(filename=filename, data=data_target, layer=None, local_time=local_time)

            print('Display:')
            if len(local_time) > 1:
                display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit='',
                                         norm='log', vmin=1e-13, vmax=1e-1, observation=False, cmap='inferno',
                                         latitude_selected=None, localtime_selected=local_time,
                                         title=f'Zonal and diurnal mean of optical thickness at 1 micron', tes=None,
                                         mvals=None,
                                         save_name=f'tau1mic_zonal_diurnal_mean')
            else:
                display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit='',
                                     norm='log', vmin=1e-13, vmax=1e-1, observation=False, cmap='inferno',
                                     latitude_selected=None, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target} ({local_time[0]:.0f}h)', tes=None, mvals=None,
                                     save_name=f'tau1mic_zonal_mean_{local_time[0]:.0f}h')

    elif name_target in ['tau']:
        print('What do you wanna do?')
        print('     1: zonal mean (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            data_processed, tmp = vars_zonal_mean(filename=filename, data=data_target, layer=None)

            print('Display:')
            print(max(data_processed))
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit='',
                                     norm='set', vmin=0., vmax=2, observation=False, cmap='inferno',
                                     latitude_selected=None, localtime_selected=local_time,
                                     title=f'Zonal mean of {name_target}, at {int(local_time[0])} h', tes=None,
                                     mvals=None,
                                     save_name=f'tau_zonal_mean_{int(local_time[0])}h')

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
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=data_processed, unit='',
                                     norm=None, vmin=0, vmax=10, observation=False, latitude_selected=None,
                                     title=name_target, tes=None, mvals=None,
                                     save_name=f'tauTES_zonal_mean_{local_time}')

    elif name_target in ['tsurf']:
        print('What do you wanna do?')
        print('     1: Zonal mean (fig: lat-ls)')
        print('     2: Polar plot every 15° ls mean, lat=60°-90° (fig: lat-ls)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            zonal_mean, tmp = vars_zonal_mean(filename, data_target[:, :, :], layer=None, local_time=local_time)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='K', norm=None,
                                     localtime_selected=local_time,
                                     vmin=125, vmax=325, observation=False, latitude_selected=None, cmap='inferno',
                                     title='Zonal mean of surface temperature', tes=True, mvals=False, layer=None,
                                     save_name='tsurf_zonal_mean')

        if view_mode == 2:
            print('Processing data:')
            data_mean, time_bin = vars_time_mean(filename=filename, data=data_target, duration=15)

            print('Display:')
            display_vars_polar_projection_multi_plot(filename=filename, data=data_mean, time=time_bin,
                                                     levels=arange(140, 310, 10), norm=None, cmap='seismic',
                                                     unit='K', localtime=local_time,
                                                     save_name='tsurf_15ls_mean_')

    elif name_target in ['ps']:
        print('What do you wanna do?')
        print('     1: Zonal mean (fig: lat-ls)')
        print('     2: Zonal mean at viking land site (fig: pressure-sols)')
        print('')
        if view_mode is None:
            view_mode = int(input('Select number:'))

        if view_mode == 1:
            print('Processing data:')
            zonal_mean, tmp = vars_zonal_mean(filename, data_target[:, :, :], layer=None)

            print('Display:')
            display_vars_latitude_ls(filename=filename, name_target=name_target, data=zonal_mean, unit='K', norm=None,
                                     vmin=None, vmax=None, observation=False, latitude_selected=None,
                                     title=None, tes=True, mvals=True, layer=None,
                                     save_name='ps_zonal_mean')
        if view_mode == 2:
            print('Processing data:')
            data_pressure_at_viking1, latitude1, longitude1, data_pressure_at_viking2, latitude2, longitude2 = \
                ps_at_viking(filename, data_target)

            print('Display:')
            display_ps_at_viking(data_pressure_at_viking1, latitude1, longitude1, data_pressure_at_viking2, latitude2,
                                 longitude2)

    # ================================================================================================================ #

    # 1-Dimension variable

    # ================================================================================================================ #
    elif name_target in ['co2_conservation', 'Sols', 'Ls']:
        print('To be Done')  # TODO

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
    files, directory_store, filename, name_target, data_target, arg_view_mode = get_argument(*argv)

    if data_target.ndim <= 2:
        plot_sim_1d(data_target=data_target, name_target=name_target, view_mode=arg_view_mode)
    else:
        plot_sim_3d(filename=filename, data_target=data_target, name_target=name_target,
                    view_mode=arg_view_mode, directory=directory_store, files=files)
    return


if '__main__' == __name__:
    import gc

    main()
    # free memory
    a = globals().keys()
    b = [x for x in a if x[:2] != '__']
    for i, value in enumerate(b):
        del value
        gc.collect()
    exit()
