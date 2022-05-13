from math import pi, exp, sqrt, log, erf
from numpy import log, zeros, linspace, arange, exp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def compute_nucleation(pco2, temperature_cell, saturation, vo2co2, nbin_cldco2, ccn_radius):
    rad_cldco2, rb_cldco2 = compute_nucleation_grid(nbin_cldco2=nbin_cldco2)

    Rn = -log(ccn_radius)
    n_aer = zeros(nbin_cldco2)
    n_derf = erf((rb_cldco2[0] + Rn) * dev2)
    for i in range(nbin_cldco2):
        n_aer[i] = -0.5 * ccn_number * n_derf
        n_derf = erf((rb_cldco2[i+1] + Rn) * dev2)
        n_aer[i] = n_aer[i] + 0.5 * ccn_number * n_derf

    rate = compute_nucleation_rate(pco2=pco2, temperature=temperature_cell, saturation=saturation, n_aer=n_aer,
                                   vo2co2=vo2co2, teta=mtetaco2, rad_cldco2=rad_cldco2, nbin_cldco2=nbin_cldco2)

    dN = 0.
    for i in range(nbin_cldco2):
        Proba = 1.0 - exp(-1. * microtimestep * rate[i])
        dN = dN + n_aer[i] * Proba

    return dN, n_aer


def compute_nucleation_grid(nbin_cldco2):
    rmin_cldco2 = 1e-9  # 1e-9  # h2o = 0.1e-6
    rmax_cldco2 = 5e-6  # 5e-6  # h2o = 10e-6
    rad_cldco2 = zeros(nbin_cldco2)

    rbmin_cldco2 = 1e-11#1e-11  # 0.0001e-6
    rbmax_cldco2 = 2e-4#2.e-4  # 1e-2
    rb_cldco2 = zeros(nbin_cldco2+1)

    vrat_cldco2 = exp(log(rmax_cldco2 / rmin_cldco2) / (nbin_cldco2 - 1) * 3.)

    rad_cldco2[0] = rmin_cldco2
    for i in range(nbin_cldco2 - 1):
        rad_cldco2[i + 1] = rad_cldco2[i] * vrat_cldco2 ** (1. / 3.)

    rb_cldco2[0] = rbmin_cldco2
    for i in range(nbin_cldco2):
        rb_cldco2[i + 1] = ((2. * vrat_cldco2) / (vrat_cldco2 + 1.)) ** (1. / 3.) * rad_cldco2[i]
    rb_cldco2[nbin_cldco2] = rbmax_cldco2
    rb_cldco2 = log(rb_cldco2)
    return rad_cldco2, rb_cldco2


def compute_nucleation_rate(pco2, temperature, saturation, n_aer, vo2co2, teta, rad_cldco2, nbin_cldco2):
    # **************************************************************************************************************** #
    #    This subroutine computes the nucleation rate as given in Pruppacher & Klett (1978) in the case of water ice
    #    forming on a solid substrate.
    #    It computes two different nucleation rates : one on the dust CCN distribution and the other one on the water
    #    ice particles distribution
    # **************************************************************************************************************** #
    # module nucleaco2_mod
    # subroutine nucleaco2(pco2,temp,sat,n_ccn,nucrate,vo2co2, teta)
    # --------------------------------------------------------------
    # rstar: Radius of the critical germ (m)
    # gstar: of molecules forming a critical embryo
    # fistar Activation energy required to form a critical embryo (J)
    # **************************************************************************************************************** #

    nco2 = pco2 / kbz / temperature
    rstar = 2. * sigco2 * vo2co2 / (kbz * temperature * log(saturation))
    gstar = 4. * pi * (rstar * rstar * rstar) / (3. * vo2co2)

    fshapeco2simple = (2.+teta) * (1.-teta) * (1.-teta) / 4.

    nucrate = zeros(nbin_cldco2)
    # Loop over size bins
    for i in range(nbin_cldco2):
        if n_aer[i] <= 1e-10:
            # no dust, no need to compute nucleation!
            nucrate[i] = 0.
        else:
            if rad_cldco2[i] >= 3000.*rstar:
                zefshapeco2 = fshapeco2simple
            else:
                zefshapeco2 = fshapeco2(teta, rad_cldco2[i]/rstar)

            fistar = (4./3.*pi) * sigco2 * (rstar * rstar) * zefshapeco2
            deltaf = (2.*desorpco2-surfdifco2-fistar) / (kbz * temperature)
            deltaf = min(max(deltaf, -100), 100)

            if deltaf == -100:
                nucrate[i] = 0.
            else:
                nucrate[i] = sqrt(fistar/(3. * pi * kbz * temperature * (gstar * gstar))) * kbz * temperature * \
                             rstar * rstar * 4. * pi * (nco2*rad_cldco2[i]) * (nco2*rad_cldco2[i]) / (
                                     zefshapeco2 * nusco2 * m0co2) * exp(deltaf)
    return nucrate


def compute_saturation(temperature):
    pressure_saturation = 1.382 * 1e12 * exp(-3182.48 / temperature)
    return pressure_saturation


def compute_co2_ice_density(temperature):
    density = 1000. * (1.72391 - 2.53e-4 * temperature - 2.87e-6 * temperature * temperature)
    return density


def plot_result(data, vector_x, vector_y, title, nbin, xscale, yscale, xlabel, ylabel, savename):
    levels = MaxNLocator(nbins=10).tick_values(0, 100)
    cmap = plt.get_cmap('jet')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cmap.set_under(color='white')

    fig, ax = plt.subplots(figsize=(11, 11))
    pcm = ax.pcolormesh(vector_x, vector_y, data.T*100., shading='flat', norm=norm,
                        cmap=cmap)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = plt.colorbar(pcm)
    cb.ax.set_title('%')

    if nbin > 0:
        rad_cldco2, rb_cldco2 = compute_nucleation_grid(nbin_cldco2=nbin)
        ax.vlines(rad_cldco2, vector_y[0], vector_y[-1], ls='--', colors='black', linewidth=2)
        ax.vlines(exp(rb_cldco2), vector_y[0], vector_y[-1], ls='-', colors='black', linewidth=2)
    ax.set_xlim(vector_x[0], vector_x[-1])
    plt.savefig(f'{savename}.png', bbox_inches='tight')


# ******************************************************************************************************************** #
#  double precision function fshapeco2(cost,rap)
#  function computing the f(m,x) factor related to energy required to form a critical embryo  *
# ******************************************************************************************************************** #
def fshapeco2(cost, rap):
    yeah = sqrt(1. - 2. * cost * rap + rap ** 2)
    # !! FSHAPECO2 = TERM A
    fshapeco2 = (1. - cost * rap) / yeah
    fshapeco2 = fshapeco2 ** 3
    fshapeco2 = 1. + fshapeco2
    # !! ... + TERM B
    yeah = (rap - cost) / yeah
    fshapeco2 = fshapeco2 + rap * rap * rap * (2. - 3. * yeah + yeah * yeah * yeah)
    # !! ... + TERM C
    fshapeco2 = fshapeco2 + 3. * cost * rap * rap * (yeah - 1.)
    # !! FACTOR 1/2
    fshapeco2 = 0.5 * fshapeco2
    return fshapeco2


def main(pression_cell, temperature_cell):
    tab_nbin = arange(1, 100)
    tab_saturation = arange(1, 50)
    distribution_ccn = zeros((tab_nbin.shape[0], tab_saturation.shape[0]))
    radius = 1e-9

    # pco2: the partial pressure of co2 vapor and its saturation ratio
    A = (1./m_co2 - 1./m_noco2)
    B = 1./m_noco2
    mmean = 1. / (A*qco2 + B)
    pco2 = qco2 * (mmean / (m_co2 * 1e3)) * pression_cell


    # T-dependant CO2 ice density
    density = compute_co2_ice_density(temperature=temperature_cell)
    vo2co2 = m0co2 / density

    # Saturation
    pressure_saturation = compute_saturation(temperature=temperature_cell)
    saturation = pco2 / pressure_saturation
    for i, value_i in enumerate(tab_nbin):
        for j, value_j in enumerate(tab_saturation):
            pco2 = pressure_saturation * value_j
            distribution_ccn[i, j], tmp = compute_nucleation(pco2=pco2, temperature_cell=temperature_cell,
                                                             saturation=value_j, vo2co2=vo2co2, nbin_cldco2=value_i,
                                                             ccn_radius=radius)
    distribution_ccn = distribution_ccn / ccn_number

    # plot result
    plot_result(data=distribution_ccn, vector_x=tab_nbin, vector_y=tab_saturation, xscale='linear', yscale='linear',
                xlabel='nbin', ylabel='Saturation', nbin=0,
                title=f'Percentage of CN used during nucleation \n'
                      f'Temperature = {temperature_cell:.2f}, r = {radius:.2e} m',
                savename=f'figure_nbin_cldco2_saturation_at_temperature_{temperature_cell:.0f}_radius_{radius:.0e}')




    # Saturation
    nbin = 10
    tab_radius = linspace(1e-9, 5e-6, 100)
    distribution_ccn = zeros((tab_radius.shape[0], tab_saturation.shape[0]))
    pressure_saturation = compute_saturation(temperature=temperature_cell)
    for i, value_i in enumerate(tab_radius):
        for j, value_j in enumerate(tab_saturation):
            pco2 = pressure_saturation * value_j
            distribution_ccn[i, j], tmp = compute_nucleation(pco2=pco2, temperature_cell=temperature_cell,
                                                             saturation=value_j, vo2co2=vo2co2, nbin_cldco2=nbin,
                                                             ccn_radius=value_i)
    distribution_ccn = distribution_ccn / ccn_number

    # plot result
    plot_result(data=distribution_ccn, vector_x=tab_radius, vector_y=tab_saturation, xscale='log', yscale='linear',
                xlabel='radius (µm)', ylabel='saturation', nbin=nbin,
                title=f'Percentage of CN used during nucleation \n'
                      f'Temperature = {temperature_cell:.2f}, nbin = {nbin:.2e} m',
                savename=f'figure_radius_saturation_at_temperature_{temperature_cell:.0f}_nbin_{nbin:.0e}')




#    tab_nbin = arange(2, 101)
#    tab_radius = linspace(1e-9, 5e-6, 100)  # m
#    distribution_ccn = zeros((tab_nbin.shape[0], tab_radius.shape[0]))
#    for i, value_i in enumerate(tab_nbin):
#        for j, value_j in enumerate(tab_radius):
#            distribution_ccn[i, j], tmp = compute_nucleation(pco2=pco2, temperature_cell=temperature_cell,
#                                                             saturation=saturation, vo2co2=vo2co2, nbin_cldco2=value_i,
#                                                             ccn_radius=value_j)
#    distribution_ccn = distribution_ccn / ccn_number
#
#    # plot result
#    plot_result(data=distribution_ccn, vector_x=tab_nbin, vector_y=tab_radius*1e6, yscale='log',
#                xlabel='nbin_cldco2', ylabel='radius (µm)',
#                title=f'Percentage of CN used during nucleation \n'
#                      f' saturation = {saturation:.2f}, P = {pression_cell:.2e} Pa, T = {temperature_cell:.2e} K',
#                savename=f'figure_nbincldco2_radius_at_sat_{int(saturation)}_pressure_'
#                         f'{pression_cell:.0e}_temperature_{int(temperature_cell)}')
#
#    # New view
#    tab_temperature = arange(30, 140)
#    tab_saturation = arange(1, 100)  # m
#    distribution_ccn = zeros((tab_temperature.shape[0], tab_saturation.shape[0]))
#    radius = 1e-9
#    nbin_cldco2 = 5
#    for i, value_i in enumerate(tab_temperature):
#        for j, value_j in enumerate(tab_saturation):
#            distribution_ccn[i, j], tmp = compute_nucleation(pco2=pco2, temperature_cell=value_i,
#                                                             saturation=value_j, vo2co2=vo2co2,
        #                                                             nbin_cldco2=nbin_cldco2,
#                                                             ccn_radius=radius)
#    distribution_ccn = distribution_ccn / ccn_number
#
#    # plot result
#    plot_result(data=distribution_ccn, vector_x=tab_temperature, vector_y=tab_saturation, yscale='linear',
#                xlabel='Temperature (K)', ylabel='Saturation',
#                title=f'Percentage of CN used during nucleation \n'
#                      f' nbin_cldco2 = {nbin_cldco2:.2f}, r = {radius:.2e} m',
#                savename=f'figure_temperature_saturation_at_nbincldco2_{nbin_cldco2}_radius_{radius:.0e}')


if '__main__' == __name__:
    # Boltzman constant
    kbz = 1.381e-23
    # Surface tension of ice/vapor (J.m-2)
    sigco2 = 0.08
    # Activation energy  for desorption of water on a dust-like substrate (J.molecule-1)
    desorpco2 = 3.07e-20
    # Estimated activation energy for surface diffusion of co2 molecules (J.molecule-1)
    surfdifco2 = desorpco2 / 10.
    # Jump frequency of a co2 molecule(s-1)
    nusco2 = 2.9e+12
    # Weight of a co2 molecule(kg)
    m0co2 = 44.e-3 / 6.023e+23
    # Contact parameter(m=cos(theta)) bachnar et al. 2016, value: 0.78
    mtetaco2 = 0.95
    # CO2 molecular mass (kg.mol-1)
    m_co2 = 44.01E-3
    # non condensible molecular mass(kg.mol-1)
    m_noco2 = 33.37E-3
    # Effective variance for sedimentation for the log-normal distribution of CO2 clouds particles
    nuiceco2_sed = 0.2
    # Microphysical time step (s)
    microtimestep = 18

    sigma_iceco2 = sqrt(log(1.+nuiceco2_sed))
    dev2 = 1. / (sqrt(2.) * sigma_iceco2)

    qco2 = 0.99
    ccn_number = 1e15

    # Pressure in Pa, temperature in K
    main(pression_cell=0.05, temperature_cell=80)
