import matplotlib.pyplot as plt
from numpy import logspace, arange, zeros, min, max, log
from matplotlib.colors import LogNorm, DivergingNorm, BoundaryNorm


def compute_opactiy(masse_co2ice, radius_co2ice, rho_co2ice):
    if radius_co2ice > 25e-6:
        coef_extinction = 2
    elif 10e-6 < radius_co2ice < 25e-6:
        coef_extinction = 4
    else:
        coef_extinction = 0

    opactiy = (3. * coef_extinction * masse_co2ice) / (4. * rho_co2ice * radius_co2ice)

    return opactiy


def compute_emissivity(alpha, opacity):
    emissivity = (1. + (alpha * opacity))**(-1./3.)
    return emissivity


# TODO afficher les valeurs du fichier optique de CO2 du GCM
def compute_variation_emissivity(opacity):
    # Metamorphism
    dtemisice = 0.4
    daysec = 88775.

    # opacity
    pemisurf = 0.95
    ptimestep = 924.739583333333
    emisref = 0.95

    zdemisurf = (emisref - pemisurf) / (dtemisice * daysec) + \
                (emisref * ((pemisurf / emisref) **(-3) + 3. * opacity * ptimestep) ** (-1 / 3.) -
                 pemisurf) / ptimestep
    return zdemisurf * ptimestep


def compute_variation_emissivity_old(masse_co2_ice, coef):
    iceradius = 100e-6
    # Metamorphism
    dtemisice = 0.4
    daysec = 88775.

    # opacity
    alpha = 0.45
    Kscat = coef * alpha / iceradius
    pemisurf = 0.95
    ptimestep = 924.739583333333
    emisref = 0.95

    zdemisurf = (emisref - pemisurf) / (dtemisice * daysec) + \
                (emisref * ((pemisurf / emisref) ** (-3) + 3. * Kscat * masse_co2_ice * ptimestep) ** (-1 / 3.) -
                 pemisurf) / ptimestep
    return zdemisurf * ptimestep


def main():
    masse_co2ice = 1.142040030011300E-008 # kg.m-2
    rho_co2ice = 1621.90715479835 # 1630  # kg.m-3

    tab_radius = logspace(-8, -3, 100)  # m
    tab_alpha = arange(0.15, 1.5, 0.05)
    emissivity = zeros(shape=(tab_alpha.shape[0], tab_radius.shape[0]))

    for r, radius in enumerate(tab_radius):
        opacity = compute_opactiy(masse_co2ice=masse_co2ice, radius_co2ice=radius, rho_co2ice=rho_co2ice)
        for a, alpha in enumerate(tab_alpha):
            emissivity[a, r] = compute_variation_emissivity(opacity=opacity)

    emissivity_old = compute_variation_emissivity_old(masse_co2_ice=masse_co2ice, coef=(0.001/3.))
    emissivity_old_2 = compute_variation_emissivity_old(masse_co2_ice=masse_co2ice, coef=(3./(rho_co2ice)))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_title(f'emis_old = {-emissivity_old:.2e}, {-emissivity_old_2:.2e}')
    pcm = ax[0].pcolormesh(tab_radius, tab_alpha, -emissivity, norm=LogNorm(vmin=1e-5, vmax=1e-2),
                           cmap='inferno')
    plt.colorbar(pcm, ax=ax[0])

    ax[1].set_title(f'emis - emis_old')
    pcm = ax[1].pcolormesh(tab_radius, tab_alpha, -emissivity + emissivity_old, norm=LogNorm(),
                           cmap='coolwarm')
    plt.colorbar(pcm, ax=ax[1])

    for axes in ax.reshape(-1):
        axes.scatter(100e-6, 0.45, c='red')
        axes.set_xscale('log')
        axes.set_xlim(1e-5, 1e-3)
        axes.set_xlabel('CO2 ice radius (µm)')
        axes.set_yscale('linear')
        axes.set_ylabel('alpha')

    plt.show()


if '__main__' == __name__:
    main()
