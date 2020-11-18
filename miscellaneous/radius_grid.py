#!/bin/bash python3
# This simple program compute the radius of CCN particule in the co2cloud.F90 program (see 0.2).
from numpy import exp, log, zeros, where


def main():
    rmax_cld = 5.e-6  # (m)
    rmin_cld = 1.e-9  # (m)
    rbmin_cld = 1.e-10  # (m)
    rbmax_cld = 2.e-4  # (m)
    nbinco2_cld = 100

    # vrat_cld is determined from the boundary values of the size grid: rmin_cld and rmax_cld.
    vrat_cld = exp(log(rmax_cld / rmin_cld) / float(nbinco2_cld - 1) * 3.)

    # rad_cldco2 is the primary radius grid used for microphysics computation.
    rad_cldco2 = zeros(nbinco2_cld)
    rad_cldco2[0] = rmin_cld
    for i in range(nbinco2_cld - 1):
        rad_cldco2[i + 1] = rad_cldco2[i] * vrat_cld ** (1. / 3.)

    # rb_cldco2 array contains the boundary values of each rad_cldco2 bin.
    rb_cldco2 = zeros(nbinco2_cld+1)
    rb_cldco2[0] = rbmin_cld
    for i in range(nbinco2_cld):
        rb_cldco2[i + 1] = ((2. * vrat_cld) / (vrat_cld + 1.)) ** (1. / 3.) * rad_cldco2[i]
    rb_cldco2[-1] = rbmax_cld

    print('Bot bound | primary radius | Top bound')
    for i in range(nbinco2_cld):
        print('{:.2e}  <    {:.2e}    < {:.2e}'.format(rb_cldco2[i], rad_cldco2[i], rb_cldco2[i+1]))

    print('{:d} points are below 100 nm radius (54 with rmax=5µm and 100 nbin).'.format(where(rad_cldco2 < 100e-9)[0].shape[0]))


if '__main__' == __name__:
    main()
