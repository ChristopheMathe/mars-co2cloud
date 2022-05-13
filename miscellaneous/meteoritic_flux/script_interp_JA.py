#!/bin/bash python3
#"from Soft import spawn
import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import re
from matplotlib.colors import LogNorm

# Ouverture et traitement fichier John Plane
#fic=spawn('ls *txt')
#fic=fic[0]
fic='Mars_dust.txt'
fichier=open(fic,'r')
lines=fichier.readlines()
nb_lines=len(lines)
res=np.empty([nb_lines,3])
for s in enumerate(lines):
    res[s[0],:]=s[1].replace('\n','').rstrip().split('\t')
res[:,2]=res[:,2]*0.1
#res[:,0]=altitudes # [:,1]=particle sizes #[:,2]= particle number

#Individualisation des bin size du fichier de John:
[a1,a2]=[res[:,1].argmin(),res[:,1].argmax()]
bins=res[a1:a2+1,1]
nbins=a2-a1+1
nalt=int(len(res)/nbins)
newone=np.empty([nalt,nbins])
for i in range(nalt):
    newone[i,:]=res[i*nbins:(i+1)*nbins,2] #newone est le tableau

#Interpol on the bins of the GCM size distribution
#[rad,radius]=distri() #this gets the GCM distribution
rmin_cld = 1e-9
rmax_cld = 5e-6
#rmin_cld = 1e-11
#rmax_cld = 3e-6
nbinco2_cld = 100

vrat_cld = np.exp(np.log(rmax_cld/rmin_cld) / float(nbinco2_cld-1) * 3.)
rad = np.zeros(nbinco2_cld)
rad[0] = rmin_cld
for i in range(1, nbinco2_cld):
	rad[i] = rad[i-1] * vrat_cld**(1./3.)

togcm=np.empty([nalt,len(rad)])
UNDEF=0.
for i in range(nalt):
    togcm[i,:]=np.interp(rad,bins,newone[i,:],right=UNDEF,left=UNDEF)

# We now have a 2D table referenced to the gcm bin sizes and altitudes
# mpl.plot(rad,togcm[0,:],'r',res[0:19,1],res[0:19,2],'b^')
# mpl.plot(rad,togcm[85,:],'r.',bins,newone[85,:],'b')
print(togcm.shape, newone.shape, res.shape, rad.shape, bins.shape)
mpl.pcolormesh(rad, np.arange(nalt), togcm[:,:], norm=LogNorm())
mpl.xscale('log')
mpl.savefig('joachim_1e-9_5e-6.png')
mpl.show()

