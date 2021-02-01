import tqdm
import sys
from math import *
import numpy as np

from Distances import Distances
from get_csection import cross_section

# Read from MICE catalogue for the lenses

size = int(sys.argv[1])
rank = int(sys.argv[2])

print("rank %d init..."%(rank))
data = np.fromfile('./8560.bin').reshape(3,-1)
dat = {}
dat['z_cgal']=data[0,:]
dat['lmstellar']=data[1,:]
dat['lmhalo']=data[2,:]

# Read the number density of source galaxies 

Nct_data = np.fromfile('Ncounts.bin')
Num = 59
redshift = Nct_data.reshape(Num,2)[:,0]
Ncounts = Nct_data.reshape(Num,2)[:,1]
dtheta = np.sqrt(5e3)/180*pi
dA = np.zeros(len(redshift))
for i in range(len(redshift)):
    d  = Distances()
    dA[i] = d.angular_diameter_distance(0,redshift[i])
Area = (dA*dtheta)**2
dNdz = Ncounts*Area
mask = dNdz != 0

# Compute for each lens, the strong lensing event rate
# total area of the sky coverage is 5000 deg^2
tot_area = 5000*(3600)**2 # arcsec^2

def main(size, rank):
    Ntot = np.zeros(len(dat['z_cgal']))
    sample_i = np.array_split(np.arange(len(dat['z_cgal'])),size)[rank]
    #print(rank, len(sample_i))
    for i in sample_i:
        if rank == 0:
            print(i)
        #print("rank{}:  {}".format(rank, i))
        d = Distances()
        zlens = np.array(dat['z_cgal'])[i]
        Mh_lens = np.array(dat['lmhalo'])[i]
        Mstar_lens = np.array(dat['lmstellar'])[i]
        ind = np.where(redshift[mask]>zlens+.02)[0]
        Nsum = []
        for j in range(len(ind)):
            try:
                z2 = redshift[mask][ind[j]]
                csection = cross_section(z1=zlens,z2=z2,M200=10**Mh_lens,Mstar=10**Mstar_lens)
                beta_caus = csection.get_loc()[1]
                cross_sec = pi*beta_caus**2
                P = cross_sec/tot_area 
                dDa = d.angular_diameter_distance(redshift[mask][ind[j]-1],redshift[mask][ind[j]])
                Nsource = dNdz[mask][ind[j]] * dDa
                Nsum.append(Nsource * P)
            except ValueError:
                print(zlens,z2)
            
            
        Ntot[i] = np.sum(Nsum)
    np.savetxt('ntot_{}.txt'.format(rank),Ntot)
        
main(size=size,rank=rank)
