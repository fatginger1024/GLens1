import tqdm
import sys
from math import *
import numpy as np
from astropy.table import Table

from Distances import Distances
from get_csection_wmag import cross_section
from get_concentration import concentration


# Read from MICE catalogue for the lenses

size = int(sys.argv[1])
rank = int(sys.argv[2])

print("rank %d init..."%(rank))
data = np.fromfile('./8560.bin').reshape(3,-1)
dat = {}
dat['z_cgal']=data[0,:]
dat['lmstellar']=data[1,:]
dat['lmhalo']=data[2,:]


# total area of the sky coverage is 5000 deg^2
tot_area = 5000*(3600)**2 # arcsec^2


# read the CCOSMOS 2015 data
# with 2 deg^2 coverage
COSMOS2015 = Table.read('/data/inspur_disk01/userdir/maotx/qing/COSMOS2015_Laigle+_v1.1.fits',format='fits')
r_mag = COSMOS2015['r_MAG_APER2']
redshift = COSMOS2015['ZPDF'] 
mask = (redshift>=0)  *  (r_mag>0) 

Nct_data = np.fromfile('COSMOS_Nct.bin')
num = 50
# rescale to MICE2 sky coverage
Nct = Nct_data.reshape(50,-1)[:,0] * 5e3
z_bin = Nct_data.reshape(50,-1)[:,1]
bin_loc = np.fromfile('bin_loc.bin',dtype=np.float32)

def mu(Mstar,mu0R=.855,betaR=1.366):
    
    return mu0R+betaR*(np.log10(Mstar)-11.4)
    
def get_Re_from_Mstar(Mstar,sigmaR=.147):
    loc = mu(Mstar)
    logRe =  np.random.normal(loc=loc,scale=sigmaR,size=1)
    return 10**logRe

def main(size, rank,h=.7,ratio=1):
    Ptot = np.zeros(len(dat['z_cgal']))
    sample_i = np.array_split(np.arange(len(dat['z_cgal'])),size)[rank]
    print(rank, len(sample_i))
    for i in sample_i:
        if rank == 0:
            print(i)
        #print("rank{}:  {}".format(rank, i))
        d = Distances()
        zlens = np.array(dat['z_cgal'])[i]
        Mh_lens = np.array(dat['lmhalo'])[i]
        Mstar_lens = np.array(dat['lmstellar'])[i]
        ind = np.where(z_bin>zlens+.02)[0]
        Psum = []
        for j in range(len(ind)):
            try:
                z2 = z_bin[ind[j]]
                rmask = (redshift[mask]<bin_loc[ind[j]+1]) * (redshift[mask]>bin_loc[ind[j]])
                rmag_pool = np.random.choice(r_mag[mask][rmask],size=1)
                source_rmag = rmag_pool
                scale_rad =  get_Re_from_Mstar(10**Mstar_lens*h,sigmaR=.147)
                conc = concentration(10**Mh_lens,zlens)
                csection = cross_section(z1=zlens,z2=z2,M200=10**Mh_lens*h,Mstar=10**Mstar_lens*h,
                                         Re=scale_rad,c=conc,ratio=ratio,source_mag=source_rmag)
                beta_caus = csection.get_loc()[1]
                cross_sec = pi*beta_caus**2
                P = cross_sec/tot_area * Nct[ind[j]]
                Psum.append(P)
                

            except ValueError:
                print(zlens,z2)
            
            
        Ptot[i] = np.sum(Psum)
        #avg_P[i] = np.trim_zeros(Ptot).mean()
    np.savetxt('output/prob_{}.txt'.format(rank),Ptot)
        
main(size=size,rank=rank)
