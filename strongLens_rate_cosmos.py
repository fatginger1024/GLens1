import sys, os
from math import *
import numpy as np
from astropy.table import Table

from get_csection_wmag import cross_section
from gnfwLens import lens_stat_gnfw
from get_concentration import concentration

from multiprocessing import Process, Queue, Value, Array
import time, os, random

# Read from MICE catalogue for the lenses

dirbase =  '../GLens'

data = np.fromfile(os.path.join(dirbase,'8560.bin')).reshape(3,-1)
dat = {}
dat['z_cgal']=data[0,:]
dat['lmstellar']=data[1,:]
dat['lmhalo']=data[2,:]


# total area of the sky coverage is 5000 deg^2
tot_area = 5000*(3600)**2 # arcsec^2


# read the CCOSMOS 2015 data
# with 2 deg^2 coverage
COSMOS2015 = Table.read(os.path.join(dirbase,'COSMOS2015_Laigle+_v1.1.fits'),format='fits')
r_mag = COSMOS2015['r_MAG_APER2']
redshift = COSMOS2015['ZPDF'] 
mask = (redshift>=0)  *  (r_mag>0) 

Nct_data = np.fromfile(os.path.join(dirbase,'COSMOS_Nct.bin'))
num = 50
# rescale to MICE2 sky coverage
Nct = Nct_data.reshape(50,-1)[:,0] * 5e3
z_bin = Nct_data.reshape(50,-1)[:,1]
bin_loc = np.fromfile(os.path.join(dirbase,'bin_loc.bin'),dtype=np.float32)

def mu(Mstar,mu0R=.855,betaR=1.366):
    
    return mu0R+betaR*(np.log10(Mstar)-11.4)
    
def get_Re_from_Mstar(Mstar,sigmaR=.147):
    loc = mu(Mstar)
    logRe =  np.random.normal(loc=loc,scale=sigmaR,size=1)
    return 10**logRe

def run_one(i,h=.7,ratio=1, cratio=1,alpha=1):
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
            csection = lens_stat_gnfw(z1=zlens,z2=z2,M200=10**Mh_lens*h,Mstar=10**Mstar_lens*h,alpha=alpha,
                                     Re=scale_rad,c=conc,cratio=cratio,ratio=ratio,source_mag=source_rmag)
            #beta_caus = csection.get_loc()[1]
            beta_caus = csection.beta_mag
            cross_sec = pi*beta_caus**2
            P = cross_sec/tot_area * Nct[ind[j]]
            Psum.append(P)
        except ValueError:
            print(zlens,z2)
    return np.sum(Psum)


def qinit(q, index):
    for i in index:
        q.put(i)

def moving_average(arr, count):
    return np.sum(arr[:])/count

def f(q, count, arr, ratio=1., cratio=1.,alpha=1.):
    while not q.empty():
        ind = q.get()
        P = run_one(ind,h=.7,ratio=ratio, cratio=cratio, alpha=alpha)
        arr[ind] = P
        count.value += 1
        num = count.value
        ma = moving_average(arr, num)
        print("pid({}):\t".format(os.getpid()),
              num,
              P,
              ma)

if __name__ == '__main__':
    parr = [0.4,0.8,1,1.2,1.6,2]
    parc = [0.4,0.8,1,1.2,1.6,2]
    alph = np.arange(.4,3,.4)[::-1]
    num_proc = 4
    np.random.seed(1234)
    num_data = len(dat['z_cgal'])
    index = np.arange(num_data)
    np.random.shuffle(index)
    # alpha:
    '''
    for ratio in parr:
        process_list = []
        q = Queue(len(index))
        num = Value('i', 0)
        arr = Array('d', np.zeros(num_data))           
        qinit(q, index)
        for i in range(num_proc):
            p = Process(target=f, args=(q,num,arr,ratio,1.))
            p.start()
            process_list.append(p)
        for i in process_list:
            p.join()
    '''
    for alpha in alph:
        process_list = []
        q = Queue(len(index))
        num = Value('i', 0)
        arr = Array('d', np.zeros(num_data))           
        qinit(q, index)
        for i in range(num_proc):
            p = Process(target=f, args=(q,num,arr,1,1.,alpha))
            p.start()
            process_list.append(p)
        for i in process_list:
            p.join() 
        fp = 'output/r{}_cr{}_alpha{}.txt'.format(1,1,alpha)
        time.sleep(60)
        np.savetxt(fp,arr[:])
