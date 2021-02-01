from math import *
import numpy as np
import tqdm
from get_csection import cross_section
import matplotlib.pyplot as plt


def get_Halo_cs_relation(M200_min=1e13,M200_max=1e15):
    Mstar = 10**11.5
    Re = 3
    z1 = .3
    z2 = 1.5
    Halo_mass = np.logspace(np.log10(M200_min),np.log10(M200_max),200)
    cs_area = np.zeros(len(Halo_mass))
    
    for i in tqdm.tqdm(range(len(Halo_mass))):
        hmass = Halo_mass[i]
        csection = cross_section(M200=hmass)
        beta_caus = csection.get_loc()[1]
        cs_area[i] = pi*beta_caus**2
    
    plt.figure(figsize=(7,5))
    plt.loglog(Halo_mass,cs_area)
    plt.title(r'$M_*=$'+"%.3e"%Mstar+r'$M_{\odot},R_e =$'+str(Re)+r'$,z_{lens}=$'+str(z1)+r'$,z_{source}=$'+str(z2))
    plt.xlabel(r'$M_h(M_{\odot})$')
    plt.ylabel(r'$\sigma_{tot}(arcsec^2)$')
    plt.savefig("./plots/Halo_mass.png")
    
    
def get_Mstar_cs_relation(Mstar_min =10**11,Mstar_max=10**11.5):
    M200 = 1e13
    Re = 3
    z1 = .3
    z2 = 1.5
    Mstar_mass = np.logspace(np.log10(Mstar_min),np.log10(Mstar_max),200)
    cs_area = np.zeros(len(Mstar_mass))
    
    for i in tqdm.tqdm(range(len(Mstar_mass))):
        Mstar = Mstar_mass[i]
        csection = cross_section(Mstar=Mstar)
        beta_caus = csection.get_loc()[1]
        cs_area[i] = pi*beta_caus**2
    
    plt.figure(figsize=(7,5))
    plt.loglog(Mstar_mass,cs_area)
    plt.title(r'$M_h=$'+"%.3e"%M200+r'$M_{\odot},R_e =$'+str(Re)+r'$,z_{lens}=$'+str(z1)+r'$,z_{source}=$'+str(z2))
    plt.xlabel(r'$M_*(M_{\odot})$')
    plt.ylabel(r'$\sigma_{tot}(arcsec^2)$')
    plt.savefig("./plots/Stellar_mass.png")
    
    
def get_Re_cs_relation(Re_min=2,Re_max=5):
    M200 = 1e13
    Mstar = 10**11.5
    z1 = .3
    z2 = 1.5
    Reff = np.linspace(Re_min,Re_max,200)
    cs_area = np.zeros(len(Reff))
    
    for i in tqdm.tqdm(range(len(Reff))):
        Re = Reff[i]
        csection = cross_section(Re=Re)
        beta_caus = csection.get_loc()[1]
        cs_area[i] = pi*beta_caus**2
    
    plt.figure(figsize=(7,5))
    plt.loglog(Reff,cs_area)
    plt.title(r'$M_h=$'+"%.3e"%M200+r'$M_{\odot},M_*=$'+"%.3e"%Mstar+r'$M_{\odot},z_{lens}=$'+str(z1)+r'$,z_{source}=$'+str(z2))
    plt.xlabel(r'$R_e(kpc)$')
    plt.ylabel(r'$\sigma_{tot}(arcsec^2)$')
    plt.savefig("./plots/Re.png")
    
    

def get_zlens_cs_relation(zlens_min=.3,zlens_max=1):
    M200 = 1e13
    Mstar = 10**11.5
    Re = 3
    z2 = 1.5

    Zlens = np.logspace(np.log10(zlens_min),np.log10(zlens_max),200)
    cs_area = np.zeros(len(Zlens))

    for i in tqdm.tqdm(range(len(Zlens))):
        zlens = Zlens[i]
        csection = cross_section(z1=zlens)
        beta_caus = csection.get_loc()[1]
        cs_area[i] = pi*beta_caus**2

    plt.figure(figsize=(7,5))
    plt.loglog(Zlens,cs_area)
    plt.title(r'$M_h=$'+"%.3e"%M200+r'$M_{\odot},M_*=$'+"%.3e"%Mstar+r'$M_{\odot},R_e =$'+str(Re)+r'$,z_{source}=$'+str(z2))
    plt.xlabel(r'$z_{lens}$')
    plt.ylabel(r'$\sigma_{tot}(arcsec^2)$')
    plt.savefig("./plots/zlens.png")
    
    
def get_zsource_cs_relation(zsource_min=1.5,zsource_max=4):
    M200 = 1e13
    Mstar = 10**11.5
    Re = 3
    z1 = .3

    Z_source = np.logspace(np.log10(zsource_min),np.log10(zsource_max),200)
    cs_area = np.zeros(len(Z_source))

    for i in tqdm.tqdm(range(len(Z_source))):
        z2 = Z_source[i]
        csection = cross_section(z2=z2)
        beta_caus = csection.get_loc()[1]
        cs_area[i] = pi*beta_caus**2

    plt.figure(figsize=(7,5))
    plt.loglog(Z_source,cs_area)
    plt.title(r'$M_h=$'+"%.3e"%M200+r'$M_{\odot},M_*=$'+"%.3e"%Mstar+r'$M_{\odot},R_e =$'+str(Re)+r'$,z_{lens}=$'+str(z1))
    plt.xlabel(r'$z_{source}$')
    plt.ylabel(r'$\sigma_{tot}(arcsec^2)$')
    plt.savefig("./plots/zsource.png")

    
if __name__=="__main__":
    get_Halo_cs_relation()
    get_Mstar_cs_relation()
    get_Re_cs_relation()
    get_zlens_cs_relation()
    get_zsource_cs_relation()
    
    