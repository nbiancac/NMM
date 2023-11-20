#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""
import sys
sys.path.append('../../src/')

import nmm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close('all')
saveDir = './tests/cavity_with_current/'

beam = nmm.beam() # initialize beam parameters
geometry = nmm.geometry(L = 0.01)
materials = nmm.materials(sigma = 0)
Np = 50
mesh = nmm.mesh(geometry, Np=Np)
P_max = 15
S_max = 5
R_max = 25
        
sim = nmm.simulation(index_max_p=P_max, index_max_r=R_max, index_max_s = S_max, Np = Np, \
                        geometry = geometry, materials = materials, beam = beam, mesh = mesh)

sim.integration='direct'
        
Zout = []
fout = []
sim.preload_matrixes()
for f in np.linspace(1e8, 6e9, 50, endpoint=True)[:]:
    print(f"frequency {f/1e9} GHz")
    sim.f = f
    sim.compute_impedance()
    
    Zout.append(sim.Z)
    fout.append(sim.f)
    
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()

savestr = f'NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}'
pd.DataFrame(index = fout, data = {'Re': Zout.real, 'Im': Zout.imag}).to_csv(saveDir+savestr+'.csv')  

#%% Just MMT

saveDir = './tests/cavity_with_current/'    
import pandas as pd
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.ylim(-60,60)
plt.xlim(0, 6)
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir+'MMT_cavity_b0.05_L0.01.png')

#%% MMT and NMM with source fields
plt.figure()
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')


df = pd.read_csv(saveDir+'/NMM_fields_cavity_b0.05_L0.01.txt', skiprows=0, index_col=0)

plt.plot(df.index/1e9, df.Re.values,'ob', label = 'Re, NMM (source fields)')
plt.plot(df.index/1e9, df.Im.values, 'om', label = 'Im, NMM (source fields)')
plt.ylim(-60,60)
plt.xlim(0,6)
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir+'NMM_fields_cavity_b0.05_L0.01_with_Im.png')
   
#%% MMT and NMM with source current, direct integration

plt.figure()
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')

df = pd.read_csv(saveDir+'/NMM_fields_cavity_b0.05_L0.01.txt', skiprows=0, index_col=0)
plt.plot(df.index/1e9, df.Re.values,'ob', label = 'Re, NMM (source fields)')
plt.plot(df.index/1e9, df.Im.values, 'om', label = 'Im, NMM (source fields)')

savestr = 'NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_direct.csv'
df = pd.read_csv(saveDir+savestr, skiprows=0, index_col=0)

# #plt.plot(fout/1e9,-hilbert(df.Re.values).imag)

corr = 0*(df.Im.values[0]/df.index*df.index[0])
plt.plot(df.index/1e9, df.Re.values,'-c', ms = 2, label = 'Re, NMM (source-$J$)')
plt.plot(df.index/1e9, df.Im.values - corr, '-', color = 'orange', ms = 2, label = 'Im, NMM (source-$J$)')

plt.ylim(-60,60)
plt.xlim(0,6)
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir+'NMM_current_cavity_b0.05_L0.01_with_Im.png')

#%% MMT and NMM with source current, direct integration, with the capacitance

plt.figure()
saveDir = './tests/cavity_with_current/'    
df = pd.read_csv(saveDir+savestr, skiprows=0, index_col=0)
corr = (df.Im.values[0]/df.index*df.index[0])
plt.plot(df.index/1e9, df.Re.values,'-c', ms = 2, label = 'Re, NMM (source-$J$)')
plt.plot(df.index/1e9, df.Im.values, '-', color = 'orange', ms = 2, label = 'Im, NMM (source-$J$)')
plt.plot(df.index/1e9, corr, '--', color = 'orange', ms = 2, label = 'Im, capacitance')

plt.ylim(-300,60)
plt.xlim(0,6)
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.legend(loc=4)
plt.tight_layout()
plt.savefig(saveDir+'NMM_current_cavity_b0.05_L0.01_with_Im_C-comp1.png')

#%% MMT and NMM with source current, direct integration, removing the capacitance

saveDir = './tests/cavity_with_current/'    
import pandas as pd
plt.figure()
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')

df = pd.read_csv(saveDir+'/NMM_fields_cavity_b0.05_L0.01.txt', skiprows=0, index_col=0)
plt.plot(df.index/1e9, df.Re.values,'ob', label = 'Re, NMM (source fields)')
plt.plot(df.index/1e9, df.Im.values, 'om', label = 'Im, NMM (source fields)')

savestr = 'NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_direct.csv'
df = pd.read_csv(saveDir+savestr, skiprows=0, index_col=0)

#plt.plot(fout/1e9,-hilbert(df.Re.values).imag)

corr = (df.Im.values[0]/df.index*df.index[0])
plt.plot(df.index/1e9, df.Re.values,'-c', ms = 2, label = 'Re, NMM (source-$J$)')
plt.plot(df.index/1e9, df.Im.values - corr, '-', color = 'orange', ms = 2, label = 'Im, NMM (source-$J$, C-comp)')

plt.ylim(-60,60)
plt.xlim(0,6)
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir+'NMM_current_cavity_b0.05_L0.01_with_Im_C-comp2.png')

#%% MMT and NMM with source current, indirect integration
saveDir = './tests/cavity_with_current/'    
plt.figure()
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')

df = pd.read_csv(saveDir+'/NMM_fields_cavity_b0.05_L0.01.txt', skiprows=0, index_col=0)
plt.plot(df.index/1e9, df.Re.values,'ob', label = 'Re, NMM (source fields)')
plt.plot(df.index/1e9, df.Im.values, 'om', label = 'Im, NMM (source fields)')

savestr = 'NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_indirect.csv'
df = pd.read_csv(saveDir+savestr, skiprows=0, index_col=0)

plt.plot(df.index/1e9, df.Re.values,'.c', ms = 5, label = 'Re, NMM (source-$J$, indirect)')
plt.plot(df.index/1e9, df.Im.values, '.', color = 'orange', ms = 5, label = 'Im, NMM (source-$J$, indirect)')

plt.ylim(-60,60)
plt.xlim(0,6)
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir+'NMM_current_cavity_b0.05_L0.01_with_Im_indirect.png')

