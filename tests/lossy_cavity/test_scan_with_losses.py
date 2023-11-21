#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:23:32 2023

@author: nbiancac
"""
# execution assumes being in the working directory is in the test file directory
import sys 
sys.path.append('./src')

import nmm as nmm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.close('all')

saveDir = './' 

# # large range scan
for sigma_ in [1e3]:
    
    beam = nmm.beam() # initialize beam parameters
    geometry = nmm.geometry(L = 0.01)
    materials = nmm.materials(sigma = sigma_)
    Np = 50
    mesh = nmm.mesh(geometry, Np=Np)
    P_max = 10
    S_max = 5
    R_max = P_max * 2
    
    sim = nmm.simulation(index_max_p=P_max, index_max_r=R_max, index_max_s = S_max, Np = Np, \
                            geometry = geometry, materials = materials, beam = beam, mesh = mesh)
    sim.integration='direct'
            
    Zout = []
    fout = []
    sim.preload_matrixes()
    for f in np.linspace(1e8,6e9,10, endpoint=True):
        print(f"frequency {f/1e9} GHz")
        sim.f = f
        sim.compute_impedance()
        Zout.append(sim.Z)
        fout.append(sim.f)
        
    fout = np.array(fout).flatten()
    Zout = np.array(Zout).flatten()
    savestr = f'NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}'
    pd.DataFrame(index = fout, data = {'Re': Zout.real, 'Im': Zout.imag}).to_csv(saveDir+savestr+'.csv')

    # refined scan    
    Zout = []
    fout = []
    for f in np.linspace(1.4e9,1.55e9, 10, endpoint=True):
        print(f"frequency {f/1e9} GHz")
        sim.f = f
        sim.compute_impedance()
        Zout.append(sim.Z)
        fout.append(sim.f)
    
    fout = np.array(fout).flatten()
    Zout = np.array(Zout).flatten()
    savestr+='_refined'
    pd.DataFrame(index = fout, data = {'Re': Zout.real, 'Im': Zout.imag}).to_csv(saveDir+savestr+'.csv')

print('Done')
#%%
# saveDir = './'    
# CSTDir = './data_CST/'
# import pandas as pd

# for sigma in [1e3]:
#     fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
#     df = pd.read_csv(CSTDir+f'/cavity_sigma{sigma:.0f}.txt', sep = '\t', skiprows=0, index_col=0)
#     ax1.plot(df.index, df.values[:,0], '-k', label = 'Re, CST')
#     ax1.plot(df.index, df.values[:,1], '-r', label = 'Im, CST')
#     ax2.plot(df.index, df.values[:,0], '-k', label = 'Re, CST')
#     ax2.plot(df.index, df.values[:,1], '-r', label = 'Im, CST')
#     savestr = f'NMM_sigma{sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}'
#     df = pd.read_csv(saveDir+savestr+'.csv', skiprows=0, index_col=0)
#     try:
#         savestr+='_refined'
#         df2 = pd.read_csv(saveDir+savestr+'.csv', skiprows=0, index_col=0)
#         df = pd.concat((df,df2)).sort_index()
#     except: pass


#     corr = 0#(df.Im.values[0]/df.index*df.index[0])
#     ax1.plot(df.index/1e9, df.Re.values,'-b', ms=1, label = 'Re, NMM')
#     ax1.plot(df.index/1e9, df.Im.values - corr, '-m', ms=1, label = 'Im, NMM')
#     ax2.plot(df.index/1e9, df.Re.values,'-b', ms=1, label = 'Re, NMM')
#     ax2.plot(df.index/1e9, df.Im.values - corr, '-m', ms=1, label = 'Im, NMM')
    
#     ax2.set_xlim(1.38, 1.55)
#     ax1.set_xlim(0, 5)
#     # ax2.set_xlim(3, 4.5)
#     # ax2.set_ylim(0, 1050)
#     # ax1.set_xlim(0, 6)
#     ax1.set_xlabel('f / GHz')
#     ax2.set_xlabel('f / GHz')
#     ax1.set_ylabel('$Z_l$ / $\Omega$')
#     ax2.set_ylabel('')
#     ax1.legend()
#     ax2.legend()
#     plt.suptitle(f'$\sigma$={sigma:.0e} S/m')
#     plt.tight_layout()
#     # plt.savefig(saveDir+f'losses_cavity_sigma{sigma}_b0.005_L0.02.png')    
