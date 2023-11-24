#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""
import sys
sys.path.append('./src/')

import nmm_CST as nmm
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
saveDir = './'
beam = nmm.beam() # initialize beam parameters
geometry = nmm.geometry(L = 0.01)
materials = nmm.materials(sigma = 0)
Np = 50
mesh = nmm.mesh(geometry, Np=Np)

P_max = 3
N_max = 100
list_modes = [1]

Zlast = 0
list_modes = []
Z_tol = 1 # tolerance for an impedance change at any frequency
for ii in np.arange(1, N_max):    
    list_modes.append(ii)
    sim = nmm.simulation_CST(index_max_p = P_max, index_max_n = N_max, Np = Np, \
                            geometry = geometry, materials = materials, beam = beam, mesh = mesh, 
                            datadir = saveDir+'cst/',
                            list_modes = list_modes)
    
    sim.integration='direct'
    sim.preload_matrixes()
    Zout = []
    fout = []
    for f in np.linspace(1e9, 5e9, 15, endpoint=True)[:]:
        print(f"frequency {f/1e9} GHz")
        sim.f = f
        sim.compute_impedance()
        Zout.append(sim.Z)
        fout.append(sim.f)
    fout = np.array(fout).flatten()
    Zout = np.array(Zout).flatten()
    
    if np.any(abs(Zout.real-Zlast.real) > Z_tol):
       pass; # keep the new mode
    else:
        list_modes.pop(); # discard the new mode
    Zlast = Zout

plt.figure()
plt.plot(fout, Zout.real, label = f'CST {len(list_modes)} modes')
plt.title('Beam current: fixed P = '+str(sim.index_max_p)+', Np = '+str(Np))
plt.xlabel('Frequency [GHz]')
plt.ylim(-2,60)
plt.tight_layout()

list_selected_modes = list_modes
print(list_selected_modes)
#%% Compare the impedance on selected modes with the one from all the modes

sim = nmm.simulation_CST(index_max_p = P_max, index_max_n = N_max, Np = Np, \
                        geometry = geometry, materials = materials, beam = beam, mesh = mesh, 
                        datadir = saveDir+'cst/',
                        list_modes = list_selected_modes)

sim.integration='direct'
sim.preload_matrixes()
Zout = []
fout = []
for f in np.linspace(1e9, 5e9, 100, endpoint=True)[:]:
    print(f"frequency {f/1e9} GHz")
    sim.f = f
    sim.compute_impedance()
    Zout.append(sim.Z)
    fout.append(sim.f)
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()

plt.figure()
plt.plot(fout, Zout.real, label = f'CST {len(list_selected_modes)} modes')

Zout = []
fout = []
list_modes = np.arange(1,100)
sim = nmm.simulation_CST(index_max_p = P_max, index_max_n = N_max, Np = Np, \
                        geometry = geometry, materials = materials, beam = beam, mesh = mesh, 
                        datadir = saveDir+'cst/',
                        list_modes = list_modes)

sim.integration='direct'
sim.preload_matrixes()
for f in np.linspace(1e9, 5e9, 100, endpoint=True)[:]:
    print(f"frequency {f/1e9} GHz")
    sim.f = f
    sim.compute_impedance()
    Zout.append(sim.Z)
    fout.append(sim.f)
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()

plt.plot(fout, Zout.real, '--', label = f'CST {len(list_modes)} modes')
plt.title('Fixed P = '+str(sim.index_max_p)+', Np = '+str(Np))
plt.xlabel('Frequency [GHz]')
plt.ylim(-2,60)
plt.tight_layout()
plt.legend()
