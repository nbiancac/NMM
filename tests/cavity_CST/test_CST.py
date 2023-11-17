#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""
import sys, os
cwd = '/home/nbiancac/HDD/Work/CERN/Finite_Length/Numerical_MMM/Codes/repository_on_git/'
os.chdir(cwd)
sys.path.append(cwd)

import field_utlis.fields as fields
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
saveDir = './tests/cavity_CST/'
beam = fields.beam() # initialize beam parameters
geometry = fields.geometry(L = 0.01)
materials = fields.materials(sigma = 0)
Np = 50
mesh = fields.mesh(geometry, Np=Np)

P_max = 30
Mode_vec = [10]

sim = fields.simulation_CST(index_max_p = P_max, index_max_n = max(Mode_vec), Np = Np, \
                        geometry = geometry, materials = materials, beam = beam, mesh = mesh, datadir = saveDir+'CST/')

sim.integration='direct'
        
Zout = []
fout = []
sim.preload_matrixes()
for f in np.linspace(1e9, 5e9, 5, endpoint=True)[:]:
    print(f"frequency {f/1e9} GHz")
    sim.f = f
    sim.compute_impedance()
    Zout.append(sim.Z)
    fout.append(sim.f)
    
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()


plt.figure()
plt.plot(fout, Zout.real)
plt.plot(fout, Zout.imag)
plt.title('Beam current: CST modes '+str(Mode_vec[0])+', fixed P = '+str(sim.index_max_p)+', Np = '+str(Np))
plt.xlabel('Frequency [GHz]')
plt.ylim(-2,60)
plt.tight_layout()