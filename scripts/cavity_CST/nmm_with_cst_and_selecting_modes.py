#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""
import path_for_scripts
with path_for_scripts.Context():
    import matplotlib.pyplot as plt
    import numpy as np
    import nmm_CST

plt.close("all")
saveDir = "./"
beam = nmm_CST.Beam()  # initialize beam parameters
geometry = nmm_CST.CST_object(length=0.01)
materials = nmm_CST.Materials(sigma=0)
mesh = nmm_CST.Mesh(geometry, Np=50)

max_num_pipe_modes = 3
max_num_cavity_modes = 20

Zlast = 0
list_modes = []
Z_tol = 1  # tolerance for an impedance change at any frequency
for _max_num_cavity_modes in np.arange(1, max_num_cavity_modes):
    list_modes.append(_max_num_cavity_modes)
    mode = nmm_CST.Mode(is_analytical=False, index_max_p=max_num_pipe_modes, max_mode_number=_max_num_cavity_modes,
                        split_rs=False, list_modes=list_modes)
    mode.datadir = './cst_large_set/'

    sim = nmm_CST.simulation_CST(
        mode=mode,
        geometry=geometry,
        materials=materials,
        beam=beam,
        mesh=mesh,
    )
    sim.integration = "direct"
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

    if np.any(abs(Zout.real - Zlast.real) > Z_tol):
        pass
        # keep the new mode
    else:
        list_modes.pop()
        # discard the new mode
    Zlast = Zout

plt.figure()
plt.plot(fout, Zout.real, label=f"CST {len(list_modes)} modes")
plt.title("Beam current: fixed P = " + str(sim.index_max_p) + ", Np = " + str(mesh.Np))
plt.xlabel("Frequency [GHz]")
plt.ylim(-2, 60)
plt.tight_layout()

list_selected_modes = list_modes
print(list_selected_modes)
#%% Compare the impedance on selected modes with the one from all the modes

mode = nmm_CST.Mode(is_analytical=False, index_max_p=max_num_pipe_modes, max_mode_number=max_num_cavity_modes, split_rs=False, list_modes=list_selected_modes)
mode.datadir = './cst_large_set/'
sim = nmm_CST.simulation_CST(
    mode=mode,
    geometry=geometry,
    materials=materials,
    beam=beam,
    mesh=mesh,
)

sim.integration = "direct"
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
plt.plot(fout, Zout.real, label=f"CST {len(list_selected_modes)} modes")

Zout = []
fout = []
mode = nmm_CST.Mode(is_analytical=False, index_max_p=max_num_pipe_modes, max_mode_number=max_num_cavity_modes, split_rs=False)
mode.datadir = './cst_large_set/'
sim = nmm_CST.simulation_CST(
    mode=mode,
    geometry=geometry,
    materials=materials,
    beam=beam,
    mesh=mesh,
)

sim.integration = "direct"
sim.preload_matrixes()
for f in np.linspace(1e9, 5e9, 100, endpoint=True)[:]:
    print(f"frequency {f/1e9} GHz")
    sim.f = f
    sim.compute_impedance()
    Zout.append(sim.Z)
    fout.append(sim.f)
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()

plt.plot(fout, Zout.real, "--", label=f"CST {len(list_modes)} modes")
plt.title("Fixed P = " + str(sim.index_max_p) + ", Np = " + str(mesh.Np))
plt.xlabel("Frequency [GHz]")
plt.ylim(-2, 60)
plt.tight_layout()
plt.legend()
plt.show()