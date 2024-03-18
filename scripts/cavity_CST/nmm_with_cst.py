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
    import nmm

saveDir = "./"
beam = nmm.Beam()
geometry = nmm.CST_object(length=0.01)
materials = nmm.Materials(sigma=0)
mesh = nmm.Mesh(geometry, Np=50)

max_num_pipe_modes = 30
max_num_cavity_modes = 10

mode = nmm.Mode(
    is_analytical=False,
    index_max_p=max_num_pipe_modes,
    max_mode_number=max_num_cavity_modes,
    split_rs=False,
)
sim = nmm.simulation_CST(
    mode=mode,
    geometry=geometry,
    materials=materials,
    beam=beam,
    mesh=mesh,
)

sim.integration = "direct"
sim.preload_matrixes()

impedance = []
frequency = np.linspace(1e9, 5e9, 5, endpoint=True)
for _f in frequency:
    print(f"frequency {_f/1e9} GHz")
    sim.f = _f
    sim.compute_impedance()
    impedance.append(sim.Z)
impedance = np.array(impedance).squeeze()

# plot real/imag part of the impedance vs frequency
plt.figure()
plt.plot(frequency, impedance.real)
plt.plot(frequency, impedance.imag)
plt.title(
    "Beam current: CST modes "
    + str(max_num_cavity_modes)
    + ", fixed P = "
    + str(sim.index_max_p)
    + ", Np = "
    + str(mesh.Np)
)
plt.xlabel("Frequency [GHz]")
plt.ylim(-2, 60)
plt.tight_layout()
# plt.show()

np.testing.assert_almost_equal(
    impedance[0], 2.21700483e-03 + -1045.15234271j
)  # assert if result is unchanged
