# %%
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
    import pandas as pd
    import nmm

saveDir = '/home/nbiancac/HDD/Work/CERN/Finite_Length/Numerical_MMM/Results/'
calculate = False
# P_max = 2
# S_max = 15
# R_max = 15

P_max = 20
S_max = 35
R_max = 35

# P_vec = np.arange(35, 36, 2, dtype=int)  # np.arange(35, 36, 2, dtype=int)
# S_max = 20  # 5#10
# R_max = 80  # 10#50s
# Np = 100

for Np_ in [150, 50]:
    for beta_ in [0.5]:
    #for beta_ in [0.5]:

        print(f'beta = {beta_}')
        beam = nmm.Beam(beta=beta_)
        geometry = nmm.Pillbox(length=0.01)
        materials = nmm.Materials(sigma=0)
        mesh = nmm.Mesh(geometry, Np=Np_)

        mode = nmm.Mode(
            is_analytical=True,
            index_max_p=P_max,
            max_mode_number=None,
            split_rs=True,
            index_max_r=R_max,
            index_max_s=S_max,
        )
        sim = nmm.Simulation(
            mode=mode,
            geometry=geometry,
            materials=materials,
            beam=beam,
            mesh=mesh,
        )
        sim.rb = 5e-3
        sim.integration = "direct"
        if calculate:
            Zout = []
            fout = []
            sim.preload_matrixes()
            for f in np.linspace(3.5e9,5e9,10, endpoint=True):
                print(f"frequency {f/1e9} GHz")
                sim.f = f
                sim.compute_impedance()
                Zout.append(sim.Z)
                fout.append(sim.f)

            fout = np.array(fout).flatten()
            Zout = np.array(Zout).flatten()

            savestr = f'NMM_beta{beam.beta}_sigma{materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_Np{Np_}for_Elias'
            import pandas as pd

            pd.DataFrame(index=fout, data={'Re': Zout.real, 'Im': Zout.imag}).to_csv(
                saveDir + savestr + '.csv')



# %%
import pandas as pd

plt.figure()
for beta_ in [0.9, 0.8, 0.7, 0.6, 0.5]:
    df_re = pd.read_csv(
            saveDir + f"MMMlong_Re_L0.01_Beta{beta_}_b0.05_t0.05_Material_CEI_empty_cavity_fmin3500000000_fmax5000000000_P55_S55.txt",
            index_col=0)
        # saveDir+f'MMMlong_Re_L0.01_Beta{beta_}_b0.05_t0.05_Material_CEI_empty_cavity_fmin3500000000_fmax5000000000_P55_S55.txt', index_col=0)
        # df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
    if (beta_ == 0.9) and (Np_ == 50):
        p, = plt.plot(df_re.index / 1e9, df_re.values, '-', label='MMT')
    else:
        p, = plt.plot(df_re.index / 1e9, df_re.values, '-', label='_nolabel_')
    # plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')

    for Np_ in [ 50, 150]:
        if Np_ ==50:
            marker = '.';
        else:
            marker = 'x'
    #for beta_ in [0.5]:
        # for beta_ in [0.6]:
        
        savestr = f'NMM_beta{beta_}_sigma{materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_Np{Np_}for_Elias'
        df = pd.read_csv(saveDir + savestr + '.csv', skiprows=0, index_col=0)
        plt.plot(df.Re.index / 1e9, df.Re.values, marker=marker, color=p.get_color(), ms=5, ls = '', label=f'NMM, $\\beta:{beta_}$, $N_p:{Np_}$')

    plt.legend(loc=3, fontsize=8, ncol=3)
    plt.xlabel('Frequency [GHz]', fontsize=14)
    plt.ylabel('Re($Z_l$) [$\Omega$]', fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(saveDir + 'scan_vs_beta_forElias.png')
    plt.savefig('/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/benchmark_pillbox_beta.png')
plt.show()
