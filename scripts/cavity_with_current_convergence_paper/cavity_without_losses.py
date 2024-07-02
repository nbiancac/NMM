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

saveDir = "./"
calculate = False

# P scan
beam = nmm.Beam()
geometry = nmm.Pillbox(length=0.01)
materials = nmm.Materials(sigma=0)
mesh = nmm.Mesh(geometry, Np=50)
P_max = 10
S_max = 50
R_max = 50
P_vec = np.arange(1,R_max, 2)
f_sample = 4.2e9 #1e7
for P_max in P_vec:
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
    sim.integration = "direct"
    if calculate:
        Zout = []
        fout = []
        sim.preload_matrixes()
        for f in [f_sample]:
            print(f"{P_max}: frequency {f/1e9} GHz")
            sim.f = f
            sim.compute_impedance()
            Zout.append(sim.Z)
            fout.append(sim.f)

        fout = np.array(fout).flatten()
        Zout = np.array(Zout).flatten()

        savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}"
        pd.DataFrame(index=fout, data={"Re": Zout.real, "Im": Zout.imag}).to_csv(
            saveDir + savestr + ".csv"
        )


saveDir = "./"
import pandas as pd
df = []
for P_max in P_vec:
    savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}"
    df_ = pd.read_csv(saveDir + savestr + '.csv', skiprows=0, index_col=0)
    df_['P'] = P_max
    df_['R'] = R_max
    df_['S'] = S_max
    df.append(df_)
df = pd.concat(df)

df.head()

df_re_mm = pd.read_csv(
    saveDir
    + "MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names = ['re']
)
df_im_mm = pd.read_csv(
    saveDir
    + "MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names = ['im']
)

target_re = df_re_mm.at[f_sample,'re']
target_im = df_im_mm.at[f_sample,'im']

plt.figure()
plt.plot(df.P.values, df.Re.values, color = 'b', ls = '-')
plt.hlines(xmin = 0, xmax = df.P.values.max(), y=target_re, ls = '--', color = 'k')
# savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}_re"
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_re_P.png")

plt.figure()
plt.plot(df.P.values, df.Im.values, color = 'b', ls = '-')
plt.hlines(xmin = 0, xmax = df.P.values.max(), y=target_im, ls = '--', color = 'k')
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_im_P.png")

df_P = df

# R, S scan
beam = nmm.Beam()
geometry = nmm.Pillbox(length=0.01)
materials = nmm.Materials(sigma=0)
mesh = nmm.Mesh(geometry, Np=50)
P_max = 20
S_max = 50
R_max = 50
vec = np.arange(1,R_max, 2)
f_sample = 4.2e9 #1e7
for R_max in vec:
    S_max = R_max
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
    sim.integration = "direct"

    if calculate:
        Zout = []
        fout = []
        sim.preload_matrixes()
        for f in [f_sample]:
            print(f"{P_max}: frequency {f/1e9} GHz")
            sim.f = f
            sim.compute_impedance()
            Zout.append(sim.Z)
            fout.append(sim.f)

        fout = np.array(fout).flatten()
        Zout = np.array(Zout).flatten()

        savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}"
        pd.DataFrame(index=fout, data={"Re": Zout.real, "Im": Zout.imag}).to_csv(
            saveDir + savestr + ".csv"
        )


saveDir = "./"
import pandas as pd
df = []
P_max = 20
for R_max in vec:
    S_max = R_max
    savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}"
    df_ = pd.read_csv(saveDir + savestr + '.csv', skiprows=0, index_col=0)
    df_['P'] = P_max
    df_['R'] = R_max
    df_['S'] = S_max
    df.append(df_)
df = pd.concat(df)

df.head()

df_re_mm = pd.read_csv(
    saveDir
    + "MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names = ['re']
)
df_im_mm = pd.read_csv(
    saveDir
    + "MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names = ['im']
)

target_re = df_re_mm.at[f_sample,'re']
target_im = df_im_mm.at[f_sample,'im']

plt.figure()
plt.plot(df.R.values, df.Re.values, color = 'b', ls = '-')
plt.hlines(xmin = 0, xmax = df.R.values.max(), y=target_re, ls = '--', color = 'k')
# savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}_re"
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_re_RS.png")

plt.figure()
plt.plot(df.R.values, df.Im.values, color = 'b', ls = '-')
plt.hlines(xmin = 0, xmax = df.R.values.max(), y=target_im, ls = '--', color = 'k')
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_im_RS.png")

df_RS = df

plt.figure()
plt.plot(df_RS.R.values, df_RS.Re.values, color = 'b', ls = '-', label= 'Scan on R = S (P = 20)')
plt.plot(df_P.P.values, df_P.Re.values, color = 'r', ls = '-', label='Scan on P (R = S = 50)')
plt.hlines(xmin = 1, xmax = df_RS.R.values.max(), y=target_re, ls = '--', color = 'k', label = 'MMT')
# savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}_conv_at_{f_sample}_re"
plt.xlabel('P, R, S parameter', fontsize = 14)
plt.ylabel("Re($Z_l$) [$\Omega$]", fontsize = 14)
plt.legend(loc=2,  fontsize = 12)
plt.ylim(0, 100)
plt.xlim(1, df_RS.R.values.max())
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_re_RSP.png")

plt.figure()
plt.plot(df_RS.R.values, df_RS.Im.values, color = 'b', ls = '-', label= 'Scan on R = S (P = 20)')
plt.plot(df_P.P.values, df_P.Im.values, color = 'r', ls = '-', label='Scan on P (R = S = 50)')
plt.hlines(xmin = 1, xmax = df.R.values.max(), y=target_im, ls = '--', color = 'k', label = 'MMT')
plt.xlabel('P, R, S parameter', fontsize = 14)
plt.ylabel("Im($Z_l$) [$\Omega$]", fontsize = 14)
plt.legend(loc=0,  fontsize = 12)
plt.xlim(1, df_RS.R.values.max())
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_current_conv_im_RSP.png")

plt.show()

