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
calculate = True
beam = nmm.Beam()
geometry = nmm.Pillbox(length=0.01)
materials = nmm.Materials(sigma=0)
mesh = nmm.Mesh(geometry, Np=50)
# P_max = 5
# S_max = 45
# R_max = 45
P_max = 20
S_max = 35
R_max = 35
# P_max = 35
# S_max = 35
# R_max = 35

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
    for f in np.linspace(1e3, 6e9, 50, endpoint=True)[:]:
        print(f"frequency {f/1e9} GHz")
        sim.f = f
        sim.compute_impedance()
        Zout.append(sim.Z)
        fout.append(sim.f)

    fout = np.array(fout).flatten()
    Zout = np.array(Zout).flatten()

    savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}"
    pd.DataFrame(index=fout, data={"Re": Zout.real, "Im": Zout.imag}).to_csv(
        saveDir + savestr + ".csv"
    )


saveDir = "./"
import pandas as pd
####################################
plt.figure()
df_re = pd.read_csv(
    saveDir
    + "MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
)
df_im = pd.read_csv(
    saveDir
    + "MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
)
plt.plot(df_re.index / 1e9, df_re.values, "-k", lw=4, label="Re, MMT")
plt.plot(df_im.index / 1e9, df_im.values, "-r", lw=4, label="Im, MMT")

df = pd.read_csv(
    saveDir + "/NMM_fields_cavity_b0.05_L0.01.txt", skiprows=0, index_col=0
)
plt.plot(df.index / 1e9, df.Re.values, "--r", color = 'r', alpha=1, label="Re, NMM (source fields)")
plt.plot(df.index / 1e9, df.Im.values, "--k", alpha=1, label="Im, NMM (source fields)")

# savestr = "NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_direct.csv"
# savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}.csv"
# df = pd.read_csv(saveDir + savestr, skiprows=0, index_col=0)
#
# # plt.plot(fout/1e9,-hilbert(df.Re.values).imag)
#
# corr = df.Im.values[0] / df.index * df.index[0]
# plt.plot(df.index[::5] / 1e9, df.Re.values[::5], ".r", ms=4, label="Re, NMM (source current)")
# plt.plot(
#     df.index[::3] / 1e9,
#     (df.Im.values - 0*corr)[::3],
#     ".",
#     color="c",
#     ms=4,
#     label="Im, NMM (source current)",
# )

plt.ylim(-60, 60)
plt.xlim(0, 6)
plt.xlabel("Frequency [GHz]", fontsize = 14)
plt.ylabel("$Z_l$ [$\Omega$]", fontsize = 14)
plt.legend(loc=2,  fontsize = 12)
plt.tight_layout()
plt.savefig(saveDir + "NMM_current_cavity_b0.05_L0.01_with_Im_C-comp2.png")
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_vs_MMT_fields.png")
####################################

####################################
plt.figure()
df_re = pd.read_csv(
    saveDir
    + "MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
)
df_im = pd.read_csv(
    saveDir
    + "MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
)
plt.plot(df_re.index / 1e9, df_re.values, "-k", lw=4, label="Re, MMT")
plt.plot(df_im.index / 1e9, df_im.values, "-r", lw=4, label="Im, MMT")

df = pd.read_csv(
    saveDir + "/NMM_fields_cavity_b0.05_L0.01.txt", skiprows=0, index_col=0
)
# plt.plot(df.index / 1e9, df.Re.values, "--", color = 'orange', alpha=1, label="Re, NMM (source fields)")
# plt.plot(df.index / 1e9, df.Im.values, "--b", alpha=1, label="Im, NMM (source fields)")

savestr = "NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_direct.csv"
savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}.csv"
df = pd.read_csv(saveDir + savestr, skiprows=0, index_col=0)

# plt.plot(fout/1e9,-hilbert(df.Re.values).imag)

corr = df.Im.values[0] / df.index * df.index[0]
plt.plot(df.index/ 1e9, df.Re.values, "--r", ms=4, label="Re, NMM (source current)")
plt.plot(
    df.index / 1e9,
    (df.Im.values - 0*corr),
    "-",
    color="orange",
    ms=4,
    label="Im, NMM (source current)",
)
plt.plot(
    df.index / 1e9,
    (df.Im.values - corr),
    "--",
    color="k",
    ms=4,
    label="Im, NMM (source current), C-compensated",
)
plt.ylim(-60, 60)
plt.xlim(0, 6)
plt.xlabel("Frequency [GHz]", fontsize = 14)
plt.ylabel("$Z_l$ [$\Omega$]", fontsize = 14)
plt.legend(loc=2,  fontsize = 12)
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir + "NMM_current_cavity_b0.05_L0.01_with_Im_C-comp2.png")
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_vs_MMT_current.png")
####################################

####################################
plt.figure()
df_re = pd.read_csv(
    saveDir
    + "MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names = ['re'],
)
df_im = pd.read_csv(
    saveDir
    + "MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt",
    index_col=0,
    names=['im']
)

df_im = df_im.append(pd.DataFrame(index = [0], data = [0], columns=['im'])).sort_index()
df_re = df_re.append(pd.DataFrame(index = [0], data = [0], columns=['re'])).sort_index()

plt.plot(df_re.index / 1e9, df_re.values, "-k", lw=4, label="Re, MMT")
plt.plot(df_im.index / 1e9, df_im.values, "-r", lw=4, label="Im, MMT")
print(df_im.head())
df = pd.read_csv(
    saveDir + "/NMM_fields_cavity_b0.05_L0.01.txt", skiprows=0, index_col=0
)
# plt.plot(df.index / 1e9, df.Re.values, "--", color = 'orange', alpha=1, label="Re, NMM (source fields)")
# plt.plot(df.index / 1e9, df.Im.values, "--b", alpha=1, label="Im, NMM (source fields)")

savestr = "NMM_sigma0_L0.01_t0.05_b0.05_rb0.005_P15_R25_S5_direct.csv"
savestr = f"NMM_sigma{sim.materials.sigma}_L{sim.L}_t{sim.t}_b{sim.b}_rb{sim.rb}_P{P_max}_R{R_max}_S{S_max}_{sim.integration}.csv"
df = pd.read_csv(saveDir + savestr, skiprows=0, index_col=0)

# plt.plot(fout/1e9,-hilbert(df.Re.values).imag)

corr = df.Im.values[0] / df.index * df.index[0]
plt.plot(df.index/ 1e9, df.Re.values, "--r", ms=4, label="Re, NMM (source current)")
plt.plot(
    df.index / 1e9,
    (df.Im.values - 0*corr),
    "-",
    color="orange",
    ms=4,
    label="Im, NMM (source current)",
)
plt.plot(
    df.index / 1e9,
    (df.Im.values - corr),
    "--",
    color="k",
    ms=4,
    label="Im, NMM (source current), C-compensated",
)
plt.ylim(-10, 20)
plt.xlim(0, 1)
plt.xlabel("Frequency [GHz]", fontsize = 14)
plt.ylabel("$Z_l$ [$\Omega$]", fontsize = 14)
plt.legend(loc=2,  fontsize = 12)
plt.legend(loc=2)
plt.tight_layout()
plt.savefig(saveDir + "NMM_current_cavity_b0.05_L0.01_with_Im_C-comp2.png")
plt.savefig("/home/nbiancac/HDD/Dropbox/Apps/Overleaf/NMM_paper/Pictures/NMM_vs_MMT_current_zoom.png")

plt.show()
