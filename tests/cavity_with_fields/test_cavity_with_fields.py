#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:17:52 2023

@author: nbiancac
"""
import sys
sys.path.append('./src')

import fields_with_source_fields as fields
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps, trapz
from scipy.constants import mu_0

plt.close('all')

beam = fields.beam() # initialize beam parameters

Zout=[]
fout = []

sim = fields.simulation(index_max_p = 15, index_max_s = 5)
sim.sigma = 0 
mesh = fields.mesh(sim, Np = 50)
mesh_loss = fields.mesh_loss(sim, Np = 50)

left =  {'direction' : -1, 
         'zmatch' : 0, 
         'ev' : [],
         'A': np.zeros((sim.index_max_p,sim.index_max_p), dtype = complex),
         'B': np.zeros((sim.index_max_p,sim.index_max_n), dtype = complex), 
         'C': np.zeros((sim.index_max_n, 1), dtype = complex), 
         'D': np.zeros((sim.index_max_n,sim.index_max_p), dtype = complex), 
         'E': np.zeros((sim.index_max_p, 1), dtype = complex),
         'S': np.zeros((sim.index_max_p, 1), dtype = complex),
         'T': np.zeros((sim.index_max_n, 1), dtype = complex),
         'Z': np.zeros((1, sim.index_max_p), dtype = complex)
         }

right = {'direction' : 1, 
         'zmatch' : sim.L, 
         'ev' : [],
         'A': np.zeros((sim.index_max_p,sim.index_max_p), dtype = complex),
         'B': np.zeros((sim.index_max_p,sim.index_max_n), dtype = complex), 
         'C': np.zeros((sim.index_max_n, 1), dtype = complex), 
         'D': np.zeros((sim.index_max_n,sim.index_max_p), dtype = complex), 
         'E': np.zeros((sim.index_max_p, 1), dtype = complex), 
         'S': np.zeros((sim.index_max_p, 1), dtype = complex),
         'T': np.zeros((sim.index_max_n, 1), dtype = complex),
         'Z': np.zeros((1, sim.index_max_p), dtype = complex)             
         }

print('Computing eigenmode fields at boundaries (only once).')
print('Number of eigenmodes: %d'%(len(sim.ix_n)))
for scenario in [left, right]:
    zmatch =    scenario['zmatch']
    for ix_n in sim.ix_n:
        cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
        cav_proj =  fields.projectors(mesh)
        cav_proj.interpolate_at_boundary(cavity, mesh, zmatch)
        scenario['ev'].append(cav_proj)

d = sim.d    
b = sim.b

WF = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
print('Computing eigenmode fields at boundaries.')
for scenario in [left, right]:
    zmatch = scenario['zmatch']
    direction = scenario['direction']
    for ix_n in sim.ix_n:
        cavity_n = fields.cavity(
            sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
        cav_proj = fields.projectors(mesh_loss)
        cav_proj.interpolate_at_boundary(cavity_n, mesh_loss, zmatch)
        argument = abs(cav_proj.Hx)**2 + abs(cav_proj.Hy)**2
        WF[ix_n, ix_n] = trapz(trapz(argument, mesh_loss.xi), mesh_loss.yi)
    scenario['WF'] = WF
    
  
for f in np.linspace(1e8, 6e9, 5):
# for f in [1.45e9]:
    print(f)
    sim.f = f


       
    for scenario in [left, right]:
        direction = scenario['direction']
        zmatch =    scenario['zmatch']
        
        # compute fields:
        v_p = []
        for ix_p in sim.ix_p:
            pipe_p  = fields.pipe(sim, ix_p, direction)
            pipe_proj_p = fields.projectors(mesh)
            pipe_proj_p.interpolate_at_boundary(pipe_p, mesh, zmatch)
            v_p.append(pipe_proj_p)    
            
        # magnetic matching
        A = np.zeros((sim.index_max_p, sim.index_max_p), dtype = complex)
        E = np.zeros((sim.index_max_p, 1), dtype = complex)
        B = np.zeros((sim.index_max_p, sim.index_max_n), dtype = complex) 
        S = np.zeros((sim.index_max_p, 1), dtype = complex)       
        C = np.zeros((sim.index_max_n, 1), dtype = complex)
        T = np.zeros((sim.index_max_n, 1), dtype = complex)
        D = np.zeros((sim.index_max_n, sim.index_max_p), dtype = complex)
        Z = np.zeros((1, sim.index_max_p), dtype = complex)
        
        # point charge at the center
        source = fields.source(sim, beam) 
        source_cav = fields.source_cav(sim, beam)
        source_proj_cav = fields.projectors(mesh)
        source_proj_cav.interpolate_at_boundary(source_cav, mesh, zmatch)         
        source_proj = fields.projectors(mesh)
        source_proj.interpolate_at_boundary(source, mesh, zmatch)      
        
        # ring of charge
        # source = fields.source_ring_top(sim, beam)
        # source_proj = fields.projectors(mesh)
        # source_proj.interpolate_at_boundary(source, mesh, zmatch)

        # source2 = fields.source_ring_bottom(sim, beam)
        # source_proj2 = fields.projectors(mesh)
        # source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
        # source_proj.add_fields(source_proj2)
        
        # sim.b = d
        
        # source_cav = fields.source_ring_top(sim, beam)
        # source_proj_cav = fields.projectors(mesh)
        # source_proj_cav.interpolate_at_boundary(source_cav, mesh, zmatch)

        # source2 = fields.source_ring_bottom(sim, beam)
        # source_proj2 = fields.projectors(mesh)
        # source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
        # source_proj_cav.add_fields(source_proj2)

        # sim.b = b

        # disk of charge
        # source = fields.source_cylinder_top(sim, beam)
        # source_proj = fields.projectors(mesh)
        # source_proj.interpolate_at_boundary(source, mesh, zmatch)

        # source2 = fields.source_cylinder_bottom(sim, beam)
        # source_proj2 = fields.projectors(mesh)
        # source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
        # source_proj.add_fields(source_proj2)
        
        # sim.b = d
        
        # source_cav = fields.source_cylinder_top(sim, beam)
        # source_proj_cav = fields.projectors(mesh)
        # source_proj_cav.interpolate_at_boundary(source_cav, mesh, zmatch)

        # source2 = fields.source_cylinder_bottom(sim, beam)
        # source_proj2 = fields.projectors(mesh)
        # source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
        # source_proj_cav.add_fields(source_proj2)

        # sim.b = b

        # gaussian beam
        # source = fields.source_gaussian(sim, beam)
        # source_proj = fields.projectors(mesh)
        # source_proj.interpolate_at_boundary(source, mesh, zmatch)
        
        # sim.b = d
        
        # source_cav = fields.source_gaussian(sim, beam)
        # source_proj_cav= fields.projectors(mesh)
        # source_proj_cav.interpolate_at_boundary(source_cav, mesh, zmatch)

        # sim.b = b
        

        
        for ix_p in sim.ix_p:
            pipe_p  = fields.pipe(sim, ix_p, direction)
            pipe_proj_p = v_p[ix_p]
            grad_x_p, grad_y_p = pipe_proj_p.Hx, pipe_proj_p.Hy
            E[ix_p, 0] = simps(simps((source_proj.Hx)*grad_x_p + 
                                        (source_proj.Hy)*grad_y_p, mesh.xi,'first'), mesh.yi,'first')
            S[ix_p, 0] = simps(simps((source_proj_cav.Hx)*grad_x_p + 
                                        (source_proj_cav.Hy)*grad_y_p, mesh.xi,'first'), mesh.yi,'first')
            # Z[0, ix_p] = - direction * 1j * sim.b / \
            #     (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (source.alpha_b - direction * pipe_p.alpha_p_tilde)  ) * \
            #     np.exp(1j * zmatch * source.alpha_b / sim.b)
            from scipy.special import jn
            # Z[0, ix_p] = - direction * 1j * sim.b * jn(0, source.rb * pipe_p.alpha_p / sim.b) / \
            #     (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (source.alpha_b - direction * pipe_p.alpha_p_tilde)) * \
            #     np.exp(1j * zmatch * source.alpha_b / sim.b)
            Z[0, ix_p] = - direction * 1j * sim.b * jn(0, source.rb * pipe_p.alpha_p / sim.b)  / \
                (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (source.alpha_b - direction * pipe_p.alpha_p_tilde)) * \
                np.exp(1j * zmatch * (source.alpha_b - 0*(direction+1)/2 * pipe_p.alpha_p_tilde) / sim.b)
            for ix_q in sim.ix_p:
                pipe_q  = fields.pipe(sim, ix_q, direction)
                pipe_proj_q = v_p[ix_q]
                grad_x_q, grad_y_q = pipe_proj_q.Hx, pipe_proj_q.Hy
                A[ix_p, ix_q] = simps(simps((pipe_proj_p.Hx)*grad_x_q + 
                                            (pipe_proj_p.Hy)*grad_y_q, mesh.xi,'first'), mesh.yi,'first')
            for ix_n in sim.ix_n:
                cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj = scenario['ev'][ix_n]
                grad_x, grad_y = pipe_proj_p.Hx, pipe_proj_p.Hy
                B[ix_p, ix_n] = simps(simps(cav_proj.Hx * grad_x + cav_proj.Hy * grad_y, mesh.xi), mesh.yi)
                argument = direction * fields.cross_prod_t(source_proj.Ex, source_proj.Ey, source_proj.Ez,
                                                           cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                C[ix_n, 0] = simps(simps(argument, mesh.xi), mesh.yi,'first')
                argument = direction * fields.cross_prod_t(source_proj_cav.Ex, source_proj_cav.Ey, source_proj_cav.Ez,
                                                           cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                T[ix_n, 0] = -simps(simps(argument, mesh.xi,'first'), mesh.yi,'first')
                argument = direction * fields.cross_prod_t(pipe_proj_p.Ex, pipe_proj_p.Ey, pipe_proj_p.Ez,
                                                           cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                D[ix_n, ix_p] = simps(simps(argument, mesh.xi,'first'), mesh.yi,'first')
            
        scenario['A'] = A
        scenario['B'] = B
        scenario['C'] = C
        scenario['D'] = D
        scenario['E'] = E
        scenario['S'] = S
        scenario['T'] = T
        scenario['Z'] = Z

    MI = np.zeros((sim.index_max_n,sim.index_max_n), dtype = complex)
    MV = np.zeros((sim.index_max_n,sim.index_max_n), dtype = complex)
    ID = np.zeros((sim.index_max_n,sim.index_max_n), dtype = complex)
    W = np.zeros((sim.index_max_n,sim.index_max_n), dtype = complex)
    for ix_n in sim.ix_n:
        cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
        MI[ix_n, ix_n] =  (1j * cavity.k_0) / (cavity.Z0 * (cavity.k_0**2 - cavity.k_ps**2 ))
        ID[ix_n, ix_n] =  1. + 0j
        MV[ix_n, ix_n] =  -1j * (cavity.k_ps) * cavity.Z0 / (cavity.k_0)        
        if (sim.sigma != 0):
            Zw = np.sqrt(mu_0 * 2 * np.pi * sim.f / 2 / sim.sigma)
            loss = (1+1j) *  2 * np.pi * sim.f * mu_0 / (cavity.Q)  \
                 - (1+1j) * Zw  * (left['WF'][ix_n,ix_n] + right['WF'][ix_n,ix_n])
            W[ix_n, ix_n] = MI[ix_n, ix_n] * loss    
        
    II = np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ left['C'] + np.linalg.inv(ID - W - MI @ left['D'] @ \
        np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
        np.linalg.inv(right['A']) @ right['B']) @ MI @ left['T'] + \
        np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ right['C'] + np.linalg.inv(ID - W - MI @ left['D'] @ \
        np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
        np.linalg.inv(right['A']) @ right['B']) @ MI @ right['T'] - \
        np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ left['D'] @ np.linalg.inv(left['A']) @ left['E'] + \
        np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ left['D'] @ np.linalg.inv(left['A']) @ left['S'] - \
        np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ right['D'] @ np.linalg.inv(right['A']) @ right['E'] + \
        np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
        left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
        @ MI @ right['D'] @ np.linalg.inv(right['A']) @ right['S']      
                  
    coeffs = {}
    coeffs['left']  = -np.linalg.inv(left['A']) @ left['E'] + np.linalg.inv(left['A']) @ \
left['S'] + np.linalg.inv(left['A']) @ left['B'] @ II
    coeffs['right'] = -np.linalg.inv(right['A']) @ right['E'] + np.linalg.inv(right['A']) @ \
right['S'] + np.linalg.inv(right['A']) @ right['B'] @ II
    coeffs['cavity'] = MV @ II
       
    # Impedance 
    Zcav = np.zeros((1, sim.index_max_n), dtype = complex)
    for ix_n in sim.ix_n:
        cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
        cav_proj_s = fields.projectors(mesh)
        cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
        Zcav[0,ix_n] = - simps(cav_proj_s.Ez * np.exp(1j * source.alpha_b / sim.b * mesh.Z), mesh.Z,'first')
        
    out =  Zcav @ coeffs['cavity'] + left['Z'] @ coeffs['left'] + right['Z'] @ coeffs['right']
    Zout.append(out)
    fout.append(f)

# Impedance
fout = np.array(fout).flatten()
Zout = np.array(Zout).flatten()
print(Zout)

plt.figure()
plt.plot(fout, Zout.real)
plt.plot(fout, Zout.imag)

#%%

saveDir = './'
import pandas as pd
df_re = pd.read_csv(saveDir+'MMMlong_Re_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
df_im = pd.read_csv(saveDir+'MMMlong_Im_L0.01_Beta0.9999_b0.05_t0.05_Material_CEI_empty_cavity_fmin10000000_fmax6000000000_P35_S35.txt', index_col = 0)
plt.plot(df_re.index/1e9, df_re.values,'-k', label = 'Re, MMT')
plt.plot(df_im.index/1e9, df_im.values, '-r', label = 'Im, MMT')
plt.plot(fout/1e9, Zout.real,'ob', label = 'Re, NMT (fields)')
plt.plot(fout/1e9, Zout.imag, 'om', label = 'Im, NMT (fields)')
plt.xlabel('f / GHz')
plt.ylabel('$Z_l$ / $\Omega$')
plt.ylim(-50,50)
plt.xlim(0, 6)
plt.legend()
plt.tight_layout()
plt.savefig(saveDir+'NMM_fields_cavity_b0.05_L0.01.png')
import pandas as pd
pd.DataFrame(index = fout, data = {'Re': Zout.real, 'Im': Zout.imag}).to_csv(saveDir+'NMM_fields_cavity_b0.05_L0.01.txt')

#%% Ez
# rb = 1e-4
# Ez_r = 0
# z_r = mesh.Z+sim.L
# direction = 1
# for ix_p in sim.ix_p[0:]:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Ez_r += (coeffs['right'][ix_p]*pipe_p.Ez(rb,0, z_r-sim.L)*pipe_p.Lz(rb,0, z_r-sim.L))

# Ez_l = 0
# z_l = mesh.Z-sim.L
# direction = -1
# for ix_p in sim.ix_p[0:]:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Ez_l += (coeffs['left'][ix_p]*pipe_p.Ez(rb,0, z_l)*pipe_p.Lz(rb,0, z_l))

# # source = fields.source_ring_top(sim, beam)
# source2 = fields.source(sim, beam)

# Ez_l += (source2.Ez(rb,0, z_l))
# Ez_r += (source2.Ez(rb,0, z_r))

# Ez_c = 0
# z_c = mesh.Z
# for ix_n in sim.ix_n[0:]:
#     cavity_ = fields.cavity(
#         sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
#     Ez_c += (coeffs['cavity'][ix_n] * cavity_.Ez(rb,0,z_c))


# source2_c = fields.source_cav(sim, beam)
# # Ez_c += (source2_c.Ez(rb,0, z_c))

# plt.plot(z_c, abs(Ez_c))  
# plt.plot(z_r, abs(Ez_r))  
# plt.plot(z_l, abs(Ez_l)) 


# #%% Er
# rb = 1e-4
# Er_r = 0
# z_r = mesh.Z+sim.L
# direction = 1
# for ix_p in sim.ix_p:
    
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Er_r += (coeffs['right'][ix_p]*pipe_p.Er(rb,0, z_r-sim.L)*pipe_p.Lz(rb,0, z_r-sim.L))

# Er_l = 0
# z_l = mesh.Z-sim.L
# direction = -1
# for ix_p in sim.ix_p:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Er_l += (coeffs['left'][ix_p]*pipe_p.Er(rb,0, z_l)*pipe_p.Lz(rb,0, z_l))

# # source = fields.source_ring_top(sim, beam)
# source2 = fields.source(sim, beam)

# Er_l += (source2.Er(rb,0, z_l))
# Er_r += (source2.Er(rb,0, z_r))

# Er_c = 0
# z_c = mesh.Z
# for ix_n in sim.ix_n:
#     cavity_ = fields.cavity(
#         sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
#     Er_c += (coeffs['cavity'][ix_n] * cavity_.Er(rb,0,z_c))


# # sim.b = d
# # source_c = fields.source_ring_top(sim, beam)
# source2_c = fields.source_cav(sim, beam)
# # sim.b = b

# Er_c += (source2_c.Er(rb,0, z_c))

# plt.plot(z_c, abs(Er_c))  
# plt.plot(z_r, abs(Er_r))  
# plt.plot(z_l, abs(Er_l)) 

# #%% Hphi
# rb = 0.6e-2
# Hphi_r = 0
# z_r = mesh.Z+sim.L
# direction = 1
# for ix_p in sim.ix_p:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Hphi_r += (coeffs['right'][ix_p]*pipe_p.Hphi(rb,0, z_r)*pipe_p.Lz(rb,0, z_r-sim.L))

# Hphi_l = 0
# z_l = mesh.Z-sim.L
# direction = -1
# for ix_p in sim.ix_p:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     Hphi_l += (coeffs['left'][ix_p]*pipe_p.Hphi(rb,0, z_l)*pipe_p.Lz(rb,0, z_l))

# source = fields.source(sim, beam)
# Hphi_l += (source.Hphi(rb,0, z_l))
# Hphi_r += (source.Hphi(rb,0, z_r))

# Hphi_c = 0
# z_c = mesh.Z
# for ix_n in sim.ix_n:
#     cavity_ = fields.cavity(
#         sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
#     Hphi_c += (II[ix_n] * cavity_.Hphi(rb,0,z_c))

# source = fields.source_cav(sim, beam)
# Hphi_c += (source.Hphi(rb,0, z_c))
    
# plt.plot(z_c, abs(Hphi_c))  
# plt.plot(z_r, abs(Hphi_r))  
# plt.plot(z_l, abs(Hphi_l)) 