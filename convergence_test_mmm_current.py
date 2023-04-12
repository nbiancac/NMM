#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""

import field_utlis.fields as fields
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz

plt.close('all')
beam = fields.beam() # initialize beam parameters

P_vec = np.arange(5, 26, 2, dtype=int)
S_vec = [15]
Np = 71

Z_conv = [];

for iP in P_vec:
    for iS in S_vec:
        print(iP, iS)
        
        plt.close('all')

        beam = fields.beam(beta=0.9999)
        beam.Q = 1
        Zout = []
        fout = []

        sim = fields.simulation(index_max_p=iP, index_max_s = iS)
        mesh = fields.mesh(sim, Np=Np)
        left = {'direction': -1,
                'zmatch': 0,
                'ev': [],
                'A': np.zeros((sim.index_max_p, sim.index_max_p), dtype=complex),
                'B': np.zeros((sim.index_max_p, sim.index_max_n), dtype=complex),
                'C': np.zeros((sim.index_max_n, 1), dtype=complex),
                'D': np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex),
                'E': np.zeros((sim.index_max_p, 1), dtype=complex),
                'S': np.zeros((sim.index_max_p, 1), dtype=complex),
                'T': np.zeros((sim.index_max_n, 1), dtype=complex),
                'Z': np.zeros((1, sim.index_max_p), dtype=complex)
                }

        right = {'direction': 1,
                 'zmatch': sim.L,
                 'ev': [],
                 'A': np.zeros((sim.index_max_p, sim.index_max_p), dtype=complex),
                 'B': np.zeros((sim.index_max_p, sim.index_max_n), dtype=complex),
                 'C': np.zeros((sim.index_max_n, 1), dtype=complex),
                 'D': np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex),
                 'E': np.zeros((sim.index_max_p, 1), dtype=complex),
                 'S': np.zeros((sim.index_max_p, 1), dtype=complex),
                 'T': np.zeros((sim.index_max_n, 1), dtype=complex),
                 'Z': np.zeros((1, sim.index_max_p), dtype=complex)
                 }

        print('Computing eigenmode fields at boundaries (only once).')
        print('Number of eigenmodes: %d' % (len(sim.ix_n)))
        for scenario in [left, right]:
            zmatch = scenario['zmatch']
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj = fields.projectors(mesh)
                cav_proj.interpolate_at_boundary(cavity, mesh, zmatch)
                scenario['ev'].append(cav_proj)

        #for f in np.concatenate(([1e6], np.linspace(1e9,6e9,100, endpoint=1))):
        for f in [4.2e9]:
            print(f)
            sim.f = f
            

            for scenario in [left, right]:
                direction = scenario['direction']
                zmatch = scenario['zmatch']

                # compute fields:
                v_p = []
                for ix_p in sim.ix_p:
                    pipe_p = fields.pipe(sim, ix_p, direction)
                    pipe_proj_p = fields.projectors(mesh)
                    pipe_proj_p.interpolate_at_boundary(pipe_p, mesh, zmatch)
                    v_p.append(pipe_proj_p)

                # magnetic matching
                A = np.zeros((sim.index_max_p, sim.index_max_p), dtype=complex)
                E = np.zeros((sim.index_max_p, 1), dtype=complex)
                B = np.zeros((sim.index_max_p, sim.index_max_n), dtype=complex)
                L = np.zeros((sim.index_max_p, sim.index_max_n), dtype=complex)
                S = np.zeros((sim.index_max_p, 1), dtype=complex)
                C = np.zeros((sim.index_max_n, 1), dtype=complex)
                H = np.zeros((sim.index_max_n, 1), dtype=complex)
                S = np.zeros((sim.index_max_n, 1), dtype=complex)
                T = np.zeros((sim.index_max_n, 1), dtype=complex)
                D = np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex)
                K = np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex)
                U = np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex)
                Z = np.zeros((1, sim.index_max_p), dtype=complex)

                source = fields.source_ring_top(sim, beam)
                source_proj = fields.projectors(mesh)
                source_proj.interpolate_at_boundary(source, mesh, zmatch)
                source2 = fields.source_ring_bottom(sim, beam)
                source_proj2 = fields.projectors(mesh)
                source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
                source_proj.add_fields(source_proj2)

                # magnetic matching
                for ix_p in sim.ix_p:
                    pipe_p = fields.pipe(sim, ix_p, direction)
                    pipe_proj_p = v_p[ix_p]
                    from scipy.special import jn
                    Z[0, ix_p] = - direction * 1j * sim.b * jn(0, source.rb * pipe_p.alpha_p / sim.b)  / \
                        (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (source.alpha_b - direction * pipe_p.alpha_p_tilde)) * \
                        np.exp(1j * zmatch * (source.alpha_b - (direction+1)/2 * pipe_p.alpha_p_tilde) / sim.b)
                    grad_x_p, grad_y_p = pipe_proj_p.Hx, pipe_proj_p.Hy
                    E[ix_p, 0] = trapz(trapz(source_proj.Hx*grad_x_p + source_proj.Hy*grad_y_p, mesh.xi), mesh.yi)
                    for ix_q in sim.ix_p:
                        pipe_q = fields.pipe(sim, ix_q, direction)
                        pipe_proj_q = v_p[ix_q]
                        grad_x_q, grad_y_q = pipe_proj_q.Hx, pipe_proj_q.Hy
                        A[ix_p, ix_q] = trapz(trapz((pipe_proj_p.Hx)*grad_x_q + (pipe_proj_p.Hy)*grad_y_q, mesh.xi), mesh.yi)
                    for ix_n in sim.ix_n:
                        cavity = fields.cavity(
                            sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                        cav_proj = scenario['ev'][ix_n]
                        B[ix_p, ix_n] = trapz(trapz(cav_proj.Hx * grad_x_p + cav_proj.Hy * grad_y_p, mesh.xi), mesh.yi)
                        argument = direction * fields.cross_prod_t(source_proj.Ex, source_proj.Ey, source_proj.Ez,
                                                                   cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                        C[ix_n, 0] = trapz(trapz(argument, mesh.xi), mesh.yi)
                        argument = direction * fields.cross_prod_t(pipe_proj_p.Ex, pipe_proj_p.Ey, pipe_proj_p.Ez,
                                                                   cav_proj.Hx,  cav_proj.Hy,  cav_proj.Hz)
                        D[ix_n, ix_p] = trapz(trapz(argument, mesh.xi), mesh.yi)
                        
                scenario['A'] = A
                scenario['B'] = B
                scenario['C'] = C
                scenario['D'] = D
                scenario['E'] = E
                scenario['Z'] = Z


            F = np.zeros((sim.index_max_n, 1), dtype=complex)
            G = np.zeros((sim.index_max_n, 1), dtype=complex)
            R = np.zeros((sim.index_max_n, 1), dtype=complex)
            MI = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            MV = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            MF = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            ID = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj_s = fields.projectors(mesh)
                cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
                F[ix_n, 0] = -(cavity.k_ps) / (cavity.k_0**2 - cavity.k_ps**2) * \
                    trapz((cav_proj_s.Ez)  * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                
                G[ix_n, 0] = 1j*(cavity.k_0 * cavity.Z0) / (cavity.k_0**2 - cavity.k_ps**2) * \
                    trapz((cav_proj_s.Ez) * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                
                R[ix_n, 0] = -trapz(cav_proj_s.Fz * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                
                MI[ix_n, ix_n] = (1j * cavity.k_0) / (cavity.Z0 * (cavity.k_0**2 - cavity.k_ps**2))
                MV[ix_n, ix_n] = -1j * cavity.k_ps * cavity.Z0 / cavity.k_0
                MF[ix_n, ix_n] = -1j * cavity.Z0 / cavity.k_0
                ID[ix_n, ix_n] = 1. + 0j

            II = np.linalg.inv(ID - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
                left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
                @ F   + np.linalg.inv(ID - MI @ left['D'] @ np.linalg.inv(left['A']) \
                @ left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ \
                right['B']) @ MI @ left['C'] + np.linalg.inv(ID - MI @ left['D'] @ \
                np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
                np.linalg.inv(right['A']) @ right['B']) @ MI @ right['C'] - \
                np.linalg.inv(ID - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
                left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
                @ MI @ left['D'] @ np.linalg.inv(left['A']) @ left['E'] - \
                np.linalg.inv(ID - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
                left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
                @ MI @ right['D'] @ np.linalg.inv(right['A']) @ right['E']

            # coefficients
            coeffs = {}
            coeffs['left']  = -np.linalg.inv(left['A']) @ left['E']   + np.linalg.inv(left['A']) @ left['B'] @ II
            coeffs['right'] = -np.linalg.inv(right['A']) @ right['E'] + np.linalg.inv(right['A']) @ right['B'] @ II
            coeffs['cavity_sol'] = MV @ (II - F) + G
            coeffs['cavity_irr'] = MF @ R
            
            # Impedance
            Zcav_sol = np.zeros((1, sim.index_max_n), dtype=complex)
            Zcav_irr = np.zeros((1, sim.index_max_n), dtype=complex)
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj_s = fields.projectors(mesh)
                cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
                Zcav_sol[0, ix_n] = - trapz((cav_proj_s.Ez) * np.exp(1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                Zcav_irr[0, ix_n] = - trapz(cav_proj_s.Fz * np.exp(1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
            
            
            out = 1./beam.Q * (Zcav_sol @ coeffs['cavity_sol'].squeeze() + \
                               Zcav_irr @ coeffs['cavity_irr'].squeeze() + \
                               (left['Z'] @ coeffs['left']) + \
                               (right['Z'] @ coeffs['right']))
            
            Zout.append(out)
            fout.append(f)

        fout = np.array(fout).flatten()
        Zout = np.array(Zout).flatten()
        Z_conv.append(Zout)
        print(Zout)

Z_conv = np.array(Z_conv).flatten()
plt.figure()
plt.plot(P_vec, Z_conv.real)
plt.plot(P_vec, Z_conv.imag)
plt.legend(['real', 'imag'])
plt.title('Beam current: P scan, fixed S = '+str(max(S_vec))+', Np = '+str(Np))
# plt.ylim(-40,20)

#%% Volume integration
iS = 3
iP = 15
sim = fields.simulation(index_max_p=iP, index_max_s = iS)
mesh = fields.mesh3D(sim, Np=80)
source = fields.source_gaussian(sim, beam)
source_proj=fields.projectors(mesh)
source_proj.interpolate_in_volume(source, mesh)

R = np.zeros((sim.index_max_n, 1), dtype=complex)
for ix_n in np.arange(sim.index_max_n)[:]:
    
    cavity = fields.cavity(
        sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    cavity.box = sim.b
    cav_proj = fields.projectors(mesh)
    cav_proj.interpolate_in_volume(cavity, mesh)

    argument = cav_proj.Fz * source_proj.Jz
    
    R[ix_n, 0] = -trapz(trapz(trapz(argument, axis=0, dx=mesh.dx),
          axis=0, dx=mesh.dy), axis=0, dx=mesh.dz)
    print(R)
#for iP in np.arange(2,20, 10, dtype=int):
mesh = fields.mesh(sim, Np=Np)
MF = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
Zcav_irr = np.zeros((1, sim.index_max_n), dtype=complex)

for ix_n in sim.ix_n:
    cavity = fields.cavity(
        sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    cav_proj_s = fields.projectors(mesh)
    cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
  
    R[ix_n, 0] = -trapz(cav_proj_s.Fz * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
    MF[ix_n, ix_n] = -1j * cavity.Z0 / cavity.k_0
    Zcav_irr[0, ix_n] = - trapz(cav_proj_s.Fz * np.exp(1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)    
            
                
# coefficients
coeffs = {}
coeffs['cavity_irr'] = MF @ R
print(iP, iS, Zcav_irr @ coeffs['cavity_irr'].squeeze())
#%%
mesh = fields.mesh3D(sim, Np=80)
source = fields.source_gaussian(sim, beam)
source_proj=fields.projectors(mesh)
source_proj.interpolate_in_volume(source, mesh)

for ix_n in np.arange(sim.index_max_n)[:]:
    
    cavity = fields.cavity(
        sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    cavity.box = sim.b
    cav_proj = fields.projectors(mesh)
    cav_proj.interpolate_in_volume(cavity, mesh)

    argument = cav_proj.Fz * source_proj.Jz
    
    print(trapz(trapz(trapz(argument, axis=0, dx=mesh.dx),
          axis=0, dx=mesh.dy), axis=0, dx=mesh.dz))

                           
#%%
import pandas as pd
pd.DataFrame(Zout, dtype = complex, index = fout).to_csv('../Results/out_J_indirect_gauss.csv')

#%% Plot phi-component continuity of the H-field
direction = -1

if direction == -1:
    coeffs_pipe = coeffs['left']
    zmatch = 0
else:
    coeffs_pipe = coeffs['right']
    zmatch = sim.L
    
v_cav_mmm = 0
r_vec = np.linspace(-sim.d, sim.d, 1000, endpoint=1)
for ix_n in sim.ix_n:
    cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    v_cav_mmm += (cavity.Hphi(r_vec, 0, zmatch)*II.squeeze()[ix_n] )  
        
plt.figure()
plt.plot(r_vec, v_cav_mmm.real)
plt.plot(r_vec, v_cav_mmm.imag)

r_vec = np.linspace(-sim.b, sim.b, 1000, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Hphi(r_vec, 0, zmatch)*coeffs_pipe.squeeze()[ix_p])
# v_pipe_mmm += source.Hphi((r_vec), 0, zmatch)

# plt.figure()
plt.plot(r_vec, v_pipe_mmm.real)
plt.plot(r_vec, v_pipe_mmm.imag)

# r_vec = np.linspace(source.rb, sim.b, 1000, endpoint=1)
# r_vec = np.sort(np.concatenate((-r_vec,r_vec)))
# v_pipe_mmm = 0
# for ix_p in sim.ix_p:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     v_pipe_mmm += (pipe_p.Hphi((r_vec), 0, zmatch)*coeffs_pipe.squeeze()[ix_p])
# v_pipe_mmm += source.Hphi((r_vec), 0, zmatch)

# plt.plot(r_vec, v_pipe_mmm.real, '--')
# plt.plot(r_vec, v_pipe_mmm.imag, '--')
#%% Plot z-component continuity of the E-field
direction = -1

if direction == -1:
    coeffs_pipe = coeffs['left']
    zmatch = 0
else:
    coeffs_pipe = coeffs['right']
    zmatch = sim.L
    
v_cav_mmm = 0
r_vec = np.linspace(-sim.d, sim.d, 1000, endpoint=1)
for ix_n in sim.ix_n:
    cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    v_cav_mmm += (cavity.Ez(r_vec, 0, zmatch)*coeffs['cavity_sol'].squeeze()[ix_n]  +\
                  cavity.Fz(r_vec, 0, zmatch)*coeffs['cavity_irr'].squeeze()[ix_n])  
        

plt.figure()
plt.plot(r_vec, v_cav_mmm.real)
# plt.plot(r_vec, v_cav_mmm.imag)

# r_vec = np.linspace(-source.rb, source.rb, 1000, endpoint=1)
# v_pipe_mmm = 0
# for ix_p in sim.ix_p:
#     pipe_p = fields.pipe(sim, ix_p, direction)
#     v_pipe_mmm += (pipe_p.Ez(r_vec, 0, zmatch)*coeffs_pipe.squeeze()[ix_p])
# v_pipe_mmm += source2.Er(r_vec, 0, zmatch)


#plt.figure()
# plt.plot(r_vec, v_pipe_mmm.real)
# plt.plot(r_vec, v_pipe_mmm.imag)

r_vec = np.linspace(-sim.b, sim.b, 1000, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Ez(r_vec, 0, zmatch)*coeffs_pipe.squeeze()[ix_p])


#plt.figure()
plt.plot(r_vec, v_pipe_mmm.real)
# plt.plot(r_vec, v_pipe_mmm.imag)
# plt.ylim(-10000,10000)

#%% Plot z-component of the E-field along z
rp = 0.0

direction = -1
z_vec = np.linspace(-sim.L, 0, 100, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Ez(rp, 0, z_vec)* coeffs['left'].squeeze()[ix_p])
v_pipe_mmm*=np.exp(1j * source.alpha_b * z_vec / sim.b)


plt.figure()
plt.plot(z_vec, v_pipe_mmm.real,'-k')
plt.plot(z_vec, v_pipe_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_pipe_mmm),'-b')

a1= -1./beam.Q * trapz(v_pipe_mmm, z_vec)
print(a1)

v_cav_mmm = 0
z_vec = np.linspace(0, sim.L, 100, endpoint=1)
for ix_n in sim.ix_n:
    cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    v_cav_mmm += (cavity.Ez(rp, 0, z_vec)*coeffs['cavity_sol'].squeeze()[ix_n]  +\
                  cavity.Fz(rp, 0, z_vec)*coeffs['cavity_irr'].squeeze()[ix_n])  
v_cav_mmm*=np.exp(1j * source.alpha_b * z_vec / sim.b)
corr1 = 1j*np.imag((v_cav_mmm[0] - v_pipe_mmm[-1]))
# corr1*=0

print(corr1)

plt.plot(z_vec, v_cav_mmm.real,'-k')
plt.plot(z_vec, v_cav_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_cav_mmm),'--b')

a2 = -1./beam.Q * trapz(v_cav_mmm, z_vec)
print(a2)

# v_cav_mmm -= (1j*np.imag(corr1)  *np.exp(-1j*omega0/v*z_vec)   )

# a2 = -1./beam.Q * trapz(v_cav_mmm * np.exp(1j * source.alpha_b * z_vec / sim.b), z_vec)
# print(a2)


# from scipy.constants import c
# v = beam.beta * c
# omega0 = 2*np.pi*f
#print(v/omega0 * corr1, v/omega0 * corr1 * np.exp(1j * 2*source.alpha_b * sim.L / sim.b))

# plt.plot(z_vec, v_cav_mmm.real,'-k')
# plt.plot(z_vec, v_cav_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_cav_mmm),'-b')

# plt.plot(z_vec, abs(v_cav_mmm) - abs(v_cav_mmm[0]) + abs(v_pipe_mmm[-1]),'-b')

direction = 1
z_vec = np.linspace(sim.L, 2*sim.L, 100, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Ez(rp, 0, z_vec)*coeffs['right'].squeeze()[ix_p])
v_pipe_mmm*=np.exp(1j * source.alpha_b * z_vec / sim.b)

plt.plot(z_vec, v_pipe_mmm.real,'-k')
plt.plot(z_vec, v_pipe_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_pipe_mmm),'-b')
# plt.ylim(-1000,1000)
a3 = -1./beam.Q * trapz(v_pipe_mmm, z_vec)
print(a3)
print(a1+a2+a3)

#%%Plot phi-component of the H-field along z
rp = 0.0001
direction = -1
z_vec = np.linspace(-1*sim.L, 0, 100, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Hphi(rp, 0, z_vec)* coeffs['left'].squeeze()[ix_p])
v_pipe_mmm += source.Hphi(rp, 0, z_vec)


plt.figure()
plt.plot(z_vec, v_pipe_mmm.real,'-k')
plt.plot(z_vec, v_pipe_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_pipe_mmm),'-b')

v_cav_mmm = 0
z_vec = np.linspace(0, sim.L, 100, endpoint=1)
for ix_n in sim.ix_n:
    cavity = fields.cavity(sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
    v_cav_mmm += (cavity.Hphi(rp, 0, z_vec)*II.squeeze()[ix_n])
        
plt.plot(z_vec, v_cav_mmm.real,'-k')
plt.plot(z_vec, v_cav_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_cav_mmm),'-b')
# plt.plot(z_vec, abs(v_cav_mmm) ,'-b')
print(np.imag(v_cav_mmm[0]) - np.imag(v_pipe_mmm[-1]))

direction = 1
z_vec = np.linspace(sim.L, 2*sim.L, 100, endpoint=1)
v_pipe_mmm = 0
for ix_p in sim.ix_p:
    pipe_p = fields.pipe(sim, ix_p, direction)
    v_pipe_mmm += (pipe_p.Hphi(rp, 0, z_vec)*coeffs['right'].squeeze()[ix_p])
v_pipe_mmm += source.Hphi(rp, 0, z_vec)


plt.plot(z_vec, v_pipe_mmm.real,'-k')
plt.plot(z_vec, v_pipe_mmm.imag,'-r')
# plt.plot(z_vec, abs(v_pipe_mmm),'-b')
# plt.ylim(-1000,1000)
