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
from scipy.constants import mu_0

plt.close('all')
beam = fields.Beam() # initialize beam parameters

P_max = 9
S_max = 1
R_vec = np.arange(1, 10, 1, dtype=int)

Np = 100


plt.close('all')

beam = fields.Beam(beta=0.9999)
beam.Q = 1

sim = fields.simulation(index_max_p=P_max, index_max_r=np.max(R_vec), index_max_s = S_max)
mesh = fields.Mesh(sim, Np=Np)

left = {'direction': -1,
        'zmatch': 0,
        'ev': [],
        'ev_all': [],
        }

right = {'direction': 1,
         'zmatch': sim.L,
         'ev': [],
         'ev_all': []
         }

print('Computing eigenmode fields at boundaries (only once).')
print(f'Number of eigenmodes: {len(sim.ix_n)}')
for scenario in [left, right]:
    zmatch = scenario['zmatch']
    for ix_n in sim.ix_n:
        cavity = fields.cavity(
            sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
        cav_proj = fields.projectors(mesh)
        cav_proj.interpolate_at_boundary(cavity, mesh, zmatch)
        scenario['ev_all'].append(cav_proj)

Z_conv = [];
for iR in R_vec:
        Zout = []
        fout = []
        sim = fields.simulation(index_max_p=P_max, index_max_r=iR, index_max_s = S_max)

        for scenario in [left, right]:
            scenario['ev'] = scenario['ev_all'][0:iR]
        

        print(f"Modes in the pipes {sim.index_max_p}.\
        \nModes in the cavity {sim.index_max_n}.\
        \nNumber of points {Np}.")                        
        # for f in np.sort(np.concatenate(([1e6], np.linspace(1e9,6e9,100, endpoint=1)))):   
        # for f in np.linspace(2e9,2.15e9,31, endpoint=1):   
        # for f in np.linspace(4.74e9,4.85e9,51, endpoint=1):   
        # for f in np.linspace(1e8,7e9,30, endpoint=1):   
        #range1 = np.linspace(1e9,7e9,100, endpoint=1)
        # range2 = np.linspace(4.2e9,4.2e9,1, endpoint=1)
        # range1 = np.arange(1e7,7e9,1e7)
        # range1 = np.array([1.3871436e9])
        range1 = np.array([4.2e9])
        for f in np.unique(np.sort(np.concatenate((range1, range1)))): 

        # for f in [2.29e9]:
            f"frequency {f/1e9} GHz"
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
                B = np.zeros((sim.index_max_p, sim.index_max_n), dtype=complex)
                C = np.zeros((sim.index_max_n, 1), dtype=complex)
                D = np.zeros((sim.index_max_n, sim.index_max_p), dtype=complex)
                WF = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
                E = np.zeros((sim.index_max_p, 1), dtype=complex)
                Z = np.zeros((1, sim.index_max_p), dtype=complex)

                source = fields.source(sim, beam)          
                source_proj = fields.projectors(mesh)
                source_proj.interpolate_at_boundary(source, mesh, zmatch)

                # source = fields.source_ring_top(sim, beam)
                # source_proj = fields.projectors(mesh)
                # source_proj.interpolate_at_boundary(source, mesh, zmatch)
                # source2 = fields.source_ring_bottom(sim, beam)
                # source_proj2 = fields.projectors(mesh)
                # source_proj2.interpolate_at_boundary(source2, mesh, zmatch)
                # source_proj.add_fields(source_proj2)

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
                        argument = abs(cav_proj.Hx)**2 + abs(cav_proj.Hy)**2 
                        WF[ix_n, ix_n] = trapz(trapz(argument, mesh.xi), mesh.yi)

                
                scenario['A'] = A
                scenario['B'] = B
                scenario['C'] = C
                scenario['D'] = D
                scenario['E'] = E
                scenario['Z'] = Z
                scenario['WF'] = WF

            F = np.zeros((sim.index_max_n, 1), dtype=complex)
            G = np.zeros((sim.index_max_n, 1), dtype=complex)
            R = np.zeros((sim.index_max_n, 1), dtype=complex)
            MI = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            MV = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            MF = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            ID = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            loss  = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj_s = fields.projectors(mesh)
                cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
                F[ix_n, 0] = -(cavity.k_rs) / (cavity.k_0**2 - cavity.k_rs**2) * \
                    trapz((cav_proj_s.Ez)  * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                
                G[ix_n, 0] = 1j*(cavity.k_0 * cavity.Z0) / (cavity.k_0**2 - cavity.k_rs**2) * \
                    trapz((cav_proj_s.Ez) * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
                
                R[ix_n, 0] = -trapz(cav_proj_s.Fz * beam.Q * np.exp(-1j * source.alpha_b * mesh.Z / sim.b) , mesh.Z)
                
                MI[ix_n, ix_n] = (1j * cavity.k_0) / (cavity.Z0 * (cavity.k_0**2 - cavity.k_rs**2))
                MV[ix_n, ix_n] = -1j * cavity.k_rs * cavity.Z0 / cavity.k_0
                MF[ix_n, ix_n] = -1j * cavity.Z0 / cavity.k_0
                ID[ix_n, ix_n] = 1. + 0j
                loss[ix_n, ix_n] = 0#(1+1j) * cavity.omega_rs * mu_0 / (cavity.Q)  \
                       # - (1+1j) / sim.sigma / cavity.deltas  * (left['WF'][ix_n,ix_n] + right['WF'][ix_n,ix_n])
                
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

            W  = np.zeros((sim.index_max_n, sim.index_max_n), dtype=complex)
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj_s = fields.projectors(mesh)
                cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)

                Zw = np.sqrt(mu_0 * 2 * np.pi * sim.f / 2 /sim.sigma)
                Zw_rs = np.sqrt(mu_0 * cavity.omega_rs / 2 / sim.sigma)
                normaz = 1#(II.T @ np.diag(left['WF'])) / (II[ix_n] * np.diag(left['WF'])[ix_n]) #3.8472506073#
                #normaz = (II.T @ np.diag(loss))
                # normaz = (II[] @ np.diag(loss))
                # print(normaz)
                fact = Zw / Zw_rs * normaz
                W[ix_n, ix_n] = fact * MI[ix_n, ix_n] * loss[ix_n, ix_n]
                

            II = np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
                left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
                @ F   + np.linalg.inv(ID - W - MI @ left['D'] @ \
                np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
                np.linalg.inv(right['A']) @ right['B']) @ MI @ left['C'] + \
                np.linalg.inv(ID - W - MI @ left['D'] @ np.linalg.inv(left['A']) @ \
                left['B'] - MI @ right['D'] @ np.linalg.inv(right['A']) @ right['B']) \
                @ MI @ right['C'] - np.linalg.inv(ID - W - MI @ left['D'] @ \
                np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
                np.linalg.inv(right['A']) @ right['B']) @ MI @ left['D'] @ \
                np.linalg.inv(left['A']) @ left['E'] - np.linalg.inv(ID - W - MI @ \
                left['D'] @ np.linalg.inv(left['A']) @ left['B'] - MI @ right['D'] @ \
                np.linalg.inv(right['A']) @ right['B']) @ MI @ right['D'] @ \
                np.linalg.inv(right['A']) @ right['E']
            
            # coefficients
            coeffs = {}
            coeffs['left']  = -np.linalg.inv(left['A']) @ left['E'] + np.linalg.inv(left['A']) @ left['B'] @ II
            coeffs['right'] = -np.linalg.inv(right['A']) @ right['E'] + np.linalg.inv(right['A']) @ right['B'] @ II
            coeffs['cavity_sol'] = MV @ (II - F) + G
            coeffs['cavity_irr'] = MF @ R    
            
            # Impedance
            from scipy.constants import epsilon_0
            Zcav_sol = np.zeros((1, sim.index_max_n), dtype=complex)
            Zcav_irr = np.zeros((1, sim.index_max_n), dtype=complex)
            for ix_n in sim.ix_n:
                cavity = fields.cavity(
                    sim, sim.ix_pairs_n[ix_n][0], sim.ix_pairs_n[ix_n][1])
                cav_proj_s = fields.projectors(mesh)
                cav_proj_s.interpolate_on_axis(cavity, mesh, source.rb, 0)
                Zcav_sol[0, ix_n] = - trapz(cav_proj_s.Ez * np.exp(1j * source.alpha_b * mesh.Z / sim.b), mesh.Z) 
                Zcav_irr[0, ix_n] = - trapz(cav_proj_s.Fz * np.exp(1j * source.alpha_b * mesh.Z / sim.b), mesh.Z)
            

            out = 1./beam.Q * (Zcav_sol @ coeffs['cavity_sol'] + \
                               Zcav_irr @ coeffs['cavity_irr'] + \
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
plt.plot(R_vec, Z_conv.real)
plt.legend(['real', 'imag'])
plt.title('Beam current: P scan, fixed P = '+str(max(sim.index_max_p))+', Np = '+str(Np))
plt.xlabel('Cavity modes')
plt.ylim(0,120)
plt.tight_layout()  