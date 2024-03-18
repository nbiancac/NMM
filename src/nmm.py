#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:53:17 2023

@author: nbiancac
"""
import numpy as np
from math_utils import cross_prod_t, cart2polar
from fields import Pipe, cavity_CST, cavity_project_on_axis, cavity_project, source_ring_top, source_ring_bottom, cavity
from projectors import projectors


class Mesh:

    def __init__(self, geometry, Np=10):  # constructor

        self.Np = Np
        self.xi = np.linspace(-geometry.b, geometry.b, self.Np, endpoint=True)
        self.yi = self.xi
        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)
        Wi = self.Xi + 1j * self.Yi
        self.R, self.PHI = cart2polar(Wi)
        self.Z = np.linspace(0, geometry.L, self.Np, endpoint=True)


class Mesh3D:

    def __init__(self, sim, Np=10):  # constructor

        self.Np = Np
        self.xi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.yi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.zi = np.linspace(0, sim.L, self.Np, endpoint=True)
        self.dx = np.diff(self.xi)[0]
        self.dy = np.diff(self.yi)[0]
        self.dz = np.diff(self.zi)[0]
        self.Xi, self.Yi, self.Z = np.meshgrid(self.xi, self.yi, self.zi)
        Wi = self.Xi + 1j * self.Yi
        self.R, self.PHI = cart2polar(Wi)


class Materials:

    def __init__(self, sigma=0):  # constructor

        self.sigma = sigma


class Geometry:

    def __init__(self, b=5e-2, t=5e-2, L=1e-2):  # constructor

        self.b = b
        self.t = t
        self.L = L
        self.d = self.t + self.b


class Pillbox:

    def __init__(self, radius=5e-2, thickness=5e-2, length=1e-2):  # constructor

        self.b = radius
        self.t = thickness
        self.L = length
        self.d = self.t + self.b


class CST_object:

    def __init__(self, radius=5e-2, thickness=0, length=1e-2):  # constructor

        self.b = radius
        self.t = thickness
        self.L = length
        self.d = self.t + self.b


class Beam:

    def __init__(self, beta=0.9999, charge=1):  # constructor
        self.beta = beta
        self.gamma = 1. / np.sqrt(1 - self.beta ** 2)
        self.charge = charge


class Mode:
    def __init__(self, is_analytical=False, index_max_p=3, max_mode_number=2, split_rs=False, **kwargs):  # constructor

        import itertools
        self.is_analytical = is_analytical
        self.index_max_p = index_max_p
        self.ix_p = np.arange(self.index_max_p)

        if is_analytical:

            if split_rs and ('index_max_r' and 'index_max_s' in kwargs):
                self.index_max_s = kwargs['index_max_s']
                self.index_max_r = kwargs['index_max_r']
            else:
                self.index_max_s = max_mode_number
                self.index_max_r = max_mode_number
            self.ix_s = np.arange(self.index_max_s)
            self.ix_r = np.arange(self.index_max_r)
            self.ix_pairs_n = list(itertools.product(
                np.arange(self.index_max_r), np.arange(self.index_max_s)))
            self.index_max_n = len(self.ix_pairs_n)
            self.ix_n = np.arange(self.index_max_n)
            self.datadir = None
        else:  # modes are from CST
            if 'list_modes' in kwargs:
                self.x_n = kwargs['list_modes']
                self.index_max_n = len(self.x_n)
                self.ix_n = np.arange(self.index_max_n)
                self.datadir = './cst/'
            else:
                self.index_max_n = max_mode_number
                self.ix_n = np.arange(self.index_max_n)
                self.x_n = self.ix_n + 1
                self.datadir = './cst/'


class Simulation:

    def __init__(self, frequency=2e9, mode=Mode, integration='indirect', geometry=Geometry,
                 materials=Materials, beam=Beam, mesh=Mesh):

        self.f = frequency
        self.index_max_p = mode.index_max_p
        self.ix_p = np.arange(self.index_max_p)
        self.index_max_n = mode.index_max_n
        self.ix_n = mode.ix_n
        self.b = geometry.b
        self.t = geometry.t
        self.L = geometry.L
        self.d = geometry.d
        self.rb = self.b / 10
        self.sigma = materials.sigma
        self.integration = integration
        self.materials = materials
        self.beam = beam
        self.mesh = mesh
        self.Np = mesh.Np
        self.datadir = mode.datadir
        self.mode = mode

    def preload_matrixes(self):

        from scipy.integrate import trapz

        print(f"\nModes in the pipes {self.index_max_p}.\
        \nModes in the cavity {self.index_max_n}.\
        \nNumber of points {self.Np}.")

        self.left = {'direction': -1,
                     'zmatch': 0,
                     'ev': [],
                     'vp': [],
                     'str': 'left'
                     }

        self.right = {'direction': 1,
                      'zmatch': self.L,
                      'ev': [],
                      'vp': [],
                      'str': 'right'
                      }

        WF = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)
        print('Computing eigenmode fields at boundaries.')
        print(f'Number of eigenmodes: {len(self.ix_n)}')
        for scenario in [self.left, self.right]:
            zmatch = scenario['zmatch']
            direction = scenario['direction']
            for ix_n in self.ix_n:
                print(f"\r >> {scenario['str']}, mode number {ix_n + 1}/{len(self.ix_n)}", end="")
                if self.mode.is_analytical:
                    # cavity_n = sim.geometry.generate_mode_field(mode)
                    cavity_n = cavity(
                        self, self.mode.ix_pairs_n[ix_n][0], self.mode.ix_pairs_n[ix_n][1])
                    cav_proj = projectors(self.mesh)
                    cav_proj.interpolate_at_boundary(cavity_n, self.mesh, zmatch)
                else:
                    # cavity_n = sim.geometry.load_mode_field(mode)
                    cav_proj = cavity_project(self.mode.x_n[ix_n], self.mesh, zmatch, radius_CST=5, stepsize_CST=0.1,
                                              datadir=self.datadir)
                scenario['ev'].append(cav_proj)
                argument = abs(cav_proj.Hx) ** 2 + abs(cav_proj.Hy) ** 2
                WF[ix_n, ix_n] = trapz(trapz(argument, self.mesh.xi), self.mesh.yi)
            scenario['WF'] = WF

            print("\n")
            for ix_p in self.ix_p:
                print(f"\r >> {scenario['str']}, pipe number {ix_p + 1}/{len(self.ix_p)}", end="")
                pipe_p = Pipe(self, ix_p, direction)
                pipe_proj_p = projectors(self.mesh)
                pipe_proj_p.interpolate_at_boundary(pipe_p, self.mesh, zmatch)
                scenario['vp'].append(pipe_proj_p)
            print("\n")

        print("Computing frequency independent matrixes.")
        A0 = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)
        M_alpha_p_tilde = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)
        for ix_p in self.ix_p:
            pipe_p = Pipe(self, ix_p, direction)
            pipe_proj_p = scenario['vp'][ix_p]
            grad_x_p, grad_y_p = pipe_proj_p.Hx, pipe_proj_p.Hy
            M_alpha_p_tilde[ix_p, ix_p] = pipe_p.alpha_p_tilde
            for ix_q in self.ix_p:
                pipe_proj_q = scenario['vp'][ix_q]
                grad_x_q, grad_y_q = pipe_proj_q.Hx, pipe_proj_q.Hy
                A0[ix_p, ix_q] = trapz(trapz((pipe_proj_p.Hx) * grad_x_q + (pipe_proj_p.Hy) * grad_y_q, self.mesh.xi),
                                       self.mesh.yi)

        for scenario in [self.left, self.right]:
            direction = scenario['direction']
            zmatch = scenario['zmatch']
            B0 = np.zeros((self.index_max_p, self.index_max_n), dtype=complex)
            D0 = np.zeros((self.index_max_n, self.index_max_p), dtype=complex)
            for ix_p in self.ix_p:
                pipe_proj_p = scenario['vp'][ix_p]
                grad_x_p, grad_y_p = pipe_proj_p.Hx, pipe_proj_p.Hy
                for ix_n in self.ix_n:
                    cav_proj = scenario['ev'][ix_n]
                    B0[ix_p, ix_n] = trapz(trapz(cav_proj.Hx * grad_x_p + cav_proj.Hy * grad_y_p, self.mesh.xi),
                                           self.mesh.yi)

                    argument = direction * cross_prod_t(pipe_proj_p.Ex, pipe_proj_p.Ey, pipe_proj_p.Ez,
                                                        cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                    D0[ix_n, ix_p] = trapz(trapz(argument, self.mesh.xi), self.mesh.yi)

            scenario['A0'] = A0
            scenario['B0'] = B0
            scenario['D0'] = D0

    def compute_impedance(self):

        from scipy.integrate import trapz
        from scipy.constants import mu_0

        for scenario in [self.left, self.right]:
            direction = scenario['direction']
            zmatch = scenario['zmatch']

            # source = source_point((self, self.beam)
            source = source_ring_top(self, self.beam)
            source_proj = projectors(self.mesh)
            source_proj.interpolate_at_boundary(source, self.mesh, zmatch)
            source2 = source_ring_bottom(self, self.beam)
            source_proj2 = projectors(self.mesh)
            source_proj2.interpolate_at_boundary(source2, self.mesh, zmatch)
            source_proj.add_fields(source_proj2)

            # matching matrixes
            C = np.zeros((self.index_max_n, 1), dtype=complex)
            E = np.zeros((self.index_max_p, 1), dtype=complex)
            Z = np.zeros((1, self.index_max_p), dtype=complex)
            M_alpha_p_tilde = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)

            for ix_p in self.ix_p:
                pipe_p = Pipe(self, ix_p, direction)
                pipe_proj_p = scenario['vp'][ix_p]
                from scipy.special import jn
                # if np.imag(pipe_p.alpha_p_tilde ) < 0:
                Z[0, ix_p] = - direction * 1j * self.b * jn(0, source.rb * pipe_p.alpha_p / self.b) / \
                             (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (
                                     source.alpha_b - direction * pipe_p.alpha_p_tilde)) * \
                             np.exp(1j * zmatch * (source.alpha_b) / self.b)
                E[ix_p, 0] = pipe_p.alpha_0 * trapz(trapz(source_proj.Hx * pipe_proj_p.Hx \
                                                          + source_proj.Hy * pipe_proj_p.Hy, self.mesh.xi),
                                                    self.mesh.yi)
                M_alpha_p_tilde[ix_p, ix_p] = pipe_p.alpha_p_tilde

            for ix_n in self.ix_n:
                cav_proj = scenario['ev'][ix_n]
                argument = direction * cross_prod_t(source_proj.Ex, source_proj.Ey, source_proj.Ez,
                                                    cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                C[ix_n, 0] = trapz(trapz(argument, self.mesh.xi), self.mesh.yi)

            scenario['A'] = scenario['A0'] * pipe_p.alpha_0 ** 2
            scenario['B'] = scenario['B0'] * pipe_p.alpha_0
            scenario['D'] = scenario['D0'] @ M_alpha_p_tilde
            scenario['Z'] = Z
            scenario['E'] = E
            scenario['C'] = C

        self.F = np.zeros((self.index_max_n, 1), dtype=complex)
        self.G = np.zeros((self.index_max_n, 1), dtype=complex)
        self.R = np.zeros((self.index_max_n, 1), dtype=complex)
        self.MI = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)
        self.MV = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)
        self.MF = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)
        self.ID = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)
        self.W = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)

        for ix_n in self.ix_n:

            if self.mode.is_analytical:
                # cavity_n = sim.geometry.generate_mode_field(mode)
                cavity_ = cavity(
                    self, self.mode.ix_pairs_n[ix_n][0], self.mode.ix_pairs_n[ix_n][1])
                cav_proj_s = projectors(self.mesh)
                cav_proj_s.interpolate_on_axis(cavity_, self.mesh, source.rb, 0)
            else:
                # cavity_n = sim.geometry.load_mode_field(mode)
                cavity_ = cavity_CST(self, mode_num=self.mode.x_n[ix_n], datadir=self.datadir)
                # cavity_n = sim.geometry.load_mode_field(mode)
                cav_proj_s = cavity_project_on_axis(self.mode.x_n[ix_n], self.mesh, datadir=self.datadir)

            self.F[ix_n, 0] = -cavity_.k_rs / (cavity_.k_0 ** 2 - cavity_.k_rs ** 2) * \
                              trapz(
                                  cav_proj_s.Ez * self.beam.charge * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b),
                                  self.mesh.Z)

            self.G[ix_n, 0] = 1j * (cavity_.k_0 * cavity_.Z0) / (cavity_.k_0 ** 2 - cavity_.k_rs ** 2) * \
                              trapz(
                                  cav_proj_s.Ez * self.beam.charge * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b),
                                  self.mesh.Z)

            self.R[ix_n, 0] = -trapz(cav_proj_s.Fz * self.beam.charge * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b),
                                     self.mesh.Z)

            self.MI[ix_n, ix_n] = (1j * cavity_.k_0) / (cavity_.Z0 * (cavity_.k_0 ** 2 - cavity_.k_rs ** 2))
            self.MV[ix_n, ix_n] = -1j * cavity_.k_rs * cavity_.Z0 / cavity_.k_0
            self.MF[ix_n, ix_n] = -1j * cavity_.Z0 / cavity_.k_0
            self.ID[ix_n, ix_n] = 1. + 0j

            if self.materials.sigma != 0:
                Zw = np.sqrt(mu_0 * 2 * np.pi * self.f / 2 / self.materials.sigma)
                loss = (1 + 1j) * 2 * np.pi * self.f * mu_0 / (cavity_.Q) \
                       - (1 + 1j) * Zw * (self.left['WF'][ix_n, ix_n] + self.right['WF'][ix_n, ix_n])
                self.W[ix_n, ix_n] = self.MI[ix_n, ix_n] * loss

        self.II = np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ np.linalg.inv(self.left['A']) @ \
                                self.left['B'] - self.MI @ self.right['D'] @ np.linalg.inv(self.right['A']) @
                                self.right['B']) \
                  @ self.F + np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ \
                                           np.linalg.inv(self.left['A']) @ self.left['B'] - self.MI @ self.right['D'] @ \
                                           np.linalg.inv(self.right['A']) @ self.right['B']) @ self.MI @ self.left[
                      'C'] + \
                  np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ np.linalg.inv(self.left['A']) @ \
                                self.left['B'] - self.MI @ self.right['D'] @ np.linalg.inv(self.right['A']) @
                                self.right['B']) \
                  @ self.MI @ self.right['C'] - np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ \
                                                              np.linalg.inv(self.left['A']) @ self.left['B'] - self.MI @
                                                              self.right['D'] @ \
                                                              np.linalg.inv(self.right['A']) @ self.right[
                                                                  'B']) @ self.MI @ self.left['D'] @ \
                  np.linalg.inv(self.left['A']) @ self.left['E'] - np.linalg.inv(self.ID - self.W - self.MI @ \
                                                                                 self.left['D'] @ np.linalg.inv(
            self.left['A']) @ self.left['B'] - self.MI @ self.right['D'] @ \
                                                                                 np.linalg.inv(self.right['A']) @
                                                                                 self.right['B']) @ self.MI @ \
                  self.right['D'] @ \
                  np.linalg.inv(self.right['A']) @ self.right['E']

        # coefficients
        coeffs = {}
        coeffs['left'] = -np.linalg.inv(self.left['A']) @ self.left['E'] + np.linalg.inv(self.left['A']) @ self.left[
            'B'] @ self.II
        coeffs['right'] = -np.linalg.inv(self.right['A']) @ self.right['E'] + np.linalg.inv(self.right['A']) @ \
                          self.right['B'] @ self.II
        coeffs['cavity_sol'] = self.MV @ (self.II - self.F) + self.G
        coeffs['cavity_irr'] = self.MF @ self.R

        # Integration
        Zcav_sol = np.zeros((1, self.index_max_n), dtype=complex)
        Zcav_irr = np.zeros((1, self.index_max_n), dtype=complex)
        for ix_n in self.ix_n:

            if self.mode.is_analytical:
                cavity_ = cavity(
                    self, self.mode.ix_pairs_n[ix_n][0], self.mode.ix_pairs_n[ix_n][1])
                cav_proj_s = projectors(self.mesh)
                if self.integration == 'indirect':
                    dir_int = 0
                    cav_proj_s.interpolate_on_axis(cavity_, self.mesh, self.geometry.b, 0)
                else:
                    dir_int = 1
                    cav_proj_s.interpolate_on_axis(cavity_, self.mesh, self.rb, 0)
            else:
                cav_proj_s = cavity_project_on_axis(self.mode.x_n[ix_n], self.mesh, datadir=self.datadir)
                dir_int = 1

            Zcav_sol[0, ix_n] = - trapz(cav_proj_s.Ez * np.exp(1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z)
            Zcav_irr[0, ix_n] = - trapz(cav_proj_s.Fz * np.exp(1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z)

        self.Z = 1. / self.beam.charge * (Zcav_sol @ coeffs['cavity_sol'] + \
                                     Zcav_irr @ coeffs['cavity_irr'] + \
                                     (dir_int) * (self.left['Z'] @ coeffs['left']) + \
                                     (dir_int) * (self.right['Z'] @ coeffs['right']))
        self.coeffs = coeffs
        self.Zcav_sol = Zcav_sol
        self.Zcav_irr = Zcav_irr
