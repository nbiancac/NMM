#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:10:17 2023

@author: nbiancac
"""
import numpy as np
import pandas as pd

class projectors:

    def __init__(self, mesh):

        self.Hx = np.zeros(mesh.R.shape, dtype=complex)
        self.Hy = np.zeros(mesh.R.shape, dtype=complex)
        self.Hz = np.zeros(mesh.R.shape, dtype=complex)
        self.Ex = np.zeros(mesh.R.shape, dtype=complex)
        self.Ey = np.zeros(mesh.R.shape, dtype=complex)
        self.Ez = np.zeros(mesh.R.shape, dtype=complex)
        self.Fx = np.zeros(mesh.R.shape, dtype=complex)
        self.Fy = np.zeros(mesh.R.shape, dtype=complex)
        self.Fz = np.zeros(mesh.R.shape, dtype=complex)
        # self.Gx = np.zeros(mesh.R.shape, dtype = complex)
        # self.Gy = np.zeros(mesh.R.shape, dtype = complex)
        # self.Gz = np.zeros(mesh.R.shape, dtype = complex)
        self.Hr = np.zeros(mesh.R.shape, dtype=complex)
        self.Hphi = np.zeros(mesh.R.shape, dtype=complex)
        self.Er = np.zeros(mesh.R.shape, dtype=complex)
        self.Ephi = np.zeros(mesh.R.shape, dtype=complex)
        self.Fr = np.zeros(mesh.R.shape, dtype=complex)
        self.Fphi = np.zeros(mesh.R.shape, dtype=complex)
        # self.Gr   = np.zeros(mesh.R.shape, dtype = complex)
        # self.Gphi = np.zeros(mesh.R.shape, dtype = complex)
        self.Jz = np.zeros(mesh.R.shape, dtype=complex)

    def interpolate_at_surface(self, obj, mesh, rmatch):
        self.Hphi = obj.Hphi(rmatch, 0, mesh.Z)
        self.Hr = obj.Hr(rmatch, 0, mesh.Z)
        self.Hz = obj.Hz(rmatch, mesh.PHI, mesh.Z)

        self.Ephi = obj.Ephi(rmatch, mesh.PHI, mesh.Z)
        self.Er = obj.Er(rmatch, mesh.PHI, mesh.Z)
        self.Ez = obj.Ez(rmatch, mesh.PHI, mesh.Z)

        self.Fphi = obj.Fphi(rmatch, mesh.PHI, mesh.Z)
        self.Fr = obj.Fr(rmatch, mesh.PHI, mesh.Z)
        self.Fz = obj.Fz(rmatch, mesh.PHI, mesh.Z)

        # self.Gphi = obj.Gphi(mesh.R,mesh.PHI, zmatch)
        # self.Gr = obj.Gr(mesh.R,mesh.PHI, zmatch)
        # self.Gz =  obj.Gz(mesh.R,mesh.PHI, zmatch)
        
    def interpolate_at_boundary(self, obj, mesh, zmatch):
        cond = ((mesh.R <= obj.box)).astype(int) * \
            ((mesh.R >= (obj.rb))).astype(int)
        c, s = np.cos(mesh.PHI), np.sin(mesh.PHI)
        self.Hphi = cond*obj.Hphi(mesh.R, mesh.PHI, zmatch)
        self.Hr = cond*obj.Hr(mesh.R, mesh.PHI, zmatch)
        self.Hphi[np.isnan(self.Hphi)] = 0
        self.Hr[np.isnan(self.Hr)] = 0
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy = self.Hphi * c + self.Hr * s
        self.Hz = cond * obj.Hz(mesh.R, mesh.PHI, zmatch)

        self.Ephi = cond*obj.Ephi(mesh.R, mesh.PHI, zmatch)
        self.Er = cond*obj.Er(mesh.R, mesh.PHI, zmatch)
        self.Ephi[np.isnan(self.Ephi)] = 0
        self.Er[np.isnan(self.Er)] = 0
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey = self.Ephi * c + self.Er * s
        self.Ez = cond*obj.Ez(mesh.R, mesh.PHI, zmatch)

        self.Fphi = cond*obj.Fphi(mesh.R, mesh.PHI, zmatch)
        self.Fr = cond*obj.Fr(mesh.R, mesh.PHI, zmatch)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy = self.Fphi * c + self.Fr * s
        self.Fz = cond*obj.Fz(mesh.R, mesh.PHI, zmatch)

        # self.Gphi = cond*obj.Gphi(mesh.R,mesh.PHI, zmatch)
        # self.Gr = cond*obj.Gr(mesh.R,mesh.PHI, zmatch)
        # self.Gx = -self.Gphi * s + self.Gr * c
        # self.Gy =  self.Gphi * c + self.Gr * s
        # self.Gz =  cond*obj.Gz(mesh.R,mesh.PHI, zmatch)

    def interpolate_in_volume(self, obj, mesh):
        cond = ((mesh.R < obj.box)).astype(int)
        c, s = np.cos(mesh.PHI), np.sin(mesh.PHI)
        self.Hphi = cond*obj.Hphi(mesh.R, mesh.PHI, mesh.Z)
        self.Hr = cond*obj.Hr(mesh.R, mesh.PHI, mesh.Z)
        self.Hphi[np.isnan(self.Hphi)] = 0
        self.Hr[np.isnan(self.Hr)] = 0
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy = self.Hphi * c + self.Hr * s
        self.Hz = cond * obj.Hz(mesh.R, mesh.PHI, mesh.Z)

        self.Ephi = cond*obj.Ephi(mesh.R, mesh.PHI, mesh.Z)
        self.Er = cond*obj.Er(mesh.R, mesh.PHI, mesh.Z)
        self.Ephi[np.isnan(self.Ephi)] = 0
        self.Er[np.isnan(self.Er)] = 0
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey = self.Ephi * c + self.Er * s
        self.Ez = cond*obj.Ez(mesh.R, mesh.PHI, mesh.Z)

        self.Fphi = cond*obj.Fphi(mesh.R, mesh.PHI, mesh.Z)
        self.Fr = cond*obj.Fr(mesh.R, mesh.PHI, mesh.Z)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy = self.Fphi * c + self.Fr * s
        self.Fz = cond*obj.Fz(mesh.R, mesh.PHI, mesh.Z)

        self.Jz = cond*obj.Jz(mesh.R, mesh.PHI, mesh.Z)

        # self.Gphi = cond*obj.Gphi(mesh.R,mesh.PHI, mesh.Z)
        # self.Gr = cond*obj.Gr(mesh.R,mesh.PHI, mesh.Z)
        # self.Gx = -self.Gphi * s + self.Gr * c
        # self.Gy =  self.Gphi * c + self.Gr * s
        # self.Gz =  cond*obj.Gz(mesh.R,mesh.PHI, mesh.Z)

    def interpolate_on_axis(self, obj, mesh, rmatch, phimatch):
        cond = int(rmatch <= obj.box)
        c, s = np.cos(phimatch), np.sin(phimatch)

        self.Hphi = cond*obj.Hphi(rmatch, phimatch, mesh.Z)
        self.Hr = cond*obj.Hr(rmatch, phimatch, mesh.Z)
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy = self.Hphi * c + self.Hr * s
        self.Hz = cond * obj.Hz(rmatch, phimatch, mesh.Z)

        self.Ephi = cond*obj.Ephi(rmatch, phimatch, mesh.Z)
        self.Er = cond*obj.Er(rmatch, phimatch, mesh.Z)
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey = self.Ephi * c + self.Er * s
        self.Ez = cond*obj.Ez(rmatch, phimatch, mesh.Z)

        self.Fphi = cond*obj.Fphi(rmatch, phimatch, mesh.Z)
        self.Fr = cond*obj.Fr(rmatch, phimatch, mesh.Z)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy = self.Fphi * c + self.Fr * s
        self.Fz = cond*obj.Fz(rmatch, phimatch, mesh.Z)

        self.Jz = cond*obj.Jz(rmatch, phimatch, mesh.Z)

    def dump_components(self, mesh, dire='./'):
        import os
        dire += '/'
        os.system('mkdir -p '+dire)
        pd.DataFrame(self.Ephi).to_csv(dire+'Ephi.csv')
        pd.DataFrame(self.Er).to_csv(dire+'Er.csv')
        pd.DataFrame(self.Ex).to_csv(dire+'Ex.csv')
        pd.DataFrame(self.Ey).to_csv(dire+'Ey.csv')
        pd.DataFrame(self.Ez).to_csv(dire+'Ez.csv')
        pd.DataFrame(self.Hphi).to_csv(dire+'Hphi.csv')
        pd.DataFrame(self.Hr).to_csv(dire+'Hr.csv')
        pd.DataFrame(self.Hx).to_csv(dire+'Hx.csv')
        pd.DataFrame(self.Hy).to_csv(dire+'Hy.csv')
        pd.DataFrame(self.Hz).to_csv(dire+'Hz.csv')
        pd.DataFrame(self.Fx).to_csv(dire+'Fx.csv')
        pd.DataFrame(self.Fy).to_csv(dire+'Fy.csv')
        pd.DataFrame(self.Fz).to_csv(dire+'Fz.csv')
        pd.DataFrame(mesh.Xi).to_csv(dire+'Mesh_x.csv')
        pd.DataFrame(mesh.Yi).to_csv(dire+'Mesh_y.csv')

    def load_components(self, mesh, dire='./'):
        dire += '/'
        self.Ephi = pd.read_csv(
            dire+'Ephi.csv', index_col=0).astype(complex).values
        self.Er = pd.read_csv(
            dire+'Er.csv', index_col=0).astype(complex).values
        self.Ex = pd.read_csv(
            dire+'Ex.csv', index_col=0).astype(complex).values
        self.Ey = pd.read_csv(
            dire+'Ey.csv', index_col=0).astype(complex).values
        self.Ez = pd.read_csv(
            dire+'Ez.csv', index_col=0).astype(complex).values
        self.Hphi = pd.read_csv(
            dire+'Hphi.csv', index_col=0).astype(complex).values
        self.Hr = pd.read_csv(
            dire+'Hr.csv', index_col=0).astype(complex).values
        self.Hx = pd.read_csv(
            dire+'Hx.csv', index_col=0).astype(complex).values
        self.Hy = pd.read_csv(
            dire+'Hy.csv', index_col=0).astype(complex).values
        self.Hz = pd.read_csv(
            dire+'Hz.csv', index_col=0).astype(complex).values
        self.Fx = pd.read_csv(
            dire+'Fx.csv', index_col=0).astype(complex).values
        self.Fy = pd.read_csv(
            dire+'Fy.csv', index_col=0).astype(complex).values
        self.Fz = pd.read_csv(
            dire+'Fz.csv', index_col=0).astype(complex).values
        
    def add_fields(self, obj2):

        self.Hphi = self.Hphi + obj2.Hphi
        self.Hr = self.Hr + obj2.Hr
        self.Hx = self.Hx + obj2.Hx
        self.Hy = self.Hy + obj2.Hy
        self.Hz = self.Hz + obj2.Hz

        self.Ephi = self.Ephi + obj2.Ephi
        self.Er = self.Er + obj2.Er
        self.Ex = self.Ex + obj2.Ex
        self.Ey = self.Ey + obj2.Ey
        self.Ez = self.Ez + obj2.Ez

        self.Fphi = self.Fphi + obj2.Fphi
        self.Fr = self.Fr + obj2.Fr
        self.Fx = self.Fx + obj2.Fx
        self.Fy = self.Fy + obj2.Fy
        self.Fz = self.Fz + obj2.Fz