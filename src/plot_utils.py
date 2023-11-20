#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:11:22 2023

@author: nbiancac
"""
import matplotlib.pyplot as plt

def plot_at_boundary(self, mesh):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    cf0_ = ax[0].contourf(mesh.Xi, mesh.Yi, (abs(self.Er)), 150)
    cf1_ = ax[1].contourf(mesh.Xi, mesh.Yi, (abs(self.Ez)), 150)
    cf2_ = ax[2].contourf(mesh.Xi, mesh.Yi, (abs(self.Hphi)), 150)
    ax[0].set_title('|$E_r$|')
    ax[1].set_title('|$E_z$|')
    ax[2].set_title('|$H_{phi}$|')
    fig.colorbar(cf0_)
    fig.colorbar(cf1_)
    fig.colorbar(cf2_)
    plt.tight_layout()
    self.fig = fig