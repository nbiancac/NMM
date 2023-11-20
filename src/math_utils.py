#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:49:07 2023

@author: nbiancac
"""

def cross_z(x, y):
    '''
    Cross product of transverse field with components (x,y) with z. 
    E.g.: E_t x z_0. If E_t has components (x,y), then E_t x z_0 has components (y,-x)
    Parameters
    ----------
    x : vector
        direction in x
    y : vector
        direction in y
    Returns
    -------
    x, y vectors after rotation.
    '''
    return y, -x


def cross_prod_t(Ax, Ay, Az, Bx, By, Bz):
    '''  
    # transverse cross product of A x B with component on z_0

    Returns
    -------
    z component
    '''
    return Ax*By - Ay * Bx


def epsilon_s(s):
    return 1 if s == 0 else 2
