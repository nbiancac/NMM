import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class mesh:

    def __init__(self, sim, Np=10): # constructor
        
        self.Np = Np
        self.xi = np.linspace(-1*sim.b, 1*sim.b, self.Np, endpoint=True)
        # pipe_mesh = np.linspace(-1*sim.b, 1*sim.b, self.Np, endpoint=True)
        # beam_mesh = np.linspace(-2*sim.rb, 2*sim.rb, self.Np, endpoint=True)
        # self.xi = np.sort(np.unique(np.concatenate((beam_mesh, pipe_mesh))))
        self.yi =self.xi
        self.Xi, self.Yi = np.meshgrid(self.xi,self.yi)
        Wi = self.Xi +1j*self.Yi
        cart2polar = lambda w: ( abs(w), np.angle(w) )
        self.R, self.PHI = cart2polar(Wi)
        self.Z = np.linspace(0, sim.L, self.Np, endpoint=True)

class mesh3D:

    def __init__(self, sim, Np=10): # constructor
        
        self.Np = Np
        self.xi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.yi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.zi = np.linspace(0, sim.L, self.Np, endpoint=True)
        self.dx = np.diff(self.xi)[0]
        self.dy = np.diff(self.yi)[0]
        self.dz = np.diff(self.zi)[0]
        self.Xi, self.Yi, self.Z = np.meshgrid(self.xi,self.yi,self.zi)

        Wi = self.Xi +1j*self.Yi
        cart2polar = lambda w: ( abs(w), np.angle(w) )

        self.R, self.PHI = cart2polar(Wi)
    
class simulation:

    def __init__(self, frequency = 2e9, index_max_p = 3, index_max_r = 3, index_max_s = 3): # constructor

        import itertools
        
        self.f = frequency
        self.index_max_p = index_max_p
        self.index_max_s = index_max_s
        self.index_max_r = index_max_r
        self.ix_p = np.arange(self.index_max_p)
        self.ix_s = np.arange(self.index_max_s)
        self.ix_r = np.arange(self.index_max_r)        
        self.b = 5e-2
        self.t = 5e-2
        self.L = 1e-2
        self.d = self.t + self.b
        self.ix_pairs_n = list(itertools.product(np.arange(self.index_max_r),np.arange(self.index_max_s)))
        self.index_max_n = len(self.ix_pairs_n)
        self.ix_n = np.arange(self.index_max_n)     
        self.rb = 0.0
        self.sigma = 1e-20

class simulation_CST:

    def __init__(self, frequency = 2e9, index_max_p = 3, index_max_mode = 3): # constructor
        
        self.f = frequency
        self.index_max_p = index_max_p
        self.index_max_mode = index_max_mode
        self.ix_p = np.arange(self.index_max_p)
        self.ix_mode = np.arange(self.index_max_mode)
        self.b = 5e-2
        self.t = 5e-2
        self.L = 1e-2
        self.d = self.t + self.b
        self.rb = 0.0
        self.sigma = 1e-20

    
class beam:

    def __init__(self, beta=0.9999): # constructor
        self.beta = beta
        self.gamma = 1./np.sqrt(1-self.beta**2)
        self.Q = 1.

class source:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        from scipy.special import kn, i0, i1
        
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        x = alpha_b / gamma        
        
        F0 = lambda r, phi, z: kn(0, r * alpha_b / b / gamma) -  i0(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        dF0 = lambda r, phi, z: - kn(1, r * alpha_b / b / gamma) -  i1(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z:   - Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma)  * dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma**2)  * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q  * alpha_b / (2* np.pi * b * gamma) * dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Hz = lambda r, phi, z: 0
                
        self.box = b
        self.rb = 0#sim.b/6
        self.alpha_b = alpha_b
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0

class source_ring_top:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        from scipy.special import kn, i0, i1
        
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        x = alpha_b / gamma    
        rb = sim.rb
        s = rb * alpha_b / b / gamma
        F0 = lambda r, phi, z: kn(0, r * alpha_b / b / gamma) -  i0(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        dF0 = lambda r, phi, z: - kn(1, r * alpha_b / b / gamma) -  i1(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z:  - Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma) * i0(s) * dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma**2) * i0(s) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q  * alpha_b / (2* np.pi * b * gamma) * dF0(r, phi, z) * i0(s) * np.exp(- 1j * z * alpha_b / b)
        self.Hz = lambda r, phi, z: 0
                
        self.box = b
        self.rb = rb
        self.alpha_b = alpha_b

        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
        
class source_ring_bottom:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        from scipy.special import kn, i0, i1
        
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        x = alpha_b / gamma    
        rb = sim.rb
        s = rb * alpha_b / b / gamma
        F0 = lambda r, phi, z: kn(0, s) -  i0(s) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z:   - Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma) * i1(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma**2) * i0(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q  * alpha_b / (2* np.pi * b * gamma) * i1(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Hz = lambda r, phi, z: 0

        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
                
        self.box = rb
        self.rb = 0
        self.alpha_b = alpha_b

class source_cylinder_bottom:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        from scipy import special
        from scipy.special import kn, i0, i1
        
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        x = alpha_b / gamma    
        rb = 0.003
        s = rb * alpha_b / b / gamma
        F0 = lambda r, phi, z: kn(0, s) -  i0(s) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        # for small radius r_s the I_0(s) -> 1
#         self.Er = lambda r, phi, z:   ((Q * ((rb) ** -2) * Z0 * (special.i1((r * x) / b)) * (-((v * gamma) / \
# omega0) + rb * (((special.i1(s)) * (special.k0(x))) / (special.i0(x)) \
# + special.k1(s)))) / beta) / np.pi
#         self.Ephi = lambda r, phi, z: 0
#         self.Ez = lambda r, phi, z:  ((((((1j * Q * (rb ** -2) * (special.i0(((r * omega0) / gamma) / v)) \
# * (-(rb * omega0 * (special.i1(((rb * omega0) / gamma) / v)) * \
# (special.k0(((b * omega0) / gamma) / v))) + (special.i0(((b * omega0) \
# / gamma) / v)) * (v * gamma + -(rb * omega0 * (special.k1(((rb * \
# omega0) / gamma) / v)))))) / (special.i0(((b * omega0) / gamma) / \
# v))) / omega0) / gamma) / v) / np.pi) / epsilon_0
        
#         self.Hr = lambda r, phi, z: 0
#         self.Hphi = lambda r, phi, z:(Q * (rb ** -2) * (special.i1(((r * omega0) / gamma) / v)) * (-((v * \
# gamma) / omega0) + (rb * (special.i1(((rb * omega0) / gamma) / v)) * \
# (special.k0(((b * omega0) / gamma) / v))) / (special.i0(((b * omega0) \
# / gamma) / v)) + rb * (special.k1(((rb * omega0) / gamma) / v)))) / \
# np.pi
#         self.Hz = lambda r, phi, z: 0

        self.Er = lambda r, phi, z:   Q * Z0 / (2*np.pi * rb**2) * r * np.exp(- 1j * z * alpha_b / b) 
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  0
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z:  Q  / (2*np.pi * rb**2) * r * np.exp(- 1j * z * alpha_b / b) 
        self.Hz = lambda r, phi, z: 0
        
        # self.Er = lambda r, phi, z:   - Q * (r/rb)**2 * Z0 * alpha_b / (2* np.pi * b * beta * gamma) * i1(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        # self.Ephi = lambda r, phi, z: 0
        # self.Ez = lambda r, phi, z:  1j * Q * (r/rb)**2 * Z0 * alpha_b / (2* np.pi * b * beta * gamma**2) * i0(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        
        # self.Hr = lambda r, phi, z: 0
        # self.Hphi = lambda r, phi, z: - Q * (r/rb)**2 * alpha_b / (2* np.pi * b * gamma) * i1(r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        # self.Hz = lambda r, phi, z: 0
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
                
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
        
        self.box = rb
        self.rb = 0
        self.alpha_b = alpha_b

class source_cylinder_top:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        #from scipy.special import kn, i0, i1
        from scipy import special
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        #x = alpha_b / gamma    
        rb = 0.003
        #s = rb * alpha_b / b / gamma
        #F0 = lambda r, phi, z: kn(0, r * alpha_b / b / gamma) -  i0(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        #dF0 = lambda r, phi, z: - kn(1, r * alpha_b / b / gamma) -  i1(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        # for small radius r_s the I_0(s) -> 1
#         self.Er = lambda r, phi, z:   ((((Q * (gamma ** -2) * (1 + epsilon_0 * (v ** 2) * (gamma ** 2) \
# * mu_0) * (special.i1(((rb * omega0) / gamma) / v)) * (((special.i1(((r \
# * omega0) / gamma) / v)) * (special.k0(((b * omega0) / gamma) / v))) \
# / (special.i0(((b * omega0) / gamma) / v)) + special.k1(((r * omega0) \
# / gamma) / v))) / v) / rb) / np.pi) / epsilon_0
#         self.Ephi = lambda r, phi, z: 0
#         self.Ez = lambda r, phi, z:  ((((((1j * Q * (special.i1(((rb * omega0) / gamma) / v)) * \
# (-((special.i0(((r * omega0) / gamma) / v)) * (special.k0(((b * \
# omega0) / gamma) / v))) + (special.i0(((b * omega0) / gamma) / v)) * \
# (special.k0(((r * omega0) / gamma) / v)))) / (special.i0(((b * \
# omega0) / gamma) / v))) / gamma) / v) / rb) / np.pi) / epsilon_0
                                        
#         self.Hr = lambda r, phi, z: 0
#         self.Hphi = lambda r, phi, z:(((Q * (special.i1(((rb * omega0) / gamma) / v)) * ((special.i1(((r \
# * omega0) / gamma) / v)) * (special.k0(((b * omega0) / gamma) / v)) + \
# (special.i0(((b * omega0) / gamma) / v)) * (special.k1(((r * omega0) \
# / gamma) / v)))) / (special.i0(((b * omega0) / gamma) / v))) / rb) / \
# np.pi
#         self.Hz = lambda r, phi, z: 0

        self.Er = lambda r, phi, z:   Q * Z0 / (2*np.pi * r) * np.exp(- 1j * z * alpha_b / b) 
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  0
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: Q  / (2*np.pi * r) * np.exp(- 1j * z * alpha_b / b) 
        self.Hz = lambda r, phi, z: 0
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
                
        self.box = b
        self.rb = rb
        self.alpha_b = alpha_b 
        
class source_gaussian:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
               
        from scipy.constants import c, epsilon_0, mu_0
        #from scipy.special import kn, i0, i1
        from scipy import special
        gamma = beam.gamma
        beta = beam.beta
        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        #x = alpha_b / gamma    
        rb = 0.003
        #s = rb * alpha_b / b / gamma
        #F0 = lambda r, phi, z: kn(0, r * alpha_b / b / gamma) -  i0(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        #dF0 = lambda r, phi, z: - kn(1, r * alpha_b / b / gamma) -  i1(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        # for small radius r_s the I_0(s) -> 1
#         self.Er = lambda r, phi, z:   ((((Q * (gamma ** -2) * (1 + epsilon_0 * (v ** 2) * (gamma ** 2) \
# * mu_0) * (special.i1(((rb * omega0) / gamma) / v)) * (((special.i1(((r \
# * omega0) / gamma) / v)) * (special.k0(((b * omega0) / gamma) / v))) \
# / (special.i0(((b * omega0) / gamma) / v)) + special.k1(((r * omega0) \
# / gamma) / v))) / v) / rb) / np.pi) / epsilon_0
#         self.Ephi = lambda r, phi, z: 0
#         self.Ez = lambda r, phi, z:  ((((((1j * Q * (special.i1(((rb * omega0) / gamma) / v)) * \
# (-((special.i0(((r * omega0) / gamma) / v)) * (special.k0(((b * \
# omega0) / gamma) / v))) + (special.i0(((b * omega0) / gamma) / v)) * \
# (special.k0(((r * omega0) / gamma) / v)))) / (special.i0(((b * \
# omega0) / gamma) / v))) / gamma) / v) / rb) / np.pi) / epsilon_0
                                        
#         self.Hr = lambda r, phi, z: 0
#         self.Hphi = lambda r, phi, z:(((Q * (special.i1(((rb * omega0) / gamma) / v)) * ((special.i1(((r \
# * omega0) / gamma) / v)) * (special.k0(((b * omega0) / gamma) / v)) + \
# (special.i0(((b * omega0) / gamma) / v)) * (special.k1(((r * omega0) \
# / gamma) / v)))) / (special.i0(((b * omega0) / gamma) / v))) / rb) / \
# np.pi
#         self.Hz = lambda r, phi, z: 0

        self.Er = lambda r, phi, z:   Q * Z0 * 1/(2*np.pi*r) * \
            (1-np.exp((-r**2)/(2*rb**2))) * np.exp(- 1j * z * alpha_b / b) 
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  0
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: Q * 1/(2*np.pi*r) * \
            (1-np.exp((-r**2)/(2*rb**2))) * np.exp(- 1j * z * alpha_b / b) 
        self.Hz = lambda r, phi, z: 0
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
        
        self.Jz = lambda r, phi, z:   Q * 1/(2*np.pi*rb**2) * \
            (np.exp((-r**2)/(2*rb**2))) * np.exp(- 1j * z * alpha_b / b) 
                
        self.box = b
        self.rb = 0
        self.alpha_b = alpha_b         
        
class source_cav:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, beam): # constructor
        
        
        from scipy.constants import c, epsilon_0, mu_0
        from scipy.special import kn, i0, i1
        
        gamma = beam.gamma
        beta = beam.beta
        b = sim.d
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        v = beta * c
        alpha_b = b * omega0 / v
        Q = beam.Q 
        x = alpha_b / gamma
        
        F0 = lambda r, phi, z: kn(0, r * alpha_b / b / gamma) -  i0(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        dF0 = lambda r, phi, z: - kn(1, r * alpha_b / b / gamma) -  i1(r * alpha_b / b / gamma) * kn(0,x) / i0( x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        # for small radius r_s the I_0(s) -> 
        self.Er = lambda r, phi, z:   - Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma)  * dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / (2* np.pi * b * beta * gamma**2)  * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q  * alpha_b / (2* np.pi * b * gamma) * dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Hz = lambda r, phi, z: 0

        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
                
       
        self.box = b
        self.rb = 0#sim.b/6
        
class pipe:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """
    def __init__(self, sim, index_p, direction): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0
        from scipy.special import jn_zeros, jn
        
        b = sim.b
        f0 = sim.f
        L = sim.L
        omega0 = 2 * np.pi * f0
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        ix_p = index_p
        Nmax_p = sim.index_max_p

        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        alpha_0 = k0 * b + 1j*0
        alpha_p= jn_zeros(0,Nmax_p)[ix_p]
        
        alpha_p_tilde = np.sqrt(alpha_0**2 - alpha_p**2).conj()
        Tau_TM = np.sqrt(np.pi) * alpha_p * abs(jn(1, alpha_p))
        self.Er = lambda r, phi, z: direction*1j * alpha_p_tilde * jn(1,r * alpha_p / b) / (alpha_p * Tau_TM) \
                                    * np.exp(-1j * direction * z * alpha_p_tilde /b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z: jn(0,r * alpha_p / b) / Tau_TM \
                                    * np.exp(-1j * direction * z * alpha_p_tilde /b)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: 1j* alpha_0 * jn(1,r * alpha_p / b) / (Z0 * alpha_p * Tau_TM) \
                                    * np.exp(-1j * direction * z * alpha_p_tilde /b)
        self.Hz = lambda r, phi, z: 0

        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0
        
        self.Jz = lambda r, phi, z:  0
        
        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
        
        self.box = b
        self.alpha_p_tilde = alpha_p_tilde
        self.alpha_p = alpha_p
        self.alpha_0 = alpha_0
        self.jalpha_p = abs(jn(1,alpha_p))
        self.rb = 0

class cavity_project:
    
    def __init__(self, mode_num, mesh, zmatch, radius_CST=5, stepsize_CST=0.1): # constructor
        from scipy.constants import epsilon_0 as eps_0
        from scipy.constants import mu_0
        from scipy.interpolate import RegularGridInterpolator
        
        if zmatch == 0:
            E_dati = np.loadtxt("E_Mode_left {}.txt".format(mode_num), skiprows=2)
            H_dati = np.loadtxt("H_Mode_left {}.txt".format(mode_num), skiprows=2)
        else:
            E_dati = np.loadtxt("E_Mode_right {}.txt".format(mode_num), skiprows=2)
            H_dati = np.loadtxt("H_Mode_right {}.txt".format(mode_num), skiprows=2)
            
        Ex = np.transpose(E_dati[:,3].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/eps_0))
        Ey = np.transpose(E_dati[:,5].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/eps_0))
        Ez = np.transpose(E_dati[:,7].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/eps_0))
        x = E_dati[:np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int), 0]
        y = E_dati[::np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int), 1]
        
        Hx = np.transpose(H_dati[:,4].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/mu_0))
        Hy = np.transpose(H_dati[:,6].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/mu_0))
        Hz = np.transpose(H_dati[:,8].reshape(np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int),np.rint((2*radius_CST+stepsize_CST)/stepsize_CST).astype(int))/np.sqrt(2/mu_0))
    
        
        interp_func_Ex = RegularGridInterpolator((x*1e-2, y*1e-2), Ex)
        interp_func_Ey = RegularGridInterpolator((x*1e-2, y*1e-2), Ey)
        interp_func_Ez = RegularGridInterpolator((x*1e-2, y*1e-2), Ez)
        
        interp_func_Hx = RegularGridInterpolator((x*1e-2, y*1e-2), Hx)
        interp_func_Hy = RegularGridInterpolator((x*1e-2, y*1e-2), Hy)
        interp_func_Hz = RegularGridInterpolator((x*1e-2, y*1e-2), Hz)

        # Calcola i valori interpolati su una griglia regolare
        xx, yy = np.meshgrid(mesh.xi, mesh.yi)
        points = np.array([xx.flatten(), yy.flatten()]).T
        
        self.Ex = interp_func_Ex(points).reshape(xx.shape)
        self.Ey = interp_func_Ey(points).reshape(xx.shape)
        self.Ez = interp_func_Ez(points).reshape(xx.shape)
        
        self.Hx = interp_func_Hx(points).reshape(xx.shape) 
        self.Hy = interp_func_Hy(points).reshape(xx.shape)
        self.Hz = interp_func_Hz(points).reshape(xx.shape)

class cavity_project_on_axis:
    
    def __init__(self, mode_num, mesh): # constructor
        from scipy.constants import epsilon_0 as eps_0
        
        E_dati_on_axis = np.loadtxt("Mode_on_axis {}.txt".format(mode_num), skiprows=2)
        self.Ez = E_dati_on_axis[:,7]/np.sqrt(2/eps_0)
        self.Z = E_dati_on_axis[:, 2]
        self.Fz = np.zeros(self.Ez.size)

class cavity:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_r, index_s: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, index_r, index_s): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0, c
        from scipy.special import jn_zeros, jnp_zeros, jn
        
        d = sim.d
        L = sim.L
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        
        ix_r = index_r
        Nmax_r = sim.index_max_r

        ix_s = index_s
        Nmax_s = sim.index_max_s
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        alpha_r = jn_zeros(0,Nmax_r)[ix_r]
        beta_r = jnp_zeros(0,Nmax_r)[ix_r]
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        
        sigma_c = sim.sigma
        alpha_0 = k0 * d + 1j*0
        s = np.arange(0,Nmax_s)[ix_s]
        alpha_s = s * np.pi * d / L
        alpha_rs = np.sqrt(alpha_r**2 + alpha_s**2)
        kn = alpha_rs / d
        lambda0 = 2*np.pi/kn
        self.omega_rs = kn * c
        deltas = np.sqrt(2/self.omega_rs/mu_0/sigma_c)
        self.deltas = deltas
        
        # beta_rs = np.sqrt(beta_r**2 + alpha_s**2)
        V_rs = d * np.sqrt(np.pi) * np.sqrt(L/epsilon_s(s)) * abs(jn(1, alpha_r)) * alpha_rs / alpha_r
        V_E = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(1, alpha_r)) * alpha_rs
        # V_H = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(0, beta_r)) * beta_rs

        # cavity TM solenoidal, nu=0
        self.Er = lambda r, phi, z: alpha_s * jn(1, r * alpha_r / d) / (alpha_r * V_rs) * np.sin(z * alpha_s / d)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  jn(0, r * alpha_r / d) / (V_rs) * np.cos(z * alpha_s / d)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: alpha_rs * jn(1, r * alpha_r / d) / (alpha_r * V_rs) * np.cos(z * alpha_s / d)
        self.Hz = lambda r, phi, z: 0

        # cavity E irrotational, nu=0
        self.Fr = lambda r, phi, z: - alpha_r * jn(1, r * alpha_r / d) / (d * V_E) * np.sin(z * alpha_s / d)
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z:   alpha_s * jn(0, r * alpha_r / d) / (d * V_E) * np.cos(z * alpha_s / d)

        # cavity H irrotational, nu=0
        # self.Gr = lambda r, phi, z: - beta_r * jn(1, r * beta_r / d) / (d * V_H) * np.cos(z * alpha_s / d)
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: - alpha_s * jn(0, r * beta_r / d) / (d * V_H) * np.sin(z * alpha_s / d)
        self.Jz = lambda r, phi, z:  0
        self.box = d
        self.Z0 = Z0
        
        self.Q = lambda0/deltas * (alpha_rs / (2*np.pi * (1 + epsilon_s(s) * d / L)))
        #self.k_rs = kn * (1 + (-1 + 1j)/(2*self.Q)*np.sqrt(self.omega_rs/omega0))
        self.k_rs = kn 
        #self.k_0 = alpha_0 / d * (1 - (-1 + 1j)/(2*self.Q))
        self.k_0 = alpha_0 / d
        self.rb = 0

class cavity_CST:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, mode_num): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0, c
        
        d = sim.d
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        

        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        alpha_0 = k0 * d + 1j*0
        
        # f_CST = np.loadtxt("Frequency {}.txt".format(mode_num), skiprows=2)
        f_CST = np.loadtxt("Frequency.txt", skiprows=2)[mode_num-1,1]

        k_rs = f_CST * 1e9 * 2 * np.pi * np.sqrt(epsilon_0 * mu_0)
        omega_rs = k_rs * c

        # self.Jz = lambda r, phi, z:  0
        # self.box = d
        self.k_0 = alpha_0 / d
        self.k_rs = k_rs
        self.omega_rs = omega_rs
        self.Z0 = Z0
        
        # self.rb = 0
        
class projectors:

    def __init__(self, mesh):
        
        self.Hx = np.zeros(mesh.R.shape, dtype = complex) 
        self.Hy = np.zeros(mesh.R.shape, dtype = complex) 
        self.Hz = np.zeros(mesh.R.shape, dtype = complex)
        self.Ex = np.zeros(mesh.R.shape, dtype = complex) 
        self.Ey = np.zeros(mesh.R.shape, dtype = complex) 
        self.Ez = np.zeros(mesh.R.shape, dtype = complex)
        self.Fx = np.zeros(mesh.R.shape, dtype = complex) 
        self.Fy = np.zeros(mesh.R.shape, dtype = complex) 
        self.Fz = np.zeros(mesh.R.shape, dtype = complex)
        # self.Gx = np.zeros(mesh.R.shape, dtype = complex) 
        # self.Gy = np.zeros(mesh.R.shape, dtype = complex) 
        # self.Gz = np.zeros(mesh.R.shape, dtype = complex)
        self.Hr   = np.zeros(mesh.R.shape, dtype = complex) 
        self.Hphi = np.zeros(mesh.R.shape, dtype = complex)
        self.Er   = np.zeros(mesh.R.shape, dtype = complex) 
        self.Ephi = np.zeros(mesh.R.shape, dtype = complex)
        self.Fr   = np.zeros(mesh.R.shape, dtype = complex) 
        self.Fphi = np.zeros(mesh.R.shape, dtype = complex) 
        # self.Gr   = np.zeros(mesh.R.shape, dtype = complex) 
        # self.Gphi = np.zeros(mesh.R.shape, dtype = complex) 
        self.Jz = np.zeros(mesh.R.shape, dtype = complex)

    def interpolate_at_boundary(self, obj, mesh, zmatch):
        cond = ((mesh.R<=obj.box)).astype(int) * ((mesh.R>=(obj.rb))).astype(int)
        c, s = np.cos(mesh.PHI), np.sin(mesh.PHI)
        self.Hphi = cond*obj.Hphi(mesh.R,mesh.PHI, zmatch)
        self.Hr = cond*obj.Hr(mesh.R,mesh.PHI, zmatch)
        self.Hphi[np.isnan(self.Hphi)] = 0
        self.Hr[np.isnan(self.Hr)] = 0
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy =  self.Hphi * c + self.Hr * s
        self.Hz = cond* obj.Hz(mesh.R,mesh.PHI, zmatch)
        
        self.Ephi = cond*obj.Ephi(mesh.R,mesh.PHI, zmatch)
        self.Er = cond*obj.Er(mesh.R,mesh.PHI, zmatch)
        self.Ephi[np.isnan(self.Ephi)] = 0
        self.Er[np.isnan(self.Er)] = 0
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey =  self.Ephi * c + self.Er * s
        self.Ez =  cond*obj.Ez(mesh.R,mesh.PHI, zmatch)
        
        self.Fphi = cond*obj.Fphi(mesh.R,mesh.PHI, zmatch)
        self.Fr = cond*obj.Fr(mesh.R,mesh.PHI, zmatch)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy =  self.Fphi * c + self.Fr * s
        self.Fz =  cond*obj.Fz(mesh.R,mesh.PHI, zmatch)
        
        # self.Gphi = cond*obj.Gphi(mesh.R,mesh.PHI, zmatch)
        # self.Gr = cond*obj.Gr(mesh.R,mesh.PHI, zmatch)
        # self.Gx = -self.Gphi * s + self.Gr * c
        # self.Gy =  self.Gphi * c + self.Gr * s
        # self.Gz =  cond*obj.Gz(mesh.R,mesh.PHI, zmatch)
        
    def interpolate_in_volume(self, obj, mesh):
        cond = ((mesh.R<obj.box)).astype(int)
        c, s = np.cos(mesh.PHI), np.sin(mesh.PHI)
        self.Hphi = cond*obj.Hphi(mesh.R,mesh.PHI, mesh.Z)
        self.Hr = cond*obj.Hr(mesh.R,mesh.PHI, mesh.Z)
        self.Hphi[np.isnan(self.Hphi)] = 0
        self.Hr[np.isnan(self.Hr)] = 0
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy =  self.Hphi * c + self.Hr * s
        self.Hz = cond* obj.Hz(mesh.R,mesh.PHI, mesh.Z)
        
        self.Ephi = cond*obj.Ephi(mesh.R,mesh.PHI, mesh.Z)
        self.Er = cond*obj.Er(mesh.R,mesh.PHI, mesh.Z)
        self.Ephi[np.isnan(self.Ephi)] = 0
        self.Er[np.isnan(self.Er)] = 0
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey =  self.Ephi * c + self.Er * s
        self.Ez =  cond*obj.Ez(mesh.R,mesh.PHI, mesh.Z)

        self.Fphi = cond*obj.Fphi(mesh.R,mesh.PHI, mesh.Z)
        self.Fr = cond*obj.Fr(mesh.R,mesh.PHI, mesh.Z)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy =  self.Fphi * c + self.Fr * s
        self.Fz =  cond*obj.Fz(mesh.R,mesh.PHI, mesh.Z)
        
        self.Jz =  cond*obj.Jz(mesh.R,mesh.PHI, mesh.Z)

        # self.Gphi = cond*obj.Gphi(mesh.R,mesh.PHI, mesh.Z)
        # self.Gr = cond*obj.Gr(mesh.R,mesh.PHI, mesh.Z)
        # self.Gx = -self.Gphi * s + self.Gr * c
        # self.Gy =  self.Gphi * c + self.Gr * s
        # self.Gz =  cond*obj.Gz(mesh.R,mesh.PHI, mesh.Z)
                  
    def interpolate_on_axis(self, obj, mesh, rmatch, phimatch):
        cond = int(rmatch < obj.box)
        c, s = np.cos(phimatch), np.sin(phimatch)
        
        self.Hphi = cond*obj.Hphi(rmatch, phimatch, mesh.Z)
        self.Hr = cond*obj.Hr(rmatch, phimatch, mesh.Z)
        self.Hx = -self.Hphi * s + self.Hr * c
        self.Hy =  self.Hphi * c + self.Hr * s
        self.Hz = cond* obj.Hz(rmatch, phimatch, mesh.Z)
        
        self.Ephi = cond*obj.Ephi(rmatch, phimatch, mesh.Z)
        self.Er = cond*obj.Er(rmatch, phimatch, mesh.Z)
        self.Ex = -self.Ephi * s + self.Er * c
        self.Ey =  self.Ephi * c + self.Er * s
        self.Ez =  cond*obj.Ez(rmatch, phimatch, mesh.Z)
        
        self.Fphi = cond*obj.Fphi(rmatch, phimatch, mesh.Z)
        self.Fr = cond*obj.Fr(rmatch, phimatch, mesh.Z)
        self.Fx = -self.Fphi * s + self.Fr * c
        self.Fy =  self.Fphi * c + self.Fr * s
        self.Fz =  cond*obj.Fz(rmatch, phimatch, mesh.Z)
        
        self.Jz =  cond*obj.Jz(rmatch, phimatch, mesh.Z)
                  
    def plot_at_boundary(self, mesh):                 
        fig, ax = plt.subplots(1,3, figsize=(20,6))
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
        
    def dump_components(self, mesh, dire='./'):
        import os
        dire+='/'
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
        pd.DataFrame(mesh.Xi).to_csv(dire+'Mesh_x.csv')
        pd.DataFrame(mesh.Yi).to_csv(dire+'Mesh_y.csv')
        
    def load_components(self, mesh, dire='./'):
        dire+='/'
        self.Ephi = pd.read_csv(dire+'Ephi.csv', index_col = 0).astype(complex).values
        self.Er = pd.read_csv(dire+'Er.csv', index_col = 0).astype(complex).values
        self.Ex = pd.read_csv(dire+'Ex.csv', index_col = 0).astype(complex).values
        self.Ey = pd.read_csv(dire+'Ey.csv', index_col = 0).astype(complex).values
        self.Ez = pd.read_csv(dire+'Ez.csv', index_col = 0).astype(complex).values
        self.Hphi = pd.read_csv(dire+'Hphi.csv', index_col = 0).astype(complex).values
        self.Hr = pd.read_csv(dire+'Hr.csv', index_col = 0).astype(complex).values
        self.Hx = pd.read_csv(dire+'Hx.csv', index_col = 0).astype(complex).values
        self.Hy = pd.read_csv(dire+'Hy.csv', index_col = 0).astype(complex).values
        self.Hz = pd.read_csv(dire+'Hz.csv', index_col = 0).astype(complex).values
        
    def add_fields(self, obj2):

        self.Hphi = self.Hphi + obj2.Hphi
        self.Hr =   self.Hr   + obj2.Hr 
        self.Hx =   self.Hx   + obj2.Hx
        self.Hy =   self.Hy   + obj2.Hy 
        self.Hz =   self.Hz   + obj2.Hz 
        
        self.Ephi = self.Ephi + obj2.Ephi
        self.Er =   self.Er   + obj2.Er 
        self.Ex =   self.Ex   + obj2.Ex
        self.Ey =   self.Ey   + obj2.Ey 
        self.Ez =   self.Ez   + obj2.Ez 
        
        self.Fphi = self.Fphi + obj2.Fphi
        self.Fr =   self.Fr   + obj2.Fr 
        self.Fx =   self.Fx   + obj2.Fx
        self.Fy =   self.Fy   + obj2.Fy 
        self.Fz =   self.Fz   + obj2.Fz 