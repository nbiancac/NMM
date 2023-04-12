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
        #self.xi = np.linspace(-1*sim.b, 1*sim.b, self.Np, endpoint=True)
        pipe_mesh = np.linspace(-1*sim.b, 1*sim.b, self.Np, endpoint=True)
        beam_mesh = np.linspace(-2*sim.rb, 2*sim.rb, self.Np, endpoint=True)
        self.xi = np.sort(np.unique(np.concatenate((beam_mesh, pipe_mesh))))
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

    def __init__(self, frequency = 2e9, index_max_p = 3, index_max_s = 3): # constructor

        import itertools
        
        self.f = frequency
        self.index_max_p = index_max_p
        self.index_max_s = index_max_s
        self.ix_p = np.arange(self.index_max_p)
        self.ix_s = np.arange(self.index_max_s)        
        self.b = 5e-2
        self.t = 5e-6
        self.L = 1e-2
        self.d = self.t + self.b
        self.ix_pairs_n = list(itertools.product(np.arange(self.index_max_p),np.arange(self.index_max_s)))
        self.index_max_n = len(self.ix_pairs_n)
        self.ix_n = np.arange(self.index_max_n)     
        self.rb = 0.005

    
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
        rb = 0.003
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
        rb = 0.003
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

class cavity:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """    
    def __init__(self, sim, index_p, index_s): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0
        from scipy.special import jn_zeros, jnp_zeros, jn
        
        d = sim.d
        L = sim.L
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        
        ix_p = index_p
        Nmax_p = sim.index_max_p

        ix_s = index_s
        Nmax_s = sim.index_max_s
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        alpha_p = jn_zeros(0,Nmax_p)[ix_p]
        beta_p = jnp_zeros(0,Nmax_p)[ix_p]
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        alpha_0 = k0 * d + 1j*0
        s = np.arange(0,Nmax_s)[ix_s]
        alpha_s = s * np.pi * d / L
        alpha_ps = np.sqrt(alpha_p**2 + alpha_s**2)
        beta_ps = np.sqrt(beta_p**2 + alpha_s**2)
        V_ps = d * np.sqrt(np.pi) * np.sqrt(L/epsilon_s(s)) * abs(jn(1, alpha_p)) * alpha_ps / alpha_p
        V_E = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(1, alpha_p)) * alpha_ps
        # V_H = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(0, beta_p)) * beta_ps

        # cavity TM solenoidal, nu=0
        self.Er = lambda r, phi, z: alpha_s * jn(1, r * alpha_p / d) / (alpha_p * V_ps) * np.sin(z * alpha_s / d)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  jn(0, r * alpha_p / d) / (V_ps) * np.cos(z * alpha_s / d)
        
        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: alpha_ps * jn(1, r * alpha_p / d) / (alpha_p * V_ps) * np.cos(z * alpha_s / d)
        self.Hz = lambda r, phi, z: 0

        # cavity E irrotational, nu=0
        self.Fr = lambda r, phi, z: - alpha_p * jn(1, r * alpha_p / d) / (d * V_E) * np.sin(z * alpha_s / d)
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z:   alpha_s * jn(0, r * alpha_p / d) / (d * V_E) * np.cos(z * alpha_s / d)

        # cavity H irrotational, nu=0
        # self.Gr = lambda r, phi, z: - beta_p * jn(1, r * beta_p / d) / (d * V_H) * np.cos(z * alpha_s / d)
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: - alpha_s * jn(0, r * beta_p / d) / (d * V_H) * np.sin(z * alpha_s / d)
        self.Jz = lambda r, phi, z:  0
        self.box = d
        self.k_0 = alpha_0 / d
        self.k_ps = alpha_ps / d
        self.Z0 = Z0
        
        self.rb = 0
        
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