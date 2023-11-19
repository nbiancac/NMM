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

    def __init__(self, geometry, Np=10):  # constructor

        self.Np = Np
        self.xi = np.linspace(-geometry.b, geometry.b, self.Np, endpoint=True)
        self.yi = self.xi
        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)
        Wi = self.Xi + 1j*self.Yi
        def cart2polar(w): return (abs(w), np.angle(w))
        self.R, self.PHI = cart2polar(Wi)
        self.Z = np.linspace(0, geometry.L, self.Np, endpoint=True)


class mesh3D:

    def __init__(self, sim, Np=10):  # constructor

        self.Np = Np
        self.xi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.yi = np.linspace(-sim.d, sim.d, self.Np, endpoint=True)
        self.zi = np.linspace(0, sim.L, self.Np, endpoint=True)
        self.dx = np.diff(self.xi)[0]
        self.dy = np.diff(self.yi)[0]
        self.dz = np.diff(self.zi)[0]
        self.Xi, self.Yi, self.Z = np.meshgrid(self.xi, self.yi, self.zi)

        Wi = self.Xi + 1j*self.Yi
        def cart2polar(w): return (abs(w), np.angle(w))

        self.R, self.PHI = cart2polar(Wi)


class materials:

    def __init__(self, sigma=0):  # constructor

        self.sigma = sigma


class geometry:

    def __init__(self, b=5e-2, t=5e-2, L=1e-2):  # constructor

        self.b = b
        self.t = t
        self.L = L
        self.d = self.t + self.b

        
class beam:

    def __init__(self, beta=0.9999, Q=1):  # constructor
        self.beta = beta
        self.gamma = 1./np.sqrt(1-self.beta**2)
        self.Q = Q


class source_point:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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
        rb = 0
        s = rb * alpha_b / b / gamma

        def F0(r, phi, z): return kn(0, r * alpha_b / b / gamma) - \
            i0(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        def dF0(r, phi, z): return - kn(1, r * alpha_b / b / gamma) - \
            i1(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0

        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z: - Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma) * i0(s) * \
            dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma**2) * i0(s) * \
            F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q * alpha_b / \
            (2 * np.pi * b * gamma) * dF0(r, phi, z) * \
            i0(s) * np.exp(- 1j * z * alpha_b / b)
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

        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0
        self.name = 'point-like_beam'


class source_ring_top:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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

        def F0(r, phi, z): return kn(0, r * alpha_b / b / gamma) - \
            i0(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        def dF0(r, phi, z): return - kn(1, r * alpha_b / b / gamma) - \
            i1(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0

        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z: - Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma) * i0(s) * \
            dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma**2) * i0(s) * \
            F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q * alpha_b / \
            (2 * np.pi * b * gamma) * dF0(r, phi, z) * \
            i0(s) * np.exp(- 1j * z * alpha_b / b)
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
        self.name = 'ring_beam'


class source_ring_bottom:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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
        def F0(r, phi, z): return kn(0, s) - i0(s) * kn(0, x) / i0(x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0

        # for small radius r_s the I_0(s) -> 1
        self.Er = lambda r, phi, z: - Q * Z0 * alpha_b / (2 * np.pi * b * beta * gamma) * i1(
            r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / (2 * np.pi * b * beta * gamma**2) * i0(
            r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q * alpha_b / (2 * np.pi * b * gamma) * i1(
            r * alpha_b / b / gamma) * F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
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
        self.name = 'ring_beam'


class source_cylinder_bottom:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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
        rb = sim.rb
        s = rb * alpha_b / b / gamma
        def F0(r, phi, z): return kn(0, s) - i0(s) * kn(0, x) / i0(x)
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

        self.Er = lambda r, phi, z:   Q * Z0 / \
            (2*np.pi * rb**2) * r * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  0

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z:  Q / \
            (2*np.pi * rb**2) * r * np.exp(- 1j * z * alpha_b / b)
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

        self.name = 'cylinder_beam'


class source_cylinder_top:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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
        rb = sim.rb
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

        self.Er = lambda r, phi, z:   Q * Z0 / \
            (2*np.pi * r) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  0

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: Q / \
            (2*np.pi * r) * np.exp(- 1j * z * alpha_b / b)
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

        self.name = 'cylinder_beam'


class source_gaussian:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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
        rb = sim.rb
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

        self.name = 'Gaussian_beam'


class source_cav:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, beam):  # constructor

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

        def F0(r, phi, z): return kn(0, r * alpha_b / b / gamma) - \
            i0(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        def dF0(r, phi, z): return - kn(1, r * alpha_b / b / gamma) - \
            i1(r * alpha_b / b / gamma) * kn(0, x) / i0(x)
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0

        # for small radius r_s the I_0(s) ->
        self.Er = lambda r, phi, z: - Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma) * \
            dF0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  1j * Q * Z0 * alpha_b / \
            (2 * np.pi * b * beta * gamma**2) * \
            F0(r, phi, z) * np.exp(- 1j * z * alpha_b / b)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: - Q * alpha_b / \
            (2 * np.pi * b * gamma) * dF0(r, phi, z) * \
            np.exp(- 1j * z * alpha_b / b)
        self.Hz = lambda r, phi, z: 0

        self.Fr = lambda r, phi, z: 0
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z: 0

        # self.Gr = lambda r, phi, z: 0
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: 0

        self.box = b
        self.rb = 0  # sim.b/6

        self.name = 'point-like_beam_in_cavity'


class pipe:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_p: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, index_p, direction):  # constructor

        from scipy.constants import epsilon_0, mu_0
        from scipy.special import jn_zeros, jn

        b = sim.b
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0

        ix_p = index_p
        Nmax_p = sim.index_max_p

        k0 = omega0 * np.sqrt(epsilon_0 * mu_0)
        alpha_0 = k0 * b + 1j*0
        alpha_p = jn_zeros(0, Nmax_p)[ix_p]

        alpha_p_tilde = np.sqrt(alpha_0**2 - alpha_p**2).conj()
        Tau_TM = np.sqrt(np.pi) * alpha_p * abs(jn(1, alpha_p))
        self.Er = lambda r, phi, z: direction*1j  * jn(1, r * alpha_p / b) / (alpha_p * Tau_TM) \
            #* np.exp(-1j * direction * z * alpha_p_tilde / b)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z: jn(0, r * alpha_p / b) / Tau_TM \
            #* np.exp(-1j * direction * z * alpha_p_tilde / b)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: 1j  * jn(1, r * alpha_p / b) / (Z0 * alpha_p * Tau_TM) \
            #* np.exp(-1j * direction * z * alpha_p_tilde / b)
        self.Hz = lambda r, phi, z: 0
        self.Lz = lambda r, phi, z: np.exp(-1j * direction * z * alpha_p_tilde / b)
        
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
        self.jalpha_p = abs(jn(1, alpha_p))
        self.rb = 0


class cavity:
    """
    sim: object gathering the simulation parameters (frequency, number of modes, etc...)
    index_r, index_s: modal index
    direction: -1 to go to left, +1 to go to right.
    """

    def __init__(self, sim, index_r, index_s):  # constructor

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
        alpha_r = jn_zeros(0, Nmax_r)[ix_r]
        beta_r = jnp_zeros(0, Nmax_r)[ix_r]
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0)

        sigma_c = sim.sigma
        alpha_0 = k0 * d + 1j*0
        s = np.arange(0, Nmax_s)[ix_s]
        alpha_s = s * np.pi * d / L
        alpha_rs = np.sqrt(alpha_r**2 + alpha_s**2)
        kn = alpha_rs / d
        lambda0 = 2*np.pi/k0
        # lambda0 = 2*np.pi/kn
        self.omega_rs = kn * c
        if sigma_c != 0:
            deltas = np.sqrt(2/omega0/mu_0/sigma_c)
            # deltas = np.sqrt(2/self.omega_rs/mu_0/sigma_c)
        else:
            deltas = np.inf
        self.deltas = deltas

        # beta_rs = np.sqrt(beta_r**2 + alpha_s**2)
        V_rs = d * np.sqrt(np.pi) * np.sqrt(L/epsilon_s(s)) * \
            abs(jn(1, alpha_r)) * alpha_rs / alpha_r
        V_E = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(1, alpha_r)) * alpha_rs
        # V_H = np.sqrt(np.pi * L / epsilon_s(s))*abs(jn(0, beta_r)) * beta_rs
        self.V_rs = V_rs
        # cavity TM solenoidal, nu=0
        self.Er = lambda r, phi, z: alpha_s * \
            jn(1, r * alpha_r / d) / (alpha_r * V_rs) * np.sin(z * alpha_s / d)
        self.Ephi = lambda r, phi, z: 0
        self.Ez = lambda r, phi, z:  jn(
            0, r * alpha_r / d) / (V_rs) * np.cos(z * alpha_s / d)

        self.Hr = lambda r, phi, z: 0
        self.Hphi = lambda r, phi, z: alpha_rs * \
            jn(1, r * alpha_r / d) / (alpha_r * V_rs) * np.cos(z * alpha_s / d)
        self.Hz = lambda r, phi, z: 0

        # cavity E irrotational, nu=0
        self.Fr = lambda r, phi, z: - alpha_r * \
            jn(1, r * alpha_r / d) / (d * V_E) * np.sin(z * alpha_s / d)
        self.Fphi = lambda r, phi, z: 0
        self.Fz = lambda r, phi, z:   alpha_s * \
            jn(0, r * alpha_r / d) / (d * V_E) * np.cos(z * alpha_s / d)

        # cavity H irrotational, nu=0
        # self.Gr = lambda r, phi, z: - beta_r * jn(1, r * beta_r / d) / (d * V_H) * np.cos(z * alpha_s / d)
        # self.Gphi = lambda r, phi, z: 0
        # self.Gz = lambda r, phi, z: - alpha_s * jn(0, r * beta_r / d) / (d * V_H) * np.sin(z * alpha_s / d)
        self.Jz = lambda r, phi, z:  0
        self.box = sim.b
        self.Z0 = Z0
# 
        self.Q = lambda0/deltas * \
            (alpha_rs / (2*np.pi * (1 + epsilon_s(s) * d / L)))
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
    def __init__(self, sim, mode_num, datadir): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0, c
        
        d = sim.d
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        alpha_0 = k0 * d + 1j*0
        
        f_CST = np.loadtxt(datadir+'Frequency.txt', skiprows=2)[mode_num-1,1]
        k_rs = f_CST * 1e9 * 2 * np.pi * np.sqrt(epsilon_0 * mu_0)
        omega_rs = k_rs * c

        self.k_0 = alpha_0 / d
        self.k_rs = k_rs
        self.omega_rs = omega_rs
        self.Z0 = Z0

class cavity_project:
    
    def __init__(self, mode_num, mesh, zmatch, radius_CST=5, stepsize_CST=0.1, datadir=''): # constructor
        from scipy.constants import epsilon_0 as eps_0
        from scipy.constants import mu_0
        from scipy.interpolate import RegularGridInterpolator
        
        if zmatch == 0:
            E_dati = np.loadtxt(datadir+'E_Mode_left {}.txt'.format(mode_num), skiprows=2)
            H_dati = np.loadtxt(datadir+'H_Mode_left {}.txt'.format(mode_num), skiprows=2)
        else:
            E_dati = np.loadtxt(datadir+'E_Mode_right {}.txt'.format(mode_num), skiprows=2)
            H_dati = np.loadtxt(datadir+'H_Mode_right {}.txt'.format(mode_num), skiprows=2)
            
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
    
    def __init__(self, mode_num, mesh, datadir): # constructor
        from scipy.constants import epsilon_0 as eps_0
        from scipy.interpolate import interp1d
      
        
        E_dati_on_axis = np.loadtxt(datadir+'Mode_on_axis {}.txt'.format(mode_num), skiprows=2)
        Ez_CST = E_dati_on_axis[:,7]/np.sqrt(2/eps_0)
        Z_CST = E_dati_on_axis[:, 2]*1e-2
        self.Fz = np.zeros(mesh.Z.size)
        # self.Fz = np.zeros(self.Ez.size)
        
        # Interpolazione
        interpolated_function = interp1d(Z_CST, Ez_CST, kind='linear', fill_value='extrapolate')
        
        # Calcola i valori interpolati su mesh.Z
        self.Ez = interpolated_function(mesh.Z)
        
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


class simulation:

    def __init__(self, frequency=2e9, index_max_p=3, index_max_r=3, index_max_s=3, Np = 50, integration = 'indirect', geometry=geometry,
                 materials=materials, beam = beam, mesh = mesh):  # constructor

        import itertools

        self.f = frequency
        self.index_max_p = index_max_p
        self.index_max_s = index_max_s
        self.index_max_r = index_max_r
        self.ix_p = np.arange(self.index_max_p)
        self.ix_s = np.arange(self.index_max_s)
        self.ix_r = np.arange(self.index_max_r)
        self.b = geometry.b
        self.t = geometry.t
        self.L = geometry.L
        self.d = geometry.d
        self.geometry = geometry
        self.ix_pairs_n = list(itertools.product(
            np.arange(self.index_max_r), np.arange(self.index_max_s)))
        self.index_max_n = len(self.ix_pairs_n)
        self.ix_n = np.arange(self.index_max_n)
        self.rb = self.b/10
        self.sigma = materials.sigma
        self.integration = integration
        self.Np = Np
        self.materials = materials
        self.beam = beam
        self.mesh = mesh
        
class simulation_CST:

    def __init__(self, frequency = 2e9, index_max_p = 3, index_max_n = 3, Np = 50, integration = 'indirect', geometry=geometry,
                 materials=materials, beam = beam, mesh = mesh, datadir=''):
        
        self.f = frequency
        self.index_max_p = index_max_p
        self.index_max_n = index_max_n
        self.ix_p = np.arange(self.index_max_p)
        self.ix_n = np.arange(self.index_max_n)
        self.b = geometry.b
        self.t = geometry.t
        self.L = geometry.L
        self.d = geometry.d
        self.rb = self.b/10
        self.sigma = materials.sigma
        self.integration = integration
        self.Np = Np
        self.materials = materials
        self.beam = beam
        self.mesh = mesh
        self.datadir = datadir
        
        
    def preload_matrixes(self):
        
        from scipy.integrate import trapz

        print(f"\nModes in the pipes {self.index_max_p}.\
        \nModes in the cavity {self.index_max_n}.\
        \nNumber of points {self.Np}.")  
        
        self.left = {'direction': -1,
                    'zmatch': 0,
                    'ev': [],
                    'vp': [],
                    'str':'left'
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
                print(f"\r >> {scenario['str']}, mode number {ix_n+1}/{len(self.ix_n)}", end="")
                # cavity_n = cavity(
                #     self, self.ix_pairs_n[ix_n][0], self.ix_pairs_n[ix_n][1])
                # cav_proj = projectors(self.mesh)
                # cav_proj.interpolate_at_boundary(cavity_n, self.mesh, zmatch)
                # cavity_ = cavity_CST(self, mode_num = ix_n + 1)
                cav_proj = cavity_project(ix_n + 1, self.mesh,  zmatch, radius_CST=5, stepsize_CST=0.1, datadir=self.datadir)
                scenario['ev'].append(cav_proj)
                # scenario['ev'].append(cav_proj)
                argument = abs(cav_proj.Hx)**2 + abs(cav_proj.Hy)**2
                WF[ix_n, ix_n] = trapz(trapz(argument, self.mesh.xi), self.mesh.yi)
            scenario['WF'] = WF
            
            print("\n")
            for ix_p in self.ix_p:
                print(f"\r >> {scenario['str']}, pipe number {ix_p+1}/{len(self.ix_p)}", end="")
                pipe_p = pipe(self, ix_p, direction)
                pipe_proj_p = projectors(self.mesh)
                pipe_proj_p.interpolate_at_boundary(pipe_p, self.mesh, zmatch)
                scenario['vp'].append(pipe_proj_p)
            print("\n")

        print("Computing frequency independent matrixes.")              
        A0 = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)
        M_alpha_p_tilde = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)
        for ix_p in self.ix_p:
            pipe_p = pipe(self, ix_p, direction)
            pipe_proj_p = scenario['vp'][ix_p]
            grad_x_p, grad_y_p = pipe_proj_p.Hx, pipe_proj_p.Hy
            M_alpha_p_tilde[ix_p, ix_p] = pipe_p.alpha_p_tilde 
            for ix_q in self.ix_p:
                pipe_proj_q = scenario['vp'][ix_q]
                grad_x_q, grad_y_q = pipe_proj_q.Hx, pipe_proj_q.Hy
                A0[ix_p, ix_q] = trapz(trapz((pipe_proj_p.Hx)*grad_x_q + (pipe_proj_p.Hy)*grad_y_q, self.mesh.xi), self.mesh.yi)

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
                    B0[ix_p, ix_n] =  trapz(trapz(cav_proj.Hx * grad_x_p + cav_proj.Hy * grad_y_p, self.mesh.xi), self.mesh.yi)
                    
                    argument = direction * cross_prod_t(pipe_proj_p.Ex, pipe_proj_p.Ey, pipe_proj_p.Ez,
                                                        cav_proj.Hx,  cav_proj.Hy,  cav_proj.Hz)
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
            source2 = source_cylinder_bottom(self, self.beam)
            source_proj2 = projectors(self.mesh)
            source_proj2.interpolate_at_boundary(source2, self.mesh, zmatch)
            source_proj.add_fields(source_proj2)
    
            # matching matrixes
            C = np.zeros((self.index_max_n, 1), dtype=complex)
            E = np.zeros((self.index_max_p, 1), dtype=complex)
            Z = np.zeros((1, self.index_max_p), dtype=complex)
            M_alpha_p_tilde = np.zeros((self.index_max_p, self.index_max_p), dtype=complex)
            
            for ix_p in self.ix_p:
                pipe_p = pipe(self, ix_p, direction)
                pipe_proj_p = scenario['vp'][ix_p]
                from scipy.special import jn
                # if np.imag(pipe_p.alpha_p_tilde ) < 0:
                Z[0, ix_p] = - direction * 1j * self.b * jn(0, source.rb * pipe_p.alpha_p / self.b)  / \
                    (np.sqrt(np.pi) * pipe_p.alpha_p * pipe_p.jalpha_p * (source.alpha_b - direction * pipe_p.alpha_p_tilde)) * \
                    np.exp(1j * zmatch * (source.alpha_b ) / self.b)
                E[ix_p, 0] = pipe_p.alpha_0 * trapz(trapz(source_proj.Hx*pipe_proj_p.Hx \
                                                        + source_proj.Hy*pipe_proj_p.Hy, self.mesh.xi), self.mesh.yi)
                M_alpha_p_tilde[ix_p, ix_p] = pipe_p.alpha_p_tilde

                
            for ix_n in self.ix_n:
                cav_proj = scenario['ev'][ix_n]
                argument = direction * cross_prod_t(source_proj.Ex, source_proj.Ey, source_proj.Ez,
                                                      cav_proj.Hx, cav_proj.Hy, cav_proj.Hz)
                C[ix_n, 0] = trapz(trapz(argument, self.mesh.xi), self.mesh.yi)
    
            scenario['A'] = scenario['A0'] * pipe_p.alpha_0**2
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
        self.W  = np.zeros((self.index_max_n, self.index_max_n), dtype=complex)

        for ix_n in self.ix_n:
            cavity_ = cavity_CST(self, mode_num = ix_n + 1, datadir=self.datadir)
            cav_proj_s = cavity_project_on_axis(ix_n + 1 , self.mesh, datadir=self.datadir)
            
            
            
            # cavity_ = cavity(
            #     self, self.ix_pairs_n[ix_n][0], self.ix_pairs_n[ix_n][1])
            # cav_proj_s = projectors(self.mesh)
            # cav_proj_s.interpolate_on_axis(cavity_, self.mesh, source.rb, 0)
            self.F[ix_n, 0] = -(cavity_.k_rs) / (cavity_.k_0**2 - cavity_.k_rs**2) * \
                trapz((cav_proj_s.Ez)  * self.beam.Q * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z)
            
            self.G[ix_n, 0] = 1j*(cavity_.k_0 * cavity_.Z0) / (cavity_.k_0**2 - cavity_.k_rs**2) * \
                trapz((cav_proj_s.Ez) * self.beam.Q * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z)
            
            self.R[ix_n, 0] = -trapz(cav_proj_s.Fz * self.beam.Q * np.exp(-1j * source.alpha_b * self.mesh.Z / self.b) , self.mesh.Z)
            
            self.MI[ix_n, ix_n] = (1j * cavity_.k_0) / (cavity_.Z0 * (cavity_.k_0**2 - cavity_.k_rs**2))
            self.MV[ix_n, ix_n] = -1j * cavity_.k_rs * cavity_.Z0 / cavity_.k_0
            self.MF[ix_n, ix_n] = -1j * cavity_.Z0 / cavity_.k_0
            self.ID[ix_n, ix_n] = 1. + 0j
            
            if self.materials.sigma != 0:
                Zw = np.sqrt(mu_0 * 2 * np.pi * self.f / 2 / self.materials.sigma)
                loss = (1+1j) *  2 * np.pi * self.f * mu_0 / (cavity_.Q)  \
                     - (1+1j) * Zw  * (self.left['WF'][ix_n,ix_n] + self.right['WF'][ix_n,ix_n])
                self.W[ix_n, ix_n] = self.MI[ix_n, ix_n] * loss
                    
        self.II = np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ np.linalg.inv(self.left['A']) @ \
            self.left['B'] - self.MI @ self.right['D'] @ np.linalg.inv(self.right['A']) @ self.right['B']) \
            @ self.F   + np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ \
            np.linalg.inv(self.left['A']) @ self.left['B'] - self.MI @ self.right['D'] @ \
            np.linalg.inv(self.right['A']) @ self.right['B']) @ self.MI @ self.left['C'] + \
            np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ np.linalg.inv(self.left['A']) @ \
            self.left['B'] - self.MI @ self.right['D'] @ np.linalg.inv(self.right['A']) @ self.right['B']) \
            @ self.MI @ self.right['C'] - np.linalg.inv(self.ID - self.W - self.MI @ self.left['D'] @ \
            np.linalg.inv(self.left['A']) @ self.left['B'] - self.MI @ self.right['D'] @ \
            np.linalg.inv(self.right['A']) @ self.right['B']) @ self.MI @ self.left['D'] @ \
            np.linalg.inv(self.left['A']) @ self.left['E'] - np.linalg.inv(self.ID - self.W - self.MI @ \
            self.left['D'] @ np.linalg.inv(self.left['A']) @ self.left['B'] - self.MI @ self.right['D'] @ \
            np.linalg.inv(self.right['A']) @ self.right['B']) @ self.MI @ self.right['D'] @ \
            np.linalg.inv(self.right['A']) @ self.right['E']
        
        # coefficients
        coeffs = {}
        coeffs['left']  = -np.linalg.inv(self.left['A']) @ self.left['E'] + np.linalg.inv(self.left['A']) @ self.left['B'] @ self.II
        coeffs['right'] = -np.linalg.inv(self.right['A']) @ self.right['E'] + np.linalg.inv(self.right['A']) @ self.right['B'] @ self.II
        coeffs['cavity_sol'] = self.MV @ (self.II - self.F) + self.G 
        coeffs['cavity_irr'] = self.MF @ self.R    
        
        # Integration
        Zcav_sol = np.zeros((1, self.index_max_n), dtype=complex)
        Zcav_irr = np.zeros((1, self.index_max_n), dtype=complex)
        for ix_n in self.ix_n:
            cavity_ = cavity_CST(self, mode_num = ix_n + 1, datadir=self.datadir)
            cav_proj_s = cavity_project_on_axis(ix_n + 1 , self.mesh, datadir=self.datadir)
            # cavity_ = cavity(
            #     self, self.ix_pairs_n[ix_n][0], self.ix_pairs_n[ix_n][1])
            # cav_proj_s = projectors(self.mesh)
            # if self.integration == 'indirect':
            #     dir_int = 0
            #     cav_proj_s.interpolate_on_axis(cavity_, self.mesh, self.geometry.b, 0)
            # else:
            dir_int = 1
            #     cav_proj_s.interpolate_on_axis(cavity_, self.mesh, self.rb, 0)
            Zcav_sol[0, ix_n] = - trapz(cav_proj_s.Ez * np.exp(1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z) 
            Zcav_irr[0, ix_n] = - trapz(cav_proj_s.Fz * np.exp(1j * source.alpha_b * self.mesh.Z / self.b), self.mesh.Z)
        

        self.Z =  1./self.beam.Q * (Zcav_sol @ coeffs['cavity_sol'] + \
                           Zcav_irr @ coeffs['cavity_irr'] + \
                            (dir_int)*(self.left['Z'] @ coeffs['left']) + \
                            (dir_int)*(self.right['Z'] @ coeffs['right']))
        self.coeffs = coeffs  
        self.Zcav_sol = Zcav_sol
        self.Zcav_irr = Zcav_irr
