import numpy as np
from math_utils import epsilon_s


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
        Nmax_r = sim.mode.index_max_r

        ix_s = index_s
        Nmax_s = sim.mode.index_max_s
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
    def __init__(self, sim, mode_num, datadir = './CST'): # constructor
        
        
        from scipy.constants import epsilon_0, mu_0, c
        
        d = sim.d
        f0 = sim.f
        omega0 = 2 * np.pi * f0
        
        Y0 = np.sqrt(epsilon_0/mu_0)
        Z0 = 1/Y0
        
        
        k0 = omega0 * np.sqrt(epsilon_0 * mu_0) 
        alpha_0 = k0 * d + 1j*0
        
        f_CST = np.loadtxt(datadir+"/Frequency.txt", skiprows=2)[mode_num-1,1]
        k_rs = f_CST * 1e9 * 2 * np.pi * np.sqrt(epsilon_0 * mu_0)
        omega_rs = k_rs * c
        q_factors = np.loadtxt(datadir + "/q_factors.txt", skiprows=2)[mode_num - 1, 1]
        self.k_0 = alpha_0 / d
        self.Q = q_factors
        self.k_rs = k_rs
        self.omega_rs = omega_rs
        self.Z0 = Z0

class cavity_project:
    
    def __init__(self, mode_num, mesh, zmatch, radius_CST=5, stepsize_CST=0.1, datadir = './CST'): # constructor
        from scipy.constants import epsilon_0 as eps_0
        from scipy.constants import mu_0
        from scipy.interpolate import RegularGridInterpolator
        
        if zmatch == 0:
            E_dati = np.loadtxt(datadir+"E_Mode_left {}.txt".format(mode_num), skiprows=2)
            H_dati = np.loadtxt(datadir+"H_Mode_left {}.txt".format(mode_num), skiprows=2)
        else:
            E_dati = np.loadtxt(datadir+"E_Mode_right {}.txt".format(mode_num), skiprows=2)
            H_dati = np.loadtxt(datadir+"H_Mode_right {}.txt".format(mode_num), skiprows=2)
            
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
    
    def __init__(self, mode_num, mesh, datadir = './CST'): # constructor
        from scipy.constants import epsilon_0 as eps_0
        from scipy.interpolate import interp1d
      
        
        E_dati_on_axis = np.loadtxt(datadir+"Mode_on_axis {}.txt".format(mode_num), skiprows=2)
        Ez_CST = E_dati_on_axis[:,7]/np.sqrt(2/eps_0)
        Z_CST = E_dati_on_axis[:, 2]*1e-2
        self.Fz = np.zeros(mesh.Z.size)
        # self.Fz = np.zeros(self.Ez.size)
        
        # Interpolazione
        interpolated_function = interp1d(Z_CST, Ez_CST, kind='linear', fill_value='extrapolate')
        
        # Calcola i valori interpolati su mesh.Z
        self.Ez = interpolated_function(mesh.Z)