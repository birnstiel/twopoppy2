import numpy as np
import astropy.constants as c
import astropy.units as u
from .fortran import impl_donorcell_adv_diff_delta

M_sun  = c.M_sun.cgs.value
R_sun  = c.R_sun.cgs.value
G      = c.G.cgs.value
au     = c.au.cgs.value
k_b    = c.k_B.cgs.value
m_p    = c.m_p.cgs.value
year   = (1. * u.year).cgs.value
sig_h2 = 2e-15


class Grid():
    "Grid object with grid centers r and grid interfaces ri"
    r = None
    ri = None
    nr = None
    
    def __init__(self, ri):
        self.ri = ri
        self.r = 0.5 * (ri[1:] + ri[:-1])
        self.nr = len(self.r)
        
    def dxdr(self, x):
        """
        derivative of quantity x on the grid. This gives the derivative dx/dr
        valid between the center points. Left and right values are kept constant so that
        """
        dxdr = np.diff(x) / np.diff(self.r)
        return np.hstack((dxdr[0], dxdr, dxdr[1]))
        
    def dlnxdlnr(self, x):
        """
        Calculate the double-log derivative of quantity x on the grid r.
        This is dln(x)/dln(r) valid between the cell centers.
        """
        dlnxdlnr = np.diff(np.log(x)) / np.diff(np.log(self.r))
        return np.hstack((dlnxdlnr[0], dlnxdlnr, dlnxdlnr[1]))

    def interpolate_at_centers(self, x):
        "Given x defined at interaces, calculate the center values"
        return 0.5 * (x[1:] + x[:-1])
        
    def interpolate_at_interfaces(self, x):
        """
        Given x defined at centers, calculate the interface values. Values
        outside will be kept constant.
        """
        return np.interp(self._grid.ri, self._grid.r, x, left=x[0], right=x[1])

        
class Twopoppy():
    """
    A new twopoppy implementation
    """
    
    M_star = None
    "stellar mass [g]"
    
    sigma_g = None
    "dust surface density [g/cm^2]"

    sigma_d = None
    "dust surface density [g/cm^2]"
    
    rho_s = 1.6
    "dust material density [g/cm^3]"
    
    T_gas = None
    "gas temperature [K]"
    
    v_frag = None
    "fragmentation velocity [cm/s]"
    
    alpha_gas = 1e-3
    "gas viscosity alpha parameter"
    
    alpha_diff = 1e-3
    "alpha value to determine dust diffusion"
    
    alpha_turb = 1e-3
    "alpha parameter to drive turbulent collisions"
    
    T_star = 4000
    "stellar temperature [K]"
    
    R_star = 2.5 * R_sun
    "stellar radius [cm]"
    
    e_drift = 1
    "drift efficiency [-]"
    
    e_stick = 1
    "sticking probability [-]"
    
    mu = 0.55
    "gas mean molecular weight in m_p [m_p]"
    
    time = 0.0
    "simulation time since beginning [s]"
    
    fudge_fr = 0.37
    "fragmentation size limit fudge factor [-]"
    
    fudge_dr = 0.55
    "drift size limit fudge factor [-]"
    
    # the attributes belonging to properties
    
    _a_0 = 1e-5
    _stokesregime = 1
    _grid = None
    _cs = None
    _hp = None
    _omega = None
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, '_' + key):
                setattr(self, '_' + key, value)
            else:
                raise ValueError(f'{key} is not an attribute of twopoppy object')
                
    @property
    def a_0(self):
        "small grain size [cm]"
        return self._a_0
        
    @property
    def r(self):
        "radial grid centers [cm]"
        return self._grid.r

    @property
    def ri(self):
        "radial interface grid [cm]"
        return self._grid.ri

    @property
    def stokesregime(self):
        """which drag regimes to include [-]
        
        stokesregime == 0: include only Epstein drag
        stokesregime == 1: include first Stokes regime
        """
        return self._stokesregime
        
    @stokesregime.setter
    def stokesregime(self, value):
        if value not in [0, 1]:
            raise ValueError('stokesregime must be 0 or 1')
        else:
            self._stokesregime = value
        
    @property
    def no_growth(self):
        "whether or not particles grow [bool]"
        return self._no_growth
        
    @no_growth.setter
    def no_growth(self, value):
        if type(value) is not bool:
            raise ValueError('no_growth must be boolean')
        else:
            self._no_growth = value
            
    def set_all_alpha(self, value):
        "helper to set all alphas to the same values"
        self.alpha_gas = value
        self.alpha_diff = value
        self.alpha_turb = value
        
    def update_omega(self):
        self._omega = np.sqrt(G * self.M_star / self._grid.r**3)
        
    @property
    def omega(self):
        "Keplerian frequency [1/s]"
        return self._omega
        
    def update_cshp(self):
        "update sound speed and disk scale height"
        self._cs = np.sqrt(k_b * self.T_gas / (self.mu * mp))
        self._hp = self._cs / self._omega
    
    @property
    def cs(self):
        "isothermal sound speed [cm/s]"
        return self._cs

    @property
    def hp(self):
        "pressure scale height [cm]"
        return self._hp
        
    @property
    def rho_mid(self):
        "gas mid-plane density [g/cm^2]"
        return self.sigma_g / (np.sqrt(2. * np.pi) * self._hp)
        
    @property
    def gas_viscosity(self):
        "gas alpha viscosity [cm^2/s]"
        return self.alpha_gas * k_b * self.T_gas / self.mu / m_p * np.sqrt(self._grid.r**3 / G / self.M_star)
    
    @property    
    def gamma(self):
        """
        Calculate the absolute of the pressure exponent
        """
        rho_mid = self.rho_mid
        P = self.rhomid * self.cs**2
        return np.abs(self._grid.dlxdlr(P))

    def update_size_limits(self, dt):
        """
        Calculates the growth limits and updates the particle sizes.
        """

        # set some constants

        mu = self.mu
        rho_s = self.rho_s
        gamma = self.gamma

        # mean free path of the particles

        n     = self.rho_mid / (mu * m_p)
        lambd = 0.5 / (sig_h2 * n)

        # calculate the sizes

        if self.no_growth:
            mask  = np.zeros_like(self.r, dtype=bool)
            a_max = self.a_0 * np.ones_like(self.r)
            a_fr  = self.a_0 * np.ones_like(self.r)
            a_dr  = self.a_0 * np.ones_like(self.r)
            a_df  = self.a_0 * np.ones_like(self.r)
        else:
            a_fr_ep = self.fudge_fr * 2 * self.sigma_g * self.v_frag**2 / (3 * np.pi * self.alpha * rho_s * self.cs**2)

            # calculate the grain size in the Stokes regime

            if self.stokesregime == 1:
                a_fr_stokes = np.sqrt(3 / (2 * np.pi)) * np.sqrt((self.sigma * lambd) / (self.alpha * rho_s)) * self.v_frag / self.cs
                a_fr = np.minimum(a_fr_ep, a_fr_stokes)
            else:
                a_fr = a_fr_ep

            # the drift limit

            a_dr = \
                self.e_stick * self.fudge_dr / self.e_drift * 2. / np.pi * \
                self.sigma_d / rho_s * self._grid.r**2 * self.omega**2 / \
                (self.gamma * self.cs**2)

            # the drift induced fragmentation limit

            N = 0.5

            a_df = self.fudge_fr * 2 * self.sigma / (rho_s * np.pi) * self.v_frag * \
                self.omega * self.r / (self.gamma * self.cs**2 * (1 - N))
                
            a_f = np.minimum(a_fr, a_df)
            a_max = np.maximum(self.a_0, np.minimum(a_dr, a_f))

            # calculate the growth time scale and thus a_1(t)

            tau_grow = self.sigma / np.maximum(1e-100, self.e_stick * self.sigma_d * self.omega)
            
            a_1 = np.minimum(a_max, self.a_1 * np.exp(np.minimum(709.0, dt / tau_grow)))

        # calculate the Stokes number of the particles and update object attributes

        self.St_0 = rho_s / self.sigma * np.pi / 2 * self.a_0
        self.St_1 = rho_s / self.sigma * np.pi / 2 * a_1
        self.a_1 = a_1
        
    def dust_bc(self, x, g, h):
        "Return the dust boundary value parameters"
        
        p_L = -(x[1] - x[0]) * h[1] / (x[1] * g[1])
        q_L = 1. / x[0] - 1. / x[1] * g[0] / g[1] * h[1] / h[0]
        r_L = 0.0
        
        p_R = 0.0
        q_R = 1.0
        r_R = 1e-100 * x[-1]
        
        return [p_L, p_R, q_L, q_R, r_L, r_R]
        
    def gas_bc(self, x, g, h):
        "Return the dust boundary value parameters"
        
        return [0.0, 0.0, 1.0, 1.0, 1e-100 * x[0], 1e-100 * x[-1]]
        
    def evolve_gas(self, dt):
        """evolve the gas surface density by time step dt"""
        nr = self._grid.nr
        x  = self._grid.r
        u  = self.sigma_g * x
        D  = 3.0 * np.sqrt(x)
        g  = self.gas_viscosity / np.sqrt(x)
        
        h = np.ones(nr)
        # K = sig_dot * x
        K = np.zeros(nr)
        L = np.zeros(nr)
        v_gas = np.zeros(nr)
    
        u = impl_donorcell_adv_diff_delta(x, D, v_gas, g, h, K, L, u, dt, *self.gas_bc(x, g, h))
        
        sig_g = u / x
        sig_g = np.maximum(sig_g, 1e-100)

        # now get the gas velocities from the exact fluxes
        
        u_flux = np.zeros(nr)
        u_flux[1:] = - 0.25 * (D[1:] + D[:-1]) * (h[1:] + h[:-1]) * (
            g[1:] / h[1] * u[1:] - g[:-1] / h[:-1] * u[:-1]) / (x[1:] - x[:-1])
        mask = u_flux > 0.0
        imask = u_flux <= 0.0
        v_gas[mask] = u_flux[mask] / u[np.maximum(0, np.where(mask)[0] - 1)]
        v_gas[imask] = u_flux[imask] / u[np.minimum(nr - 1, np.where(imask)[0] + 1)]
        
        # update values
        
        self.v_gas = v_gas
        self.sigma_g = sig_g
            
            
def impl_donorcell_adv_diff_delta(x, Diff, v, g, h, K, L, u, dt, pl, pr, ql, qr, rl, rr):
    """
    Implicit donor cell advection-diffusion scheme with piecewise constant values

    NOTE: The cell centers can be arbitrarily placed - the interfaces are assumed
    to be in the middle of the "centers", which makes all interface values
    just the arithmetic mean of the center values.

        Perform one time step for the following PDE:

           du    d  /    \    d  /              d  /       u   \ \
           -- + -- | u v | - -- | h(x) Diff(x) -- | g(x) ----  | | = K + L u
           dt   dx \    /    dx \              dx \      h(x) / /

        with boundary conditions

            dgu/h |            |
          p ----- |      + q u |       = r
             dx   |x=xbc       |x=xbc

    Arguments:
    ----------
    n_x : int
        number of grid points

    x : array-like
        the grid

    Diff : array-like
        value of Diff @ cell center

    v : array-like
        the values for v @ interface (array[i] = value @ i-1/2)

    g : array-like
        the values for g(x)

    h : array-like
        the values for h(x)

    K : array-like
        the values for K(x)

    L : array-like
        the values for L(x)

    u : array-like
        the current values of u(x)

    dt : float
        the time step


    Output:
    -------

    u : array-like
        the updated values of u(x) after timestep dt

    """
    n_x = len(x)
    D05 = np.zeros(n_x)
    h05 = np.zeros(n_x)
    
    A = np.zeros(n_x)
    B = np.zeros(n_x)
    C = np.zeros(n_x)
    D = np.zeros(n_x)
    rhs = np.zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    D05[1:] = 0.5 * (Diff[:-1] + Diff[1:])
    h05[1:] = 0.5 * (   h[:-1] +    h[1:])
    #
    # calculate the entries of the tridiagonal matrix
    #
    vol = 0.5 * (x[2:] - x[:-2])
    A[1:-1] = -dt / vol * (
        np.maximum(0., v[1:-1]) + 
        D05[1:-1] * h05[1:-1] * g[:-2] / ((x[1:-1] - x[:-2]) * h[:-2]))
    B[1:-1] = 1. - dt * L[1:-1] + dt / vol * \
        (
        np.maximum(0., v[2:]) -
        np.minimum(0., v[1:-1]) +
        D05[2:] * h05[2:] * g[1:-1] / ((x[2:] - x[1:-1]) * h[1:-1]) +
        D05[1:-1] * h05[1:-1] * g[1:-1] / ((x[1:-1] - x[:-2]) * h[1:-1])
        )
    C[1:-1] = dt / vol *  \
        (
        np.minimum(0., v[2:]) -
        D05[2:] * h05[2:] * g[2:] / ((x[2:] - x[1:-1]) * h[2:])
        )
    D[1:-1] = -dt * K[1:-1]
    #
    # boundary Conditions
    #
    A[0] = 0.
    B[0] = ql - pl * g[0] / (h[0] * (x[1] - x[0]))
    C[0] = pl * g[1] / (h[1] * (x[1] - x[0]))
    D[0] = u[0] - rl

    A[-1] = - pr * g[-2] / (h[-2] * (x[-1] - x[-2]))
    B[-1] = qr + pr * g[-1] / (h[-1] * (x[-1] - x[-2]))
    C[-1] = 0.
    D[-1] = u[-1] - rr

    # the delta-way
    #for i in range(1, n_x - 1):
    #    rhs[i] = u[i] - D[i] - \
    #        (A[i] * u[i - 1] + B[i] * u[i] + C[i] * u[i + 1])
            
    rhs[1:-1] = u[1:-1] - D[1:-1] - (A[1:-1] * u[:-2] + B[1:-1] * u[1:-1] + C[1:-1] * u[:-2])
    rhs[0] = rl - (B[0] * u[0] + C[0] * u[1])
    rhs[-1] = rr - (A[-1] * u[-2] + B[-1] * u[-1])

    # solve for du
    
    du = tridag(A, B, C, rhs, n_x)

    return u + du
    
    
def tridag(a, b, c, r, n):
    """
    Solves a tridiagnoal matrix equation

        M * u  =  r

    where M is tridiagonal, and u and r are vectors of length n.

    Arguments:
    ----------

    a : array
        lower diagonal entries

    b : array
        diagonal entries

    c : array
        upper diagonal entries

    r : array
        right hand side vector

    n : int
        size of the vectors

    Returns:
    --------

    u : array
        solution vector
    """
    import numpy as np

    gam = np.zeros(n)
    u = np.zeros(n)

    if b[0] == 0.:
        raise ValueError('tridag: rewrite equations')

    bet = b[0]

    u[0] = r[0] / bet

    for j in np.arange(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]

        if bet == 0:
            raise ValueError('tridag failed')
        u[j] = (r[j] - a[j] * u[j - 1]) / bet

    for j in np.arange(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u

    def compute_vr_twopoppy(self):
        """
        calculate the velocities of the two populations:
        """

        # first: get the gas velocity

        self.compute_v_gas_interfaces(upwind=True)
        ri = 0.5 * (self.r[1:] + self.r[:-1])
        v_gas = np.interp(self.r, ri, self.vr)

        v_0 = v_gas / (1 + self.twopoppy['St_0']**2)
        v_1 = v_gas / (1 + self.twopoppy['St_1']**2)
        #
        # Second: drift velocity
        #
        v_dr = self.cs**2 / (2 * self.omega * self.r) * self.gamma
        #
        # level of at the peak position
        #
        v_0 = v_0 + 2 / (self.twopoppy['St_0'] + 1 / self.twopoppy['St_0']) * v_dr
        v_1 = v_1 + 2 / (self.twopoppy['St_1'] + 1 / self.twopoppy['St_1']) * v_dr
        #
        # set the mass distribution ratios
        #
        f_m = 0.75 * np.invert(self.twopoppy['driftlimited']) + 0.97 * self.twopoppy['driftlimited']
        #
        # calculate the mass weighted transport velocity
        #
        self.v_bar = v_0 * (1 - f_m) + v_1 * f_m

    def compute_twopoppy_next_timestep(self, dt, alphamodel=True, fixgas=False, extracond=None):
        """
        Advance the dust component one time step into the future.
        Radial drift and turbulent mixing included, as well as the
        gas drag as the gas is moving inward.

        ARGUMENTS:
        dt          = Time step in seconds
        alphamodel  = If True, then recompute self.nu from alpha-recipe (default)
        fixgas      = If True, then do *not* include the inward gas motion in dust drift.
        extracond   = (for special purposes only) List of extra internal conditions

        Note: If self.diskradialmodel.alphamix is present, then this alpha will be used (instead of the
        usual self.alpha) for the turbulent mixing.

        ** BEWARE: **
        Always make sure to have updated the midplane density and temperature,
        and then call the compute_stokes_from_agrain() before calling this subroutine,
        if you have evolved the gas beforehand.
        """
        #
        # If requested, compute nu and dmix from alpha
        #
        if alphamodel:
            self.compute_nu()
            if hasattr(self, 'alphamix'):
                self.dmix = self.alphamix * self.cs * self.cs / self.omega / self.Sc
            else:
                self.dmix = self.nu / self.Sc
            self.dmix[:] *= 1.0 / (1.0 + self.St**2)
        #
        # Cast into diffusion equation form
        #
        x    = self.r
        y    = 2 * np.pi * self.r * self.sigma_d  # Dust
        g    = 2 * np.pi * self.r * self.sigma    # Gas
        d    = self.dmix
        di   = 0.5 * (d[1:] + d[:-1])
        self.compute_dustvr_at_interfaces(alphamodel=alphamodel, fixgas=fixgas)
        vi   = self.vr
        s    = np.zeros(len(x))    # For now no source term
        #
        # Set boundary conditions
        #
        bcl  = (1, 0, 0, 1)  # Simply set dy/dx=0 at inner edge
        bcr  = (1, 0, 0, 1)  # Simply set dy/dx=0 at outer edge
        #
        # Get the new value of y after one time step dt
        #
        y    = solvediffonedee(x, y, vi, di, g, s, bcl, bcr, dt=dt,
                               int=True, upwind=True, extracond=extracond)
        #
        # Obtain new sigdust
        #
        sigma = y / (2 * np.pi * self.r)
        #
        # Return
        #
        return sigma
