import numpy as np

from .fortran import impl_donorcell_adv_diff_delta, advect, diffuse
from .constants import M_sun, R_sun, year, sig_h2, m_p, k_b, G


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
        valid between the center points. Left and right values are kept constant.
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

    def dlnxdlnrc(self, x):
        """
        Calculate the double-log derivative interpolated on the grid centers.
        """
        return np.interp(self.r, self.ri, self.dlnxdlnr(x))

    def interpolate_at_centers(self, x):
        "Given x defined at interaces, calculate the center values"
        return 0.5 * (x[1:] + x[:-1])

    def interpolate_at_interfaces(self, x):
        """
        Given x defined at centers, calculate the interface values. Values
        outside will be kept constant.
        """
        return np.interp(self.ri, self.r, x, left=x[0], right=x[-1])


class Twopoppy():
    """
    A new twopoppy implementation
    """

    snapshots = np.logspace(2, np.log10(5e6), 50) * year
    "the times at which to store snapshots"

    M_star = M_sun
    "stellar mass [g]"

    sigma_g = None
    "dust surface density [g/cm^2]"

    sigma_d = None
    "dust surface density [g/cm^2]"

    rho_s = 1.6
    "dust material density [g/cm^3]"

    T_gas = None
    "gas temperature [K]"

    v_frag = 1000
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

    f_mf = 0.75
    "mass fraction of large grains in the fragmentation limit [-]"

    f_md = 0.97
    "mass fraction of large grains in the drift limit [-]"

    _floor = 1e-100
    _dust_floor = _floor
    _CFL = 0.4

    # the attributes belonging to properties

    _a_0 = 1e-5
    _a_1 = 1e-5
    _stokesregime = 1
    _grid = None
    _cs = None
    _hp = None
    _omega = None
    _no_growth = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, '_' + key):
                setattr(self, '_' + key, value)
            else:
                raise ValueError(f'{key} is not an attribute of twopoppy object')

    def initialize(self):
        """
        This calls all the relevant update functions so that all arrays are set.
        It also initializes the output arrays and sets the time to zero.
        """
        self.update_omega()

        for key in ['T_gas', 'sigma_d', 'sigma_g']:
            if getattr(self, key) is None:
                raise ValueError(f'"{key}" needs to be set!')

        self.update_cshp()
        self.evolve_gas(0.0)
        self.update_size_limits(0.0)
        self.update_v_bar()
        self.update_v_bar_i()
        self.update_diffusivity()
        self.update_diffusivity_i()
        self._initialize_data()
        self.time = 0.0
        self.snapshots.sort()
        if self.snapshots[0] != 0.0:
            self.snapshots = np.hstack((0.0, self.snapshots))

    def _initialize_data(self):
        self.data = {}
        self.data['sigma_g'] = None
        self.data['sigma_d'] = None
        self.data['T_gas']   = None
        self.data['a_1']     = None
        self.data['a_df']    = None
        self.data['a_fr']    = None
        self.data['a_dr']    = None
        self.data['v_bar']   = None
        self.data['time']    = None

    def _update_data(self):
        "update the evolution data arrays"
        for key in self.data.keys():
            if hasattr(self, key):
                self_key = key
            elif hasattr(self, '_' + key):
                self_key = '_' + key
            else:
                raise NameError(f'The attribute {key} cannot be stored as it does not exist')

            _data = getattr(self, self_key)
            if self.data[key] is None:
                self.data[key] = _data
            else:
                self.data[key] = np.vstack((self.data[key], _data))

    @property
    def a_0(self):
        "small grain size [cm]"
        return self._a_0

    @property
    def a_1(self):
        "small grain size [cm]. Setting this will update St_1."
        return self._a_1

    @a_1.setter
    def a_1(self, value):
        self._a_1 = value * np.ones_like(self._a_1)
        self._St_1 = self.StokesNumber(self._a_1)

    @property
    def St_0(self):
        return self._St_0

    @property
    def St_1(self):
        "Stokes number of the large grains. Is updated by setting a_1"
        return self._St_1

    def StokesNumber(self, a):
        mfp = self.mu * m_p / (self.rho_mid * sig_h2)
        epstein = np.pi / 2. * a * self.rho_s / self.sigma_g
        stokesI = np.pi * 2. / 9. * a**2. * self.rho_s / (mfp * self.sigma_g)
        return np.where((a > 9. / 4. * mfp) & (self.stokesregime == 1), stokesI, epstein)

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
        self._cs = np.sqrt(k_b * self.T_gas / (self.mu * m_p))
        self._hp = self._cs / self._omega

    def update_diffusivity(self):
        "updates the cell center diffusion constant"
        self.Diff = self.alpha_diff * k_b * self.T_gas / self.mu / m_p / self._omega

    def update_diffusivity_i(self):
        "updates the interface diffusion constant, based on cell centers"
        self.Diff_i = self._grid.interpolate_at_interfaces(self.Diff)

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
        Calculate the pressure exponent
        """
        P = self.rho_mid * self.cs**2
        return self._grid.dlnxdlnrc(P)

    def update_v_bar(self):
        "sets self.v_bar which is defined at the cell centers"
        v_0   = self.v_gas / (1.0 + self.St_0**2)
        v_1   = self.v_gas / (1.0 + self.St_1**2)
        v_eta = self.cs**2 / (2 * self.omega * self.r) * self.gamma
        v_0   = v_0 + 2 / (self.St_0 + 1 / self.St_0) * v_eta
        v_1   = v_1 + 2 / (self.St_1 + 1 / self.St_1) * v_eta

        # set the mass distribution ratios

        f_m = self.f_mf * np.invert(self.mask_drift) + self.f_md * self.mask_drift

        # calculate the mass weighted transport velocity

        self.v_bar = v_0 * (1.0 - f_m) + v_1 * f_m

    def update_v_bar_i(self):
        "calculates self.v_bar_i (at interfaces ) from the current v_bar."
        self.v_bar_i = self._grid.interpolate_at_interfaces(self.v_bar)

    def get_dt_adv(self):
        "calculates the advection time step limit"
        dt = np.min(np.diff(self.ri) / np.abs(self.v_bar))
        return self._CFL * dt

    def get_dt_diff(self):
        "calculates the diffusion time step limit"
        dt = 0.5 * np.min(np.diff(self.ri)**2 / (self.Diff + self._floor))
        return self._CFL * dt

    def calculate_size_limits(self, dt):
        """
        Calculates the growth limits. Returns dict with keys
        - a_1
        - St_0
        - St_1
        """

        # set some constants

        mu = self.mu
        rho_s = self.rho_s

        # mean free path of the particles

        n     = self.rho_mid / (mu * m_p)
        lambd = 0.5 / (sig_h2 * n)

        gamma = np.abs(self.gamma)

        # calculate the sizes

        if self.no_growth:
            a_max = self.a_0 * np.ones_like(self.r)
            a_fr  = self.a_0 * np.ones_like(self.r)
            a_dr  = self.a_0 * np.ones_like(self.r)
            a_df  = self.a_0 * np.ones_like(self.r)
        else:
            a_fr_ep = self.fudge_fr * 2 * self.sigma_g * self.v_frag**2 / (3 * np.pi * self.alpha_turb * rho_s * self.cs**2)

            # calculate the grain size in the Stokes regime

            if self.stokesregime == 1:
                a_fr_stokes = np.sqrt(3 / (2 * np.pi)) * np.sqrt((self.sigma_g * lambd) / (self.alpha_turb * rho_s)) * self.v_frag / self.cs
                a_fr = np.minimum(a_fr_ep, a_fr_stokes)
            else:
                a_fr = a_fr_ep

            # the drift limit

            a_dr = \
                self.e_stick * self.fudge_dr / self.e_drift * 2. / np.pi * \
                self.sigma_d / rho_s * self._grid.r**2 * self.omega**2 / \
                (gamma * self.cs**2)

            # the drift induced fragmentation limit

            N = 0.5

            a_df = self.fudge_fr * 2 * self.sigma_g / (rho_s * np.pi) * self.v_frag * \
                self.omega * self.r / (gamma * self.cs**2 * (1 - N))

            a_f = np.minimum(a_fr, a_df)
            a_max = np.maximum(self.a_0, np.minimum(a_dr, a_f))

            # calculate the growth time scale and thus a_1(t)

            tau_grow = self.sigma_g / np.maximum(1e-100, self.e_stick * self.sigma_d * self.omega)

            a_1 = np.minimum(a_max, self.a_1 * np.exp(np.minimum(709.0, dt / tau_grow)))

        # calculate the Stokes number of the particles

        St_0 = self.StokesNumber(self.a_0)
        St_1 = self.StokesNumber(a_1)

        # where drift is limiting

        mask_drift = (a_dr < a_df) & (a_dr < a_fr)

        return {
            'St_0': St_0,
            'St_1': St_1,
            'a_1': a_1,
            'a_dr': a_dr,
            'a_fr': a_fr,
            'a_df': a_df,
            'mask_drift': mask_drift,
            }

    def update_size_limits(self, dt):
        """
        Updates the size limits St_0, St_1, a_1
        """
        limits = self.calculate_size_limits(dt)
        self._St_0 = limits['St_0']
        self._St_1 = limits['St_1']
        self._a_1  = limits['a_1']
        self._a_dr = limits['a_dr']
        self._a_fr = limits['a_fr']
        self._a_df = limits['a_df']
        self.mask_drift = limits['mask_drift']

    def dust_bc(self, x, g, h):
        "Return the dust boundary value parameters"

        p_L = -(x[1] - x[0]) * h[1] / (x[1] * g[1])
        q_L = 1. / x[0] - 1. / x[1] * g[0] / g[1] * h[1] / h[0]
        r_L = 0.0

        p_R = 0.0
        q_R = 1.0
        r_R = 1e-100 * x[-1]

        return [p_L, p_R, q_L, q_R, r_L, r_R]

    def gas_bc_zerotorque(self, x, g, u, h):
        "Return the dust boundary value parameters"
        return [0.0, 0.0, 1.0, 1.0, 1e-100 * x[0], 1e-100 * x[-1]]

    def gas_bc_constmdot(self, x, g, u, h):
        "Return the dust boundary value parameters"
        return [1.0, 0.0, 0.0, 1.0, g[0] * u[0] / x[0], 1e-100 * x[-1]]

    def dust_bc_zerodensity(self, x, g, u, h):
        return [0.0, 0.0, 1.0, 1.0, 1e-100 * x[0], 1e-100 * x[-1]]

    def dust_bc_zero_d2g_gradient(self, x, g, u, h):
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

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

        u = impl_donorcell_adv_diff_delta(x, D, v_gas, g, h, K, L, u, dt, *self.gas_bc_constmdot(x, g, u, h))

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

    def _dust_step_impl(self, dt):
        """Do an implicit dust time step (advection and diffusion).

        Returns:
        --------
        sigma_d : array
            the updated dust-density

        dt : float
            the time step taken (same as input, just for consistency with
            explicit method)
        """
        x = self.r
        v = np.sign(self.v_bar) * np.minimum(self.cs, np.abs(self.v_bar))
        D = self.Diff
        D[:2]  = 0      # noqa
        D[-2:] = 0      # noqa
        v[-2:] = 0      # noqa
        #
        # set up the equation
        #
        h = self.sigma_g * x
        g = np.ones(self._grid.nr)
        K = np.zeros(self._grid.nr)
        L = np.zeros(self._grid.nr)
        u = self.sigma_d * x

        # do the update

        u = impl_donorcell_adv_diff_delta(x, D, v, g, h, K, L, u, dt, *self.dust_bc_zero_d2g_gradient(x, g, u, h))
        # mask = abs(u_dust[2:-1] / u_in[2:-1] - 1) > 0.05
        sigma_d = u / x
        sigma_d = self._enforce_dust_floor(sigma_d=sigma_d)

        # now get the velocities from the exact fluxes

        u_flux = np.zeros(self._grid.nr)
        u_flux[1:] = - 0.25 * (D[1:] + D[:-1]) * (h[1:] + h[:-1]) * (
            g[1:] / h[1] * u[1:] - g[:-1] / h[:-1] * u[:-1]) / (x[1:] - x[:-1])

        return sigma_d, dt

    def _dust_step_expl(self, t_max=np.inf):
        """Carry out an explicit dust time step (advection and diffusion).

        It calculates a dt from the courant conditions and then does the
        advection followed by the diffusion. It returns the updated array and dt.
        """
        dt_adv = self.get_dt_adv()
        dt_dif = self.get_dt_diff()
        dt = min(dt_adv, dt_dif)
        dt = min(dt, t_max - self.time)

        sigma_d = self.sigma_d + advect(dt, self.r, self.ri, self.v_bar_i, self.sigma_d * self.r) / self.r
        sigma_d += diffuse(dt, self.r, self.ri, self.Diff_i, self.sigma_g * self.r, sigma_d * self.r) / self.r
        sigma_d = self._enforce_dust_floor(sigma_d=sigma_d)

        return sigma_d, dt

    def _dust_step_implexpl(self, t_max=np.inf):
        """Carry out an explicit dust advection and implicit dust diffusion step.

        It calculates a dt from the courant conditions and then does the
        advection followed by the diffusion. It returns the updated array and dt.
        """
        dt = self.get_dt_adv()
        dt = min(dt, t_max - self.time)

        sigma_d = self.sigma_d + advect(dt, self.r, self.ri, self.v_bar_i, self.sigma_d * self.r) / self.r

        x = self.r
        v = np.zeros_like(x)
        D = self.Diff
        D[:2]  = 0      # noqa
        D[-2:] = 0      # noqa
        #
        # set up the equation
        #
        g = np.ones(self._grid.nr)
        K = np.zeros(self._grid.nr)
        L = np.zeros(self._grid.nr)
        h = self.sigma_g * x
        u = sigma_d * x

        # do the update

        u = impl_donorcell_adv_diff_delta(x, D, v, g, h, K, L, u, dt, *self.dust_bc_zero_d2g_gradient(x, g, u, h))

        sigma_d = self._enforce_dust_floor(sigma_d=u / x)

        return sigma_d, dt

    def _enforce_dust_floor(self, sigma_d=None):
        "limits the dust surface density to the floor value"
        if sigma_d is None:
            self.sigma_d = np.maximum(self._dust_floor, self.sigma_d)
        else:
            return np.maximum(self._dust_floor, sigma_d)

    def dust_step(self, t_max=np.inf):
        """Twopoppy time step.

        """
        dt = min(max(self.time / 200.0, year), t_max - self.time)

        self.update_size_limits(dt)
        self.update_v_bar()
        self.sigma_d = self._dust_step_impl(dt)[0]
        self.time += dt

    def run(self):
        "Run the simulation and store the snapshots in `data`"
        start_index = np.where(self.time >= self.snapshots)[0][0]

        for i_snap in range(start_index, len(self.snapshots)):

            t_next = self.snapshots[i_snap]
            while self.time < t_next:
                self.dust_step(t_max=t_next)

            self._update_data()
            print(f'\rRunning ... {i_snap / (len(self.snapshots) - 1) * 100:.1f}%', end='', flush=True)

        print('\r------ DONE! ------')
