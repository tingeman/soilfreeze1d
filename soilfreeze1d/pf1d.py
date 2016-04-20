# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:22:02 2013

Permafrost modelling of 1d half space using finite volumes.

@author: thin
"""



def solver_FE(I, a, L, Nx, F, T,
              user_action=None, version='scalar'):
    """
    Vectorized implementation of solver_FE_simple.
    """
    import time
    t0 = time.clock()

    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    u   = zeros(Nx+1)   # solution array
    u_1 = zeros(Nx+1)   # solution at t-dt
    u_2 = zeros(Nx+1)   # solution at t-2*dt

    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    for n in range(0, Nt):
        # Update all inner points
        if version == 'scalar':
            for i in range(1, Nx):
                u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])

        elif version == 'vectorized':
            u[1:Nx] = u_1[1:Nx] +  \
                      F*(u_1[0:Nx-1] - 2*u_1[1:Nx] + u_1[2:Nx+1])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        u[0] = 0;  u[Nx] = 0
        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_1 before next step
        #u_1[:] = u  # safe, but slow
        u_1, u = u, u_1  # just switch references

    t1 = time.clock()
    return u, x, t, t1-t0
    
    
    
    



class BasePf1D(object):
    def __init__(self, **kwargs):

        # upper boundary condition
        self.ub_type = 1             # 1: Dirichlet (specified value)
                                     # 2: Neumann   (specified flux)
                                     # 3: Robin     (combination of 1 and 2) NOT IMPLEMENTED

        self.ub_temp = 0.            # 2D array of ordinals and upper boundary temperatures [degC]
        self.ub_grad_type = 2        # upper boundary type: 2 = gradient of temperature, 1 = heat flux
        self.ub_grad = 0.015         # value of upper boundary condition

        # lower boundary condition
        self.lb_type = 1             # 1: Dirichlet (specified value)
                                     # 2: Neumann   (specified flux)
                                     # 3: Robin     (combination of 1 and 2) NOT IMPLEMENTED

        self.lb_temp = 0.            # 2D array of ordinals and upper boundary temperatures [degC]
        self.lb_grad_type = 2        # lower boundary type: 2 = gradient of temperature, 1 = heat flux
        self.lb_grad = 0.015         # value of lower boundary condition

        # Initial conditions
        self.init_ground_temp = 0.   # 2D array of depths and initial ground temperatures [degC]

        # snow handling
        # should this be a separate class which is linked on top of the ground model
        # How could this be implemented?
        #self.k_snow = None           # 2D array of ordinals and snow thermal conductivities [W/m*K]
        #self.snow_depth = None       # 2D array of ordinals and snow dpths [m]

        
        # grid properties
        self.grid_z = None           # list or array of grid node depths [m]
        self.grid_dz = None          # list or array of grid cell sizes [m]
        self.surface_z = 0.          # elevation of the ground surface [m]

        self._mesh = None             # will hold the generated mesh

        # layer properties
        self.layer_depths = None     # list or array of depth to bottom of each layer in model

        # calculation settings
        self.dt_max = 86400.0     # maximum time step [s]
        self.dt_min = 1.0         # minimum time step [s]
        self.dt_init = 0.5*86400  # initial time step size of calculation [s]
        self._dt = self.dt_max    # time step of current iteration
        self._t = 0.              # keeps track of time

        self.uwk = 0.1            # decrease time step if change in unfrozen water is larger than uwk
        self.dT_max = 0.5         # decrease time step if temperature change at any node is larger than E1 [degC]

        self.out_step = 1.0       # output step size for result file [days]
        self.nmean = 30.0         # number of steps to average for each output to average file [-]

        self.itmax = 5            # maximum number of iteration before decreasing time step

        self.unfrw = True         # solution type: True: unfrozen water formula, False: Stefan problem
        self.Tf = 0.0             # phase change temperature for Stefan problem [degC]
        self.ET = 0.1             # interval of smoothing for Stefan problem [degC]

        self.T_init = 25.         # initial temperature condition
        
        self.action = None        # function handle to perform an action at beginning of each iteration
        
        # other switches
        pass

        # run options
        pass

        # Update attributes from keyword arguments
        for k,v in kwargs.iteritems():
            if k in self.__dict__.keys():
                try:
                    self.__dict__[k].setValue(v)
                except:
                    self.__dict__[k] = v






        # Generate mesh
        self.x = np.linspace(self.surface_z, self.layer_depths[-1], self.nx)
        self.dx = np.diff(self.x)
        
        
        # generate solution arrays, T0 = current step, T1 = previous step, T2 = the step before previous
        self.T_2 = np.zeros([self.nx,1])
        self.T_1 = np.zeros([self.nx,1])
        self.T_0 = np.zeros([self.nx,1])
        
        
        # Set initial condition T(x,0) = I(x)
        if not hasattr(self.T_init, "__len__"):
            self.T_init = self.T_init*np.ones_like(self.T_0)
            
        for i in arange(0, self.nx):
            self.T_1[i] = self.T_init[i]
            
        # allow for an action to be performed at each iteration
        if action is not None:
            action(self.T_1, self.x, self._t, 0)
            
        
        

        

        


























                    
    def set_layer_properties(self, depths, **kwargs):
        """Sets the layer properties passed, and converts to cell properties."""
        if self.nlayers is None:
            self.nlayers = len(depths)
        elif len(depths) != self.nlayers:
            raise ValueError("Number of layer do not match previously registered layer properties.")

        self.layer_depths = depths

        # Generate layer properties
        for k,v in kwargs:
            if not k.startswith('l_'):
                # Make sure the variable name starts with 'l_'
                # to identify it as a layer property
                k = 'l_'+k

            self.__dict___[k] = v  # set the attribute

    def generate_mesh_properties(self):
        """Iterate over all layer properties, and update the corresponding
        CellVariables or FaceVariables, if they exist, expanding the user
        specified layer properties to values across the mesh.
        """

        has_layer_properties = False
        has_mesh_properties = False

        # Iterate over all class attributes
        for key in self.__dict__.keys():
            if key.startswith('l_'):
                # Only handle layer properties

                has_layer_properties = True

                mesh_property = 'm_'+key[2:]  # generate mesh property name
                if hasattr(self, mesh_property):
                    # If the mesh property has been defined
                    # get the relevant mesh depths (Cell or Face centers)
                    if isinstance(mesh_property, CellVariable):
                        depths = self._mesh.getCellCenters()
                    elif isinstance(mesh_property, FaceVariable):
                        depths = self._mesh.getFaceCenters()

                    # Find values at the relevant depths and set the mesh property values.
                    mesh_values = self._pick_values(depths, key)
                    self.__dict__[mesh_property].setValue(mesh_values)
                    has_mesh_properties = True
                else:
                    # If the mesh property has not been previously defined,
                    # do nothing, as we don't know what type of variable
                    # this is
                    warnings.warn("No mesh property with the name {0} has been defined.".format(mesh_property))

        if not has_layer_properties:
            raise ValueError("Model has no layer properties defined!")
        elif not has_mesh_properties:
            raise ValueError("Model has no mesh properties defined!")

    def _pick_values(self, depths, lprop):
        """Returns an array of values corresponding to the depths passed. The values
        are picked from the layer property (lprop) parameter based on the layer depths
        defined in the class attribute layer_depths.
        """
        ldepths = [self.surface_z]
        ldepths.extend([d for d in self.layer_depths])

        # set up list of conditions
        condlist = []
        for lid in xrange(self.nlayers):
            condlist.append(ldepths[lid]<=depths[0]<ldepths[lid+1])

        return numerix.select(condlist, self.__dict__[lprop])

    def set_mesh_properties(self):
        """To be overridden by derived classes!
        In this method, the derived class should define CellVariables and FaceVariables to
        hold mesh properties used in the differential equation.
        The method should call self.generate_mesh_propeties() at the end, to fill the
        mesh variables with values from the corresponding layer properties.

        """
        self.generate_mesh_properties()

    def set_equations(self):
        abstract()

    def run(self):
        abstract()

    def set_ubc(self):
        faces = self._mesh.facesLeft

        if self.ub_type == 1:
            # Dirichlet type:
            # This is a fixed value, function of time or
            # interpolation of lb_temp
            self._set_bc_dirichlet(faces, self.lb_temp)

        elif self.ub_type == 2:
            raise NotImplementedError("Neumann (gradient) boundary condition not implemented "
                                      "for upper boundary.")
        else:
            raise ValueError("Unknown upper boundary type: {0}".format(self.ub_type))

    def set_lbc(self):
        faces = self._mesh.facesRight

        if self.lb_type == 1:
            # Dirichlet type:
            # This is a fixed value, function of time or
            # interpolation of lb_temp
            self._set_bc_dirichlet(faces, self.lb_temp)

        elif self.lb_type == 2:
            raise NotImplementedError("Neumann (gradient) boundary condition not implemented "
                                      "for lower boundary.")
        else:
            raise ValueError("Unknown lower boundary type: {0}".format(self.lb_type))

    def setup(self):
        # construct mesh
        if self.grid_dz is None:
            self.grid_dz = numerix.diff(self.grid_z)
        self._mesh = Grid1D(dx=self.grid_dz)

        # initialize temperature field
        self._T = CellVariable(name="Temperature",
                               mesh=self._mesh,
                               value=self.get_mesh_Tinit())

        # set up cell properties and define equations
        self.set_mesh_properties()
        self.set_equations()

        # set boundary constraints
        self.set_lbc()
        self.set_ubc()

    def _set_bc_dirichlet(self, faces, value):
        if hasattr(value, '__call__'):
            # value is a function (must not take any arguments)
            # constrain boundary to function value
            self._T.constrain(value, faces)
        elif hasattr(value, '__iter__'):
            # value is a 2D array
            # set up interpolator to get value at arbitrary time
            bc_interp = Interpolator(value, self._t)
            # constrain boundary to interpolated value
            self._T.constrain(bc_interp, faces)
        else:
            # value is a float, make a fixed constraint.
            self._T.constrain(value, faces)

    def get_mesh_Tinit(self):
        """Constructs the initial ground temperature profile for the depths,
        based on interpolation of the temperature profile passed.

        :param depths: iterable of float
            The depths at which to get the temperature
        :param init_ground_temp: float or two column array
            If a float is passed, this is used as a constant grund temperature
            for all nodes. If an array is passed, first column must hold
            depths and second column the ground temperature at that depth.

        :return: array
            Iterable of initial ground temperatures for mesh cell centers.
        """

        depths = numerix.array(self._mesh.getCellCenters())

        result = numerix.zeros((len(depths),1))

        if not hasattr(self.init_ground_temp, '__iter__'):
            # this is a single value
            result[:] = self.init_ground_temp
        else:
            # this is an array or iterable

            result[:] = numerix.interp(depths,
                                       self.init_ground_temp[:,0],
                                       self.init_ground_temp[:,1])
            # outside the range of depths passed in init_ground_temp
            # interp just uses a constant value.

            # Handle top cells, if not encompassed by init_ground_temp
            if self.ub_type == 1 and self.ub_temp is not None:
                if self.init_ground_temp[0,0] > depths[0]:
                    # get index of cels that are not assigned a temperature.
                    top_ids = numerix.nonzero(depths<self.init_ground_temp[0,0])[0]

                    if hasattr(self.ub_temp, '__iter__'):
                        ub_temp = self.ub_temp
                    else:
                        ub_temp = self.ub_temp[0,1]

                    # interpolate between upper boundary temperature and
                    # first ground temperature measurement.
                    result[top_ids, 1] = numerix.interp(depths[top_ids],
                                                            [0, self.init_ground_temp[0,0]],
                                                            [ub_temp, self.init_ground_temp[0,1]])

            # Handle bottom cells, if not encompassed by init_ground_temp
            if self.init_ground_temp[0,-1] < depths[-1]:
                warnings.warn("Model mesh is deeper than the deepest provided "
                              "initial temperature. Deepest cells will same "
                              "temperature as lowermost specified initial "
                              "temperature!")

                # Possibly try to handle geothermal gradient or similar
                # to calculate initial temperatures of deep nodes.

        return result

    # -----------------------------------------------------------------------------------------------------------------
    # Methods simulating the behavior of dictionaries, for dict-style attribute access
    # -----------------------------------------------------------------------------------------------------------------
    def __setitem__(self, key, item):
        """Permits dictionary-like setting of attribute values."""
        self.__dict__[key] = item

    def __getitem__(self, key):
        """Permits dictionary-like attribute value lookup."""
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()


class Pf1D_basic(BasePf1D):
    def set_cell_properties(self, k, c, layer_depths):
        self._k = FaceVariable(name=r'$\lambda$', mesh=self._mesh)
        self._C = CellVariable(name=r'$C$', mesh=self._mesh, hasOld=False)

    def set_equations(self):
        # Set differential equation to solve:
        self._eq = TransientTerm(coeff=self._C) == DiffusionTerm(coeff=self._k)

    def run(self):
        pass

        
class Pf1D_unfrw(BasePf1D):
    def __init__(self, **kwargs):
        # individual layer property lists
        self.n = None       # Porosity [-]
        self.Cth = None     # Heat capacity, thawed [J/(m*m*m*K)]
        self.Cfr = None     # Heat capacity, frozen [J/(m*m*m*K)]
        self.kth = None     # Themal conductivity, thawed [W/(m*K)]
        self.kfr = None     # Themal conductivity, frozen [W/(m*K)]
        self.rho = None     # density
        self.a = None       # Unfrozen water, a constant
        self.b = None       # Unfrozen water, b exponent
        self.T_star = None  # Unfrozen water, T_star [degC]

        super(Pf1D_unfrw, self).__init__(**kwargs)

    def set_mesh_properties(self):
        self._n = CellVariable(name=r'$n$', mesh=self._mesh, hasOld=False)
        self._kth = CellVariable(name=r'$\lambda_{th}$', mesh=self._mesh, hasOld=False)
        self._kfr = CellVariable(name=r'$\lambda_{fr}$', mesh=self._mesh, hasOld=False)
        self._Cth = CellVariable(name=r'$C_{th}$', mesh=self._mesh, hasOld=False)
        self._Cfr = CellVariable(name=r'$C_{fr}$', mesh=self._mesh, hasOld=False)
        self._a = CellVariable(name=r'$a$', mesh=self._mesh, hasOld=False)
        self._b = CellVariable(name=r'$b$', mesh=self._mesh, hasOld=False)
        self._Tstar = CellVariable(name=r'$T_{*}$', mesh=self._mesh, hasOld=False)

    def set_equations(self):
        # Set function to calculate unfrozen water content fraction (phi_uw)
        # Takes temperature CellVariable as input
        self._f_phi_uw = lambda T: numerix.where(T < self._Tstar,
                                                 self._a*numerix.power((T-self._Tstar),-self._b),
                                                 numerix.ones_like(T))

        # Set function to calculate unfrozen water content (Theta_uw) [m^3/m^3]
        # Takes temperature CellVariable as input
        self._f_Theta_uw = lambda T: self._f_phi_uw(T) * self._n()

        # Set function to calculate effective thermal conductivity
        # Takes temperature CellVariable as input
        self._f_keff = lambda T: numerix.power(self._kth(), self._f_phi_uw(T)) * \
                                 numerix.power(self._kfr(), (1-self._f_phi_uw(T)))

        # Set function to calculate effective heat capacity
        # Takes temperature CellVariable as input
        self._f_Ceff = lambda T: self._Cth()*self._f_phi_uw(T) + self._Cfr*(1-self._f_phi_uw(T))

        # Set function to calculate temperature dependent latent heat
        # Takes temperature CellVariable as input
        self._f_L = lambda T: 333.2e6 + 4.955e+6*self._T() + 2.987e+4*numerix.power(self._T(),2)

        # Set function to calculate temperature dependent apparent heat capacity
        # Takes temperature CellVariable as input
        self._f_Capp = lambda T: self._f_Ceff(T) - \
                                 self._f_L(T) * (self._f_Theta_uw(T)-self._f_Theta_uw(T.old()))/self._dt

        def update_C(self):
            self._Capp.setValue(self._f_Capp(self._T))

        def update_k(self):
            self._keff.setValue(self._f_keff(self._T.arithmeticFaceValue()))

        self.update_C = update_C
        self.update_k = update_k

        # set variables to hold effective properties
        self._Capp = CellVariable(name=r'$C_{app}$', mesh=self._mesh, hasOld=False)
        self._keff = FaceVariable(name=r'$lambda_{eff}$', mesh=self._mesh)

        # Set differential equation to solve:
        self._eq = TransientTerm(coeff=self._Capp) == DiffusionTerm(coeff=self._keff)

        # Try setting up latent heat term both as C_app and as separate source term
        # and compare results!!!!

        # Should _keff be a FaceVariable?
        # Should _Ceff / C_app be a FaceVariable?

    def run(self):
        pass

       
        
# --------------------------------------------------------------
#
# Calculation of unfrozen water content
#z
# --------------------------------------------------------------    

def f_T_star(T_f, S_w, a, b):
    """Calculation of the effective freezing point, T_star."""
    return T_f-np.power((S_w/a),(-1/b))
        
        
def f_phi(T, a, b, T_star):
    """Calculates the unfrozen water fraction."""
    return np.where(T < T_star,
                         a*np.power((T-T_star),-b),
                         np.ones_like(T)*S_w)

                         
def f_unfrozen_water(T, a, b, T_star, n):
    """Calculates the unfrozen water content [m^3/m^3]."""
    return phi(T, a, b, T_star) * n

# --------------------------------------------------------------
#
# Calculation of thermal conductivities
#
# --------------------------------------------------------------    

def f_k_f(k_s, k_i, n):
    """Calculates the frozen thermal conductivity []."""
    return k_s**(1-n)*k_i**(n)
    

def f_k_t(k_s, k_w, n):
    """Calculates the thawed thermal conductivity []."""
    return k_s**(1-n)*k_w**(n)
    
    
def f_k_eff(k_f, k_t, phi):
    """Calculates the effective thermal conductivity []."""
    return k_f**(1-phi)*k_t**(phi)
    

# --------------------------------------------------------------
#
# Calculation of heat capacities
#
# --------------------------------------------------------------        

def f_C_eff(C_f, C_t, phi):
    """Calculates the effective heat capacity []."""
    return C_f*(1-phi)+C_t*(phi)    

    
def f_C_f(C_s, C_i, n):
    """Calculates the frozen heat capacity []."""
    return C_s*(1-n)+C_i*(n)

    
def f_C_t(C_s, C_w, n):
    """Calculates the thawed heat capacity []."""
    return C_s*(1-n)+C_w*(n)


# --------------------------------------------------------------
#
# Calculation of apparent heat capacity
#
# --------------------------------------------------------------        

def f_C_app(C_eff, T_0, T_1, a, b, T_star, n)

    
# --------------------------------------------------------------
#
# Interpolation functions
#
# --------------------------------------------------------------        
    
class Interpolator(object):
    """Class to make interpolation of time series, keeping
    track of the last location in the time series for quick
    positioning.
    """
    def __init__(self, x, xp, fp):
        self.xp = xp    # interpolation x-values
        self.fp = fp    # interpolation y-values
        self.x = x      # point at which to find interpolation
        self.id = 0     # id into xp for closest value less than x

    def __call__(self):
        # rewind if time has been stepped backwards
        while self.x<self.xp[self.id] or self.id == 0:
            self.id -= 1

        # fast forward if time has been stepped up
        while self.x>self.xp[self.id+1] or self.id == len(self.xp)-2:
            self.id += 1

        # do interpolation
        return np.interp(self.x(), self.xp[self.id:self.id+2], self.fp[self.id:self.id+2])

    def reset(self):
        self.id = 0