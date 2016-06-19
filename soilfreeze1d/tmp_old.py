#!/usr/bin/env python
# As v1, but using scipy.sparse.diags instead of spdiags
"""
Functions for solving a 1D diffusion equations of simplest types
(constant coefficient, no source term):
      u_t = a*u_xx on (0,L)
with boundary conditions u=0 on x=0,L, for t in (0,T].
Initial condition: u(x,0)=I(x).
The following naming convention of variables are used.
===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
F     The dimensionless number a*dt/dx**2, which implicitly
      specifies the time step.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_1   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
user_action is a function of (u, x, t, n), u[i] is the solution at
spatial mesh point x[i] at time t[n], where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""
import pdb
import sys, time
#from scitools.std import *
import numpy as np
from numpy import linspace, zeros, exp, pi, sin
from matplotlib.pyplot import plot, savefig, gcf, gca, draw, show, figure
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.sparse
import scipy.sparse.linalg


class LayeredModel(object):
    _descriptor_unfrw = {'names': ('Thickness', 'n', 'C_th', 'C_fr', 'k_th', 'k_fr', 'alpha',  'beta',  'Tf', 'Soil_type'), 
                         'formats': ('f8',      'f8',  'f8',   'f8',   'f8',   'f8',   'f8', 'f8', 'f8', 'S50')}
    _descriptor_std =   {'names': ('Thickness', 'C',  'k',  'Soil_type'), 
                         'formats': ('f8',      'f8', 'f8', 'S50')}
                         
    def __init__(self, type='std', surface_z=0.):
        self._layers = None
        self.parameter_set = type
        self._descriptor = getattr(self, '_descriptor_'+type)
        self.surface_z = surface_z  # z-axis is positive down!
    
    def add(self, **kwargs):
        if self._layers is None:
            # Create layer structured array if it does not exist
            self._layers = np.zeros((1,), dtype=self._descriptor)
            self._layers[-1]['Soil_type'] = ''
        else:
            # Or extend it, if it already exists
            self._layers = np.resize(self._layers, len(self._layers)+1)
        
        # Add the 
        for k,v in kwargs.items():
            try:
                self._layers[-1][k] = v
            except:
                pass
        
    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self._layers[key]
        else:
            return dict(zip(self._descriptor['names'],self._layers[np.int(key)]))
    
    def __setitem__(self, key, value):
        raise NotImplementedError('Setter not implemented for the LayeredModel class...')

    def __getattr__(self, attr):
        if attr == 'z_max':
            return self.surface_z+np.sum(self._layers['Thickness'])
        elif attr in self._descriptor['names']:
            return self._layers[attr]
            
        else:
            raise ValueError('No such attribute defined')
    
    def __len__(self):
        return len(self._layers)
    
    def pick_values(self, depths, param):
        """Returns an array of values corresponding to the depths passed. 
        A point on a layer boundary will be asigned the value corresponding
        to the layer below the point.
        
        THIS IS PROBABLY A SLOW IMPLEMENTATION! 
        COULD JUST ITERATE OVER ALL POINTS ONCE AND ASSIGN.
        
        """
        ldepths = [self.surface_z]
        ldepths.extend(self.surface_z+np.cumsum(self._layers['Thickness']))
       
        # set up list of conditions
        condlist = []
        for lid in xrange(len(self._layers)):
            condlist.append(np.logical_and(depths>=ldepths[lid], depths<ldepths[lid+1]))
        
        result = np.select(condlist, self._layers[param])
        
        # Handle the case where the last depth is exactly at or below the 
        # lower-most layer boundary, and therefore not assigned a value.
        ids = depths >= ldepths[-1]
        result[ids] = self._layers[-1][param]
                
        return result
        
    def show(self, T1=-10, T2=2):
        
        nlayers = len(self)
        if nlayers > 5:
            raise NotImplementedError('Visualization of more than 5 layers is not yet implemented')
        
        ax1 = plt.subplot2grid((nlayers,2), (0,0), rowspan=nlayers)
        
        axes = []
        
        T = np.linspace(T1,T2,300)
        
        for n in xrange(nlayers):
            axes.append(plt.subplot2grid((nlayers,2), (n,1)))
            Tstar = f_Tstar(self[n]['Tf'], 1.0, self[n]['alpha'], self[n]['beta'])
            unfrw = f_unfrozen_water(T, self[n]['alpha'], self[n]['beta'], Tstar, self[n]['n'])
            axes[-1].plot(T,unfrw,'-k')
        
        show()
        draw()
    
            
        
    
        
    

            
                
                
        
    
    # DONE: Add method to retrieve all parameters for a specific layers
    
    # DONE: Add method to calculate the grid point values of a specified parameter, given an array of gridpoint depths.
    
    # Add method to visualize the layered model somehow (including the unfrozen water content curve...)
        
        

Layers = LayeredModel()
Layers.add(Thickness=1,  phi=0.60, C_th=2.00E6, C_fr=1.50E6, k_th=0.9, k_fr=1.3, alpha=2.1/0.60, beta=0.408, Tf=0.0, Soil_type='Peat')
Layers.add(t=49, phi=0.35, C_th=2.00E6, C_fr=1.50E6, k_th=1.2, k_fr=1.6, a=4.8/0.35, b=0.326, c=0.0, name='Fairbanks Silt')

# Layers[n] should return an ordered dictionary of all parameters for layer n
# Layers.C_th should return an array of C_th for all layers
# get_node_values(Layers.C_th, x) should return an array same shape as x, with values generated from C_th layer parameter
# get_cell_values(Layers.k_th, x) should return an array of shape len(x)-1, with values generated from k_th layer parameter

def plug(scheme='FE', F=0.5, Nx=50):
    L = 1.
    a = 1
    T = 0.1

    def I(x):
        """Plug profile as initial condition."""
        if abs(x-L/2.0) > 0.1:
            return 0
        else:
            return 1

    u, cpu = viz(I, a, L, Nx, F, T,
                 umin=-0.1, umax=1.1,
                 scheme=scheme, animate=True, framefiles=True)
    print 'CPU time:', cpu

def gaussian(scheme='FE', F=0.5, Nx=50, sigma=0.05):
    L = 1.
    a = 1
    T = 0.1

    def I(x):
        """Gaussian profile as initial condition."""
        return exp(-0.5*((x-L/2.0)**2)/sigma**2)

    u, cpu = viz(I, a, L, Nx, F, T,
                 umin=-0.1, umax=1.1,
                 scheme=scheme, animate=True, framefiles=True)
    print 'CPU time:', cpu

    
def solver_theta(I, Layers, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0., theta=0.5,
                 user_action=None):
    """
    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    """
    import time
    t0 = time.clock()

    x = linspace(0, Z, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    F = dt/dx**2. * Layers['k']/Layers['C']
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    
    u   = zeros(Nx+1)   # solution array at t[n+1]
    u_1 = zeros(Nx+1)   # solution at t[n]

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx)
    upper    = zeros(Nx)
    b        = zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    Fl = F*theta
    Fr = F*(1-theta)
    diagonal[:] = 1 + 2*Fl
    lower[:] = -Fl  #1
    upper[:] = -Fl  #1
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    diags = [0, -1, 1]
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
        format='csr')
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        print '.'
        b[1:-1] = u_1[1:-1] + Fr*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:])
        b[0] = ub(t[n]); b[-1] = lb(t[n])  # boundary conditions
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_1 before next step
        #u_1[:] = u
        u_1, u = u, u_1

    t1 = time.clock()
    return u, x, t, t1-t0
    
def solver_thetaPF(I, k, C, Z, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0., theta=0.5,
                 user_action=None):
    """
    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    """
    import time
    t0 = time.clock()

    x = linspace(0, Z, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    F = dt/dx**2. * k/C
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    
    u   = zeros(Nx+1)   # solution array at t[n+1]
    u_1 = zeros(Nx+1)   # solution at t[n]

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx)
    upper    = zeros(Nx)
    b        = zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    Fl = F*theta
    Fr = F*(1-theta)
    diagonal[:] = 1 + 2*Fl
    lower[:] = -Fl  #1
    upper[:] = -Fl  #1
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    diags = [0, -1, 1]
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
        format='csr')
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        print '.'
        b[1:-1] = u_1[1:-1] + Fr*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:])
        b[0] = ub(t[n]); b[-1] = lb(t[n])  # boundary conditions
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_1 before next step
        #u_1[:] = u
        u_1, u = u, u_1

    t1 = time.clock()
    return u, x, t, t1-t0


def solver_CN(I, Layers, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0., theta=0.5,
                 user_action=None, version=None):
    """
    Full solver for the model problem using the Crank-Nicolson
    difference approximation 
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    """
    import time
    t0 = time.clock()

    L = 334*1e6 # [kJ/kg] => *1000[J/kJ]*1000[kg/m^3] => [J/m^3]
    
    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    
    u   = zeros(Nx+1)   # solution array at t[tid+1]
    u_1 = zeros(Nx+1)   # solution at t[tid]

    dudT = np.ones(Nx+1)*-999.   # will hold derivative of unfrozen water
    dT = np.ones(Nx+1)*-999.     # will hold vertical temperature difference
                         # averaged over two cells except for first and last.
    
    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx)
    upper    = zeros(Nx)
    b        = zeros(Nx+1)


    # Get constant layer parameters distributed on the grid
    if Layers.parameter_set == 'unfrw':
        print "Using unfrozen water parameters"
        k_th = Layers.pick_values(x, 'k_th')
        C_th = Layers.pick_values(x, 'C_th')
        k_fr = Layers.pick_values(x, 'k_fr')
        C_fr = Layers.pick_values(x, 'C_fr')
        n = Layers.pick_values(x, 'n')
        
        alpha = Layers.pick_values(x, 'alpha')
        beta = Layers.pick_values(x, 'beta')
            
        Tf = Layers.pick_values(x, 'Tf')
        Tstar = f_Tstar(Tf, 1.0, alpha, beta)
    else:
        print "Using standard parameters"
        k_eff = Layers.pick_values(x, 'k')
        C_app = Layers.pick_values(x, 'C')  

        
    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    u = u_1 + 0.001    # initialize u for finite differences
    
    if user_action is not None:
        user_action(u_1, x, t, 0)
       
    F = dt/(2*dx**2)
    
    symbols = r'|/-\|/-\\'
    
    # Time loop
    for tid in range(0, Nt):
        #pdb.set_trace()
        print '{0:d}'.format(tid) ,
        
        # Algorithm for iterations:
        # 1. Use slope of unfrozen water to calculate Latent heat
        # 2. Solve for T(n+1)
        # 3. Use T(n) and T(n+1) to calculate updated Latent heat
        # 4. Solve again for T(n+1)
        # 5. Repeat from 3.
        #       - until change in T(n+1) smaller than XX
        #       - until max iterations reached
        # 6. If max iterations reached, reduce time-step and restart from 1.?
        #    If convergence, prepare for next time step
        # 7. If time step did not change previous 2 steps, try to increase.        
        
        
        # For now we only allow unfrw solution.
        if Layers.parameter_set == 'unfrw':
            pass
        
        phi = f_phi(u_1, alpha, beta, Tstar, 1.0)
        k_eff = f_k_eff(k_fr, k_th, phi)
        C_eff = f_C_eff(C_fr, C_th, phi)
        
        for iter in xrange(1):
   
            C_add = + L * n * alpha * beta * np.abs(u_1-Tf)**(-beta-1)
            
            # We calculate finite difference of unfrozen water based on the vertical
            # temperature gradient. We can do this because we have no significant
            # heat sources, and thus it is the vertical temperature difference
            # that drives the flow of energy.

            unfrw_u1 = n*phi
            
            dT[0:Nx-1] = u_1[0:Nx-1]-u_1[1:Nx]
            dT[Nx] = 0.
            dT[1:Nx] += dT[0:Nx-1]
            dT[1:Nx-1] = dT[1:Nx-1]/2
            
            dudT[0:Nx-1] = (unfrw_u1[0:Nx-1]-unfrw_u1[1:Nx])/(u_1[0:Nx-1]-u_1[1:Nx])
            dudT[Nx] = 0.
            dudT[1:Nx] += dudT[0:Nx-1]
            dudT[1:Nx-1] = dudT[1:Nx-1]/2
            
            if tid == 10:
                pdb.set_trace()
            
#            C_add = L * dudT
            
#            if iter == 0:
#                # If this is the first iteration, use derivative
#                print "Derivative" ,
#                
#            else:
#                # ...Otherwise, use finite difference
#                # u1 contains temperatures from last time step
#                # u contains temperatures from previous iteration of current time step.
#                print "FD" ,
#                unfrw_u1 = n*phi
#                unfrw_u = n*f_phi(u, alpha, beta, Tstar, 1.0)
#                
#            C_add = np.where(np.abs(u-u_1)<1e-4, C_add, L * (unfrw_u-unfrw_u1)/(u-u_1))
#            C_add = np.where(np.logical_or(np.abs(dT)<1e-6, np.isnan(dudT)), C_add, L * dudT)
            
            # Apparent heat capacity is the heat capacity + the latent heat effect
            #C_app = C_eff + C_add # np.where(u_1 < Tstar, C_eff+C_add, C_eff)
            
            C_app = C_eff + C_add
            
            if np.any(np.isnan(C_app)):
                pdb.set_trace()
            
            # Update all inner points
            #A_m = F*(k_eff[1:Nx]+k_eff[0:Nx-1])/C_app[1:Nx]
            #B_m = F*(k_eff[0:Nx-1]+2*k_eff[1:Nx]+k_eff[2:Nx+1])/C_app[1:Nx]
            #C_m = F*(k_eff[1:Nx]+k_eff[2:Nx+1])/C_app[1:Nx]    
            A_m = F*(2*k_eff[1:Nx])/C_app[1:Nx]
            B_m = F*(4*k_eff[1:Nx])/C_app[1:Nx]
            C_m = F*(2*k_eff[1:Nx])/C_app[1:Nx]    
            
            
            # Compute sparse matrix (scipy format)
            diagonal[1:-1] = 1 + 0.5*B_m
            lower[0:-1] = -0.5*A_m  #1
            upper[1:] = -0.5*C_m  #1
            
            # Insert boundary conditions (Dirichlet)
            diagonal[0] = 1
            upper[0] = 0
            diagonal[Nx] = 1
            lower[-1] = 0

            U = scipy.sparse.diags(
                diagonals=[diagonal, lower, upper],
                offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                format='csr')
            #print A.todense()
                
            # Compute known vector
            b[1:-1] = 0.5*A_m*u_1[:-2] + (1-0.5*B_m)*u_1[1:-1] + 0.5*C_m*u_1[2:]
            b[0] = ub(t[tid]); b[-1] = lb(t[tid])  # boundary conditions
            u[:] = scipy.sparse.linalg.spsolve(U, b)

        print "Done!"
            
        if user_action is not None:
            user_action(u, x, t, tid+1)

        # Update u_1 before next step
        #u_1[:] = u
        u_1, u = u, u_1

    t1 = time.clock()
    return u, x, t, t1-t0




    
def solver_FEPF(I, k, C, Z, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0.,
              user_action=None, version='vectorized'):
    """
    ======= ==========================================================
    Name    Description
    ======= ==========================================================
    I       Initial condition (Python function of x).
    Layers  LayeredModel instance 
               Containing all layer parameters and depths
    k       Thermal conductivity
    C       Heat capacity
    Z       Total depth of the domain ([0,Z])
    Nx      The total number of mesh cells; mesh points are numbered
            from 0 to Nx.
    dt      Time step (s)          
    F       The dimensionless number a*dt/dx**2, which implicitly
            specifies the time step.
    T       The stop time for the simulation.
            
    a       Variable coefficient (constant).
            
    x       Mesh points in space.
    t       Mesh points in time.
    n       Index counter in time.
    u       Unknown at current/new time level.
    u_1     u at the previous time level.
    dx      Constant mesh spacing in x.
    dt      Constant mesh spacing in t.
    ======= ==========================================================
    """
    
    import time
    t0 = time.clock()

    x = linspace(0, Z, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    F = dt/dx**2. * k/C
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
        print '.'
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
        u[0] = ub(t[n]);  u[Nx] = lb(t[n])
        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_1 before next step
        #u_1[:] = u  # safe, but slow
        u_1, u = u, u_1  # just switch references

    t1 = time.clock()
    return u, x, t, t1-t0    
    
    
    
def solver_FEPFL(I, Layers, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0.,
              user_action=None, version='vectorized'):
    """
    ======= ==========================================================
    Name    Description
    ======= ==========================================================
    I       Initial condition (Python function of x).
    Layers  LayeredModel instance 
               Containing all layer parameters and depths
    k       Thermal conductivity
    C       Heat capacity
    Z       Total depth of the domain ([0,Z])
    Nx      The total number of mesh cells; mesh points are numbered
            from 0 to Nx.
    dt      Time step (s)          
    F       The dimensionless number a*dt/dx**2, which implicitly
            specifies the time step.
    T       The stop time for the simulation.
            
    a       Variable coefficient (constant).
            
    x       Mesh points in space.
    t       Mesh points in time.
    n       Index counter in time.
    u       Unknown at current/new time level.
    u_1     u at the previous time level.
    dx      Constant mesh spacing in x.
    dt      Constant mesh spacing in t.
    ======= ==========================================================
    """
    
    import time
    t0 = time.clock()

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
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

    if version == 'scalar':        
        F = dt/dx**2. * Layers[0]['k']/Layers[0]['C']
        for n in range(0, Nt):
            print '.' ,
            # Update all inner points
        
            for i in range(1, Nx):
                u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])


            # Insert boundary conditions
            u[0] = ub(t[n]);  u[Nx] = lb(t[n])
            if user_action is not None:
                user_action(u, x, t, n+1)

            # Update u_1 before next step
            #u_1[:] = u  # safe, but slow
            u_1, u = u, u_1  # just switch references

    elif version == 'vectorized':            
        F = dt/dx**2. * Layers[0]['k']/Layers[0]['C']
        for n in range(0, Nt):
            print '.' , 
            # Update all inner points
            u[1:Nx] = u_1[1:Nx] +  \
                      F*(u_1[0:Nx-1] - 2*u_1[1:Nx] + u_1[2:Nx+1])

            # Insert boundary conditions
            u[0] = ub(t[n]);  u[Nx] = lb(t[n])
            if user_action is not None:
                user_action(u, x, t, n+1)

            # Update u_1 before next step
            #u_1[:] = u  # safe, but slow
            u_1, u = u, u_1  # just switch references        
    
    elif version == 'layered':
        print 'Hello!!!'
        
        F = dt/(2*dx**2)
        
        if Layers.parameter_set == 'unfrw':
            k_th = Layers.pick_values(x, 'k_th')
            C_th = Layers.pick_values(x, 'C_th')
            k_fr = Layers.pick_values(x, 'k_fr')
            C_fr = Layers.pick_values(x, 'C_fr')

            alpha = Layers.pick_values(x, 'alpha')
            beta = Layers.pick_values(x, 'beta')
                
            Tstar = f_Tstar(Layers.pick_values(x, 'Tf'), 1.0, alpha, beta)
        else:
            k_eff = Layers.pick_values(x, 'k')
            C_app = Layers.pick_values(x, 'C')
        
        
        for n in range(0, Nt):
            print '.' ,
            
            if Layers.parameter_set == 'unfrw':
                phi = f_phi(u_1, alpha, beta, Tstar, 1.0)
                k_eff = f_k_eff(k_fr, k_th, phi)
                C_eff = f_C_eff(C_fr, C_th, phi)
                C_app = C_eff + 0.    # This is where we add the Latent heat contribution
            
            #if n > 10:
            #    pdb.set_trace()
            
            # Update all inner points
            A_m = F*(k_eff[1:Nx]+k_eff[0:Nx-1])/C_app[1:Nx]
            B_m = 1-F*(k_eff[0:Nx-1]+2*k_eff[1:Nx]+k_eff[2:Nx+1])/C_app[1:Nx]
            C_m = F*(k_eff[1:Nx]+k_eff[2:Nx+1])/C_app[1:Nx]
            
            u[1:Nx] = A_m*u_1[0:Nx-1] + B_m*u_1[1:Nx] + C_m*u_1[2:Nx+1]
            
            # Insert boundary conditions
            u[0] = ub(t[n]);  u[Nx] = lb(t[n])
            if user_action is not None:
                user_action(u, x, t, n+1)

            # Update u_1 before next step
            #u_1[:] = u  # safe, but slow
            u_1, u = u, u_1  # just switch references        

    else:
        raise ValueError('version=%s' % version)
        
    t1 = time.clock()
    return u, x, t, t1-t0   

    
def vizTIN(I, k, C, Z, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0.,
        umin=0., umax=1.1, scheme='FE', animate=True, framefiles=False):

    def plot_u(u, x, t, n):
        ax = gca()
        ax.hold(False)
        h = plot(u, x, 'r-')
        ax.set_title('t=%f' % (t[n]/(3600*24.)))
        ax.set_ylim([0,Z])
        ax.set_xlim([umin,umax])
        ax.invert_yaxis()
        if framefiles:
            savefig('tmp_frame%04d.png' % n)
        if t[n] == 0:
            time.sleep(1)
        elif not framefiles:
            # It takes time to write files so pause is needed
            # for screen only animation
            draw()
            show()
            #time.sleep(0.05)
    
    def update_u(u,x,t,n):
        ax = gca()
        ax.lines[0].set_xdata(u)
        ax.set_title('t=%f' % (t[n]/(3600*24.)))
        draw()
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)
    x = linspace(0, Z, Nx+1)   # mesh points in space
    u = zeros(Nx+1) 
    plot_u(u, x, t, 0)
    
    user_action = update_u if animate else lambda u,x,t,n: None

    u, x, t, cpu = eval('solver_'+scheme)\
                   (I, k, C, Z, Nx, dt, T, ub=ub, lb=lb,
                    user_action=user_action)
    return u, cpu    

def vizTINL(I, Layers, Nx, dt, T, ub=lambda x: 0., lb=lambda x: 0.,
        umin=0., umax=1.1, scheme='FE', version='vectorized', animate=True, framefiles=False):

    def plot_u(u, x, t, n, ax):
        #ax = gca()
        ax.hold(False)
        h = plot(u, x, 'r-', marker='.', ms=5)
        ax.set_title('t=%f' % (t[n]/(3600*24.)))
        #ax.set_ylim([Layers.surface_z,Layers.z_max])
        ax.set_ylim([Layers.surface_z-0.1 ,np.min([5., Layers.z_max])])
        ax.set_xlim([umin,umax])
        ax.invert_yaxis()
        if framefiles:
            savefig('tmp_frame%04d.png' % n)
        if t[n] == 0:
            time.sleep(1)
        elif not framefiles:
            # It takes time to write files so pause is needed
            # for screen only animation
            draw()
            show()
            #time.sleep(0.05)
    
    def update_u(u,x,t,n):
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax1.lines[0].set_xdata(u)
        ax1.set_title('t=%f' % (t[n]/(3600*24.)))
        ax2 = fig.axes[1]
        ax2.lines[0].set_xdata(np.diff(u))
        ax2.set_title('t=%f' % (t[n]/(3600*24.)))
        xl = ax2.get_xlim()
        
        xl2 = [np.min([np.min(np.diff(u)), xl[0]]), np.max([np.max(np.diff(u)), xl[1]])]
        print xl2
        ax2.set_xlim(xl2)
        
        draw()
   
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)
    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    u = zeros(Nx+1) 
    fig = plt.figure(99)
    ax1 = plt.subplot(1, 2, 1)
    plot_u(u, x, t, 0, ax1)
    
    ax2 = plt.subplot(1, 2, 2)
    #pdb.set_trace()
    ax2.hold(False)
    ax2.plot(np.diff(u[:]),np.arange(len(u)-1),'b-')
    ax2.set_ylim([0, len(u)-1])
    ax2.invert_yaxis()
    
    user_action = update_u if animate else lambda u,x,t,n: None

    u, x, t, cpu = eval('solver_'+scheme)\
                   (I, Layers, Nx, dt, T, ub=ub, lb=lb,
                    user_action=user_action, version=version)
    return u, cpu  


def pf_test(scheme='CN', Nx=100, version='vectorized'):
    pylab.ion()
    Layers = LayeredModel(type='unfrw')

#    Layers.add(Thickness=5,  n=0.40, C_th=2.00E6, C_fr=1.50E6, k_th=0.9, k_fr=1.3, alpha=0.06, beta=0.408, Tf=-0.0001, soil_type='Peat')
    Layers.add(Thickness=5,  n=0.35, C_th=1.00E6, C_fr=2.00E6, k_th=1.2, k_fr=2.0, alpha=0.21, beta=0.19, Tf=-0.0001, soil_type='Fairbanks Silt')
    
    #plot_unfrw(Layers[0])
    #plot_unfrw(Layers[1])
    
    #k = 0.8             # W/m*K
    #C = 1500*2000        # J/kg*K * kg/m3 = J/m^3*K
    T = 10*365*3600*24.     # seconds
    dt = 1*3600*24.    # seconds

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    F = dt/dx**2. * Layers[0]['k_th']/Layers[0]['C_th']
    
    #while True:
    #    F = dt/dx**2. * Layers[0]['k_th']/Layers[0]['C_th']
    #    if F >= 0.5:
    #        dt = dt/2.
    #    else:
    #        break
    
    print dt/(3600*24), F
    
    def I(x):
        """Gaussian profile as initial condition."""
        return -1.

    u, cpu = vizTINL(I, Layers, Nx, dt, T, 
                 ub=lambda x: 10*sin(2*pi/(365*3600*24)*x)-1, lb=lambda x: -1.,
                 umin=-10, umax=10,
                 scheme=scheme, version=version, animate=True, framefiles=False)
    print u                 
    print 'CPU time:', cpu    


def std_test(scheme='CN', Nx=100, version='layered'):
    Layers = LayeredModel(type='std')
    Layers.add(Thickness=0.25, C=1., k=0.2, soil_type='Peat')
    Layers.add(Thickness=0.25, C=1., k=0.4, soil_type='Fairbanks Silt')
    Layers.add(Thickness=0.25, C=1., k=4., soil_type='Fairbanks Silt')
    #k = 0.8             # W/m*K
    #C = 1500*2000        # J/kg*K * kg/m3 = J/m^3*K
    T = 365*3600*24.     # seconds
    dt = 1.*3600*24.    # seconds

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    F = dt/dx**2. * Layers[0]['k']/Layers[0]['C']
    
    #while True:
    #    F = dt/dx**2. * Layers[0]['k']/Layers[0]['C']
    #    if F >= 0.5:
    #        dt = dt/2.
    #    else:
    #        break
    
    print dt/(3600*24), F
    
    def I(x):
        """Gaussian profile as initial condition."""
        return 0.

    u, cpu = vizTINL(I, Layers, Nx, dt, T, 
                 ub=lambda x: 5, lb=lambda x: 0.5,
                 umin=-10, umax=10,
                 scheme=scheme, version=version, animate=True, framefiles=False)
    print u                 
    print 'CPU time:', cpu        

    
    
    
def expsin(scheme='FE', F=0.5, m=3):
    L = 10.0
    a = 1
    T = 3

    def exact(x, t):
        return exp(-m**2*pi**2*a/L**2*t)*sin(m*pi/L*x)

    def I(x):
        return exact(x, 0)

    Nx = 80
    viz(I, a, L, Nx, F, T, -1, 1, scheme=scheme, animate=True,
        framefiles=True)

    # Convergence study
    def action(u, x, t, n):
        e = abs(u - exact(x, t[n])).max()
        errors.append(e)

    errors = []
    Nx_values = [10, 20, 40, 80, 160]
    for Nx in Nx_values:
        eval('solver_'+scheme)(I, a, L, Nx, F, T, user_action=action)
        dt = F*(L/Nx)**2/a
        print dt, errors[-1]



def test_solvers():
    def u_exact(x, t):
        return (L-x)**2*5*t  # fulfills BC at x=0 and x=L

    def I(x):
        return u_exact(x, 0)

    def f(x, t):
        return (L-x)**2*t - a*2*5*t

    a = 3.5
    L = 1.5
    import functools
    s = functools.partial  # object for calling a function w/args
    solvers = [
        s(solver_FE_simple, I=I, a=a, L=L, F=0.5, T=2, Nx=100),
        s(solver_FE,        I=I, a=a, L=L, F=0.5, T=2, Nx=100,
          user_action=None, version='scalar'),
        s(solver_FE,        I=I, a=a, L=L, F=0.5, T=2, Nx=100,
          user_action=None, version='vectorized'),
        s(solver_BE_simple, I=I, a=a, L=L, F=0.5, T=2, Nx=100),
        s(solver_BE,        I=I, a=a, L=L, F=0.5, T=2, Nx=100,
          user_action=None),
        s(solver_theta,     I=I, a=a, L=L, F=0.5, T=2, Nx=100,
          theta=0, u_L=0, u_R=0, user_action=None),
        ]
    for solver in solvers:
        print 'A'
        u, x, t, cpu = solver()
        u_e = u_exact(x, t[-1])
        diff = abs(u_e - u).max()
        tol = 1E-14
        assert diff < tol, 'max diff: %g' % diff




        
# --------------------------------------------------------------
#
# Calculation of unfrozen water content
#
# --------------------------------------------------------------    

def f_Tstar(Tf, S_w, a, b):
    """Calculation of the effective freezing point, T_star."""
    return Tf-np.power((S_w/a),(-1/b))
        
        
def f_phi(T, a, b, Tstar, S_w):
    """Calculates the unfrozen water fraction."""
    return np.where(T < Tstar,
                         a*np.power(np.abs(T-Tstar),-b),
                         np.ones_like(T)*S_w)

                         
def f_unfrozen_water(T, a, b, Tstar, n):
    """Calculates the unfrozen water content [m^3/m^3]."""
    return f_phi(T, a, b, Tstar, 1.0) * n

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

def f_C_app(C_eff, T_0, T_1, alpha, beta, T_star, n):
    return C_eff + L * alpha * beta * np.abs(Tf-T)**(-beta-1)



# --------------------------------------------------------------
#
# Plot unfrozen water content
#
# --------------------------------------------------------------        
        
def plot_unfrw(params, T1=-10, T2=2):

    T = np.linspace(T1,T2,300)
    
    Tstar = f_Tstar(params['Tf'], 1.0, params['alpha'], params['beta'])
    unfrw = f_unfrozen_water(T, params['alpha'], params['beta'], Tstar, params['n'])
    
    fig = figure()
    plot(T,unfrw,'-k')
    
    show()
    draw()
    
    pdb.set_trace()
    
    
    
        
        
        
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print """Usage %s function arg1 arg2 arg3 ...""" % sys.argv[0]
        sys.exit(0)
    cmd = '%s(%s)' % (sys.argv[1], ', '.join(sys.argv[2:]))
    print cmd
    eval(cmd)