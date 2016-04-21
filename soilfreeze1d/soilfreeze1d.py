#!/usr/bin/env python
# As v1, but using scipy.sparse.diags instead of spdiags
"""
Functions for solving the 1d heat equation:
      (k*u_x)_x = C*u_t + L*dtheta/dt 
with boundary conditions u(t,0)=ub(t) and u(t,L)=lb(t) 
or du(t,L)/dx=grad, for t in [t0,T+t0].

Upper boundary is a dirichlet type specified forcing
temperature, while lower boundary is either dirichlet
type or neumann type (specified gradient).
The initial condition is u(x,0)=initialTemperature(x).

The following naming convention of variables are used.

===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
t0    The start time of the simulation      
T     The duration of the simulation.
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
import os.path
import numpy as np
from numpy import linspace, zeros, pi
from matplotlib.pyplot import plot, savefig, draw, show, figure
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.sparse
import scipy.sparse.linalg



days = 3600.*24      # number of seconds in a day


class HarmonicTemperature(object):
    """Calculates harmonically varying temperature, to be used as
    upper boundary condition in a fixed upper boundary solution.
    
    time       Time in seconds.
    maat       Mean Annual Airt Temperature.
    amplitude  Amplitude of the yearly variation.
    lag        Phase lag (number of days to delay the harmonic oscillation).

    The period of the oscillation is 365.242 days = 1 year
    """
    def __init__(self, maat=-2, amplitude=8, lag=0*days):
        self.maat = maat
        self.amplitude = amplitude
        self.lag = lag
    
    def __call__(self, time):    
        """Returns the temperature at the specified time using a harmonic
        temperature variation.
        """
        return self.maat - self.amplitude * np.cos(2*pi*(time-self.lag)/(365.242*days))


class TimeInterpolator(object):
    """Class to make linear interpolation of time series, keeping
    track of the last location in the time series for quick
    positioning.
    This class will be usefull for sequential time stepping
    with occasional reverse time steps.
    It will be slow for general purpose interpolation.
    
    Usage:
    
    # Initialization
    time_interp = TimeInterpolator(0, times, temperatures)
    
    # Get interpolated temperature at t=27.15
    time_interp(27.15)
    
    # Get interpolated temperature at t=27.30
    time_interp(27.30)
    
    # Reset the interpolation tracking:
    time_interp.reset()
    """

    def __init__(self, x, xp, fp):
        
        self.xp = xp       # interpolation x-values (times)
        self.fp = fp       # interpolation y-values (temperatures)
        self.x = None      # point at which to find interpolation
        self.id = 0        # id into xp for closest value less than x

    def __call__(self, x):
        self.x = x
        
        # rewind if time has been stepped backwards
        while self.x < self.xp[self.id] and self.id > 0:
            self.id -= 1

        # fast forward if time has been stepped up
        while self.x > self.xp[self.id+1] and self.id < len(self.xp)-2:
            self.id += 1

        # do interpolation
        return np.interp(self.x, self.xp[self.id:self.id+2], self.fp[self.id:self.id+2])

    def reset(self):
        self.id = 0



class DistanceInterpolator(object):
    """Class to make linear interpolation of temperature profiles.
    
    Usage:
    
    # Initialization
    z_interp = DistanceInterpolator(depths, temperatures)
    
    # Get interpolated temperatures at the locations specified in
    # the interpolation_depths array.
    temp = z_interp(intepolation_depths)    
    
    The interpolator will return fp[0] for distances less than depths[0]
    and fp[-1] for distances larger than depths[-1]
    """

    def __init__(self, depths=None, temperatures=None):
        self.zp = depths          # interpolation x-values (times)
        self.fp = temperatures    # interpolation y-values (temperatures)

    def __call__(self, z_int):
        # do interpolation
        return np.interp(z_int, self.zp, self.fp)


class FileStorage(object):
    """Class to handle storage of modelling results.
    
    FileStorage.add(t, u) will add the results given by u, if
    t (rounded to the specified number of decimals) is a 
    multiple of "interval" 
    
    Values will be stored to disk when the buffer has been filled.
    
    """
    
    def __init__(self, filename, depths, interval, decimals=6, 
                 buffer_size=20, append=False):
        self.filename = filename
        self.append = append
        self.depths = depths
        self.interval = interval
        self.decimals = decimals
        self.buffer_size = buffer_size
        self._buffer = np.ones((buffer_size, len(depths)+2))
        self.count = 0
        self.initialize()
    
    def initialize(self):
        """Creates the storage file and writes column headers.
        If file exists and should be appended, do not write header!        
        """
        
        if self.append:
            if os.path.exists(self.filename):
                # The file already exists, so do nothing
                return
        
        with open(self.filename, 'w') as f:
            # Write the header row
            f.write('{0:16s}'.format('Time[seconds]'))
            f.write('; {0:12s}'.format('SurfTemp[C]'))
            # Loop over all depths 
            for did in xrange(len(self.depths)):
                # write separator and temperature
                f.write('; {0:+8.3f}'.format(self.depths[did]))
            # write line termination
            f.write('\n')
        # file is automatically closed when using the "with .. as" construct
                
    def add(self, t, st, u):
        """Adds the current time step, if t is a multiple of the
        specified interval."""
        
        t = np.round(t, self.decimals) # round t to specified number of decimals
        
        if np.mod(t, self.interval) == 0:
            # t is a multiple of interval...
            self._buffer[self.count, 0] = t  # store time
            self._buffer[self.count, 1] = st  # store time
            self._buffer[self.count, 2:] = u  # store results 
            self.count += 1    # increment counter
            
            if self.count == self.buffer_size:
                # We have filled the buffer, now write to disk
                self.flush()
        else:
            # t is not at a regular storage interval, so do nothing
            pass
    
    def flush(self):
        """Writes buffer to disk file."""
        with open(self.filename, 'a') as f:  # open the file
            # loop over all rows in buffer
            for rid in xrange(self.count):
                # Write the time step
                f.write('{0:16.3f}'.format(self._buffer[rid, 0])) 
                f.write('; {0:+12.3f}'.format(self._buffer[rid, 1]))
                # Loop over all the temperatures in the row
                for cid in xrange(len(self.depths)):
                    # write separator and temperature
                    f.write('; {0:+8.3f}'.format(self._buffer[rid, cid+2]))
                
                # write line termination
                f.write('\n')
        self.count = 0
        # file is automatically closed when using the "with .. as" construct




class LayeredModel(object):
    _descriptor_unfrw = {'names': ('Thickness', 'n', 'C_th', 'C_fr', 'k_th', 'k_fr', 'alpha',  'beta',  'Tf', 'Soil_type'), 
                         'formats': ('f8',      'f8',  'f8',   'f8',   'f8',   'f8',   'f8', 'f8', 'f8', 'S50')}

    _descriptor_std =   {'names': ('Thickness', 'C',  'k',  'Soil_type'), 
                         'formats': ('f8',      'f8', 'f8', 'S50')}
                         
    _descriptor_stefan = {'names': ('Thickness', 'n',  'C_th', 'C_fr', 'k_th', 'k_fr', 'Tf', 'interval', 'Soil_type'), 
                          'formats': ('f8',      'f8', 'f8',   'f8',   'f8',   'f8',   'f8', 'f8',       'S50')}
                         
    def __init__(self, type='stefan', surface_z=0., interval=1, Tf=0.):
        self._layers = None
        self.parameter_set = type
        self._descriptor = getattr(self, '_descriptor_'+type)
        self.surface_z = surface_z  # z-axis is positive down!
        if type=='stefan':
            self.interval = interval
            self.Tf = Tf
            
    
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


    # Layers[n] should return an ordered dictionary of all parameters for layer n
    # Layers.C_th should return an array of C_th for all layers
    # get_node_values(Layers.C_th, x) should return an array same shape as x, with values generated from C_th layer parameter
    # get_cell_values(Layers.k_th, x) should return an array of shape len(x)-1, with values generated from k_th layer parameter
        
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
    
    
    def show(self, T1=-10, T2=2, fig=None):
        
        # allow a maximum of five layers for plotting
        nlayers = len(self)
        if nlayers > 5:
            raise NotImplementedError('Visualization of more than 5 layers is not yet implemented')
        
        # Select figure window to plot
        if fig is None:
            fig = plt.figure() # use new window in no window specified
        else:
            fig = plt.figure(fig)
        
        # Create axes for the layered model display
        ax1 = plt.subplot2grid((nlayers,2), (0,0), rowspan=nlayers)

        # Prepare to plot unfrozen water        
        axes = []
        T = np.linspace(T1,T2,300)
        
        # make list of all depths, including surface
        ldepths = [self.surface_z]
        ldepths.extend(self.surface_z+np.cumsum(self._layers['Thickness']))        
        
        # loop over all layers
        for n in xrange(nlayers):
            # plot top of layer as line in ax1
            ax1.axhline(y=ldepths[n], ls='-', color='k')
            ax1.set_ylim([ldepths[0], ldepths[-1]])            
            ax1.invert_yaxis()
            
            # Create axis for unfrozen water plot
            axes.append(plt.subplot2grid((nlayers,2), (n,1)))
            
            # Select type of unfrozen water
            if self.parameter_set == 'unfrw':
                # unfrw = n*a*|T-Tf|**-b
                Tstar = f_Tstar(self[n]['Tf'], 1.0, self[n]['alpha'], self[n]['beta'])
                unfrw = f_unfrozen_water(T, self[n]['alpha'], self[n]['beta'], 
                                         self[n]['Tf'], Tstar, self[n]['n'])
            elif self.parameter_set == 'stefan':
                # unfrozen water is linear between Tf-interfal and Tf
                #phi = np.ones_like(T)*np.nan
                #phi[T>self[n]['Tf']] = 1.0  # The 1.0 is the water saturation
                #phi[T<=self[n]['Tf']-self[n]['interval']] = 0.0  # No unfrozen water
                #phi = np.where(np.isnan(phi), self[n]['interval']*T+1, phi)
                phi = f_phi_stefan(T, self[n]['Tf'], self[n]['interval'])
                unfrw = phi*self[n]['n']

            axes[-1].plot(T,unfrw,'-k')
            axes[-1].set_ylim([0,np.round(np.max(unfrw)*1.1, 2)])
            
        show()
        draw()
    
            
        
    
        
    

            
                
                
        
    
    # DONE: Add method to retrieve all parameters for a specific layers
    
    # DONE: Add method to calculate the grid point values of a specified parameter, given an array of gridpoint depths.
    
    # Add method to visualize the layered model somehow (including the unfrozen water content curve...)
        
        


    

def solver_theta(Layers, Nx, dt, T, t0=0, theta=1,
                 Tinit=lambda x: -2., 
                 ub=lambda x: 10., lb=lambda x: -2., lb_type=1, grad=0.09,
                 user_action=None,
                 outfile='model_result.txt',
                 outint=1*days):
    """Full solver for the model problem using the theta based finite difference 
    approximation. Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    
    lb_type=1   Dirichlet type lower boundary condition (specified temperature)
    lb_type=2   Neumann type lower boundary condition (specified gradient)    
    grad        The gradient [K/m] to use for the lower boundary
    """
    
    tstart = time.clock()
    
    L = 334*1e6 # [kJ/kg] => *1000[J/kJ]*1000[kg/m^3] => [J/m^3]
    
    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    Nt = int(round((T-t0)/float(dt)))
    t = linspace(t0, T, Nt+1)   # mesh points in time
    
    u   = zeros(Nx+1)   # solution array at t[tid+1]
    u_1 = zeros(Nx+1)   # solution at t[tid]
    u_bak = zeros(Nx+1)   # solution at t[tid+1], result from previous iteration

    dudT = np.ones(Nx+1)*-999.   # will hold derivative of unfrozen water
    dT = np.ones(Nx+1)*-999.     # will hold vertical temperature difference
                         # averaged over two cells except for first and last.
    
    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx)
    upper    = zeros(Nx)
    b        = zeros(Nx+1)
    A_m      = zeros(Nx)
    B_m      = zeros(Nx+1)
    C_m      = zeros(Nx)
    unfrw_u  = zeros(Nx+1)
    unfrw_u1 = zeros(Nx+1)

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
    elif Layers.parameter_set == 'stefan':
        print "Using stefan solution parameters"
        k_th = Layers.pick_values(x, 'k_th')
        C_th = Layers.pick_values(x, 'C_th')
        k_fr = Layers.pick_values(x, 'k_fr')
        C_fr = Layers.pick_values(x, 'C_fr')
        n = Layers.pick_values(x, 'n')
        
        #interval = Layers.interval
        #Tf = Layers.Tf
        interval = Layers.pick_values(x, 'interval')
        Tf = Layers.pick_values(x, 'Tf')
    else:
        print "Using standard parameters"
        k_th = Layers.pick_values(x, 'k')
        C_th = Layers.pick_values(x, 'C')
        k_fr = np.nan
        C_fr = np.nan
        n = Layers.pick_values(x, 'n')
    
    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = Tinit(x[i])

    u = u_1 + 0.001    # initialize u for finite differences
    
    if user_action is not None:
        user_action(u_1, x, t[0])

    datafile = FileStorage(outfile, depths=x, 
                           interval=outint, buffer_size=30)        
    datafile.add(t[0], ub(t[0]), u)
       
    # Time loop
    for tid in range(0, Nt):

        # u_1 holds the temperatures at time step n
        # u   will eventually hold calculated temperatures at step n+1
        
        u_bak = u_1 
        
        #pdb.set_trace()
        print '{0:d}'.format(tid) ,
        
        if Layers.parameter_set == 'unfrw':
            phi = f_phi_unfrw(u_1, alpha, beta, Tf, Tstar, 1.0)
        elif Layers.parameter_set == 'stefan':
            phi = f_phi_stefan(u_1, Tf, interval)
        else:
            phi = 1.  # for standard solution there is no phase change

        k_eff = f_k_eff(k_fr, k_th, phi)
        C_eff = f_C_eff(C_fr, C_th, phi)
        unfrw_u1 = n*phi
        
        #if tid == 600:
        #    pdb.set_trace()
        
        
        F = dt/(2*dx**2)        


        if Layers.parameter_set == 'unfrw':
            maxiter = 10
        else:
            maxiter = 1

        
        for iter in xrange(maxiter):
            
            if Layers.parameter_set == 'unfrw':
                # We calculate finite difference of unfrozen water based on the vertical
                # temperature gradient. We can do this because we have no significant
                # heat sources, and thus it is the vertical temperature difference
                # that drives the flow of energy.
                
                if False:
                    # THIS IS THE ORIGINAL ATTEMPT, WHICH DOES NOT WORK WELL
                    C_add_1 = - L * n * alpha * beta * np.abs(u_1-Tf)**(-beta-1)
                    
    #                dT[0:Nx-1] = u_1[0:Nx-1]-u_1[1:Nx]
    #                dT[Nx] = 0.
    #                dT[1:Nx] += dT[0:Nx-1]
    #                dT[1:Nx-1] = dT[1:Nx-1]/2
    #                
    #                dudT[0:Nx-1] = (unfrw_u1[0:Nx-1]-unfrw_u1[1:Nx])/(u_1[0:Nx-1]-u_1[1:Nx])
    #                dudT[Nx] = 0.
    #                dudT[1:Nx] += dudT[0:Nx-1]
    #                dudT[1:Nx-1] = dudT[1:Nx-1]/2                
    
    
                    dT[0:-1] = (u_1[0:-1]-u_1[1:])
                    dT[-1] = 0
                    dT[1:] += (u_1[1:]-u_1[0:-1])
                    dT[1:-1] *= 0.5   # divide by two inplace
    
                    
                    dudT[0:-1] = (unfrw_u1[0:-1]-unfrw_u1[1:])/(u_1[0:-1]-u_1[1:])
                    dudT[-1] = 0
                    dudT[1:] += (unfrw_u1[1:]-unfrw_u1[0:-1])/(u_1[1:]-u_1[0:-1])
                    dudT[1:-1] *= 0.5   # divide by two inplace
    
                    C_add_2 = L * dudT
                    
                    #C_add = np.where(np.logical_or(np.abs(dT)<1e-6, np.isnan(dudT)), C_add_1, C_add_2)
                    #C_add = np.where(np.isinf(dudT), C_add_1, C_add)
                    
                    C_add = C_add_1
            
                # NEW APPROACH TRYING AN ITERATION SCHEME
                if iter == 0:
                    # This is first iteration, approximate the latent heat 
                    # component by the analytical derivative
                    C_add_1 = L * n * alpha * beta * np.abs(u_1-Tf)**(-beta-1)
                    C_add = C_add_1
                    #print "I:{0:.2f}".format(C_add[10]),                    
                    
                else:
                    # A previous iteration exist, so an estimate of the
                    # next time step exists. Use that to calculate a finite
                    # difference.
                    
                    dT = (u-u_1)
                    dudT = (unfrw_u-unfrw_u1)/(u-u_1)
                    
                    C_add = np.where(np.isfinite(dudT), L*dudT, C_add_1)
                    #print "*:{0:.2f}".format(C_add[10]),                    
            
                # Apparent heat capacity is the heat capacity + the latent heat effect
                C_app = C_eff + C_add
                #print "C:{0:.2f}".format(C_app[10]),
                
            elif Layers.parameter_set == 'stefan':
                C_app = np.where(np.logical_and(np.less(u,Tf),np.
                                                greater_equal(u, Tf-interval)), 
                                 C_eff + L*n/interval, C_eff)
            else:
                C_app = C_eff
            
            if np.any(np.isnan(C_app)) or np.any(np.isinf(C_app)):
                pdb.set_trace()
            
            # Update all inner points
            A_m       = F*(k_eff[1:]+k_eff[0:-1])/C_app[1:]
            B_m[1:-1] = F*(k_eff[0:-2]+2*k_eff[1:-1]+k_eff[2:])/C_app[1:-1]
            C_m       = F*(k_eff[0:-1]+k_eff[1:])/C_app[0:-1]    
            #A_N = F*(k_eff[Nx]+k_eff[Nx-1])/C_app[Nx]    
            #B_N = F*(k_eff[Nx-1]+3*k_eff[Nx])/C_app[Nx]
            #C_N = F*(2*k_eff[Nx])/C_app[Nx]    
            #A_m = F*(2*k_eff[1:Nx])/C_app[1:Nx]
            #B_m = F*(4*k_eff[1:Nx])/C_app[1:Nx]
            #C_m = F*(2*k_eff[1:Nx])/C_app[1:Nx]    
            #A_N = B_m[-1]
            #B_N = B_m[-1]
            #C_N = C_m[-1]
            
            # Compute sparse matrix (scipy format)
            diagonal[1:-1] = 1 + theta*B_m[1:-1]
            lower = -theta*A_m  #1
            upper = -theta*C_m  #1
            
            # Insert boundary conditions (Dirichlet)
            diagonal[0] = 1
            upper[0] = 0
            
            if lb_type == 1:
                diagonal[Nx] = 1
                lower[-1] = 0
            elif lb_type == 2:
                #diagonal[Nx] = 1 + theta*B_N
                #lower[-1] = -theta*(A_N+C_N)  
                diagonal[Nx] = 1
                lower[-1] = -1

            else:
                raise ValueError('Unknown lower boundary type')

            U = scipy.sparse.diags(
                diagonals=[diagonal, lower, upper],
                offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                format='csr')
            #print A.todense()
                
            # Compute known vector
            b[1:-1] = u_1[1:-1] + (1-theta) * (A_m[0:-1]*u_1[:-2] - B_m[1:-1]*u_1[1:-1] + C_m[1:]*u_1[2:])
            b[0] = ub(t[tid])    # upper boundary conditions
            
            # Add lower boundary condition
            if lb_type == 1:
                b[-1] = lb(t[tid])  
            elif lb_type == 2:
                #b[-1] = u_1[Nx] + (1-theta) * ((A_N+C_N)*u_1[-2] - B_N*u_1[-1]) - 2*C_N*dx*grad
                b[-1] = dx*grad
                
            u[:] = scipy.sparse.linalg.spsolve(U, b)

            if Layers.parameter_set == 'unfrw':
                phi_u = f_phi_unfrw(u, alpha, beta, Tf, Tstar, 1.0)
            elif Layers.parameter_set == 'stefan':
                phi_u = f_phi_stefan(u, Tf, interval)
            else:
                phi_u = 1.  # for standard solution there is no phase change
            
            unfrw_u = n*phi_u
            
            #if np.any(np.abs(unfrw_u-unfrw_u1)>0.05):
            #    dt = dt/2.
            #else:
            #    break
            
            if Layers.parameter_set == 'unfrw':
                #print "{0:.5f}".format(u[10]),
                change = (u-u_bak)/u_bak*100
                print "{0:.8f}".format(np.max(change)),
                
                #if np.any(change>=1.0):
                #    pdb.set_trace
    
                if np.max(change) < 0.001:
                    # break iteration loop
                    # since no improvement
                    break                
            
            u_bak = u.copy()
            
             
        print "Time step completed: {0:d} iterations".format(iter)
        #print "{0:d} iterations. Done!".format(iter)
        
        #dt = dt*2.

        u[0] = ub(t[tid+1])            
        if user_action is not None:
            user_action(u, x, t[tid+1])

        # Update u_1 before next step
        #u_1[:] = u

        datafile.add(t[tid+1], ub(t[tid+1]), u)
        
        u_1, u = u, u_1
    
    datafile.flush()
    tstop = time.clock()
    return u, x, t, tstop-tstart
    

class Visualizer_T_dT(object):
    def __init__(self, fig=None, ax1=None, ax2=None, z_max=np.inf, 
                 Tmin=None, Tmax=None):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.z_max = z_max
        self.Tmin = Tmin
        self.Tmax = Tmax
    
    def initialize(self, u, x, t, Layers, name=''):
        plt.figure(self.fig)

        if self.ax1 is None:
            self.ax1 = plt.subplot(1, 2, 1)
            self.ax1.hold(False)
            
        self.ax1.plot(u, x, 'r-', marker='.', ms=5)
        self.ax1.hold(True)
        self.ax1.axvline(x=0, ls='--', color='k')
        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
        self.ax1.set_ylim([Layers.surface_z-0.1 ,np.min([self.z_max, Layers.z_max])])
        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        
        if self.ax2 is None: 
            self.ax2 = plt.subplot(1, 2, 2)
            self.ax2.hold(False)
            
        self.ax2.plot(np.diff(u[:]),np.arange(len(u)-1),'b-')
        self.ax2.hold(True)
        self.ax2.axvline(x=0, ls='--', color='k')
        self.ax2.set_ylim([0, len(u)-1])
        self.ax2.invert_yaxis()
        
        if name:
            plt.figure(self.fig).suptitle(name)        
            
        plt.draw()
        plt.show()


    def __call__(self, u, x, t):
        self.ax1.lines[0].set_xdata(u)
        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
        self.ax2.lines[0].set_xdata(np.diff(u))
        self.ax2.set_title('t=%f' % (t/(3600*24.)))
        xl = self.ax2.get_xlim()
        
        xl2 = [np.min([np.min(np.diff(u)), xl[0]]), np.max([np.max(np.diff(u)), xl[1]])]
        self.ax2.set_xlim(xl2)
        
        plt.draw()
        
    def update(self, u, x, t):
        self.ax1.lines[0].set_xdata(u)
        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
        self.ax2.lines[0].set_xdata(np.diff(u))
        self.ax2.set_title('t=%f' % (t/(3600*24.)))
        xl = self.ax2.get_xlim()
        
        xl2 = [np.min([np.min(np.diff(u)), xl[0]]), np.max([np.max(np.diff(u)), xl[1]])]
        self.ax2.set_xlim(xl2)
        
        plt.draw()


class Visualizer_T(object):
    def __init__(self, fig=None, ax1=None, z_max=np.inf, 
                 Tmin=None, Tmax=None):
        self.fig = fig
        self.ax1 = ax1
        self.title = None
        self.z_max = z_max
        self.Tmin = Tmin
        self.Tmax = Tmax
    
    def initialize(self, u, x, t, Layers, name=''):
        plt.figure(self.fig)

        if self.ax1 is None:
            self.ax1 = plt.subplot(1, 1, 1)
            self.ax1.hold(False)
            
        self.ax1.set_ylim([Layers.surface_z-0.1 ,np.min([self.z_max, Layers.z_max])])

        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        self.ax1.axvline(x=0, ls='--', color='k')
        
        plt.draw()
        plt.show()

        self.background = plt.figure(self.fig).canvas.copy_from_bbox(self.ax1.bbox)
            
        self.ax1.plot(u, x, 'r-', marker='.', ms=5)
        self.ax1.hold(True)
        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
        self.ax1.set_ylim([Layers.surface_z-0.1 ,np.min([self.z_max, Layers.z_max])])
        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        
        if name:
            self.title = plt.figure(self.fig).suptitle(name)        
            
        plt.draw()
        plt.show()


    def __call__(self, u, x, t):
        self.ax1.lines[0].set_xdata(u)
        self.ax1.title.set_text('t=%f' % (t/(3600*24.)))

        # restore background
        plt.figure(self.fig).canvas.restore_region(self.background)

        # redraw just the points
        self.ax1.draw_artist(self.ax1.lines[0])
        self.ax1.draw_artist(self.ax1.title)

        # fill in the axes rectangle
        plt.figure(self.fig).canvas.blit(self.ax1.bbox)
        
        
    def update(self, u, x, t):
         self(u, x, t)       
#        self.ax1.lines[0].set_xdata(u)
#        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
#        plt.draw()

        
    
def test_FD_unfrw(scheme='theta', Nx=100, version='vectorized', fignum=99, theta=1.0, z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='unfrw')

    Layers.add(Thickness=1,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    Layers.add(Thickness=29,  n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    dt = 1*days          # seconds
    T = 10*365*days  # seconds
    

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    def initialTemperature(x):
        """Constant temperature as initial condition."""
        constant_temperature = -2.
        return np.ones_like(x)*constant_temperature

    surf_T = HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)

    # Set up the plotting
    plot_solution = Visualizer_T(Tmin=-8, Tmax=4, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., 0, Layers, scheme=scheme)
        
    if animate:
        user_action = plot_solution
    else:
        user_action = None
        
    u, x, t, cpu = solver_theta(Layers, Nx, dt, T, theta=theta, 
                                Tinit=initialTemperature, 
                                ub=surf_T, lb=lambda x: -2.,
                                user_action=user_action,
                                outfile='test_dat.txt',
                                outint=1*days)

    plot_solution.update(initialTemperature(x), x, t[-1], 0, Layers)
    
    print u                 
    print 'CPU time:', cpu    
    
    return u, dt, dx, surf_T(T)



def test_FD_stefan(scheme='theta', Nx=100, fignum=99, theta=1., z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='stefan', interval=1)

    Layers.add(Thickness=30,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Test 1')    
    #Layers.add(Thickness=28, n=0.3,  C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=-0.0001, soil_type='Test 2')
    
    dt = 0.2*days          # seconds
    T = (5*155+1)*0.2*days  # seconds
    

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    def initialTemperature(x):
        """Constant temperature as initial condition."""
        constant_temperature = -2.
        return np.ones_like(x)*constant_temperature

    surf_T = HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)

    # Set up the plotting
    plot_solution = Visualizer_T_dT(Tmin=-10, Tmax=6, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., Layers, scheme=scheme)
        
    if animate:
        user_action = plot_solution
    else:
        user_action = None
        
    u, x, t, cpu = solver_theta(Layers, Nx, dt, T, theta=theta, 
                                Tinit=initialTemperature, 
                                ub=surf_T, lb=lambda x: -2.,
                                user_action=user_action,
                                outfile='test_dat.txt',
                                outint=1*days)

    plot_solution.update(u, x, t[-1])
    
    print u                 
    print 'CPU time:', cpu    
    
    return u, dt, dx, surf_T(T)


def test_FD_stefan_grad(scheme='theta', Nx=100, fignum=99, theta=1., z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='stefan', interval=1)

    Layers.add(Thickness=30,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Test 1')    
    
    dt = 0.2*days          # seconds
    T = (5*155+1)*0.2*days  # seconds
    

    x = linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    def initialTemperature(x):
        """Constant temperature as initial condition."""
        constant_temperature = -2.
        return np.ones_like(x)*constant_temperature

    surf_T = HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)

    # Set up the plotting
    plot_solution = Visualizer_T_dT(Tmin=-10, Tmax=6, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., Layers, scheme=scheme)
        
    if animate:
        user_action = plot_solution
    else:
        user_action = None
        
    u, x, t, cpu = solver_theta(Layers, Nx, dt, T, theta=theta, 
                                Tinit=initialTemperature, 
                                ub=surf_T, lb_type=2, grad=0.08333,
                                user_action=user_action,
                                outfile='test_dat.txt',
                                outint=1*days)

    plot_solution.update(u, x, t[-1])
    
    print u                 
    print 'CPU time:', cpu    
    
    return u, dt, dx, surf_T(T)

        
# --------------------------------------------------------------
#
# Calculation of unfrozen water content
#
# --------------------------------------------------------------    

def f_Tstar(Tf, S_w, a, b):
    """Calculation of the effective freezing point, T_star."""
    return Tf-np.power((S_w/a),(-1/b))
        
        
def f_phi_unfrw(T, a, b, Tf, Tstar, S_w):
    """Calculates the unfrozen water fraction."""
    return np.where(T < Tstar,
                         a*np.power(np.abs(T-Tf),-b),
                         np.ones_like(T)*S_w)


def f_phi_stefan(T, Tf, interval):
    """Calculates the unfrozen water fraction."""
    # unfrozen water is linear between Tf-interfal and Tf
    phi = np.ones_like(T)*np.nan
    phi[np.greater(T,Tf)] = 1.0              # The 1.0 is the water saturation
    phi[np.less_equal(T,Tf-interval)] = 0.0  # No unfrozen water
    return np.where(np.isnan(phi), interval*T+1, phi)
    
                         
def f_unfrozen_water(T, a, b, Tf, Tstar, n, S_w=1.0):
    """Calculates the unfrozen water content [m^3/m^3]."""
    return f_phi_unfrw(T, a, b, Tf, Tstar, S_w) * n

# --------------------------------------------------------------
#
# Calculation of thermal conductivities
#
# --------------------------------------------------------------    

#def f_k_f(k_s, k_i, n):
#    """Calculates the frozen thermal conductivity []."""
#    return k_s**(1-n)*k_i**(n)
#    
#
#def f_k_t(k_s, k_w, n):
#    """Calculates the thawed thermal conductivity []."""
#    return k_s**(1-n)*k_w**(n)
    
    
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

    
#def f_C_f(C_s, C_i, n):
#    """Calculates the frozen heat capacity []."""
#    return C_s*(1-n)+C_i*(n)
#
#    
#def f_C_t(C_s, C_w, n):
#    """Calculates the thawed heat capacity []."""
#    return C_s*(1-n)+C_w*(n)


# --------------------------------------------------------------
#
# Plot unfrozen water content
#
# --------------------------------------------------------------        
        
def plot_unfrw(params, T1=-10, T2=2):

    T = np.linspace(T1,T2,300)
    
    Tstar = f_Tstar(params['Tf'], 1.0, params['alpha'], params['beta'])
    unfrw = f_unfrozen_water(T, params['alpha'], params['beta'], Tstar, params['n'])
    
    figure()
    plot(T,unfrw,'-k')
    
    show()
    draw()
    
    pdb.set_trace()
    


def plot_trumpet(fname, start=0, end=-1, **kwargs):
    figBG   = 'w'        # the figure background color
    axesBG  = '#ffffff'  # the axies background color
    textsize = 8        # size for axes text
    
    fh = None

    if kwargs.has_key('axes'):
        ax = kwargs.pop('axes')
    elif kwargs.has_key('Axes'):
        ax = kwargs.pop('Axes')
    elif kwargs.has_key('ax'):
        ax = kwargs.pop('ax')
    else:
        fh = plt.figure(facecolor=figBG)
        ax = plt.axes(axisbg=axesBG)

    if fh is None:
        fh = ax.get_figure()

    hstate = ax.ishold()

    data = np.loadtxt(fname, skiprows=1, delimiter=';')
    with open(fname, 'r') as f:
        line = f.readline()
    
    time = data[:,0]/(1*days)
    surf_T = data[:,1]
    data = data[start:end,2:]
    
    depths = np.array(line.split(';')[2:], dtype=float)    

    Tmax = data.max(axis=0)
    Tmin = data.min(axis=0)    

    ax.axvline(x=0,linestyle='--', color='k')    
    
    ax.plot(Tmax,depths, '-r')
    ax.plot(Tmin,depths, '-b')
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.tick_params(axis='both', direction='out')
    ax.hold(hstate)    

    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Temperature [C]')

    textstr = '{0}\nstart day = {1:.2f}\nend day = {2:.2f}\nnumber of days = {3:.2f}'.format(fname,
                                                                                        time[start],
                                                                                        time[end],
                                                                                        (time[end]-time[start]))
                                                                                        
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    pylab.draw()

    return ax
    
    

def plot_surf(fname, annotations=True, figsize=(15,6),
                cmap=plt.cm.bwr, node_depths=True, cax=None,
                cont_levels=[0], **kwargs):

    figBG   = 'w'        # the figure background color
    axesBG  = '#ffffff'  # the axies background color
    textsize = 8        # size for axes text
    left, width = 0.1, 0.8
    rect1 = [left, 0.2, width, 0.6]     #left, bottom, width, height

    fh = None

    if kwargs.has_key('axes'):
        ax = kwargs.pop('axes')
    elif kwargs.has_key('Axes'):
        ax = kwargs.pop('Axes')
    elif kwargs.has_key('ax'):
        ax = kwargs.pop('ax')
    else:
        fh = plt.figure(figsize=figsize,facecolor=figBG)
        ax = plt.axes(rect1, axisbg=axesBG)

    if fh is None:
        fh = ax.get_figure()

    hstate = ax.ishold()

    data = np.loadtxt(fname, skiprows=1, delimiter=';')
    with open(fname, 'r') as f:
        line = f.readline()
    
    time = data[:,0]/(1*days)
    surf_T = data[:,1]
    data = data[:,2:]
    
    depths = np.array(line.split(';')[2:], dtype=float)    
    
    # Find the maximum and minimum temperatures, and round up/down
    mxn = np.max(np.abs([np.floor(data.min()),
           np.ceil(data.max())]))
    levels = np.arange(-mxn,mxn+1)

    xx, yy  = np.meshgrid(time, depths)

    cf = ax.contourf(xx, yy, data.T, levels, cmap=cmap)
    ax.hold(True)
    if cont_levels is not None:
        ct = ax.contour(xx, yy, data.T, cont_levels, colors='k')
        cl = ax.clabel(ct, cont_levels, inline=True, fmt='%1.1f $^\circ$C',fontsize=12, colors='k')

    if node_depths:
        xlim = ax.get_xlim()
        ax.plot(np.ones_like(depths)*xlim[0]+0.01*np.diff(xlim),depths,
                'ko', ms=3)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.tick_params(axis='both', direction='out')
    ax.hold(hstate)
    #ax.xaxis_date()

    cbax = plt.colorbar(cf, orientation='horizontal', ax=cax, shrink=1.0)
    cbax.set_label('Temperature [$^\circ$C]')
    #fh.autofmt_xdate()

    if annotations:
        fh.suptitle(fname)

    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Time [days]')

    pylab.draw()

    return pylab.gca()


    
    
# --------------------------------------------------------------
#
# Test functionality
#
# --------------------------------------------------------------         

def test_FileStorage():
    datafile = FileStorage('testdata.txt', depths=[0, 1, 2, 3, 4], 
                           interval=1.0*days, buffer_size=5)
    dt = 0.5*days
    for n in np.arange(20):
        datafile.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4]))

    datafile.append = False
    datafile.initialize()
    dt = 0.5*days
    for n in np.arange(20):
        datafile.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4])*2)

    
    datafile.append = True
    dt = 0.5*days
    for n in np.arange(20):
        datafile.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4])*3)
    
    
def test_LayeredModel():
    Layers = LayeredModel(type='unfrw')

#    Layers.add(Thickness=5,  n=0.40, C_th=2.00E6, C_fr=1.50E6, k_th=0.9, k_fr=1.3, alpha=0.06, beta=0.408, Tf=-0.0001, soil_type='Peat')
#    Layers.add(Thickness=5,  n=0.35, C_th=1.00E6, C_fr=2.00E6, k_th=1.2, k_fr=2.0, alpha=0.21, beta=0.19, Tf=-0.0001, soil_type='Fairbanks Silt')

#    Layers.add(Thickness=30,  n=0.35, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.21, beta=0.19, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    Layers.add(Thickness=2,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    Layers.add(Thickness=28,  n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    Layers.show(fig=100)


    Layers = LayeredModel(type='stefan')    
    Layers.add(Thickness=2,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=-0.0001, soil_type='Test 1')    
    Layers.add(Thickness=28, n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=-0.0001, soil_type='Test 2')    
    
    Layers.show(fig=101)        
        
        
if __name__ == '__main__':
    test_FD_stefan_grad