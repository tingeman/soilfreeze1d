#!/usr/bin/env python
# As v1, but using scipy.sparse.diags instead of spdiags


"""
TODOs:
20200125:   Look at how to optimize with Numba (can't use **kwargs)


Completed:
20200125:   MemoryStorage implemented for solver_theta_nug
"""


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
import time
import os.path
import copy
import warnings
import fractions
import decimal
import numpy as np
import numbers
import array
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.sparse
import scipy.sparse.linalg



days = 3600.*24      # number of seconds in a day


class ConstantTemperature(object):
    """Defines a constant temperature, which could be used to as upper 
    boundary condition.
    
    When called with time as argument, the class returns the temperature
    passed dureing initialization.
    
    T = constant temperature
    
    time       Time in seconds.
    """
    
    def __init__(self, T=0.):
        self.T = T
        
    def __call__(self, time):    
        """Returns the constant temperature, regardless of the time passed.
        """
        return self.T


class HarmonicTemperature(object):
    """Calculates harmonically varying temperature, which could be used to 
    approximate the seasonal variation air temperature and applied as upper 
    boundary condition.
    
    When called with time as argument, the class returns a temperature
    according to the following formula:
    
    T = maat - amplitude * cos(2*pi*(time-lag)/(365.242*days))
    
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
        return self.maat - self.amplitude * \
                    np.cos(2*np.pi*(time-self.lag)/(365.242*days))


class SteppedInterpolator(object):
    """Produces a stepped time series based on input time series.
    If timeseries values X_n and X_n+1 are specified for time t_n and t_n+1, 
    X_n will be returned for all times in the range [t_n <= t < t_n+1],
    and X_n+1 will be returned for all times [t_n+1 <= t]
    
    times       Time in seconds.
    values      Timeseries values.
    """
    
    def __init__(self, times, values):
        self.times = times
        self.values = values
    
    def __call__(self, time):    
        """Returns the value of the current step"""
        try:
            id = np.max(np.where(self.times<=time)[0])
        except:
            id = 0
        return self.values[id]


class TimeInterpolator(object):
    """Class to make linear interpolation of time series, keeping track of the 
    last location in the time series for quick positioning.
    This class will be usefull for sequential time stepping with occasional 
    reverse time steps. It could be used e.g. for interpolation of observed
    air temperature data to be used as upper boundary condition.
    It will be slow for general purpose interpolation.
    
    Usage:
    
    # Initialization
    time_interp = TimeInterpolator(times, temperatures)
    
    # Get interpolated temperature at t=27.15
    time_interp(27.15)
    
    # Get interpolated temperature at t=27.30
    time_interp(27.30)
    
    # Reset the interpolation tracking:
    time_interp.reset()
    """

    def __init__(self, xp, fp, x=0):
        
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
    
    def __init__(self, filename, interval=1., depths=None, nodes=None, decimals=6, 
                 buffer_size=20, append=False):
        self.filename = filename
        self.append = append
        self.interval = interval
        self.depths = depths
        self.nodes = nodes
        self.decimals = decimals
        self.buffer_size = buffer_size
        if self.nodes is None:
            self._buffer = np.ones((buffer_size, len(self.depths)+2))
        else:
            self._buffer = np.ones((buffer_size, len(self.nodes)+2))
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
        
        if os.path.exists(os.path.dirname(os.path.abspath(self.filename))):
            pass
        else:
            os.mkdir(os.path.dirname(os.path.abspath(self.filename)))
        
        with open(self.filename, 'w') as f:
            # Write the header row
            f.write('{0:16s}'.format('Time[seconds]'))
            f.write('; {0:12s}'.format('SurfTemp[C]'))
            # Loop over all depths 
            if self.nodes is not None:
                for did in self.nodes:
                    # write separator and temperature, if node is to be output
                    f.write('; {0:+8.3f}'.format(self.depths[did]))
            else:
                for did in range(len(self.depths)):
                    # write separator and temperature, if node is to be output
                    f.write('; {0:+8.3f}'.format(self.depths[did]))
                    
            # write line termination
            f.write('\n')
        # file is automatically closed when using the "with .. as" construct
                
    def add(self, t, ub, u):
        """Adds the current time step, if t is a multiple of the
        specified interval."""
        
        t = np.round(t, self.decimals) # round t to specified number of decimals
        #print('FileStorage time: {0} ...'.format(t), end='')
        if np.mod(t, self.interval) == 0:
            # t is a multiple of interval...
            self._buffer[self.count, 0] = t  # store time
            self._buffer[self.count, 1] = ub  # store upper boundary value
            if self.nodes is not None:
                self._buffer[self.count, 2:] = u[self.nodes]  # store results 
            else:
                self._buffer[self.count, 2:] = u  # store results 
            self.count += 1    # increment counter
            
            if self.count == self.buffer_size:
                # We have filled the buffer, now write to disk
                self.flush()
            
            #print('stored!')
        else:
            # t is not at a regular storage interval, so do nothing
            pass
        
    def add_comment(self, text):
        """Adds a comment line to the file."""
        self.flush()

        with open(self.filename, 'a') as f:  # open the file
            f.write('# {0}\n'.format(text))
        
    
    def flush(self):
        """Writes buffer to disk file."""
        with open(self.filename, 'a') as f:  # open the file
            # loop over all rows in buffer
            for rid in range(self.count):
                # Write the time step
                f.write('{0:16.3f}'.format(self._buffer[rid, 0])) 
                f.write('; {0:+12.3f}'.format(self._buffer[rid, 1]))
                # Loop over all the temperatures in the row
                if self.nodes is None:
                    for cid in range(len(self.depths)):
                        # write separator and temperature
                        f.write('; {0:+8.3f}'.format(self._buffer[rid, cid+2]))
                else:
                    for cid in range(len(self.nodes)):
                        # write separator and temperature
                        f.write('; {0:+8.3f}'.format(self._buffer[rid, cid+2]))
                # write line termination
                f.write('\n')
        self.count = 0
        # file is automatically closed when using the "with .. as" construct
        
    def finalize(self):
        return self.filename


class MemoryStorage(object):
    """Class to handle storage of modelling results in memory.
    
    FileStorage.add(t, u) will add the results given by u, if
    t (rounded to the specified number of decimals) is a 
    multiple of "interval" (given in seconds)
    
    Values will be stored to memory only.
    
    """
    
    def __init__(self, interval=1., depths=None, nodes=None, decimals=6, append=False, buffer_size=None):
        # append has no meaning for this storage class
        # buffer_size has no meaning for this storage class
        self.interval = interval
        self.depths = depths
        if isinstance(nodes, numbers.Number) and not hasattr(nodes, '__iter__'):
            nodes = [nodes]
        self.nodes = nodes
        self.decimals = decimals
        
        self._header = []
        self._buffer = array.array('d')
        self._comments = []
        self.count = 0
        self.initialize()
    
    def initialize(self):
        """Creates the storage file and writes column headers.
        If file exists and should be appended, do not write header!        
        """
        
        self._header = []
        self._header.append('Time[seconds]')
        self._header.append('UpperBoundary')

        # Loop over all depths 
        if self.nodes is not None:
            for did in self.nodes:
                self._header.append('{0:+.3f}'.format(self.depths[did]))
        else:
            for did in range(len(self.depths)):
                self._header.append('{0:+.3f}'.format(self.depths[did]))
        
        self.count = 0
        self._buffer = array.array('d')

                
    def add(self, t, ub, u):
        """Adds the current time step, if t is a multiple of the
        specified interval."""
        
        t = np.round(t, self.decimals) # round t to specified number of decimals
        
        if np.mod(t, self.interval) == 0:
            # t is a multiple of interval...
            self._buffer.append(t)   # store time
            self._buffer.append(ub)  # store upper boundary value
            if self.nodes is not None:
                self._buffer.extend(u[self.nodes])  # store results 
            else:
                self._buffer.extend(u)              # store results 
            self.count += 1    # increment counter
        else:
            # t is not at a regular storage interval, so do nothing
            pass
        
    def add_comment(self, text):
        """Adds a comment line."""
        self._comments.append([self.count, text])
    
    def flush(self):
        """This method does nothing, but must be present to not break
        compatibilty with FileStorage."""
        pass
        
    def finalize(self):
        """Returns the buffer in numpy array format"""
        arr = np.array(self._buffer)
        if self.nodes is not None:
            cols = len(self.nodes)+2
        else:
            cols = len(self.depths)+2
        arr = np.array(self._buffer).reshape([-1, cols])
        
        return self._header, arr, self._comments


class NoStorage(object):
    """Class to handle when no storage is wanted.
    all methods are just 'pass'.
    """
    
    def __init__(self, interval=1., depths=None, nodes=None, decimals=6, append=False):
        pass
    
    def initialize(self):
        pass
                
    def add(self, t, ub, u):
        pass
        
    def add_comment(self, text):
        pass
    
    def flush(self):
        pass
        
    def finalize(self):
        return None


class LayeredModel(object):
    _descriptor =   {'names': ('Thickness', 'C',  'k',  'Soil_type'), 
                     'formats': ('f8',      'f8', 'f8', 'S50')}

    def __init__(self, type='std', surface_z=0., **kwargs):
        self._layers = None
        self.parameter_set = type
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
        for k,v in list(kwargs.items()):
            try:
                self._layers[-1][k] = v
            except:
                pass

    # Layers[n] should return an ordered dictionary of all parameters for layer n
    # Layers.C_th should return an array of C_th for all layers
    # get_node_values(Layers.C_th, x) should return an array same shape as x, with values generated from C_th layer parameter
    # get_cell_values(Layers.k_th, x) should return an array of shape len(x)-1, with values generated from k_th layer parameter
        
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._layers[key]
        else:
            return dict(list(zip(self._descriptor['names'],self._layers[np.int(key)])))
    
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
        for lid in range(len(self._layers)):
            condlist.append(np.logical_and(depths>=ldepths[lid], depths<ldepths[lid+1]))
        
        result = np.select(condlist, self._layers[param])
        
        # Handle the case where the last depth is exactly at or below the 
        # lower-most layer boundary, and therefore not assigned a value.
        ids = depths >= ldepths[-1]
        result[ids] = self._layers[-1][param]
                
        return result
    
    def show(self, T1=-10, T2=2, fig=None):
        raise NotImplementedError('Visualization of standard model not yet implemented')

    def f_unfrw_fraction(**kwargs):
        raise NotImplementedError('Unfrozen water is not implemented for standard model')
    
    def f_unfrozen_water(**kwargs):
        raise NotImplementedError('Unfrozen water is not implemented for standard model')
    
    def f_k_eff(k, **kwargs):
        return k
    
    def f_C_eff(C, **kwargs):
        return C



# change unfrw_swi model so that:
# 1) you fix the grid using set_grid method
# 2) this triggers the generation of C_s, C_w etc arrays, holding values for each grid point
# 3) f_xxxxxx methods use these precalculated arrays in the calculation of effective parameters

# The idea is to make the solver completely unaware of the type of layered model, and thus how effective parameters are calculated.
# so the code in the solver can be generic and independent of the parameters used in the layered model.        

# It will be difficult to make it unaware of unfrozen water, as the iterative scheme is
# based on convergence of the unfrozen water calculation...
# could maybe put the convergence criteria in a LayeredModel method...
# but that does not observe the single responsibility concept...

# Could also make a separate Convergence class, which would handle all
# about convergence tests and keep track of iterations and when to 
# step time or change time step...

class LayeredModel_std(LayeredModel):
    pass
        
class LayeredModel_unfrw_swi(LayeredModel):
    _descriptor = {'names': ('Thickness', 'n', 'C_s', 'C_w', 'C_i', 'k_s', 'k_w', 'k_i', 'alpha', 'beta', 'Tf', 'Soil_type'), 
                   'formats': ('f8',      'f8', 'f8', 'f8',  'f8',  'f8',  'f8',  'f8',  'f8',    'f8',   'f8', 'S50')}
                         
    def __init__(self, **kwargs):
        kwargs['type'] = 'unfrw_swi'
        super(LayeredModel_unfrw_swi, self).__init__(**kwargs)
            
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
        for n in range(nlayers):
            # plot top of layer as line in ax1
            ax1.axhline(y=ldepths[n], ls='-', color='k')
            ax1.set_ylim([ldepths[0], ldepths[-1]])            
            ax1.invert_yaxis()
            
            # Create axis for unfrozen water plot
            axes.append(plt.subplot2grid((nlayers,2), (n,1)))
            
            # unfrw = n*a*|T-Tf|**-b
            Tstar = self.f_Tstar(self[n]['Tf'], 1.0, self[n]['alpha'], self[n]['beta'])
            unfrw = self.f_unfrozen_water(T, self[n]['alpha'], self[n]['beta'], 
                                          self[n]['Tf'], Tstar, self[n]['n'])
                                     
            axes[-1].plot(T,unfrw,'-k')
            axes[-1].set_ylim([0,np.round(np.max(unfrw)*1.1, 2)])
            
        plt.draw()        
        plt.show()
        
    def f_Tstar(self, Tf, S_w, a, b):
        """Calculation of the effective freezing point, T_star."""
        return Tf-np.power((S_w/a),(-1/b))
            
    def f_unfrw_fraction(self, T, a, b, Tf, Tstar, S_w):
        """Calculates the unfrozen water fraction."""
        return np.where(T < Tstar,
                             a*np.power(np.abs(T-Tf),-b),
                             np.ones_like(T)*S_w)

    def f_unfrozen_water(self, T, a, b, Tf, Tstar, n, S_w=1.0):
        """Calculates the unfrozen water content [m^3/m^3]."""
        return self.f_unfrw_fraction(T, a, b, Tf, Tstar, S_w) * n

    def f_k_eff(self, k_s, k_w, k_i, n, phi):
        """Calculates the effective thermal conductivity []."""
        return k_s**(1-n)*k_w**(n*phi)*k_i**(n*(1-phi))        

    def f_C_eff(self, C_s, C_w, C_i, n, phi):
        """Calculates the effective heat capacity []."""
        return C_s*(1-n) + C_w*(n*phi) + C_i*(n*(1-phi))
        
    
class LayeredModel_unfrw_thfr(LayeredModel):
    _descriptor = {'names': ('Thickness', 'n', 'C_th', 'C_fr', 'k_th', 'k_fr', 'alpha',  'beta',  'Tf', 'Soil_type'), 
                   'formats': ('f8',      'f8',  'f8',   'f8',   'f8',   'f8',   'f8', 'f8', 'f8', 'S50')}
                         
    def __init__(self, **kwargs):
        kwargs['type'] = 'unfrw_thfr'
        super(LayeredModel_unfrw_thfr, self).__init__(**kwargs)
            
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
        for n in range(nlayers):
            # plot top of layer as line in ax1
            ax1.axhline(y=ldepths[n], ls='-', color='k')
            ax1.set_ylim([ldepths[0], ldepths[-1]])            
            ax1.invert_yaxis()
            
            # Create axis for unfrozen water plot
            axes.append(plt.subplot2grid((nlayers,2), (n,1)))
            
            # unfrw = n*a*|T-Tf|**-b
            Tstar = self.f_Tstar(self[n]['Tf'], 1.0, self[n]['alpha'], self[n]['beta'])
            unfrw = self.f_unfrozen_water(T, self[n]['alpha'], self[n]['beta'], 
                                          self[n]['Tf'], Tstar, self[n]['n'])
                                     
            axes[-1].plot(T,unfrw,'-k')
            axes[-1].set_ylim([0,np.round(np.max(unfrw)*1.1, 2)])
            
        plt.draw()        
        plt.show()
        
    def f_Tstar(self, Tf, S_w, a, b):
        """Calculation of the effective freezing point, T_star."""
        return Tf-np.power((S_w/a),(-1/b))
            
    def f_unfrw_fraction(self, T, a, b, Tf, Tstar, S_w):
        """Calculates the unfrozen water fraction."""
        return np.where(T < Tstar,
                             a*np.power(np.abs(T-Tf),-b),
                             np.ones_like(T)*S_w)

    def f_unfrozen_water(self, T, a, b, Tf, Tstar, n, S_w=1.0):
        """Calculates the unfrozen water content [m^3/m^3]."""
        return self.f_unfrw_fraction(T, a, b, Tf, Tstar, S_w) * n

    def f_k_eff(self, k_f, k_t, phi):
        """Calculates the effective thermal conductivity []."""
        return k_f**(1-phi)*k_t**(phi)        

    def f_C_eff(self, C_f, C_t, phi):
        """Calculates the effective heat capacity []."""
        return C_f*(1-phi)+C_t*(phi)          
                             

class LayeredModel_stefan(LayeredModel):
    _descriptor = {'names': ('Thickness', 'n',  'C_th', 'C_fr', 'k_th', 'k_fr', 'Tf', 'interval', 'Soil_type'), 
                   'formats': ('f8',      'f8', 'f8',   'f8',   'f8',   'f8',   'f8', 'f8',       'S50')}
                         
    def __init__(self, interval=1, Tf=0., **kwargs):
        kwargs['type'] = 'stefan'
        super(LayeredModel_stefan, self).__init__(**kwargs)
        self.interval = interval
        self.Tf = Tf
            
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
        for n in range(nlayers):
            # plot top of layer as line in ax1
            ax1.axhline(y=ldepths[n], ls='-', color='k')
            ax1.set_ylim([ldepths[0], ldepths[-1]])            
            ax1.invert_yaxis()
            
            # Create axis for unfrozen water plot
            axes.append(plt.subplot2grid((nlayers,2), (n,1)))
            
            # unfrozen water is linear between Tf-interfal and Tf
            phi = self.f_unfrw_fraction(T, self[n]['Tf'], self[n]['interval'])
            unfrw = phi*self[n]['n']
                                     
            axes[-1].plot(T,unfrw,'-k')
            axes[-1].set_ylim([0,np.round(np.max(unfrw)*1.1, 2)])
            
        plt.draw()        
        plt.show()
        
    def f_unfrw_fraction(self, T, Tf, interval):
        """Calculates the unfrozen water fraction."""
        # unfrozen water is linear between Tf-interfal and Tf
        phi = np.ones_like(T)*np.nan
        phi[np.greater(T,Tf)] = 1.0              # The 1.0 is the water saturation
        phi[np.less_equal(T,Tf-interval)] = 0.0  # No unfrozen water
        return np.where(np.isnan(phi), (interval**-1)*T+1, phi)

    def f_unfrozen_water(self, T, Tf, interval, n, S_w=1.0):
        """Calculates the unfrozen water content [m^3/m^3]."""
        return self.f_phi_unfrw(T, Tf, interval) * n

    def f_k_eff(self, k_f, k_t, phi):
        """Calculates the effective thermal conductivity []."""
        return k_f**(1-phi)*k_t**(phi)        

    def f_C_eff(self, C_f, C_t, phi):
        """Calculates the effective heat capacity []."""
        return C_f*(1-phi)+C_t*(phi)
        

        
        
class ConvergenceCriteria(object):
    unit = ''
    def __init__(self, threshold=0.05, max_iter=5):
        self.threshold = threshold
        self.max_iter = max_iter
        self.iteration = -1
        self.change = None
    
    def calc_change(self, u_0, u_1, unfrw_0, unfrw_1, dt_fraction):
        return None
    
    def has_converged(self, u_0, u_1, unfrw_0, unfrw_1, dt_fraction):
        self.change = self.calc_change(u_0, u_1, unfrw_0, unfrw_1, dt_fraction)

        if np.max(self.change) < self.threshold:
            return True
        else:
            return False
    
    def iterator(self):
        while self.iteration < self.max_iter:
            self.iteration += 1
            yield self.iteration

    def reset_iterator(self):
        self.iteration = -1
        
    def show(self):
        success = False
        attempts = 0
        while success == False and attempts < 10:
            try:
                if self.unit == '%':
                    print("{1:.8f}{0}".format(self.unit,np.max(self.change)*100), end=' ')
                else:
                    print("{1:.8f}{0}".format(self.unit,np.max(self.change)), end=' ')
                success = True
            except:
                pass
            attempts += 1
        #print "Max = {1:.8f} {0},   Min = {2:.8f} {0} ".format(self.unit, np.max(self.change), np.min(self.change))
        
    

class ConvCritNoIter(ConvergenceCriteria):
    def has_converged(self, *args):
        return True


class ConvCritUnfrw1(ConvergenceCriteria):
    unit = '%'
    def calc_change(self, u_0, u_1, *args):
        return (u_1-u_0)/u_0

        
class ConvCritUnfrw2(ConvergenceCriteria):
    unit = '%'
    def calc_change(self, u_0, u_1, *args):
        return np.abs((u_1-u_0)/u_0)

        
class ConvCritUnfrw3(ConvergenceCriteria):
    unit = '%'
    def calc_change(self, u_0, u_1, *args):
        return np.abs((u_1-u_0)/(u_0+273.15))
        
        
class ConvCritUnfrw4(ConvergenceCriteria):
    unit = 'C'
    def calc_change(self, u_0, u_1, uw_0, uw_1, dt_fraction):
        return np.abs(u_1-u_0)/np.float(dt_fraction)

        
        
        
        
        
def new_layered_model(type='', **kwargs):
    """Function to instantiate a layered model based on the type passed."""
    
    if type != '':
        return globals()['LayeredModel_'+type](**kwargs)
    else:
        return LayeredModel(**kwargs)
    
        
class SolverTime(object):
    """Class that handles adaptive time stepping for finite difference solver.
    """
    
    def __init__(self, t0, dt, dt_min=360, optimistic=False):
        #self.time = fractions.Fraction(t0).limit_denominator()
        #self.time = decimal.Decimal(round(t0, 8))
        self.time = np.float64(t0)
        self.previous_time = None
        self.dt_max = fractions.Fraction(dt).limit_denominator()
        self.dt_min = dt_min
        self.dt_fraction = fractions.Fraction(1,1)      
        self.dt = self.dt_max*self.dt_fraction
        self.step_up_allowed = False
        self.optimistic = optimistic
        self._o_counter = 0
        
    def _is_power2(self, num):
        """Tests if num is a power of two."""
        return ((num & (num - 1)) == 0) and num > 0
        
    def step(self):
        """Evolve time with the current time step size. Time step is increased
        if flag is set and time step fraction allows (we only step up when we 
        we are sure the new time step is in sync with the nominal (maximum) 
        time step.)
        
        """
        
        #if self.time == 33570196.875:
        #    pdb.set_trace()
        
        if self.optimistic and self._o_counter > 1:
            # If we are allowed to be optimistic, set step-up flag if the
            # last two steps did not result in a step size decrease.
            # = if today is no worse than yesterday, assume tomorrow will 
            #   be even better.
            self.step_up_allowed = True
        
        if self.step_up_allowed and self.dt_fraction < 1:
            # We are allowed to increase the time step
            if self.time%(self.dt*2) == 0:
                # yes new time step will be in sync, so we can increase
                self.dt_fraction *= 2
                self.dt = self.dt_max*self.dt_fraction
                self.step_up_allowed = False  # reset flag
            else:
                # No, we are not at a power-of-two fraction
                # do nothing and step up next time we get the chance
                pass
        else:
            # No we are not allowed to increase time step, do nothing
            pass
        
        self.previous_time = self.time  # store present time step
        self.time += self.dt  # take new step
        self._o_counter += 1
        self.show()
        
    def step_back(self):
        """Take backward step, restoring the previous time."""
        if self.previous_time is not None:
            self.time = self.previous_time
            self.previous_time = None
        else:
            raise ValueError('Cannot step backwards.')
        self.show()

    def increase_step(self, force=False):
        """Increase the current time step, dt, by a factor of 2."""
        if force and self.dt_fraction<1:
            self.dt_fraction *= 2
            self.dt = self.dt_max*self.dt_fraction
        else:
            self.step_up_allowed = True
        self.show()
        
    def decrease_step(self):
        """Reduce the current time step, dt, by a factor of 2."""
        if self.dt_fraction/2*self.dt_max >= self.dt_min:
            self.dt_fraction /= 2
            self.dt = self.dt_max*self.dt_fraction
            success = True
        else:
            success = False
        self.step_up_allowed = False
        self._o_counter = 0
        self.show()
        return success
        
    def __call__(self):
        """Return the current time."""
        return np.round(self.time, decimals=9)

    def show(self):
        return
        print("time:        {0}".format(self.time))
        print("time-1:      {0}".format(self.previous_time))
        print("dt_fraction: {0}".format(self.dt_fraction))        
        print("allow incr:  {0}".format(self.step_up_allowed))    


def solver_theta(Layers, Nx, dt, t_end, t0=0, dt_min=360, theta=1,
                 Tinit=lambda x: -2., 
                 ub=lambda x: 10., lb=lambda x: -2., lb_type=1, grad=0.09,
                 user_action=None,
                 conv_crit=None,
                 outfile='model_result.txt',
                 outint=1*days,
                 outnodes=None,
                 silent=False,
                 show_solver_time=True):
    """Uniform grid solver using the theta based finite difference 
    approximation. Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    
    Layers       LayeredModel (or subclass) instance defining layer parameters
    Nx           Number of equidistant grid points in domain
    dt           The maximum permissible time step [s]
    t_end        The end time of the model [s]
    t_0          The starting time of the model [s]
    dt_min       The minumum permissible time step [s]
    theta        Parameter determining the type of finite difference solution.
                     theta = 0:   Forward Euler solution
                     theta = 0.5: Crank-Nicholson solution
                     theta = 1:   Backward Euler solution
    Tinit        Function defining initial temperatures at all node points 
                     (takes depth as input argument)
    ub           Function returning the upper boundary temperature for any point in time. 
                     (takes time [s] as input argument)
    lb           Function returning the lower boundary value for any point in time. 
                     (takes time [s] as input argument)
    lb_type      Flag to set the type of lower boundary values:
                     lb_type=1   Dirichlet type lower boundary condition (specified temperature)
                     lb_type=2   Neumann type lower boundary condition (specified gradient)    
    grad         The gradient [K/m] to use for the lower boundary (presently not a function)
    user_action  Function to be called at every iteration, can handle any user plotting etc.
    conv_crit    ConvergenceCriteria (or subclass) instance defining the convergence
                     criteria for iterative search for unfrozen water content.
    outfile      Filename of the output data file
    outint       Frequency of data output [s]
    outnodes     List of indices of nodes to output in results file
    silent       Flag to determine if status messages are written to stdout.
    show_solver_time    If silent, solver time may still be printed to indicate progress.
    """

    tstart = time.monotonic()
    
    dt_stats = {}
    iter_stats = {}
    
    if conv_crit is None:
        if Layers.parameter_set in ['std','stefan']:
            conv_crit = ConvCritNoIter()
        else:
            conv_crit = ConvCritUnfrw4(threshold=1e-3, max_iter=5)
            
    L = 334*1e6 # [kJ/kg] => *1000[J/kJ]*1000[kg/m^3] => [J/m^3]
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # mesh points in space
    dx = x[1] - x[0]
    
    u   = np.zeros(Nx)   # solution array at t[tid+1]
    u_1 = np.zeros(Nx)   # solution at t[tid]
    u_bak = np.zeros(Nx)   # solution at t[tid+1], result from previous iteration

    dudT = np.ones(Nx)*-999.   # will hold derivative of unfrozen water
    
    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nx)
    lower    = np.zeros(Nx-1)
    upper    = np.zeros(Nx-1)
    b        = np.zeros(Nx)
    A_m      = np.zeros(Nx-1)
    B_m      = np.zeros(Nx)
    C_m      = np.zeros(Nx-1)
    unfrw_u  = np.zeros(Nx)
    unfrw_u1 = np.zeros(Nx)

    # Get constant layer parameters distributed on the grid
    if Layers.parameter_set == 'unfrw_thfr':
        if not silent: print("Using unfrozen water parameters")
        k_th = Layers.pick_values(x, 'k_th')
        C_th = Layers.pick_values(x, 'C_th')
        k_fr = Layers.pick_values(x, 'k_fr')
        C_fr = Layers.pick_values(x, 'C_fr')
        _n = Layers.pick_values(x, 'n')
        
        alpha = Layers.pick_values(x, 'alpha')
        beta = Layers.pick_values(x, 'beta')
            
        Tf = Layers.pick_values(x, 'Tf')
        Tstar = Layers.f_Tstar(Tf, 1.0, alpha, beta)
    elif Layers.parameter_set == 'unfrw_swi':
        if not silent: print("Using unfrozen water parameters")
        k_s = Layers.pick_values(x, 'k_s')
        C_s = Layers.pick_values(x, 'C_s')
        k_w = Layers.pick_values(x, 'k_w')
        C_w = Layers.pick_values(x, 'C_w')
        k_i = Layers.pick_values(x, 'k_i')
        C_i = Layers.pick_values(x, 'C_i')
        _n = Layers.pick_values(x, 'n')
        
        alpha = Layers.pick_values(x, 'alpha')
        beta = Layers.pick_values(x, 'beta')
            
        Tf = Layers.pick_values(x, 'Tf')
        Tstar = Layers.f_Tstar(Tf, 1.0, alpha, beta)        
    elif Layers.parameter_set == 'stefan':
        if not silent: print("Using stefan solution parameters")
        k_th = Layers.pick_values(x, 'k_th')
        C_th = Layers.pick_values(x, 'C_th')
        k_fr = Layers.pick_values(x, 'k_fr')
        C_fr = Layers.pick_values(x, 'C_fr')
        _n = Layers.pick_values(x, 'n')
        
        #interval = Layers.interval
        #Tf = Layers.Tf
        interval = Layers.pick_values(x, 'interval')
        Tf = Layers.pick_values(x, 'Tf')
    else:
        if not silent: print("Using standard parameters")
        k_eff = Layers.pick_values(x, 'k')
        C_eff = Layers.pick_values(x, 'C')
        k_th = np.nan
        C_th = np.nan
        k_fr = np.nan
        C_fr = np.nan
        phi = 1.
        unfrw_u1 = 0.
    
    # Set initial condition
    for i in range(0,Nx):
        u_1[i] = Tinit(x[i])

    u = copy.copy(u_1)    # initialize u for finite differences   
    
    if user_action is not None:
        user_action(u_1, x, t0)

    datafile = FileStorage(outfile, depths=x, nodes=outnodes, 
                           interval=outint, buffer_size=30)        
    datafile.add(t0, ub(t0), u)
       
    # solver_time always holds the time of the time-step
    # that is being calculated.
    solver_time = SolverTime(t0, dt, dt_min=dt_min, optimistic=True)
    step = 0    
    solver_time.step()
    
    if silent and show_solver_time:
        print('day:' + ' '*12, end=' ')
    
    # Time loop    
    while solver_time() <= t_end:      
        step += 1        
        
        # u_1 holds the temperatures at time step n
        # u   will eventually hold calculated temperatures at step n+1
        
        u_bak = u_1 
        
        try:
            if not silent:
                print('{0:6d}, t: {1:10.2f}, dtf: {2:>7s}   '.format(step, solver_time()/(1*days), solver_time.dt_fraction), end=' ')
            elif show_solver_time:
                print('\b'*11 + '{0:10.2f}'.format(solver_time()/(1*days)), end=' ')
        except:
            pass
                
        if Layers.parameter_set == 'std':
            #phi = 1.  # for standard solution there is no phase change
            #k_eff = k_th
            #C_eff = C_th
            #unfrw_u1 = 0.
            pass
        else:
            if Layers.parameter_set == 'unfrw_thfr':
                phi = Layers.f_unfrw_fraction(u_1, alpha, beta, Tf, Tstar, 1.0)
                
                k_eff = Layers.f_k_eff(k_fr, k_th, phi)
                C_eff = Layers.f_C_eff(C_fr, C_th, phi)
                unfrw_u1 = _n*phi
                
            elif Layers.parameter_set == 'unfrw_swi':
                phi = Layers.f_unfrw_fraction(u_1, alpha, beta, Tf, Tstar, 1.0)
                
                k_eff = Layers.f_k_eff(k_s, k_w, k_i, _n, phi)
                C_eff = Layers.f_C_eff(C_s, C_w, C_i, _n, phi)
                unfrw_u1 = _n*phi
                
            elif Layers.parameter_set == 'stefan':
                phi = Layers.f_unfrw_fraction(u_1, Tf, interval)

                k_eff = Layers.f_k_eff(k_fr, k_th, phi)
                C_eff = Layers.f_C_eff(C_fr, C_th, phi)
                unfrw_u1 = _n*phi
        
        F = solver_time.dt/(2*dx**2)        

        # Iterative scheme for estimating unfrozen water content
        convergence = False
        conv_crit.reset_iterator()
        for it in conv_crit.iterator():
            
            if Layers.parameter_set in ['unfrw_thfr', 'unfrw_swi']:
                if conv_crit.iteration == 0:
                    # This is first iteration, approximate the latent heat 
                    # component by the analytical derivative
                    C_add_1 = L * _n * alpha * beta * np.abs(u_1-Tf)**(-beta-1)
                    C_add = C_add_1
                else:
                    # A previous iteration exist, so an estimate of the
                    # next time step exists. Use that to calculate a finite
                    # difference for the unfrozen water content.

                    # The latent heat contribution is estimated based on
                    # a finite difference, where the temperature change
                    # is non-zero, and the slope of the unfrozen water
                    # content curve where the temperature change is
                    # exactly zero (produces infinite result the finite
                    # difference).
                    
                    # Temporarily ignore division by zero warning.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dudT = (unfrw_u-unfrw_u1)/(u-u_1)
                                        
                    C_add = np.where(np.isfinite(dudT), L*dudT, C_add_1)
            
                # Apparent heat capacity is the heat capacity + the latent heat effect
                C_app = C_eff + C_add
                
            elif Layers.parameter_set == 'stefan':
                C_app = np.where(np.logical_and(np.less(u,Tf),np.
                                                greater_equal(u, Tf-interval)), 
                                 C_eff + L*_n/interval, C_eff)
            else:
                C_app = C_eff
            
            if np.any(np.isnan(C_app)) or np.any(np.isinf(C_app)) or np.any(C_app <= 0):
                raise ValueError('Invalid NaN, Inf or zero value encountered in apparent heat capacity!')
            
            # Compute diagonal elements for inner points
            A_m       = F*(k_eff[1:]+k_eff[0:-1])/C_app[1:]
            B_m[1:-1] = F*(k_eff[0:-2]+2*k_eff[1:-1]+k_eff[2:])/C_app[1:-1]
            C_m       = F*(k_eff[0:-1]+k_eff[1:])/C_app[0:-1]    
            
            # Compute diagonal elements for lower boundary point
            A_N       = F*(k_eff[Nx-1]+k_eff[Nx-2])/C_app[Nx-1]        # node id Nx-1 is the last node (N)
            B_N       = F*(k_eff[Nx-2]+3*k_eff[Nx-1])/C_app[Nx-1]
            C_N       = F*(2*k_eff[Nx-1])/C_app[Nx-1]    

            # Compute sparse matrix (scipy format)
            diagonal[1:-1] = 1 + theta*B_m[1:-1]
            lower = -theta*A_m  #1
            upper = -theta*C_m  #1
            
            # Insert boundary conditions (Dirichlet)
            diagonal[0] = 1
            upper[0] = 0
            
            if lb_type == 1:
                # Dirichlet solution for lower boundary
                diagonal[Nx-1] = 1
                lower[-1] = 0
            elif lb_type == 2:
                # First order Neumann solution for lower boundary
                diagonal[Nx-1] = 1
                lower[-1] = -1
            elif lb_type == 3:
                # Second order Neumann solution for lower boundary
                diagonal[Nx-1] = 1 + theta*B_N
                lower[-1] = -theta*(A_N+C_N)  
            else:
                raise ValueError('Unknown lower boundary type')

            U = scipy.sparse.diags(diagonals=[diagonal, lower, upper],
                                   offsets=[0, -1, 1], shape=(Nx, Nx),
                                   format='csr')
                
            # Compute known vector
            b[1:-1] = u_1[1:-1] + (1-theta) * (A_m[0:-1]*u_1[:-2] - B_m[1:-1]*u_1[1:-1] + C_m[1:]*u_1[2:])
            b[0] = ub(solver_time())    # upper boundary conditions
            
            # Add lower boundary condition
            if lb_type == 1:
                # Dirichlet solution for lower boundary
                b[-1] = lb(solver_time())  
            elif lb_type == 2:
                # First order Neumann solution for lower boundary
                b[-1] = dx*grad
            elif lb_type == 3:
                # First order Neumann solution for lower boundary
                b[-1] = u_1[-1] + (1-theta) * ((A_N+C_N)*u_1[-2] - B_N*u_1[-1]) + 2*C_N*dx*grad

            # Solve the system of equations
            u[:] = scipy.sparse.linalg.spsolve(U, b)


# Should not be necessary ???
#            # Add upper boundary condition
#            u[0] = ub(solver_time())    # upper boundary condition
#            
#            # Add lower boundary condition
#            if lb_type == 1:
#                # Dirichlet solution for lower boundary
#                u[-1] = lb(solver_time())  
#            elif lb_type == 2:
#                # First order Neumann solution for lower boundary
#                # u[-1] = dx*grad
#                pass
#            elif lb_type == 3:
#                # First order Neumann solution for lower boundary
#                # u[-1] = u[-1] + (1-theta) * ((A_N+C_N)*u[-2] - B_N*u[-1]) + 2*C_N*dx*grad
#                pass
                
            
            
            # NOW HANDLE CONVERGENCE TESTING
            
            if Layers.parameter_set == 'std':
                unfrw_u = 0.
                convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
            else:
                if Layers.parameter_set  in ['unfrw_thfr', 'unfrw_swi']:
                    phi_u = Layers.f_unfrw_fraction(u, alpha, beta, Tf, Tstar, 1.0)
                    unfrw_u = _n*phi_u    
                    
                    if conv_crit.iteration != 0:       # Always do at least 1 iteration
                        convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
                        
                        if not silent:
                            conv_crit.show()
                    
                elif Layers.parameter_set == 'stefan':
                    phi_u = Layers.f_unfrw_fraction(u, Tf, interval)
                    unfrw_u = _n*phi_u

                    convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
            
            u_bak = u.copy()

            if convergence:
                # break iteration loop since no significant is observed.
                break    

        if not convergence:
            if not silent:
                print("No convergence.", end=' ')
                
            timestep_decreased = solver_time.decrease_step()
            
            # We decrease time step, because we did not see
            # sufficient improvement within the maximum numberof iterations.
            
            # Since solver_time is optimistic, it will automatically
            # increase time step gradually, whenever possible.
            
            # If timestep_decreased is False, we have reached the minimum time step.            
            # do not step forward in time, we need to recalculate for time t+dt
            # where dt is the new decreased time step.
            
            if timestep_decreased:
                if not silent:
                    print("dt reduced.")
            else:
                if not silent: 
                    print("Reduction impossible.", end=' ')

        if convergence or not timestep_decreased:
            # We had convergence, prepare for next time step.             
            if not silent: 
                print("Done! {0:d} iters".format(conv_crit.iteration))
            
            # Perform any defined user action
            if user_action is not None:
                user_action(u, x, solver_time())
            
            # Add data to data file buffer
            datafile.add(solver_time(), ub(solver_time()), u)
            
            # Store some statistics for time step and number of iterations.
            if solver_time.dt not in dt_stats:
                dt_stats[solver_time.dt] = {'N': 1, 'max_iter': conv_crit.iteration}
            else:
                dt_stats[solver_time.dt]['N'] += 1
                if conv_crit.iteration > dt_stats[solver_time.dt]['max_iter']:
                    dt_stats[solver_time.dt]['max_iter'] = conv_crit.iteration
            
            if conv_crit.iteration not in iter_stats:
                iter_stats[conv_crit.iteration] = 1
            else:
                iter_stats[conv_crit.iteration] += 1

            # Prepare for next time step...
            u_1, u = u, u_1

            # Step time forward. Time step will be increased
            # if possible by the optimistic stepping algorithm.
            solver_time.step()

            
    # The model run has completed, now wrap up and print/output results
    
    # Screen output
    try:
        if not silent:
            print('{0:6d}, t: {1:10.2f}, dtf: {2:>7s}   '.format(step, solver_time()/(1*days), solver_time.dt_fraction), end=' ')
        elif show_solver_time:
            print('\b'*11 + '{0:10.2f}'.format(solver_time()/(1*days)), end=' ')
    except:
        pass
    
    # Output statistics to datafile.
    datafile.flush()
    tstop = time.monotonic()    
    datafile.add_comment('cpu: {0:.3f} sec'.format(tstop-tstart))
    
    datafile.add_comment('--- Step-size statistics ----')
    for key in sorted(dt_stats.keys()):
        datafile.add_comment('{1} steps with step-size: {0} s.   Max. {2} iterations at this step size.'.format(key, dt_stats[key]['N'], dt_stats[key]['max_iter']))
    
    datafile.add_comment('--- Iterations statistics ----')    
    for key in sorted(iter_stats.keys()):
        datafile.add_comment('{0} steps with {1} iterations'.format(iter_stats[key], key))        
    
    return u, x, solver_time(), tstop-tstart


def solver_theta_nug(Layers, x, dt, t_end, t0=0, dt_min=360, theta=1, sigma=0,
                     Tinit=lambda x: -2., 
                     ub=lambda x: 10., ub_type=1, lb=lambda x: -2., lb_type=1, grad=0.09, grad_ub=None,
                     user_action=None,
                     conv_crit=None,
                     outfile='model_result.txt',
                     storage=None,
                     outint=1*days,
                     outnodes=None,
                     silent=False,
                     show_solver_time=True,
                     use_sparse=False,
                     L=None,
                     ):
                 
    """Full solver for the model problem using the theta based finite difference 
    approximation. Non Uniform Grid implemented using approximating Lagrange polynomials.
    Vectorized implementation and sparse (n-diagonal) coefficient matrix.
    
    ub_type=1   Dirichlet type upper boundary condition (specified temperature)
    ub_type=2   Neumann type upper boundary condition (specified gradient),
                    implemented as linear gradient between two first nodes.
    ub_type=3   Neumann type upper boundary condition (specified gradient),
                    implemented using the legendre polynomial gradient.
    ub_type=4   Neumann type upper boundary condition (time varying flux),
                    implemented using the legendre polynomial gradient,
                    dividing the current time step flux (retrieved from ub)
                    by the effective thermal conductivity at the current step
                    to calculate the gradient.
    lb_type=1   Dirichlet type lower boundary condition (specified temperature)
    lb_type=2   Neumann type lower boundary condition (specified gradient),
                    implemented as linear gradient between two first nodes.
    lb_type=3   Neumann type lower boundary condition (specified gradient),
                    implemented using the legendre polynomial gradient.
    grad        The gradient [K/m] to use for the lower boundary
    grad_ub     The gradient [K/m] to use for the upper boundary
    ub          Timeseries of upper boundary condition
    lb          Timeseries of upper boundary condition
    outnodes    List of indices of nodes to output in results file
    L           Latent heat of fusion (optional)
    
    storage     Instance of a storage class (FileStorage or MemoryStorage)
                   If None, no data will be stored, only final result returned.
                   If storage is None, and outfile is specified (default)
                       a FileStorage class is instantiated.
                   If storage is FileStorage or MemoryStorage, 
                       outfile is neglected.
                   
    """
    
    def calc_Dp(x):
        """Calculates the square diagonal matrix Dpp used to approximate the first derivative
        of a function with known values at the points given by the vector x. 
        The matrix contains the coefficients of the Lagrange basis polynomials on the diagonals.
        
        Arguments:
        x:  array of node depths
        
        Returns:
        Dp: sparse matrix of Lagrange basis polynomial coefficients
        """
        
        # Since python uses 0-indexing, the variable names and indices
        # in the code differs from those in the theoretical derivation.
        
        N = len(x)
        h = np.squeeze(x[1:]-x[:N-1]).astype(float)  # ensure that the array is one-dimensional and floating point!
        
        # Coefficients of the first row
        a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
        b0 = (h[0]+h[1])/(h[0]*h[1]) 
        c0 = -h[0]/(h[1]*(h[0]+h[1]))
        
        # Coefficients of the inner rows (1 to N-1)
        # The following indicing corresponds to:
        # h[1:]   =  h_{k+1}
        # h[:n-2] =  h_k 
        ak = -h[1:]/(h[:N-2]*(h[:N-2]+h[1:]))
        bk = (h[1:]-h[:N-2])/(h[:N-2]*h[1:])
        ck =  h[:N-2]/(h[1:]*(h[:N-2]+h[1:]))
        
        # Coefficients of the last row
        aN = h[-1]/(h[-2]*(h[-1]+h[-2]))
        bN = -(h[-1]+h[-2])/(h[-1]*h[-2])
        cN = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
        
        # Staack everything up nicely in a sparse matrix
        
        # First create an array of all the values in the matrix
        val  = np.hstack((a0,ak,aN,b0,bk,bN,c0,ck,cN))
        # generate the row indices of each element (from 0 to N-1 three times)
        row = np.tile(np.arange(N),3)
        # generate the column indices, [0,0,1,...,k,...,N-2,N-1,N-1]
        dex = np.hstack((0,np.arange(N-2),N-3))
        col = np.hstack((dex,dex+1,dex+2))
        
        # create and return the sparse matrix
        return scipy.sparse.csr_matrix((val,(row,col)),shape=(N,N))

        
    def calc_Dpp(x):
        """Calculates the square diagonal matrix Dpp used to approximate the second derivative
        of a function with known values at the points given by the vector x. 
        The matrix contains the coefficients of the Lagrange basis polynomials on the diagonals.
        
        Arguments:
        x:  array of node depths
        
        Returns:
        Dpp: sparse matrix of Lagrange basis polynomial coefficients
        
        NOTICE: The approximation at the boundaries (first and last row) are based on one-sided
        Lagrange polynomials, and are only first order accurate, whereas the inner rows use
        centered Lagrange polynomials, and are second order accurate.
        """
        
        # Since python uses 0-indexing, the variable names and indices
        # in the code differs from those in the theoretical derivation.
        
        N = len(x)
        h = np.squeeze(x[1:]-x[:N-1]).astype(float)  # ensure that the array is one-dimensional and floating point!
        
        # Coefficients of the inner rows (1 to N-1)
        # The following indicing corresponds to:
        # h[1:]   =  h_{k+1}
        # h[:n-2] =  h_k 
        ak = 2/(h[:N-2]*(h[1:]+h[:N-2]))
        bk = -2/(h[1:]*h[:N-2])
        ck = 2/(h[1:]*(h[1:]+h[:N-2]))
        
        # Staack everything up nicely in a sparse matrix
        
        # First create an array of all the values in the matrix
        val  = np.hstack((ak[0],ak,ak[-1],bk[0],bk,bk[-1],ck[0],ck,ck[-1]))
        # generate the row indices of each element (from 0 to N-1 three times)
        row = np.tile(np.arange(N),3)
        # generate the column indices, [0,0,1,...,k,...,N-2,N-1,N-1]
        dex = np.hstack((0,np.arange(N-2),N-3))
        col = np.hstack((dex,dex+1,dex+2))
        
        # create and return the sparse matrix
        return scipy.sparse.csr_matrix((val,(row,col)),shape=(N,N))        
    
    
    #raise NotImplementedError('Non-uniform grid version of the solver not yet implemented')
    
    tstart = time.monotonic()
    
    if isinstance(storage, (FileStorage, MemoryStorage, NoStorage)):
        data_storage = storage
    elif isinstance(outfile, str):
        data_storage = FileStorage(outfile, depths=x, interval=outint, buffer_size=30)
    else:
        raise ValueError('Unreckognized data storage option.')
    
    dt_stats = {}
    iter_stats = {}
    
    if conv_crit is None:
        if Layers.parameter_set in ['std','stefan']:
            conv_crit = ConvCritNoIter()
        else:
            conv_crit = ConvCritUnfrw4(threshold=1e-3, max_iter=5)
            
    if L is None:
        L = 334*1e6 # [kJ/kg] => *1000[J/kJ]*1000[kg/m^3] => [J/m^3]
    
    dx = x[1:] - x[0:-1]
    Nx = len(x)   # Number of nodes in grid
    
    u   = np.zeros(Nx)   # solution array at t[tid+1]
    u_1 = np.zeros(Nx)   # solution at t[tid]
    u_bak = np.zeros(Nx)   # solution at t[tid+1], result from previous iteration

    dudT = np.ones(Nx)*-999.   # will hold derivative of unfrozen water
    
    # Representation of right-hand side and unfrozen water contents from two successive calculations
    d = np.zeros(Nx)
    
    # Representation of unfrozen water contents from two successive calculations
    unfrw_u  = np.zeros(Nx)
    unfrw_u1 = np.zeros(Nx)

    # Calculate matrices of lagrange basis polynomial coefficients
    s_Dp = calc_Dp(x) 
    s_Dpp = calc_Dpp(x) 
    if not use_sparse:
        Dp = s_Dp.todense()
        Dpp = s_Dpp.todense()
    
    # indexes into diagonal of the different matrices would be [idx[i], idx[i]]
    idx = np.arange(Nx)
    
    # Get constant layer parameters distributed on the grid
    if Layers.parameter_set == 'unfrw_thfr':
        if not silent: print("Using unfrozen water parameters")
        k_th = Layers.pick_values(x, 'k_th')
        C_th = Layers.pick_values(x, 'C_th')
        k_fr = Layers.pick_values(x, 'k_fr')
        C_fr = Layers.pick_values(x, 'C_fr')
        n = Layers.pick_values(x, 'n')
        
        alpha = Layers.pick_values(x, 'alpha')
        beta = Layers.pick_values(x, 'beta')
            
        Tf = Layers.pick_values(x, 'Tf')
        Tstar = Layers.f_Tstar(Tf, 1.0, alpha, beta)
    elif Layers.parameter_set == 'unfrw_swi':
        if not silent: print("Using unfrozen water parameters")
        k_s = Layers.pick_values(x, 'k_s')
        C_s = Layers.pick_values(x, 'C_s')
        k_w = Layers.pick_values(x, 'k_w')
        C_w = Layers.pick_values(x, 'C_w')
        k_i = Layers.pick_values(x, 'k_i')
        C_i = Layers.pick_values(x, 'C_i')
        n = Layers.pick_values(x, 'n')
        
        alpha = Layers.pick_values(x, 'alpha')
        beta = Layers.pick_values(x, 'beta')
            
        Tf = Layers.pick_values(x, 'Tf')
        Tstar = Layers.f_Tstar(Tf, 1.0, alpha, beta)        
    elif Layers.parameter_set == 'stefan':
        if not silent: print("Using stefan solution parameters")
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
        if not silent: print("Using standard parameters")
        k_eff = Layers.pick_values(x, 'k')
        C_eff = Layers.pick_values(x, 'C')
        k_th = np.nan
        C_th = np.nan
        k_fr = np.nan
        C_fr = np.nan
        phi = 1.
        unfrw_u1 = 0.
    
    # Set initial condition
    for i in range(0,Nx):
        u_1[i] = Tinit(x[i])

    u = copy.copy(u_1)    # initialize u for finite differences
    
    if user_action is not None:
        user_action(u_1, x, t0)

    data_storage.add(t0, ub(t0), u)
       
    # solver_time always holds the time of the time-step
    # that is being calculated.
    solver_time = SolverTime(t0, dt, dt_min=dt_min, optimistic=True)
    step = 0
    solver_time.step()
    
    if silent and show_solver_time:
        if t_end <= 24*3600:
            print('seconds:' + ' '*12, end=' ')
        else:
            print('day:' + ' '*12, end=' ')
    
    # Time loop    
    while solver_time() <= t_end:
        step += 1        
        
        # u_1 holds the temperatures at time step n
        # u   will eventually hold calculated temperatures at step n+1
        
        u_bak = u_1 
        
        try:
            if not silent:
                print('{0:6d}, t: {1:10.2f}, dtf: {2:>7s}   '.format(step, solver_time()/(1*days), solver_time.dt_fraction), end=' ')
            elif show_solver_time:
                if t_end <= 24*3600:
                    print('\b'*11 + '{0:10.2f}'.format(solver_time()), end=' ')
                else:
                    print('\b'*11 + '{0:10.2f}'.format(solver_time()/(1*days)), end=' ')
        except:
            pass
                
        if Layers.parameter_set == 'std':
            #phi = 1.  # for standard solution there is no phase change
            #k_eff = k_th
            #C_eff = C_th
            #unfrw_u1 = 0.
            pass
        else:
            if Layers.parameter_set == 'unfrw_thfr':
                phi = Layers.f_unfrw_fraction(u_1, alpha, beta, Tf, Tstar, 1.0)
                
                k_eff = Layers.f_k_eff(k_fr, k_th, phi)
                C_eff = Layers.f_C_eff(C_fr, C_th, phi)
                unfrw_u1 = n*phi
                
            elif Layers.parameter_set == 'unfrw_swi':
                phi = Layers.f_unfrw_fraction(u_1, alpha, beta, Tf, Tstar, 1.0)
                
                k_eff = Layers.f_k_eff(k_s, k_w, k_i, n, phi)
                C_eff = Layers.f_C_eff(C_s, C_w, C_i, n, phi)
                unfrw_u1 = n*phi
                
            elif Layers.parameter_set == 'stefan':
                phi = Layers.f_unfrw_fraction(u_1, Tf, interval)

                k_eff = Layers.f_k_eff(k_fr, k_th, phi)
                C_eff = Layers.f_C_eff(C_fr, C_th, phi)
                unfrw_u1 = n*phi
        
        #pdb.set_trace()
        
        # Iterative scheme for estimating unfrozen water content
        convergence = False
        conv_crit.reset_iterator()
        for it in conv_crit.iterator():
            
            if Layers.parameter_set in ['unfrw_thfr', 'unfrw_swi']:
                if conv_crit.iteration == 0:
                    # This is first iteration, approximate the latent heat 
                    # component by the analytical derivative
                    C_add_1 = L * n * alpha * beta * np.abs(u_1-Tf)**(-beta-1)
                    C_add = C_add_1
                else:
                    # A previous iteration exist, so an estimate of the
                    # next time step exists. Use that to calculate a finite
                    # difference for the unfrozen water content.

                    # The latent heat contribution is estimated based on
                    # a finite difference, where the temperature change
                    # is non-zero, and the slope of the unfrozen water
                    # content curve where the temperature change is
                    # exactly zero (produces infinite result the finite
                    # difference).
                    
                    # Temporarily ignore division by zero warning.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dudT = (unfrw_u-unfrw_u1)/(u-u_1)
                                        
                    C_add = np.where(np.isfinite(dudT), L*dudT, C_add_1)
            
                # Apparent heat capacity is the heat capacity + the latent heat effect
                C_app = C_eff + C_add
                
            elif Layers.parameter_set == 'stefan':
                C_app = np.where(np.logical_and(np.less(u,Tf),np.
                                                greater_equal(u, Tf-interval)), 
                                 C_eff + L*n/interval, C_eff)
            else:
                C_app = C_eff
            
            if np.any(np.isnan(C_app)) or np.any(np.isinf(C_app)) or np.any(C_app <= 0):
                raise ValueError('Invalid NaN, Inf or zero value encountered in apparent heat capacity!')

            #if True:
            if use_sparse:
                # Calculate matrices in sparse format
                s_diag_k_eff = scipy.sparse.diags(k_eff, 0)
                
                s_Dp_dot_k = s_Dp.dot(k_eff.reshape(-1,1))
                
                s_diag_Dp_dot_k = scipy.sparse.diags(s_Dp_dot_k.reshape(-1), 0)
                
                s_TMP1 = (s_diag_k_eff*s_Dpp + s_diag_Dp_dot_k*s_Dp)
                s_TMP2 = scipy.sparse.csr_matrix((float(solver_time.dt)/C_app,(idx,idx)),shape=(Nx,Nx))
                s_I = scipy.sparse.identity(Nx, dtype='float', format='csr')
                TMP = s_TMP2*s_TMP1
            else:
                # Calculate matrices in dense format
                Dp_dot_k = Dp.dot(k_eff.reshape(-1,1))
                
                diag_Dp_dot_k = np.diag(np.asarray(Dp_dot_k).reshape(-1))
                
                TMP1 = np.einsum('ij,i->ij',Dpp,k_eff)   #   == (Dpp.T dot np.diag(k_eff)).T
                TMP1 += diag_Dp_dot_k*Dp                 #   == diag(Dp*k) dot Dp
                if sigma > 0:
                    TMP1 += np.einsum('ij,i->ij',Dp,sigma*k_eff/x)
                    
                TMP = np.einsum('ij,i->ij',TMP1,float(solver_time.dt)/C_app)
                I = np.identity(Nx)

            # Now calculate G1 and G2 matrices
            G1 = I - theta * TMP
            G2 = I + (1-theta) * TMP
            
            # Calculate the known vector d
            d = G2.dot(u_1)    # u_1 holds temperatures at time step n
            
            # ----------------------------------------------------------
            # TODO:
            # Implement different types of upper boundary condition
            # ----------------------------------------------------------

            # Insert upper boundary condition (Dirichlet)
            
            if ub_type == 1:
                # Dirichlet solution for lower boundary
                G1[0,0] = 1
                G1[0,1:] = 0
                d[0] = ub(solver_time())  
            elif ub_type == 2:
                # First order Neumann solution for lower boundary
                G1[0,0] = -1
                G1[0,1] = 1
                G1[0,2:] = 0
                d[0] = dx[0]*grad_ub
            elif ub_type == 3:
                # Second order Neumann solution for lower boundary
                
                # NO THIS SOLUTION IS STILL ONLY FIRST ORDER ACCURATE!!!
                
                # Use first derivative solution in G1 first row (a'N, b'N, c'N)
                G1[0,:] = Dp[0,:]
                d[0] = grad_ub
            elif ub_type == 4:
                # Neumann solution for upper boundary
                # With flux specified instead of gradient
                
                # Use first derivative solution in G1 first row (a'N, b'N, c'N)
                G1[0,:] = Dp[0,:]
                d[0] = -ub(solver_time())/k_eff[0]
            else:
                raise ValueError('Unknown lower boundary type')
            
            ## Insert upper boundary condition (Dirichlet)
            #G1[0,0] = 1
            #G1[0,1:] = 0
            #
            #d[0] = ub(solver_time())    # upper boundary conditions      
            
            # Insert lower boundary condition
            
            if lb_type == 1:
                # Dirichlet solution for lower boundary
                G1[-1,-1] = 1
                G1[-1,:-1] = 0
                d[-1] = lb(solver_time())  
            elif lb_type == 2:
                # First order Neumann solution for lower boundary
                G1[-1,-2] = -1
                G1[-1,-1] = 1
                G1[-1,:-2] = 0
                d[-1] = dx[-1]*grad
            elif lb_type == 3:
                # Second order Neumann solution for lower boundary
                
                # NO THIS SOLUTION IS STILL ONLY FIRST ORDER ACCURATE!!!
                
                # Use first derivative solution in G1 last row (a'N, b'N, c'N)
                G1[-1,:] = Dp[-1,:]
                d[-1] = grad
            else:
                raise ValueError('Unknown lower boundary type')
            
            
            if use_sparse:
                # Solve system of equations
                u[:] = scipy.sparse.linalg.spsolve(G1, d)
            else:
                u[:] = np.linalg.solve(G1, d)
            
            #pdb.set_trace()

# Should not be necessary            
#            # Add upper boundary condition
#            u[0] = ub(solver_time())    # upper boundary conditions      
#            
#            # Add lower boundary condition
#            
#            if lb_type == 1:
#                # Dirichlet solution for lower boundary
#                G1[-1,-1] = 1
#                G1[-1,:-2] = 0
#                u[-1] = lb(solver_time())  
#            elif lb_type == 2:
#                # First order Neumann solution for lower boundary
#                G1[-1,-1] = 1
#                G1[-1,-2] = -1
#                #u[-1] = dx[-1]*grad
#            elif lb_type == 3:
#                # Second order Neumann solution for lower boundary
#                
#                # NO THIS SOLUTION IS STILL ONLY FIRST ORDER ACCURATE!!!
#                
#                # Use first derivative solution in G1 last row (a'N, b'N, c'N)
#                G1[-1,:] = Dp[-1,:]
#                #u[-1] = grad
#            else:
#                raise ValueError('Unknown lower boundary type')



            
            # NOW HANDLE CONVERGENCE TESTING
            
            if Layers.parameter_set == 'std':
                unfrw_u = 0.
                convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
            else:
                if Layers.parameter_set  in ['unfrw_thfr', 'unfrw_swi']:
                    phi_u = Layers.f_unfrw_fraction(u, alpha, beta, Tf, Tstar, 1.0)
                    unfrw_u = n*phi_u    
                    
                    if conv_crit.iteration != 0:       # Always do at least 1 iteration
                        convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
                        
                        if not silent:
                            conv_crit.show()
                    
                elif Layers.parameter_set == 'stefan':
                    phi_u = Layers.f_unfrw_fraction(u, Tf, interval)
                    unfrw_u = n*phi_u

                    convergence = conv_crit.has_converged(u_bak, u, None, None, solver_time.dt_fraction)
            
            u_bak = u.copy()

            if convergence:
                # break iteration loop since no significant is observed.
                break    
            
        if not convergence:
            if not silent:
                print("No convergence.", end=' ')
                
            timestep_decreased = solver_time.decrease_step()
            
            # We decrease time step, because we did not see
            # sufficient improvement within the maximum number of iterations.
            
            # Since solver_time is optimistic, it will automatically
            # increase time step gradually, whenever possible.
            
            # If timestep_decreased is False, we have reached the minimum time step.            
            # do not step forward in time, we need to recalculate for time t+dt
            # where dt is the new decreased time step.
            
            if timestep_decreased:
                if not silent:
                    print("dt reduced.")
            else:
                if not silent: 
                    print("Reduction impossible.", end=' ')

        if convergence or not timestep_decreased:
            # We had convergence, prepare for next time step.             
            if not silent: 
                print("Done! {0:d} iters".format(conv_crit.iteration))
            
             # Perform any defined user action
            if user_action is not None:
                user_action(u, x, solver_time())
            
             # Add data to data file buffer
            data_storage.add(solver_time(), ub(solver_time()), u)     
            
            # Store some statistics for time step and number of iterations.
            if solver_time.dt not in dt_stats:
                dt_stats[solver_time.dt] = {'N': 1, 'max_iter': conv_crit.iteration}
            else:
                dt_stats[solver_time.dt]['N'] += 1
                if conv_crit.iteration > dt_stats[solver_time.dt]['max_iter']:
                    dt_stats[solver_time.dt]['max_iter'] = conv_crit.iteration
            
            if conv_crit.iteration not in iter_stats:
                iter_stats[conv_crit.iteration] = 1
            else:
                iter_stats[conv_crit.iteration] += 1
            
            # Prepare for next time step...
            u_1, u = u, u_1
            
            # Step time forward. Time step will be increased
            # if possible by the optimistic stepping algorithm.
            solver_time.step()

            
    # The model run has completed, now wrap up and print/output results
    
    # Screen output
    try:
        if not silent:
            print('{0:6d}, t: {1:10.2f}, dtf: {2:>7s}   '.format(step, solver_time()/(1*days), solver_time.dt_fraction), end=' ')
        elif show_solver_time:
            if t_end <= 24*3600:
                print('\b'*11 + '{0:10.2f}'.format(solver_time()), end=' ')
            else:
                print('\b'*11 + '{0:10.2f}'.format(solver_time()/(1*days)), end=' ')
    except:
        pass
    
    # Output statistics to data_storage.
    data_storage.flush()
    tstop = time.monotonic()    
    data_storage.add_comment('cpu: {0:.3f} sec'.format(tstop-tstart))
    
    data_storage.add_comment('--- Step-size statistics ----')
    for key in sorted(dt_stats.keys()):
        data_storage.add_comment('{1} steps with step-size: {0} s.   Max. {2} iterations at this step size.'.format(key, dt_stats[key]['N'], dt_stats[key]['max_iter']))
    
    data_storage.add_comment('--- Iterations statistics ----')    
    for key in sorted(iter_stats.keys()):
        data_storage.add_comment('{0} steps with {1} iterations'.format(iter_stats[key], key))        
    
    return u, x, solver_time(), tstop-tstart
    

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
            
        self.ax1.plot(u, x, 'r-', marker='.', ms=5)
        self.ax1.axvline(x=0, ls='--', color='k')
        self.ax1.set_title('t=%f' % (t/(3600*24.)))
        
        self.ax1.set_ylim([Layers.surface_z-0.1 ,np.min([self.z_max, Layers.z_max])])
        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        
        if self.ax2 is None: 
            self.ax2 = plt.subplot(1, 2, 2)
            
        self.ax2.plot(np.diff(u[:]),np.arange(len(u)-1),'b-')
        self.ax2.axvline(x=0, ls='--', color='k')
        self.ax2.set_ylim([0, len(u)-1])
        self.ax2.invert_yaxis()
        
        if name:
            plt.figure(self.fig).suptitle(name)        
            
        plt.draw()
        plt.show(block=False)


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
    """A class to handle automatic plotting of the model temperature distribution
    during soilfreeze1d calculations.
    """
    
    def __init__(self, Layers, fig=None, ax1=None, z_max=np.inf, 
                 Tmin=None, Tmax=None, figxy=(0,0)):
        self.fignum = fig
        self.fig = None
        self.ax1 = ax1
        self.time_txt = None
        self.line = None
        self.vline = None
        self.z_max = z_max
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.initialized = False
        self.Layers = Layers
        self.figxy = figxy
        self.backend = matplotlib.get_backend()
        if self.backend not in ['TkAgg', 'Qt4Agg', 'Qt5Agg']:
            raise NotImplementedError('The matplotlib {0} backend is not supported. '
                                      'Only Qt and TkAgg backends are supported.')
        
    def initialize(self, u, x, t, name='', figxy=None):
        """Initialize the plot window. The initial condition of the model
        domain is plotted, and preparations are done for fast updating
        of the plot window using blitting.
        
        This initialize method must be invoked before the update method 
        can be invoked.
        """
        plt.close(self.fignum)
        
        self.fig = plt.figure(self.fignum)
        self.fig.clear()

        if figxy is not None:
            self.figxy = figxy

        if figxy is not None:
            self.move_figure(self.figxy[0],self.figxy[1])
            
        if self.ax1 is None:
            self.ax1 = plt.subplot(1, 1, 1)
        
        self.ax1.clear()
            
        self.ax1.set_ylim([self.Layers.surface_z-0.1 ,np.min([self.z_max, self.Layers.z_max])])
        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        self.vline = self.ax1.axvline(x=0, ls='--', color='k')
        
        self.fig.canvas.draw_idle()
        plt.show(block=False)

        self.background = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            
        self.line = self.ax1.plot(u, x, 'r-', marker='.', ms=5)[0]
        self.time_txt = self.ax1.text(0.95, 0.05, 't=%f' % (t/(3600*24.)),
                                      horizontalalignment='right',
                                      verticalalignment='center',
                                      transform=self.ax1.transAxes)
        
        self.ax1.set_ylim([self.Layers.surface_z-0.1 ,np.min([self.z_max, self.Layers.z_max])])
        self.ax1.set_xlim([self.Tmin,self.Tmax])
        self.ax1.invert_yaxis()
        
        if name:
            self.title = self.fig.suptitle(name)        
        
        self.fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.01)
        
        self.initialized = True
        print("VIZUALIZER INITIALIZED...")

    def __call__(self, u, x, t):
        """Method to update the plot of model domain temperature.
        The class .initialize() method must be called before 
        updates to the plot can be accomplished.
        """
        self.line.set_xdata(u)
        self.time_txt.set_text('t=%f' % (t/(3600*24.)))
        
        # restore background
        self.fig.canvas.restore_region(self.background)

        # redraw selected artists
        self.ax1.draw_artist(self.line)
        self.ax1.draw_artist(self.vline)
        self.ax1.draw_artist(self.time_txt)
        
        # redraw the figure canvas
        if self.backend in ['TkAgg']:
            self.fig_redraw = self.fig.canvas.blit(self.ax1.bbox)
        else:
            self.fig_redraw = self.fig.canvas.update()
        
        self.fig.canvas.flush_events()
        
    def update(self, u, x, t):
        self(u, x, t)       

    def add(self, u, x, t, color='b'):
        """Adds an aditional temperature distribution to the figure. 
        This additional distribution will not be redrawn during a
        an update call.
        """
        self.ax1.plot(u, x, color+'-', marker='.', ms=5)
        
        plt.draw()
        plt.show(block=False)
    
    def show(self, block=True):
        plt.show(block=block)
    
    def move_figure(self, x, y):
        """Move figure's upper left corner to pixel (x, y)"""
        mngr = self.fig.canvas.manager            
        
        if self.backend == 'TkAgg':
            mngr.window.wm_geometry("+{0}+{1}".format(x,y))
        elif self.backend == 'WXAgg':
            mngr.window.SetPosition((x,y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            try:
                mngr.window.move(x,y)
            except:
                pass

        
    
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
    

# --------------------------------------------------------------
#
# Calculation of heat capacities
#
# --------------------------------------------------------------        

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
    
    plt.figure()
    plt.plot(T,unfrw,'-k')
    
    plt.show()
    plt.draw()


def plot_trumpet(fname, start=0, end=-1, **kwargs):
    figBG   = 'w'        # the figure background color
    axesBG  = '#ffffff'  # the axies background color
    
    fh = None

    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    elif 'Axes' in kwargs:
        ax = kwargs.pop('Axes')
    elif 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fh = plt.figure(facecolor=figBG)
        ax = plt.axes(axisbg=axesBG)

    if fh is None:
        fh = ax.get_figure()

    data = np.loadtxt(fname, skiprows=1, delimiter=';')
    with open(fname, 'r') as f:
        line = f.readline()
    
    time = data[:,0]/(1*days)
    #surf_T = data[:,1]
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
    
    

def plot_surf(fname=None, data=None, time=None, depths=None, annotations=True, 
              figsize=(15,6), cmap=plt.cm.bwr, node_depths=True, cax=None,
              levels=None, cont_levels=[0], title='', **kwargs):

    figBG   = 'w'        # the figure background color
    axesBG  = '#ffffff'  # the axies background color
    left, width = 0.1, 0.8
    rect1 = [left, 0.2, width, 0.6]     #left, bottom, width, height

    fh = None

    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    elif 'Axes' in kwargs:
        ax = kwargs.pop('Axes')
    elif 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fh = plt.figure(figsize=figsize,facecolor=figBG)
        ax = plt.axes(rect1, axisbg=axesBG)

    if fh is None:
        fh = ax.get_figure()

    if fname is not None:
        data = np.loadtxt(fname, skiprows=1, delimiter=';')
        with open(fname, 'r') as f:
            line = f.readline()
    
        time = data[:,0]/(1*days)
        #surf_T = data[:,1]
        data = data[:,2:]
    
        depths = np.array(line.split(';')[2:], dtype=float)    
    
    # Find the maximum and minimum temperatures, and round up/down
    if levels is None:
        mxn = np.max(np.abs([np.floor(data.min()),
               np.ceil(data.max())]))
        levels = np.arange(-mxn,mxn+1)
        
        if len(levels) < 2:
            levels = np.array([-0.5,0,0.5])

    xx, yy  = np.meshgrid(time, depths)

    cf = ax.contourf(xx, yy, data.T, levels, cmap=cmap)
    
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
    #ax.xaxis_date()

    cbax = plt.colorbar(cf, orientation='horizontal', ax=cax, shrink=1.0)
    cbax.set_label('Temperature [$^\circ$C]')
    #fh.autofmt_xdate()

    if annotations:
        if title == '' and fname is not None:
            fh.suptitle(fname)
        else:
            fh.suptitle(title)

    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Time [days]')

    plt.draw()
    plt.show()

    return ax

    
# --------------------------------------------------------------
#
# Test functionality
#
# --------------------------------------------------------------         

def test_FileStorage():
    data_storage = FileStorage('testdata.txt', depths=[0, 1, 2, 3, 4], 
                           interval=1.0*days, buffer_size=5)
    dt = 0.5*days
    for n in np.arange(20):
        data_storage.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4]))

    data_storage.append = False
    data_storage.initialize()
    dt = 0.5*days
    for n in np.arange(20):
        data_storage.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4])*2)

    
    data_storage.append = True
    dt = 0.5*days
    for n in np.arange(20):
        data_storage.add(dt*n, np.ones(5)*n+np.array([0, 0.1, 0.2, 0.3, 0.4])*3)
    
    
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


def test_FD_unfrw(scheme='theta', Nx=100, version='vectorized', fignum=99, theta=1.0, z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='unfrw')

    Layers.add(Thickness=1,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    Layers.add(Thickness=29,  n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    dt = 1*days          # seconds
    T = 10*365*days  # seconds
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
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
    
    print(u)                 
    print('CPU time:', cpu)    
    
    return u, dt, dx, surf_T(T)



def test_FD_stefan(scheme='theta', Nx=100, fignum=99, theta=1., z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='stefan', interval=1)

    Layers.add(Thickness=30,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Test 1')    
    #Layers.add(Thickness=28, n=0.3,  C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=-0.0001, soil_type='Test 2')
    
    dt = 0.2*days          # seconds
    T = (5*155+1)*0.2*days  # seconds
    

    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
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
    
    print(u)                 
    print('CPU time:', cpu)    
    
    return u, dt, dx, surf_T(T)


def test_FD_stefan_grad(scheme='theta', Nx=100, fignum=99, theta=1., z_max=np.inf, animate=True):
    pylab.ion()
    Layers = LayeredModel(type='stefan', interval=1)

    Layers.add(Thickness=30,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Test 1')    
    
    dt = 0.2*days          # seconds
    T = (5*155+1)*0.2*days  # seconds
    

    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
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
    
    print(u)                 
    print('CPU time:', cpu)    
    
    return u, dt, dx, surf_T(T)
        
        
if __name__ == '__main__':
    #test_FD_stefan_gra
    pass