# -*- coding: utf-8 -*-
"""
@author: thin

===============================================================================

This model script compares the use of a 1st order and 2nd order Neumann
solution for the lower boundary condition (gradient of temperature).

===============================================================================

"""

# import standard pythom modules
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import own modules
import soilfreeze1d   # import the module containing the Finite Difference 
                      # modelling code and supporting functions.

# Define variables and constants
days = 24*3600  # Define a constant for conversion from days to seconds
hours = 1*3600

# Define any supporting functions

def initialTemperature(z):
    """A function used to set up the initial condition in the model domain.
    The finite difference algorithm needs a starting temperature at each node,
    and will call this this function with an array of node depths (z) expecting
    in return an array of equal shape (length) containing the initial temperature
    at each node location.
    
    This function will return a constant temperature, independent of the node
    location.
    """
    constant_temperature = -2.
    return np.ones_like(z)*constant_temperature



if __name__ == '__main__':
    # Run this code only if the module is executed using th %run command
    # from IPython console, or using "python.exe model1.py" from command prompt.

    # Define the model layers and properties
    Layers = soilfreeze1d.LayeredModel(type='std')
    Layers.add(Thickness=30, C=2.5E6, k=1.1, soil_type='Fairbanks Silt')    
    
    # Thickness:    Thickness of the layer [m]
    # C:            Heat capacity [J/(m^3*C)]
    # k:            Thermal conductivity [W/(m*C)]
    # soil_type:    Character string for identification, not used by model

    
    # Define model domain properties
    Nx = 400        # The number of nodes in the model domain is Nx+1
    T = 10*365*days    # The total calculation period

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.ConstantTemperature(T=-2.)    

    # Define the geothermal gradient (lower boundary)    
    grad=0.08333     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99    
    animate = False  # Plot result of each model time step    
                    # If set to False, only first and last
                    # time step will be plotted.
    
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9      # The maximum value on the temperature axis
    z_max = 10      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outint = 1*days  # The interval at which results will be written to the file    
    
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    # Switch animationoff
    user_action = None

    outint = 1*days  # The interval at which results will be written to the file   

    cpu_list = []
    
    
    runs = [dict(scheme='BE', dt=2*hours, dt_min=1, theta=1,   outfile='model6_2h-1s_BE_lb2.txt',  lb_type=2),
            dict(scheme='BE', dt=2*hours, dt_min=1, theta=1,   outfile='model6_2h-1s_BE_lb3.txt',  lb_type=3),
            dict(scheme='CN', dt=2*hours, dt_min=1, theta=0.5, outfile='model6_2h-1s_CN_lb2.txt',  lb_type=2),
            dict(scheme='CN', dt=2*hours, dt_min=1, theta=0.5, outfile='model6_2h-1s_CN_lb3.txt',  lb_type=3),]
            
    if True:
        for rid, run in enumerate(runs):
            # Call Finite Difference engine    
            print "Running {0}   ".format(run['outfile']),
            sys.stdout.flush()
            u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, run['dt'], T, dt_min=run['dt_min'], 
                                                     theta=run['theta'],
                                                     Tinit=initialTemperature, 
                                                     ub=surf_T, lb_type=run['lb_type'], grad=grad,
                                                     user_action=user_action,
                                                     outfile=run['outfile'],
                                                     outint=outint,
                                                     silent=True)
            
            runs[rid]['cpu'] = cpu
            cpu_list.append([run['outfile'],cpu])
            print cpu
        

    if True:        
        data = {}
        for rid,run in enumerate(runs):
            dat = pd.read_csv(run['outfile'], delimiter = ";")
            runs[rid]['cpu'] = float(dat.icol(0).irow(-1).split(':')[-1])
            times = np.array(list(dat.ix[1:dat.index.max()-1,0].values), dtype='f8')
            data[run['outfile']] = dat.ix[1:dat.index.max()-1,2:]
            # skip first and last rows, as well as index columns
            
        params = pd.DataFrame(runs)
        params['dT_max'] = 0.
        params['dT_min'] = 0.
        params['dT_avg'] = 0.
        
        ref = 'model6_2h-1s_CN_lb3.txt'
        refdat = data[ref]

#        soilfreeze1d.plot_surf(data=refdat, time=times, 
#                               depths=x, levels=np.linspace(-5,5,21), 
#                               cont_levels=[0.],
#                               title="{0}".format(ref))

            
        
        for did, dat_key in enumerate(data.keys()):
            if dat_key != ref:
                params.ix[params['outfile']==dat_key, 'dT_max'] = (refdat-data[dat_key]).max().max()
                params.ix[params['outfile']==dat_key, 'dT_min'] = (refdat-data[dat_key]).min().min()
                params.ix[params['outfile']==dat_key, 'dT_avg'] = (refdat-data[dat_key]).mean().mean()
                
                fig = plt.figure()
                plt.plot(data[dat_key].irow(-1).values, -x, '.-r')
                plt.gca().axvline(x=0)
                fig.suptitle(dat_key)
                
                #soilfreeze1d.plot_surf(data=data[dat_key], time=times, 
                #                       depths=x, levels=np.linspace(-5,5,21),
                #                       cont_levels=[0.],
                #                       title="{0}".format(dat_key))
                                       
                soilfreeze1d.plot_surf(data=refdat-data[dat_key], time=times, 
                                       depths=x, levels=np.linspace(-1,1,21), 
                                       cont_levels=[-0.05, 0.05],
                                       title="{0} - {1}".format(ref, dat_key))
        
        
        if False:
            fig1 = plt.figure(); ax1 = plt.axes()
            fig2 = plt.figure(); ax2 = plt.axes()
            fig3 = plt.figure(); ax3 = plt.axes()
            fig4 = plt.figure(); ax4 = plt.axes()
            
            m = ['s','d']
            c = ['b','g']
            
            for sid,scheme in enumerate(['BE', 'CN']):   # shapes:   squares, diamonds
                for tid,lb_type in enumerate([2,3]):     # colors:   bblue,   green
                    xdat = params[np.logical_and(params['scheme'] == scheme, params['lb_type'] == lb_type)]['dt']/(1*hours)
                    ydat = params[np.logical_and(params['scheme'] == scheme, params['lb_type'] == lb_type)]['dT_max']
                    ax1.plot(xdat, ydat, marker=m[sid], color=c[tid])
                    
                    ydat = params[np.logical_and(params['scheme'] == scheme, params['lb_type'] == lb_type)]['dT_min']
                    ax2.plot(xdat, ydat, marker=m[sid], color=c[tid])
    
                    ydat = params[np.logical_and(params['scheme'] == scheme, params['lb_type'] == lb_type)]['dT_avg']
                    ax3.plot(xdat, ydat, marker=m[sid], color=c[tid])
    
                    ydat = params[np.logical_and(params['scheme'] == scheme, params['lb_type'] == lb_type)]['cpu']
                    ax4.loglog(xdat, ydat, marker=m[sid], color=c[tid])                
                    
            fig1.suptitle('Maximum difference')
            fig2.suptitle('Minimum difference')
            fig3.suptitle('Average difference')
            fig4.suptitle('Calculation time')
                
        plt.show()

    
