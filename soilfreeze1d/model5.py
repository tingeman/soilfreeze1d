# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 01:06:32 2016

@author: thin
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
minutes = 1*60

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
    Layers = soilfreeze1d.new_layered_model(type='unfrw_thfr')
    Layers.add(Thickness=3,  n=0.6, C_th=2.5E6, C_fr=2.5E6, k_th=1.1, k_fr=1.1, alpha=0.19, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    Layers.add(Thickness=28,  n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.1, k_fr=1.1, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    
    # Thickness:    Thickness of the layer [m]
    # n:            Porosity [-]   Soil is considered fully saturated
    # C_th:         Heat capacity, thawed state [J/(m^3*C)]
    # C_fr:         Heat capacity, frozen state [J/(m^3*C)]    
    # k_th:         Thermal conductivity, thawed state [W/(m*C)]
    # k_fr:         Thermal conductivity, frozen state [W/(m*C)]
    # interval:     Phase change interval [C]   default = 1C
    # Tf:           Freezing point [C]  default = 0C
    # soil_type:    Character string for identification, not used by model
    
    
    # Define model domain properties
    Nx = 400        # The number of nodes in the model domain is Nx+1
    T = 2*365*days    # The total calculation period

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)    

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
    
    
    runs = [dict(scheme='BE', dt=360,      dt_min=1,   theta=1,   outfile='model5_6m-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt=1800,     dt_min=1,   theta=1,   outfile='model5_30m-1s_BE.txt', cpu=None),
            dict(scheme='BE', dt= 1*hours, dt_min=1,   theta=1,   outfile='model5_1h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 2*hours, dt_min=1,   theta=1,   outfile='model5_2h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 3*hours, dt_min=1,   theta=1,   outfile='model5_3h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 4*hours, dt_min=1,   theta=1,   outfile='model5_4h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 6*hours, dt_min=1,   theta=1,   outfile='model5_6h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 8*hours, dt_min=1,   theta=1,   outfile='model5_8h-1s_BE.txt',  cpu=None),
            dict(scheme='BE', dt=12*hours, dt_min=1,   theta=1,   outfile='model5_12h-1s_BE.txt', cpu=None),
            dict(scheme='BE', dt=16*hours, dt_min=1,   theta=1,   outfile='model5_16h-1s_BE.txt', cpu=None),
            dict(scheme='BE', dt=20*hours, dt_min=1,   theta=1,   outfile='model5_20h-1s_BE.txt', cpu=None),
            dict(scheme='BE', dt=24*hours, dt_min=1,   theta=1,   outfile='model5_24h-1s_BE.txt', cpu=None),
            dict(scheme='CN', dt=360,      dt_min=1,   theta=0.5, outfile='model5_6m-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt=1800,     dt_min=1,   theta=0.5, outfile='model5_30m-1s_CN.txt', cpu=None),
            dict(scheme='CN', dt= 1*hours, dt_min=1,   theta=0.5, outfile='model5_1h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 2*hours, dt_min=1,   theta=0.5, outfile='model5_2h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 3*hours, dt_min=1,   theta=0.5, outfile='model5_3h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 4*hours, dt_min=1,   theta=0.5, outfile='model5_4h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 6*hours, dt_min=1,   theta=0.5, outfile='model5_6h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 8*hours, dt_min=1,   theta=0.5, outfile='model5_8h-1s_CN.txt',  cpu=None),
            dict(scheme='CN', dt=12*hours, dt_min=1,   theta=0.5, outfile='model5_12h-1s_CN.txt', cpu=None),
            dict(scheme='CN', dt=16*hours, dt_min=1,   theta=0.5, outfile='model5_16h-1s_CN.txt', cpu=None),
            dict(scheme='CN', dt=20*hours, dt_min=1,   theta=0.5, outfile='model5_20h-1s_CN.txt', cpu=None),
            dict(scheme='CN', dt=24*hours, dt_min=1,   theta=0.5, outfile='model5_24h-1s_CN.txt', cpu=None),
            dict(scheme='BE', dt= 1*hours, dt_min=360, theta=1,   outfile='model5_1h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 2*hours, dt_min=360, theta=1,   outfile='model5_2h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 3*hours, dt_min=360, theta=1,   outfile='model5_3h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 4*hours, dt_min=360, theta=1,   outfile='model5_4h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 6*hours, dt_min=360, theta=1,   outfile='model5_6h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt= 8*hours, dt_min=360, theta=1,   outfile='model5_8h-6m_BE.txt',  cpu=None),
            dict(scheme='BE', dt=12*hours, dt_min=360, theta=1,   outfile='model5_12h-6m_BE.txt', cpu=None),
            dict(scheme='BE', dt=16*hours, dt_min=360, theta=1,   outfile='model5_16h-6m_BE.txt', cpu=None),
            dict(scheme='BE', dt=20*hours, dt_min=360, theta=1,   outfile='model5_20h-6m_BE.txt', cpu=None),
            dict(scheme='BE', dt=24*hours, dt_min=360, theta=1,   outfile='model5_24h-6m_BE.txt', cpu=None),
            dict(scheme='CN', dt= 1*hours, dt_min=360, theta=0.5, outfile='model5_1h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 2*hours, dt_min=360, theta=0.5, outfile='model5_2h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 3*hours, dt_min=360, theta=0.5, outfile='model5_3h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 4*hours, dt_min=360, theta=0.5, outfile='model5_4h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 6*hours, dt_min=360, theta=0.5, outfile='model5_6h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt= 8*hours, dt_min=360, theta=0.5, outfile='model5_8h-6m_CN.txt',  cpu=None),
            dict(scheme='CN', dt=12*hours, dt_min=360, theta=0.5, outfile='model5_12h-6m_CN.txt', cpu=None),
            dict(scheme='CN', dt=16*hours, dt_min=360, theta=0.5, outfile='model5_16h-6m_CN.txt', cpu=None),
            dict(scheme='CN', dt=20*hours, dt_min=360, theta=0.5, outfile='model5_20h-6m_CN.txt', cpu=None),
            dict(scheme='CN', dt=24*hours, dt_min=360, theta=0.5, outfile='model5_24h-6m_CN.txt', cpu=None)]
            
    if True:
        for rid, run in enumerate(runs):
            # Call Finite Difference engine    
            print "Running {0}   ".format(run['outfile']),
            sys.stdout.flush()
            u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, run['dt'], T, dt_min=run['dt_min'], 
                                                     theta=run['theta'],
                                                     Tinit=initialTemperature, 
                                                     ub=surf_T, lb_type=2, grad=grad,
                                                     user_action=user_action,
                                                     outfile=run['outfile'],
                                                     outint=outint,
                                                     silent=True)
            
            runs[rid][cpu] = cpu
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
            #data[run['outfile']] = np.loadtxt(run['outfile'],  skiprows=1, delimiter=';', comment='#')
            
        params = pd.DataFrame(runs)
        params['dT_max'] = 0.
        params['dT_min'] = 0.
        params['dT_avg'] = 0.
        
        ref = 'model5_1h-1s_CN.txt'
        refdat = data[ref]
        
        for did, dat_key in enumerate(data.keys()):
            if dat_key != ref:
                params.ix[params['outfile']==dat_key, 'dT_max'] = (refdat-data[dat_key]).max().max()
                params.ix[params['outfile']==dat_key, 'dT_min'] = (refdat-data[dat_key]).min().min()
                params.ix[params['outfile']==dat_key, 'dT_avg'] = (refdat-data[dat_key]).mean().mean()
                
                #soilfreeze1d.plot_surf(data=refdat-data[dat_key], time=times, depths=x, cont_levels=[-0.05, 0.05])
        
        fig1 = plt.figure(); ax1 = plt.axes()
        fig2 = plt.figure(); ax2 = plt.axes()
        fig3 = plt.figure(); ax3 = plt.axes()
        fig4 = plt.figure(); ax4 = plt.axes()
        
        m = ['s','d']
        c = ['b','g']
        
        lb_scheme = ['Backward Euler', 'Cranck-Nicholson']
        lb_dtmin  = ['dt_min=1s', 'dt_min=360s']
        
        for sid,scheme in enumerate(['BE', 'CN']):
            for did,dt_min in enumerate([1,360]):
                xdat = params[np.logical_and(params['scheme'] == scheme, params['dt_min'] == dt_min)]['dt']/(1*minutes)
                ydat = params[np.logical_and(params['scheme'] == scheme, params['dt_min'] == dt_min)]['dT_max']
                ax1.plot(xdat, ydat, marker=m[sid], color=c[did], label=lb_scheme[sid]+', '+lb_dtmin[did])
                
                ydat = params[np.logical_and(params['scheme'] == scheme, params['dt_min'] == dt_min)]['dT_min']
                ax2.plot(xdat, ydat, marker=m[sid], color=c[did], label=lb_scheme[sid]+', '+lb_dtmin[did])

                ydat = params[np.logical_and(params['scheme'] == scheme, params['dt_min'] == dt_min)]['dT_avg']
                ax3.plot(xdat, ydat, marker=m[sid], color=c[did], label=lb_scheme[sid]+', '+lb_dtmin[did])

                ydat = params[np.logical_and(params['scheme'] == scheme, params['dt_min'] == dt_min)]['cpu']
                ax4.loglog(xdat, ydat, marker=m[sid], color=c[did], label=lb_scheme[sid]+', '+lb_dtmin[did])                
                
        ax1.set_title('Referece: Cranck-Nicholson, dt_max=1h, dt_min=1s')        
        ax1.set_xlabel('Maximum time step [min]')
        ax1.set_ylabel('Maximum temperature difference [C]')
        ax1.legend(loc=0)
        
        ax2.set_title('Referece: Cranck-Nicholson, dt_max=1h, dt_min=1s')        
        ax2.set_xlabel('Mmaximum time step [min]')
        ax2.set_ylabel('Minimum temperature difference [C]')
        ax2.legend(loc=0)
        
        ax3.set_title('Referece: Cranck-Nicholson, dt_max=1h, dt_min=1s')        
        ax3.set_xlabel('Maximum time step [min]')
        ax3.set_ylabel('Average temperature difference [C]')
        ax3.legend(loc=0)
              
        ax4.set_xlabel('Maximum time step [min]')
        ax4.set_ylabel('Calculation time [s]')
        ax4.legend(loc=0)
        
        plt.show()
        1/0

                
#                soilfreeze1d.plot_surf(data=dat1[1][:,2:]-dat2[1][:,2:],  
#                                       time=dat1[1][:,0], depths=x, 
#                                       cont_levels=[-0.05, 0.05],
#                                       title="{0} - {1}".format(dat1[0], dat2[0]))
        
    if False:
        data_1h  = np.loadtxt('model5_1h.txt',  skiprows=1, delimiter=';')
        data_3h  = np.loadtxt('model5_3h.txt',  skiprows=1, delimiter=';')
        data_6h  = np.loadtxt('model5_6h.txt',  skiprows=1, delimiter=';')
        data_12h = np.loadtxt('model5_12h.txt', skiprows=1, delimiter=';')
        data_24h = np.loadtxt('model5_24h.txt', skiprows=1, delimiter=';')
        data_24h2 = np.loadtxt('model5_24h2.txt', skiprows=1, delimiter=';')

        print "diff 1h- 3h  = {0}".format(np.max( data_1h[:,2:]-data_3h[:,2:]))        
        print "diff 1h- 6h  = {0}".format(np.max( data_1h[:,2:]-data_6h[:,2:]))        
        print "diff 1h-12h  = {0}".format(np.max( data_1h[:,2:]-data_12h[:,2:]))        
        print "diff 1h-24h  = {0}".format(np.max( data_1h[:,2:]-data_24h[:,2:]))        
        print "diff 1h-24h2 = {0}".format(np.max( data_1h[:,2:]-data_24h2[:,2:]))        

        print "diff  3h- 6h  = {0}".format(np.max( data_3h[:,2:]-data_6h[:,2:]))        
        print "diff  3h-12h  = {0}".format(np.max( data_3h[:,2:]-data_12h[:,2:]))        
        print "diff  3h-24h  = {0}".format(np.max( data_3h[:,2:]-data_24h[:,2:]))        
        print "diff  6h-12h  = {0}".format(np.max( data_6h[:,2:]-data_12h[:,2:]))        
        print "diff  6h-24h  = {0}".format(np.max( data_6h[:,2:]-data_24h[:,2:]))
        print "diff 12h-24h  = {0}".format(np.max(data_12h[:,2:]-data_24h[:,2:]))

        print "avg  3h- 6h  = {0}".format(np.mean( data_3h[:,2:]-data_6h[:,2:]))        
        print "avg  3h-12h  = {0}".format(np.mean( data_3h[:,2:]-data_12h[:,2:]))        
        print "avg  3h-24h  = {0}".format(np.mean( data_3h[:,2:]-data_24h[:,2:]))        
        print "avg  6h-12h  = {0}".format(np.mean( data_6h[:,2:]-data_12h[:,2:]))        
        print "avg  6h-24h  = {0}".format(np.mean( data_6h[:,2:]-data_24h[:,2:]))
        print "avg 12h-24h  = {0}".format(np.mean(data_12h[:,2:]-data_24h[:,2:]))
        
        
        soilfreeze1d.plot_surf(data=data_1h[:,2:]-data_3h[:,2:],  time=data_1h[:,0], depths=x, cont_levels=[-0.05, 0.05])
        soilfreeze1d.plot_surf(data=data_1h[:,2:]-data_6h[:,2:],  time=data_1h[:,0], depths=x, cont_levels=[-0.05, 0.05])
        soilfreeze1d.plot_surf(data=data_1h[:,2:]-data_12h[:,2:], time=data_1h[:,0], depths=x, cont_levels=[-0.05, 0.05])    
        soilfreeze1d.plot_surf(data=data_1h[:,2:]-data_24h2[:,2:], time=data_1h[:,0], depths=x, cont_levels=[-0.05, 0.05])    
        soilfreeze1d.plot_surf(data=data_24h[:,2:]-data_24h2[:,2:], time=data_1h[:,0], depths=x, cont_levels=[-0.05, 0.05])    
        
    #    soilfreeze1d.plot_surf(data=data_3h[:,2:]- data_6h[:,2:],  time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])
    #    soilfreeze1d.plot_surf(data=data_3h[:,2:]- data_12h[:,2:], time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])    
    #    soilfreeze1d.plot_surf(data=data_3h[:,2:]- data_24h[:,2:], time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])    
    #    soilfreeze1d.plot_surf(data=data_6h[:,2:]- data_12h[:,2:], time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])
    #    soilfreeze1d.plot_surf(data=data_6h[:,2:]- data_24h[:,2:], time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])
    #    soilfreeze1d.plot_surf(data=data_12h[:,2:]-data_24h[:,2:], time=data_3h[:,0], depths=x, cont_levels=[-0.05, 0.05])

    
    
