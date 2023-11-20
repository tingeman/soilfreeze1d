# -*- coding: utf-8 -*-
"""
@author: thin

# ===============================================================================
# 
# This model script illustrates the use of the soilfreeze1d module for 
# calculating the thermal regime in a 1D model with one layer using the stefan 
# solution (phase change occurs linearly over a specified temperature interval).
# 
# The model domain has an initial termperature of -2C at all nodes, and the 
# model is forced by a harmonic temperature variation at the upper boundary. 
# The lower boundary has a specified constant gradient of 0.02 C/m.
# Results will be written to the file model1_results.txt at a daily interval.
# 
# ===============================================================================

REWRITE!!!!


"""

# import standard pythom modules
import numpy as np
import pdb

# import own modules
import soilfreeze1d   # import the module containing the Finite Difference 
                      # modelling code and supporting functions.

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define variables and constants
sec = 1
hour = 3600*sec
days = 24*3600  # Define a constant for conversion from days to seconds


def plot_trumpet(fname, year, days_in_year=365):
    data = pd.read_csv(fname, sep=';', skipinitialspace=True, comment='#')
    cols = [s.strip() for s in data.columns]
    data.columns = cols
    depths = np.array([float(s) for s in cols[2:]])

    plt.figure()
    days_in_month = np.round(days_in_year/12)
    days = np.arange(12)*days_in_month
    
    for day in days:
        plt.plot(data[data['Time[seconds]']>=3600.*24*(day+year*days_in_year)].iloc[0,2:], -depths, '-', color='0.5', lw=1)

    plt.grid(True)
    plt.xlabel('Temperature [$^\circ C$]')
    plt.ylabel('Depth [$m$]')
    ax = plt.gca()
    
    data_1yr = data[(data['Time[seconds]']>=3600.*24*(0+year*days_in_year)) & (data['Time[seconds]']<=3600.*24*(days_in_year+year*days_in_year))]
    stats = data_1yr.describe().T

    #pdb.set_trace()

    ax.plot(stats['min'][2:], -depths, '-b', lw=2, label='Min T')
    ax.plot(stats['max'][2:], -depths, '-r', lw=2, label='Max T')
    ax.plot(stats['mean'][2:], -depths, '-g', lw=2, label='Mean T')
    ax.legend()




# Define any supporting functions

if __name__ == '__main__':
    # Run this code only if the module is executed using th %run command
    # from IPython console, or using "python.exe model1.py" from command prompt.

    if False:
        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model()
        Layers.add(Thickness=1,  C=2.5E6, k=2, soil_type='undefined material')    

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 100            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 20*days    # The total calculation period

        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.ConstantTemperature(T=10)    
        lb_T = soilfreeze1d.ConstantTemperature(T=10)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*10    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = True  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = 4        # The minimum value on the temperature axis
        Tmax = 12       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_1a.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show()
        
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.ConstantTemperature(T=5)    
        lb_T = soilfreeze1d.ConstantTemperature(T=10)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*10    # 10C at all depths
        
        # Set up result output 
        outfile = 'test_bc_results_1b.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file   

        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)

        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show()


        T = 100*days    # The total calculation period

        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.ConstantTemperature(T=5)    
        lb_T = soilfreeze1d.ConstantTemperature(T=10)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*10    # 10C at all depths
        
        # Set up result output 
        outfile = 'test_bc_results_1c.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file   

        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)

        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=2, grad=1.0, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show()    
        

        T = 100*days    # The total calculation period

        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.ConstantTemperature(T=5)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*10    # 10C at all depths
        
        # Set up result output 
        outfile = 'test_bc_results_1d.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file   

        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)

        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=2, grad=0.0, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show() 
        
        
    if False:
    
        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model()
        Layers.add(Thickness=10,  C=2.5E6, k=2, soil_type='undefined material')    

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 100            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 3*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)    
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = True  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_2a.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show()
        
 
        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model()
        Layers.add(Thickness=10,  C=2.5E6, k=2, soil_type='undefined material')    

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 100            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 3*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)   
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = True  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_2b.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=3, grad=0.02, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
        plot_solution.show() 
        
    if True:
        anim = False
    
        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=10,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 50            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3a.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=3, grad=0.02, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 

        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=10,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 50            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3b.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 

        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=30,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 150            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3c.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=3, grad=0.02, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 

        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=30,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 150            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3d.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=3, grad=0.00, 
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 
        
        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=30,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 150            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-2)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-2    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3e.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T,  
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 


        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=30,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 150            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-1)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-1    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3f.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T,  
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 


        # Define the model layers and properties
        Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
        #Layers.add(Thickness=10,  n=0.4, C_th=2.5E6, C_fr=2.5E6, k_th=2, k_fr=2, interval=1.0, Tf=0.0, soil_type='Soil 1')     
        Layers.add(Thickness=30,  n=0.4, C_th=2.8E6, C_fr=2.1E6, k_th=1.4, k_fr=2.3, interval=1.0, Tf=0.0, soil_type='Soil 1')     

        # Thickness:    Thickness of the layer [m]
        # C:            Heat capacity  [J/(m^3*C)]
        # k:            Thermal conductivity  [W/(m*C)]
        # soil_type:    Character string for identification, not used by model
        
        # Define model domain properties
        Nx = 150            # The number of nodes in the model domain is Nx
        dt = 1*days         # The calculation time step
        T = 11*360*days    # The total calculation period
        
        # We use 360 days to make plotting easy
        
        # Define the forcing upper boundary temperature
        ub_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days, period=360)  
        lb_T = soilfreeze1d.ConstantTemperature(T=-4)    

        # Define initial temperatures:
        initialTemperature = lambda z: np.ones_like(z)*-4    # 10C at all depths
        
        # Set up plotting
        fignum  = 99    # Plot results in figure number 99    
        animate = anim  # Plot result of each model time step    
                        # If set to False, only first and last
                        # time step will be plotted.
        
        Tmin = -11        # The minimum value on the temperature axis
        Tmax = 10       # The maximum value on the temperature axis
        z_max = 10      # Maximum plot depth on the z-axis    
        
        # Set up result output 
        outfile = 'test_bc_results_3g.txt' # Name of the result file
        outint = 1*hour  # The interval at which results will be written to the file    
        
        x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # equidistant mesh points in space
        dx = x[1] - x[0]
        
        # Plot initial condition
        plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
        plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)
        
        # Switch animation on or off
        if animate:
            user_action = plot_solution
        else:
            user_action = None
        
        # Call Finite Difference engine    
        u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub_type=1, ub=ub_T, 
                                                 lb_type=1, lb=lb_T,  
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
        
        # plot final result
        plot_solution.update(u, x, t)
        
        print(' ')
        print(' ')
        print('CPU time:', cpu)
        print(' ')
        print('Close figure to return focus to the terminal...')
#        plot_solution.show() 
        
    if True:
        plot_trumpet('test_bc_results_3a.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3b.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3c.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3d.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3e.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3f.txt', year=10, days_in_year=360)
        plot_trumpet('test_bc_results_3g.txt', year=10, days_in_year=360)