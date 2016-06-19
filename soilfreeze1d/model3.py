# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 01:06:32 2016

@author: thin
"""

# import standard pythom modules
import pdb
import time
import numpy as np


# import own modules
import soilfreeze1d   # import the module containing the Finite Difference 
                      # modelling code and supporting functions.

# Define variables and constants
days = 24*3600  # Define a constant for conversion from days to seconds



if __name__ == '__main__':
    # Run this code only if the module is executed using th %run command
    # from IPython console, or using "python.exe model1.py" from command prompt.

    # Define the model layers and properties
    Layers = soilfreeze1d.LayeredModel(type='stefan')
    Layers.add(Thickness=30,  n=0.10, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Soil 1')    
    
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
    Nx = 200        # The number of nodes in the model domain is Nx+1
    dt = 1*days   # The calculation time step


    # Define initial condition (temperatures in domain)
    initialTemperature = soilfreeze1d.DistanceInterpolator(depths=[0,15,30], temperatures=[-2,-2,-1])
    # Here we defined a piece-wise linear temperature profile with temperature
    # -2C at top of model (0 m) to center of model (15 m), increasing to  -1C 
    # at bottom of model (30 m).
    
    # This method could be used to feed the result of a previous calculation
    # as initial condition for a new calculation.
    # Keep in mind also to ensure that the upper boundary temperature is 
    # consisten (continuous) between the two model runs...
    

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)    

    # Define the geothermal gradient (lower boundary)    
    grad=0.08333     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99        
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9       # The maximum value on the temperature axis
    z_max = 30      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outfile = 'model3_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    # Plot initial condition
    plot_solution = soilfreeze1d.Visualizer_T(Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., Layers, name=outfile)

    #==========================================================================
    # First we spin up the model by running it for 10 years 
    # ... with no plotting since that will be much quicker
    #==========================================================================
                       
    animate = True
    t0 = 0*days         # Start time of the model    
    T = 10*365*days+t0  # The total calculation period
    
    # Switch animation on or off
    if animate:
        user_action = plot_solution
    else:
        user_action = None
    
    # Call Finite Difference engine    
    u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, t0=t0,
                                             Tinit=initialTemperature, 
                                             ub=surf_T, lb_type=2, grad=grad,
                                             user_action=user_action,
                                             outfile=outfile,
                                             outint=outint)
    
    # plot final result
    plot_solution.update(u, x, t)
    
    # Print the time spent
    print 'Spin-up run, CPU time: {0:.3f} s'.format(cpu)

    print "Waiting 5 seconds (inspect model result)"
    time.sleep(5)
    
    #==========================================================================
    # ...Then we use the output to set up the next model run
    # Which now has an extra layer added.
    #==========================================================================
 
    # Define the model layers and properties
    Layers = soilfreeze1d.LayeredModel(type='stefan', surface_z=-5.)
    Layers.add(Thickness=5,   n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Top Soil')
    Layers.add(Thickness=30,  n=0.10, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Soil 1')

    # Define initial condition (temperatures in domain)
    initialTemperature = soilfreeze1d.DistanceInterpolator(depths=x, temperatures=u)
    
    # update the grid points
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    # We should have also updated the Nx value proportionally to the 
    # expansion of the domain... Instead we get a less dense
    # node distribution.


    animate = True  # Now do the animation  
    dt = 0.5*days   # The calculation time step
    t0 = t[-1]      # Set start time to the end time of previous run
    T = 365*days+t0 # The total calculation period is one year

    # Set up result output 
    outfile = 'model3b_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    # Reinitialize plotting
    plot_solution = soilfreeze1d.Visualizer_T(Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, t0, Layers, name=outfile)
    
    # Switch animation on or off
    if animate:
        user_action = plot_solution
    else:
        user_action = None
    
    print "Waiting 5 seconds before commencing new run (inspect initial model)"
    time.sleep(5)
    
    # Call Finite Difference engine    
    u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, t0=t0,
                                             Tinit=initialTemperature, 
                                             ub=surf_T, lb_type=2, grad=grad,
                                             user_action=user_action,
                                             outfile=outfile,
                                             outint=outint)
    
    # plot final result
    plot_solution.update(u, x, t[-1])
    
    # Print the time spent
    print 'Final run, CPU time: {0:.3f} s'.format(cpu)    
    
    
    