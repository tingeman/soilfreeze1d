# -*- coding: utf-8 -*-
"""
@author: thin

===============================================================================

This model script illustrates the use of the soilfreeze1d module for
calculating the thermal regime in embankment constructions.
The model first sets up and initializes a model of the natural soil.
After the model has been run for a sufficient amount of time to obtain
stable thermal conditions (spin-up), the construction of an embankment is 
modelled by adding a five meter thick layer on top of the model.

The spin-up is deliberately set to be 10 years, which is typically much too 
short for at real model spin-up.

This model is implemented using the stefan solution (phase change occurs 
linearly over a specified temperature interval).

The model is forced by a harmonic temperature variation at the upper boundary. 
The lower boundary has a specified constant gradient of 0.02 C/m.
Results will be written to the file model1_results.txt at a daily interval.

This model run is split across two scripts:
- model3_spinup.py:       Calculates the model spin-up (without embankment)
- model3_embankment.py:   Reads the temperature distribution from the last 
                            timestep of the spin-up, adds the embankment
                            to the model domain, and calculates the final 
                            model response

This file is model3_embankment.py.
                            
===============================================================================

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

    #==========================================================================
    # First read in the data from the previous model spin-up run
    #==========================================================================
    
    # Read the final temperature distribution from the spin-up run
    outreader = soilfreeze1d.FileReader('model3_spinup_results.txt')
    spinup_result = outreader.get(index=-1)
    
    # spinup_result['t']    contains the time (seconds) of the model output
    # spinup_result['x']    grid point locations
    # spinup_result['u']    temperature profile


    #==========================================================================
    # ...Then we use the output to set up the next model run
    # Which now has an extra layer added.
    #==========================================================================

    # Define the model layers and properties
    Layers = soilfreeze1d.new_layered_model(type='stefan_thfr', surface_z=-5.)
    Layers.add(Thickness=5,  n=0.05, C_th=2.6E6, C_fr=1.7E6, k_th=1.6, k_fr=3.0, interval=1.0, Tf=0.0, soil_type='Top Soil')
    Layers.add(Thickness=30,  n=0.30, C_th=2.6E6, C_fr=1.7E6, k_th=1.6, k_fr=3.0, interval=1.0, Tf=0.0, soil_type='Soil 1')
    
    # Thickness:    Thickness of the layer [m]
    # n:            Porosity [-]   Soil is considered fully saturated
    # C_th:         Heat capacity, thawed state [J/(m^3*C)]
    # C_fr:         Heat capacity, frozen state [J/(m^3*C)]    
    # k_th:         Thermal conductivity, thawed state [W/(m*C)]
    # k_fr:         Thermal conductivity, frozen state [W/(m*C)]
    # interval:     Phase change interval [C]   default = 1C
    # Tf:           Freezing point [C]  default = 0C
    # soil_type:    Character string for identification, not used by model
    
    # The new model definition has an extra layer (5 m thick) added on top of the original
    # soil column, and the surface of the model is specified at -5 m (5 m above the original surface).
    
    # Define initial condition (temperatures in domain)
    initialTemperature = soilfreeze1d.DistanceInterpolator(depths=spinup_result['x'], 
                                                           temperatures=spinup_result['u'])
    
    # The arrays x and u contain the final model temperatures from the intial simulation.
    # Temperatures are present for depths from 0 to -30 m.
    # As we now have extended the model domain upward by 5 m (to x = -5 m) the model 
    # will automatically use the upper-most temperature specified (in this case at x = 0 m)
    # for any nodes above that (from 0 to -5 m).

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)    

    # Define the geothermal gradient (lower boundary)    
    grad=0.02     # [K/m]


    # Define model domain properties
    Nx = 200      # The number of nodes in the model domain is Nx+1
    dt = 1*days   # The calculation time step

    x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # mesh points in space
    dx = x[1] - x[0]

    # Set up plotting
    fignum  = 99    # Plot results in figure number 99        
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9       # The maximum value on the temperature axis
    z_max = 30      # Maximum plot depth on the z-axis    
        
    # define the grid points
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # mesh points in space
    dx = x[1] - x[0]
    # In this second part of the model we have used the same number of node points
    # as in the original model. As the model domain is larger, the nodes will be 
    # less densely spaced.
    # We could have also chosen a larger Nx value proportionally to the 
    # expansion of the domain... to obtain the same node distance.

    animate = False  # Show animation of the model results during calculations
    dt = 1*days               # The maximum calculation time step
    t0 = spinup_result['t']   # Set start time to the end time of previous run
    T = 10*365*days+t0        # The total calculation period is ten years

    # Set up result output 
    outfile = 'model3_embankment_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    # Reinitialize plotting
    plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, t0, name=outfile)
    
    # Switch animation on or off
    if animate:
        user_action = plot_solution
    else:
        user_action = None
    
    print("Waiting 5 seconds before commencing new run (inspect initial model)")
    time.sleep(5)
    
    # Call Finite Difference engine    
    u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, t0=t0,
                                             Tinit=initialTemperature, 
                                             ub=surf_T, lb_type=3, grad=grad,
                                             user_action=user_action,
                                             outfile=outfile,
                                             outint=outint,
                                             silent=True)
    
    # plot final result
    plot_solution.update(u, x, t)
    
    print(' ')
    print(' ')
    print('Final run, CPU time: {0:.3f} s'.format(cpu))    
    print(' ')
    print('Close figure to return focus to the terminal...')
    plot_solution.show()
