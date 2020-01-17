# -*- coding: utf-8 -*-
"""
@author: thin

===============================================================================

This model script illustrates the use of the soilfreeze1d module for 
calculating the thermal regime in a 1D model with one layer using the stefan 
solution (phase change occurs linearly over a specified temperature interval).

The model domain is initialized with a piece wise linear temperature,
illustrating how the DistanceInterpolator class may be used to initialize
the model temperature, e.g. by interpolating between measurements from at
thermistor string.

The model is forced by a harmonic temperature variation at the upper boundary. 
The lower boundary has a specified constant gradient of 0.02 C/m.
Results will be written to the file model2_results.txt at a daily interval.

===============================================================================

"""

# import standard pythom modules
import pdb
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
    Layers = soilfreeze1d.new_layered_model(type='stefan')
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
    dt = 0.5*days   # The calculation time step
    T = 5*365*days    # The total calculation period

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
    grad=0.02     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99    
    animate = True  # Plot result of each model time step    
                    # If set to False, only first and last
                    # time step will be plotted.
    
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9       # The maximum value on the temperature axis
    z_max = 30      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outfile = 'model2_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx)   # mesh points in space
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
                                             ub=surf_T, lb_type=3, grad=grad,
                                             user_action=user_action,
                                             outfile=outfile,
                                             outint=outint)
    
   # plot final result
    plot_solution.update(u, x, t)
    
    print ' '
    print ' '
    print 'CPU time:', cpu
    print ' '
    print 'Close figure to return focus to the terminal...'
    plot_solution.show()