# -*- coding: utf-8 -*-
"""
@author: thin

===============================================================================

This model script illustrates the use of the soilfreeze1d module for 
calculating the thermal regime in a 1D model with one layer using the stefan 
solution (phase change occurs linearly over a specified temperature interval).
The model domain has an initial termperature of -2C at all nodes, and the 
model is forced by a harmonic temperature variation at the upper boundary. 
The lower boundary has a specified constant gradient of 0.08333 C/m.
Results will be written to the file model1_results.txt at a daily interval.
This verstion uses the non-uniform grid code, with a node spacing increase
factor of 1.1

===============================================================================

"""

# import standard pythom modules
import numpy as np
import pdb
import warnings

# import own modules
import soilfreeze1d   # import the module containing the Finite Difference 
                      # modelling code and supporting functions.


# Define variables and constants
days = 24*3600  # Define a constant for conversion from days to seconds


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
    Layers = soilfreeze1d.LayeredModel(type='stefan')
    Layers.add(Thickness=30,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, interval=1.0, Tf=0.0, soil_type='Soil 1')    

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
    dx0 = 0.05
    dxf = 1.1
    
    Nx = 0
    xtmp = [0, 1]
    xsum = dx0
    while xsum<(Layers.z_max-Layers.surface_z):
        Nx += 1
        xtmp.append(dxf**Nx)
        xsum += xtmp[-1]*dx0
    
    x = np.cumsum(xtmp)*dx0+Layers.surface_z
    x = x[:-1]
    
    dt = 1*days   # The calculation time step
    T = 10*365*days    # The total calculation period

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)    

    # Example of a function that will give a constant
    # upper boundary temperature
    
    #def surf_T(t):
    #    return -2.
    
    # Define the geothermal gradient (lower boundary)    
    grad=0.08333     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99    
    animate = False  # Plot result of each model time step    
                    # If set to False, only first and last
                    # time step will be plotted.
    
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9      # The maximum value on the temperature axis
    z_max = 30      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outfile = 'model1_nug_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    # Plot initial condition
    plot_solution = soilfreeze1d.Visualizer_T(Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., Layers, name=outfile)
        
    # Switch animation on or off
    if animate:
        user_action = plot_solution
    else:
        user_action = None
    
    # Call Finite Difference engine    
    
    # # upgrade warnings to errors
    # with warnings.catch_warnings():
    #     warnings.simplefilter('error')
    
    u, x, t, cpu = soilfreeze1d.solver_theta_nug(Layers, x, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub=surf_T, lb_type=2, grad=grad,
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
    
    # plot final result
    plot_solution.update(u, x, t)
    
    print 'CPU time:', cpu
    
    
    
    
    Nx = 100
    
    # Set up result output 
    outfile = 'model1_ug_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    # Call Finite Difference engine    
    u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                                 Tinit=initialTemperature, 
                                                 ub=surf_T, lb_type=2, grad=grad,
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint)
    
    plot_solution.add(u, x, t, 'b')