# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 01:06:32 2016

@author: thin
"""

# import standard pythom modules
import numpy as np

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
    Layers = soilfreeze1d.LayeredModel(type='unfrw')
    Layers.add(Thickness=3,  n=0.02, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.19, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    Layers.add(Thickness=28,  n=0.3, C_th=2.5E6, C_fr=2.5E6, k_th=1.8, k_fr=1.8, alpha=0.05, beta=0.4, Tf=-0.0001, soil_type='Fairbanks Silt')    
    
    
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
    Nx = 100        # The number of nodes in the model domain is Nx+1
    dt = 0.5*days   # The calculation time step
    T = 365*days    # The total calculation period

    # Define the forcing upper boundary temperature
    surf_T = soilfreeze1d.HarmonicTemperature(maat=-2, amplitude=8, lag=14*days)    

    # Define the geothermal gradient (lower boundary)    
    grad=0.08333     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99    
    animate = True  # Plot result of each model time step    
                    # If set to False, only first and last
                    # time step will be plotted.
    
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9      # The maximum value on the temperature axis
    z_max = 10      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outfile = 'model4_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    
    x = np.linspace(Layers.surface_z, Layers.z_max, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    
    # Plot initial condition
    plot_solution = soilfreeze1d.Visualizer_T_dT(Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., Layers, name=outfile)
        
    # Switch animation on or off
    if animate:
        user_action = plot_solution
    else:
        user_action = None
    
    # Call Finite Difference engine    
    u, x, t, cpu = soilfreeze1d.solver_theta(Layers, Nx, dt, T, 
                                             Tinit=initialTemperature, 
                                             ub=surf_T, lb_type=2, grad=grad,
                                             user_action=user_action,
                                             outfile=outfile,
                                             outint=outint)
    
    # plot final result
    plot_solution.update(u, x, t[-1])
    
    print 'CPU time:', cpu