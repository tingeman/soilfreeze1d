# -*- coding: utf-8 -*-
"""
@author: thin

===============================================================================

This model script illustrates the use of the soilfreeze1d module for
calculating the thermal regime in embankment constructions.
The model first sets up and initializes a model of the natural soil.
After the model has been run for a sufficient amount of time to obtain
stable thermal conditions, the construction of an embankment is 
modelled by adding a five meter thick layer on top of the model.
 
This model is implemented using the stefan solution (phase change occurs 
linearly over a specified temperature interval).

The model is forced by a harmonic temperature variation at the upper boundary. 
The lower boundary has a specified constant gradient of 0.02 C/m.
Results will be written to the file model3_nug_results.txt at a daily interval.

The difference in this model compared with model3.py is that here we use a 
non-uniform grid to reduce the number of grid-points and thus speed up the
calculations.

NB: It turns out the non-uniform-grid solution is not faster, at least not in 
this particular case!

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

    # Define the model layers and properties
    Layers = soilfreeze1d.new_layered_model(type='stefan_thfr')
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
    
    
    # Define model domain properties
    Nx = 200      # The number of nodes in the model domain is Nx+1
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
    grad=0.02     # [K/m]
    
    # Set up plotting
    fignum  = 99    # Plot results in figure number 99        
    Tmin = -11      # The minimum value on the temperature axis
    Tmax = +9       # The maximum value on the temperature axis
    z_max = 30      # Maximum plot depth on the z-axis    
    
    # Set up result output 
    outfile = 'model3_nug_results.txt' # Name of the result file
    outint = 1*days  # The interval at which results will be written to the file    
    
    # Define model domain properties

    # We use a minimum node spacing of 1 cm, and increase the node spacing by
    # 10% (factor of 1.1) for each deeper node from the ground surface and down.

    dx0 = 0.01
    dxf = 1.1
    
    Nx = 0
    xtmp = [0, 1]
    xsum = dx0
    while xsum<(Layers.z_max-Layers.surface_z):
        Nx += 1
        xtmp.append(dxf**Nx)
        xsum += xtmp[-1]*dx0
    
    x = np.cumsum(xtmp)*dx0+Layers.surface_z
    x[-1] = Layers.z_max
    
    # Plot initial condition
    plot_solution = soilfreeze1d.Visualizer_T(Layers, Tmin=Tmin, Tmax=Tmax, z_max=z_max, fig=fignum)
    plot_solution.initialize(initialTemperature(x), x, 0., name=outfile)

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
    u, x, t, cpu = soilfreeze1d.solver_theta_nug(Layers, x, dt, T, 
                                                 t0=t0,
                                                 Tinit=initialTemperature, 
                                                 ub=surf_T, lb_type=3, grad=grad,
                                                 user_action=user_action,
                                                 outfile=outfile,
                                                 outint=outint,
                                                 silent=True)
    
    # plot final result
    plot_solution.update(u, x, t)
    
    # Print the time spent
    print(' ')
    print(' ')
    print('Spin-up run, CPU time: {0:.3f} s'.format(cpu))
    print(' ')
    print("Waiting 5 seconds (inspect model result)")
    time.sleep(5)
    
    
    #==========================================================================
    # ...Then we use the output to set up the next model run
    # Which now has an extra layer added.
    #==========================================================================
 
    # Define the model layers and properties

    Layers = soilfreeze1d.new_layered_model(type='stefan_thfr', surface_z=-5.)
    Layers.add(Thickness=5,  n=0.05, C_th=2.6E6, C_fr=1.7E6, k_th=1.6, k_fr=3.0, interval=1.0, Tf=0.0, soil_type='Top Soil')
    Layers.add(Thickness=30,  n=0.30, C_th=2.6E6, C_fr=1.7E6, k_th=1.6, k_fr=3.0, interval=1.0, Tf=0.0, soil_type='Soil 1')
    
    # The new model definition has an extra layer (5 m thick) added on top of the original
    # soil column, and the surface of the model is specified at -5 m (5 m above the original surface).
    
    
    # Define initial condition (temperatures in domain)
    initialTemperature = soilfreeze1d.DistanceInterpolator(depths=x, temperatures=u)
    
    # The arrays x and u contain grid node locations and the final model temperatures 
    # from the intial simulation. Temperatures are present for depths from 0 to -30 m.
    
    # The distance interpolator is a class to make linear interpolation of temperature profiles.
    # We use it in this case to transfer the final model temperatures from the initial simulation 
    # to the new node locations that we setup for the second run with added embankment structure.

    # The class instance will be passed to the solver, which will call it with the new grid node 
    # locations and receive as output the initial temperatures interpolated on the new grid.

    # As we now have extended the model domain upward by 5 m (to x = -5 m) the interpolator class
    # will automatically use the upper-most temperature specified (in this case at x = 0 m)
    # for any nodes above that (from 0 to -5 m).


    # Now we redefine grid node locations for the new model:

    # Again we use a minimum node spacing of 1 cm, and increase the node spacing by
    # 10% (factor of 1.1) for each deeper node.
    # This time we start from the top of the embankment - This is automatically handled
    # since we specified the surface of the layered model now is at -5 m elevation
    # (5 m above the previous natural soil surface)

    dx0 = 0.01
    dxf = 1.1
    
    Nx = 0
    xtmp = [0, 1]
    xsum = dx0
    while xsum<(Layers.z_max-Layers.surface_z):
        Nx += 1
        xtmp.append(dxf**Nx)
        xsum += xtmp[-1]*dx0
    
    x = np.cumsum(xtmp)*dx0+Layers.surface_z
    x[-1] = Layers.z_max


    # set up other parameters:4

    animate = True  # Show animation of the model results during calculations
    dt = 0.5*days   # The maximum calculation time step
    t0 = t          # Set start time to the end time of previous run
    T = 10*365*days+t0 # The total calculation period is ten years

    # Set up result output 
    outfile = 'model3b_results.txt' # Name of the result file
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
    u, x, t, cpu = soilfreeze1d.solver_theta_nug(Layers, x, dt, T,
                                                 t0=t0,
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
