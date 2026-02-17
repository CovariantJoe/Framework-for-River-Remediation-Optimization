"""
Script to replicate the benchmarks in the article.
This script controls our Shallow Waters solver in velocity.py to obtain the velocity and water height fields,
direct comparison is then performed to analytical solutions from the SWASHES code

---------------------- Assumptions ---------------------------------
- THIS SCRIPT WAS ONLY TESTED IN A LINUX OPERATING SYSTEM 
- The mass unit is grams [g], the rest of units are SI units (e.g concentration g/m^3). 
- The analytical solution comes from the Python package SWASHES whose purpose is precisely to conduct these tests
   you can install this via conda install conda-forge::swashes
---------------------- Dependencies --------------------------------------------
Tested dependencies:
    Python: 3.10.16
    dolfin: 2019.1.0
    pandas: 2.3.3
    swashes.__version__: 1.05.00  Installed through conda in the path returned by swashes.__file__
"""
from velocity import main # Own numerical result
from Solver import PollutionSolver
import pandas as pd
import numpy as np
from dolfin import *
from pathlib import Path as path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import sys
import io
import subprocess
#---------------------------------------- Configure -------------------------------------------------
ANIMATE = False # Whether to save an animation, or plot the last frame only (False)
TEST = 2 # Either 1 or 2, respectively for SWE module or advection-diffusion module
#----------------------------------------------------------------------------------------------------

g = 9.81

def SWASHES(params):
    import swashes # Solution to compare
    lib_path = path(swashes.__file__).resolve().parent
    executable = "data/bin/swashes" if not sys.platform.startswith("win") else "data/bin/swashes.exe"
    swashes_path = os.path.join(lib_path, executable)

    args = [swashes_path] + [str(p) for p in params]    
    try:
        run = subprocess.run(args, capture_output=True, text=True)
        output = run.stdout
    except:
        raise Exception(f"Failed to run SWASHES. Make sure you can run it in {swashes_path} \n passing: {[str(p) for p in params]}")
    
    df = pd.read_csv(
        io.StringIO(output), 
        sep=r'\s+', 
        comment='#', 
        header=None
    )
    return df

def SWE_Comparison(u_sim, h_sim, df, Z, axes):
    """
    Compares 2D FEniCS solutions to 1D analytical data in a DataFrame for SWE.
    
    Parameters:
    u_sim: dolfin.Function (Vector field, P1)
    h_sim: dolfin.Function (Scalar field, P2)
    df: pandas.DataFrame containing analytical solution at a certain moment
    """
    
    # Extract coordinates and analytical values
    x_coords = df.x.values
    h_analytical = df.h.values
    u_analytical = df.u.values
    
    # Sample FEniCS solutions at (x, 0)
    h_numeric = np.array([h_sim(x, 0) for x in x_coords])
    # For u, we take the x-component (index 0) of the vector field
    u_numeric = np.array([u_sim(x, 0)[0] for x in x_coords])
    
    # Calculate relative errors
    error_h = np.abs(h_numeric - h_analytical)/np.abs(h_analytical)
    error_u = np.abs(u_numeric - u_analytical)/np.abs(u_analytical)
    
    # Plotting
    for ax in axes.flat:
        ax.clear()
    
    # Scalar Field h Comparison
    axes[0, 0].plot(x_coords, h_analytical, 'k--', lw=2.0, label='SWASHES')
    axes[0, 0].plot(x_coords, h_numeric, 'r-', alpha=0.6, lw=1.5, label='FEniCS (This work)')
    axes[0, 0].set_ylabel('Amplitude',fontsize=14)
    axes[0, 0].set_title(r'$h$',fontsize=16)
    axes[0, 0].legend(fontsize=12)
    axes[0,0].tick_params(axis='both', labelsize=13)
    
    axes[1, 0].plot(x_coords, error_h, 'g-')
    axes[1, 0].set_ylabel('Relative Error',fontsize=14)
    axes[1, 0].set_xlabel(r'$x$',fontsize=14)
    axes[1,0].tick_params(axis='both', labelsize=13)
    
    # Vector Field u (x-component) Comparison
    axes[0, 1].plot(x_coords, u_analytical, 'k--', lw = 2.0, label='SWASHES')
    axes[0, 1].plot(x_coords, u_numeric, alpha=0.6, lw = 1.5, label='FEniCS (This work)')
    axes[0, 1].set_title(r'$u_x$',fontsize=16)
    axes[0, 1].legend(fontsize=12)
    axes[0,1].tick_params(axis='both', labelsize=13)
    
    axes[1, 1].plot(x_coords, error_u, 'g-')
    axes[1, 1].set_xlabel(r'$x$',fontsize=14)
    axes[1,1].tick_params(axis='both', labelsize=13)
    
    plt.tight_layout()
    fig.subplots_adjust(
    left=0.08,
    right=0.987,
    bottom=0.078,
    top=0.952,
    wspace=0.242,
    hspace=0.058)

    return

def update(frame, u_data, h_data, axes, df, Z):
    """
    Function to create the animation
    """
    u_curr = u_data[frame]
    h_curr = h_data[frame]
    SWE_Comparison(u_curr, h_curr, df, Z, axes)
    
def SWE_Sim(ANIMATE):
    """
    Function to run the simulation for the SWE benchmark
    """
    from DATA import Path
    Lx = 25
    nx,ny = 250,1
    Ly = 0.625
    time_steps = 418
    dt = 0.25

    def outlet(x, on_boundary):
        return on_boundary and near(x[0],Lx, Ly/2)
    
    def inlet(x,on_boundary):
        return on_boundary and near(x[0], 0, 1e-2)
    
    class OutletBoundary(SubDomain):
        # Class to initiate the outlet wall boundary.
        def inside(self, x, on_boundary):
            return outlet(x,on_boundary)
        
    class InletBoundary(SubDomain):
        # Class to initiate the inlet wall boundary.
        def inside(self, x, on_boundary):
            return inlet(x,on_boundary)

    mesh = RectangleMesh(Point(0.0, -Ly/2), Point(Lx, Ly/2), nx, ny)
    x = SpatialCoordinate(mesh);
    
    Z0 = conditional(
        ge(0.2 - 0.05*(x[0] - 10)**2, 0.0),
        0.2 - 0.05*(x[0] - 10)**2,
        0.0)
    
    t = Constant(0.0)
    base_height = 2.0
    delta_h = 0.001 # Height of incoming inlet velocity field
    u_desired = 2.21; u0 = 1e-5 # Desired inlet velocity; initial velocity everywhere
    
    # Gradually ramp the influx over 40 seconds, using u_inlet = u_desired directly will break the simulation
    u_inlet = Expression(("u0 + (u_desired - u0) * min(1.0, t/40)","0.0"), degree=1, u0=u0, u_desired=u_desired, t=0.0)
    # Gradually ramp the topography over 40 seconds, using Z = Z0 directly will break the simulation
    Z = conditional( gt( t/40.0, 1.0 ), Z0,  Z0*t/40.0 )
    
    params = [1, 1, 1, 1, nx] 
    df = SWASHES(params)
    df.columns = ["x", "h", "u", "z", "q", "z+h", "Fr", "z+h*c"]
    v, h = main(u_inlet, delta_h, mesh, [time_steps, dt], bc = [""], u_ini = (u0,0), base_height = base_height, z = Z, t=t, ANIMATE = ANIMATE, InletClass = InletBoundary, OutletClass = OutletBoundary)
    return v, h, df, Z0

def toMatrix(coords, x, y, f_list):
    '''
    Convert fenics output vectors to a matrix on the mesh
    '''
    nx = len(x)
    ny = len(y)
    
    f_matrix = np.zeros((ny, nx))
    for i, (x_, y_) in enumerate(coords):
        row = np.where(y == y_)[0][0]
        col = np.where(x == x_)[0][0]
        f_matrix[row, col] = f_list[i]
    return f_matrix
        
def AD_Analytical(mesh, u_sim, a_sim, sigma, point, a0, D, t):
    """
    Function for the advection-diffusion (AD) analytical solution
    """
    x0 = point[0]; y0 = point[1]
    coords = mesh.coordinates()
    u_values = np.array([u_sim(coord)[0] for coord in coords]) # x component
    a_values = np.array([a_sim(coord) for coord in coords])
    
    x = np.sort(np.unique(coords[:, 0]))
    y = np.sort(np.unique(coords[:, 1]))
    X,Y = np.meshgrid(x,y) 
    u_matrix = toMatrix(coords, x, y, u_values)
    
    a_sim = toMatrix(coords, x, y, a_values)
    a_analytical = a0/(2*np.pi*(sigma**2 + 2*D*t)) * np.exp(- ((X-x0-u_matrix*t)**2 + (Y-y0)**2 )/(2*(sigma**2 + 2*D*t) ) )
    return a_sim, a_analytical, X, Y

def AD_Comparison(sim, analytical, X, Y):
    """
    Function for the advection-diffusion (AD) analytical vs numerical comparison
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 6))
    
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.3
    )
    
    ax3 = fig.add_subplot(gs[:, 0])      # full left column
    ax1 = fig.add_subplot(gs[0, 1])      # top-right
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)  # bottom-right
    
    # Plot simulation as filled contours
    cf = ax3.contourf(X, Y, sim, levels=10, cmap='inferno', alpha=0.8)
    # Plot analytical as dashed lines for direct comparison
    cl = ax3.contour(X, Y, analytical, levels=cf.levels, colors='white', linestyles='dashed', linewidths=1)
    ax3.set_title("FEniCS (This work, filled) vs. Analytical (Dashed)",fontsize=16)
    ax3.set_xlabel("$x$",fontsize=14)
    ax3.set_ylabel("$y$",fontsize=14)
    ax3.set_xlim([2.7,5.0])
    ax3.set_ylim([-1.5,1.5])
    plt.colorbar(cf, ax=ax3, shrink = 0.92)
    ax3.tick_params(axis='both', labelsize=13)

    # Centerline Slice (along y-midpoint or peak)
    mid_idx = sim.shape[0] // 2 - 1 # Assuming the pulse passes through the center
    analytical_slice = analytical[mid_idx, :]
    sim_slice = sim[mid_idx, :]
    x_slice = X[mid_idx, :]
    ax1.plot(x_slice, analytical_slice, 'k--', lw = 2.0, label='Analytical')
    ax1.plot(x_slice, sim_slice, 'r-', lw = 1.5, alpha = 0.6, label='FEniCS (This work)')
    ax1.set_title(f"Profile at $y$ = {Y[mid_idx, 0]:.2f}",fontsize=16)
    ax1.set_ylabel("Amplitude",fontsize=14)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=13)

    # Absolute error with respect to maximum value (prevents division by very small numbers where solution is near 0)
    # Uncomment both lines for relative error with respect to peak's maximum in order to avoid division by 0
    
    # scale = np.max(np.abs(analytical_slice))
    abs_error = np.abs(sim_slice - analytical_slice) #/ scale
    
    ax2.plot(x_slice, abs_error)
    ax2.set_ylabel("Absolute Error", fontsize=14)
    ax2.set_xlabel("$x$",fontsize=14)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(axis='both', labelsize=13)
    return fig

def AD_Sim():
    """
    Function for the advection-diffusion (AD) benchmark
    """
    from Solver import AdsorbentRegion
    Lx = 5
    nx,ny = 45,45
    Ly = 5
    time_steps = 230
    dt = 0.02

    def outlet(x, on_boundary):
        return on_boundary and near(x[0],Lx, Ly/2)
    
    def inlet(x,on_boundary):
        return on_boundary and near(x[0], 0, 1e-2)
    
    class OutletBoundary(SubDomain):
        # Class to initiate the outlet wall boundary.
        def inside(self, x, on_boundary):
            return outlet(x,on_boundary)
        
    class InletBoundary(SubDomain):
        # Class to initiate the inlet wall boundary.
        def inside(self, x, on_boundary):
            return inlet(x,on_boundary)

    mesh = RectangleMesh(Point(0.0, -Ly/2), Point(Lx, Ly/2), nx, ny)
    x = SpatialCoordinate(mesh);
    
    base_height = 2.0
    delta_h = 0.000 # Height of incoming inlet velocity field
    u_inlet = (0.5,0.0); u0 = u_inlet # Desired inlet velocity; initial velocity everywhere
    
    # Create a uniform, constant vector field
    v, h = main(u_inlet, delta_h, mesh, [time_steps, dt], bc = [""], u_ini = u0, base_height = base_height, z = 0, t=0, ANIMATE = ANIMATE, InletClass = InletBoundary, OutletClass = OutletBoundary)
    
    # Run the advection-difussion simulation (the field will be the adsorbent field, and pesticide will be off)
    D = 0.02
    pollutant = {"Name":"Benchmark", "Conc":0,"Dm":0}
    adsorbent = {"Name":"Benchmark","Pollutant":"Benchmark","Model":1,"Values":{"qmax":0,"b":0,"k1":1e-10}, "Transport":{"rho_s":1,"porosity":0}}
    a0 = 1; y0 = 0; width = 0.3
    
    Final_Pesticide_conc, Pesticide_dist = PollutionSolver(adsorbent, pollutant, mesh, mode = "normal", FROM_CHECKPOINT = False, ONLY_OUTLET = False, MAX_CONC = 0, a_0 = a0, y = y0, width = width, Config = [dt,dt,1, D, ""])    
    a_sim = PollutionSolver(adsorbent, pollutant, mesh, mode = "normal", FROM_CHECKPOINT = True, ONLY_OUTLET = False, N_TIME_STEPS_CHKP = 1, MAX_CONC = 0, a_0 = a0, y = y0, width = width, Config = [dt,dt,1, D, "benchmark"])

    adsorbentDistribution = AdsorbentRegion(a0, y0, width, mesh)
    return AD_Analytical(mesh, v, a_sim, adsorbentDistribution.sigma, adsorbentDistribution.point, adsorbentDistribution.a0, D, time_steps*dt)
    
if __name__ == "__main__":
    if TEST == 1:
        v, h, df, Z = SWE_Sim(ANIMATE)
        Path = str(path(__file__).resolve().parent)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex='col')    
        if ANIMATE:
            loc = os.path.join(Path,"Benchmark.mp4")
            print(f"Saving animation to: {loc}")
            ani = FuncAnimation(fig, update, frames=len(v), fargs=(v, h, ax, df, Z), repeat=False)
            writer = FFMpegWriter(fps=16)
            ani.save(loc, writer=writer)
            plt.close(fig)
        else:
            SWE_Comparison(v, h, df, Z, ax)
    elif TEST == 2:
        a_sim, a_analytical, X, Y = AD_Sim()
        fig = AD_Comparison(a_sim, a_analytical, X, Y)
        if ANIMATE:
            print("\n Simulation not available with this test")
    else:
        raise Exception("Only test = 1 and test = 2 available")
