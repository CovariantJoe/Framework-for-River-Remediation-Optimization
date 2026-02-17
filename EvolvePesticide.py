"""
Script to manually control Solver.py (transport-adsorption) for testing, preparations for optimization, etc.
Also includes most figures in the article as functions below.

It serves two main modes of usage:
    
    1) CREATING THIS CHECKPOINT IS ESSENTIAL TO RUN OPTIMIZATION
        FROM_CHECKPOINT = False: Evolve a pesticide field from the inlet without adsorption until it fills the river
        to a concentration near MAX_CONC percentage of that set in DATA.py (e.g fill river to 99.5% of target conc.)
        
    2) FROM_CHECKPOINT = True: Conduct a dual transport-adsorbent simulation from the checkpoint and initial conditions of the
        adsorbent. Return the mean pesticide concentration in the region selected by ONLY_OUTLET as a function of time

---------------------------- Variables ----------------------------------------
- adsorbent: Dictionary of parameters for a single adsorbent as defined in DATA.py
- pollutant: Dictionary of parameters for a single pesticide as defined in DATA.py
- mode ("fast", "normal"). Determines whether to write every single time step to a file (normal),
  or just the final state (fast) often at at minimum pollution concentration, noticeably impacting the runtime
- FROM_CHECKPOINT (True or False): determines whether the program runs to evolve the pesticide alone up to a checkpoint time, 
  or if it begins from a checkpoint of pesticide concentration and evolves with an adsorbent from that time forward.
- ONLY_OUTLET (True or False): determines whether the program returns concentrations only in the outlet region defined by OutletRegion class,
  or in the complete mesh domain. This materially changes the optimization problem from all the mesh to only the region defined by OutletRegion
- MAX_CONC (float, 0 to 1): When using with FROM_CHECKPOINT = False, simulation attempts to fill the river to this percentage of Conc in DATA.py 
- a_0, y, width (floats): Initial conditions for the adsorbent leading to (x_0, y_0) centered Gaussian distribution through class AdsorbentRegion in Solver.py.

-------------------------- Assumptions ----------------------------------------
- THIS SCRIPT WAS ONLY TESTED IN A LINUX OPERATING SYSTEM 
- There are .h5 files in the same directory as the script containing the full 
   evolution of the velocity field during the same time as intended here
- The file mesh.xml contains the same mesh used to evolve the velocity field
- The mass unit is grams [g], the rest of units are SI units (e.g concentration g/m^3). 

---------------------- If using different mesh ---------------------------------
Some mesh coordinates are hard coded to define the different regions.
Read this section inside Solver.py and velocity.py for an overview
---------------------------- Dependencies -------------------------------------
Tested dependencies:
    Python 3.10.16
    dolfin 2019.1.0    
"""
from Solver import *
from DATA import MeshData
# ----------------------------------- CONFIG ----------------------------------------
FROM_CHECKPOINT = False # Whether to create a checkpoint evolving without adsorption (False), or True
ONLY_OUTLET = False # Whether to return concentrations only in the outlet subdomain defined in Solver.py
SOLVE_FOR = 0 # Adsorbent-pesticide pair [0,3] to solve for according to definition in DATA.py
mode = "fast" # Output mode, save only final state to file inside Fields directory (fast), save every timestep to file (normal)
MAX_CONC = 0.99 # Value [0,1] of percentage of Conc in DATA.py to attempt to fill the river to when using FROM_CHECKPOINT = False

# Relevant only when FROM_CHECKPOINT = True:
y0 = -700  # Initial adsorbent y0
w0 = 0.5 # Initial width in river [0,1] to place adsorbent, leading to internal x0,y0 pair
a0 = 600 # Initial adsorbent dose
#------------------------------------------------------------------------------------
try: 
    Adsorbents[SOLVE_FOR]
    Pollutants[SOLVE_FOR]
except:
    raise Exception("Invalid value for variable SOLVE_FOR")
if FROM_CHECKPOINT:
    Pesticide_conc, Pesticide_dist, adsorbent_dist = PollutionSolver(Adsorbents[SOLVE_FOR], Pollutants[SOLVE_FOR], MeshData, mode = mode, FROM_CHECKPOINT = FROM_CHECKPOINT, ONLY_OUTLET = ONLY_OUTLET, MAX_CONC = MAX_CONC, a_0 = a0, y = y0, width = w0)
else:
    Final_Pesticide_conc, Pesticide_dist = PollutionSolver(Adsorbents[SOLVE_FOR], Pollutants[SOLVE_FOR], MeshData, mode = mode, FROM_CHECKPOINT = FROM_CHECKPOINT, ONLY_OUTLET = ONLY_OUTLET, MAX_CONC = MAX_CONC, a_0 = a0, y = y0, width = w0)

'''
Code for some figures in the article
'''
def plot_C(Pesticide_dist):
    pollutant = Pollutants[SOLVE_FOR]
    levels = np.linspace(22.5, 107.5, 32)
    cu = fe.plot(100*Pesticide_dist/(Pesticide_conc[0]), levels = levels, cmap = "plasma", extend="max")
    col = plt.colorbar(cu, shrink = 0.92)
    
    col.set_label(r"Percent of $C_{0}$", fontsize = 13)
    col.ax.tick_params(labelsize=10)
    plt.xlabel(r"$x$", fontsize = 13)
    plt.ylabel(r"$y$", fontsize = 13)
    plt.ylim([-330,1])
    plt.xlim([-29,232.7])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(r"$\text{Distribution of pesticide at } t = t_{min}$")
    return cu, col

def plot_C2(Pesticide_conc, Pesticide_dist):
    pollutant = Pollutants[SOLVE_FOR]
    levels = np.linspace(17.5, 102, 22)
    fig,ax = plt.subplots()
    cu = fe.plot(100*Pesticide_dist/(Pesticide_conc[0]), levels=levels,cmap = "plasma", extend="max")
    col = plt.colorbar(cu, shrink = 0.92)
    
    col.set_label(r"Percent of $C_{0}$", fontsize = 16)
    col.ax.tick_params(labelsize=14)
    
    ax.scatter([174.4],[-301.1], marker='*', s=200, c='red', edgecolors='k', zorder=10)
    ax.set_xlabel(r"$x$", fontsize = 15)
    ax.set_ylabel(r"$y$", fontsize = 15)
    ax.set_ylim([-330,1])
    ax.set_xlim([-29,232.7])
    ax.tick_params(labelsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(r"$\text{Distribution of pesticide at } t = t_{min}$",fontsize=18)
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_fontsize(14)
    return cu, col    

def plot_A():
    '''
    This should be the code for Fig. 8
    '''
    from sklearn.metrics import r2_score
    # IMPORTANT: As the code is, Cs is in the opposite order of As (225 grams gave 0.94182, not 0.269688)
    As = [225, 543, 936.7, 1271, 1697, 2113, 2595, 3262, 3860, 4673]
    Cs = np.array([0.269688, 0.318057, 0.3629068, 0.43176847, 0.49359, 0.552646, 0.636077, 0.716757, 0.8289, 0.94182])

    x = np.array(As[::-1])/1000
    logy = np.log10(Cs)
    
    # Fit
    slope, intercept = np.polyfit(x, logy, 1)
    fit_line = slope * x + intercept
    r_squared = r2_score(logy, fit_line)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.plot(x, logy, 'o', label='Data', markersize=5)
    ax.plot(x, fit_line, '-', label='Fit: ' + r'$y = $' +  f"{slope:.4f}" + r"$x$" + f"{intercept:.4f}" + "\n" + r"$R^2$" + f"= {r_squared:.3f}")
    
    ax.set_xlabel(r'$A_0$ [Kg]', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(C_{min}/C_0)$ ', fontsize=12)
    ax.set_title('Log-Linear Fit of Data', fontsize=13)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(frameon=False, fontsize=10)
    
    # Ticks
    ax.tick_params(direction='in', length=4)
   # ax.set_xlim([-0.05,5.08])
    
    # Tight layout and display
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_C(ns, C, a):
    C = C/C[0]; a = a/a[0] # Normalize pesticide (C) adsorbent (a) concentrations to initial value
    indx = np.argmin(C) # Minimum concentration
    ns = ns - ns[0]
    ns = ns*5/3600
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(ns, a, label = r"$|\Omega|\mathcal{F}_\Omega(A)$", linewidth = 1)
    ax.plot(ns, C, label = r"$\mathcal{F}_\Omega(C)$")
    ax.axvline(ns[indx], ymax = 0.95*a[indx], linestyle=':', linewidth = 1.0, color = 'black', label = r"$t = t_{min}$")
    
    ax.set_xlabel(r'time since $t_{ads}$ [h]', fontsize=12)
    ax.set_ylabel(r"Value relative to $C_0$ and $A_0$" , fontsize=12)
    ax.set_title(r'Relative substance concentrations', fontsize=13)
    
    # Grid and legend
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(frameon=False, fontsize=10)
    
    # Ticks
    ax.tick_params(direction='in', length=4)
    
    # Tight layout and display
    fig.tight_layout()
    plt.show()
    return
    
def plot_C_sep(ns,C,a):
    from matplotlib.ticker import MaxNLocator
    C = C/C[0]; a = a/a[0]
    indx = np.argmin(C)
    ns = ns - ns[0]
    ns = ns*5/3600
    fig, ax = plt.subplots(2,1,figsize=(6, 4), dpi=300,sharex=True)

    ax[0].plot(ns, a, label = r"$| \Omega_{out} |$ $\mathcal{F}_{\Omega_{out}} (A)$")
    ax[1].plot(ns, C, label = r"$\mathcal{F}_{\Omega_{out}}(C)$",color="orange")
    ax[0].axvline(ns[indx], ymax = 0.95*a[indx], linestyle=':', linewidth = 1.0, color = 'black', label = r"$t = t_{min}$")
    ax[1].axvline(ns[indx], ymax = 0.13, linestyle=':', linewidth = 1.0, color = 'black', label = r"$t = t_{min}$")

    ax[1].set_xlabel(r'Time since $t_{ads}$ [h]', fontsize=10)
    ax[0].set_ylabel(r"Value relative to $A_0$" , fontsize=10)
    ax[1].set_ylabel(r"Value relative to $C_0$" , fontsize=10)
    #ax[0].set_title(r'Relative substance concentrations', fontsize=13)
    fig.suptitle(r'Relative substance concentrations', fontsize=13)
    ax[1].set_ylim([0.15,1.06])

    # Grid and legend
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(frameon=False, fontsize=10)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(frameon=False, fontsize=10)

    # Ticks
    ax[0].tick_params(direction='in', length=4)
    ax[1].tick_params(direction='in', length=4)
    #ax[1].set_yticks(np.linspace(0.75, 1, 4))
    #ax[1].yaxis.set_major_locator(MaxNLocator(nbins=4)) 
    # Tight layout and display
    fig.subplots_adjust(hspace=0.3,bottom=0.116)
    #fig.tight_layout()
    plt.show()
    return fig, ax

def plot_C2(ns, C, a):
    C = C/C[0]; a = a/a[0]
    indx = np.argmin(C)
    ns = ns - ns[0]
    ns = ns*5/3600
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(ns, a, label = r"$| \Omega_{out} |$ $\mathcal{F}_{\Omega_{out}} (A)$", linewidth = 1)
    ax.plot(ns, C, label = r"$\mathcal{F}_{\Omega_{out}}(C)$")
    ax.axvline(ns[indx], ymax = 0.9*a[indx], linestyle=':', linewidth = 1.0, color = 'black', label = r"$t = t_{min}$")

    ax.set_xlabel(r'time since $t_{ads}$ [h]', fontsize=12)
    ax.set_ylabel(r"Value relative to $C_0$ and $A_0$" , fontsize=12)
    ax.set_title(r'Relative substance concentrations', fontsize=13)

    # Grid and legend
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(frameon=False, fontsize=10)

    # Ticks
    ax.tick_params(direction='in', length=4)

    # Tight layout and display
    fig.tight_layout()
    plt.show()
    return  
