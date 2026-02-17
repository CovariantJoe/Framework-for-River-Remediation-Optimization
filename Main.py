"""
This script is the main executable in the project: Numerical study of optimal pesticide remediation in the Santiago River with a 2D model

The objective is to optimize the initial conditions of an adsorbent by controlling the transport and adsorption simulation from here

--------------------------------- Assumptions ---------------------------------
- THIS SCRIPT WAS ONLY TESTED IN A LINUX OPERATING SYSTEM 
- There are .h5 files in the same directory as the script containing the initial state of the pesticide, and the velocity
    (created with EvolvePesticide.py, which in turn needs .h5 files for the state of the velocity field) 
- The file mesh.xml contains the mesh used to create the velocity field and evolve the pesticide alone

--------------------------- If using different mesh ------------------------
Some mesh coordinates are hard coded to define the different regions.
Read this section inside Solver.py and velocity.py for an overview
---------------------- Optimization Variables ----------------------

The 3 fundamental variables to optimize are: 
A_0 : float
Initial adsorbent dose 
y_0 : float
Initial adsorbent placement height in river, 
w_0 : float between 0 and 1
Initial adsorbent placement width within river at given y_0 internally leading to a x_0, y_0 pair

As defined in DATA.py, substance pairs follow indices throughout the project as follows:
0 : Biochar / Glyphosate
1 : Egg shell / Malathion
2 : Montmorillonite / Aldrin
3 : Leaves / Malathion

The mass unit is grams [g], the rest of units are SI units (e.g concentration g/m^3). 

---------------------- Dependencies --------------------------------
Tested dependencies:
    Python 3.10.16
    dolfin 2019.1.0
    scipy 1.15.2    
"""

from DATA import *
import time
from datetime import datetime
from Solver import *
from fenics import assemble, dx, Constant
import nlopt
from scipy.optimize import minimize
import subprocess
import sys

if not sys.platform.startswith("linux"):
    raise Exception("The project was only tested on a native Linux system. If you acknowledge this comment this line in Main.py")

# ---------------------------  GENERAL CONFIG ----------------------------------------
mode = "LOCAL" # LOCAL or GLOBAL optimization mode (L-BFGS-B or nlopt.GN_DIRECT_L, respectively)
ONLY_OUTLET = False # (False or True) Whether to conduct optimization only in the defined outlet subdomain region (True), or the whole mesh (False). This region is defined in Solver.py
OPT = 1 # Optimize for either 1 (A_0), 2 (A_0, w_0) or 3 (y_0, A_0, w_0) variables
SOLVE_FOR =  0 # Integer from -1 to 3: Substance pair to solve for (explained in DATA.PY and above) SOLVE_FOR = -1 will solve for every value (0-3) in sequence
weight = 0.2 # Optimization weight omega between 0 and 1 as defined in Eq. 15 of the article 

a0s = [4673,594,598,449] # (Optional) array for initial A_0 values when using SOLVE_FOR = -1 in substance order from 0 to 3

# Select boundaries ( format: (min,max) ) for each optimized variable:
A_bounds = (50,5000)
y_bounds = (-810,-70)
w_bounds = (0.15,0.85) # Suggested to keep (0.15,0.85) for w_0 to ensure the Gaussian projects entirely to the mesh

# ------------------------- CONFIG LOCAL OPTIMIZATION MODE ---------------------------
# Suggested to use the output from global mode, may change max_eval below if needed
y0 = -800 # Initial y_0
w0 =  0.5 # Initial Gaussian placement width at y_0, value in [0,1] 
a0 =  2500  # Initial adsorbent A_0. For units see above

ftol = 2e-3 # Tolerance in relative reduction of cost for convergence (severely impacts convergence time, suggested >= 2e-3)
pgtol = -1 # Tolerance in relative reduction of gradient for convergence. Set -1 to use internal default, suggested relatively small

# ------------------------- CONFIG GLOBAL OPTIMIZATION MODE --------------------------
# Initial values will be at center of parameter space, changing value assumed unimportant due to global nature of this optimization 
# If need to change, change "center" inside opti.optimize() below in the correct variable order to new initial values
max_eval = 30 # Maximum simulations for the global optimization
# -----------------------------------------------------------------------------------

print(f"Commencing program, see opt-{mode.lower()}.log to follow optimizer status")

if OPT == 3:
    bounds = [y_bounds,A_bounds, w_bounds] 
elif OPT == 1:
    bounds = [A_bounds]
else:
    bounds = [A_bounds, w_bounds] 
    
def log(message, mode = "GLOBAL"):
    now = datetime.now()
    T0 = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(Path,f"opt-{mode.lower()}.log")
    try:
        with open(filename, "a") as f:
            f.write("[ " + T0 + " ] " + f"{message}" + "\n")
    except FileNotFoundError:
            try:
                fd = os.open(filename, os.O_CREAT)
                os.close(fd)
                with open(filename, "a") as f:
                    f.write("[ " + T0 + " ] " + f"{message}" + "\n")
            except:
                raise Exception(f"Fatal error, couldn't find nor create and write to log. Perhaps check permissions inside {Path}")

    return 0

def unscale(u, boundaries): # Take array of normalized variables, unscale to physical units
    lower =np.array([ x[0] for x in boundaries ])
    upper =np.array([ x[1] for x in boundaries ])
    return lower + np.array(u)*(upper - lower)

def scale(x, boundaries): # Take array of physical variables, normalize to boundaries
    lower =np.array([ x[0] for x in boundaries ])
    upper =np.array([ x[1] for x in boundaries ])
    return (np.array(x) - lower)/(upper - lower) 

def run_fem(a0, w = w0, y = y0): # Call the FEM simulation with given initial conditions for adsorbent a0, w0, y0. The function called is defined in Solver.py
    total_c, C, total_a = PollutionSolver(Adsorbents[SOLVE], Pollutants[SOLVE], MeshData, FROM_CHECKPOINT = True, ONLY_OUTLET = ONLY_OUTLET, a_0 = a0, y = y, width = w)
    return np.min(total_c)

class LocalObjective:
    '''
    Local optimization happens inside this class.
    
    Variables:
        Most importantly "x" are the variables in physical units while "u" are variables normalized to their bounds
        
    Functions:
        fun - Run simulation, return cost function, prevents simulations from repeating
        grad - Find gradient by calling fun using central differences near point u
        fun_grad -  Actual function passed to optimizer returning pair (cost, grad) at u, everything normalized
        
    '''
    def __init__(self, run_fem, pollutant, boundaries, OPT, weight):
        if len(boundaries) != OPT:
            raise Exception(f"Boundaries dont correspond to number of variables to optimize ({OPT}) in format [ (min, max) , (min, max), ...]")
        elif OPT == 1 or OPT == 2:
            self.MIN_A = boundaries[0][0]; self.MAX_A = boundaries[0][1]
        else:
            self.MIN_A = boundaries[1][0]; self.MAX_A = boundaries[1][1]
            
        self.run_fem   = run_fem
        self._f_cache  = {} # Cache to store previous simulation results to avoid useless repetition
        self._g_cache  = {} # Cache to store previous simulation gradients to avoid useless repetition
        self.mode = mode
        self.lower =np.array([ x[0] for x in boundaries ])
        self.upper =np.array([ x[1] for x in boundaries ])
        self.counter = 0 # Counter of the number of simulations
        self.pollutant = pollutant
        self.weight = weight # Optimization weight \omega
        self.MIN_c = pollutant["Conc"]*0.00001; self.MAX_c = pollutant["Conc"]*0.999 # min/max pesticide concentration values (0.0001 or 0.999 of DATA.py value) for normalization to [0,1]
        self.boundaries = boundaries
        self.OPT = OPT
        
    def _key(self, x):
        return tuple(np.round(x, 6)) # Round simulation results/params to use in the cache above

    def fun(self, u): # Cost function at u
        x = unscale(u, self.boundaries)
        k = self._key(x)
        self.counter = self.counter + 1
        if k not in self._f_cache:
            if self.OPT == 3:
                mean_c   = self.run_fem(x[1], x[2], x[0])
            elif self.OPT == 2:
                mean_c = self.run_fem(x[0], x[1])
            else:
                mean_c   = self.run_fem(x[0])
        # Cost function evaluations
            if self.OPT == 3:
                cost = (1-self.weight)*(mean_c - self.MIN_c)/(self.MAX_c - self.MIN_c) + self.weight*(x[1] - self.MIN_A)/(self.MAX_A - self.MIN_A)
            elif self.OPT == 1 or self.OPT == 2:
                cost = (1-self.weight)*(mean_c - self.MIN_c)/(self.MAX_c - self.MIN_c) + self.weight*(x[0] - self.MIN_A)/(self.MAX_A - self.MIN_A)
            self._f_cache[k] = cost
            self.logcall(f' Sim: {self.counter} - Normalized Params: {u} - Params: {x} - Cost func: {cost} - mean conc / DATA.py conc: {mean_c/self.pollutant["Conc"]}')
        else:
            self.logcall(f"Skipping sim {self.counter} with params: {x}. Result was found before")
        return self._f_cache[k]

    def grad(self, u): # Gradient of cost function at u
        x = unscale(u, self.boundaries)
        k = self._key(x)
        
        if k in self._g_cache:
            self.logcall(f"Gradient at {x} was found before, skipping calculation")
            return self._g_cache[k]
        
        self.logcall(f"Evaluating gradient at {x}")
        g  = np.zeros_like(x)
        # Gradient steps in physical units, sensitive to change!
        # Follows the same order as bounds in config. section
        if self.OPT == 3:
            abs_steps = [-4, -25, 0.025]
        elif self.OPT == 1:
            abs_steps = [-25]
        else:
            abs_steps = [-25, 0.025]
            
        for i in range(len(x)):
            h2 = abs_steps[i]
            xp = x.copy(); xm = x.copy()
            xp[i] += h2; xm[i] -= h2  # Plus and minus steps for central diff eval
            if self.OPT == 3 and i == 0 and np.abs(x[0]-self.lower[0]) < 1:
                g[i] = 0 # Gradient component = 0 if very close to lower boundary of y_0 
                continue
            # Actual gradient evaluation, calling cost function "fun" for plus and minus steps near x (scaled to u)
            g[i] = (self.fun(scale(xp, self.boundaries)) - self.fun(scale(xm, self.boundaries))) / (2*h2)
                        
        self._g_cache[k] = g*(self.upper - self.lower) # Normalize gradient for optimizer to see

        self.logcall(f"Gradient at {x} was: {g} - Normalized to {self._g_cache[k]}")
        return self._g_cache[k]
    
    def logcall(self, message):
        log(message,self.mode)
        
    def fun_grad(self, u):
        return self.fun(u), self.grad(u)

class GlobalObjective:
    '''
    Global optimization happens inside this class.
    Notice the global optimizer used is gradient-free
 
    Variables:
        Most importantly "x" are the variables in physical units while "u" are variables normalized to their bounds
        
    Functions:
        objective - Run simulation, return cost function, prevents simulations from repeating
        global_search - Define the problem for nlopt 
       
    '''
    def __init__(self, boundaries, pollutant, weight):
        self.boundaries = boundaries
        self.pollutant = pollutant
        self.weight = weight
        self.counter = 0 # Simulation counter
        self.cache_fun = {}  # Cache to store previous simulation results to avoid useless repetition
        self.mode = "GLOBAL"
        
    def key(self, x):
        return tuple(np.round(x, 6))
    
    def logcall(self, message):
        log(message,self.mode)
    
    def objective(self, u, b = []): # Cost function
        x = unscale(u, self.boundaries)
        k = self.key(x)
        self.counter += 1
        self.MAX_c = self.pollutant["Conc"]*0.999; self.MIN_c = self.pollutant["Conc"]*0.00001# 0.001
            
        if k in self.cache_fun:
            log(f"Skipping sim {self.counter} with params: {x}. Result is cached")
            return self.cache_fun[k]

        if len(x) == 2: # (A_0, w_0)
            mean_c = run_fem(x[0], x[1])
            cost = (1-self.weight)*(mean_c - self.MIN_c)/(self.MAX_c - self.MIN_c) + self.weight*(x[0] - self.MIN_A)/(self.MAX_A - self.MIN_A)
        elif len(x) == 3: # (y_0, A_0, w_0)
            mean_c = run_fem(x[1], x[2], x[0])
            cost = (1-self.weight)*(mean_c - self.MIN_c)/(self.MAX_c - self.MIN_c) + self.weight*(x[1] - self.MIN_A)/(self.MAX_A - self.MIN_A)
        elif len(x) == 1: # (A_0)
            mean_c = run_fem(x[0])
            cost = (1-self.weight)*(mean_c - self.MIN_c)/(self.MAX_c - self.MIN_c) + self.weight*(x[0] - self.MIN_A)/(self.MAX_A - self.MIN_A)

        self.cache_fun[k] = cost
        self.logcall(f' Sim: {self.counter} - Normalized Params: {scale(x,self.boundaries)} - Params: {x} - Cost func: {cost} - mean conc / DATA.py conc: {mean_c/self.pollutant["Conc"]}')

        return self.cache_fun[k]
    
    
    def global_search(self, N_OPT):
        '''
        Define the parameters of the global search
        '''
        global w0; global y0
        opti = nlopt.opt(nlopt.GN_DIRECT_L, N_OPT)
        opti.set_min_objective(self.objective)
        opti.set_maxeval(max_eval)
        bounds = self.boundaries
        
        if len(bounds) != N_OPT:
            raise Exception(f"Boundaries dont correspond to number of variables to optimize ({OPT}) in format [ (min, max) , (min, max), ...]")
        elif N_OPT == 1 or N_OPT == 2:
            self.MIN_A = bounds[0][0]; self.MAX_A = bounds[0][1]
        else:
            self.MIN_A = bounds[1][0]; self.MAX_A = bounds[1][1]
            
        # Center of parameter space to use as initial guess
        center = [(b[0]+b[1])/2 for b in bounds]
        # N_OPT = 1: (A_0), 2: (A_0, w_0) or 3: (y_0, A_0, w_0) 
        if N_OPT == 1:
            w0 = (w_bounds[0]+w_bounds[1])/2; y0 = (y_bounds[0]+y_bounds[1])/2
            opti.set_lower_bounds([0])
            opti.set_upper_bounds([1])
            xopti = opti.optimize(scale(center,bounds))
        if N_OPT == 2:
            w0 = center[1]; y0 = (y_bounds[0]+y_bounds[1])/2
            opti.set_lower_bounds([0, 0])
            opti.set_upper_bounds([1, 1])
            xopti = opti.optimize(scale(center,bounds))
        elif N_OPT == 3:
            w0 = center[-1]; y0 = center[0]
            opti.set_lower_bounds([0, 0, 0])
            opti.set_upper_bounds([1, 1, 1])
            xopti = opti.optimize(scale(center,bounds))
            
        return xopti


if mode != "LOCAL" and mode != "GLOBAL":
    raise Exception("Must choose for mode either 'GLOBAL' or 'LOCAL' optimization")
if OPT not in [1,2,3]:
    raise Exception("Can only optimize 1 to 3 vars")
if SOLVE_FOR not in [-1,0,1,2,3]:
    raise Exception("Invalid value for variable SOLVE_FOR")
if OPT == 3 and bounds[0][0] > bounds[0][1]:
    raise Exception("Boundaries for y_0 are not (min, max), perhaps check sign")
if not isinstance(OPT,(int)) or not isinstance(SOLVE_FOR,(int)):
    raise Exception("OPT and SOLVE_FOR should be integers")
if weight < 0 or abs(weight) > 1:
    raise Exception("Optimization weight not in [0,1]")
if not isinstance(ONLY_OUTLET,(bool)):
    raise Exception("Option ONLY_OUTLET should be True or False")

if SOLVE_FOR == -1:
    log("Selected solving for every adsorbent/pesticide pair in DATA.py",mode)
for SOLVE in [0,1,2,3]:
    SOLVE = SOLVE_FOR if SOLVE_FOR != -1 else SOLVE
    a0 = a0 if SOLVE_FOR != -1 else a0s[SOLVE]
    if OPT == 3:
        theta0 = scale([y0, a0, w0], bounds) # Normalized initial guess
        bnds = [[0,1],[0,1],[0,1]]
    elif OPT == 1:
        theta0 = scale([a0], bounds)     # Normalized initial guess
        bnds = [[0,1]]
    else:
        theta0 = scale([a0, w0], bounds)    # Normalized initial guess
        bnds = [[0,1],[0,1]]
    
    if mode == "LOCAL":
                 
            problem = LocalObjective(run_fem, Pollutants[SOLVE], bounds, OPT, weight)
            log(f"LOCAL Optimizing for {Adsorbents[SOLVE]['Name']} adsorbing {Pollutants[SOLVE]['Name']} with bounds: {bounds} and normalized initial guess: {theta0}", mode)
            log("Format of output per simulation is: Sim. count - Initial params normalized to [0 , 1] - Initial params in physical units - Cost function - final pesticide concentration / initial concentration set in DATA.py",mode)
            if pgtol > 0:
                res = minimize(problem.fun_grad, x0 = theta0, method = 'L-BFGS-B', jac = True, bounds =  bnds, tol = ftol,
                               options = {"maxiter":6, "pgtol": pgtol})
            elif pgtol == -1:
                res = minimize(problem.fun_grad, x0 = theta0, method = 'L-BFGS-B', jac = True, bounds =  bnds, tol = ftol,
                               options = {"maxiter":6})
            else:
                raise Exception("Incorrect (<= 0 and not -1) value for pgtol")
            res.x = unscale(res.x, bounds)
            log(res,"LOCAL")
            print(f"RESULT: {res}")
    
    elif mode == "GLOBAL":
        problem = GlobalObjective(bounds, Pollutants[SOLVE], weight)

        log(f"GLOBAL Optimizing for {Adsorbents[SOLVE_FOR]['Name']} adsorbing {Pollutants[SOLVE_FOR]['Name']} with bounds: {bounds}", mode)
        log("Format of output per simulation is: Sim. count - Initial params normalized to [0 , 1] - Initial params in physical units - Cost function - final pesticide concentration / initial concentration set in DATA.py",mode)
        
        xopt = problem.global_search(OPT)
        log(f"RESULT: {xopt}", "GLOBAL")
        print(f"RESULT: {xopt}")
    if SOLVE_FOR != -1:
        break
