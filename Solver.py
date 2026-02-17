"""
Finite element script to find the evolution and interaction of pesticide alone or pesticide + adosorbent.
This script is internally called by Main.py and EvolvePesticide.py

---------------------- Assumptions ---------------------------------
- THIS SCRIPT WAS ONLY TESTED IN A LINUX OPERATING SYSTEM 
- There are .h5 files in the same directory as the script containing the full 
   evolution of the velocity field during the same time as intended here
- The file mesh.xml contains the same mesh used to evolve the velocity field
- The mass unit is grams [g], the rest of units are SI units (e.g concentration g/m^3). 
- Molecular weights: Glyphosate: 169.07 g/mol, Malathion: 330.4 g/mol, Aldrin: 364.9 g/mol

---------------------- If using different mesh ---------------------------------
Some mesh coordinates are hard coded to define the different regions.
Examples where to manually change values in this script if using another mesh:
- classes: InletBoundary, OutletBoundary, OutletRegion
- functions: Inlet, outlet, PollutionWall
- variables: inside PollutionSolver: D and D_ads. N_TIME_STEPS_CHKP is an empirical choice of timesteps
             that the simulation takes to fill the river to 99% of target concentration in our mesh/velocity
             If using other mesh, also change the value of t_vel and n0_vel reading their explanation below 
---------------------- Dependencies --------------------------------
Tested dependencies:
    Python 3.10.16
    dolfin 2019.1.0
"""

from DATA import *
import os
import numpy as np
import fenics as fe
from dolfin import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from ufl import conditional, ge, And, max_value
import sys

set_log_level(LogLevel.ERROR)
if not sys.platform.startswith("linux"):
    raise Exception("The project was only tested on a native Linux system. If you assume this risk comment this line in Solver.py")

'''
Several classes to mark different regions of the mesh. Some may not be necessary
'''
class PollutionRegion(SubDomain):
    def inside(self, x, on_boundary):
        return PollutionWall(x,on_boundary)

class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        isOut = outlet(x,on_boundary)
        return isOut

class InletBoundary(SubDomain):   
    def inside(self, x, on_boundary):
        return Inlet(x,on_boundary)
    
class OutletRegion(UserExpression):
    '''
    To define the outlet region (not to confuse with the outlet wall!), a vector
    transformation is performed to a different base. Then for every point in the mesh
    [ x[0],x[1] ], it's component in this basis vector is evaluated. 
    In this particular case, components > -16 are within the region
    
    Needless to say the function eval_cell needs to change if you use a different mesh.

    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval_cell(self, values, x, cell):
            base = np.array([-18.262, -56.346])
            M = np.matrix([[0.81525954,-0.36584678],[ 0.57909574,  0.9306751]])
            component = np.matmul(M, base-np.array( [ x[0],x[1] ] ))[0,0] 
            if component > -16:
                values[0] = 1.0   # inside region
            else:
                values[0] = 0.0   # outside
    def value_shape(self):
        return ()
    
class AdsorbentRegion(UserExpression):
    '''
    Very important class to obtain the pair (x_0, y_0) of initial coordinates
    for the adsorbent, and create their initial Gaussian distribution
    '''
    def __init__(self, a0, y, width, mesh, **kwargs):
        super().__init__(**kwargs)

        # Force 0.1 < width < 0.9 to ensure all A_0 adsorbent is actually projected to mesh
        width = 0.1 if float(width) < 0.1 else width
        width = 0.9 if float(width) > 0.9 else width

        def isInMesh(mesh,point):
            # Simple test of whether point is inside mesh based on cell_id of first collision
            tree = mesh.bounding_box_tree()
            if tree.compute_first_entity_collision(Point(point)) > mesh.num_cells():
                raise Exception(f"The optimizer seems to have intended to place the adsorbent outside the mesh at {self.point}. Check especially that y_bounds is set correctly. \n This code should prevent that an intended position (inside) is rounded to outside")
            return 0
        
        def AdsRegion(y, width, mesh):
            # This may fail if your mesh only consists of boundary points
            temp_points = list(mesh.coordinates())
            boundary_mesh = BoundaryMesh(mesh, "exterior")
            boundary_points = boundary_mesh.coordinates()
            all_points = []
            for i in range(0,len(temp_points)):
                p = temp_points[i]
                if not np.any(np.all(boundary_points == p, axis=1)):
                    all_points.append(p)
                    
            def dist(p1,p2):
                return np.sqrt((p1[1] - p2[1])**2 + (p1[0] - p2[0])**2)
            
            def n_closest(points, threshold):
                pts = []
                N = len(points)
                for i in range(N):
                    temp = np.abs(points - y)
                    loc = np.argmin(temp[:,1])
                    
                    pt = points[loc]
                    if np.abs((pt[1] - y)) > 5 and threshold >2:
                        continue
                    
                    pts.append(pt)
                    
                    if len(pts) >= threshold:
                        return pts
                    else:    
                        points = np.vstack([points[:loc], points[loc+1:]])
                    
                return pts
                    
            pts = n_closest(boundary_points, 2)
            d = dist(pts[0], pts[1])
            
            pts_all = n_closest(np.array(all_points), 100)
            
            if pts[1][0] < pts[0][0]:
                pts = pts[::-1]
                
            dists = []
            for pt in pts_all:
                dists.append(dist(pt, pts[0]) + (pt[1] - y) )
                
            loc = np.argmin(np.abs(dists - d*width))
            point = pts_all[loc]
            
            err = np.abs(dist(point,pts[0])-d*width)
            if err  > 5:
                print(f"Warning, adsorbent projected {err} meters from intended release point")
            return point
            
        self.point = AdsRegion(float(y), float(width), mesh)
        isInMesh(mesh,self.point)
        self.a0 = float(a0);
        V0 = FunctionSpace(mesh, "DG", 0)
        h_ufl = CellDiameter(mesh)
        h = project(h_ufl, V0)
        self.sigma = h(self.point) # Use local mesh resolution as sigma in gaussian distribution
        
    def eval_cell(self, values, x, cell):
        # Gaussian distribution for the adsorbent
        quant = self.a0*np.exp(-( (self.point[0] - x[0])**2 + (self.point[1] - x[1])**2 )/(2*self.sigma**2))
        values[0] = quant/(2*np.pi*self.sigma**2) if quant > 1e-4 else 1e-16

    def value_shape(self):
        return ()
    
def Inlet(x, on_boundary):
    # Define the inlet region. Notice MeshData["DBC"][x][y] from DATA.py introduces coordinates from example mesh you need to change if using another mesh
    return on_boundary and x[1] > MeshData["DBC"][1][1]-1e-4 and x[1] < MeshData["DBC"][0][1]+1e-4 and x[0] < MeshData["DBC"][1][0] + 1e-4

def outlet(x, on_boundary):
    # Define the outlet region. Notice MeshData["DBC"][x][y] from DATA.py introduces coordinates from example mesh you need to change if using another mesh
    return on_boundary and x[1]> MeshData["DBC"][3][1]-1e-4 and x[0] > MeshData["DBC"][2][0]-1e-3

def PollutionWall(x,on_boundary):
    # Define the inlet region. Notice some coordinates from the example mesh you need to change if using another mesh
    if x[0] < 121:
        return on_boundary and x[1] < -368 and not Inlet(x,on_boundary)
    else:
        return on_boundary and x[1] < -513 and not Inlet(x,on_boundary)

def max_velocity_index(xdmf_file, u, name="u"):
#    Returns the maximum valid checkpoint index in the velocity simulation using exponential + binary search.
#    fenics doesn't supply another way to do this
    def exists(n):
        try:
            xdmf_file.read_checkpoint(u, name, n)
            return True
        except RuntimeError:
            return False

    low = 0
    high = 1
    while exists(high):
        low = high
        high *= 5

    while low + 1 < high:
        mid = (low + high) // 2
        if exists(mid):
            low = mid
        else:
            high = mid

    return low

    
def TestFiles(FROM_CHECKPOINT,Adsorbent):
    # Function to test whether the required files exist inside the directory Fields before running. Some specific filenames are assumed

    required = ["u.xdmf","u.h5"] if not FROM_CHECKPOINT else ["u.xdmf","u.h5", f'{Adsorbent.split(" ")[0]}-Checkpoint.h5']
    previous = [f'C-{Adsorbent.split(" ")[0]}.xdmf',f'q-{Adsorbent.split(" ")[0]}.xdmf',f'a-{Adsorbent.split(" ")[0]}.xdmf',f'C-{Adsorbent.split(" ")[0]}.h5',f'q-{Adsorbent.split(" ")[0]}.h5',f'a-{Adsorbent.split(" ")[0]}.h5']
    fields_dir = os.path.join(Path, "Fields")
    if not os.path.isdir(fields_dir):
        raise Exception(f"Directory not found: {fields_dir}. The program expects this directory containing some fields and needs to save some output here")

    for filename in required:
        file_path = os.path.join(fields_dir, filename)
        if not os.path.isfile(file_path):
            if not FROM_CHECKPOINT:
                raise Exception(f"Missing: {file_path}. The program expects a velocity field over time in u.xdmf and u.h5, you can generate with velocity.py (takes hours)")
            else:
                raise Exception(f"Missing: {file_path}. The program expects 1) a velocity field over time in u.xdmf and u.h5, you can generate with velocity.py (takes hours) or use the example \n 2) a checkpoint (final state of evolution without adsorbent) for the selected pesticide's concentration, you can generate it with EvolvePesticide.py")

    # Remove the previous solution before running the simulation to avoid file corruption
    fields_dir = os.path.join(Path, "Fields/Solutions")
    for filename in previous:
        file_path = os.path.join(fields_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return 0

def PollutionSolver(adsorbent, pollutant, MeshData, mode = "fast", FROM_CHECKPOINT = False, ONLY_OUTLET = False, MAX_CONC = 0.999, N_TIME_STEPS_CHKP = 5200, a_0 = 600, y = -440, width = 0.5, InletClass = InletBoundary, OutletClass = OutletBoundary, Config = [5,5,600,np.nan,""]):
    '''
    Main function in this script to solve the simultaneous transport / adsorption problem, or the pesticide transport problem alone

    Variables:
       - C, q, a are respectively: Concentration of pesticide, adsorbed pesticide, and adsorbent concentration. The output files under Fields directory follow this naming convention.
    
    Input:
       - adsorbent: Dictionary of parameters for a single adsorbent as defined in DATA.py
       - pollutant: Dictionary of parameters for a single pesticide as defined in DATA.py
       - MeshData: Dictionary of mesh info with path to xml file, or a dolfin mesh object 
       - mode ("fast", "normal"). Determines whether to write every single time step to a file (normal),
         or just the final state (fast) often at at minimum pollution concentration, mild impact on the runtime
       - FROM_CHECKPOINT (True or False): determines whether the program runs to evolve the pesticide alone up to a checkpoint time, 
         or if it begins from a checkpoint of pesticide concentration and evolves with an adsorbent from that time forward.
       - ONLY_OUTLET (True or False): determines whether the program returns concentrations only in the outlet region defined by OutletRegion class,
         or in the complete mesh domain. This materially changes the optimization problem from all the mesh to only the region defined by OutletRegion
       - MAX_CONC (float, 0 to 1): When using with FROM_CHECKPOINT = False, simulation attempts to fill the river to this percentage of Conc in DATA.py 
       - a_0, y, width (floats): Initial conditions for the adsorbent leading to (x_0, y_0) centered Gaussian distribution through class AdsorbentRegion above.
       - InletClass: A class marking the inlet for the mesh in MeshData, see example at top of this file
       - OutletClass: A class marking the outlet for the mesh in MeshData, see example at top of this file
       - Config: This is best shown in lines 313,314 of Benchmark.py, or seeing where each value is assigned below

    Output:
       - FROM_CHECKPOINT = True: np.array(avg_c), C, np.array(avg_a). An array over time of the average concentration of pesticide measured in the domain selected by ONLY_OUTLET. The simulation breaks after this quantity reaches a minimum, 
           C is the fenics pesticide field object to plot with fe.plot(), and np.array(avg_a) is an array over time of the average concentration of adsorbent measured in the domain selected by ONLY_OUTLET
         FROM_CHECKPOINT = False: final_avg_c, C. The final value of the average concentration of pesticide measured in the domain selected by ONLY_OUTLET, and the fenics pesticide field object to plot with fe.plot()

    '''
    from ufl import conditional, And, Or, Not
    
    if mode != "normal" and mode != "fast":
        raise Exception("Mode is either 'normal' or 'fast'")
    elif MAX_CONC < 0 or MAX_CONC > 1 or not isinstance(MAX_CONC,(float,int)):
        raise Exception("Please set parameter MAX_CONC to value in [0,1]")    
    elif not isinstance(FROM_CHECKPOINT,(bool)) or not isinstance(ONLY_OUTLET,(bool)):
        raise Exception("Please set parameters FROM_CHECKPOINT and ONLY_OUTLET to a bool")  
        
    try:
        # Test provided mesh .xml works, and Path exists
        if isinstance(MeshData,(dict)):
            mesh = Mesh(MeshData["xml"])
        elif "dolfin.cpp.generation" in str(type(MeshData)):
            mesh = MeshData
        else:
            raise Exception("Argument MeshData is neither a mesh nor a dictionary leading to a mesh file in .xml format")
        x = SpatialCoordinate(mesh)
        os.path.exists(Path)
        N = mesh.hash() == 16975793265595423454
    except:
        raise Exception(f"Please in DATA.py make sure Path is correct \n Also ensure either MeshData['xml'] is an existing mesh file, or argument MeshData is a dolfin mesh already")
    
    # Note on the mesh volumes: vol = 96394.12737347104 (ONLY_OUTLET = False), 1501.9820507362065 (ONLY_OUTLET = True), ratio: 64
    
# --------------------------------------------------------- Parameters -------------------------------------------------------
    # This is an empirical choice for our case, 5200 steps fills the river from the inlet to about 99% of Conc from DATA.py
    # N_TIME_STEPS_CHKP = 5200 # This value is now an argument when calling the function

    dt = Config[0] # Time step to use in the evolution of pesticides and adsorbents
    dt_vel = Config[1] # Time step utilized when running velocity.py to generate the velocity field
    n0_vel = Config[2] # These two are the time and iteration number n0_vel of the velocity simulation that will act as reference initial time/iteration here
    t_vel = n0_vel*dt_vel # so set according to your needs (e.g set to when the velocity field is ready to be used here because it has stabilized)
    factor = dt/dt_vel if dt/dt_vel > 1 else 1 # Factor to skip velocity time steps if dt > dt_vel
    

    # Some code for filenames and directories
    TestFiles(FROM_CHECKPOINT, adsorbent["Name"])
    FILE_u = os.path.join(Path, "Fields/u.xdmf")
    FILE_C = os.path.join(Path, "Fields/Solutions/" + f'C-{adsorbent["Name"].split(" ")[0]}.xdmf') 
    FILE_CHECKPOINT = os.path.join(Path, f'Fields/{adsorbent["Name"].split(" ")[0]}-Checkpoint.h5')
    chkp = 1 if FROM_CHECKPOINT else 1e-15 # Turn on and turn off adsorption if just creating the checkpoint
    FILE_q = os.path.join(Path, "Fields/Solutions/" + f'q-{adsorbent["Name"].split(" ")[0]}.xdmf')
    FILE_a = os.path.join(Path, "Fields/Solutions/" +  f'a-{adsorbent["Name"].split(" ")[0]}.xdmf')
    
# ---------------------------------------------------- Parameters from DATA.py-------------------------------------------------
    # Se below in this section to understand Dm_pol and Dm_ads
    Dm_pol = pollutant["Dm"]
    Dm_ads = 1e-13 # This contribution comes from the Einstein relation and is assumed negliglible compared to Dm_pol, which is the case for the substances tested
    F = 1e-15 # Source term for the equation for testing
    aL = 0.83*np.log10(100)**2.414 # Dispersivity to field scale relation by Xu and Eckstein
    Conc = Constant(pollutant["Conc"]) # Target pesticide concentration set in DATA.py
    
    model = adsorbent["Model"] # Isotherm
    k1 = chkp*adsorbent["Values"]["k1"] # Adsorption rate
    theta = 1.0
    rho = adsorbent["Transport"]["rho_s"]*(1-adsorbent["Transport"]["porosity"]) # bulk density
    
    if model == 1: # Isotherm 1:
        Q = chkp*adsorbent["Values"]["qmax"]
        b = adsorbent["Values"]["b"]
    elif model == 2: # Isotherm 2:
        K = chkp*adsorbent["Values"]["k"]
        n_val = adsorbent["Values"]["n"]
    
    '''
    Values for D (pest) and D_ads (adsorbent). Quoting from article "Moreover, we model each Di following the literature with a molecular (Dm,i) and longitudinal (aL)
    component scaled by the average speed (|u|): Di = Dm,i + aL|u|."  Here, we change the value of |u| with a conditional depending on the region in the mesh
    with values that were measured manually (e.g for y >= -36 we use |u| = 0.209 ). These coordinates should change with a different mesh
    '''

    if N:
        D = conditional(ge(x[1], -36), Constant(Dm_pol + aL*0.209), conditional(gt(x[1], -81.42),  Constant(Dm_pol + aL*0.1432), conditional(ge(x[1],-234), Constant(Dm_pol + aL*0.075), Constant(Dm_pol + aL*0.04)) )  )
        D_ads = conditional(ge(x[1], -36), Constant(Dm_ads + aL*0.209), conditional(gt(x[1], -81.42),  Constant(Dm_ads + aL*0.1432), conditional(ge(x[1],-234), Constant(Dm_ads + aL*0.075), Constant(Dm_ads + aL*0.04)) )  )
    else:
        D = 1e-10; D_ads = Config[3]

    # -----------------------------------------------------------------------------------------------------------------

    element1 = FiniteElement('P', mesh.ufl_cell(), 1) # Elements for pesticide concentration C, adsorbed quantity q, and adsorbent concentration a respectively
    element2 = FiniteElement('P', mesh.ufl_cell(), 1)
    element3 = FiniteElement('P', mesh.ufl_cell(), 1)
    mix = MixedElement([element1,element2,element3])
    Space = FunctionSpace(mesh, mix)
    V = VectorFunctionSpace(mesh, element1, 1)
    V1 = Space.sub(0).collapse() # Spaces for C, q and a respectively
    V2 = Space.sub(1).collapse()
    V3 = Space.sub(2).collapse()
    
    w = Function(Space)
    w_n = Function(Space)
    u = Function(V)
    C0 = Function(V1)
    q0 = Function(V2)
    a0 = Function(V3)
    
    N_TIME_STEPS =  max_velocity_index(XDMFFile(FILE_u),u) - 1 if FROM_CHECKPOINT else N_TIME_STEPS_CHKP
    
    (C_n, q_n, a_n) = split(w_n)
    (C,q,a) = split(w)
    (test_C, test_q, test_a) = TestFunctions(Space)
    dw = TrialFunction(Space)
    
    a_0 = Constant(a_0); y = Constant(y); width = Constant(width)
    outletSubdomain = OutletRegion(degree = 2)
    # This mask is the implementation behind ONLY_OUTLET:
    Mask = project(outletSubdomain, V3, form_compiler_parameters={"quadrature_degree": 20}) if ONLY_OUTLET else Constant(1.0)
    adsRegion = AdsorbentRegion(a0 = a_0, y = y, width = width, mesh = mesh, degree=5)
    q0.interpolate(Constant(0.0)) 
    a0 = project(adsRegion, V3, solver_type="lu", form_compiler_parameters={"quadrature_degree": 40}) # Project gaussian to space V3
    a_array = a0.vector().get_local()
    a_array[a_array < 0] = 1e-16
    a0.vector().set_local(a_array)
    a0.vector().apply("insert")
    
    if not FROM_CHECKPOINT:
        C0.interpolate(Constant(0.0))
        t = t_vel
        n0 = n0_vel
    else:
        n0 = N_TIME_STEPS_CHKP
        w0 = Function(Space)
        h5 = HDF5File(mesh.mpi_comm(), FILE_CHECKPOINT, "r")
        h5.read(w0,       "w") 
        h5.close()
        C0 = w0.split()[0]
        t = t_vel + dt_vel*(n0 - n0_vel - factor)
        
    assign(w.sub(0), C0)
    assign(w.sub(1), q0)
    assign(w.sub(2), a0)
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    out = OutletBoundary()
    out.mark(boundaries,2)
    inl = InletBoundary()
    inl.mark(boundaries, 1)
    file = File(os.path.join(Path, "Fields/BoundaryMarksPest.pvd"))
    file << boundaries

    if FROM_CHECKPOINT:
        XDMFileC = XDMFFile(mesh.mpi_comm(), FILE_C)
        XDMFileC.parameters["flush_output"] = True
        XDMFileC.parameters["functions_share_mesh"] = True  
        XDMFileq = XDMFFile(mesh.mpi_comm(), FILE_q)
        XDMFilea = XDMFFile(mesh.mpi_comm(), FILE_a)
        XDMFileq.parameters["flush_output"] = True
        XDMFilea.parameters["flush_output"] = True
        XDMFileq.parameters["functions_share_mesh"] = True    
        XDMFilea.parameters["functions_share_mesh"] = True
    else:
        h5 = HDF5File(mesh.mpi_comm(), FILE_CHECKPOINT, "w")
   
    normal = FacetNormal(mesh)
    
    C_mid = 0.5*(C_n + C); a_mid = 0.5*(a_n + a); q_mid = 0.5*(q_n + q) # Crank-Nicolson quantities from numerical method
    
    XDMFFile(FILE_u).read_checkpoint(u,"u",n0) # Read velocity at selected initial step from velocity simulation
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    bc = [] # Placeholder for empty Dirichlet boundary conditions IF choosing to use weak conditions
    bc = [ DirichletBC(Space.sub(0), Conc, inl) ] # Dirichlet condition, comment if choosing to use weak condition, then uncomment lines 476,477 below

    F1 = (C_n - C)*test_C/dt * dx
    F1 -= dot(u*C_mid,grad(test_C))*dx
    F1 += D*dot(grad(C_mid),grad(test_C))*dx
    F1 += (rho/theta)*(q_n - q)/dt *test_C*dx
    
    if model == 1 and FROM_CHECKPOINT:
        expr = Q*0.5*(b*C_n/(1+b*C_n) + b*C/(1+b*C)) # Langmuir with average from Cranck-Nicolson
    elif not FROM_CHECKPOINT:
        expr = 0.0 # Adsorption off while creating the checkpoint
    elif model == 2:
        C_base = 0.5*pollutant["Conc"] # The following expression is a Taylor approximation of the Cranck-Nicolson average of the Freudlich isotherm near 50% adsorbent concentration. This is due to problems when raising to real exponents internally inside fenics (i.e this avoids real exponents and includes up to order 3 contributions)
        expr =  0.5*(K*C_base**(1/n_val) +K/n_val *C_base**(1/n_val - 1)*(C_n-C_base ) +0.5*K*((1-n_val)/n_val**2)*C_base**(1/n_val -2)*(C_n-C_base)**2 + K/(6*n_val**3)*(1-n_val)*(1-2*n_val)*C_base**(1/n_val -3)*(C_n-C_base)**3 + K*C_base**(1/n_val) +K/n_val *C_base**(1/n_val - 1)*(C-C_base ) +0.5*K*((1-n_val)/n_val**2)*C_base**(1/n_val -2)*(C-C_base)**2 + K/(6*n_val**3)*(1-n_val)*(1-2*n_val)*C_base**(1/n_val -3)*(C-C_base)**3 )
        
    F2 = (q_n - q)/dt *test_q *dx
    F2 -= k1*a_mid*(expr - q_mid)*test_q*dx
    
    F3 = (a_n - a)*test_a/dt * dx
    F3 -= a_mid*dot(u,grad(test_a))*dx
    F3 += D_ads*dot(grad(a_mid),grad(test_a))*dx

    vol_local = CellDiameter(mesh) # local mesh volume
    vel_norm = sqrt(dot(u, u))+1e-1 # velociry norm
    tau = 1*vol_local/(2*vel_norm)  # crude SUPG parameter proposed
    
    # Residuals for each quantity from their respective equation, used for supg
    res_C = ( (C - C_n)/dt
          + div(u*C_mid)
          - D*div(grad(C_mid))
          - (q_n - q)/dt* rho/theta )
    
    res_a = ( (a - a_n)/dt
          + div(u*a_mid)
          - D_ads*div(grad(a_mid)))
    
    res_q = ( (q - q_n)/dt + k1*a_mid*(expr - q_mid))
    

    # Boundary conditions. No need to explictly put the terms on every boundary
    # Only prescribe the fluxes where they're not zero:
    # Make the fields exit through outlet wall
    F1 += dot(u, normal)*C_mid*test_C*ds(2)
    F3 += dot(u, normal)*a_mid*test_a*ds(2)
    F1 -= D*dot(grad(C_mid), normal)*test_C*ds(2)
    F3 -= D_ads*dot(grad(a_mid), normal)*test_a*ds(2)
        
    # UNCOMMENT to weakly impose the influx of pesticide, then comment the Dirichlet alternative near line 428
#    F1 += dot(u, normal)*(pollutant["Conc"] - C_mid)*test_C*ds(1) #  Setup the Newmann boundary condition at inlet to fill river with influx
#    F1 -= D*dot(grad(C_mid), normal)*test_C*ds(1)
    
    F = F1 + F2 + F3 # Actual quantity fenics solves for
    A0 = assemble(a*dx(domain=V3.mesh()))  # The actual quantity of adsorbent deployed in the mesh, close but not equal to the input a_0 due to coarseness in projection
    if FROM_CHECKPOINT:
        print("Actual adsorbent quantity projected to mesh:", A0)
        F = F + tau*dot(u, grad(test_C))*res_C*dx # SUPG term included when solving from checkpoint (with adsorption)

    problem = NonlinearVariationalProblem(F, w_n, bc, J=derivative(F, w_n, dw))
    solver  = NonlinearVariationalSolver(problem)    

    solver.parameters.update({ # Set up some internal solver parameters for fenics
    "nonlinear_solver": "newton",
    "newton_solver": {
        "linear_solver": "mumps",            
        "maximum_iterations": 50,          
        "absolute_tolerance": 1e-12,         
        "relative_tolerance": 1e-13,
        "relaxation_parameter": 1.0,
        "report": False,
        }
    })
    # vol will be either the volume of the outlet region or the full mesh depending on the value of ONLY_OUTLET
    vol = assemble(Mask*dx(domain=V1.mesh())); avg_a = []; avg_c = []; avg_q = []
    ns = np.arange(n0, N_TIME_STEPS, int(factor) ) # Total number of time steps
    test_break = ns[::1][1:] # Slices of ns to test whether to break the simulation (sampling rate to measure pesticide concentration)
    n_snapshot = ns[::10]
    
# ------------------------------------------------ Iteration ------------------------------------------------
    for n in tqdm(ns):
        XDMFFile(FILE_u).read_checkpoint(u,"u",n) # Read current velocity value
        uv = u.vector().get_local()

        mag = np.abs(uv)
        # zero out one single velocity value at a boundary that is very large while not affecting neighbours for unknown reasons
        uv[mag > 0.8] = 0.0
        u.vector().set_local(uv)
        u.vector().apply("insert")
        solver.solve()

        # Force the pesticide concentration to zero if solver over-removes pesticide to negatives
        c_n_array = C_n.ufl_operands[0].vector().get_local()
        c_n_array[c_n_array < 0] = 1e-16
        C_n.ufl_operands[0].vector().set_local(c_n_array)
        C_n.ufl_operands[0].vector().apply("insert")
        ctest = project(C_n,V1)
        w.assign(w_n)
        
        # Force the adsorbent concentration to zero if solver over-removes adsorbent to negatives
        a_n_array = a_n.ufl_operands[0].vector().get_local()
        a_n_array[a_n_array < 1e-13] = 1e-16
        a_n.ufl_operands[0].vector().set_local(a_n_array)
        a_n.ufl_operands[0].vector().apply("insert")
        atest = project(a_n,V3)
        assign(w.sub(0),ctest)
        assign(w.sub(2),atest)
        t+=dt
        
        (C, q, a) = split(w)
        if n in test_break:
            conc_c = assemble(Mask*C*dx(domain=V1.mesh())) # Pesticide concentration in either outlet or full mesh according to ONLY_OUTLET
            if FROM_CHECKPOINT:
                conc_q = assemble(Mask*q*dx(domain=V2.mesh()))
                conc_a = assemble(a*dx(domain=V3.mesh()))
                avg_c.append( np.abs(conc_c / vol) ); avg_a.append( conc_a/vol ); avg_q.append( conc_q ) 
                if conc_a <= 0.05*A0 or (np.argmin(avg_c) < len(avg_c) - 5 and np.argmin(avg_c) > 2): # Condition to stop de simulation (95% adsorbent left, or a minimum for pesticide concentration was found previously)
                    print(f"Breaking simulation at n = {n}, either 95% of adsorbent left or a pesticide concentration minimum was reached")
                    break
                elif conc_c/vol <= 0.005*pollutant["Conc"]:
                    avg_c[np.argmin(avg_c)] = 0
                    print(f"Breaking simulation at n = {n}, pollution reduced to 0%")
                    break
            elif conc_c/vol >= MAX_CONC*pollutant["Conc"]:
                h5.write(w,       "/w")
                h5.close()
                print(f"Breaking simulation at n = {n}, t = {t}. {100*MAX_CONC}% of target pollution in DATA.py reached")
                break
        
        if not FROM_CHECKPOINT:
            if n == ns[-1]:
                msg = "at the outlet" if ONLY_OUTLET else "in the complete mesh"
                print(f' \n Simulation has ended at iteration n = {n-n0}. \n {round(100*(conc_c/vol)/pollutant["Conc"],4)} of target pest concentration in DATA.py reached {msg}. Checkpoint created')
                h5.write(w,       "/w")
                h5.close()
        elif mode == "normal" and n != ns[-1]:
            XDMFileC.write_checkpoint(w.sub(0), "C", t, XDMFFile.Encoding.HDF5, append=True)
            XDMFileq.write_checkpoint(w.sub(1), "q", t, XDMFFile.Encoding.HDF5, append=True)
            XDMFilea.write_checkpoint(w.sub(2), "a", t, XDMFFile.Encoding.HDF5, append=True)
    
    if FROM_CHECKPOINT:
        XDMFileC.write_checkpoint(w.sub(0), "C", t, XDMFFile.Encoding.HDF5, append=True)
        XDMFileq.write_checkpoint(w.sub(1), "q", t, XDMFFile.Encoding.HDF5, append=True)
        XDMFilea.write_checkpoint(w.sub(2), "a", t, XDMFFile.Encoding.HDF5, append=True) 
        if Config[-1] != "benchmark":
            return np.array(avg_c), C, np.array(avg_a)
        else: 
            return w.split(deepcopy=True)[-1] # This is extracting "a" from the mixed field w, as it's the third (-1) component. "a" is the field in the benchmark
    else:
        final_avg_c = assemble(C*Mask*dx(domain=V1.mesh()))/vol
        return final_avg_c, C
