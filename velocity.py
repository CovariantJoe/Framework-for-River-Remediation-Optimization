"""
Finite element script to find the evolution of the velocity and height fields in the river
This script is needed to generate the velocity field used by every other script in the project.
Doing so can take several hours

---------------------- Assumptions ---------------------------------
- THIS SCRIPT WAS ONLY TESTED IN A LINUX OPERATING SYSTEM 
- The file mesh.xml contains the same mesh used to evolve the velocity field
- The mass unit is grams [g], the rest of units are SI units (e.g concentration g/m^3). 

---------------------- If using different mesh ---------------------------------
If using a custom mesh with inlet/outlet BCs, you need to re-create your own functions 
outlet/inlet and the classes InletBoundary and OutletBoundary instead of the ones given below.
---------------------- Dependencies --------------------------------------------
Tested dependencies:
    Python 3.10.16
    dolfin 2019.1.0    
"""
import os
import glob
import numpy as np
import fenics as fe
from dolfin import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ufl import conditional, ge, And, max_value
set_log_level(LogLevel.ERROR)
#---------------------------------------- Configure -------------------------------------------------
if __name__ == "__main__":
    HOURS = 13.3334 # Hour length of simulation
    dt = 5 # Define time step (seconds)
    u_inlet_norm = np.array([0.3405449 , 0.94022825]) # Define according to your inlet normal vector
    u_inlet_mag = 0.045 # Magnitude of velocity at the inlet boundary
    delta_h = 0.0203 # Incoming velocity wave height with respect to the base height of 2 
#-----------------------------------------------------------------------------------------------------

    u_inlet = u_inlet_mag * u_inlet_norm 
    SIMULATION_TIME = HOURS*60*60
    base_height = 2
    N_TIME_STEPS = int(SIMULATION_TIME / dt)
    Config = [N_TIME_STEPS, dt]

def outlet(x, on_boundary):
    #  1e-4 is used as small error to ensure every point in wall fulfills the condition
    return on_boundary and x[1]> MeshData["DBC"][3][1]-1e-4 and x[0] > MeshData["DBC"][2][0]-1e-3

def inlet(x,on_boundary):
    #  1e-4 is used as small error to ensure every point in wall fulfills the condition
    return on_boundary and x[1] > MeshData["DBC"][1][1]-1e-4 and x[1] < MeshData["DBC"][0][1]+1e-4 and x[0] < MeshData["DBC"][1][0] + 1e-4

class OutletBoundary(SubDomain):
    # Class to initiate the outlet wall boundary.
    def inside(self, x, on_boundary):
        return outlet(x,on_boundary)
    
class InletBoundary(SubDomain):
    # Class to initiate the inlet wall boundary.
    def inside(self, x, on_boundary):
        return inlet(x,on_boundary)

def CheckFiles(files):
    found = []
    for file in files:
        if os.path.isfile(file):
            found.append(file)
    if found == []:
        return
    else:
        if input(f" Replace existing solutions? [y/n]: \n \n {found} ").strip().lower() in ("y", "yes"):
            for file in found:
                os.remove(file)
        else:
            raise Exception(f" \n The program was stopped to prevent overwriting the existing solution.")
        return

def main(u_inlet, delta_h, MeshData, Config, bc = [""], u_ini=(1e-2,1e-2), base_height= 2, z = Constant(0), t = None, ANIMATE = False, InletClass = InletBoundary, OutletClass = OutletBoundary):
    from DATA import Path
    try:
        # Test provided mesh .xml works, and Path exists
        if isinstance(MeshData,(dict)):
            mesh = Mesh(MeshData["xml"])
        elif "dolfin.cpp.generation" in str(type(MeshData)) or "dolfin.cpp.mesh.Mesh" in str(type(MeshData)):
            mesh = MeshData
        else:
            raise Exception("Argument MeshData is neither a mesh nor a dictionary leading to a mesh file in .xml format")
        os.path.exists(Path)
        N = mesh.hash() == 16975793265595423454
    except:
        raise Exception(f"Please in DATA.py make sure Path is correct \n Also ensure either MeshData['xml'] is an existing mesh file, or argument MeshData is a dolfin mesh already")
    if len(Config) != 2 or not isinstance(Config[0],(int)) or not isinstance(Config[1],(int,float)):
        raise Exception("Expected Config argument to be integer, then int or float: [ N_TIME_STEPS (int), dt (int or float time step length in seconds) ]")
    else:
        N_TIME_STEPS = Config[0]
        dt = Config[1]
    if isinstance(u_ini,(list,tuple,np.ndarray)) and isinstance(u_ini[0],(float,int)):
        u_0 = Constant(u_ini)
    else:
        u_0 = u_ini
    if t is None:
        t = Constant(0.0)
    elif isinstance(t,(float,int)):
        t = Constant(t)
    elif "dolfin.function.constant.Constant" not in str(type(t)):
        raise Exception(f"Invalid argument passed for t: {type(t)} either leave blank or pass initial time")
    
    FILE_u = os.path.join(Path, "Fields/u.xdmf")
    FILE_h = os.path.join(Path, "Fields/h.xdmf")
    FILE_u5 = os.path.join(Path, "Fields/u.h5")
    FILE_h5 = os.path.join(Path, "Fields/h.h5")
    CheckFiles([FILE_u, FILE_h, FILE_u5, FILE_h5])

    # Define functions space    
    element_u = VectorElement('CG', mesh.ufl_cell(), 1)
    element_h = FiniteElement('CG', mesh.ufl_cell(), 2)
    V_grad = VectorFunctionSpace(mesh, element_h,1) # Vector space for gradient
    mixed_element = MixedElement([element_u, element_h])
    MixedSpace = FunctionSpace(mesh, mixed_element)

    w = Function(MixedSpace)
    w_n = Function(MixedSpace)
    V_u = MixedSpace.sub(0).collapse() 
    V_h = MixedSpace.sub(1).collapse()
    u0 = Function(V_u)
    h0 = Function(V_h)   
    
    try:
        u0.interpolate( u_0 ) # Initiate velocity to u_0 in river (often 0,0)
    except AttributeError:
        u0 = project(u_0, V_u)
                        
    try:
        h0.interpolate(base_height) # Initiate height to base
    except AttributeError:
        h0 = project(base_height,V_h)

    # Initiate functions
    (u, h) = split(w)
    (psi, phi) = TestFunctions(MixedSpace)
    (u_n, h_n) = split(w_n)    
    assigner = FunctionAssigner(MixedSpace, [u0.function_space(), h0.function_space()])
    assigner.assign(w, [u0, h0])

    g = Constant(9.81)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)    
    boundaries.set_all(0)  
    
    if bc == [""]:
    # [""] is default case when calling function without specific boundary conds
    # The reason to put this here is so it doesn't run in every case, other conds can be supplied
     
        # Initiate boundary marking
        inl = InletClass()
        out = OutletClass()
        out.mark(boundaries,4)
        inl.mark(boundaries, 1)
        inlet = InletClass.inside
        outlet = OutletClass.inside
        
        # Visualization to verify with Paraview the boundaries are correct
        file = File(os.path.join(Path, "Fields/BoundaryMarksVel.pvd"))
        file << boundaries

        # Boundary conditions (again, so that this doesn't run when doing benchmarks)
        bc  = [DirichletBC(MixedSpace.sub(0), u_inlet , inl)]
        bc.append( DirichletBC(MixedSpace.sub(1),base_height, out ) )
        bc.append( DirichletBC(MixedSpace.sub(1),base_height + delta_h, inl) )

    # Normal to all walls
    normal = FacetNormal(mesh)
    vol = CellDiameter(mesh)

    u_mid = 0.5*(u+u_n); h_mid = 0.5*(h+h_n)
    # Formulacion variacional velocidad
    penalty = Constant(0.2) # This small penalty will prevent spurious velocity components normal to each boundary
    x = SpatialCoordinate(mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx_metadata = dx(metadata={"quadrature_degree": 12})
    parameters["form_compiler"]["quadrature_degree"] = 12

    # Velocity equation
    F1 = inner((u_n - u), psi)/dt * dx
    F1 += ( 0.5*dot(  dot(u_n,nabla_grad(u_n))  ,psi) + 0.5*dot(  dot(u,nabla_grad(u))  ,psi) ) *dx
    F1 += g*dot(grad(h_mid + z), psi)* dx # minus because RHS_u should be in the LHS for this form

    # This penalty will prevent spurious velocity components normal to the walls
    F1 += penalty/(vol)*dot(u_n,normal)*dot(psi,normal)*ds(0)

    flux_avg = 0.5*(h_n*u_n + h*u)
    F2 = inner((h_n - h), phi)/dt * dx
    F2 -= dot(flux_avg, nabla_grad(phi))*dx
        
    u_norm = sqrt(dot(u_mid,u_mid))+5e0 # 5e0 ~ sqrt(g*h)
    modulator = Expression("n >= 6000 ? -5.5e-05*n +0.772 : 0", n = 0, degree = 1)
    if N:
        tau = conditional(
        lt(x[1], -81.4),
        conditional(gt(x[1], -810), 1*vol/(2.0*u_norm), Constant(0.0)),
        modulator*vol/(2.0*u_norm))
    else:
        tau = 0.025*vol/(2.0*(u_norm)) #SUPG term. May need to fine-tune to your velocities
        
    # Residuals for each quantity from their respective equation, used for SUPG
    R_mom = ( (u - u_n)/dt
        - dot(u_mid, nabla_grad(u_mid))
        - g*0.5*(grad(h_n+z) + grad(h+z))
       )
    
    R_con = (h - h_n)/dt + 0.5*div(h*u + h_n*u_n)

    # SUPG behaves incorrectly if acting too early in the simulation in part due to the velocity field
    # not being present everywhere in the domain yet, so this delays the SUPG term 2% of time steps
    boolean = Expression("n >= t0 ? (1) : (0)", t0 = int(0.02*N_TIME_STEPS), n = 0, degree = 1)
    F1 += inner(boolean*tau*dot(u_mid,nabla_grad(psi)),R_mom)*dx
    F2 += inner(boolean*tau*dot(u_mid,nabla_grad(phi)),R_con)*dx

    F = F1 + F2

    problem = NonlinearVariationalProblem(F, w_n, bc, J=derivative(F, w_n))
    solver = NonlinearVariationalSolver(problem)

    XDMFileU = XDMFFile(mesh.mpi_comm(),FILE_u)
    XDMFileU.parameters["flush_output"] = True
    XDMFileU.parameters["functions_share_mesh"] = True
    XDMFileH = XDMFFile(mesh.mpi_comm(), FILE_h)
    XDMFileH.parameters["flush_output"] = True
    XDMFileH.parameters["functions_share_mesh"] = True
    
    solver.parameters.update({
    "nonlinear_solver": "newton",
    "newton_solver": {
        "linear_solver": "mumps",
        "maximum_iterations": 9,
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 1e-9,
        "relaxation_parameter": 1.0,
        "report":False,
        }})
    w2 = Function(MixedSpace)

    XDMFileU.write_checkpoint(u0, "u", float(t), XDMFFile.Encoding.HDF5, append=True)
    XDMFileH.write_checkpoint(h0, "h", float(t), XDMFFile.Encoding.HDF5, append=True)
    u_ret = []; h_ret = []
# ------------------------------------------------ Iteration ------------------------------------------------
    for n in tqdm(range(N_TIME_STEPS)):
        try:
            solver.solve()    
            t.assign(t+dt)
        except:
            print(f" \n Newton solver failed at iteration {n}")
            if ANIMATE:
                return u_ret, h_ret
            else:
                (u_save, h_save) = split(w_n,deepcopy=True)
                return u_n, h_n

        modulator.n = n
        boolean.n = n
        try:
            u_inlet.t = float(t)
        except AttributeError:
            pass

        w.assign(w_n)
        XDMFileU.write_checkpoint(w.sub(0), "u", float(t), XDMFFile.Encoding.HDF5, append=True)
        XDMFileH.write_checkpoint(w.sub(1), "h", float(t), XDMFFile.Encoding.HDF5, append=True)
        
        if n % 10 == 0:
            (u_save, h_save) = w_n.split(deepcopy=True)
            u_ret.append(u_save); h_ret.append(h_save)

    if ANIMATE:
        return u_ret, h_ret
    else:
        (u_save, h_save) = w_n.split(deepcopy=True)
        return u_save, h_save
# -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__": # Solver call
    from DATA import MeshData, Path
    v, h = main(u_inlet, delta_h, MeshData, Config, u_ini = (1e-5,1e-5), base_height = base_height, z = 0)
