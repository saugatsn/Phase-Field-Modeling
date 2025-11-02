# Phase field fracture implementation in FEniCS
# AT1/AT2 and Miehe/Amor splits

from dolfin import *

use_AT1 = False        
use_Amor = False       

mesh = Mesh('mesh.xml')

# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)

# Trial and test functions for phase-field and displacement
p, q = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)   

# Material parameters 
Gc =  2.7
l = 0.015
lmbda = 121.1538e3
mu = 80.7692e3

# normalization constant c_w 
c_w = 2.0/3.0 if use_AT1 else 1.0

# Bulk modulus for Amor split if needed
K = lmbda + 2.0*mu/3.0

# Degradation function
def g(phi):
    """Degradation function for AT1 or AT2"""
    if use_AT1:
        return (1.0 - phi)**2 / (1.0 + 2.0*phi)
    else:
        return (1.0 - phi)**2

# Constitutive helpers
def epsilon(w):
    return sym(grad(w))

def sigma(w):
    return 2.0*mu*epsilon(w) + lmbda*tr(epsilon(w))*Identity(len(w))

# helper: positive part
def pos(x):
    return 0.5*(x + abs(x))

# deviatoric part helper
def dev_eps(w):
    return dev(epsilon(w))

if use_Amor:
    # Amor (volumetric-deviatoric) split: degrade only volumetric tensile part
    def psi(w):
        tr_e = tr(epsilon(w))
        psi_vol_pos = 0.5 * K * (pos(tr_e))**2
        psi_dev = mu * inner(dev_eps(w), dev_eps(w))
        return psi_vol_pos + psi_dev
else:
    # Miehe spectral split
    def psi(w):
        return 0.5*(lmbda + mu) * (pos(tr(epsilon(w))))**2 + mu * inner(dev_eps(w), dev_eps(w))

# History field (irreversibility)
def H(uold, unew, Hold):
    return conditional(lt(psi(uold), psi(unew)), psi(unew), Hold)

# Boundary conditions
top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")

def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= 0.0

load = Expression("t", t = 0.0, degree=1)

bcbot = DirichletBC(W, Constant((0.0, 0.0)), bot)
bctop = DirichletBC(W.sub(1), load, top)
bc_u = [bcbot, bctop]
bc_phi = [DirichletBC(V, Constant(1.0), Crack)]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries, 1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form 
unew, uold = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)

# displacement bilinear form: use trial function 'u' and test 'v'
E_du = g(pold) * inner(grad(v), sigma(u)) * dx

# Phase-field weak form: use c_w to switch AT1/AT2 normalization
E_phi = ( (Gc * l / c_w) * inner(grad(p), grad(q))
          + ( (Gc / (c_w * l)) + 2.0 * H(uold, unew, Hold) ) * inner(p, q)
          - 2.0 * H(uold, unew, Hold) * q ) * dx

# Problems & solvers (staggered scheme)
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)

# Initialization & output 
t = 0
u_r = 0.007
deltaT = 0.1
tol = 1e-3
conc_f = File ("./ResultsDir/phi.pvd")
fname = open('ForcevsDisp.txt', 'w')

# Staggered scheme
while t <= 1.0:
    t += deltaT
    if t >= 0.7:
        deltaT = 0.01
    load.t = t * u_r
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()

        err_u = errornorm(unew, uold, norm_type='l2', mesh=None)
        err_phi = errornorm(pnew, pold, norm_type='l2', mesh=None)
        err = max(err_u, err_phi)

        uold.assign(unew)
        pold.assign(pnew)
        Hold.assign(project(psi(unew), WW))

        if err < tol:
            print('Iterations:', iter, ', Total time', t)

            if round(t * 1e4) % 10 == 0:
                conc_f << pnew

                Traction = dot(sigma(unew), n)
                fy = Traction[1] * ds(1)
                fname.write(str(t * u_r) + "\t")
                fname.write(str(assemble(fy)) + "\n")

fname.close()
print('Simulation completed')
