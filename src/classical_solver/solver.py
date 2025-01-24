"""
Classical finite-element solver for a 2D waveguide problem using FEniCS.

PDE (Helmholtz-like equation):
    -∇²u - k^2 * n^2(x,z) * u = 0
subject to user-defined boundary conditions.
"""

import numpy as np
from dolfin import *

def refractive_index_expression(x_core_half, n_core, n_cladding):
    """
    Returns a dolfin Expression or UserExpression that defines the refractive
    index distribution n(x, z) for a step-index waveguide.
    """
    # In this example, we define n as piecewise constant in x:
    class RefractiveIndex(UserExpression):
        def eval(self, values, x):
            # x is a 2D point [x_coord, z_coord]
            if abs(x[0]) <= x_core_half:
                values[0] = n_core
            else:
                values[0] = n_cladding

        def value_shape(self):
            return ()

    return RefractiveIndex(degree=0)

def solve_waveguide_2D(
    x_min=-1.0, x_max=1.0,
    z_min=0.0,  z_max=5.0,
    nx=50,      nz=250,
    n_core=1.5, n_cladding=1.0,
    k=2*np.pi,  # free-space wavenumber = 2π / λ₀, set λ₀=1 for example
    bc_type="dirichlet"
):
    """
    Solve the Helmholtz-like equation using FEniCS.

    PDE: -∇²u - (k^2 * n^2) u = 0

    Parameters
    ----------
    x_min, x_max : float
        Transverse domain boundaries (waveguide width).
    z_min, z_max : float
        Propagation direction boundaries.
    nx, nz : int
        Number of elements in x and z directions.
    n_core, n_cladding : float
        Refractive indices for the core and cladding.
    k : float
        Free-space wavenumber (2π / λ₀).
    bc_type : str
        "dirichlet" or "neumann" for boundary conditions.

    Returns
    -------
    mesh : Mesh
        The FEniCS mesh object.
    u : Function
        The solution field.
    """

    # Create rectangular mesh
    mesh = RectangleMesh(Point(x_min, z_min), Point(x_max, z_max), nx, nz)
    V = FunctionSpace(mesh, "Lagrange", degree=1)

    # Define refractive index expression
    x_core_half = 0.3  # half-thickness of the waveguide core, example
    n_expr = refractive_index_expression(x_core_half, n_core, n_cladding)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Convert the refractive index to a dolfin Function
    n_func = Function(V)
    n_project = project(n_expr, V)
    n_func.assign(n_project)

    # Define the PDE in variational (weak) form:
    # a(u,v) = L(v), where
    # a(u,v) = ∫(∇u·∇v) dx - ∫(k^2 * n^2 * u * v) dx
    # L(v) = 0
    a = dot(grad(u), grad(v))*dx - (k**2)*n_func**2 * u * v * dx
    L = Constant(0.0)*v*dx  # Zero on the right-hand side

    # -- Boundary conditions --
    if bc_type.lower() == "dirichlet":
        # For Dirichlet: u = 0 on all boundaries
        def boundary_all(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(0.0), boundary_all)

        # Solve
        u_solution = Function(V)
        solve(a == L, u_solution, bc)

    elif bc_type.lower() == "neumann":
        # No explicit boundary condition needed for homogeneous Neumann
        # (the condition is naturally applied in the weak form if the boundary term is zero).
        # Solve
        u_solution = Function(V)
        solve(a == L, u_solution)

    else:
        raise ValueError(f"Unknown bc_type: {bc_type}. Use 'dirichlet' or 'neumann'.")

    return mesh, u_solution

def save_solution(mesh, u_solution, filename="waveguide_solution.npy"):
    """
    Save the FEM solution on vertices to a NumPy array for later comparison.
    """
    # Get vertex coordinates and solution values
    coords = mesh.coordinates()
    values = u_solution.compute_vertex_values(mesh)

    # Combine coords and values in one array [x, z, u]
    # Note: The ordering in `coords` and `values` will match by vertex index.
    data = np.column_stack((coords[:, 0], coords[:, 1], values))

    # Save
    np.save(filename, data)
    print(f"Solution saved to {filename}")

if __name__ == "__main__":
    # Example usage:
    mesh, solution = solve_waveguide_2D(
        x_min=-1.0, x_max=1.0,
        z_min=0.0,  z_max=5.0,
        nx=50, nz=250,
        n_core=1.5, n_cladding=1.0,
        k=2*np.pi,   # Wavenumber for λ₀ = 1
        bc_type="dirichlet"
    )

    # Save the solution in a NumPy file
    save_solution(mesh, solution, filename="waveguide_solution.npy")