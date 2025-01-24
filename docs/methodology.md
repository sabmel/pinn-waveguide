# Methodology

This document describes the mathematical formulation and problem setup for the **2D Waveguide PINN** project. We cover the PDE, domain specifics, refractive index profile, and boundary conditions used in this physics-informed neural network approach.

---

## 1. Overview

We aim to solve a **2D scalar wave equation** that approximates the propagation of electromagnetic waves in a planar waveguide. Specifically, we focus on a **transverse electric (TE)** mode, where the electric field is predominantly in the \( y \)-direction.

In this simplified setting, the equation of interest is:

\[
\frac{\partial^2 E_y}{\partial x^2}
+ \frac{\partial^2 E_y}{\partial z^2}
+ k^2 \, n^2(x,z) \, E_y = 0,
\]

where:

- \( E_y(x,z) \) is the electric field component in the out-of-plane \( y \)-direction.  
- \( k = \frac{2\pi}{\lambda_0} \) is the free-space wavenumber (\(\lambda_0\) is the free-space wavelength).  
- \( n(x,z) \) is the refractive index distribution in the waveguide.

Alternatively, written in **scalar form**:

\[
\nabla^2 u + k^2 \, n^2(x,z) \, u = 0,
\]

which, in 2D Cartesian coordinates, becomes:

\[
\frac{\partial^2 u}{\partial x^2}
+ \frac{\partial^2 u}{\partial z^2}
+ k^2 \, n^2(x,z) \, u = 0.
\]

---

## 2. Domain Definition

We define a rectangular domain in the \((x,z)\) plane:

- \( x \in [-a,\, +a] \) to represent the transverse direction.  
- \( z \in [0,\, L] \) along the wave propagation direction.

\( x \) captures the thickness or width of the waveguide core and cladding, whereas \( z \) represents the propagation length.

---

## 3. Refractive Index Profile

A simple **step-index** planar waveguide has a uniform refractive index in the core and a slightly lower index in the cladding. If the core has thickness \( t \) and refractive index \( n_\text{core} \), while the cladding has refractive index \( n_\text{cladding} \), the index profile \( n(x) \) can be described by:

\[
n(x) = 
\begin{cases}
n_\text{core}, & |x| \leq \frac{t}{2},\\
n_\text{cladding}, & |x| > \frac{t}{2}.
\end{cases}
\]

---

## 4. Boundary Conditions

### 4.1 Dirichlet Boundaries

For a perfect electrical conductor (PEC) boundary at \( x = \pm a \)

\[
E_y(x = \pm a, z) = 0 \quad \text{and/or} \quad
E_y(x, z = 0) = 0, \quad
E_y(x, z = L) = 0.
\]

This approach is used to approximate conducting walls or to enforce a specific mode shape.

### 4.2 Neumann Boundaries

\[
\frac{\partial E_y}{\partial n} = 0 \quad \text{on the boundary},
\]
where \(\partial/\partial n\) indicates the outward normal derivative.

## 5. Summary

**Governing PDE**  
\[
\frac{\partial^2 E_y}{\partial x^2}
+ \frac{\partial^2 E_y}{\partial z^2}
+ k^2 \, n^2(x,z) \, E_y = 0.
\]

**Domain**  
\[
x \in [-a, +a], \quad z \in [0, L].
\]

**Refractive Index**  
\[
n(x,z) = 
\begin{cases}
n_\text{core}, & |x| \le \frac{t}{2},\\
n_\text{cladding}, & |x| > \frac{t}{2}.
\end{cases}
\]

**Boundary Conditions**  
- **Dirichlet** (\( E_y = 0 \)), **Neumann** (\( \partial E_y/\partial n = 0 \)), or  
- **Absorbing/PML** (for open boundaries).

---

## 6. Next Steps

1. **Classical Solver**: We will implement (or utilize) a finite-element or finite-difference solver to generate a reference solution.  
2. **PINN**: We will then create a physics-informed neural network to solve the same PDE and compare results (error metrics, convergence, etc.).  
3. **Sampling Strategy**: We will define a set of collocation points in the interior (for the PDE residual) and on the boundary (for BCs) for the PINN training.  
4. **Performance Metrics**: We will measure the \(\ell_2\) error norm, maximum absolute error, and computation time to compare with the classical solver.

---

**Author**: Sabrina Smith
**Last Updated**: January 23, 2025

> **Notes**  