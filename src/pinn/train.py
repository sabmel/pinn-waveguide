import numpy as np
import tensorflow as tf
from model import PINN

##############################################################################
# 1. Synthetic or user-defined data: domain, refractive index, boundary, etc.
##############################################################################

def generate_collocation_points(num_interior=2000, num_boundary=400):
    """
    Sample collocation points within and on the boundary of the domain.
    Domain example: x in [-1,1], z in [0,2].
    
    Returns
    -------
    x_int, z_int : tf.Tensor
        Interior points for PDE residual.
    x_bnd, z_bnd : tf.Tensor
        Boundary points for BC enforcement.
    """
    # Define domain corners
    x_min, x_max = -1.0, 1.0
    z_min, z_max = 0.0, 2.0

    # Interior points
    x_int = np.random.uniform(x_min, x_max, (num_interior, 1))
    z_int = np.random.uniform(z_min, z_max, (num_interior, 1))

    # Boundary points
    # For Dirichlet BC: we'll sample edges: x = +/-1, z = 0 or z = 2
    # We'll collect them in a single array, though you can treat each edge separately.
    x_left   = np.ones((num_boundary//4, 1)) * x_min
    z_left   = np.random.uniform(z_min, z_max, (num_boundary//4, 1))
    x_right  = np.ones((num_boundary//4, 1)) * x_max
    z_right  = np.random.uniform(z_min, z_max, (num_boundary//4, 1))
    x_bottom = np.random.uniform(x_min, x_max, (num_boundary//4, 1))
    z_bottom = np.ones((num_boundary//4, 1)) * z_min
    x_top    = np.random.uniform(x_min, x_max, (num_boundary//4, 1))
    z_top    = np.ones((num_boundary//4, 1)) * z_max

    x_bnd = np.vstack([x_left, x_right, x_bottom, x_top])
    z_bnd = np.vstack([z_left, z_right, z_bottom, z_top])

    return (
        tf.convert_to_tensor(x_int, dtype=tf.float32),
        tf.convert_to_tensor(z_int, dtype=tf.float32),
        tf.convert_to_tensor(x_bnd, dtype=tf.float32),
        tf.convert_to_tensor(z_bnd, dtype=tf.float32),
    )


def refractive_index(x, z):
    """
    Example: step-index waveguide with n_core=1.5 for |x|<0.3, n_clad=1.0 otherwise.
    Adjust to your geometry.
    """
    core_half_thickness = 0.3
    n_core = 1.5
    n_clad = 1.0

    # Condition: if |x| <= 0.3 => n_core, else n_clad
    return tf.where(tf.abs(x) <= core_half_thickness, n_core, n_clad)


##############################################################################
# 2. Define the PDE residual and boundary condition losses
##############################################################################

@tf.function
def wave_equation_residual(model, x, z, k):
    """
    Compute the PDE residual for the 2D wave/Helmholtz equation:
        ∂²u/∂x² + ∂²u/∂z² + (k * n(x,z))^2 * u = 0
    
    Parameters
    ----------
    model : tf.keras.Model
        The neural network (PINN) that approximates u(x,z).
    x, z : tf.Tensor, shape=(batch_size,1)
        Input coordinates for PDE residual.
    k : float
        Free-space wavenumber = 2π / λ0 (for instance).
    
    Returns
    -------
    residual : tf.Tensor, shape=(batch_size,1)
        The PDE residual evaluated at each point.
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, z])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, z])
            u = model(tf.concat([x, z], axis=1))  # shape=(batch_size,1)
        # First derivatives
        u_x = tape1.gradient(u, x)
        u_z = tape1.gradient(u, z)
    # Second derivatives
    u_xx = tape2.gradient(u_x, x)
    u_zz = tape2.gradient(u_z, z)
    del tape1
    del tape2

    # Refractive index
    n_values = refractive_index(x, z)  # shape=(batch_size,)

    # PDE: u_xx + u_zz + (k * n)^2 * u = 0
    return u_xx + u_zz + (k * n_values)**2 * u


@tf.function
def boundary_condition_loss(model, x_bnd, z_bnd):
    """
    Dirichlet boundary condition: u(x_bnd, z_bnd) = 0
    For a waveguide with perfect electric conductor (PEC) walls or simply Dirichlet edges.
    
    Returns
    -------
    bc_loss : tf.Tensor
        A scalar loss enforcing u=0 on the boundary.
    """
    u_bnd = model(tf.concat([x_bnd, z_bnd], axis=1))
    # We want u_bnd to be 0 on the boundary, so the MSE is used here
    bc_loss = tf.reduce_mean(tf.square(u_bnd))
    return bc_loss


##############################################################################
# 3. Training Loop
##############################################################################

def train_pinn(num_epochs=5000, learning_rate=1e-3, print_every=500):
    # Step 1: Instantiate model
    model = PINN(layers=[32, 32, 32], activation='tanh')

    # Step 2: Generate collocation points
    x_int, z_int, x_bnd, z_bnd = generate_collocation_points(
        num_interior=2000,
        num_boundary=400
    )

    # Convert k to a Tensor
    lambda_0 = 1.0             # example free-space wavelength
    k_val = 2.0 * np.pi / lambda_0  # if λ0=1 => k=2π
    k_tensor = tf.constant(k_val, dtype=tf.float32)

    # Step 3: Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # PDE residual on interior points
            res_int = wave_equation_residual(model, x_int, z_int, k_tensor)
            loss_pde = tf.reduce_mean(tf.square(res_int))

            # Boundary condition loss
            loss_bc = boundary_condition_loss(model, x_bnd, z_bnd)

            # Total loss
            loss = loss_pde + loss_bc

        # Compute gradients and update
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Print progress
        if epoch % print_every == 0:
            print(
                f"Epoch: {epoch}, "
                f"PDE Loss: {loss_pde.numpy():.5e}, "
                f"BC Loss: {loss_bc.numpy():.5e}, "
                f"Total Loss: {loss.numpy():.5e}"
            )

    return model


if __name__ == "__main__":
    trained_model = train_pinn(
        num_epochs=5000,
        learning_rate=1e-3,
        print_every=500
    )
    # Now you can use trained_model(...) to predict u(x,z) anywhere in the domain.
