import tensorflow as tf

class PINN(tf.keras.Model):
    """
    A simple fully-connected neural network (MLP) to represent the solution u(x,z).
    """
    def __init__(self, layers=[32, 32, 32], activation='tanh'):
        """
        Parameters
        ----------
        layers : list of int
            Width (number of neurons) in each hidden layer.
        activation : str
            Activation function used in each hidden layer.
        """
        super().__init__()
        self.hidden_layers = []

        for units in layers:
            self.hidden_layers.append(
                tf.keras.layers.Dense(units, activation=activation)
            )

        # Output layer: single value (u) with no activation
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        """
        Forward pass through the network.
        inputs.shape = (batch_size, 2) for (x, z)
        """
        x = inputs
        for hl in self.hidden_layers:
            x = hl(x)
        return self.output_layer(x)
