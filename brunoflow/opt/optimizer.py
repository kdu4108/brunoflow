"""
This module implements common optimization methods
"""

from jax import numpy as jnp
import types


class Optimizer:
    """
    Represents an optimization method.

    Attributes:
        params: a list of Parameters to be optimized
        step_size: gradient step size hyperparameter
    """

    def __init__(self, params_to_optimize, step_size=0.001):
        if isinstance(params_to_optimize, types.GeneratorType):
            params_to_optimize = [p for p in params_to_optimize]
        self.params = params_to_optimize
        self.step_size = step_size

    def step(self):
        """
        Take one optimization step
        """
        raise NotImplementedError("Optimizer step method not implemented!")

    def zero_gradients(self):
        """
        Zero out the gradients for all parameters to be optimzed.
        Must be called before computing the backward pass to prevent incorrect gradient
            accumulation across iterations.
        """
        for p in self.params:
            p.zero_gradients()


# --------------------------------------------------------------------------------


class SGD(Optimizer):
    def __init__(self, params_to_optimize, step_size=0.001, momemtum=0.0):
        super(SGD, self).__init__(params_to_optimize, step_size=step_size)
        del params_to_optimize  # Safety deleting because you should only use self.params from here on out.

        self.mu = momemtum
        self.v = [jnp.zeros_like(p.val, dtype=float) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.v[i] * self.mu - p.grad * self.step_size
            p.val += self.v[i]


# --------------------------------------------------------------------------------


class AdaGrad(Optimizer):
    def __init__(self, params_to_optimize, step_size=0.001, eps=1e-8):
        super(AdaGrad, self).__init__(params_to_optimize, step_size)
        del params_to_optimize  # Safety deleting because you should only use self.params from here on out.

        self.eps = eps
        self.g2 = [jnp.zeros_like(p.val, dtype=float) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.g2[i] += p.grad * p.grad
            p.val -= self.step_size * (p.grad / (jnp.sqrt(self.g2[i]) + self.eps))


# --------------------------------------------------------------------------------


class RMSProp(Optimizer):
    def __init__(self, params_to_optimize, step_size=0.001, decay_rate=0.9, eps=1e-8):
        super(RMSProp, self).__init__(params_to_optimize, step_size)
        del params_to_optimize  # Safety deleting because you should only use self.params from here on out.

        self.decay_rate = decay_rate
        self.eps = eps
        self.g2 = [jnp.zeros_like(p.val, dtype=float) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.g2[i] = self.decay_rate * self.g2[i] + (1 - self.decay_rate) * p.grad * p.grad
            p.val -= self.step_size * (p.grad / (jnp.sqrt(self.g2[i]) + self.eps))


# --------------------------------------------------------------------------------


class Adam(Optimizer):
    def __init__(self, params_to_optimize, step_size=0.001, beta1=0.9, beta2=0.99, eps=1e-8):
        super(Adam, self).__init__(params_to_optimize, step_size)
        del params_to_optimize  # Safety deleting because you should only use self.params from here on out.

        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [jnp.zeros_like(p.val, dtype=float) for p in self.params]
        self.v = [jnp.zeros_like(p.val, dtype=float) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad * p.grad
            m = self.m[i] / (1 - self.beta1)
            v = self.v[i] / (1 - self.beta2)
            p.val -= self.step_size * (m / (jnp.sqrt(v) + self.eps))
