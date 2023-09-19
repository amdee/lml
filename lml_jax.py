import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

@jax.custom_vjp
def LML_jax(x, N, eps, n_iter, branch=None, verbose=0):
    y, res = lml_forward(x, N, eps, n_iter, branch, verbose)
    return y, res

def lml_forward(x, N, eps, n_iter, branch, verbose):
    branch = branch if branch is not None else (10 if x.device().platform == 'cpu' else 100)

    single = x.ndim == 1
    if single:
        x = jnp.expand_dims(x, 0)
    
    n_batch, nx = x.shape
    if nx <= N:
        return jnp.ones((n_batch, nx), dtype=x.dtype), None
    
    x_sorted = jnp.sort(x, axis=1)[:, ::-1]
    nu_lower = -x_sorted[:, N-1] - 7.
    nu_upper = -x_sorted[:, N] + 7.

    ls = jnp.linspace(0, 1, branch)

    for _ in range(n_iter):
        r = nu_upper - nu_lower
        I = r > eps

        if not jnp.any(I):
            break

        Ix = jnp.where(I)[0]

        nus = r[Ix][:, None] * ls + nu_lower[Ix][:, None]
        _xs = x[Ix][:, None, :] + nus[:, :, None]
        fs = jnp.sum(jax.nn.sigmoid(_xs), axis=-1) - N

        i_lower = jnp.sum(fs < 0, axis=-1) - 1
        i_lower = jnp.where(i_lower < 0, 0, i_lower)

        i_upper = i_lower + 1

        nu_lower = jnp.where(I, jnp.take_along_axis(nus, i_lower[:, None], axis=1).squeeze(), nu_lower)
        nu_upper = jnp.where(I, jnp.take_along_axis(nus, i_upper[:, None], axis=1).squeeze(), nu_upper)

    nu = nu_lower + r / 2.
    y = jax.nn.sigmoid(x + nu[:, None])

    return y, (y, nu, x, N)

def lml_backward(res, grad_output):
    y, nu, x, N = res

    if y is None:
        return (jnp.zeros_like(x), None, None, None, None, None)

    Hinv = 1. / (1. / y + 1. / (1. - y))
    dnu = jnp.sum(Hinv * grad_output, axis=1) / jnp.sum(Hinv, axis=1)
    dx = -Hinv * (-grad_output + dnu[:, None])

    return (dx, None, None, None, None, None)

LML_jax.defvjp(lml_forward, lml_backward)


class LML(nn.Module):
    N: int = 1
    eps: float = 1e-4
    n_iter: int = 100
    branch: int = None
    verbose: int = 0

    @nn.compact
    def __call__(self, x):
        """This function is called when you use the syntax y = model(x) and returns a tupple (y, res)

        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, n_features

        Returns:
            (y, res) tupple: y is the output of the LML function and res is a tuple of (y, nu, x, N)
        """
        return LML_jax(x, N=self.N, eps=self.eps, n_iter=self.n_iter, branch=self.branch, verbose=self.verbose)


if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    import numpy.random as npr
    import jax
    from jax import grad
    from flax import linen as nn  # Assuming LML is defined here
    import cvxpy as cp
    import numdifftools as nd

    # Setting up for debugging
    import sys
    from IPython.core import ultratb

    sys.excepthook = ultratb.FormattedTB(
        mode='Verbose', color_scheme='Linux', call_pdb=1
    )

    # Constants
    m = 10
    n = 2
    batch_size = 2

    # Seed and data
    npr.seed(0)
    x = npr.random(m)

    # CVXPY
    y = cp.Variable(m)
    obj = cp.Minimize(-x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y)))
    cons = [0 <= y, y <= 1, cp.sum(y) == n]
    prob = cp.Problem(obj, cons)
    prob.solve(cp.SCS, verbose=True)
    assert 'optimal' in prob.status
    y_cp = y.value

    # Assuming that LML is a flax.linen module in your codebase
    model = LML(N=n)
    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    dumy_input = jax.random.normal(key1, (batch_size, m)) # Dummy input
    params = model.init(jax.random.PRNGKey(0), dumy_input)

    # Using JAX and Flax's LML
    x = jnp.stack([x, x])
    LML_state = model.bind(params)  # Use this if you want to use the syntax y = model(x)
    y_jax, _ = LML_state(x)

    # Check almost equality
    y_jax_check = np.array(y_jax, copy=False)
    print(f'y_jax[0] = {y_jax_check[0]}\ny_cp = {y_cp}')
    np.testing.assert_almost_equal(y_jax_check[0], y_cp, decimal=3)
    print("Test passed!")

    # # Compute the gradient using JAX
    f = lambda X: LML_state(X)  # This reyurns a tuple (y, res)
    dy0, grad_dy0= jax.value_and_grad(f, argnums=0)(x)    
    # # dy0 = jnp.squeeze(dy0)
    print(f"dy0 = {dy0}, \n\ngrad_dyo = {grad_dy0}")

    # Function for numdifftools
    def func(x):
        y, _ = LML_state(jnp.array(x))
        return np.array(y[0])

    # # Numerical gradient
    df = nd.Jacobian(func)
    dy0_fd = df(x)[0]
    print(f"dy0_fd = {dy0_fd}")

    # Test again
    np.testing.assert_almost_equal(dy0, dy0_fd, decimal=3)


##########Test Again with torch implimentation to see same result##############
    # import torch
    # from jax import random
    # from lml import LML as LML_Torch
    # # Initialize the model and its state
    # model = LML(N=2, eps=1e-5, n_iter=100, verbose=True)
    # rng = random.PRNGKey(0)
    # params = model.init(rng, jnp.zeros((5,))) # Using zeros as a dummy input to initialize the model

    # # Apply the initialized model to the input
    # module_state = model.bind(params)  # Use this if you want to use the syntax y = model(x)
    # x = jnp.array([[ -4.0695,  10.8666,  13.0867,  -7.1431, -14.7220], [ -7.0695,  1.8666,  3.0867,  -10.1431, -4.7220]])
    # # x = random.normal(random.PRNGKey(1), (5,))
    # print(f"x created using jax {x}")
    # y = module_state(x)
    

    # # x_torch = 10.*torch.randn(5) # tensor([ -4.0695,  10.8666,  13.0867,  -7.1431, -14.7220])
    # x_torch = torch.tensor([[ -4.0695,  10.8666,  13.0867,  -7.1431, -14.7220],[ -7.0695,  1.8666,  3.0867,  -10.1431, -4.7220]])
    # print(f"x created using torch {x_torch}")
    # y_torch = LML_Torch(N=2, eps=1e-5, n_iter=100, verbose=True)(x_torch)

    # print("\n\ny from jax version:", y[0])
    # print("\n\ny from torch version:", y_torch)
    