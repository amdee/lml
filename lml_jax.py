import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial

# Unbatched LML_jax function
@jax.custom_vjp
def LML_jax(x, N, eps, n_iter, branch=None, verbose=0):
    y, res = lml_forward(x, N, eps, n_iter, branch, verbose)
    return y, res

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def lml_forward(x, N, eps, n_iter, branch, verbose):
    branch = branch if branch is not None else 10 if jax.devices()[0].platform == 'cpu' else 100

    nx = x.shape[0]
    if nx <= N:
        return jnp.ones(nx, dtype=x.dtype), None

    x_sorted = jnp.sort(x)[::-1]
    nu_lower = -x_sorted[N-1] - 7.
    nu_upper = -x_sorted[N] + 7.

    ls = jnp.linspace(0, 1, branch)

    @jax.jit
    def calculate_fs_single(nu, x, N):
        _xs = x + nu
        return jnp.sum(jax.nn.sigmoid(_xs)) - N

    # Vectorize the helper function
    calculate_fs_vectorized = jax.vmap(calculate_fs_single, in_axes=(0, None, None))

    # Replace the loop body with a single call to the vectorized function
    for _ in range(n_iter):
        r = nu_upper - nu_lower
        nus = r * ls + nu_lower
        fs = calculate_fs_vectorized(nus, x, N)

        i_lower = jnp.sum(fs < 0) - 1
        i_lower = jnp.where(i_lower < 0, 0, i_lower)
        i_upper = i_lower + 1

        nu_lower = nus[i_lower]
        nu_upper = nus[i_upper]


    nu = nu_lower + r / 2.
    y = jax.nn.sigmoid(x + nu)

    return y, (y, nu, x, N)

@jax.jit
def lml_backward(res, grad_output):
    y, nu, x, N = res
    if y is None:
        return (jnp.zeros_like(x), None, None, None, None, None)

    Hinv = 1. / (1. / y + 1. / (1. - y))
    dnu = jnp.sum(Hinv * grad_output) / jnp.sum(Hinv)
    dx = -Hinv * (-grad_output + dnu)

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
        return LML_jax(x, N=self.N, eps=self.eps, n_iter=self.n_iter, branch=self.branch, verbose=self.verbose)

if __name__ == '__main__':
    import numpy as np
    import cvxpy as cp
    import numdifftools as nd

    # Setting up for debugging
    import sys
    from IPython.core import ultratb

    sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

    # Constants
    m = 10
    n = 2

    # Seed and data
    np.random.seed(0)
    x = np.random.random(m)

    # CVXPY
    y = cp.Variable(m)
    obj = cp.Minimize(-x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y)))
    cons = [0 <= y, y <= 1, cp.sum(y) == n]
    prob = cp.Problem(obj, cons)
    prob.solve(cp.SCS, verbose=True)    # print(f"value of y vyo_unbatched: {vyo_unbatched}\n\ngradient of y dy0 vyo_unbatched: {dyo_unbatched}")
    # print(f"value of y vyo_batched: {vyo_batched}\n\ngradient of y dy0 vyo_batched: {dyo_batched}") 
    assert 'optimal' in prob.status
    y_cp = y.value

    # Assuming that LML is a flax.linen module
    x_jax_unbatched = jnp.array(x)
    x_jax_batched = jnp.stack([x_jax_unbatched, x_jax_unbatched])
    x = jnp.stack([x, x])
    model = LML(N=n)
    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    dumy_input = jax.random.normal(key1, (n,m)) # Dummy input
    params = model.init(jax.random.PRNGKey(0), dumy_input)
    LML_state = model.bind(params)
    lml = lambda x_input: LML_state(x_input)[0] # for single instance

    # calculating the forward pass for unbatched and batched inputs
    y_unbatched = lml(x_jax_unbatched)
    y_batched = jax.vmap(lml)(x_jax_batched)
    
    # Check almost equality for single inputs
    y_unbatched_check = np.array(y_unbatched, copy=False)
    np.testing.assert_almost_equal(y_unbatched_check, y_cp, decimal=3)
    print("Test 1 Unbatched : passed!")

    # Check almost equality for batched inputs
    y_batched_check = np.array(y_batched, copy=False)
    np.testing.assert_almost_equal(y_batched_check[0], y_cp, decimal=3)
    print("Test 2 batched : passed!")

    # Compute the gradient using JAX
    # below function similar to lml but returns only the value of y amd for sanity check
    def func_4jax_gradient(x_input):
        return LML_state(x_input)[0]
    
    def f(X_input):
        y, _ = LML_state(jnp.array(X_input))
        return np.array(y)

    vy0, dy0 = jax.value_and_grad(func_4jax_gradient)(x_jax_unbatched)
    # print(f"value of y vy0: {vy0}\n\ngradient of y dy0: {dy0}")

    vyo_unbatched, dyo_unbatched = jax.value_and_grad(lml)(x_jax_unbatched)
    vyo_batched, dyo_batched = jax.vmap(jax.value_and_grad(lml))(x_jax_batched)
    # print(f"value of y vyo_unbatched: {vyo_unbatched}\n\ngradient of y dy0 vyo_unbatched: {dyo_unbatched}")
    # print(f"value of y vyo_batched: {vyo_batched}\n\ngradient of y dy0 vyo_batched: {dyo_batched}") 

    # # Compute the gradient using numdifftools
    x_ = np.array(x[0])
    df_unbatched = nd.Jacobian(f)
    df_fd_unbatched = df_unbatched(x_)

    # np.testing.assert_almost_equal(np.array(dy0[0]), dy0_fd[0], decimal=3)
    np.testing.assert_almost_equal(np.array(dy0), df_fd_unbatched[0], decimal=3)
    np.testing.assert_almost_equal(np.array(dyo_unbatched), df_fd_unbatched[0], decimal=3)
    np.testing.assert_almost_equal(np.array(dyo_batched[0]), df_fd_unbatched[0], decimal=3)
    print(f"Test 3: Passed!")    # print(f"value of y vyo_unbatched: {vyo_unbatched}\n\ngradient of y dy0 vyo_unbatched: {dyo_unbatched}")
    # print(f"value of y vyo_batched: {vyo_batched}\n\ngradient of y dy0 vyo_batched: {dyo_batched}") 

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
    