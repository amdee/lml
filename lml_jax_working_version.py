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

    # Constants
    m = 10
    n = 2

    # Seed and data
    np.random.seed(0)
    x = np.random.random(m)

    # Assuming that LML is a flax.linen module in your codebase
    model = LML(N=n)
    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    dumy_input = jax.random.normal(key1, (n,m)) # Dummy input
    params = model.init(jax.random.PRNGKey(0), dumy_input)
    LML_state = model.bind(params)
    y, _ = LML_state(jnp.stack([x, x]))

    # # # Compute the gradient using JAX
    dy0, grad_dy0 = jax.value_and_grad(lambda x: model.apply(params, jnp.array([x, x]))[0, 0])(x)
    print(f"dy0: {dy0}\n\ngrad_dy0: {grad_dy0}")