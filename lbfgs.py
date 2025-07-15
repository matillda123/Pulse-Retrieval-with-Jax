import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from utilities import MyNamespace



# the two loops need to have differing counting directions

def backward_loop(q, i, rho, s, y):
    alpha = rho[i]*jnp.vdot(s[i], q)
    q = q - alpha*y[i]
    return q, alpha

def forward_loop(r, i, alpha, rho, s, y):
    beta = rho[i]*jnp.vdot(y[i], r)
    r = r + s[i] * (alpha[i] - beta)
    return r, None


def do_lbfgs(grad_current, lbfgs_state, descent_info):
    grad_prev, newton_direction_prev, step_size_prev = lbfgs_state.grad_prev, lbfgs_state.newton_direction_prev, lbfgs_state.step_size_prev

    m = descent_info.lbfgs_memory
    m_backward = jnp.arange(0,m,1)
    m_forward = jnp.arange(m,0,-1)

    s = -1*step_size_prev*newton_direction_prev
    y = grad_current - grad_prev


    rho = 1/jnp.real(jnp.vecdot(y,s) + 1e-12)
    rho = jnp.maximum(rho, 0)
    alpha = jnp.zeros(m, dtype=jnp.complex64)
    
    q = grad_current
    # for i in jnp.arange(0,m,1):
    #     jax.debug.print("{error}", error=i)
    #     alpha = alpha.at[i].set(rho[i]*jnp.vdot(s[i], q))
    #     q = q - alpha[i]*y[i]

    do_backward = Partial(backward_loop, rho=rho, s=s, y=y)
    q, alpha = jax.lax.scan(do_backward, q, m_backward)

    n = jnp.shape(grad_prev)[-1]
    r = jnp.eye(n) @ q
    # for i in jnp.arange(m,0,-1):
    #     jax.debug.print("{error}", error=i)
    #     beta = rho[i]*jnp.vdot(y[i], r)
    #     r = r + s[i] * (alpha[i] - beta)
    do_forward = Partial(forward_loop, alpha=alpha, rho=rho, s=s, y=y)
    r, _ = jax.lax.scan(do_forward, r, m_forward)

    newton_direction = r
    return newton_direction, rho




def get_pseudo_newton_direction(grad, lbfgs_state, descent_info):
    newton_direction, rho = jax.vmap(do_lbfgs, in_axes=(0,0,None))(grad, lbfgs_state, descent_info)

    #jax.debug.print("{error}", error=rho[0])


    # this updateing goes wrong somehow the past is changed ??
    grad_arr = lbfgs_state.grad_prev
    grad_arr = grad_arr.at[:,1:].set(grad_arr[:,:-1])
    grad_arr = grad_arr.at[:,0].set(grad)

    jax.debug.print("{error}", error=jnp.sum(grad_arr[0], axis=-1))

    newton_arr = lbfgs_state.newton_direction_prev
    newton_arr = newton_arr.at[:,1:].set(newton_arr[:,:-1])
    newton_arr = newton_arr.at[:,0].set(newton_direction)

    lbfgs_state = MyNamespace(grad_prev = grad_arr, 
                              newton_direction_prev=newton_arr,
                              step_size_prev = lbfgs_state.step_size_prev)
    return newton_direction, lbfgs_state