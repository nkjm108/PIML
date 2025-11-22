import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial
import piml_library.lagrangian as lag
import piml_library.hamiltonian as ham
import piml_library.util as util

class BaselineNN_h(nn.Module):
    '''
    Input : s(t, q, v) → Output : (qdot, pdot)
    '''
    hidden_dim : int
    output_dim : int
    
    @nn.compact
    def __call__(self, state):
        t = ham.time(state)
        q = ham.coordinate(state)
        p = ham.momentum(state)
        
        #タプルから1次元配列への変換
        q_flat, _ = ravel_pytree(q)
        p_flat, _ = ravel_pytree(p)
        
        inputs = jnp.concatenate([q_flat, p_flat])
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.output_dim*2)(x)
        return x.squeeze()


@partial(jax.jit, static_argnames = ('model_apply_fn'))
def compute_loss(params, model_apply_fn, batch_states, batch_true_derivatives):
    """
    batch_states: (t, q, p)
    batch_true_derivatives: (q_dot_true, p_dot_true)
    """
    t_batch, q_batch, p_batch = batch_states
    q_dot_true, p_dot_true = batch_true_derivatives

    vmap_model_apply = jax.vmap(
        lambda t, q, p: model_apply_fn({'params':params}, (t, q, p)),
        in_axes=(0, 0, 0) 
    )
    
    predicted_derivatives = vmap_model_apply(t_batch, q_batch, p_batch)
    
    if q_dot_true.ndim == 1:
        q_dot_true = q_dot_true[:, None] # (N,) -> (N, 1)
    if p_dot_true.ndim == 1:
        p_dot_true = p_dot_true[:, None] # (N,) -> (N, 1)
        
    true_derivatives_flat = jnp.concatenate([q_dot_true, p_dot_true], axis=1)
    
    diff = predicted_derivatives - true_derivatives_flat
    
    # 二乗平均誤差 (MSE)
    loss = jnp.mean(diff**2)
    return loss


@partial(jax.jit, static_argnames = ('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, batch_states, batch_true_derivatives):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, model_apply_fn, batch_states, batch_true_derivatives
    )
    # update parameter
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


def create_trajectory(model_apply_fn, trained_params, q_dim):
    """
    q_dim: 座標qの次元数 (分割のために必要)
    """
    # NNの出力関数 (s -> [q_dot, p_dot])
    deriv_learned = lambda s : model_apply_fn({'params':trained_params}, s)
    
    @jax.jit
    def state_derivative(state, args=None):
        deriv_pred = deriv_learned(state)
        q_dot_pred, p_dot_pred = jnp.split(deriv_pred, 2)
        return jnp.array(1.0), q_dot_pred, p_dot_pred
    
    solver = util.ode_solver(state_derivative)
    return solver