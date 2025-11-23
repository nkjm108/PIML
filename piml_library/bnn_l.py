import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial
import piml_library.lagrangian as lgr
import piml_library.util as util

class BaselineNN_l(nn.Module):
    '''
    Input : s(t, q, v) → Output : a
    '''
    hidden_dim : int
    output_dim : int
    
    @nn.compact
    def __call__(self, state):
        t = lgr.time(state)
        q = lgr.coordinate(state)
        v = lgr.velocity(state)
        
        #タプルから1次元配列への変換
        q_flat, _ = ravel_pytree(q)
        v_flat, _ = ravel_pytree(v)
        
        inputs = jnp.concatenate([q_flat, v_flat])
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.output_dim)(x)
        return x.squeeze()


@partial(jax.jit, static_argnames = ('model_apply_fn'))
def compute_loss(params, model_apply_fn, batch_states, batch_true_accelerations):
    t_batch, q_batch, v_batch = batch_states
    
    vmap_model_apply = jax.vmap(
        lambda t, q, v: model_apply_fn({'params':params}, (t, q, v)),
        in_axes=(0, 0, 0) 
    )
    
    predicted_accelerations = vmap_model_apply(t_batch, q_batch, v_batch)
    
    #同じ葉の部分でlamdaで定義した関数を適応する
    diff = tree_map(lambda pred, true: pred - true,
                    predicted_accelerations,
                    batch_true_accelerations)
    
    diff_flat, _ = ravel_pytree(diff)
    loss = jnp.mean(diff_flat**2)
    return loss


@partial(jax.jit, static_argnames = ('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, _, batch_true_accel):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, model_apply_fn, _, batch_true_accel
    )

    #update parameter
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


def create_trajectory(model_apply_fn, trained_params):
    a_learned = lambda s : model_apply_fn({'params':trained_params}, s)
    
    @jax.jit
    def state_derivative(state, args=None):
        t, q, v = state
        a_pred = jnp.array([a_learned(state)]) #aはスカラーとしてNNから出るためｖと形状をあわせる
        return jnp.array(1.0), v, a_pred
    
    solver = util.ode_solver(state_derivative)
    return solver