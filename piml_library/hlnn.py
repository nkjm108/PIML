import jax
import jax.numpy as jnp
import optax
from flax import linen as nn 
from flax.linen import initializers
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial
import piml_library.lagrangian as lag
import piml_library.hamiltonian as ham
import piml_library.util as util

class HLNN(nn.Module):
    '''
    Input : s=(t, q, p) → Output : H
    '''
    hidden_dim : int 
    out_dim : int
    
    @nn.compact
    def __call__(self, state):
        t = ham.time(state) 
        q = ham.coordinate(state)
        p = ham.momentum(state)
        
        q_flat, _ = ravel_pytree(q)
        p_flat, _ = ravel_pytree(p)
        
        inputs = jnp.concatenate([q_flat, p_flat])
        
        '''
        LNN Initialization
        n = hidden units number
        '''
        n = self.hidden_dim
        sqrt_n = jnp.sqrt(n)
        
        # input layer → hidden layer 1
        var_input = 2.2 / sqrt_n
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(var_input)),
            bias_init=nn.initializers.zeros
        )(inputs) 
        x = nn.softplus(x) 
        
        # hidden layer 1 → hidden layer 2
        var_hidden1 = (0.58 * 1) / sqrt_n
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(var_hidden1)),
            bias_init=nn.initializers.zeros
        )(x)
        x = nn.softplus(x)
        
        # hidden layer 2 → output layer
        var_output = n / sqrt_n 
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(var_output)),
            bias_init=nn.initializers.zeros
        )(x)
        
        return x.squeeze()


@partial(jax.jit, static_argnames=('model_apply_fn',))
def compute_loss(params, model_apply_fn, batch_states, batch_true_accelerations):
    
    H_learned = lambda s: model_apply_fn({'params': params}, s)
    L_learned = ham.hamiltonian_to_lagrangian(H_learned)
    
    a_func = lag.lagrangian_to_acceleration(L_learned)
    
    t_batch, q_batch, v_batch = batch_states
    
    vmap_a_func = jax.vmap(
        lambda t, q, v: a_func((t, q, v)), 
        in_axes=(0, 0, 0)
    )
    
    predicted_accelerations = vmap_a_func(t_batch, q_batch, v_batch)
    
    diff = tree_map(lambda pred, true: pred - true, 
                          predicted_accelerations, 
                          batch_true_accelerations)
    diff_flat, _ = ravel_pytree(diff)
    
    loss = jnp.mean(diff_flat**2)
    return loss


@partial(jax.jit, static_argnames=('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, batch_states, batch_true_accel):

    loss, grads = jax.value_and_grad(compute_loss)( 
        params, 
        model_apply_fn, 
        batch_states, 
        batch_true_accel
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

