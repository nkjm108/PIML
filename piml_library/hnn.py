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

class HNN(nn.Module):
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
def compute_loss(params, model_apply_fn, batch_states, batch_true_derivatives):

    t_batch, q_batch, p_batch = batch_states
    q_dot_true_batch, p_dot_true_batch = batch_true_derivatives
    
    # model_apply_fnにparamsを適応させて∂Hを計算する
    H_learned = lambda s: model_apply_fn({'params': params}, s)
    H_learned_ = util.tuple_to_multi_arg(H_learned) # H_(t, q, p)
    dH1 = jax.grad(H_learned_, 1)
    dH2 = jax.grad(H_learned_, 2)
    
    vmap_grad_H1 = jax.vmap(
        lambda t, q, p: dH1(t, q, p),
        in_axes=(0,0,0)
    )
    vmap_grad_H2 = jax.vmap(
        lambda t, q, p: dH2(t, q, p),
        in_axes=(0,0,0)
    )
    
    grad_H1_pred = vmap_grad_H1(t_batch, q_batch, p_batch)
    grad_H2_pred = vmap_grad_H2(t_batch, q_batch, p_batch)
    
    loss_term1 = tree_map(lambda pred, true: pred - true,
                          grad_H2_pred, q_dot_true_batch)
    
    loss_term2 = tree_map(lambda pred, true: pred + true,
                          grad_H1_pred, p_dot_true_batch)
    
    flat1, _ = ravel_pytree(loss_term1)
    flat2, _ = ravel_pytree(loss_term2)
    all_diffs = jnp.concatenate([flat1, flat2])
    loss = jnp.mean(all_diffs**2)
    
    return loss

@partial(jax.jit, static_argnames=('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, batch_states, batch_true_derivative):
    loss, grads = jax.value_and_grad(compute_loss)( 
        params, 
        model_apply_fn,
        batch_states, 
        batch_true_derivative
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
    
def create_trajectory(model_apply_fn, trained_params):
    H_learned = lambda s: model_apply_fn({'params': trained_params}, s) #s=(t, q, p)
    ds = ham.state_derivative(H_learned)
    solver = util.ode_solver(ds)
    return solver


    
    
    


