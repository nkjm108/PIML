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

class LHNN(nn.Module): 
    '''
    Input : s(t, q, v) → Output : L
    '''
    hidden_dim : int
    out_dim : int
    
    
    @nn.compact
    def __call__(self, state):
        t = lag.time(state)
        q = lag.coordinate(state)
        v = lag.velocity(state)
        q_flat, _ = ravel_pytree(q)
        v_flat, _ = ravel_pytree(v)
        inputs = jnp.concatenate([q_flat, v_flat])
        
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
def compute_loss(params, model_apply_fn, batch_states, batch_targets):
    """
    LHNN パラメータからL(s)を定義 -> H(t,q,p)へ変換
    HNNと同様のスタイルでLoss計算
    """
    t_batch, q_batch, p_batch = batch_states
    q_dot_true_batch, p_dot_true_batch = batch_targets
    
    # 1. パラメータを使って L(s) を定義
    L_fn = lambda s: model_apply_fn({'params': params}, s)
    
    # 2. L -> H へルジャンドル変換
    H_fn = ham.lagrangian_to_hamiltonian(L_fn)
    H_multi_arg = util.tuple_to_multi_arg(H_fn)
    dH_dq_fn = jax.grad(H_multi_arg, 1) # ∂H/∂q
    dH_dp_fn = jax.grad(H_multi_arg, 2) # ∂H/∂p
    
    vmap_dH_dq = jax.vmap(
        lambda t, q, p: dH_dq_fn(t, q, p),
        in_axes=(0, 0, 0)
    )
    vmap_dH_dp = jax.vmap(
        lambda t, q, p: dH_dp_fn(t, q, p),
        in_axes=(0, 0, 0)
    )
    
    pred_dH_dq = vmap_dH_dq(t_batch, q_batch, p_batch)
    pred_dH_dp = vmap_dH_dp(t_batch, q_batch, p_batch)
    
    loss_term_q = tree_map(lambda pred, true: pred - true,
                           pred_dH_dp, q_dot_true_batch)
    
    loss_term_p = tree_map(lambda pred, true: pred + true,
                           pred_dH_dq, p_dot_true_batch)
    
    flat_q, _ = ravel_pytree(loss_term_q)
    flat_p, _ = ravel_pytree(loss_term_p)
    all_diffs = jnp.concatenate([flat_q, flat_p])
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