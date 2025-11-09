import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial
import piml_library.lagrangian as lgr
import piml_library.hamiltonian as ham
import piml_library.util as util

class HamiltonianNN(nn.Module): #nn.Moduleを継承。NNの雛形
    '''
    Input : s=(t, q, p) → Output : H
    '''
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, state):
        t = ham.time(state)
        q = ham.coordinate(state)
        p = ham.momentum(state)
        
        q_flat, _ = ravel_pytree(q)
        p_flat, _ = ravel_pytree(p)
        
        inputs = jnp.concatenate([q_flat, p_flat])
        
        #MLP
        x = nn.Dense(self.hidden_dim)(inputs) #全結合層
        x = nn.softplus(x) #活性化関数
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(1)(x)
        # (batch_size, 1) -> (batch_size,)
        return x.squeeze()
    
    
@partial(jax.jit, static_argnames=('model_apply_fn',))
def compute_loss(params, model_apply_fn, batch_states, batch_true_derivatives):
    """
    
    
    Args:
        params: NNのパラメータ
        model_apply_fn: NNのapply関数 (e.g., lnn_model.apply), paramsを受け取ることでs→Lに対応させる関数の'形状'を意味する
        batch_states(t, q, p)
    """
    
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
    
    diff_tuple = (loss_term1, loss_term2)
    diff_flat, _ = ravel_pytree(diff_tuple)
    
    loss = jnp.mean(diff_flat**2)
    return loss


@partial(jax.jit, static_argnames=('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, batch_states, batch_true_derivative):
    loss, grads = jax.value_and_grad(compute_loss)( 
        params, 
        model_apply_fn, 
        batch_states, 
        batch_true_derivative
    )
    
    # 2. update parameter
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
    
#ODEソルバーを用いて軌道を生成する
def create_trajectory(model_apply_fn, trained_params):
    H_learned = lambda s: model_apply_fn({'params': trained_params}, s) # H(s)とするためのwrapper
    ds = ham.state_derivative(H_learned)
    solver = util.ode_solver(ds)
    v_from_H_func = ham.to_velocity(H_learned)
    vmap_v_func = jax.vmap(
        lambda t, q, p: v_from_H_func((t, q, p)),
        in_axes=(0, 0, 0) # t, q, p のバッチ軸に沿って並列化
    )
    
    # @partial(jax.jit) # 必要に応じて JIT 化
    def final_hnn_solver(initial_state_lgr, t_eval, *args, **kwargs):
        """
        内部で (q,p) を計算し、(t,q,v) を返すソルバー
        """
        t0, q0, v0 = initial_state_lgr 
        initial_state_hnn = (t0, q0, v0) 

        t_traj, q_traj, p_traj = solver(initial_state_hnn, t_eval, *args, **kwargs)
        v_traj = vmap_v_func(t_traj, q_traj, p_traj)
        
        return (t_traj, q_traj, v_traj)
    return final_hnn_solver
    
    
    


