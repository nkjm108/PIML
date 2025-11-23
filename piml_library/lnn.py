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

class LagrangianNN(nn.Module): #nn.Moduleを継承。NNの雛形
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
        
        #MLP
        x = nn.Dense(self.hidden_dim)(inputs) 
        x = nn.softplus(x) 
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.out_dim)(x)
        # (batch_size, 1) -> (batch_size,)
        return x.squeeze()
    
# defining loss function
# model_apply_fnはlnn_model.apply()という関数オブジェクトなので、jax配列として扱わない
@partial(jax.jit, static_argnames=('model_apply_fn',))
def compute_loss(params, model_apply_fn, batch_states, batch_true_accelerations):
    """
    LNNから導出される加速度と、真の加速度との間のMSE（平均二乗誤差）を計算する
    
    Args:
        params: NNのパラメータ
        model_apply_fn: NNのapply関数 (e.g., lnn_model.apply), paramsを受け取ることでs→Lに対応させる関数の'形状'を意味する
        batch_states: (t, q, v) のタプル。各要素は (Batch, ...) の形状
        batch_true_accelerations: (Batch, ...) の形状の真の加速度
    """
    
    # model_apply_fnにparamsを適応させる
    L_learned = lambda s: model_apply_fn({'params': params}, s)
    
    a_func = lag.lagrangian_to_acceleration(L_learned)
    
    # vmapを用いることで並列処理
    # q_batch = ((q0_data0, q0_data1, ...), (q1_data0, q2_data1, ....))の可能性も
    t_batch, q_batch, v_batch = batch_states
    
    vmap_a_func = jax.vmap(
        lambda t, q, v: a_func((t, q, v)), 
        in_axes=(0, 0, 0)
    )
    
    predicted_accelerations = vmap_a_func(t_batch, q_batch, v_batch)
    
    # 4. 損失（MSE）を計算
    '''
    diff((e1_data0, e1_data1, .....), (e2_data0, e2_data1, ....))
    → diff_flat(e1_data0, e1_data1, ....., e2_data0, e2_data1, ....)
    '''
    diff = tree_map(lambda pred, true: pred - true, 
                          predicted_accelerations, 
                          batch_true_accelerations)
    diff_flat, _ = ravel_pytree(diff)
    
    #各成分の2乗誤差の平均を取る
    loss = jnp.mean(diff_flat**2)
    return loss


@partial(jax.jit, static_argnames=('optimizer', 'model_apply_fn'))
def train_step(params, opt_state, optimizer, model_apply_fn, batch_states, batch_true_accel):
    # compute loss and grads
    #compute_lossに必要な引数を渡すことで、
    loss, grads = jax.value_and_grad(compute_loss)( 
        params, 
        model_apply_fn, 
        batch_states, 
        batch_true_accel
    )
    
    # 2. update parameter
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
    
#ODEソルバーを用いて軌道を生成する
#s=(t, q, v)の形
def create_trajectory(model_apply_fn, trained_params):
    L_learned = lambda s: model_apply_fn({'params': trained_params}, s) #s=(t, q, v)
    ds = lag.state_derivative(L_learned)
    solver = util.ode_solver(ds) 
    return solver

def create_trajectory_for_hnn(HNN_from_LNN_fn) : #s=(t, q, p)
    H_learned = lambda s: HNN_from_LNN_fn(s)
    ds = ham.state_derivative(H_learned)
    solver = util.ode_solver(ds)
    return solver
