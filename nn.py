import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial
import lagrangian as lgr
import util

class BaselineNN(nn.Module):
    '''
    Input : s(t, q, v) → Output : a
    '''
    hidden_dim : int = 128
    output_dim : int = 0
    
    @nn.compact
    def __call__(self, state):
        t = lgr.time(state)
        q = lgr.coordinate(state)
        v = lgr.velocity(state)
        
        #タプルから1次元配列への変換
        q_flat, _ = ravel_pytree(q)
        v_flat, _ = ravel_pytree(v)
        
        inputs = jnp.concatenate([q_flat, v_flat])
    