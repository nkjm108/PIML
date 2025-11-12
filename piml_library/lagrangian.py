import jax.numpy as np
from .util import compose, apply_iter, nest_iter, tuple_to_multi_arg, multi_arg_to_tuple, solve_linear_left, state_mapper
from jax import jacfwd, jacrev, jvp, grad, vmap, jit

def time(local):
    return local[0]
def coordinate(local):
    return local[1]
def velocity(local):
    return local[2]
def state(t, q, v):
    return (t,q,v)

def state_to_qv(local):
    return local[1:3]

vmap = state_mapper(state)

def Gamma(q, n=3):
    Dq_n = apply_iter(nest_iter(jacfwd, q, n-2))
    def gamma_fun(t, *args):
        return (t,) + tuple(Dq_n(t, *args))
    return gamma_fun

def Gamma_ugly(q,n=3):
    r"""
     (t -> q) -> (t -> *args -> (t, q, Dq, ...))
    """
    if n == 1:
        return lambda t,*args: t
    elif n == 2:
        return lambda t,*args: (t,q(t,*args))
    elif n == 3:
        Dq = jacfwd(q)
        return lambda t,*args: (t,q(t,*args),Dq(t,*args))
    elif n == 4:
        Dq = jacfwd(q)
        D2q = jacfwd(Dq)
        return lambda t,*args: (t,q(t,*args),Dq(t,*args),D2q(t,*args))

#重要(1.130)
def lagrangian_to_acceleration(L):  # L=(s=(t,q,v))
    L_ = tuple_to_multi_arg(L) # L_(t, q, v)
    P  = grad(L_, 2) #p = dL/dv
    F  = grad(L_, 1) #F = dL/dq
    dP_2 = jacfwd(P, 2) # \partial_2\partial_2 L 質量行列に対応
    #↑までは関数の定義で具体的な値をもつものではない
    def the_acceleration(local):
        t, q, v = local
        '''
        EL-eqを解いている
        0 = F - dP/dtのdP/dtを各成文(t, q, v)で分解し、
        ∂P/∂t, ∂P/∂q, ∂P/∂vを自動微分で計算し、方向ベクトル(1, v, 0)とのヤコビ積を取ることで、
        F - (∂P/∂t + ∂P/∂q dq/dt) = Md^2q/dt^2の形を作り出す
        '''
        b = F(t,q,v) - jvp(P, (t,q,v),  #ヤコビ・ベクトル積
                            (np.array(1.0), v, np.zeros_like(v))
                           )[1]
        M = dP_2(t, q, v)
        # Ma =bをaについて解き、aを出力
        return solve_linear_left(M, b)
    return the_acceleration

def to_momentum(L):
    L_ = tuple_to_multi_arg(L) # L_(t, q, v)
    P_ = grad(L_, 2)
    return multi_arg_to_tuple(P_)

#ここが重要。実際に利用するところ
def state_derivative(L):
    A = lagrangian_to_acceleration(L)
    def ds(local): #ds(local, tprime):
        t,q,v = local
        a = A(local) #ここでthe_accelerationの内部の計算が実行される。
        return np.array(1.0), v, a
    return ds    


#座標変換
def f_to_c(F): # F((t, x')) = x
    def C(local): # local = (t, x', v')
        # (t,x,v) = C(t, x', v') = (t, F(t,x'), \partial_0 F + \partial_1F*v')
        t, x_prime, v_prime = time(local), coordinate(local), velocity(local)
        tangent = (1.0, v_prime)
        # xにはFそのもの、vには方向微分係数(∂F/∂t, ∂F/∂x')・(1, v')に((t, x_prime),)の点を代入したもの
        #jvp(F, state, tangent)のstate, tangent各要素はタプルを要請
        x, v = jvp(F, ((t, x_prime),), (tangent,)) # (F, ((t,x'),), ((1,v'),))
        return t, x, v    
    return C
