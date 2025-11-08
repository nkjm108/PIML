import jax 
from jax import numpy as np
from jax import grad, jvp, jacrev, jacfwd
from util import compose, tuple_to_multi_arg, principal_value, ode_solver, principal_value, compatible_zero, solve_linear_left, state_mapper
import lagrangian as lgr


def state(t,q,p):
    return (t,q,p)
def time(s):
    return s[0]
def coordinate(s):
    return s[1]
def momentum(s):
    return s[2]
def state_to_qp(s):
    return s[1:3]

vmap = state_mapper(state)

def lagrangian_state_to_hamiltonian_state(L):
    L_ = tuple_to_multi_arg(L) # L_(t, q, v)
    P_ = grad(L_, 2) # P(t,q,v) = (\partial_2 L_)(t, q, v)
    def Hstate(lstate):
        t,q,v = lgr.time(lstate), lgr.coordinate(lstate), lgr.velocity(lstate)
        p = P_(t, q, v)
        return state(t, q, p)
    return Hstate

def hamiltonian_state_to_lagrangian_state(H):
    H_ = tuple_to_multi_arg(H)
    V_ = grad(H_, 2) # V(t, q, p) = (\partial_2 H_)(t, q, p)
    def Lstate(hstate):
        t,q,p = time(hstate), coordinate(hstate), momentum(hstate)
        return lgr.state(t, q, V_(t, q, p))
    return Lstate

def to_velocity(H):
    H_ = tuple_to_multi_arg(H)
    V_ = grad(H_, 2) # V(t, q, p) = (\partial_2 H_)(t, q, p)
    return multi_arg_to_tuple(V_)

def state_derivative(H): # H((t, q, p))
    # qdot = \partial_2 H, pdot = -\partial_1 H
    H_ = tuple_to_multi_arg(H) # H_(t, q, p)
    dH1_ = grad(H_, 1)
    dH2_ = grad(H_, 2)
    def ds(s):
        return state(np.array(1.0), dH2_(*s), -dH1_(*s))
    return ds

def legendre_transformation(F):
    r"""
    F is a real function of v, 
    w = DF(v)
    v = DG(w)
    F,G are related by Legendre transformation

    F(v) = 1/2 v.M.v + b v + c
    w(v) = DF(v) = Mv + b
    M = Dw(v)   ※Mはvの値によらない定数行列なので、v=0の点で計算しても良い。
    b = w(0)

    v = M^{-1}(w-b) = V(w)
    Mv= w-b

    G(w) = w V(w) - F(V(w))
    """
    w_of_v = jacrev(F) #w(v) = DF(v)
    M_of_v = jacfwd(w_of_v) #M = Dw(v)
    
    def G(w):
        zero = compatible_zero(w)
        M = M_of_v(zero)
        b = w_of_v(zero)
        v = solve_linear_left(M, w-b)
        return np.sum(w*v) - F(v)
    return G

# H = Σpq_dot - L
#F = L(q_dot), w = p, v = q_dot
def lagrangian_to_hamiltonian(the_lagrangian):
    def the_hamiltonian(hstate):
        t = time(hstate)
        q = coordinate(hstate)
        p = momentum(hstate)
        L = lambda qdot : the_lagrangian(lgr.state(t, q, qdot))
        return legendre_transformation(L)(p)
    return the_hamiltonian

# L = Σq_dotp - H
#F = H(p), w = q_dot, v = pとしている
def hamiltonian_to_lagrangian(the_hamiltonian):
    def the_lagrangian(lstate):
        t = lgr.time(lstate)
        q = lgr.coordinate(lstate)
        v = lgr.velocity(lstate)
        H = lambda p : the_hamiltonian(state(t, q, p))
        return legendre_transformation(H)(v)
    return the_lagrangian
