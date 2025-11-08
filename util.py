import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from functools import partial, wraps, update_wrapper, reduce
from itertools import accumulate

def compose_two(f, g):
    def the_composition(*args, **kwargs):
        return f(g(*args, **kwargs))
    return the_composition

def state_ref(s, n):
    r"""Return element at position n of a state tuple

    Args:
      s : state tuple (t, a, b, c, ...)

    Returns:
      The element at potions n
    """    
    return s[n]


def nest(f, initial, n, *args, **kwargs):
    r"""Nested function evaluation, repeatedly applies `f` to `initial` (n
    times) :math:`f(f(\cdots f(x)))`

    Args:
      f : function to apply
      initial : initial argument
      n : number of times to apply `f`
      *args : tuple of additional arguments to `f`
      **kwargs : dictiionary of additional keyword arugments to `f`
    Returns:
      The nested result of applying f to initial (n times)
    """
    return reduce(lambda a,_: f(a, *args, **kwargs), range(n), initial)

def nest_iter(f, initial, n, *args, **kwargs):
    r"""Nested function evaluation, repeatedly applies `f` to `initial` (n
    times), returning iterator with all intermediate results

    .. math::

      \left(x, f(x), f(f(x)), \cdots, f(f(\cdots f(x)))\right )

    Args:
      f : function to apply
      initial : initial argument
      n : number of times to apply `f`
      *args : tuple of additional arguments to `f`
      **kwargs : dictiionary of additional keyword arugments to `f`

    Returns:
      Iterator for the accumulated application of `f` to `inital` (n times)

    Note:
      Output will be of size ``n+1``
    """
    return tuple(accumulate(range(n), func=lambda a,_:f(a, *args, **kwargs),
                            initial=initial))

def apply_iter(iterable):
    r"""Generate a function to evaluate over an iterable of functions

    The haskel-like type signature is

    .. code-block:: haskell

      map_iter :: [x->y] -> (x -> [y])

    Args:
      iterable : iterable of functions

    Returns:
      Function to evaluate the iterables, itself returning a map object
    """
    return lambda *args, **kwargs : map(lambda f: f(*args, **kwargs), iterable)

def compose(*funs):
    """Function composition

    .. code-block:: haskell

      compose :: (y->z) -> (x->y) -> ... -> (b->c) -> (a->b) -> (a->z)

    Args:
      *funs : iterator of functions to compose

    Note:
      Functions are applied from right to left

    Returns:
      The function composition

    Warnign:
      Empty composition is the identity function, but we will require `funs` to
      be non-empty
    """
    return reduce(compose_two, funs)

def tuple_to_multi_arg(f):
    """Map between a function a function of a tuple of arguments (i.e., state)
    and a function of multiple arguments

    Args:
      f : function of state ``(t,q,v,...)->l``

    Returns:
      function of multiple arguments ``t->q->v->...-> l``
    """
    def f_multi(*args):
        return f(args)
    return f_multi

def multi_arg_to_tuple(f):
    """Map between a function of multiple arguments and a function of a tuple
    of the arguments

    Args:
      f : function of multiple arguments ``t->q->v->...->l``

    Returns:
      function of tuple of arguments (i.e., state) ``(t,q,v,...)->l``
    """
    def f_tuple(local):
        return f(*local)
    return f_tuple


def ode_solver(state_derivative):
    r"""State advancer

    Returns function to integrates ode system given the state derivative.

    The Haskell-like type-signature is

    .. code-block:: haskell

     state_advancer :: (s -> ds) -> (s -> [t] -> [s])

    where time is necessarily included in the state tuple.

    Args:
      state_derivative : state derivative function ``s -> ds``

    Returns:
      function to integrate ode system gien initial state and time-grid
     ``s -> [t] -> [s]``

    Note:
      Uses the jax.experimental.odeint function (Dopri5)
    """

    # odeint wrapper, odeint expects f : x, t, *args
    f = lambda x, _, *args: state_derivative(x, *args)

    @partial(jax.jit, static_argnames=['rtol', 'atol', 'mxstep', 'hmax'])
    def solver(x0, t, *args,
               rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, hmax=jnp.inf):
        r"""ODE integrator using jax.odeint

        Args :
           x0 : initial state
           t : time grid for solution
           *args : tuple of additional params for state_derivative
           rtol : float, relative local error tolerance (optional)
           atol : float, absolute local error tolerance (optional)
           mxstep : int, maximum nubmer of steps for each timepoint (optional)
           hmax : float, maximum step stize (optional)

        Returns :
           Value of the solution (i.e., the state at each time)

        Note:
           kwargs are (rtol=relative tolerance, atol=absolute tolerance, mxstep, hmax)
        """
        return jax.experimental.ode.odeint(
            f, x0, t, *args, rtol=rtol, atol=atol, mxstep=mxstep, hmax=hmax
        )

    return solver


def principal_value(cuthigh):
    r"""Return fucntion to compute principal value between
    [cuthigh-2pi,cuthigh]

    Args:
      cuthigh : maximum value

    Returns:
      function to compute principal value ``x->x``
    """
    twopi = 2 * jnp.pi

    def fun(x):
        r"""Compute principal value"""
        y = x - (twopi * jnp.floor(x / twopi))  # y in [0,2pi]
        return jnp.where(y<cuthigh, y, y - twopi)

    return fun

def pseudosolve_linear_left(A, b):
    b, unravel_pytree = ravel_pytree(b)  # flatten pytree to 1d array
    n = b.size
    A = A.reshape(n, n)
    return unravel_pytree(jnp.linalg.pinv(A) @ b)
def directsolve_linear_left(A, b):
    b, unravel_pytree = ravel_pytree(b)  # flatten pytree to 1d array
    n = b.size
    A = A.reshape(n, n)
    return unravel_pytree(jnp.linalg.solve(A, b))
#別名で関数オブジェクトを作成
solve_linear_left = directsolve_linear_left

def compatible_zero(tree):
    return tree_map(lambda leaf: jnp.zeros_like(leaf), tree)

def state_mapper(state):
    r"""Generator for mapping function over states
      Args:
        state: state generating function t->q->vp->(t,q,vp)
    """    
    def vmap_wrapper(fun, *args, **kwargs):
        r""" Mapping function over state variables
        Args:
          fun: function of state (t,q,vp) -> ... -> f
          *args: vmap *args
          **kwargs : vmap **kwargs
        Returns:
          State function mapped over state variables, e.g., t, q, v, ...
          [t] -> [q] -> [vp] -> *extra_args
        """        
        def fun_(t, x, vp, *extra_args): 
            return fun(state(t, x, vp), *extra_args)
        return jax.vmap(fun_, *args, **kwargs) # parallelized multi-arg function
    return vmap_wrapper

def hessian(argnums_col=0, argnums_row=0):
    """Hessian matrix calculation"""

    def wrapper(fun, has_aux=False, holomorphic=False):
        return jacfwd(
            jacrev(fun, argnum_rows, has_aux=has_aux, holomorphic=holomorphic),
            argnum_cols,
            has_aux=has_aux,
            holomorphic=holomorphic,
        )

    return wrapper
