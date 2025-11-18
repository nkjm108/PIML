# plotting.py

import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree

def plot_trajectory_comparison(t_true, q_true, v_true, t_learned, q_learned, v_learned, title_prefix=""):
    """
    学習済み軌道と真の軌道 (q, v) を時間に対してプロットします。
    """
    plt.figure(figsize=(12, 8)) #
    
    # 座標q の比較
    plt.subplot(2, 1, 1) #
    plt.plot(t_true, q_true, 'b-', label='True q(t) (Analytical)', linewidth=2) #
    plt.plot(t_learned, q_learned, 'r--', label='Learned q(t) (NN)', markersize=4) #
    plt.title(f'{title_prefix} Position (q)') #
    plt.ylabel('q') #
    plt.legend() #
    
    # 速度v の比較
    plt.subplot(2, 1, 2) #
    plt.plot(t_true, v_true, 'b-', label='True v(t) (Analytical)', linewidth=2) #
    plt.plot(t_learned, v_learned, 'r--', label='Learned v(t) (NN)', markersize=4) #
    plt.title(f'{title_prefix} Velocity (v)') #
    plt.xlabel('Time (t)') #
    plt.ylabel('v') #
    plt.legend() #
    
    plt.tight_layout() #
    plt.show() #
    
def plot_q_squared_error(t, q_true, q_learned, title_prefix=""):
    """
    学習済み軌道と真の軌道との「2乗誤差」をプロット
    """
    # ravel_pytree を使って、q がベクトルでも安全に
    q_true_flat, _ = ravel_pytree(q_true)
    q_learned_flat, _ = ravel_pytree(q_learned)
    q_error = q_learned_flat - q_true_flat
    q_squared_error = q_error ** 2
    
    plt.figure(figsize=(12, 6)) # Figureサイズを調整
    plt.plot(t, q_squared_error, 'r-', label='Squared Error (q_learned - q_true)^2')
    
    plt.title(f'{title_prefix} Position Squared Error (q)')
    plt.xlabel('Time (t)')
    plt.ylabel('Squared Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_phase_space(q_true, p_true, q_learned, p_learned, title_prefix=""):
    """
    学習済み軌道と真の軌道の位相空間 (q vs p) をプロット
    """
    
    plt.figure(figsize=(6, 6))
    plt.plot(q_true, p_true, 'b-', label='True Phase Space') 
    plt.plot(q_learned, p_learned, 'r--', label='Learned Phase Space') 
    plt.title(f'{title_prefix} Phase Space (q vs p)') 
    plt.xlabel('Position (q)') 
    plt.ylabel('Velocity (p)') 
    plt.legend() 
    plt.axis('equal') 
    plt.show() 
    
def plot_energy_comparison(t, E_true, E_learned, title_prefix=""):
    """
    
    """
    plt.figure(figsize=(10, 5))
    
    # トータルエネルギーのプロット
    plt.plot(t, E_true, 'b-', label='True', linewidth=2)
    plt.plot(t, E_learned, 'r--', label='Learned', markersize=4)

    plt.title('L or H')
    plt.xlabel('t')
    plt.ylabel('L or H')
    plt.legend()
    plt.tight_layout()
    plt.show()