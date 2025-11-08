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
    plt.plot(t_learned, q_learned, 'r--', label='Learned q(t) (LNN)', markersize=4) #
    plt.title(f'{title_prefix} Position (q)') #
    plt.ylabel('q') #
    plt.legend() #
    
    # 速度v の比較
    plt.subplot(2, 1, 2) #
    plt.plot(t_true, v_true, 'b-', label='True v(t) (Analytical)', linewidth=2) #
    plt.plot(t_learned, v_learned, 'r--', label='Learned v(t) (LNN)', markersize=4) #
    plt.title(f'{title_prefix} Velocity (v)') #
    plt.xlabel('Time (t)') #
    plt.ylabel('v') #
    plt.legend() #
    
    plt.tight_layout() #
    plt.show() #
    
def plot_trajectory_error(t, q_true, v_true, q_learned, v_learned, title_prefix=""):
    """
    学習済み軌道と真の軌道とのズレ (誤差) をプロットします。
    """
    # ravel_pytree を使って、q や v がベクトルでも安全に差を取れるようにする
    q_true_flat, _ = ravel_pytree(q_true)
    q_learned_flat, _ = ravel_pytree(q_learned)
    v_true_flat, _ = ravel_pytree(v_true)
    v_learned_flat, _ = ravel_pytree(v_learned)

    # 誤差（ズレ）を計算
    q_error = q_learned_flat - q_true_flat
    v_error = v_learned_flat - v_true_flat
    
    plt.figure(figsize=(12, 8))
    
    # 座標q の誤差
    plt.subplot(2, 1, 1)
    plt.plot(t, q_error, 'r-', label='Error in q(t) (q_learned - q_true)')
    plt.title(f'{title_prefix} Trajectory Error: Position (q)')
    plt.ylabel('Error (q)')
    plt.legend()
    plt.grid(True)
    
    # 速度v の誤差
    plt.subplot(2, 1, 2)
    plt.plot(t, v_error, 'r-', label='Error in v(t) (v_learned - v_true)')
    plt.title(f'{title_prefix} Trajectory Error: Velocity (v)')
    plt.xlabel('Time (t)')
    plt.ylabel('Error (v)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_phase_space(q_true, v_true, q_learned, v_learned, title_prefix=""):
    """
    学習済み軌道と真の軌道の位相空間 (q vs v) をプロットします。
   
    """
    plt.figure(figsize=(6, 6))
    plt.plot(q_true, v_true, 'b-', label='True Phase Space (Energy Conservation)') #
    plt.plot(q_learned, v_learned, 'r--', label='Learned Phase Space (LNN)') #
    plt.title(f'{title_prefix} Phase Space (q vs v)') #
    plt.xlabel('Position (q)') #
    plt.ylabel('Velocity (v)') #
    plt.legend() #
    plt.axis('equal') #
    plt.show() #
    
def plot_energy_comparison(t, E_true, E_learned, title_prefix=""):
    """
    学習済み系と真の系のトータルエネルギー (E = T + V) を時間に対してプロットします。
    """
    plt.figure(figsize=(10, 5))
    
    # トータルエネルギーのプロット
    plt.plot(t, E_true, 'b-', label='True Energy (E_true)', linewidth=2)
    plt.plot(t, E_learned, 'r--', label='Learned Energy (E_learned)', markersize=4)

    plt.title(f'{title_prefix} Total Energy (T+V)')
    plt.xlabel('Time (t)')
    plt.ylabel('Energy (E)')
    plt.legend()
    plt.tight_layout()
    plt.show()