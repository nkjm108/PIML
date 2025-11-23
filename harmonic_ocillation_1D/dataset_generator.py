# ---------------------------------------------
# ファイル名: dataset_generator.py (修正版)
# ---------------------------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
from functools import partial
import numpy as np
from jax.tree_util import tree_map

import piml_library.lagrangian as lag
import piml_library.hamiltonian as ham
import piml_library.util as util

# --- データ生成関数 (メイン) ---
def create_trajectory_datasets(
    L_analytical, #L((t, q, v))
    H_analytical,  #H((t, q, p))
    key, 
    q_dim=1, 
    num_trajectories=50, 
    N_points_per_traj=500, 
    t_end=25.0, 
    split_ratio=0.5,
    noise_std=0.1
):
    
    """
    指定された L (ラグランジアン) と H (ハミルトニアン) を使って、
    軌道データセットを生成し、時間で学習用とテスト用に分割する。
    """
    
    print("--- 学習用・テスト用データセットの生成開始 ---")

    # 軌道を計算するために使用
    ds_true = lag.state_derivative(L_analytical) 
    solver_true = util.ode_solver(ds_true)
    a_true_func = lag.lagrangian_to_acceleration(L_analytical) 
    vmap_a_true_func = jax.vmap(
        lambda t, q, v: a_true_func((t, q, v)), 
        in_axes=(0, 0, 0)
    )
    
    L_multi_arg = util.tuple_to_multi_arg(L_analytical) #L((t, q, p))→L(t, q, p)
    
    #p=∂L/∂q_dot
    p_func = jax.grad(L_multi_arg, 2)
    vmap_p_func = jax.vmap(
        lambda t, q, v: p_func(t, q, v),
        in_axes=(0,0,0)
    )
    
    #p_dot = ∂L/∂q
    p_dot_func = jax.grad(L_multi_arg, 1)
    vmap_p_dot_func = jax.vmap(
        lambda t, q, v: p_dot_func(t, q, v),
        in_axes=(0,0,0)
    )

    t_eval = jnp.linspace(0.0, t_end, N_points_per_traj)
    N_points_train = int(N_points_per_traj * split_ratio)

    # データを一時的に保存するリスト
    train_t_list, train_q_list, train_v_list, train_p_list, train_a_list = [], [], [], [], []
    test_t_list, test_q_list, test_v_list, test_p_list, test_a_list = [], [], [], [], []
    train_hnn_target_list, test_hnn_target_list = [], []
    initial_energies_list = [] 

    print(f"Generating {num_trajectories} trajectories...")

    for i in range(num_trajectories):
        
        if q_dim == 1:
            key, y_key, r_key, noise_key = jax.random.split(key, 4)
            y0 = jax.random.uniform(y_key, shape=(q_dim*2,))*2-1 #[-1.0, 1.0]の範囲で乱数を2つ生成する
            radius = jax.random.uniform(r_key)*0.9 + 0.1 #エネルギーの範囲を決定
            y0 = y0 / jnp.sqrt((y0**2).sum()) * radius #初期状態をここで決定する
            '''
            データにガウシアンノイズを加える必要がある?
            '''
            noise = jax.random.normal(noise_key, shape=y0.shape) * noise_std
            y0_noisy = y0 + noise
            initial_lgr_state = (0.0, y0_noisy[0], y0_noisy[1])
            initial_energies_list.append(radius)
        
        t_traj, q_traj, v_traj = solver_true(initial_lgr_state, t_eval)
        a_traj = vmap_a_true_func(t_traj, q_traj, v_traj)
        p_traj = vmap_p_func(t_traj, q_traj, v_traj)
        dq_dt_traj = v_traj
        dp_dt_traj = vmap_p_dot_func(t_traj, q_traj, v_traj)
        
        # 軌道を学習用 (前半) とテスト用 (後半) に分割
        train_t_list.append(t_traj[:N_points_train])
        train_q_list.append(q_traj[:N_points_train])
        train_v_list.append(v_traj[:N_points_train])
        train_p_list.append(p_traj[:N_points_train])
        train_a_list.append(a_traj[:N_points_train])
        
        test_t_list.append(t_traj[N_points_train:])
        test_q_list.append(q_traj[N_points_train:])
        test_v_list.append(v_traj[N_points_train:])
        test_p_list.append(p_traj[N_points_train:])
        test_a_list.append(a_traj[N_points_train:])
        
        train_hnn_target_list.append( (dq_dt_traj[:N_points_train], dp_dt_traj[:N_points_train]) )
        test_hnn_target_list.append( (dq_dt_traj[N_points_train:], dp_dt_traj[N_points_train:]) )

    # --- データを巨大なJAX配列に連結 ---
    train_t = jnp.concatenate(train_t_list, axis=0)
    train_q = jnp.concatenate(train_q_list, axis=0)
    train_v = jnp.concatenate(train_v_list, axis=0)
    train_targets = jnp.concatenate(train_a_list, axis=0)

    test_t = jnp.concatenate(test_t_list, axis=0)
    test_q = jnp.concatenate(test_q_list, axis=0)
    test_v = jnp.concatenate(test_v_list, axis=0)
    test_targets = jnp.concatenate(test_a_list, axis=0)

    # LNN & BNN用データ
    train_states_lnn = (train_t, train_q, train_v)
    train_targets_lnn = jnp.concatenate(train_a_list, axis=0)
    test_states_lnn = (test_t, test_q, test_v)
    test_targets_lnn = jnp.concatenate(test_a_list, axis=0)
    initial_energies = jnp.array(initial_energies_list)
    
    # HNN用データ
    train_p = jnp.concatenate(train_p_list, axis=0)
    test_p = jnp.concatenate(test_p_list, axis=0)
    train_states_hnn = (train_t, train_q, train_p)
    test_states_hnn = (test_t, test_q, test_p)
    train_targets_hnn = tree_map(lambda *x: jnp.concatenate(x, axis=0), *train_hnn_target_list)
    test_targets_hnn = tree_map(lambda *x: jnp.concatenate(x, axis=0), *test_hnn_target_list)

    initial_energies = jnp.array(initial_energies_list) # 初期エネルギー
    N_train_total = train_q.shape[0]
    N_test_total = test_q.shape[0]
    
    print(f"--- データセット生成完了 ---")
    print(f"Total Train Points: {N_train_total}")
    print(f"Total Test Points:  {N_test_total}")

    # データを辞書形式で返す
    return {
        # LNN / Standard NN 用
        "train_states_lnn": train_states_lnn,
        "train_targets_lnn": train_targets_lnn,
        "test_dataset_states_lnn": test_states_lnn,
        "test_dataset_targets_lnn": test_targets_lnn,
        
        # HNN 用
        "train_states_hnn": train_states_hnn,
        "train_targets_hnn": train_targets_hnn,
        "test_dataset_states_hnn": test_states_hnn,
        "test_dataset_targets_hnn": test_targets_hnn,
        
        # 共通
        "initial_energies": initial_energies,
        "N_train_total": N_train_total,
        "N_test_total": N_test_total
    }