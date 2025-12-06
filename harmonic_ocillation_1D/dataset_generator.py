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

class DataScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # データの平均と標準偏差を計算 (Time, Dim) or (Batch, Dim)
        self.mean = jnp.mean(data, axis=0)
        self.std = jnp.std(data, axis=0) + 1e-6 # ゼロ除算防止

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

def create_trajectory_datasets(
    L_analytical, # L((t, q, v))
    H_analytical, # H((t, q, p))
    key, 
    q_dim=1, 
    num_trajectories=50, 
    N_points_per_traj=30, # 論文の設定(30点)
    t_end=3.0,            # 論文の設定(dt=0.1 * 30 = 3.0)
    test_split_ratio=0.5, # 軌道の何割をテストにするか
    noise_std=0.1
):
    
    """
    指定された L (ラグランジアン) と H (ハミルトニアン) を使って、
    独立した軌道データセットを生成し、軌道単位で学習用とテスト用に分割して返す。
    
    Args:
        test_split_ratio (float): 全軌道数のうち、テストデータにする割合 (0.0 ~ 1.0)
    """
    
    print("--- 学習用・テスト用データセットの生成開始 ---")

    # --- 1. 物理量の計算関数を定義 ---
    # 運動方程式 (Lagrangian -> ODE)
    ds_true = lag.state_derivative(L_analytical) 
    solver_true = util.ode_solver(ds_true)
    
    # 加速度 (LNN/BNN Target)
    a_true_func = lag.lagrangian_to_acceleration(L_analytical) 
    vmap_a_true_func = jax.vmap(lambda t, q, v: a_true_func((t, q, v)), in_axes=(0, 0, 0))
    
    # 解析的な p と p_dot (HNN Target)
    L_multi_arg = util.tuple_to_multi_arg(L_analytical) # L(t, q, v)
    
    # p = ∂L/∂v
    p_func = jax.grad(L_multi_arg, 2)
    vmap_p_func = jax.vmap(lambda t, q, v: p_func(t, q, v), in_axes=(0,0,0))
    
    # p_dot = ∂L/∂q
    p_dot_func = jax.grad(L_multi_arg, 1)
    vmap_p_dot_func = jax.vmap(lambda t, q, v: p_dot_func(t, q, v), in_axes=(0,0,0))

    # 時間軸 (全期間) ここのやり方が多分おかしい
    t_eval = jnp.linspace(0.0, t_end, N_points_per_traj)
    
    # 全軌道のデータを一時保存するリスト
    all_t, all_q, all_v, all_p, all_a = [], [], [], [], []
    all_hnn_targets = []
    initial_energies_list = [] 

    print(f"Generating {num_trajectories} independent trajectories...")

    # --- 2. 軌道生成ループ ---
    for i in range(num_trajectories):
        
        # 初期値生成
        if q_dim == 1:
            key, y_key, r_key, noise_key = jax.random.split(key, 4)
            
            # ランダムな方向とエネルギー(半径)を決定
            y0 = jax.random.uniform(y_key, shape=(q_dim*2,))*2-1 # [-1.0, 1.0]
            radius = jax.random.uniform(r_key)*0.9 + 0.1         # エネルギー範囲 [0.2, 1.0]
            y0 = y0 / jnp.sqrt((y0**2).sum()) * radius           # 正規化してスケーリング
            
            initial_lgr_state = (0.0, y0[0], y0[1])
            initial_energies_list.append(radius)
        
        # ODEソルバーで軌道を計算
        t_traj, q_clean, v_clean = solver_true(initial_lgr_state, t_eval)
        
        # 各物理量（真値）を計算
        a_clean = vmap_a_true_func(t_traj, q_clean, v_clean)       
        p_clean = vmap_p_func(t_traj, q_clean, v_clean)             
        dp_dt_clean = vmap_p_dot_func(t_traj, q_clean, v_clean)     
        dq_dt_clean = v_clean                                 # 位置時間微分 (HNN Target)
        
        # 軌道全体に観測ノイズを加える
        # それぞれの物理量に対して個別にノイズを生成
        key, nq_key, nv_key, np_key = jax.random.split(key, 4)
        q_noisy = q_clean + jax.random.normal(nq_key, shape=q_clean.shape) * noise_std
        v_noisy = v_clean + jax.random.normal(nv_key, shape=v_clean.shape) * noise_std
        p_noisy = p_clean + jax.random.normal(np_key, shape=p_clean.shape) * noise_std
        
        # リストに追加
        # 入力データ(states)には「ノイズあり」を使う
        all_t.append(t_traj)
        all_q.append(q_noisy) 
        all_v.append(v_noisy)
        all_p.append(p_noisy)
        
        #ターゲットに使うからクリーンなものを
        all_a.append(a_clean) 
        all_hnn_targets.append((dq_dt_clean, dp_dt_clean))

    # Train / Test 分割と結合
    # 軌道単位で分割数を決定
    num_test = int(num_trajectories * test_split_ratio)
    num_train = num_trajectories - num_test
    
    print(f"Splitting: {num_train} Train trajectories / {num_test} Test trajectories")

    # リストをスライスして結合するヘルパー関数
    def concat_split(data_list, idx_start, idx_end):
        # 該当範囲のリストを結合 (Batch, Time, Dim) -> (Batch*Time, Dim)
        return jnp.concatenate(data_list[idx_start:idx_end], axis=0)

    # Train Data (前半 num_train 本)
    train_t = concat_split(all_t, 0, num_train)
    train_q = concat_split(all_q, 0, num_train)
    train_v = concat_split(all_v, 0, num_train)
    train_p = concat_split(all_p, 0, num_train)
    
    # Targets
    train_targets_lnn = concat_split(all_a, 0, num_train)
    train_dq_dt = concat_split([x[0] for x in all_hnn_targets], 0, num_train)
    train_dp_dt = concat_split([x[1] for x in all_hnn_targets], 0, num_train)
    train_targets_hnn = (train_dq_dt, train_dp_dt)

    # Test Data (後半 num_test 本)
    test_t = concat_split(all_t, num_train, num_trajectories)
    test_q = concat_split(all_q, num_train, num_trajectories)
    test_v = concat_split(all_v, num_train, num_trajectories)
    test_p = concat_split(all_p, num_train, num_trajectories)
    
    # Targets
    test_targets_lnn = concat_split(all_a, num_train, num_trajectories)
    test_dq_dt = concat_split([x[0] for x in all_hnn_targets], num_train, num_trajectories)
    test_dp_dt = concat_split([x[1] for x in all_hnn_targets], num_train, num_trajectories)
    test_targets_hnn = (test_dq_dt, test_dp_dt)

    # 状態タプルにまとめる
    train_states_lnn = (train_t, train_q, train_v)
    test_states_lnn = (test_t, test_q, test_v)
    
    train_states_hnn = (train_t, train_q, train_p)
    test_states_hnn = (test_t, test_q, test_p)

    initial_energies = jnp.array(initial_energies_list)
    N_train_total = train_q.shape[0]
    N_test_total = test_q.shape[0]
    
    print(f"--- データセット生成完了 ---")
    print(f"Total Train Points: {N_train_total}")
    print(f"Total Test Points:  {N_test_total}")

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