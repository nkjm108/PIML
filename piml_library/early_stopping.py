import numpy as np
import os
import pickle
import warnings

class EarlyStopping:
    """
    検証用損失が改善しなくなった時点で学習を早期終了させるクラス。
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pkl', mode='min', save_best_on_disk=True):
        """
        Args:
            patience (int): 改善が見られなくなってから停止するまでの待機エポック数。
            verbose (bool): Trueの場合、改善メッセージを出力する。
            delta (float): 改善とみなされる最小の変化量。これより小さい変化は改善とみなさない。
            path (str): ベストモデルを保存するファイルパス。
            mode (str): 'min' (Lossなど、低い方が良い) または 'max' (Accuracyなど、高い方が良い)。
            save_best_on_disk (bool): Trueならベストモデルをディスクに保存、Falseならメモリに保持。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.save_best_on_disk = save_best_on_disk
        self.best_params = None # メモリ保持用

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta = -delta # スコア(符号反転後)に対するdelta調整
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta = delta
        else:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_val, model_params=None):
        """
        Args:
            current_val (float): 監視する指標の値（Validation Lossなど）
            model_params (object): 保存するモデルのパラメータ（JAXのparamsやPyTorchのstate_dict）
        """
        # 内部処理のため 'min' モードなら符号を反転させて「大きい方が良い」スコアとして扱う実装例もありますが、
        # ここでは直感的にわかりやすく、monitor_opで直接比較します。
        
        score = current_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model_params)
        
        # 改善しなかった場合
        elif not self.monitor_op(score, self.best_score - self.delta if self.monitor_op == np.less else self.best_score + self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        # 改善した場合
        else:
            self.save_checkpoint(score, model_params)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model_params):
        '''検証スコアが改善したときにモデルを保存する'''
        if self.verbose:
            if self.val_loss_min == np.inf:
                 print(f'Validation metric set to {score:.6f}.  Saving model ...')
            else:
                print(f'Validation metric improved from {self.val_loss_min:.6f} to {score:.6f}.  Saving model ...')
        
        self.val_loss_min = score
        
        if model_params is not None:
            if self.save_best_on_disk:
                # 汎用的にpickleで保存（フレームワーク固有の保存方法を使いたい場合はここを書き換える）
                # JAXの場合: flax.serialization.save_msgpack などが推奨されますが、pickleでも動作はします
                try:
                    with open(self.path, 'wb') as f:
                        pickle.dump(model_params, f)
                except Exception as e:
                    warnings.warn(f"Failed to save checkpoint to disk: {e}")
            else:
                # メモリ上に保存（JAXなどのImmutableなオブジェクトの場合はこれでOK）
                # 注意: PyTorchなどの場合、deepcopyが必要なことがあります
                self.best_params = model_params