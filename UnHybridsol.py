#5032_ir


import numpy as np
from enum import Enum, auto
from typing import List, Callable, Union, Optional
import time

# --- 設定と定数 ---

class PropagationMethod(Enum):
    LINEAR = auto()       # 線形誤差伝播（高速、ガウス分布近似）
    MONTE_CARLO = auto()  # モンテカルロ法（高精度、非線形対応、計算コスト大）
    AUTO = auto()         # 状況に応じて自動選択

class SDEType(Enum):
    ITO = auto()          # 伊藤積分
    STRATONOVICH = auto() # ストラトノビッチ積分

# --- コアエンジン: 計算グラフとコンテキスト管理 ---

class ComputationContext:
    """計算のコンテキスト（手法、設定）を管理するシングルトン的クラス"""
    def __init__(self, method: PropagationMethod = PropagationMethod.LINEAR, mc_samples: int = 10000):
        self.method = method
        self.mc_samples = mc_samples
        self.seed = 42
        self._rng = np.random.default_rng(self.seed)

    def set_method(self, method: PropagationMethod):
        self.method = method

    def get_rng(self):
        return self._rng

# グローバルコンテキスト
_ctx = ComputationContext()

def set_global_context(method: PropagationMethod = PropagationMethod.LINEAR, samples: int = 10000):
    _ctx.method = method
    _ctx.mc_samples = samples
    _ctx._rng = np.random.default_rng(_ctx.seed)

# --- 変数クラス (Uncertain Variable) ---

class UVar:
    """
    不確実性を持つ変数 (Uncertain Variable)。
    値(mean)と不確実性(std/variance)を持ち、演算時に自動的に誤差を伝播させる。
    """
    def __init__(self, mean: float, std: float = 0.0, samples: Optional[np.ndarray] = None, name: str = "Var"):
        self.mean = mean
        self.std = std
        self.name = name
        
        # モンテカルロ用サンプルの初期化
        if samples is not None:
            self.samples = samples
            # サンプルから統計量を再計算して整合性を取ることも可能だが、ここでは初期値を優先
        else:
            # まだ生成しない（遅延生成、またはメソッドに応じて生成）
            self.samples = None

    def _ensure_samples(self):
        """モンテカルロモードが必要な場合にサンプルを生成または取得"""
        if self.samples is None:
            rng = _ctx.get_rng()
            # 正規分布を仮定して初期サンプル生成
            self.samples = rng.normal(self.mean, self.std, _ctx.mc_samples)
        return self.samples

    def __repr__(self):
        return f"UVar({self.name}: {self.mean:.4f} ± {self.std:.4f})"

    # --- 演算子のオーバーロード ---
    # これにより、通常の数式 (c = a + b) で計算グラフが構築・実行される

    def _propagate(self, other, op_func_linear, op_func_mc, op_name):
        """
        誤差伝播の共通ロジック。
        Global Contextの設定(AUTO/LINEAR/MC)に基づいて計算手法をスイッチする。
        """
        # AUTOモードの簡易ロジック: 
        # 相手が定数ならLinear、相手もUVarでかつ非線形演算ならMC...などの判定を入れる場所。
        # ここでは簡易的に、「非線形関数」かつ「標準偏差が大きい(>10%ofMean)」場合にMCを選択する等のヒューリスティックが可能。
        
        current_method = _ctx.method
        
        # 相手が数値型の場合はUVar(定数, std=0)に変換
        if isinstance(other, (int, float)):
            other = UVar(other, 0.0, name="Const")
        
        if not isinstance(other, UVar):
            raise TypeError(f"Unsupported type for operation: {type(other)}")

        # --- 自動選択ロジック (Heuristic) ---
        if current_method == PropagationMethod.AUTO:
            # 簡易ヒューリスティック: 
            # 変動係数が大きい(>0.2)場合、または演算が高度に非線形(exp, sin等)な場合にMCを選択
            cv_self = abs(self.std / self.mean) if self.mean != 0 else 0
            cv_other = abs(other.std / other.mean) if other.mean != 0 else 0
            
            is_nonlinear_op = op_name in ['mul', 'div', 'exp', 'sin', 'cos', 'pow']
            high_variance = (cv_self > 0.2) or (cv_other > 0.2)

            if is_nonlinear_op and high_variance:
                use_method = PropagationMethod.MONTE_CARLO
            else:
                use_method = PropagationMethod.LINEAR
        else:
            use_method = current_method

        # --- 計算実行 ---
        if use_method == PropagationMethod.MONTE_CARLO:
            # モンテカルロ法
            s_self = self._ensure_samples()
            s_other = other._ensure_samples()
            
            new_samples = op_func_mc(s_self, s_other)
            new_mean = np.mean(new_samples)
            new_std = np.std(new_samples)
            return UVar(new_mean, new_std, samples=new_samples, name=f"({self.name}_{op_name}_{other.name})")

        else:
            # 線形誤差伝播法 (Linear Error Propagation)
            # σ_f^2 = (df/dx)^2 * σ_x^2 + (df/dy)^2 * σ_y^2 + 2*cov... (共分散は簡易化のため0と仮定)
            new_mean, new_std = op_func_linear(self, other)
            return UVar(new_mean, new_std, name=f"({self.name}_{op_name}_{other.name})")

    # --- 基本演算 ---

    def __add__(self, other):
        def linear_op(x, y):
            val = x.mean + y.mean
            var = x.std**2 + y.std**2 # 加算の分散は和
            return val, np.sqrt(var)
        
        def mc_op(sx, sy): return sx + sy

        return self._propagate(other, linear_op, mc_op, "add")

    def __sub__(self, other):
        def linear_op(x, y):
            val = x.mean - y.mean
            var = x.std**2 + y.std**2 # 減算の分散も和
            return val, np.sqrt(var)
        
        def mc_op(sx, sy): return sx - sy

        return self._propagate(other, linear_op, mc_op, "sub")

    def __mul__(self, other):
        def linear_op(x, y):
            val = x.mean * y.mean
            # 積の誤差伝播 (近似式)
            # (σf/f)^2 = (σx/x)^2 + (σy/y)^2
            if val == 0: return 0.0, 0.0
            rel_var = (x.std/x.mean)**2 + (y.std/y.mean)**2 if x.mean!=0 and y.mean!=0 else 0
            return val, abs(val) * np.sqrt(rel_var)

        def mc_op(sx, sy): return sx * sy

        return self._propagate(other, linear_op, mc_op, "mul")

    def __truediv__(self, other):
        def linear_op(x, y):
            val = x.mean / y.mean
            # 商の誤差伝播
            if y.mean == 0: raise ValueError("Division by zero mean variable")
            rel_var = (x.std/x.mean)**2 + (y.std/y.mean)**2 if x.mean!=0 else 0
            return val, abs(val) * np.sqrt(rel_var)

        def mc_op(sx, sy): return sx / sy

        return self._propagate(other, linear_op, mc_op, "div")

    # --- 単項演算と特殊関数 ---
    
    def exp(self):
        """Exponential function e^x"""
        # 単項演算用のヘルパー
        dummy = UVar(0, 0) # ダミー
        
        def linear_op(x, _):
            val = np.exp(x.mean)
            # d(e^x)/dx = e^x -> sigma = e^x * sigma_x
            std = val * x.std
            return val, std

        def mc_op(sx, _): return np.exp(sx)

        return self._propagate(dummy, linear_op, mc_op, "exp")
    
    def log(self):
        """Natural logarithm ln(x)"""
        dummy = UVar(0, 0)
        def linear_op(x, _):
            val = np.log(x.mean)
            # d(ln x)/dx = 1/x -> sigma = sigma_x / x
            std = abs(x.std / x.mean)
            return val, std
        def mc_op(sx, _): return np.log(sx)
        return self._propagate(dummy, linear_op, mc_op, "log")


# --- SDE Solver (確率微分方程式ソルバー) ---

class SDESolver:
    """
    確率微分方程式 (SDE) を解くためのクラス。
    形式: dX_t = drift(X_t, t)dt + diffusion(X_t, t)dW_t
    """
    
    @staticmethod
    def solve(
        initial_val: UVar,
        drift_func: Callable[[float, float], float],    # mu(x, t)
        diffusion_func: Callable[[float, float], float], # sigma(x, t)
        t_span: tuple,
        dt: float = 0.01,
        method: SDEType = SDEType.ITO
    ) -> dict:
        """
        Euler-Maruyama法を用いてSDEをシミュレーションし、不確実性の時間発展を計算する。
        戻り値は時刻ごとのUVarのリスト。
        
        ここでは内部的にモンテカルロパスを多数生成して分布を追跡するアプローチを取る。
        """
        t0, t_end = t_span
        num_steps = int((t_end - t0) / dt)
        ts = np.linspace(t0, t_end, num_steps)
        
        # モンテカルロパスの初期化
        current_samples = initial_val._ensure_samples()
        rng = _ctx.get_rng()
        num_paths = len(current_samples)
        
        history_mean = []
        history_std = []
        
        current_t = t0
        
        # 時間発展ループ
        for _ in range(num_steps):
            # 現在のサンプル群に対するドリフトと拡散係数
            # ベクトル化演算を想定
            drift = drift_func(current_samples, current_t)
            diffusion = diffusion_func(current_samples, current_t)
            
            # Wiener過程の増分 dW ~ N(0, dt)
            dW = rng.normal(0, np.sqrt(dt), num_paths)
            
            # Euler-Maruyama Update
            # X_{t+1} = X_t + mu*dt + sigma*dW
            current_samples = current_samples + drift * dt + diffusion * dW
            
            # 統計量を記録
            history_mean.append(np.mean(current_samples))
            history_std.append(np.std(current_samples))
            
            current_t += dt
            
        # 結果のパッケージング
        result = {
            "time": ts,
            "mean": np.array(history_mean),
            "std": np.array(history_std),
            "final_uvar": UVar(np.mean(current_samples), np.std(current_samples), samples=current_samples, name="SDE_Result")
        }
        return result

# --- ユーザー向けラッパー関数 ---

def sin(x: UVar):
    dummy = UVar(0, 0)
    def linear_op(v, _):
        val = np.sin(v.mean)
        std = abs(np.cos(v.mean) * v.std)
        return val, std
    def mc_op(sx, _): return np.sin(sx)
    return x._propagate(dummy, linear_op, mc_op, "sin")

def cos(x: UVar):
    dummy = UVar(0, 0)
    def linear_op(v, _):
        val = np.cos(v.mean)
        std = abs(-np.sin(v.mean) * v.std)
        return val, std
    def mc_op(sx, _): return np.cos(sx)
    return x._propagate(dummy, linear_op, mc_op, "cos")

# --- デモ実行用 ---

if __name__ == "__main__":
    print("=== FluxUncertainty Engine Demo ===")
    
    # 1. 線形誤差伝播の例
    set_global_context(PropagationMethod.LINEAR)
    print("\n--- [Linear Mode] Basic Calculation ---")
    a = UVar(10.0, 0.5, name="A")
    b = UVar(5.0, 0.2, name="B")
    c = a * b + UVar(2.0, 0.1)
    print(f"Result (Linear): {c}")
    
    # 2. 自動選択モード (非線形性が強い場合)
    set_global_context(PropagationMethod.AUTO)
    print("\n--- [Auto Mode] Non-linear Calculation (exp) ---")
    x = UVar(2.0, 0.5, name="X") # 分散が大きい
    # exp(x) は非線形で分散が大きいとLinear近似の誤差が大きくなる -> AutoでMCが選ばれるはず
    y = x.exp() 
    print(f"Result (Auto-Select): {y}")
    
    # 確認: 手動でLinearと比較
    set_global_context(PropagationMethod.LINEAR)
    y_lin = x.exp()
    print(f"Comparison (Forced Linear): {y_lin}")
    print(f"-> Difference due to non-linearity skew: Mean diff = {abs(y.mean - y_lin.mean):.4f}")

    # 3. SDE ソルバー (幾何ブラウン運動: dS = mu*S*dt + sigma*S*dW)
    print("\n--- [SDE Solver] Geometric Brownian Motion ---")
    set_global_context(PropagationMethod.MONTE_CARLO, samples=5000)
    
    S0 = UVar(100.0, 0.0, name="StockPrice") # 初期値は確定的
    mu = 0.05
    sigma = 0.2
    
    def drift(x, t): return mu * x
    def diffusion(x, t): return sigma * x
    
    start_time = time.time()
    result = SDESolver.solve(S0, drift, diffusion, t_span=(0, 1.0), dt=0.01)
    end_time = time.time()
    
    final_res = result["final_uvar"]
    
    # 理論値 (対数正規分布の期待値): E[St] = S0 * e^(mu*t)
    theoretical_mean = 100.0 * np.exp(0.05 * 1.0)
    
    print(f"Simulation Time: {end_time - start_time:.4f}s")
    print(f"SDE Final State (t=1.0): {final_res}")
    print(f"Theoretical Mean: {theoretical_mean:.4f}")
    

