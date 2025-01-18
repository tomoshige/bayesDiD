# bayesian difference-in-difference
Bayesian Difference-in-Differences (DiD) モデルを Python で実装したパッケージです。
異質性のある因果効果 (Heterogeneous Treatment Effects) を推定することができます。

## 特徴
- DiD モデルにおいて、**効果異質性** $`\tau = \tau_0 + \tau_1^T X `$ を考慮可能
- **個体ランダム効果 (random intercept)** を組み込むことで、個体差をモデル化
- Gibbs サンプリングにより事後分布をサンプル
- 後処理として、様々なパラメータや CATE (Conditional Average Treatment Effect) の可視化・推定が可能

## インストール

GitHub リポジトリから直接インストールできます。  
（`<your_account>` の部分はご自身の GitHub アカウント名などに置き換えてください）

```bash
pip install git+https://github.com/<your_account>/bayesian_did.git
```

# 使い方

## import

```python
import numpy as np
from bayesian_did import gibbs_did_cate_p3, plot_posterior_distributions
```

## data preparation
`gibbs_did_cate_p3` 関数は、以下を入力として受け取ります。
- `X` : shape (N,p) の共変量
- `D` : shape (N, )の処置フラグ (0/1)
- `Y0`: shape (N, )のアウトカム（事前時点）
- `Y1`: shape (N, )のアウトカム（事後時点）

シミュレーション例：
``` python
N = 100
p = 3
np.random.seed(42)

# 共変量
X = np.random.normal(0, 1.0, size=(N,p))

# D の生成 (ロジットモデルを使用した一例)
w_logit = np.array([0.4, -0.1, 0.3])
logit_p = X @ w_logit
prob_d  = 1.0/(1.0 + np.exp(-logit_p))
D = (np.random.rand(N) < prob_d).astype(int)

# 真のパラメータ例
alpha_true = 2.0
theta_true = 1.0
beta_true  = np.array([0.5, -0.3, 0.8])
tau0_true  = 1.2
tau1_true  = np.array([0.5, -0.2, 0.3])
sigma_true = 1.0
sigma_gamma_true = 0.7

# 個体ランダム効果
gamma_i = np.random.normal(0, sigma_gamma_true, size=N)

# アウトカム (Y0, Y1) を生成
Y0 = np.zeros(N)
Y1 = np.zeros(N)
for i in range(N):
    tau_i = tau0_true + X[i,:] @ tau1_true
    Y0[i] = alpha_true + gamma_i[i] + X[i,:] @ beta_true + np.random.normal(0, sigma_true)
    Y1[i] = alpha_true + gamma_i[i] + theta_true + X[i,:] @ beta_true \
            + tau_i * D[i] + np.random.normal(0, sigma_true)

data = {
    'X': X,
    'D': D,
    'Y0': Y0,
    'Y1': Y1
}
```
## サンプリングの実行
```python
posterior = gibbs_did_cate_p3(
    data,
    n_iter=6000,   # 総イテレーション数
    burn_in=2000   # バーンイン
)
```
`posterior`には、以下のようなキーを持つ辞書が返ってくる。

|キー|形状|説明|
|--------|------------|-------------------------|
|$`\alpha`$|(n_samp,)|intercept のサンプル|
|$`\theta`$|(n_samp,)|time effect のサンプル|
|$`\beta`$|(n_samp,p)|共変量の係数$`\beta`$のサンプル|
|$`\tau_0`$|(n_samp,)|処置のベースライン効果 $`\tau_0`$のサンプル|
|$`\tau_1`$|(n_samp,p)|処置効果の異質性の係数 $`\tau_1`$のサンプル|
|$`\gamma`$|(n_samp,N)|個体ランダム効果$`\gamma`$のサンプル|
|$`\sigma2`$|(n_samp,)|residualsの分散$`\sigma^2`$のサンプル|
|$`\sigma2_gamma`$|(n_samp,)|個体のランダム効果の分散$`\sigma_{\gamma}^{2}`$のサンプル|


## 推定結果の可視化
```python
plot_posterior_distributions(posterior)
```
- $`\alpha`$(ベースライン)や、$`\theta`$（時点効果）などのトレース・ヒストグラムを表示
- $`\beta, \tau_0, \tau_1`$なども可視化可能。

## 結果のまとめ
```python
alpha_post = posterior['alpha'].mean()
theta_post = posterior['theta'].mean()
beta_post  = posterior['beta'].mean(axis=0)
tau0_post  = posterior['tau0'].mean()
tau1_post  = posterior['tau1'].mean(axis=0)

print(f"Posterior mean of alpha : {alpha_post:.3f}")
print(f"Posterior mean of theta : {theta_post:.3f}")
print(f"Posterior mean of beta  : {beta_post}")
print(f"Posterior mean of tau0  : {tau0_post:.3f}")
print(f"Posterior mean of tau1  : {tau1_post}")
```

## Conditional Average Treatment Effectの計算例
$`\tau(X) = \tau_0 + \tau_{1}^{\top}X`$の効果の異質性をモデル化しているため、
ある共変量ベクトル $`x`$ に対する CATE $`\tau(x)`$ は以下のように計算可能。

```python
tau0_samps = posterior['tau0']
tau1_samps = posterior['tau1']

x_candidates = [
    np.array([-1, -1, -1]),
    np.array([-1,  0,  1]),
    np.array([ 0,  0,  0]),
    np.array([ 1,  0, -1]),
    np.array([ 1,  1,  1])
]

for x_val in x_candidates:
    # サンプルごとに tau(x_val) を計算 => 事後分布に基づく平均と信頼区間を算出
    tau_x = tau0_samps + np.sum(tau1_samps * x_val, axis=1)
    mean_est = np.mean(tau_x)
    ci_lower = np.percentile(tau_x, 2.5)
    ci_upper = np.percentile(tau_x, 97.5)
    print(f"x={x_val} => CATE mean={mean_est:.3f}, 95%CI=({ci_lower:.3f}, {ci_upper:.3f})")
)
```
