# bayesian_did

Bayesian Difference-in-Differences (DiD) モデルを Python で実装したパッケージです。
共変量との交互作用を含む効果異質性 (Heterogeneous Treatment Effects) や、個体ランダム効果 (Random Intercept) を考慮したベイジアン推定を行うことができます。

## 特徴

- DiD モデルにおいて、**効果異質性** (\( \tau = \tau_0 + \tau_1^T X \)) を考慮可能
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
```python
import numpy as np
from bayesian_did import gibbs_did_cate_p3, plot_posterior_distributions
```



