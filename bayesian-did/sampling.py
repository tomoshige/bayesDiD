# bayesian_did/sampling.py

import numpy as np

def gibbs_did_cate_p3(data, n_iter=5000, burn_in=1000):
    """
    Gibbsサンプリング: 効果異質性 (tau0 + tau1^T X) & random intercept付きDiD。
    
    Parameters
    ----------
    data : dict
        {
            'X': (N, p),
            'D': (N,),
            'Y0': (N,),
            'Y1': (N,)
        }
    n_iter : int
        サンプリングの総イテレーション数
    burn_in : int
        バーンインのイテレーション数
    
    Returns
    -------
    chain_out : dict of np.ndarray
        burn-in 後のサンプルを格納した辞書
        {
            'alpha': (n_samp,),
            'theta': (n_samp,),
            'beta': (n_samp, p),
            'tau0': (n_samp,),
            'tau1': (n_samp, p),
            'gamma': (n_samp, N),
            'sigma2': (n_samp,),
            'sigma2_gamma': (n_samp,)
        }
    """
    X = data['X']
    D = data['D']
    Y0= data['Y0']
    Y1= data['Y1']
    N, p = X.shape
    
    # ハイパーパラメータ
    sigma0_sq = 100.0
    a0 = 2.0
    b0 = 2.0
    
    # 初期値
    alpha = 0.0
    theta = 0.0
    beta  = np.zeros(p)
    tau0  = 0.0
    tau1  = np.zeros(p)
    gamma_i = np.zeros(N)
    sigma2 = 1.0
    sigma2_gamma = 1.0
    
    # 結果格納用
    chain = {
        'alpha': [],
        'theta': [],
        'beta': [],
        'tau0': [],
        'tau1': [],
        'gamma': [],
        'sigma2': [],
        'sigma2_gamma': []
    }
    
    def rinvgamma(shape, rate):
        return 1.0 / np.random.gamma(shape, 1.0/rate)
    
    for it in range(n_iter):
        # (1) gamma_i の更新
        for i in range(N):
            # 残差
            r0 = Y0[i] - (alpha + X[i,:]@beta)
            r1 = Y1[i] - (alpha + theta + X[i,:]@beta + (tau0 + X[i,:]@tau1)*D[i])
            prec = 2.0/sigma2 + 1.0/sigma2_gamma
            var_i = 1.0 / prec
            mu_i  = (r0 + r1)/sigma2 * var_i
            gamma_i[i] = np.random.normal(mu_i, np.sqrt(var_i))
        
        # (2) (alpha,theta,beta,tau0,tau1) の更新 (block)
        dim_w = 2 + p + 1 + p  # = 2p + 3
        Z = np.zeros((2*N, dim_w))
        y = np.zeros(2*N)
        
        # w = [alpha, theta, beta(1..p), tau0, tau1(1..p)]
        for i in range(N):
            # Y0
            y[2*i] = Y0[i] - gamma_i[i]
            Z[2*i, 0] = 1.0  # alpha
            Z[2*i, 1] = 0.0  # theta
            Z[2*i, 2:2+p] = X[i,:]   # beta
            Z[2*i, 2+p]   = 0.0      # tau0
            Z[2*i, (2+p+1):(2+p+1+p)] = 0.0  # tau1
            
            # Y1
            y[2*i+1] = Y1[i] - gamma_i[i]
            Z[2*i+1, 0] = 1.0
            Z[2*i+1, 1] = 1.0
            Z[2*i+1, 2:2+p] = X[i,:]
            Z[2*i+1, 2+p]   = D[i]  # tau0
            Z[2*i+1, (2+p+1):(2+p+1+p)] = X[i,:]*D[i]  # tau1 * X_i * D_i
        
        prec_prior = (1.0/sigma0_sq)*np.eye(dim_w)
        prec_lik   = (Z.T @ Z)/sigma2
        V_inv      = prec_prior + prec_lik
        V          = np.linalg.inv(V_inv)
        
        mean_part  = (Z.T @ y)/sigma2
        w_hat      = V @ mean_part
        
        w_sample   = np.random.multivariate_normal(w_hat, V)
        
        alpha = w_sample[0]
        theta = w_sample[1]
        beta  = w_sample[2:2+p]
        tau0  = w_sample[2+p]
        tau1  = w_sample[(2+p+1):(2+p+1+p)]
        
        # (3) sigma^2 の更新
        rss = 0.0
        for i in range(N):
            r0 = Y0[i] - (alpha + gamma_i[i] + X[i,:]@beta)
            r1 = Y1[i] - (alpha + gamma_i[i] + theta + X[i,:]@beta 
                          + (tau0 + X[i,:]@tau1)*D[i])
            rss += r0**2 + r1**2
        shape_post = a0 + (2*N)/2
        rate_post  = b0 + 0.5*rss
        sigma2     = rinvgamma(shape_post, rate_post)
        
        # (4) sigma_gamma^2 の更新
        g_sumsq = np.sum(gamma_i**2)
        shape_g = a0 + N/2
        rate_g  = b0 + 0.5*g_sumsq
        sigma2_gamma = rinvgamma(shape_g, rate_g)
        
        # チェーン格納
        chain['alpha'].append(alpha)
        chain['theta'].append(theta)
        chain['beta'].append(beta.copy())
        chain['tau0'].append(tau0)
        chain['tau1'].append(tau1.copy())
        chain['gamma'].append(gamma_i.copy())
        chain['sigma2'].append(sigma2)
        chain['sigma2_gamma'].append(sigma2_gamma)
    
    # burn-in 後のサンプルを返す
    chain_out = {}
    for k,v in chain.items():
        chain_out[k] = np.array(v)[burn_in:]
    return chain_out
