# bayesian_did/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_posterior_distributions(posterior):
    """
    posterior: dict
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
    sns.set_style('whitegrid')
    
    # alpha, theta, sigma^2
    fig, axes = plt.subplots(3, 2, figsize=(12,12))
    # alpha trace
    axes[0,0].plot(posterior['alpha'], color='blue', alpha=0.5)
    axes[0,0].set_title("Trace: alpha")
    # alpha hist
    sns.histplot(posterior['alpha'], kde=True, ax=axes[0,1], color='blue')
    axes[0,1].set_title("Posterior dist: alpha")
    
    # theta trace
    axes[1,0].plot(posterior['theta'], color='red', alpha=0.5)
    axes[1,0].set_title("Trace: theta")
    # theta hist
    sns.histplot(posterior['theta'], kde=True, ax=axes[1,1], color='red')
    axes[1,1].set_title("Posterior dist: theta")
    
    # sigma2 trace
    axes[2,0].plot(posterior['sigma2'], color='green', alpha=0.5)
    axes[2,0].set_title("Trace: sigma^2")
    # sigma2 hist
    sns.histplot(posterior['sigma2'], kde=True, ax=axes[2,1], color='green')
    axes[2,1].set_title("Posterior dist: sigma^2")
    
    plt.tight_layout()
    plt.show()
    
    # --- beta, tau0, tau1 などを別枠で可視化 ---
    p = posterior['beta'].shape[1]
    fig, axes = plt.subplots(p+1, 2, figsize=(10, 5*(p+1)))
    for j in range(p):
        # beta_j
        param_chain = posterior['beta'][:, j]
        axes[j,0].plot(param_chain, alpha=0.5)
        axes[j,0].set_title(f"Trace: beta[{j}]")
        sns.histplot(param_chain, kde=True, ax=axes[j,1])
        axes[j,1].set_title(f"Posterior dist: beta[{j}]")
    
    # tau0
    row_last = p
    param_t0 = posterior['tau0']
    axes[row_last,0].plot(param_t0, alpha=0.5, color='darkorange')
    axes[row_last,0].set_title("Trace: tau0")
    sns.histplot(param_t0, kde=True, ax=axes[row_last,1], color='darkorange')
    axes[row_last,1].set_title("Posterior dist: tau0")
    
    plt.tight_layout()
    plt.show()
    
    # --- tau1: p次元 ---
    fig, axes = plt.subplots(p, 2, figsize=(10, 4*p))
    for j in range(p):
        param_t1j = posterior['tau1'][:, j]
        axes[j,0].plot(param_t1j, alpha=0.5)
        axes[j,0].set_title(f"Trace: tau1[{j}]")
        sns.histplot(param_t1j, kde=True, ax=axes[j,1])
        axes[j,1].set_title(f"Posterior dist: tau1[{j}]")
    plt.tight_layout()
    plt.show()
    
    # --- sigma^2_gamma ---
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].plot(posterior['sigma2_gamma'], alpha=0.5, color='purple')
    axes[0].set_title("Trace: sigma^2_gamma")
    sns.histplot(posterior['sigma2_gamma'], kde=True, ax=axes[1], color='purple')
    axes[1].set_title("Posterior dist: sigma^2_gamma")
    plt.tight_layout()
    plt.show()
