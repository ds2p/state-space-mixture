import numpy as np
from numba import jit
from utils.data import load_network
from utils.distributions_jit import samp_normal, samp_uniform, normal_logpdf, uniform_logpdf
from binomial_ssm_csmc import bssm_log_likelihood
from dirichlet_process import infer_dp

SEED = 1236
np.random.seed(SEED)

network_df = load_network()
obs_df = network_df.apply(lambda x: x.sum(1))

obs_all = obs_df.values.tolist()
shock_trial = -30

# base distribution (G)
@jit(nopython=True, nogil=True)
def samp_base(n):
    jumps = samp_normal(np.zeros(n), 2 * np.ones(n))
    variances = samp_uniform(np.full(n, -15), np.full(n, 0))
    all_samps = np.vstack((jumps, variances)).T
    return all_samps

@jit(nopython=True, nogil=True)
def base_logpdf(params):
    n = params.shape[0]
    jump_logpdf = normal_logpdf(params[:, 0], np.zeros(n), 2 * np.ones(n))
    variances_logpdf = uniform_logpdf(params[:, 1], np.full(n, -15), np.full(n, 0))
    total_logpdf = jump_logpdf + variances_logpdf
    return total_logpdf

# Log likelihood function
@jit(nopython=True, nogil=True)
def calc_bssm_log_like_shock_mod_cue(ob, param, n):
    jump, log_var = param
    p_init = ob[:shock_trial].sum() / (2000 * (ob.shape[0] + shock_trial))
    mean_init = np.log(p_init / (1 - p_init))
    var = np.exp(log_var)
    return bssm_log_likelihood(ob[shock_trial:], var=var, n_particles=64, n_trials=2000, 
                               max_iters=3, mean_init=mean_init+jump, var_init=1e-10)

output = infer_dp(obs_all, calc_bssm_log_like_shock_mod_cue, 1.0, samp_base, base_logpdf, n_gibbs_iters=10000, 
                  sds_mh_proposal=1/2 * np.array([0.5, 0.5]), n_aux=5, dump_file='../pickle/shock_mod_cue.p', seed=SEED)
