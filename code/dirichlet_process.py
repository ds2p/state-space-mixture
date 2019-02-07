import numpy as np
from numba import jit, prange
from utils.distributions_jit import samp_normal, softmax, samp_uniform, samp_multinoulli
from utils.data import write_pickle
import time

# Run DPnSSM inference algorithm
def infer_dp(obs, calc_log_like, concentration, samp_base, base_logpdf, 
             n_gibbs_iters=500, n_clusters_init=1, sds_mh_proposal=None, n_aux=5, 
             max_per_line=89, dump_file=None, seed=None):
    
    # Set seed for reproducible results
    if seed:
        np.random.seed(seed)
        @jit(nopython=True, nogil=True)
        def set_seed():
            np.random.seed(seed * 2)
        set_seed()
    
    # Create batch log like function
    batch_calc_log_like = create_batch_calc_log_like(calc_log_like)
    
    # Initialize variables
    n_obs = len(obs)
    n_clusters = n_clusters_init
    cluster_ids = samp_multinoulli(n=n_obs, pvals=np.full(n_clusters, 1 / n_clusters))
    params = samp_base(n_clusters)
    n_params = params.shape[1]
    if sds_mh_proposal is None:
        sds_mh_proposal = 1 / n_params * np.ones(n_params)
    cluster_ids_samps = []
    params_samps = []
    n_clusters_samps = []
    n_accept = 0
    n_total = 0
    
    for i in range(n_gibbs_iters):
        
        t = time.time()
        
        if i > 0:
            cluster_ids = cluster_ids_samps[-1].copy()
            params = params_samps[-1].copy()
        
        # Run one iteration
        cluster_ids, params, n_clusters, accept_stats = iterate_metropolis_within_gibbs_for_dp(
            obs, cluster_ids, params, batch_calc_log_like, 
            concentration, samp_base, base_logpdf, 
            sds_mh_proposal, n_aux)
        n_accept += accept_stats[1]
        n_total += np.sum(accept_stats)
        
        # Save samples
        cluster_ids_samps.append(cluster_ids)
        params_samps.append(params)
        n_clusters_samps.append(n_clusters)
        
        # Print and log output
        cluster_counts_str = np.array2string(np.bincount(cluster_ids))
        params_str = np.array2string(params.round(1).T)
        print('iter: #%3d | time: %4.2f | num clusters: %2d | counts: %s' % (i, time.time() - t, n_clusters, cluster_counts_str))
        params_str = ' ' * 11 + ('| params: %s' % params_str.replace('\n', '\n' + ' ' * 21))
        print(params_str)
        print(' ' * 11 + ('| accept prob: %.3f' % (n_accept / n_total)))
        
        # Write pickle
        if dump_file:
            write_pickle((cluster_ids_samps, params_samps, n_clusters_samps), dump_file)
            
    return cluster_ids_samps, params_samps, n_clusters_samps


# Helper function for parallelizing batch calculation of log likelihood
def create_batch_calc_log_like(calc_log_like):
    
    @jit(nopython=True, parallel=True)
    def batch_calc_log_like(obs, params, ns):
        n_log_likes = ns.size
        log_likes = np.empty(n_log_likes)
        for n in prange(ns.size):
            log_likes[n] = calc_log_like(obs[n], params[n], ns[n])
        return log_likes
    
    return batch_calc_log_like


# Run one iteration of Metropolis-within-Gibbs inference
@jit(nopython=True, nogil=True)
def iterate_metropolis_within_gibbs_for_dp(obs, cluster_ids, params, batch_calc_log_like, 
                         concentration, samp_base, base_logpdf, 
                         sds_mh_proposal, n_aux):
   
    n_obs = len(obs)
    params_curr_loglikes = np.empty(n_obs)
    
     # Sample cluster assignment
    for n in range(n_obs):
        n_states = len(obs[n])
        
        # Drop current index and renumber all others
        cluster_ids_temp, n_clusters_temp, params_temp, counts_temp = renumber(cluster_ids, n, params)

        # Add auxiliary parameters
        params_temp = np.vstack((params_temp, samp_base(n_aux)))
        n_clusters_temp += n_aux

        # Compute prior probability of each cluster using CRP
        class_probs = np.ones((n_clusters_temp,))
        nums = np.hstack((counts_temp, np.full(n_aux, concentration / n_aux)))
        denom = n_obs - 1 + concentration
        class_probs *= nums / denom

        # Use log space for numerical stability
        log_class_probs = np.log(class_probs)

        # Compute posterior probability of each cluster using likelihood
        log_class_cond_likes = batch_calc_log_like(obs[n] * np.ones((n_clusters_temp, n_states)), 
                                                   params_temp, 
                                                   np.full(n_clusters_temp, n))
        class_post_probs = softmax(log_class_probs + log_class_cond_likes)

        # Sample new cluster identity and record parameter likelihood
        cluster_id = samp_multinoulli(1, class_post_probs)[0]
        cluster_ids[n] = cluster_id
        params_curr_loglikes[n] = log_class_cond_likes[cluster_id]

        # Update number of clusters and renumber them
        clusters = np.unique(cluster_ids)
        n_clusters = clusters.shape[0]
        for k in range(n_clusters):
            cluster_ids[cluster_ids == clusters[k]] = k
        params = params_temp[clusters]
        
    # Sample parameters
    params_prop = samp_normal(params, sds_mh_proposal)
    
    # Calculate param priors and likelihoods
    params_curr_logprobs = base_logpdf(params)
    params_prop_logprobs = base_logpdf(params_prop)
    params_prop_loglikes = batch_calc_log_like(obs, params_prop[cluster_ids], np.arange(n_obs))
    params_curr_loglikes = np.bincount(cluster_ids, params_curr_loglikes)
    params_prop_loglikes = np.bincount(cluster_ids, params_prop_loglikes)
    
    # Calculate param posteriors and Metropolis acceptances probs
    params_curr_logposts = params_curr_logprobs + params_curr_loglikes
    params_prop_logposts = params_prop_logprobs + params_prop_loglikes
    accept_probs = np.minimum(np.exp(params_prop_logposts - params_curr_logposts), np.ones(n_clusters))
    accepts = samp_uniform(np.zeros(n_clusters), np.ones(n_clusters)) < accept_probs
    accept_stats = [np.sum(~accepts), np.sum(accepts)]
    
    # Determine new parameters
    params = np.where(accepts.reshape(-1, 1) * np.ones(params.shape), params_prop, params)
            
    return cluster_ids, params, n_clusters, accept_stats


# Helper function for renumbering tables of Chinese restaurant process by dropping a given index
@jit(nopython=True, nogil=True)
def renumber(cluster_ids, idx, params):
    # Remove ith cluster assignments
    cluster_ids_temp = cluster_ids[np.arange(cluster_ids.shape[0]) != idx]
    counts_temp = np.bincount(cluster_ids_temp)
    if (cluster_ids_temp == cluster_ids[idx]).any():
        # Number of clusters remains the same
        n_clusters_temp = params.shape[0]
        return cluster_ids_temp, n_clusters_temp, params, counts_temp
    else:
        # Number of clusters decreases by one
        cluster_ids_temp[cluster_ids_temp > idx] -= 1
        params_temp = params[np.arange(params.shape[0]) != cluster_ids[idx]]
        n_clusters_temp = params_temp.shape[0]
        counts_temp = counts_temp[counts_temp > 0]
        return cluster_ids_temp, n_clusters_temp, params_temp, counts_temp