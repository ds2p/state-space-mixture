import numpy as np
from utils.distributions_jit import samp_normal
from numba import jit

# Basic bootstrap particle filter with twisting parameters A, B, C
@jit(nopython=True, nogil=True)
def particle_filter(obs, var, inputs=None, A=None, B=None, C=None, 
                    n_particles=32, n_trials=50, mean_init=0, var_init=1):

    n_states = len(obs)
    curr_log_likelihood = 0

    # Define variables that will be stored and returned
    particles = np.empty((n_states, n_particles))
    log_weights = np.empty((n_states, n_particles))
    log_likelihoods = np.empty((n_states,))
    log_state_trans_norms = np.empty((n_states, n_particles))
    eff_samp_sizes = np.empty((n_states,))
    
    # If inputs not supplied, then set them equal to 0
    if inputs is None:
        inputs = np.zeros((n_states,))
    
    # If A, B, C are not supplied, then run bootstrap particle filter
    if A is None:
        A = np.zeros((n_states,))
    if B is None:
        B = np.zeros((n_states,))
    if C is None:
        C = np.zeros((n_states,))

    # Intialize state t=0
    curr_twisted_var = 1 / (1 / var_init + 2 * A[0])
    curr_twisted_means = np.full(shape=(n_particles,), 
                                 fill_value=((mean_init + inputs[0]) / var_init - B[0]) * curr_twisted_var)
    log_init_prior_norm = 1/2 * np.log(curr_twisted_var) - 1/2 * np.log(var_init) + \
            1/2 * curr_twisted_var * ((mean_init + inputs[0]) / var_init - B[0])**2 - \
            1/2 * (mean_init + inputs[0])**2 / var_init - C[0]

    for t in range(n_states):

        # Sample new particles and compute new weights
        curr_particles = samp_normal(curr_twisted_means, np.sqrt(curr_twisted_var))
        curr_log_state_dep_likes = -(n_trials - obs[t]) * curr_particles - n_trials * np.log(1 + np.exp(-curr_particles))
        # curr_log_state_dep_likes = obs[t] * curr_particles - np.exp(curr_particles)
        curr_log_weights = curr_log_state_dep_likes + A[t] * curr_particles**2 + B[t] * curr_particles + C[t]
        if t == 0:
            curr_log_weights += log_init_prior_norm
        if t == n_states-1:
            curr_log_state_trans_norms = np.zeros((n_particles,))
        else:
            next_twisted_var = 1 / (1 / var + 2 * A[t+1])
            curr_log_state_trans_norms = 1/2 * np.log(next_twisted_var) - 1/2 * np.log(var) + \
                    1/2 * next_twisted_var * ((curr_particles + inputs[t+1]) / var - B[t+1])**2 - \
                    1/2 * (curr_particles + inputs[t+1])**2 / var - C[t+1]            
        curr_log_weights += curr_log_state_trans_norms

        # Compute cumulative log likelihood and effective sample size
        max_log_weights = np.max(curr_log_weights)
        curr_weights = np.exp(curr_log_weights - max_log_weights)
        curr_log_likelihood += max_log_weights + np.log(np.mean(curr_weights))
        curr_weights_norm = curr_weights / curr_weights.sum()
        curr_ess = 1 / ((curr_weights_norm ** 2).sum())

        # Save necessary information
        particles[t] = curr_particles
        log_weights[t] = curr_log_weights
        log_likelihoods[t] = curr_log_likelihood
        log_state_trans_norms[t] = curr_log_state_trans_norms
        eff_samp_sizes[t] = curr_ess

        # Set up next iteration
        if t < n_states-1:
            curr_ancestors = systematic_resampling(curr_particles, curr_weights_norm)
            curr_twisted_var = next_twisted_var
            curr_twisted_means = ((curr_ancestors + inputs[t+1]) / var - B[t+1]) * curr_twisted_var

    return particles, log_weights, log_likelihoods, log_state_trans_norms, eff_samp_sizes

# Systematic resampling scheme
@jit(nopython=True, nogil=True)
def systematic_resampling(particles, weights):
    n_particles = weights.shape[0]
    indices = np.array([0] * n_particles)
    weights *= n_particles
    j = 0
    csw = weights[j]
    u = np.random.rand()
    for k in range(n_particles):
        while csw < u:
            j += 1
            csw += weights[j]
        indices[k] = j
        u += 1
    return particles[indices]

# Approximate backwards recursion to find approximation to optimal twisting procedure
@jit(nopython=True, nogil=True)
def approx_back_recursion(var, particles, inputs, log_weights, log_state_trans_norms, A, B, C, var_init=1):
    
    n_states, n_particles = particles.shape
    
    # Define variables that will be stored and returned
    A_new = np.empty((n_states,))
    B_new = np.empty((n_states,))
    C_new = np.empty((n_states,))

    for t in range(n_states-1, -1, -1):
        
        # Define variables (X, y) for regression
        x = particles[t]
        X = np.vstack((x**2, x, np.ones((n_particles,)))).T
        y = -log_weights[t]
        if t < n_states-1:
            next_twisted_var_new = 1 / (1 / var + 2 * A_new[t+1])
            curr_log_state_trans_norms_new = 1/2 * np.log(next_twisted_var_new) - 1/2 * np.log(var) + \
                    1/2 * next_twisted_var_new * ((x + inputs[t+1]) / var - B_new[t+1])**2 - \
                    1/2 * (x + inputs[t+1])**2 / var - C_new[t+1]
            y += -curr_log_state_trans_norms_new + log_state_trans_norms[t]
            
        # Run linear regression to get new twisting coefficients A_new, B_new, C_new
        output = np.linalg.lstsq(X, y)
        a, b, c = output[0]
#        a, b, c = np.linalg.pinv(X.T @ X) @ X.T @ y
        A_new[t] = A[t] + a
        B_new[t] = B[t] + b
        C_new[t] = C[t] + c
        
        # Check that twisting coefficients do not violate constraints
        if t == 0:
            if A_new[t] < -1 / (2 * var_init):
                raise ValueError('Constraint violated: A_new is less than it should be.')
        else:
            if A_new[t] < -1 / (2 * var):
                raise ValueError('Constraint violated: A_new is less than it should be.')
    
    return A_new, B_new, C_new

# Controlled Sequential Monte Carlo wrapper function to compute log likelihood for binomial state space model
@jit(nopython=True, nogil=True)
def bssm_log_likelihood(obs, var, inputs=None, n_particles=128, n_trials=50, mean_init=0, var_init=1, ess_threshold=80, max_iters=3):
    
    n_states = obs.shape[0]
    A = np.zeros((n_states,))
    B = np.zeros((n_states,))
    C = np.zeros((n_states,))
    
    # If inputs not supplied, then set them equal to 0
    if inputs is None:
        inputs = np.zeros((n_states,))
    
    # Run bootstrap particle filter
    particles, log_weights, log_likelihoods, log_state_trans_norms, eff_samp_sizes = particle_filter(obs, var, inputs, A, B, C, n_particles, n_trials, mean_init, var_init)
    rel_ess = eff_samp_sizes / n_particles * 100
        
    # Run controlled SMC until all ess are below threshold
    iters = 0
    while ((rel_ess < ess_threshold).any()) & (iters < max_iters):
        A, B, C = approx_back_recursion(var, particles, inputs, log_weights, log_state_trans_norms, A, B, C, var_init=1)
        particles, log_weights, log_likelihoods, log_state_trans_norms, eff_samp_sizes = particle_filter(obs, var, inputs, A, B, C, n_particles, n_trials, mean_init, var_init)
        rel_ess = eff_samp_sizes / n_particles * 100
        iters += 1
    return log_likelihoods[-1]