#' Markov chain Monte Carlo for Binomial CAR model
#'
#' @importFrom coda mcmc
#' @importFrom stats qlogis
#' @import Matrix
#' @inheritParams psi_car
#' @inheritParams approximate_binomial_car
#' @param n_particles Number of particles used in the SMC. If set to zero, approximate MCMC is used.
#' @param n_iter Number of iterations for the MCMC.
#' @param n_burnin Number of iterations to discard as burn-in.
#' @param S A lower triangular matrix defining the Cholesky decomposition of the Gaussian proposal distribution.
#' @export
mcmc_binomial_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE,
  n_iter, n_burnin, n_particles = 0, initial_mode, max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1), S = NULL, 
  ratio_correction = TRUE, reorder = TRUE, ess_threshold=1) {
  
  y <- split(y, idx)
  u <- split(u, idx)
  n_y <- lengths(y)
  y <- list_to_matrix(y)
  u <- list_to_matrix(u)
  
  if (missing(initial_mode)) {
    initial_mode <- colMeans((y + 0.5) / (u + 1), na.rm = TRUE)
    initial_mode[is.na(initial_mode)] <- 1 #check this!!
    initial_mode <- qlogis(initial_mode) - mu * use_mu
  }
  
  if (use_mu) {
    initial_theta <- c(log(tau), log(d), mu)
  } else {
    initial_theta <- c(log(tau), log(d))
  }
  
  if (is.null(S)) {
    S <- diag(0.1 * pmax(0.1, abs(initial_theta)), length(initial_theta))
  }
  
  storage.mode(nbs) <- storage.mode(nnbs) <- storage.mode(n_y) <- "integer"
  storage.mode(y) <- storage.mode(u) <- "double"
  if (n_particles < 1) {
    out <- R_amcmc_binomial_car(nnbs, nbs, tau, d, y, n_y, u, mu, use_mu,
      n_iter, n_burnin, initial_theta, initial_mode, S, max_iter, conv_tol, 
      seed, ratio_correction, reorder)
  } else {
    out <- R_mcmc_binomial_car(nnbs, nbs, tau, d, y, n_y, u, mu, use_mu,
      n_iter, n_burnin, initial_theta, initial_mode, S, max_iter, conv_tol, 
      seed, n_particles, reorder, ess_threshold)
  }
  out$theta[,1:2] <- exp(out$theta[,1:2])
  colnames(out$theta) <- c("tau", "d", if(use_mu) "mu")
  out$theta <- mcmc(out$theta)
  out
} 
