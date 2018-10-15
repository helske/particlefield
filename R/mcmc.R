#' MCMC for Binomial CAR model
#'
#' @importFrom coda mcmc
#' @param nnbs Vector defining the number of neighbours for each vertex.
#' @param nbs Matrix of indices of defining neighbours for each vertex.
#' @param phi Autoregressive parameter of the latent process.
#' @param sigma Standard deviation of the AR process noise.
#' @param nu Intercept term of the AR process (stationary mean is nu / (1 - phi)).
#' @param y Vector of observations.
#' @param u Vector of exposures. Default to 1.
#' @param idx Vector defining the dependencies between y and x.
#' @param mu Intercept of the linear predictor. Defaults to 1.
#' @param initial_mode Initial mode estimate of x.
#' @param max_iter Maximum number of iterations for the approximation algorithm.
#' @param conv_tol Tolerance parameter for the approximation algorithm.
#' @param n_particles Number of particles for SMC.
#' @export
mcmc_binomial_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE,
  n_iter, n_burnin, n_particles = 0, initial_mode, max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1), S = NULL, 
  ratio_correction = TRUE, delayed_acceptance = TRUE, reorder = TRUE, ess_threshold=1) {
  
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
    initial_theta <- c(log(tau), d)
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
  out$theta <- mcmc(out$theta)
  out
} 
