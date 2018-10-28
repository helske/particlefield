#' Twisted SMC for CAR model with Binomial observations
#' 
#' @inheritParams approximate_binomial_car
#' @param nbs Matrix of indices of defining neighbours for each vertex.
#' @param n_particles Number of particles for SMC.
#' @param ess_threshold Resampling is done when the effective sample size estimator is less than this threshold times the number of particles.
#' For example if \code{ess_threshold=1} resampling is done at each iteration (default), whereas if \code{ess_threshold=0} the algorithm reduces to simple importance sampling.
#' @param seed Seed for the random number generator.
#' @export
psi_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE, n_particles,
                    ess_threshold = NULL, reorder = TRUE, initial_mode, max_iter = 100, conv_tol = 1e-8, 
                    seed = sample(.Machine$integer.max, size = 1)) {
  
  if(n_particles < 1) stop("Argument 'n_particles' must be positive integer.")
  if (is.null(ess_threshold)) ess_threshold <- 2
  y <- split(y, idx)
  u <- split(u, idx)
  n_y <- lengths(y)
  y <- list_to_matrix(y)
  u <- list_to_matrix(u)
  
  if (missing(initial_mode)) {
    initial_mode <- colMeans((y + 0.5) / (u + 1), na.rm = TRUE)
    initial_mode[is.na(initial_mode)] <- 1 #check this!!
    initial_mode <- qlogis(initial_mode) - use_mu * mu
  }
  
  storage.mode(nbs) <- storage.mode(nnbs) <- storage.mode(n_y) <- "integer"
  storage.mode(y) <- storage.mode(u) <- "double"
  R_psi_binomial_car(nnbs, nbs, tau, d, y, n_y, u, mu, use_mu, n_particles,
                     initial_mode, max_iter, conv_tol, seed, reorder, ess_threshold)
  
} 
#' Boostrap SMC for CAR model with Binomial observations
#' 
#' @inheritParams psi_car
#' @export
bsf_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE, n_particles,
                    ess_threshold = NULL, reorder = TRUE,
                    seed = sample(.Machine$integer.max, size = 1)) {
  
  if(n_particles < 1) stop("Argument 'n_particles' must be positive integer.")
  if (is.null(ess_threshold)) ess_threshold <- 2
  y <- split(y, idx)
  u <- split(u, idx)
  n_y <- lengths(y)
  y <- list_to_matrix(y)
  u <- list_to_matrix(u)
  
  storage.mode(nbs) <- storage.mode(nnbs) <- storage.mode(n_y) <- "integer"
  storage.mode(y) <- storage.mode(u) <- "double"
  R_bsf_binomial_car(nnbs, nbs, tau, d, y, n_y, u, mu, use_mu, n_particles,
                     seed, reorder, ess_threshold)
  
} 
