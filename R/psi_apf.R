#' psi-APF for Poisson model with latent AR(1) process
#'
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
psi_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE, n_particles,
  initial_mode, max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1),
  reorder = TRUE, ess_threshold=NULL) {
  
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

#' @export
bsf_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, use_mu=TRUE, n_particles,
                    seed = sample(.Machine$integer.max, size = 1),
                    reorder = TRUE, ess_threshold = 2) {
  
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
