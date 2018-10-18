#' Gaussian approximation for Binomial model with latent CAR field
#'
#' @param nnbs Vector defining the number of neighbours for each vertex.
#' @param nbs Matrix of indices of defining neighbours for each vertex.
#' @param tau Precision parameter for CAR field.
#' @param d Properness parameter for the CAR field.
#' @param y Vector of observations.
#' @param u Vector of trials. Default to 1.
#' @param idx Vector defining the dependencies between y and x.
#' @param mu Intercept of the linear predictor. Defaults to 0.
#' @param use_mu Use mu in the model or not. Default is \code{TRUE}.
#' @param initial_mode Initial mode estimate of x.
#' @param max_iter Maximum number of iterations for the approximation algorithm.
#' @param conv_tol Tolerance parameter for the approximation algorithm.
#' @param reorder If \code{TRUE} (default), reordering is performed for increased efficiency.
#' @param ratio_correction Should the returned log-likelihood estimate contain the ratio correction
#' term? Default is \code{TRUE}.
#' @export
approximate_binomial_car <- function(nnbs, nbs, tau, d, y, u, idx, mu = 0, 
  use_mu = TRUE, initial_mode, max_iter = 100, conv_tol = 1e-8, reorder = TRUE,
  ratio_correction = TRUE) {
  
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
  R_approx_binomial_car(nnbs, nbs, tau, d, y, n_y, u, mu, use_mu,
    initial_mode, max_iter, conv_tol, reorder, ratio_correction)
  
} 

