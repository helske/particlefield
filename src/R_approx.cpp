// Gaussian approximation for binomial CAR

#include "car_model.h"
#include "binomial_model.h"
#include "gaussian_approx.h"

// [[Rcpp::export]]
Rcpp::List R_approx_binomial_car(
    const Eigen::Map<Eigen::VectorXi> nnbs, 
    const Eigen::Map<Eigen::MatrixXi> nbs,
    const double tau, 
    const double d,
    const Eigen::Map<Eigen::MatrixXd> y,
    const Eigen::Map<Eigen::MatrixXi> n_y,
    const Eigen::Map<Eigen::MatrixXd> u, 
    const double mu,
    const bool use_mu,
    const Eigen::Map<Eigen::VectorXd> initial_mode,
    const unsigned int max_iter, 
    const double conv_tol, 
    const bool reorder,
    const bool ratio_correction) {
  
  graph G(nnbs, nbs);
  car_model model_x(G, tau, d);
  binomial_model model_y(y, n_y, u, mu, use_mu);
  
  if (reorder) model_x.reorder(model_y);
  Eigen::VectorXd mode = model_x.P * initial_mode;
  Eigen::SparseMatrix<double> L(model_x.number_of_states, model_x.number_of_states);
  double loglik = gaussian_approx(model_y, model_x, mode, L, max_iter, conv_tol,ratio_correction);
  if (reorder) mode = model_x.P.transpose() * mode;
  return Rcpp::List::create(
    Rcpp::Named("mean") = mode,
    Rcpp::Named("L") = L,
    Rcpp::Named("logLik") = loglik,
    Rcpp::Named("P") = model_x.P.toDenseMatrix()); 
  //note: all.equal(as.matrix(L2%*%t(L2)), as.matrix(t(P)%*%L%*%t(L)%*%P)), 
  // where L2 is from non-reordered case
}

