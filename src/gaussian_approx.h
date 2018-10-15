// Gaussian approximation for exponential family models
// with univariate case psi(y_i | x) = psi(y_i |x_j(i))
// where j(i) defines the state for which y_i depends

#ifndef APPROX_H
#define APPROX_H

#include <RcppEigen.h>

// returns the approximate mode and Cholesky L of Q no it doesn't, L is not correct!
template <class T1, class T2>
double gaussian_approx(
    const T1& y_model, // model for observations
    T2& x_model, // model for latent variables
    Eigen::VectorXd& x,
    Eigen::SparseMatrix<double>& L, // I don't understand why Ref does not work here...
    const unsigned int max_iter = 50,
    const double conv_tol = 1e-8, 
    const bool ratio_correction = true) {
  
  // prior for x
  Eigen::SparseMatrix<double> Q_x_prior = x_model.build_Q();
  Eigen::VectorXd prior_mean = x_model.mean();
  Eigen::VectorXd b_x_prior = Q_x_prior * prior_mean;
  Eigen::VectorXd b_smoothed = b_x_prior + y_model.build_b(x);
  Eigen::SparseMatrix<double> Q_smoothed = Q_x_prior;
  Q_smoothed.diagonal() +=  y_model.build_Q(x);
  // no reorder, must be done beforehand
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> solver;
  solver.analyzePattern(Q_smoothed); //sparsity structure does not change
  solver.factorize(Q_smoothed);
  
  // maximize approximate p(x|y)
  
  x = solver.solve(b_smoothed);

  double log_pxy = -0.5 * x.dot(Q_smoothed * x) + b_smoothed.dot(x);
  // Use simple Newton algorithm, we might need line-search in some cases...
  unsigned int i = 0;
  double diff = conv_tol + 1;
  
  while(i < max_iter && conv_tol < diff) {
    
    b_smoothed = b_x_prior + y_model.build_b(x);
    Q_smoothed = Q_x_prior;
    Q_smoothed.diagonal() += y_model.build_Q(x);
    solver.factorize(Q_smoothed);
    Eigen::VectorXd x_new = solver.solve(b_smoothed);
  
    double log_pxy_new = -0.5 * x_new.dot(Q_smoothed * x_new) + b_smoothed.dot(x_new);
    diff = std::abs(log_pxy_new - log_pxy) / std::abs(log_pxy);
    x = x_new;
    log_pxy = log_pxy_new;
    
    i = i + 1;
  }
 
  L = solver.matrixL().eval() * (solver.vectorD().cwiseSqrt().asDiagonal());
  // did not converge
  if (i == max_iter) {
    Rcpp::Rcout<<"approximation did not converge. Hyperparameters theta: tau="<<x_model.tau<<", d="<<x_model.d<<", mu="<<y_model.mu<<std::endl;
    return -std::numeric_limits<double>::infinity();
  }
  
  // log-likelihood of the approximating Gaussian model
  // loglik = log_p(x,y) - log_p(x|y)  = log_p(x) + log_p(y|x) - log_p(x|y) 
  const unsigned int m = x_model.number_of_states;
  // log_p(y | x) part:
  double loglik = 0.0;
  if (ratio_correction) {
    //additional correction term p(y|x)/tildep(y|x), cancels tildep(y|x)
    for (unsigned int j = 0; j < m; j++) {
      loglik += y_model.log_density(j, x(j)); 
    }
  } else {
    for (unsigned int j = 0; j < m; j++) {
      loglik += y_model.approx_log_density(j, x(j), x(j)); 
    }
  }
  // -log_p(x | y) part:
  double log_det = solver.vectorD().array().log().sum();
  loglik -= 0.5 * log_det;
  
  // log_p(x) part: 
  solver.factorize(Q_x_prior);
  double tolerance = Eigen::NumTraits<double>::dummy_precision();
  log_det = 
    (solver.vectorD().array() > tolerance).select(solver.vectorD().array(), 1.0).log().sum();
  loglik += 0.5 * (log_det - x.dot(Q_x_prior * x) - prior_mean.dot(Q_x_prior * prior_mean)) + 
    (Q_x_prior * prior_mean).dot(x) ;
  
  return loglik;
}

#endif

