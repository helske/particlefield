// psi-PF with stratified resampling

#ifndef PSI_H
#define PSI_H

#include<random>
#include "sampling.h"
#include "gaussian_approx.h"

template <class T1, class T2>
double psi_filter(
    const T1& y_model,
    T2& x_model,
    const unsigned int n_particles,
    std::mt19937& engine,
    Eigen::Ref<Eigen::MatrixXd> states,
    Eigen::Ref<Eigen::VectorXd> weights,
    Eigen::Ref<Eigen::VectorXd> ess,
    const Eigen::Ref<const Eigen::VectorXd>& initial_mode,
    const unsigned int max_iter = 50,
    const double conv_tol = 1e-8,
    const double ess_threshold = 2.0) {
  
  // compute a Gaussian approximation, returns the mode and L
  Eigen::SparseMatrix<double> L(x_model.number_of_states, x_model.number_of_states);
  Eigen::VectorXd mode = initial_mode;
  double loglik = gaussian_approx(y_model, x_model, mode, L, max_iter, conv_tol, false);
  
  const unsigned int m = x_model.number_of_states;
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  // sample x_m
  double q_smoothedm = std::pow(L.coeff(m - 1, m - 1), 2);
  for (unsigned int k = 0; k < n_particles; k++) {
    states(m - 1, k) = sample_from_normal(mode(m - 1), q_smoothedm, engine);
    // log[p(y|x)/g(y|x)]
    weights(k) = y_model.log_density(m - 1, states(m - 1, k)) -
      y_model.approx_log_density(m - 1, mode(m - 1), states(m - 1, k));
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  // Sample x_{m-1},...,x_1
  
  
  double max_weight = weights.maxCoeff();
  Eigen::VectorXd expweights = exp(weights.array() - max_weight);
  double sum_weights = expweights.sum();
  Eigen::VectorXd normalized_weights = expweights / sum_weights;
  weights = log(normalized_weights.array());
  loglik += max_weight + log(sum_weights / n_particles);
  ess(m - 1) = 1.0 / normalized_weights.array().square().sum();
  
  for (int j = m - 2; j >= 0; j--) {
    
    // check the need for resampling
    if (ess(j + 1) < ess_threshold * n_particles) {
      
      Eigen::VectorXd r(n_particles);
      for (unsigned int k = 0; k < n_particles; k++) {
        r(k) = unif(engine);
      }
      Eigen::VectorXi indices = stratified_sample(normalized_weights, r, n_particles);
      Eigen::MatrixXd state_tmp = states;
      for (unsigned int k = 0; k < n_particles; k++) {
        states.col(k) = state_tmp.col(indices(k));
      }
      weights.fill(-log(n_particles));
    }
    
    
    double q_smoothed = std::pow(L.coeff(j, j), 2);
    
    for (unsigned int k = 0; k < n_particles; k++) {
      double mean_smoothed = mode(j) -
        L.col(j).bottomRows(m - j - 1).dot(states.col(k).bottomRows(m - j - 1) -
        mode.bottomRows(m - j - 1)) / L.coeff(j, j) ;
      
      states(j, k) = sample_from_normal(mean_smoothed, q_smoothed, engine);
      
      weights(k) += y_model.log_density(j, states(j, k)) -
        y_model.approx_log_density(j, mode(j), states(j, k));
    }
    
    max_weight = weights.maxCoeff();
    expweights = exp(weights.array() - max_weight);
    sum_weights = expweights.sum();
    normalized_weights = expweights / sum_weights;
    weights = log(normalized_weights.array());
    loglik += max_weight + log(sum_weights);
    ess(j) = 1.0 / normalized_weights.array().square().sum();
  }
  weights = exp(weights.array());
  
  return loglik;
}

#endif

