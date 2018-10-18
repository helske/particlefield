// bootstrap filter with stratified resampling

#ifndef BSF_H
#define BSF_H

#include<random>
#include "sampling.h"

template <class T1, class T2>
double bootstrap_filter(
    const T1& y_model,
    T2& x_model,
    const unsigned int n_particles,
    std::mt19937& engine,
    Eigen::Ref<Eigen::MatrixXd> states,
    Eigen::Ref<Eigen::VectorXd> weights,
    Eigen::Ref<Eigen::VectorXd> ess,
    const double ess_threshold = 2.0) {
  
  double loglik = 0.0;
  
  const unsigned int m = x_model.number_of_states;
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  // prior for x
  Eigen::SparseMatrix<double> Q_x_prior = x_model.build_Q();
  Eigen::VectorXd prior_mean = x_model.mean();
  
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> solver;
  solver.analyzePattern(Q_x_prior); //sparsity structure does not change
  solver.factorize(Q_x_prior);
  Eigen::SparseMatrix<double> L(x_model.number_of_states, x_model.number_of_states);
  L = solver.matrixL().eval() * (solver.vectorD().cwiseSqrt().asDiagonal());
  // sample x_1
  double q_m = std::pow(L.coeff(m - 1, m - 1), 2);
  for (unsigned int k = 0; k < n_particles; k++) {
    states(m - 1, k) = sample_from_normal(prior_mean(m - 1), q_m, engine);
    // log[p(y|x)/g(y|x)]
    weights(k) = y_model.log_density(m - 1, states(m - 1, k));
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  
  double max_weight = weights.maxCoeff();
  Eigen::VectorXd expweights = exp(weights.array() - max_weight);
  double sum_weights = expweights.sum();
  Eigen::VectorXd normalized_weights = expweights / sum_weights;
  weights = log(normalized_weights.array());
  loglik += max_weight + log(sum_weights / n_particles);
  
  ess(m - 1) = 1.0 / normalized_weights.array().square().sum();
  
  // Sample x_{m-1},...,x_1
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
    
    double q_j = std::pow(L.coeff(j, j), 2);
    
    for (unsigned int k = 0; k < n_particles; k++) {
      
      double mean = prior_mean(j) -
        L.col(j).bottomRows(m - j - 1).dot(states.col(k).bottomRows(m - j - 1) -
        prior_mean.bottomRows(m - j - 1)) / L.coeff(j, j) ;
      
      states(j, k) = sample_from_normal(mean, q_j, engine);
      
      weights(k) += y_model.log_density(j, states(j, k));
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

