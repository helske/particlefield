// pseudo-marginal MCMC with delayed acceptance
// Using the Gaussian approximation in first step
// and twisted SMC in the second step

#ifndef DAMCMC_H
#define DAMCMC_H

#include <random>
#include <RcppEigen.h>
#include "ram.h"
#include "psi_filter.h"

template <class T1, class T2>
double da_mcmc(
    T1& y_model,
    T2& x_model,
    const unsigned int n_iter,
    const unsigned int n_burnin,
    Eigen::Ref<Eigen::MatrixXd> theta_storage,
    Eigen::Ref<Eigen::MatrixXd> state_storage,
    Eigen::Ref<Eigen::VectorXd> posterior,
    Eigen::VectorXd theta,
    Eigen::Ref<Eigen::MatrixXd> S,
    const Eigen::Ref<const Eigen::VectorXd>& initial_mode,
    std::mt19937& engine,
    const unsigned int n_particles,
    const unsigned int max_iter = 50,
    const double conv_tol = 1e-8,
    const double ess_threshold = 2) {
  
  
  Eigen::MatrixXd states = 
    Eigen::MatrixXd::Zero(x_model.number_of_states, n_particles);
  Eigen::VectorXd weights(n_particles);
  Eigen::VectorXd ess(x_model.number_of_states);
  
  // compute the log-likelihood
  Eigen::SparseMatrix<double> L(x_model.number_of_states, x_model.number_of_states);
  Eigen::VectorXd mode = initial_mode;
  double approx_loglik = 
    gaussian_approx(y_model, x_model, mode, L, max_iter, conv_tol, true);
  
  // psi-filter computes the approximation again but as it uses the mode from previous approx run
  // it only takes one step to converge
  double loglik = 
    psi_filter(y_model, x_model,  n_particles, engine, states, weights, ess, mode,
      max_iter, conv_tol, ess_threshold);
  
  std::discrete_distribution<unsigned int> sample0(weights.data(), weights.data() + weights.size());
  Eigen::VectorXd state = states.col(sample0(engine));
  
  double logprior = 
    x_model.log_prior_pdf(theta.head(x_model.n_par)) + 
    y_model.log_prior_pdf(theta.tail(y_model.n_par));
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  std::normal_distribution<> normal(0.0, 1.0);
  double acceptance_prob = 0.0;
  double acceptance_rate = 0.0;
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    Eigen::VectorXd u(theta.size());
    for(unsigned int j = 0; j < theta.size(); j++) {
      u(j) = normal(engine);
    }
    
    // propose new theta
    Eigen::VectorXd theta_prop = theta + S * u;
    // compute prior
    double logprior_prop =  x_model.log_prior_pdf(theta_prop.head(x_model.n_par)) + 
      y_model.log_prior_pdf(theta_prop.tail(y_model.n_par));
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      
      // update parameters
      x_model.update_model(theta_prop.head(x_model.n_par));
      y_model.update_model(theta_prop.tail(y_model.n_par));
      
      // compute the approximate log-likelihood with proposed theta
      mode = initial_mode;
      double approx_loglik_prop = 
        gaussian_approx(y_model, x_model, mode, L, max_iter, conv_tol, true);
      
      if (approx_loglik_prop > -std::numeric_limits<double>::infinity() && !std::isnan(approx_loglik_prop)) {
        
        //compute the first stage acceptance probability
        // use explicit min(...) as we need this value later in RAM
        acceptance_prob = std::min(1.0,
          std::exp(approx_loglik_prop - approx_loglik + logprior_prop - logprior));
        
        //first stage accept
        if (unif(engine) < acceptance_prob) {
          
          double loglik_prop = 
            psi_filter(y_model, x_model,  n_particles, engine, states, weights, ess, mode,
              max_iter, conv_tol, ess_threshold);
          // second stage accept
          if (log(unif(engine)) < loglik_prop + approx_loglik - loglik - approx_loglik_prop) {
            if (i > n_burnin) {
              acceptance_rate++;
            }
            loglik = loglik_prop;
            approx_loglik = approx_loglik_prop;
            logprior = logprior_prop;
            theta = theta_prop;
            
            std::discrete_distribution<unsigned int> sample(weights.data(), weights.data() + weights.size());
            state = states.col(sample(engine));
          }
        }
      } else acceptance_prob = 0.0;
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      posterior(i - n_burnin - 1) = logprior + loglik;
      theta_storage.col(i - n_burnin - 1) = theta;
      state_storage.col(i - n_burnin - 1) = state;
    }
    if (i <= n_burnin) {
      // fixed at the moment for simplicity
      adapt_S(S, u, acceptance_prob, 0.234, i, 2.0 / 3.0);
    }
    
  }
  
  return acceptance_rate / (n_iter - n_burnin);
}

#endif

