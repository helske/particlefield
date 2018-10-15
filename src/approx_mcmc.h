// approximate marginal MCMC

#ifndef AMCMC_H
#define AMCMC_H

#include <random>
#include <RcppEigen.h>
#include "ram.h"
#include "gaussian_approx.h"

template <class T1, class T2>
double approx_mcmc(
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
    const unsigned int max_iter = 50,
    const double conv_tol = 1e-8, 
    const bool ratio_correction = true) {
  
  // compute the Gaussian approximation
  Eigen::SparseMatrix<double> L(x_model.number_of_states, x_model.number_of_states);
  Eigen::VectorXd mode = initial_mode;
  double loglik = 
    gaussian_approx(y_model, x_model, mode, L, max_iter, conv_tol, ratio_correction);
  double logprior = 
    x_model.log_prior_pdf(theta.head(x_model.n_par)) + 
    y_model.log_prior_pdf(theta.tail(y_model.n_par));
  std::normal_distribution<> normal(0.0, 1.0);
  Eigen::VectorXd state(x_model.number_of_states);
  for(unsigned int j = 0; j < state.size(); j++) {
    state(j) = normal(engine);
  }
  state = mode + L.triangularView<Eigen::Lower>().transpose().solve(state);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
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
      
      // compute log-likelihood with proposed theta
      mode = initial_mode;
      double loglik_prop = 
        gaussian_approx(y_model, x_model, mode, L, max_iter, conv_tol, ratio_correction);
      
      if (loglik_prop > -std::numeric_limits<double>::infinity() && !std::isnan(loglik_prop)) {
        
        //compute the acceptance probability
        // use explicit min(...) as we need this value later
        
        acceptance_prob = std::min(1.0,
          std::exp(loglik_prop + logprior_prop - loglik - logprior));
        
        //accept
        if (unif(engine) < acceptance_prob) {
          if (i > n_burnin) {
            acceptance_rate++;
          }
          loglik = loglik_prop;
          logprior = logprior_prop;
          theta = theta_prop;
          for(unsigned int j = 0; j < state.size(); j++) {
            state(j) = normal(engine);
          }
          state = mode + L.triangularView<Eigen::Lower>().transpose().solve(state);
       
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

