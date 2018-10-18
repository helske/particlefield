// Bootstrap particle filter
#include<random>

#include "graph.h"
#include "car_model.h"
#include "binomial_model.h"
#include "bootstrap_filter.h"


// [[Rcpp::export]]
Rcpp::List R_bsf_binomial_car(
    const Eigen::Map<Eigen::VectorXi> nnbs, 
    const Eigen::Map<Eigen::MatrixXi> nbs,
    const double tau, 
    const double d,
    const Eigen::Map<Eigen::MatrixXd> y,
    const Eigen::Map<Eigen::MatrixXi> n_y,
    const Eigen::Map<Eigen::MatrixXd> u, 
    const double mu,
    const bool use_mu,
    const unsigned int n_particles,
    const unsigned int seed,
    const bool reorder,
    const double ess_threshold) {
  
  graph G(nnbs, nbs);
  car_model model_x(G, tau, d);
  binomial_model model_y(y, n_y, u, mu, use_mu);
  Eigen::MatrixXd states = 
    Eigen::MatrixXd::Zero(model_x.number_of_states, n_particles);
  Eigen::VectorXd weights(n_particles);
  Eigen::VectorXd ess(model_x.number_of_states);
  
  if (reorder) model_x.reorder(model_y);
  
  
  std::mt19937 engine(seed);
  double logLik = bootstrap_filter(model_y, model_x, n_particles, engine, states, 
    weights, ess, ess_threshold);
  
  return Rcpp::List::create(
    Rcpp::Named("states") =  model_x.P.transpose() * states,
    Rcpp::Named("weights") = weights,  // note that these in respect to processing order!
    Rcpp::Named("ess") = ess.reverse(),  // note that these in respect to processing order!
    Rcpp::Named("logLik") = logLik);
}

