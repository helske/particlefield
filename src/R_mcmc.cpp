// MCMC

#include "graph.h"
#include "car_model.h"
#include "binomial_model.h"
#include "approx_mcmc.h"
#include "da_mcmc.h"

// [[Rcpp::export]]
Rcpp::List R_amcmc_binomial_car(
    const Eigen::Map<Eigen::VectorXi> nnbs, 
    const Eigen::Map<Eigen::MatrixXi> nbs,
    const double tau,
    const double d,
    const Eigen::Map<Eigen::MatrixXd> y,
    const Eigen::Map<Eigen::VectorXi> n_y,
    const Eigen::Map<Eigen::MatrixXd> u,
    const double mu,
    const bool use_mu,
    const unsigned int n_iter,
    const unsigned int n_burnin,
    const Eigen::Map<Eigen::VectorXd> initial_theta,
    const Eigen::Map<Eigen::VectorXd> initial_mode,
    Eigen::Map<Eigen::MatrixXd> S,
    const unsigned int max_iter, 
    const double conv_tol, 
    const unsigned int seed, 
    const bool ratio_correction,
    const bool reorder) {
  
  graph G(nnbs, nbs);
  car_model model_x(G, tau, d);
  binomial_model model_y(y, n_y, u, mu, use_mu);
  
  if (reorder) model_x.reorder(model_y);
  Eigen::MatrixXd theta(initial_theta.size(), n_iter - n_burnin);
  Eigen::MatrixXd states(initial_mode.size(), n_iter - n_burnin);
  Eigen::VectorXd posterior(n_iter - n_burnin);
  
  std::mt19937 engine(seed);
  double acceptance_rate = approx_mcmc(
    model_y,
    model_x,
    n_iter,
    n_burnin,
    theta,
    states,
    posterior,
    initial_theta,
    S,
    model_x.P * initial_mode,
    engine,
    max_iter,
    conv_tol,
    ratio_correction);
  
  return Rcpp::List::create(
    Rcpp::Named("theta") = theta.transpose(),
    Rcpp::Named("states") = states.transpose() * model_x.P,
    Rcpp::Named("posterior") = posterior,
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = S);
}

// [[Rcpp::export]]
Rcpp::List R_mcmc_binomial_car(
    const Eigen::VectorXi nnbs,
    const Eigen::MatrixXi nbs,
    const double tau,
    const double d,
    const Eigen::MatrixXd y,
    const Eigen::VectorXi n_y,
    const Eigen::MatrixXd u,
    const double mu,
    const bool use_mu,
    const unsigned int n_iter,
    const unsigned int n_burnin,
    const Eigen::VectorXd initial_theta,
    const Eigen::VectorXd initial_mode,
    Eigen::MatrixXd S,
    const unsigned int max_iter,
    const double conv_tol,
    const unsigned int seed,
    const unsigned int n_particles,
    const bool reorder,
    const double ess_threshold) {

  graph G(nnbs, nbs);
  car_model model_x(G, tau, d);
  binomial_model model_y(y, n_y, u, mu, use_mu);

  if (reorder) model_x.reorder(model_y);
  Eigen::MatrixXd theta(initial_theta.size(), n_iter - n_burnin);
  Eigen::MatrixXd states(initial_mode.size(), n_iter - n_burnin);
  Eigen::VectorXd posterior(n_iter - n_burnin);

  std::mt19937 engine(seed);
  double acceptance_rate;

  acceptance_rate = da_mcmc(
      model_y,
      model_x,
      n_iter,
      n_burnin,
      theta,
      states,
      posterior,
      initial_theta,
      S,
      model_x.P * initial_mode,
      engine,
      n_particles,
      max_iter,
      conv_tol,
      ess_threshold);
  
  return Rcpp::List::create(
    Rcpp::Named("theta") = theta.transpose(),
    Rcpp::Named("states") = states.transpose() * model_x.P,
    Rcpp::Named("posterior") = posterior,
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = S);
}
