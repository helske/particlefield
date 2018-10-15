// Binomial observation density, y_i ~ Binomial(logit(mu + x_j(i)), u_i), where
// where j(i) defines the state for which y_i depends
// u_i is "exposure" at point i
// mu is overall mean in linear-predictor scale

#include "binomial_model.h"

binomial_model::binomial_model(
  const Eigen::MatrixXd y_,
  const Eigen::VectorXi n_y_,
  const Eigen::MatrixXd u_, 
  const double mu_, 
  const bool use_mu_) : 
  obs_model(y_, n_y_, use_mu_), u(u_), mu(use_mu_ * mu_), use_mu(use_mu_){
}

// log-density of the observations depending on the same state
double binomial_model::log_density(
    const unsigned int j, 
    const double state) const {
  // double ld=0.0;
  // double p = 1.0/(1.0 + exp(-state));
  // for(int i=0;i<n_y(j);i++){
  //   ld += R::dbinom(y(i,j),u(i,j), p, 1);
  // }
  // return ld;
  return y.col(j).head(n_y(j)).sum() * (mu + state) - 
    u.col(j).head(n_y(j)).sum() * std::log(1.0 + std::exp(mu + state));
}

// functions returning the necessary components for the Gaussian approximation
// i.e. the canonical parameters of the Gaussian approximation wrt state

// log-density of the approximating Gaussian model depending on the same state
double binomial_model::approx_log_density(
    const unsigned int j, 
    const double mode,
    const double state) const {
  return -0.5 * std::pow(b_function(j, mode) / c_function(j, mode) -  state, 2) * c_function(j, mode);
}

double binomial_model::b_function(
    const unsigned int j, 
    const double state) const {
  const double tmp = std::exp(mu + state);
  return y.col(j).head(n_y(j)).sum() - u.col(j).head(n_y(j)).sum() * 
    tmp / std::pow(1.0 + tmp, 2) * (tmp - state + 1);
}

double binomial_model::c_function(
    const unsigned int j, 
    const double state) const {
  return u.col(j).head(n_y(j)).sum() * 
    std::exp(mu + state) / std::pow(1.0 + std::exp(mu + state), 2);
}


double binomial_model::log_prior_pdf(
    const Eigen::Ref<const Eigen::VectorXd>& theta) const {
  // currently fixed
  double log_prior = 0.0;
  if (use_mu) log_prior = R::dnorm(theta(0), 0, 10, 1);
  return log_prior;
}

void binomial_model::update_model(
    const Eigen::Ref<const Eigen::VectorXd>& theta) {
  if (use_mu) mu = theta(0);
}
