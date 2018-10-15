// Binomial observation density, y_i ~ Binomial(logit(mu + x_j(i)), u_i), where
// where j(i) defines the state for which y_i depends
// u_i is the number of trials at point i
// mu is overall mean in linear-predictor scale
// and n_y is a vector defining number of observations depending on x_j

// Note that dim(y) >= dim(x), i.e. there should be at least NA (missing) value 
// corresponding to each latent variable => idx ranges from 1 to dim(x).
// But note that this condition is not currently checked anywhere!

#ifndef BINOMIAL_H
#define BINOMIAL_H

#include "obs_model.h"

class binomial_model: public obs_model {
  
public:
  
  // constructor
  binomial_model(
    const Eigen::MatrixXd y_, 
    const Eigen::VectorXi n_y_,
    const Eigen::MatrixXd u_, 
    const double mu_,
    const bool use_mu_ = true);
  
  double log_density(
      const unsigned int j, 
      const double state) const;
  
  // functions returning the necessary components for the Gaussian approximation
  // i.e. the canonical parameters of the Gaussian approximation wrt state
  
  double approx_log_density(
      const unsigned int j, 
      const double mode,
      const double state) const;
  
  double b_function(
      const unsigned int j, 
      const double state) const;
  
  double c_function(
      const unsigned int j, 
      const double state) const;
  
  double log_prior_pdf(
      const Eigen::Ref<const Eigen::VectorXd>& theta) const;
  
  void update_model(
      const Eigen::Ref<const Eigen::VectorXd>& theta);
  
  Eigen::MatrixXd u;
  double mu;
  const bool use_mu;
};

#endif
