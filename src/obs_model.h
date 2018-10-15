// Base class for observation densities
// psi(y_i | x) = psi(y_i | x_j(i))
// where j(i) defines the latent variable for which y_i depends
// y are the observations, as k x m matrix, each column corresponds to one state
// and n_y(j) gives the number of observations corresponding to state_j (i.e y is padded with NA's)

// Note that dim(y) >= dim(x), i.e. there should be at least NA (missing) value 
// corresponding to each latent variable => idx ranges from 1 to dim(x).
// But note that this condition is not currently checked anywhere!

#ifndef OBS_H
#define OBS_H

#include <RcppEigen.h>

class obs_model {
  
public:
  
  // constructor
  obs_model(
    const Eigen::MatrixXd y_, 
    const Eigen::VectorXi n_y_,
    const unsigned int n_par_ = 1);
  
  // unnormalized log-density p(y_i | x_i, mu)
  virtual double log_density(
      const unsigned int i, 
      const double state) const = 0;
  
  // functions returning the canonical parameters of the Gaussian approximation
  
  virtual double approx_log_density(
      const unsigned int j, 
      const double mode,
      const double state) const = 0;
  
  virtual double b_function(
      const unsigned int i, 
      const double state) const = 0;
  
  virtual double c_function(
      const unsigned int i,
      const double state) const = 0;
  
  Eigen::VectorXd build_Q(
      const Eigen::Ref<const Eigen::VectorXd>& x) const;
  
  Eigen::VectorXd build_b(
      const Eigen::Ref<const Eigen::VectorXd>& x) const;
  
  virtual double log_prior_pdf(
      const Eigen::Ref<const Eigen::VectorXd>& theta) const = 0;
  
  virtual void update_model(
      const Eigen::Ref<const Eigen::VectorXd>& theta) = 0;
  
  Eigen::MatrixXd y; // observations as k x m matrix
  Eigen::VectorXi n_y; // how many observations depend on each x_j

  const unsigned int n_par;
};



#endif
