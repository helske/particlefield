// Base class for observation densities

#include "obs_model.h"

obs_model::obs_model(
  const Eigen::MatrixXd y_,
  const Eigen::VectorXi n_y_,
  const unsigned int n_par_) : 
  y(y_), 
  n_y(n_y_),  
  n_par(n_par_) {
}

// build the diagonal of the "Q_y"
// not really Q_y because we sum elements of with same idx
Eigen::VectorXd obs_model::build_Q(
    const Eigen::Ref<const Eigen::VectorXd>& x) const {
  
  Eigen::VectorXd diag(x.size());
  for (unsigned int j = 0; j < x.size(); j++) {
    diag(j) = c_function(j, x(j));
  }
  return diag;
}

// build b
Eigen::VectorXd obs_model::build_b(
    const Eigen::Ref<const Eigen::VectorXd>& x) const {
  
  Eigen::VectorXd b(x.size());
  for (unsigned int j = 0; j < x.size(); j++) {
    b(j) = b_function(j, x(j));
  }
  return b;
}
