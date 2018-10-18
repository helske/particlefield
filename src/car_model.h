// Besag's CAR model for latent field
// with additional "properness" parameter d
// In INLA this is besagproper

#ifndef CAR_H
#define CAR_H

#include "state_model.h"

class graph;

class car_model: public state_model {
  
public:
  
  // constructor
  car_model(
    const graph& G,
    const double tau_,
    const double d_ = 0);
  
  // return the value of Q_ij
  double precision(
      const unsigned int i,
      const unsigned int j) const;
  
  Eigen::VectorXd mean() const;
  
  double log_prior_pdf(
      const Eigen::Ref<const Eigen::VectorXd>& theta) const;
  
  void update_model(
      const Eigen::Ref<const Eigen::VectorXd>& theta);
  
  double tau;
  double d;
};

#endif
