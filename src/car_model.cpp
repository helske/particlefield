// Besag's CAR model for latent field
// with additional "properness" parameter d
// In INLA this is besagproper

#include "car_model.h"
#include "graph.h"

car_model::car_model(
  const graph& G_,
  const double tau_,
  const double d_) :
  state_model(G_, 2), tau(tau_), d(d_) {
}

// note this is called only when i ~ j or i == j!
double car_model::precision(
    const unsigned int i,
    const unsigned int j) const{
  
  double q = 0.0;
  if (i == j) { // diagonal
    q = G.number_of_neighbours(i) + d;
  } else {
    q = -1.0;
  }
  return q * tau;
}

Eigen::VectorXd car_model::mean() const {
  return Eigen::VectorXd::Constant(G.number_of_vertices, 0.0);
}

double car_model::log_prior_pdf(
    const Eigen::Ref<const Eigen::VectorXd>& theta) const {
  // currently hard coded...
  
  // prior on tau, use gamma like in INLA by default
  // R's C function uses scale and not rate...
  double  log_prior = R::dgamma(exp(theta(0)), 1, 100, 1) + 
                      R::dgamma(exp(theta(1)), 1, 1, 1) + theta(0) + theta(1);
 
  return log_prior;
}

void car_model::update_model(
    const Eigen::Ref<const Eigen::VectorXd>& theta) {
  tau = exp(theta(0));
  d = exp(theta(1));
}

