#ifndef RAM_H
#define RAM_H
// from ramcmc package : https://cran.r-project.org/web/packages/ramcmc/index.html
#include <RcppEigen.h>
// Cholesky update
// Given the lower triangular matrix L obtained from the Cholesky decomposition of A,
// updates L such that it corresponds to the decomposition of A + u*u'.
//
void chol_update(
  Eigen::Ref<Eigen::MatrixXd> L, 
  Eigen::Ref<Eigen::VectorXd> u) {
  
  unsigned int n = u.size() - 1;
  
  for (unsigned int i = 0; i < n; i++) {
    double r = sqrt(L(i,i) * L(i,i) + u(i) * u(i));
    double c = r / L(i, i);
    double s = u(i) / L(i, i);
    L(i, i) = r;
    L.col(i).tail(n - i)  =
      (L.col(i).tail(n -  i) + s * u.tail(n - i)) / c;
    u.tail(n - i) = c * u.tail(n - i) -
      s * L.col(i).tail(n - i);
  }
  L(n, n) = sqrt(L(n, n) * L(n, n) + u(n) * u(n));
}


// Cholesky downdate
// Given the lower triangular matrix L obtained from the Cholesky decomposition of A,
// updates L such that it corresponds to the decomposition of A - u*u'.
//
// NOTE: The function does not check that the downdating produces a positive definite matrix!
void chol_downdate(
  Eigen::Ref<Eigen::MatrixXd> L, 
  Eigen::Ref<Eigen::VectorXd> u) {
  
  unsigned int n = u.size() - 1;
  
  for (unsigned int i = 0; i < n; i++) {
    double r = sqrt(L(i,i) * L(i,i) - u(i) * u(i));
    double c = r / L(i, i);
    double s = u(i) / L(i, i);
    L(i, i) = r;
    L.col(i).tail(n - i)  =
      (L.col(i).tail(n -  i) - s * u.tail(n - i)) / c;
    u.tail(n - i) = c * u.tail(n - i) -
      s * L.col(i).tail(n - i);
  }
  L(n, n) = sqrt(L(n, n) * L(n, n) - u(n) * u(n));
}

// Update the Cholesky factor of the covariance matrix of the proposal distribution

// only case where there can be problems is a pathological case
// where target = 1 and current = 0
// this is not checked as it never happens with reasonable targets.
void adapt_S(
  Eigen::Ref<Eigen::MatrixXd> S, 
  Eigen::Ref<Eigen::VectorXd> u, 
  double current, 
  double target,
  unsigned int n, 
  double gamma) {

  double change = current - target;
  u = S * u / u.norm() * std::sqrt(std::min(1.0, u.size() * std::pow(n, -gamma)) *
    std::abs(change));
  if(change > 0.0) {
    chol_update(S, u);
  } else {
    chol_downdate(S, u);
  }
}

#endif
