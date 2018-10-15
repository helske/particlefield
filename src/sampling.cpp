#include "sampling.h"

// stratified sampling of indices from 0 to length(p)
// p is the target distribution (will be overwritten)
// r are random numbers from U(0, 1)
// N is the number of samples
Eigen::VectorXi stratified_sample(
    Eigen::Ref<Eigen::VectorXd> p, 
    Eigen::Ref<Eigen::VectorXd> r, 
    const unsigned int N) {
  
  for (unsigned int i = 1; i < p.size(); i++) {
    p(i) += p(i-1);
  }
  p(p.size() - 1) = 1;
  
  unsigned int j = 0;
  double alpha = 1.0 / N;
  Eigen::VectorXi xp(N);
  for(unsigned int k = 0; k < p.size() && j < N; k++) {
    while (j < N && (r(j) + j) * alpha <= p(k)) {
      xp(j) = k;
      j++;
    }
  }
  while (j < N) {
    xp(j) = N;
    j++;
  }
  return xp;
}


Eigen::MatrixXd sample_from_canonical(
    const unsigned int nsim,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::SparseMatrix<double>& Q, std::mt19937& engine) {
  
  // LDL decomposition without permutations for Q = LDL'
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int> > chol(Q);
  Eigen::VectorXd mean = chol.solve(b);
  unsigned int m = b.size();
  
  Eigen::SparseMatrix<double> LDt = chol.vectorD().cwiseSqrt().asDiagonal() * chol.matrixU().eval();
  // sample
  Eigen::MatrixXd X(m, nsim);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    
    Eigen::VectorXd z(m);
    for (unsigned int j = 0; j < m; j++) {
      z(j) = normal(engine);
    }
    
    X.col(i) =  mean + LDt.triangularView<Eigen::Upper>().solve(z);
  }
  return X;
}

Eigen::MatrixXd sample_from_normal(
    const unsigned int nsim,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::SparseMatrix<double>& Q, std::mt19937& engine) {
  
  // LDL decomposition without permutations for Q = LDL'
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int> > chol(Q);
  
  unsigned int m = mean.size();
  
  Eigen::SparseMatrix<double> LDt = chol.vectorD().cwiseSqrt().asDiagonal() * chol.matrixU().eval();
  // sample
  Eigen::MatrixXd X(m, nsim);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    
    Eigen::VectorXd z(m);
    for (unsigned int j = 0; j < m; j++) {
      z(j) = normal(engine);
    }
    
    X.col(i) =  mean + LDt.triangularView<Eigen::Upper>().solve(z);
  }
  return X;
}

double sample_from_canonical(const double& b,
  const double& Q, std::mt19937& engine) {
  
  // sample
  std::normal_distribution<> normal(0.0, 1.0);
  return b / Q + std::sqrt(1.0 / Q) * normal(engine);
}

double sample_from_normal(const double& mean,
  const double& Q, std::mt19937& engine) {
  
  // sample
  std::normal_distribution<> normal(0.0, 1.0);
  return mean + normal(engine) / std::sqrt(Q);
}

