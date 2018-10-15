#ifndef SAMPLE_H
#define SAMPLE_H

#include<random>
#include <RcppEigen.h>

Eigen::VectorXi stratified_sample(
    Eigen::Ref<Eigen::VectorXd> p, 
    Eigen::Ref<Eigen::VectorXd> r, 
    const unsigned int N);

Eigen::MatrixXd sample_from_canonical(
    const unsigned int nsim,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::SparseMatrix<double>& Q, std::mt19937& engine);

Eigen::MatrixXd sample_from_normal(
    const unsigned int nsim,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::SparseMatrix<double>& Q, std::mt19937& engine);

double sample_from_canonical(const double& b,
  const double& Q, std::mt19937& engine);

double sample_from_normal(const double& mean,
  const double& Q, std::mt19937& engine);



#endif
