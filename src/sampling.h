#ifndef SAMPLE_H
#define SAMPLE_H

#include<random>
#include <RcppEigen.h>

Eigen::VectorXi stratified_sample(
    Eigen::Ref<Eigen::VectorXd> p, 
    Eigen::Ref<Eigen::VectorXd> r, 
    const unsigned int N);

double sample_from_normal(const double& mean,
  const double& Q, std::mt19937& engine);

#endif
