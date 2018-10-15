// Base class for GMRF
#ifndef STATE_H
#define STATE_H

#include <RcppEigen.h>
#include "graph.h"

class state_model {
  
public:
  
  state_model(
    const graph& G_,
    const unsigned int n_par_ = 0);
  
  // return the value of Q_ij
  virtual double precision(
      const unsigned int i,
      const unsigned int j) const = 0;
  
  // return the mean vector mu
  virtual Eigen::VectorXd mean() const = 0;
  
  // return Q
  Eigen::SparseMatrix<double> build_Q() const; 
  
  // reorder observations and construct P
  template <class T>
  void reorder(
      T& y_model);
  
  virtual double log_prior_pdf(
      const Eigen::Ref<const Eigen::VectorXd>& theta) const = 0;
  
  virtual void update_model(
      const Eigen::Ref<const Eigen::VectorXd>& theta) = 0;
  
  const unsigned int number_of_states;
  const unsigned int n_par;
  
  // permutation matrix for reordering
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>  P;
  
protected:
  graph G;
};

#endif
