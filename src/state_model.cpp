// Generic state model

#include "state_model.h"
#include "binomial_model.h"
#include "graph.h"

state_model::state_model(
    const graph& G_, 
    const unsigned int n_par_) :
  
  number_of_states(G_.number_of_vertices), 
  n_par(n_par_),
  P(Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>(G_.number_of_vertices)),
  G(G_) {
  P.setIdentity();
}

//** build precision matrix Q **//
Eigen::SparseMatrix<double> state_model::build_Q() const {
  
  std::vector< Eigen::Triplet<double> > triplet_list;
  triplet_list.reserve(G.number_of_neighbours.sum() + G.number_of_vertices);
  
  for (unsigned int i = 0; i < G.number_of_vertices; i++) {
    // itself
    triplet_list.push_back(Eigen::Triplet<double>(i, i, precision(i, i)));
    for (int j = 0; j < G.number_of_neighbours(i); j++) {
      // neighbours
      unsigned int k = G.neighbours(j, i) - 1;
      if (k < G.number_of_vertices) {
        triplet_list.push_back(Eigen::Triplet<double>(k, i, precision(k, i)));
      }
    }
  }
  
  Eigen::SparseMatrix<double> Q(G.number_of_vertices, G.number_of_vertices);
  Q.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return Q;
}

// Reorder the states and observervations to minimize the fill-in in Cholesky
// Based on the prior Q_x, so essentially it is assumed that Q_y is diagonal

template <class T>
void state_model::reorder(
    T& y_model) {
  
  // prior Q for x
  Eigen::SparseMatrix<double> Q = build_Q();
  
  Eigen::AMDOrdering<int> ordering;
  ordering(Q.selfadjointView<Eigen::Lower>(), P);
  P = P.transpose();
  // reorder graph
  G.reorder(P);
  // reorder observations
  y_model.n_y = P * y_model.n_y;
  y_model.y = y_model.y * P.transpose();
  y_model.u = y_model.u * P.transpose();
}

template void state_model::reorder(binomial_model& y_model);
