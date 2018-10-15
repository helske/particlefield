// Functions for testing

#include "car_model.h"

// build and print a graph
// [[Rcpp::export]]
void R_print_graph(const Eigen::Map<Eigen::VectorXi> nnbs, const Eigen::Map<Eigen::MatrixXi> nbs) {

  graph test_graph(nnbs, nbs);
  test_graph.print_graph();

}

// build graph of CAR-model, and the corresponding precision matrix
// [[Rcpp::export]]
Eigen::SparseMatrix<double> R_car_Q(
    const Eigen::Map<Eigen::VectorXi> nnbs, 
    const Eigen::Map<Eigen::MatrixXi> nbs,
    const double tau, const double d) {

  graph G(nnbs, nbs);
  car_model model(G, tau, d);
  return model.build_Q();
}
