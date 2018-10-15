// Graph object constructor

#include "graph.h"

graph::graph(
  const Eigen::Ref<const Eigen::VectorXi>& number_of_neighbours_,
  const Eigen::Ref<const Eigen::MatrixXi>& neighbours_) :
  number_of_neighbours(number_of_neighbours_),
  neighbours(neighbours_.transpose()), number_of_vertices(number_of_neighbours_.size()),
  number_of_edges(number_of_neighbours.sum() / 2){
}

void graph::print_graph() {
  Rcpp::Rcout<<"The graph contains "<<number_of_vertices<<" vertices"<<std::endl;
  for(unsigned int i = 0; i < number_of_vertices; i++) {
    Rcpp::Rcout<<"The neighbours of vertex "<<i + 1<<" are "<<
      neighbours.col(i).head(number_of_neighbours(i)).transpose().array()<<std::endl;
  }
}

// reorder graph based on permutation P
// this is likely not the most optimal way, but maybe easiest..
void graph::reorder(Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>& P) {
  
  //build Q without diagonal
  std::vector< Eigen::Triplet<double> > triplet_list;
  triplet_list.reserve(number_of_neighbours.sum() + number_of_vertices);
  
  for (unsigned int i = 0; i < number_of_vertices; i++) {
    for (int j = 0; j < number_of_neighbours(i); j++) {
      // neighbours
      unsigned int k = neighbours(j, i) - 1;
      if (k < number_of_vertices) {
        triplet_list.push_back(Eigen::Triplet<double>(k, i, 1));
      }
    }
  }
  
  Eigen::SparseMatrix<double> Q(number_of_vertices, number_of_vertices);
  Q.setFromTriplets(triplet_list.begin(), triplet_list.end());
  
  // permute
  
  Q = Q.twistedBy(P);
  // permute nnbs
  number_of_neighbours = P * number_of_neighbours;
  
  // refill neighbourhood matrix
  neighbours.setZero();
  for (unsigned int i = 0; i < number_of_vertices; i++) {
    unsigned int k = 0;
    for (Eigen::SparseMatrix<double>::InnerIterator j(Q, i); j; ++j) {
      neighbours(k, i) = j.index() + 1;
      k++;
    }
  }
}
