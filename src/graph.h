// Graph object

#ifndef GRAPH_H
#define GRAPH_H

#include <RcppEigen.h>

class graph {
  
public:
  // constructor
  graph(
    const Eigen::Ref<const Eigen::VectorXi>& number_of_neighbours_, 
    const Eigen::Ref<const Eigen::MatrixXi>& neighbours_);
  // print graph
  void print_graph();
  // reoder graph
  void reorder(Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>& P);
  
  Eigen::VectorXi number_of_neighbours;
  Eigen::MatrixXi neighbours;
  const unsigned int number_of_vertices;
  const unsigned int number_of_edges;
};



#endif
