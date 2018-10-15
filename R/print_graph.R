#' Print the graph structure to console
#'
#' @param nnbs Vector defining the number of neighbours for each vertex.
#' @param nbs Matrix of indices of defining neighbours for each vertex.
#'
#' @export
#'
#' @examples
#' # Graph on line with 10 vertices
#' nnbs <- c(1, rep(2, 8), 1)
#' nbs <- matrix(0, 10, 10)
#' nbs[1, 1] <- 2
#' for (i in 2:9) nbs[i, 1:2] <- c(i - 1, i + 1)
#' nbs[10, 1] <- 9
#' print_graph(nnbs, nbs)
#' 
print_graph <- function(nnbs, nbs) {
  
  storage.mode(nbs) <- storage.mode(nnbs) <- "integer"
  R_print_graph(nnbs, nbs)
  
} 