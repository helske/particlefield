list_to_matrix <- function(x) {
  m <- matrix(0, max(lengths(x)), length(x))
  for (i in seq_along(x)) {
    m[seq_along(x[[i]]), i] <- x[[i]]
  }
  m
}
