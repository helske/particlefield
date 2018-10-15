context("Test graph building and printing")

test_that("Building graph works", {
  
  nnbs <- c(1, rep(2, 8), 1)
  nbs <- matrix(0, 10, 10)
  nbs[1, 1] <- 2
  for (i in 2:9) nbs[i, 1:2] <- c(i - 1, i + 1)
  nbs[10, 1] <- 9
  expect_output_file(print_graph(nnbs, nbs), file = "graph_output.txt")
})


