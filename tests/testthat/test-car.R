context("Test CAR model building")

test_that("Building CAR model works", {
  require("Matrix")
  set.seed(1)
  m <- 10
  tau <- 1
  d <- 1
  idx <- sample(m * (m + 1) / 2, 2 * m)
  Q <- matrix(0, m, m)
  Q[upper.tri(Q)] <- sample(0:1, size = m * (m - 1) / 2, replace = TRUE)
  Q <- Q + t(Q)
  Q <- -Q
  diag(Q) <- 0
  diag(Q) <- rowSums(Q != 0)
  nnbs <- diag(Q)
  nbs <- matrix(0, m, m)
  for(i in 1:m) nbs[i, 1:nnbs[i]] <- which(Q[i, ] == -1)
  
  diag(Q) <- (diag(Q) + d) * tau
  
  storage.mode(nnbs) <- storage.mode(nbs) <- "integer"
  expect_identical(Matrix(particlefield:::R_car_Q(nnbs, nbs, 1,1)), Matrix(Q))
  
})


