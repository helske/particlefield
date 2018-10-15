context("Test Gaussian approximation")


test_that("Approximation of Binomial CAR with SMCfields and INLA coincide", {
  
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
  
  # states
  L <- t(chol(Q))
  x <- L %*%rnorm(m)
  
  # observations
  mu <- 2
  u <- sample(1:10, size = m, replace = TRUE)
  y <- rbinom(m, u, plogis(mu + x)) # p observations per state
  extra <- c(1, 1, 4, 10, 6)
  y <- c(y, rbinom(5, u[extra], plogis(mu + x[extra])))
  u <- c(u, u[extra])
  idx <- c(1:m, extra)
  mode <- approximate_binomial_car(nnbs, nbs, tau, d, y, u, idx, mu,TRUE,reorder=FALSE)
  
  library(INLA)
  inla_g <- inla.read.graph(Q)
  fit <- inla(y ~  f(region, model = "besagproper", graph = inla_g), 
    family = "binomial", Ntrials = u, data = data.frame(y = y, region = idx), 
    control.fixed = list(mean.intercept = mu, prec.intercept = exp(15)),
    control.mode = list(theta = c(0,0), fixed = TRUE), control.compute = list(config = TRUE), 
    control.inla = list(strategy = "gaussian"), num.threads = 1)
  
  fit$misc$configs$config[[1]]$mean[(length(y)+1):(length(y)+m)]
  mode$mean
  expect_equal(fit$misc$configs$config[[1]]$mean[(length(y)+1):(length(y)+m)], mode$mean, tolerance = 1e-3)
})

  