# Benchmarking running times of SMC-twist, SMC-base and the Gaussian approximation
# Note that runtime t(SMC-twist) - t(Laplace) > t(SMC-base), 
# as the core of SMC-twist does not perform Cholesky factorization whereas SMC-base 
# and Laplace (which SMC-twist uses) does. And there are extra density computations
# 
# Also note that here we have used fixed number of particles. In reality, 
# One should increase the number of particles as data size increases.
# See ire.R for another benchmark relating to this.
#
#
library(particlefield)
library(microbenchmark)
library(ggplot2)
library(scales)  

set.seed(1)

sizes <- 2^(7:10)
n <- length(sizes)

levs <- c("SMC-base", "SMC-twist", "IS-twist", 
  "SMC-base", "SMC-twist", "IS-twist",
  "Laplace",  "SMC-twist - Laplace", "SMC-twist - Laplace",
  "IS-twist - Laplace", "IS-twist - Laplace")

dat <- data.frame(
  n_particles = factor(c(10, 10, 10, 100, 100, 100, "NA", 10, 100, 10, 100)),
  method = factor(levs), 
  lq = NA, median = NA, uq = NA, T = factor(rep(sizes, each = 11)))

tau <- 1
d <- 1 # ensures positive definite covariance matrix
mu <- 0 # intercept
nsim <- 10
for(j in 1:n) {
  m <- sizes[j]
  
  idx <- 1:m # each observation depends on one state
  u <- rep(10, m) # number of trials for Binomial
  Q <- matrix(0, m, m)
  Q[upper.tri(Q)] <- sample(0:1, size = m * (m - 1) / 2, prob = c(0.9, 0.1), replace = TRUE)
  Q[cbind(1:(m-1),2:m)] <- 1
  Q <- -(Q + t(Q))
  diag(Q) <- rowSums(Q != 0)
  nnbs <- diag(Q)
  nbs <- matrix(0, m, m)
  for(i in 1:m) nbs[i, 1:nnbs[i]] <- which(Q[i, ] == -1)
  diag(Q) <- (diag(Q) + d) * tau
  
  # sample states
  L <- solve(t(chol(Q)))
  x <- c(L %*%rnorm(m))
  
  # and Binomial observations
  y <- rbinom(m, u, plogis(mu + x))
  
  bm <- summary(microbenchmark(
    bsf_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 10), 
    psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 10),
    psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 10, ess = 0),
    bsf_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 100),
    psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 100),
    psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles = 100, ess = 0),
    approximate_binomial_car(nnbs, nbs, tau, d, y, u, idx, mu, ratio_correction = FALSE), 
    times = nsim), unit="s")[,c("lq", "median", "uq")]
  
  dat[(j - 1) * 11 + 1:11, c("lq", "median", "uq")] <- 
    rbind(bm,
      c(NA, bm[2, "median"] - bm[7, "median"], NA),
      c(NA, bm[5, "median"] - bm[7, "median"], NA),
      c(NA, bm[3, "median"] - bm[7, "median"], NA),
      c(NA, bm[6, "median"] - bm[7, "median"], NA))
  
  print(j)
}

ggplot(data = dat, aes(x=T, y = median, group = interaction(n_particles, method),
  color = method)) + scale_y_continuous(trans = log2_trans(),
    breaks = trans_breaks("log2", function(x) 2^x),
    labels = trans_format("log2", math_format(2^.x))) +
  geom_point(aes(shape = n_particles),size=3, alpha = 0.5) + ylab("Time (seconds)")
