# Benchmarking of sd(logLik)
#
library(particlefield)
library(dplyr)

set.seed(1)

n_particles <- 2^(4:8)
sizes <- 2^c(7:11)
n <- length(sizes)

methods <- c("SMC-base", "SMC-twist", "IS-twist", "approx")
nsim <- 100

n_m <- length(methods)
n_p <- length(n_particles)
n_T <- length(sizes)
dat <- data.frame(
  method = rep(factor(methods), each = nsim),
  n_particles = rep(n_particles, each = n_m * nsim),
  T = rep(sizes, each = n_p * n_m * nsim),
  tau = rep(c(0.1, 1, 10), each = n_T * n_p * n_m * nsim),
  logLik = NA, time = NA)
dat <- dat[-which(dat$method %in% c("approx") & dat$n_particles != n_particles[1]), ]
dat$n_particles[dat$method %in% c( "approx")] <- 0

d <- 1 # ensures positive definite covariance matrix
mu <- 1 # intercept
trials <- 1
m <- 0
tau <- 0
for(j in 1:nrow(dat)) {
  
  if(m != dat$T[j] | tau != dat$tau[j]) {
    tau <- dat$tau[j]
    m <- dat$T[j]
    
    idx <- 1:m # each observation depends on one state
    u <- rep(trials, m) # number of trials for Binomial
    Q <- matrix(0, m, m)
    # sparse Q
    Q[upper.tri(Q)] <- sample(0:1, size = m * (m - 1) / 2, prob = c(0.95, 0.05), replace = TRUE)
    Q[cbind(1:(m-1),2:m)] <- 1 # ensure connectivity
    Q <- -(Q + t(Q))
    diag(Q) <- rowSums(Q != 0)
    nnbs <- diag(Q)
    nbs <- matrix(0, m, m)
    for(i in 1:m) nbs[i, 1:nnbs[i]] <- which(Q[i, ] == -1)
    diag(Q) <- (diag(Q) + d)
    Q <- tau * Q
    
    # sample states
    L <- solve(t(chol(Q)))
    x <- c(L %*%rnorm(m))
    
    # and Binomial observations
    y <- rbinom(m, u, plogis(mu + x))
    
    # inla_g <- inla.read.graph(Q)
  }
  gc()
  switch(as.character(dat$method[j]),
    "SMC-base" =
      dat[j, "time"] <- system.time(gcFirst = TRUE,
        dat[j, "logLik"] <- bsf_car(nnbs, nbs, tau, d, y, u, idx, mu, 
          n_particles = dat$n_particles[j])$log)[3],
    "SMC-twist" =
      dat[j, "time"] <- system.time(gcFirst = TRUE,
        dat[j, "logLik"] <- psi_car(nnbs, nbs, tau, d, y, u, idx, mu, 
          n_particles = dat$n_particles[j], ess = 1)$log)[3],
    "IS-twist" = 
      dat[j, "time"] <- system.time(gcFirst = TRUE,
        dat[j, "logLik"] <-psi_car(nnbs, nbs, tau, d, y, u, idx, mu, 
          n_particles = dat$n_particles[j], ess = 0)$log)[3],
    "approx" =
      dat[j, "time"] <- system.time(gcFirst = TRUE,
        dat[j, "logLik"] <- approximate_binomial_car(nnbs, nbs, tau, 
          d, y, u, idx, mu)$log)[3]
    # "INLA" =
    #   dat[j, "time"] <- system.time(gcFirst = TRUE,
    #     dat[j, "logLik"] <- inla(y ~ f(region, model = "besagproper", graph = inla_g), 
    #       family = "binomial", Ntrials = u, data = data.frame(y = y, region = idx), 
    #       control.fixed = list(mean.intercept = mu, prec.intercept = exp(20)),
    #       control.mode = list(theta = c(log(tau),0), fixed = TRUE), 
    #       control.inla = list(strategy = "simplified.laplace"), 
    #       num.threads = 1)$mlik[1])[3]
  )
  
  saveRDS(dat, "simulated_data_experiment_sparse.rda")
  print(round(j/nrow(dat)*100, 2))
}

