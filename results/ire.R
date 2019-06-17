# Benchmarking of sd(logLik)
#
library(particlefield)
library(microbenchmark)
library(ggplot2)
library(scales)  
library(INLA)
library(dplyr)

set.seed(1)

n_particles <- 2^(4:8)
sizes <- 2^c(7:10)
n <- length(sizes)

methods <- c("SMC-base", "SMC-twist", "IS-twist", "INLA", "approx")
nsim <- 500

dat <- data.frame(
  method = rep(factor(methods), each = nsim),
  n_particles = rep(n_particles, each = length(methods) * nsim),
  T = rep(sizes, each = length(methods) * length(n_particles) * nsim),
  logLik = NA, time = NA)
dat <- dat[-which(dat$method %in% c("INLA", "approx") & dat$n_particles != n_particles[1]), ]
dat$n_particles[dat$method %in% c("INLA", "approx")] <- 0

tau <- 0.1
d <- 1 # ensures positive definite covariance matrix
mu <- 0 # intercept
trials <- 10
m <- 0
for(j in 1:nrow(dat)) {
  
  if(m != dat$T[j]) {
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
    
    inla_g <- inla.read.graph(Q)
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
          d, y, u, idx, mu)$log)[3],
    "INLA" =
      dat[j, "time"] <- system.time(gcFirst = TRUE,
        dat[j, "logLik"] <- inla(y ~ f(region, model = "besagproper", graph = inla_g), 
          family = "binomial", Ntrials = u, data = data.frame(y = y, region = idx), 
          control.fixed = list(mean.intercept = mu, prec.intercept = exp(20)),
          control.mode = list(theta = c(log(tau),0), fixed = TRUE), 
          control.inla = list(strategy = "simplified.laplace"), 
          num.threads = 1)$mlik[1])[3]
  )
  
  print(j)
  # if(j %% 100 == 0) {
  #   dat2 <- dat
  #   dat2$n_particles <- as.factor(dat2$n_particles)
  #   dat2$T <- as.factor(dat2$T)
  #   dat2 %>% group_by(method, n_particles, T) %>%
  #     summarize(sd = sd(logLik, na.rm = TRUE), time = mean(time, na.rm = TRUE)) -> results
  #   print(
  #     ggplot(data = results[results$n_particles != 0, ], 
  #       aes(x=factor(n_particles), y = time, group = method, color = method)) + 
  #       geom_point(size=3, alpha = 0.5) + scale_y_log10() + 
  #       ylab("IRE") + xlab("Number of particles") + 
  #       facet_wrap(~T, labeller = label_both) +
  #       geom_hline(data = results[results$n_particles == 0, ], 
  #         aes(yintercept = time, color = method, group = method))
  #   )
  # }
}

dat <- readRDS("results/simulated_data_experiment_sparse.rda")

dat2 <- dat
dat2$n_particles <- as.factor(dat2$n_particles)
dat2$T <- as.factor(dat2$T)
dat2 %>% group_by(method, n_particles, T) %>%
  summarize(sd = sd(logLik), mean = mean(logLik),
    time = mean(time)) -> results

results$MSE <- 0

for(N in unique(results$T)) {
  for(tau in unique(results$tau)) {
    for(method in c("SMC-twist", "approx", "IS-twist", "SMC-base")) {
      if (method == "SMC-twist") {
        ref <- mean(dat2$logLik[dat2$method==method & dat2$T == N & dat2$n_particles == 256 & dat2$tau == tau])
      }
      for(n in unique(results$n_particles)) {
        results$MSE[results$method==method & results$T == N & results$n_particles == n & results$tau == tau] <- 
          mean((dat2$logLik[dat2$method==method & dat2$T == N & dat2$n_particles == n & dat2$tau == tau] - ref)^2)
      }
    }
  }
}


ggplot(data = results[results$n_particles != 0, ],
  aes(x=factor(n_particles), y = mean, group = method, color = method)) +
  geom_point(size=3, alpha = 0.5) + #scale_y_log10() +
  ylab("logLik") + xlab("Number of particles") +
  facet_wrap(~T, labeller = label_both, scales = "free") +
  geom_hline(data = results[results$n_particles == 0, ],
    aes(yintercept = mean, color = method, group = method))

ggplot(data = results[results$n_particles != 0, ],
  aes(x=factor(n_particles), y = time, group = method, color = method)) +
  geom_point(size=3, alpha = 0.5) + scale_y_log10() +
  ylab("time") + xlab("Number of particles") +
  facet_wrap(~interaction(T,tau), labeller = label_both) +
  geom_hline(data = results[results$n_particles == 0, ],
    aes(yintercept = time, color = method, group = method))


ggplot(data = results[results$n_particles != 0, ],
  aes(x=factor(n_particles), y = MSE, group = method, color = method)) +
  geom_point(size=3, alpha = 0.5) + scale_y_log10() +
  ylab("MSE") + xlab("Number of particles") +
  facet_wrap(~interaction(T,tau), labeller = label_both, scales = "free") +
  geom_hline(data = results[results$n_particles == 0, ],
    aes(yintercept = MSE, color = method, group = method))

ggplot(data = results[results$n_particles != 0, ],
  aes(x=factor(n_particles), y = MSE * time, group = method, color = method)) +
  geom_point(size=3, alpha = 0.5) + scale_y_log10() +
  ylab("IRE") + xlab("Number of particles") +
  facet_wrap(~interaction(T,tau), labeller = label_both, scales = "free") +
  geom_hline(data = results[results$n_particles == 0, ],
    aes(yintercept = MSE * time, color = method, group = method))

ggplot(data = results[results$n_particles != 0, ],
  aes(x=factor(n_particles), y = sd, group = method, color = method)) +
  geom_point(size=3, alpha = 0.5) + scale_y_log10() +
  ylab("SD") + xlab("Number of particles") +
  facet_wrap(~interaction(T,tau), labeller = label_both, scales = "free") 

ggplot(data = results[results$n_particles != 0, ], 
  aes(x=factor(n_particles), y = time, group = method, color = method)) + 
  geom_point(size=3, alpha = 0.5) + scale_y_log10() + 
  ylab("IRE") + xlab("Number of particles") + 
  facet_wrap(~T, labeller = label_both) +
  geom_hline(data = results[results$n_particles == 0, ], 
    aes(yintercept = time, color = method, group = method))

ggplot(data = dat, 
  aes(x=factor(n_particles), y = IRE, group = method, color = method)) + 
  geom_point(size=3, alpha = 0.5) +  scale_y_log10() +
  ylab("IRE") + xlab("Number of particles") + facet_wrap(~T, labeller = label_both)

ggplot(data = dat, 
  aes(x=factor(n_particles), y = time, group = method, color = method)) + 
  geom_point(size=3, alpha = 0.5) + scale_y_log10() +
  ylab("time") + xlab("Number of particles") + facet_wrap(~T, labeller = label_both)

ggplot(data = dat, 
  aes(x=factor(n_particles), y = MSE, group = method, color = method)) + 
  geom_point(size=3, alpha = 0.5) +  scale_y_log10() +
  ylab("MSE") + xlab("Number of particles") + facet_wrap(~T, labeller = label_both)
