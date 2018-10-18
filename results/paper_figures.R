library(particlefield)
# install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
library(INLA) # for map of Germany

library(smccar)
library(Matrix)
library(foreach)
library(doParallel)
set.seed(1)

experiment <- function(tau, nsim, n_cores = 24) {
  set.seed(42)
  
  # randomly reorder the states i.e. the processing order in SMC
  random_permutation <- function(nnbs, nbs, idx, y, u, Q, tau){
    jdx <- sample(idx)
    Qj <- Q[jdx,jdx]
    nnbs <- nnbs[jdx]
    y <- y[jdx]
    u <- u[jdx]
    nbs <- matrix(0, m, m)
    for(i in 1:nrow(Q)) nbs[i, 1:nnbs[i]] <- which(Qj[i, ] == -tau)
    
    list(nnbs=nnbs, nbs=nbs,y=y,u=u,jdx=jdx)
  }
  
  
  n_particles <- c(64, 1024)
  replication <- 1:nsim
  ordering <- factor(c("random", "AMD"))
  method <- factor(c("SMC-twist", "SIS", "SMC-base"))
  
  
  # use the spatial structure of Germany
  graph <- system.file("demodata/germany.graph", package = "INLA")
  g <- inla.read.graph(graph)
  
  # build the graph for smccar
  nnbs <- g$nnbs
  m <- length(nnbs)
  nbs <- matrix(0, m, m)
  for(i in 1:m) nbs[i, 1:nnbs[i]] <- g$nbs[[i]]
  
  
  
  d <- 1 # ensures positive definite covariance matrix
  idx <- 1:m # each observation depends on one state
  u <- rep(10, m) # number of trials for Binomial
  mu <- 0 # intercept
  
  Q <- matrix(0, g$n, g$n)
  diag(Q) <- tau * (d + g$nnbs)
  for(i in 1:g$n) {
    if (g$nnbs[i] > 0) {
      Q[i, g$nbs[[i]]] <- -tau
      Q[g$nbs[[i]], i] <- -tau
    }
  }
  
  # sample states
  L <- solve(t(chol(Q)))
  x <- L %*%rnorm(m)
  
  # and Binomial observations
  y <- rbinom(m, u, plogis(mu + x))
  
  
  results <- data.frame(logLik=NA, time = NA, 
                                 expand.grid(method=method, 
                                             ordering=ordering, 
                                             n_particles=n_particles, 
                                             replication=replication))
  
  
  attr(results, "approx_loglik") <- approximate_binomial_car(
    nnbs, nbs, tau, d, y, u, idx, mu)$log
  
  attr(results, "twist_1e5") <- psi_car(nnbs, nbs, tau, d, y, u, idx, 
                                        mu, n_particles=1e5,
                                        reorder=TRUE,
                                        ess_threshold = 1)$log
  attr(results, "base_1e5") <- bsf_car(nnbs, nbs, tau, d, y, u, idx, 
                                       mu, n_particles=1e5,
                                       reorder=TRUE,
                                       ess_threshold = 1)$log
  
  cl <- makeCluster(n_cores, outfile="")
  registerDoParallel(cl)
  results[, c("time", "logLik")] <- foreach(k = 1:nrow(results), .packages="smccar", .combine='rbind', .inorder = TRUE) %dopar% {
    
    # reorder observations randomly
    perm <- random_permutation(nnbs, nbs, idx, y, u, Q, tau)
    nnbsj <- perm$nnbs
    nbsj <- perm$nbs
    yj <- perm$y
    uj <- perm$u
    
    print(k)
    switch(as.character(results$method[k]),
           "SMC-base" = {
             time <- system.time(
               loglik <- bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                 mu, n_particles=results$n_particles[k], 
                                 reorder=results$ordering[k] == "AMD",
                                 ess_threshold = 0.5)$log)[3]
             c(time, loglik)
           },
           SIS = {
             time <- system.time(
               loglik <- psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                 mu, n_particles=results$n_particles[k],
                                 reorder=results$ordering[k] == "AMD",
                                 ess_threshold = 0)$log)[3]
             c(time, loglik)
           },
           "SMC-twist" = {
             time <- system.time(
               loglik <- psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                 mu, n_particles=results$n_particles[k],
                                 reorder=results$ordering[k] == "AMD",
                                 ess_threshold = 0.5)$log)[3]
             c(time, loglik)
           })
    
    
  }
  stopCluster(cl)
  results
}

tau_0.1_results <- experiment(tau = 0.1, nsim = 10000, n_cores = 56)
saveRDS(tau_0.1_results, file="tau_0.1_d_1_results.rda")

###

results <- readRDS("comparison/tau_0.1_d_1_results.rda")

library(dplyr)
library(ggplot2)
approx <- data.frame(n_particles = factor(c(64, 1024)), laplace=attr(results, "approx_loglik"))
attr(results,"twist_1e5") #-3294.345
attr(results,"base_1e5") #-3294.346

truth <- data.frame(n_particles = factor(c(64, 1024)), truth=attr(results,"twist_1e5"))

ggplot(results,
       aes(x = factor(method, labels = c("SIS", "SMC-Base", "SMC-Twist")), y = logLik, 
           fill=factor(ordering, labels = c("AMD", "Random")))) + 
  theme(legend.position="bottom") +
  geom_hline(aes(yintercept=laplace), data=approx, lty = 3) +
  geom_hline(aes(yintercept=truth), data=truth, lty = 2) +
  geom_boxplot(coef=100, position = position_dodge(0.8), fatten = 1) + 
  facet_wrap(~factor(n_particles, labels=c("N = 64","N = 1024")), scales="free") + 
  scale_y_continuous("Log Z") + scale_x_discrete("Method") + 
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black")) + 
  theme(legend.justification="right", legend.position=c(0.45,0.16)) + 
  labs(fill="Ordering")


library(patchwork)
filter(results, n_particles == 64) -> data64
filter(results, n_particles == 1024) -> data1024

# compute lower and upper whiskers
ylim64 <- c(
  boxplot.stats(filter(data64, method == "SMC-base" & ordering == "random")$logLik)$stats[1],
  boxplot.stats(filter(data64, method == "SMC-base" & ordering == "AMD")$logLik)$stats[5])

ylim1024 <-  c(
  boxplot.stats(filter(data1024, method == "SMC-base" & ordering == "random")$logLik)$stats[1],
  boxplot.stats(filter(data1024, method == "SMC-base" & ordering == "random")$logLik)$stats[5])


p1 <- ggplot(data64,
             aes(x = method, y = logLik, 
                 fill=ordering)) + ggtitle("N = 64") +
  geom_hline(aes(yintercept=laplace), data=approx, lty = 3) +
  geom_hline(aes(yintercept=truth), data=truth, lty = 2) +
  geom_boxplot(outlier.color = NA, position = position_dodge(0.8), fatten = 1) + 
  scale_y_continuous("log Z") + scale_x_discrete("method") + 
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black")) + 
  coord_cartesian(ylim = ylim64 + c(-1, 1)) + 
  theme(legend.position=c(0.83, 0.2))

p2 <- ggplot(data1024,
             aes(x = factor(method), y = logLik, 
                 fill=ordering)) + ggtitle("N = 1024") +
  geom_hline(aes(yintercept=laplace), data=approx, lty = 3) +
  geom_hline(aes(yintercept=truth), data=truth, lty = 2) +
  geom_boxplot(outlier.color=NA, position = position_dodge(0.8), fatten = 1,
               show.legend = FALSE) + 
  scale_y_continuous("log Z") + scale_x_discrete("Method") + 
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black")) + 
  coord_cartesian(ylim = ylim1024 + c(-0.1, 0.1))

p1 + p2




## remove outliers manually in order to tighten the plotting range
ggsave("results/car_example.pdf")

