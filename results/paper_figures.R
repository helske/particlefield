library(particlefield)
library(INLA) # for map of Germany
library(ggplot2)

# randomly reorder the states i.e. the processing order in SMC
random_permutation <- function(nnbs, nbs, idx, y, u, Q, tau){
  jdx <- sample(idx)
  Qj <- Q[jdx, jdx]
  nnbs <- nnbs[jdx]
  y <- y[jdx]
  u <- u[jdx]
  nbs <- matrix(0, m, m)
  for(i in 1:nrow(Q)) nbs[i, 1:nnbs[i]] <- which(Qj[i, ] == -tau)
  
  list(nnbs = nnbs, nbs = nbs, y = y, u = u, jdx = jdx)
}


## use the spatial structure of Germany
graph <- system.file("demodata/germany.graph", package = "INLA")
g <- inla.read.graph(graph)
# build the graph for SMC
nnbs <- g$nnbs
m <- length(nnbs)
nbs <- matrix(0, m, m)
for(i in 1:m) nbs[i, 1:nnbs[i]] <- g$nbs[[i]]


# different values for the precision
tauseq <- c(0.1, 1)
# number of particles
n_particles <- 2^c(6, 8, 10)
# adaptive resampling settings
ess <- c(1, 0.5, 0)
# number of replications
nsim <- 1000
replication <- 1:nsim
ordering <- factor(c("random", "AMD"))
method <- factor(c("SMC-twist", "BSF"))
# create table for results
results <- data.frame(
  logLik = NA, 
  time = NA, 
  expand.grid(method = method, 
              ordering = ordering, 
              ess = ess, 
              n_particles = n_particles, 
              replication = replication, 
              tau = tauseq))
results <- results[!(results$method == "BSF" & results$ess == 0), ]
nrows <- nrow(results)

d <- 0.1 # ensures positive definite covariance matrix
idx <- 1:m # each observation depends on one state
u <- rep(10, m) # number of trials for Binomial
mu <- 0 # intercept
tau <- 0 #dummy for start

set.seed(1)
for (k in 1:nrows) {
  
  if (results$tau[k] != tau) { # new parameters
    
    # create precision matrix
    tau <- results$tau[k]
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
  }
  
  # reorder observations randomly
  perm <- random_permutation(nnbs, nbs, idx, y, u, Q, tau)
  nnbsj <- perm$nnbs
  nbsj <- perm$nbs
  yj <- perm$y
  uj <- perm$u
  
  if (results$method[k] == "BSF") {
    results$time[k] <- system.time(results$logLik[k] <- bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
     mu, n_particles=results$n_particles[k], reorder=results$ordering[k] == "AMD",
     ess_threshold = results$ess[k])$log)[3]
  } else {
    results$time[k] <- system.time(results$logLik[k] <- psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
      mu, n_particles=results$n_particles[k], reorder=results$ordering[k] == "AMD",
      ess_threshold = results$ess[k])$log)[3]
  }
  
  saveRDS(results, file="tau_results.rda")
  print(k)
}

ggplot(results, aes(x = n_particles, y = logLik, fill=interaction(ordering, ess, method))) + 
  geom_boxplot() + facet_wrap(~tau, scales = "free")

results <- readRDS("tau_0.1_results.rda")
results$n_particles <- factor(results$n_particles)
ggplot(filter(results, n_particles %in% c(64,256,1024)), aes(x = n_particles, y = logLik, fill=interaction(ordering, ess, method))) + 
  geom_boxplot()

sum(!is.na(results$logLik))
results[970,]$replication

true <- mean(results$logLik[as.numeric(results$n_particles)==6 & results$method=="SMC-twist" & results$ess==1], na.rm = TRUE)

mse <- sapply(split(results,interaction(results$n_particles, results$method, results$ordering, results$ess)), 
              function(x) mean(x$time,na.rm=TRUE) * mean((x$logLik - true)^2,na.rm=TRUE))

compute_ire <- function(time, logLik) mean(time,na.rm=TRUE) * mean((logLik - true)^2,na.rm=TRUE)

library(dplyr)

results %>% 
  group_by(n_particles, method, ordering, ess) %>% 
  summarise(IRE = compute_ire(time, logLik)) %>% 
  ungroup() -> ires

ggplot(ires, aes(x = n_particles, y = log(IRE), 
                 color = interaction(ordering, ess, method),
                 group=interaction(ordering, ess, method))) + geom_line()

