library(smccar)
library(INLA)
library(spam) #for germany.plot
library(ggplot2)

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

colSds <- function(x) apply(x,2,sd)

## use the spatial structure of Germany
graph <- system.file("demodata/germany.graph", package="INLA")
g <- inla.read.graph(graph)
#inla.spy(g)

# create the precision matrix
d <- 0.1
tau <- 0.1 #0.05 was good
Q <- matrix(0, g$n, g$n)
diag(Q) <- tau * (d + g$nnbs)
for(i in 1:g$n) {
  if (g$nnbs[i] > 0) {
    Q[i, g$nbs[[i]]] <- -tau
    Q[g$nbs[[i]], i] <- -tau
  }
}

# build the graph for SMC
nnbs <- g$nnbs
m <- length(nnbs)
nbs <- matrix(0, m, m)
for(i in 1:m) nbs[i, 1:nnbs[i]] <- which(Q[i, ] == -tau)

set.seed(42)
# sample states
L <- solve(t(chol(Q)))
x <- L %*%rnorm(m)
germany.plot(x)

# and Binomial observations
mu <- 0
u <- rep(10, m)
y <- rbinom(m, u, plogis(mu + x))
idx <- 1:m
germany.plot(y)

psi_8192 <- function(n=2^13){
  x <- replicate(100, psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=n, reorder=TRUE)$log)
  c(mean(x), sd(x))
}
bsf_8192 <- function(n=2^13){
  x <- replicate(100, bsf_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=n, reorder=TRUE)$log)
  c(mean(x), sd(x))
}
sis_8192 <- function(n=2^13){
  x <- replicate(100, psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=n, reorder=TRUE,resamp=0)$log)
  c(mean(x), sd(x))
}
# psi_8192()
# bsf_8192()
# sis_8192()

# number of replications
nsim <- 500

# number of particles
n <- 2^(5:12)
psi_ordered <- bsf_ordered <- psi_random <- bsf_random <- sis_random <- sis_ordered <- matrix(NA, nsim, length(n))

set.seed(1)

for (i in seq_along(n)) {
  for (j in 1:nsim) {
    # reorder observations randomly
    perm <- random_permutation(nnbs, nbs, idx, y, u, Q, tau)
    nnbsj <- perm$nnbs
    nbsj <- perm$nbs
    yj <- perm$y
    uj <- perm$u
    psi_random[j, i] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=FALSE)$log
    bsf_random[j, i] <-  bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=FALSE)$log
    psi_ordered[j, i] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=TRUE)$log
    bsf_ordered[j, i] <-  bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=TRUE)$log
    sis_random[j, i] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=FALSE,resamp=0)$log
    sis_ordered[j, i] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, mu,n_particles=n[i], reorder=TRUE,resamp=0)$log
  }
  print(i)
  
  
  ts.plot(cbind(colSds(psi_random),
                colSds(psi_ordered),
                colSds(bsf_random),
                colSds(bsf_ordered),
                colSds(sis_random),
                colSds(sis_ordered)),col = c(1,1,2,2,3,3),lty=1:2,type="b")
  # ts.plot(cbind(colMeans(psi_random),
  #               colMeans(psi_ordered),
  #               colMeans(bsf_random),
  #               colMeans(bsf_ordered),
  #                 colMeans(sis_random),
  #                 colMeans(sis_ordered)),
  #          col = c(1,1,2,2),lty=c(1:2,1:2))
}


ll <- c(psi_random, psi_ordered, bsf_random, bsf_ordered, sis_random, sis_ordered)

results <- data.frame(loglik = ll, 
                      method = rep(factor(c("SMC-twist", "SMC-BSF", "SIS-twist"), ordered=TRUE), each = 2 * length(n) * nsim),
                      ordering = rep(factor(c("random", "AMD"), ordered=TRUE), each = length(n) * nsim),
                      particles = rep(factor(n, ordered=TRUE), each = nsim))

library(ggplot2)
ggplot(results, aes(x = particles, y = loglik, fill=interaction(ordering, method))) + 
  stat_boxplot(geom ='errorbar')+
  geom_boxplot()


##### change tau

# number of particles
tauseq <- c(0.1, 1, 10)
n_particles <- 2^(5:10)
nsim <- 1000
psi_ordered <- bsf_ordered <- psi_random <- bsf_random <- sis_random <- sis_ordered <- 
  psi_ordered_time <- bsf_ordered_time <- psi_random_time <- bsf_random_time <- sis_random_time <- sis_ordered_time <- 
  array(NA, c(nsim, length(n_particles),length(tauseq)))

for (k in seq_along(tauseq)) {
  
  d <- 0.1
  tau <- tauseq[k]
  Q <- matrix(0, g$n, g$n)
  diag(Q) <- tau * (d + g$nnbs)
  for(i in 1:g$n) {
    if (g$nnbs[i] > 0) {
      Q[i, g$nbs[[i]]] <- -tau
      Q[g$nbs[[i]], i] <- -tau
    }
  }
  
  # build the graph for SMC
  nnbs <- g$nnbs
  m <- length(nnbs)
  nbs <- matrix(0, m, m)
  for(i in 1:m) nbs[i, 1:nnbs[i]] <- which(Q[i, ] == -tau)
  
  set.seed(42)
  # sample states
  L <- solve(t(chol(Q)))
  x <- L %*%rnorm(m)
  
  # and Binomial observations
  mu <- 0
  u <- rep(10, m)
  y <- rbinom(m, u, plogis(mu + x))
  idx <- 1:m
  
  
  for (n in seq_along(n_particles)) {
    for (j in 1:nsim) {
      # reorder observations randomly
      perm <- random_permutation(nnbs, nbs, idx, y, u, Q, tau)
      nnbsj <- perm$nnbs
      nbsj <- perm$nbs
      yj <- perm$y
      uj <- perm$u
      psi_random_time[j,n,k] <- system.time(
        psi_random[j, n, k] <-  psi_car(nnbsj, nbsj,tau, d, yj, uj, idx, 
                                        mu,n_particles=n_particles[n], reorder=FALSE)$log)[3]
      bsf_random_time[j,n,k] <- system.time(
        bsf_random[j, n, k] <-  bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                        mu,n_particles=n_particles[n], reorder=FALSE)$log)[3]
      psi_ordered_time[j,n,k] <- system.time(
        psi_ordered[j, n, k] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                       mu,n_particles=n_particles[n], reorder=TRUE)$log)[3]
      bsf_ordered_time[j,n,k] <- system.time(
        bsf_ordered[j, n, k] <-  bsf_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                       mu,n_particles=n_particles[n], reorder=TRUE)$log)[3]
      sis_random_time[j,n,k] <- system.time(
        sis_random[j, n, k] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                      mu,n_particles=n_particles[n], reorder=FALSE,resamp=0)$log)[3]
      sis_ordered_time[j,n,k] <- system.time(
        sis_ordered[j, n, k] <-  psi_car(nnbsj, nbsj, tau, d, yj, uj, idx, 
                                       mu,n_particles=n_particles[n], reorder=TRUE,resamp=0)$log)[3]
    }
  }
  print(k)
}


ll <- c(psi_random, psi_ordered, bsf_random, bsf_ordered, sis_random, sis_ordered)

results <- data.frame(loglik = ll, 
                      method = rep(factor(c("SMC-twist", "SMC-BSF", "SIS-twist"), ordered=TRUE), 
                                   each = 2 * length(tauseq) * length(n_particles) * nsim),
                      ordering = rep(factor(c("random", "AMD"), ordered=TRUE), each = length(tauseq) * length(n_particles) * nsim),
                      tau = rep(factor(tauseq, ordered=TRUE), each = nsim * length(n_particles)),
                      n_particles = rep(factor(n_particles), each = nsim))
saveRDS(list(psi_random, psi_ordered, bsf_random, bsf_ordered, sis_random, sis_ordered, 
          psi_random_time, psi_ordered_time, bsf_random_time, bsf_ordered_time, sis_random_time, sis_ordered_time),
     file="tau_results.rda")
ggplot(results, aes(x = n_particles, y = loglik, fill=interaction(ordering, method))) + 
  geom_boxplot() + facet_wrap(~tau, scales = "free")

colMeans(sis_ordered_time)
colMeans(bsf_ordered_time)
colMeans(psi_ordered_time)

truths<-colMeans(psi_ordered[,6,])
psi_mse <- sapply(1:3, function(i) colMeans(psi_ordered[,,i]-truths[i])^2)
sis_mse <- sapply(1:3, function(i) colMeans(sis_ordered[,,i]-truths[i])^2)
ts.plot(cbind(psi_mse, sis_mse),col=1:6)
