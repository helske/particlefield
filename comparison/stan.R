library(rstan)
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
d <- 1
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
for(i in 1:m) nbs[i, 1:nnbs[i]] <- g$nbs[[i]]

set.seed(42)
# sample states
L <- t(chol(solve(Q)))
x <- c(L %*%rnorm(m))

# and Binomial observations
mu <- 0
u <- rep(10, m)
y <- rbinom(m, u, plogis(mu + x))
idx <- 1:m
###


out <- psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=100000)
truth <- c(diagis::weighted_mean(t(out$states), out$weights))

stan_x <- psi_x <- matrix(NA, 544, 1000)
time_stan <- time_psi <- numeric(1000)
for(i in 1:1000){
  
  #time_stan[i] <- system.time(fit_stan <- stan("../stan/binomial_car_fixed.stan",verbose=FALSE, 
  #                 data = list(y = y, u = u, N = m,
  #                             nbs=nbs,nnbs1=nnbs), init = list(list(x=x)),
  #                 pars = c("x"), refresh=-1,
  #                 iter = 700, warmup = 200, chains = 1))[3]
  #stan_x[, i] <- summary(fit_stan)[[1]][1:544,"mean"]
  
  time_psi[i] <- system.time(
    out <- psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=200))[3]
  psi_x[, i] <- diagis::weighted_mean(t(out$states), out$weights)
  print(i)
  if(i > 1)
    print(c(mean(time_stan,na.rm=TRUE)*mean((stan_x-truth)^2,na.rm=TRUE), 
            mean(time_psi,na.rm=TRUE)*mean((psi_x-truth)^2,na.rm=TRUE)))
}


ts.plot(cbind(mean(time_stan,na.rm=TRUE)*apply(stan_x,1,var,na.rm=TRUE), mean(time_psi,na.rm=TRUE)*apply(psi_x, 1, var, na.rm=TRUE)),col=1:2)

mean(time_stan, na.rm=TRUE)
mean(time_psi, na.rm=TRUE)
ts.plot(rowMeans(stan_x,na.rm=TRUE) - rowMeans(psi_x, na.rm=TRUE))
ts.plot(cbind(apply(stan_x,1,sd,na.rm=TRUE), apply(psi_x, 1, sd, na.rm=TRUE)),col=1:2)

xhat <- diagis::weighted_mean(t(out$states), out$weights)
xhat_stan <- summary(fit_stan)[[1]][1:544,"mean"]

ts.plot(xhat-xhat_stan)

print(fit_stan, pars=c("d", "tau", "mu"),digits=5)

out <- mcmc_binomial_car(nnbs, nbs, tau, d, y, u, idx, use_mu=TRUE, n_iter=15000,n_burnin=5000, n_particles=100)
summary(out$theta)$stat

####
###

sd(replicate(100, psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=100, reorder=TRUE)$log))

fit_stan <- stan("../stan/binomial_car.stan",verbose=TRUE, 
                 data = list(y = y, u = u, N = m, tau=tau, d=d,
                             nbs=nbs,nnbs=nnbs), pars = c("x", "prob"),
                 iter = 11000, warmup = 1000, chains = 1)

out <- psi_car(nnbs, nbs, tau, d, y, u, idx, mu, n_particles=1000000, reorder=TRUE)

xhat <- diagis::weighted_mean(t(out$states),out$weights)

ts.plot(xhat-summary(fit_stan)$summary[1:544,"mean"])

xsd <- sqrt(diag(diagis::weighted_var(t(out$states),out$weights)))
ts.plot(xsd - summary(fit_stan)$summary[1:544, "sd"])


###
