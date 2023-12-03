using POMDPs
using POMDPTools
using DSAPOMDPs
using Random 
using ParticleFilters
using Distributions

N = 100
P = DSAPOMDP()
rng = MersenneTwister(1)
up = BootstrapFilter(P, N, rng)
b0 = initialize_belief(up)
n_reps = 100
max_steps = 24

hr = HistoryRecorder(rng=rng, max_steps=max_steps)