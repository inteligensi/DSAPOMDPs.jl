using POMDPs
using POMDPTools
using DSAPOMDPs
using Random 
using ParticleFilters
using Distributions

N = 100
P = DSAPOMDP()
π_random = RandomPolicy(P)
rng = MersenneTwister(1)
up = BootstrapFilter(P, N, rng)
b0 = initialize_belief(up)
n_reps = 50
max_steps = 24

hr = HistoryRecorder(rng=rng, max_steps=max_steps)

# h_random = simulate(hr, P, π_random, up, b0)
# r_random = compute_rdisc(P, h_random);
