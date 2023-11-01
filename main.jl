using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators
using BasicPOMCP

include("DSAPOMDP.jl")
include("utils.jl")

P = DSAPOMDP()
rng = MersenneTwister(1)
up = DSABeliefUpdater(P)
b0 = initialize_belief(up)
s0 = rand(b0)

#simulating random policy with NothingUpdater
policy = RandomPolicy(P)
(r1_tot, r1_disc) = simulate(P, NothingUpdater(), s0, b0, policy, rng)

#simulating POMCP policy our DSABeliefUpdater
solver = POMCPSolver(c=10.0)
planner = solve(solver, P);
(r2_tot, r2_disc) = simulate(P, up, s0, b0, planner, rng)

#compare both policies
println("Random policy: total reward = $r1_tot, discounted reward = $r1_disc")
println("POMCP policy: total reward = $r2_tot, discounted reward = $r2_disc")

# try to evaluate the policy for k=30 replications
# report the mean and standard deviation for the total discounted reward 
# make sure that each policy is evaluated on the same set of 30 replication for fairness
