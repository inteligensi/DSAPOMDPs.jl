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

#simulating POMCP policy our DSABeliefUpdater
solver = POMCPSolver()
pomcp_planner = solve(solver, P)
#(r_tot, r_disc) = simulate(P, up, s0, b0, planner, rng)

num_runs = 30
num_steps = 12
benchmarks = ["Random", "POMCP"]
policies = [RandomPolicy(P), pomcp_planner]
(means, stds) = evaluate(P=P, up=up, b0=b0, policies=policies, rng=rng, num_reps=num_runs, num_steps=num_steps)

#print each policy mean ± std
for (i, policy) in enumerate(benchmarks)
    println("Policy $policy: $(means[i]) ± $(stds[i])")
end

#simulate
simulate(P, up, s0, b0, pomcp_planner, rng, max_steps=12, verbose=true)

#implement simple PBVI

policy = RandomPolicy(P)
a = action(policy, b0)
(s, o, r) = gen(P, s0, a, rng)