using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators
using BasicPOMCP
using ARDESPOT
using FIB
using PointBasedValueIteration

include("DSAPOMDP.jl")
include("utils.jl")

P = DSAPOMDP()
rng = MersenneTwister(1)
up = DSABeliefUpdater(P)
b0 = initialize_belief(up)
s0 = rand(b0)

#simulating random policy with NothingUpdater
policy = RandomPolicy(P)
# (r1_tot, r1_disc) = simulate(P, NothingUpdater(), s0, b0, policy, rng)

#simulating POMCP policy our DSABeliefUpdater
solver = POMCPSolver(c=10.0)
planner = solve(solver, P);
# (r2_tot, r2_disc) = simulate(P, up, s0, b0, planner, rng)

#compare both policies
# println("Random policy: total reward = $r1_tot, discounted reward = $r1_disc")
# println("POMCP policy: total reward = $r2_tot, discounted reward = $r2_disc")

online_solver = DESPOTSolver(bounds=(-20.0, 0.0))
planner_despot = solve(online_solver, P)

offline_solver = PBVISolver()
pbvi_policy = solve(offline_solver, P)

# try to evaluate the policy for k=30 replications
function evaluate(v_policy::Vector, b0::Belief, up::DSABeliefUpdater, rng::AbstractRNG, P::DSAPOMDP, n::Int64)
    b = deepcopy(b0)
    result = Array{Float64}(undef, n, length(v_policy))
    for i in 1:n
        s0 = rand(b)
        for j in 1:length(v_policy)
            (r_tot, r_disc) = simulate(P, up, s0, b, v_policy[j], rng)
            result[i, j] = r_disc
        end
        println(i)
    end
    res_mean = mean(result, dims=1)
    res_std = std(result, dims=1)
    return (res_mean=res_mean, res_std=res_std)
end

planners = [policy, planner, planner_despot]
res_mean, res_std = evaluate(planners, b0, up, rng, P, 5)

(r2_tot, r2_disc) = simulate(P, up, s0, b0, planner_despot, rng)

# report the mean and standard deviation for the total discounted reward 
# make sure that each policy is evaluated on the same set of 30 replication for fairness
