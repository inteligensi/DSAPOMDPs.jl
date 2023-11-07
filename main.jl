using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators
using BasicPOMCP
using ParticleFilters

include("DSAPOMDP.jl")
include("utils.jl")

P = DSAPOMDP()
rng = MersenneTwister(1)
b0 = initialize_belief(DSABeliefUpdater(P))
s0 = rand(b0)

N = 1000
up = BootstrapFilter(P, N, rng)
policy = RandomPolicy(P)

hr = HistoryRecorder(rng=rng, max_steps=12)
history = simulate(hr, P, policy, up, b0)


using Plots

@gif for (i, b) in enumerate(belief_hist(history))
    local ane_values = [s.ane for s in particles(b)]
    local avm_values = [s.avm for s in particles(b)]
    local occ_values = [s.occ for s in particles(b)]
    local time_values = [s.time for s in particles(b)]
    
    p1 = histogram(
        ane_values,
        xlim = (0,1.5),
        ylim = (0,1),
        nbins = 2,
        title = "IsAneurysm (t: $i)",
        normalize = :probability,
        ticks = [false, true], 
        legend = false
    )
    p2 = histogram(
        avm_values,
        ylim = (0,1),
        xlim = (0,1.5),
        nbins = 2,
        title = "IsAvm (t: $i)",
        normalize = :probability,
        ticks = [false, true], 
        legend = false
    )
    p3 = histogram(
        occ_values,
        ylim = (0,1),
        xlim = (0,1.5),
        nbins = 2,
        title = "IsOcc (t: $i)",
        normalize = :probability,
        ticks = [false, true], 
        legend = false
    )
    plot(p1, p2, p3, layout = (1, 3), link = :y)
end

#simulate random policy
# policy = RandomPolicy(P)
# for (s, a, sp, o) in stepthrough(P, policy, NothingUpdater(), s0, max_steps=12, rng=rng)
#     println("s: $s, a: $a, o: $o")
# end

#simulating POMCP policy our DSABeliefUpdater
# solver = POMCPSolver()
# pomcp_planner = solve(solver, P)
# #(r_tot, r_disc) = simulate(P, up, s0, b0, planner, rng)

# num_runs = 30
# num_steps = 12
# benchmarks = ["Random", "POMCP"]
# policies = [RandomPolicy(P), pomcp_planner]
# (means, stds) = evaluate(P=P, up=up, b0=b0, policies=policies, rng=rng, num_reps=num_runs, num_steps=num_steps)

# #print each policy mean ± std
# for (i, policy) in enumerate(benchmarks)
#     println("Policy $policy: $(means[i]) ± $(stds[i])")
# end

# #simulate
# simulate(P, up, s0, b0, pomcp_planner, rng, max_steps=12, verbose=true)

# #implement simple PBVI

# policy = RandomPolicy(P)
# a = action(policy, b0)
# (s, o, r) = gen(P, s0, a, rng)

