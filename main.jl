using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators
using BasicPOMCP
using ParticleFilters
using D3Trees
using MCTS
using ARDESPOT
using PointBasedValueIteration
using FIB

include("DSAPOMDP.jl")
include("utils.jl")

P = DSAPOMDP()
rng = MersenneTwister(1)
b0 = initialize_belief(DSABeliefUpdater(P))
s0 = rand(b0)
b0.probs = [0.5, 0.5, 0, 0, 0, 0, 0, 0]

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
    p4 = histogram(
        time_values,
        ylim = (0,1),
        xlim = (0,12),
        nbins = 13,
        title = "Time (t: $i)",
        normalize = :probability, 
        legend = false
    )
    plot(p1, p2, p3, p4, layout = (2, 2))
end

#simulate random policy
# policy = RandomPolicy(P)
# for (s, a, sp, o) in stepthrough(P, policy, NothingUpdater(), s0, max_steps=12, rng=rng)
#     println("s: $s, a: $a, o: $o")
# end

#simulating POMCP policy our DSABeliefUpdater
solver = POMCPSolver()
pomcp_planner = solve(solver, P)
history_pomcp = simulate(hr, P, pomcp_planner, up, b0)


rsum = 0.0
d = 1.0
for step in eachstep(history_pomcp)
    # @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum += (step.r * d)
    global d *= discount(P)
end
@show rsum;

rsum = 0.0
d = 1.0
for step in eachstep(history)
    # @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum += (step.r * d)
    global d *= discount(P)
end
@show rsum;

a, info = action_info(pomcp_planner, b0, tree_in_info=true)

inchrome(D3Tree(info[:tree], init_expand=1))
#(r_tot, r_disc) = simulate(P, up, s0, b0, planner, rng)

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
# ardespot, mcvi, pbvi, pomcp, mcts, random (time, r_disc)
# visualize each step action and belief

policy = RandomPolicy(P)
a = action(policy, b0)
(s, o, r) = gen(P, s0, a, rng)

# SOLVER
P = DSAPOMDP()
rng = MersenneTwister(1)
b0 = initialize_belief(DSABeliefUpdater(P))
b0.probs = [0.5, 0.5, 0, 0, 0, 0, 0, 0]

N = 1000
up = BootstrapFilter(P, N, rng)
hr = HistoryRecorder(rng=rng, max_steps=12)

rsum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Online : MCTS, ARDESPOT, BasicPOMCP
# MCTS
mdp = GenerativeBeliefMDP(P, up)
mcts_solver = MCTSSolver()
mcts_planner = solve(mcts_solver, mdp)
mcts_history = simulate(hr, mdp, mcts_planner)

for step in eachstep(mcts_history)
    @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum[1] += (step.r * d[1])
    global d[1] *= discount(P)
end
@show rsum[1];

# ARDESPOT
despot_solver = DESPOTSolver()
despot_planner = solve(despot_solver, P)
despot_history = simulate(hr, P, despot_planner, up, b0)

for step in eachstep(despot_history)
    @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum[2] += (step.r * d[2])
    global d[2] *= discount(P)
end
@show rsum[2];

# BasicPOMCP
pomcp_solver = POMCPSolver()
pomcp_planner = solve(pomcp_solver, P)
pomcp_history = simulate(hr, P, pomcp_planner, up, b0)

for step in eachstep(pomcp_history)
    @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum[3] += (step.r * d[3])
    global d[3] *= discount(P)
end
@show rsum[3];

# Offline : PBVI, FIB
# PBVI
pbvi_solver = PBVISolver()
pbvi_policy = solve(pbvi_solver, P)
pbvi_history = simulate(hr, P, pbvi_policy)

for step in eachstep(pbvi_history)
    @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum[4] += (step.r * d[4])
    global d[4] *= discount(P)
end
@show rsum[4];

# FIB
fib_solver = FIBSolver()
fib_policy = solve(fib_solver, P)
fib_history = simulate(hr, P, fib_policy)

for step in eachstep(fib_history)
    @show("action: %s, observation: %s\n", step.a, step.o)
    global rsum[5] += (step.r * d[5])
    global d[5] *= discount(P)
end
@show rsum[5];

# Random & Dokter
@show rsum;