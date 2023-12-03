using BasicPOMCP
using D3Trees
using MCTS
using ARDESPOT
using PointBasedValueIteration


include("main.jl")

π_pomcp = solve(POMCPSolver(), P)
# h_pomcp = simulate(hr, P, π_pomcp, up, b0)

π_despot = solve(DESPOTSolver(), P)
# h_despot = simulate(hr, P, π_despot, up, b0)

mdp = GenerativeBeliefMDP(P, up)
π_mcts = solve(MCTSSolver(), mdp)
# h_mcts = simulate(hr, mdp, π_mcts)

# r_pomcp = compute_rdisc(P, h_pomcp);
# r_despot = compute_rdisc(P, h_despot);
# r_mcts = compute_rdisc(P, h_mcts);

policies = [π_pomcp, π_despot, π_mcts, π_random]
means, stds, t_means, t_stds = evaluate_policies_replication(hr, P, policies, up, b0, n_reps, max_steps)
# evaluate_policies(hr, P, policies, up, b0, max_steps)

# println("Test running an offline solver")
# π_pbvi = solve(PBVISolver(), P)
# pbvi_history = simulate(hr, P, π_pbvi, up, b0)

file_path = "output-rdisc-time-rep.txt"
file = open(file_path, "w")
println(file, "means = $(means)")
println(file, "stds = $(stds)")
println(file, "t_means = $(t_means)")
println(file, "t_stds = $(t_stds)")