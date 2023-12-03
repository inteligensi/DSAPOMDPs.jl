using BasicPOMCP
using D3Trees
using MCTS
using ARDESPOT


include("main.jl")

π_pomcp = solve(POMCPSolver(), P)
π_despot = solve(DESPOTSolver(), P)
π_random = RandomPolicy(P)

policies = [π_pomcp, π_despot, π_random]
means, stds, t_means, t_stds = evaluate_policies_replication(hr, P, policies, up, b0, n_reps, max_steps)

file_path = "output-rdisc-time-rep.txt"
file = open(file_path, "w")
println(file, "means = $(means)")
println(file, "stds = $(stds)")
println(file, "t_means = $(t_means)")
println(file, "t_stds = $(t_stds)")