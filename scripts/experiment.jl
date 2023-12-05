using BasicPOMCP
using ARDESPOT
using DataFrames
using CSV

include("main.jl")

π_pomcp = solve(POMCPSolver(), P)
π_despot = solve(DESPOTSolver(bounds=IndependentBounds(-5000, 5000, check_terminal=true)), P)
π_random = RandomPolicy(P)
π_dsa = DSAPolicy(P)
π_hosp = HOSPPolicy(P)

policies = [π_pomcp, π_despot, π_random, π_dsa, π_hosp]
π_names = ["POMCP", "DESPOT", "Random", "ExpertDSA", "ExpertHOSP"]
s0 = states(P)[76]
rs, ttrs, hcs = evaluate_policies(hr, P, policies, up, b0, s0, pol_names = π_names, save_to_file=true)

# repeat the experiment num_reps times with s0=rand(b0)
# num_reps = 100
# dict_result = replicate_policy_eval(hr, P, policies, up, b0; num_reps=num_reps, pol_names=π_names)
# df = dict2df(dict_result)

# CSV.write("results_$(num_reps).csv", df)

# hist = simulate(hr, P, π_despot, up, b0, s0);
# summarize_rollout(P, hist)

