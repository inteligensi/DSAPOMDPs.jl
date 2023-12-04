using BasicPOMCP
using ARDESPOT
using DataFrames
using CSV
using Plots

include("main.jl")

π_pomcp = solve(POMCPSolver(), P)
π_despot = solve(DESPOTSolver(bounds=IndependentBounds(-5000, 5000, check_terminal=true)), P)
π_random = RandomPolicy(P)
π_dsa = DSAPolicy(P)
π_hosp = HOSPPolicy(P)

policies = [π_pomcp, π_despot, π_random, π_dsa, π_hosp]
π_names = ["POMCP", "DESPOT", "Random", "ExpertDSA", "ExpertHOSP"]

s0 = states(P)[101]
hist = simulate(hr, P, π_pomcp, up, b0, s0);

# t=5
# plt = plot_belief_hist(hist, t)
# savefig("outputs/despot_belief_hist_$(t).pdf")

@gif for (i, b) in enumerate(belief_hist(hist))
    plt = plot_belief_hist(hist, i-1)
end
