using BasicPOMCP
using D3Trees
using MCTS
using ARDESPOT
using PointBasedValueIteration


include("main.jl")

# π_pomcp = solve(POMCPSolver(), P)
# h_pomcp = simulate(hr, P, π_pomcp, up, b0)

# π_despot = solve(DESPOTSolver(), P)
# h_despot = simulate(hr, P, π_despot, up, b0)

# mdp = GenerativeBeliefMDP(P, up)
# π_mcts = solve(MCTSSolver(), mdp)
# h_mcts = simulate(hr, mdp, π_mcts)

# r_pomcp = compute_rdisc(P, h_pomcp);
# r_despot = compute_rdisc(P, h_despot);
# r_mcts = compute_rdisc(P, h_mcts);


println("Test running an offline solver")
π_pbvi = solve(PBVISolver(), P)
pbvi_history = simulate(hr, P, π_pbvi, up, b0)

