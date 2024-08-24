using Test
using POMDPs
using POMDPTools
using Random
using Distributions
rng = MersenneTwister(123)

include("../src/pomdp.jl")
include("../src/functions.jl")

@testset "DSAPOMDP Tests" begin
    # Initialize the POMDP
    pomdp = DSAPOMDP()

    @testset "State Space" begin
        state_space = states(pomdp)
        @test length(state_space) == 2 * 2 * 2 * (pomdp.max_duration + 1)
        @test State(false, false, false, 0) in state_space
        @test State(true, true, true, pomdp.max_duration) in state_space
        @test !(State(false, false, false, pomdp.max_duration + 1) in state_space)  # Edge case
    end

    @testset "Action Space" begin
        action_space = actions(pomdp)
        @test length(action_space) == 7
        @test action_space == [WAIT, HOSP, DSA, COIL, EMBO, REVC, DISC]
    end

    @testset "Observation Space" begin
        obs_space = observations(pomdp)
        @test length(obs_space) == 6 + 8  # 6 WHObs + 8 DSAObs
        @test any(obs isa WHObs for obs in obs_space)
        @test any(obs isa DSAObs for obs in obs_space)
    end

    @testset "Transition Function" begin
        # Test normal transition
        s = State(false, false, false, 0)
        sp = rand(rng, transition(pomdp, s, WAIT))
        @test sp[4] == 1

        # Test transition to null_state
        s_terminal = State(true, true, true, pomdp.max_duration)
        sp_terminal = rand(rng, transition(pomdp, s_terminal, WAIT))
        @test sp_terminal == pomdp.null_state

        s_discharge = State(false, false, false, 5)
        sp_discharge = rand(rng, transition(pomdp, s_discharge, DISC))
        @test sp_discharge == pomdp.discharge_state

        # Edge case: transition from max_duration to null_state
        s_max = State(false, false, false, pomdp.max_duration)
        sp_max = rand(rng, transition(pomdp, s_max, WAIT))
        @test sp_max == pomdp.null_state

        # Edge case: transition with DSA action
        s_dsa = State(true, false, false, 3)
        sp_dsa = rand(rng, transition(pomdp, s_dsa, DSA))
        @test sp_dsa[4] == 4

        # Test terminal state coverage
        s_terminal_action = State(false, false, false, -99)
        # Ensure that terminal function is invoked, assuming isterminal function correctly identifies terminal states.
        is_terminal = isterminal(pomdp, s_terminal_action)
        if is_terminal
            sp_terminal_action = rand(rng, transition(pomdp, s_terminal_action, WAIT))
            @test Deterministic(sp_terminal_action) == Deterministic(s_terminal_action)
        end
    end

    @testset "Reward Function" begin
        s = State(true, false, false, 5)
        sp = State(true, false, false, 6)

        # Non-terminal state tests
        @test reward(pomdp, s, COIL, sp) > reward(pomdp, s, EMBO, sp)
        @test reward(pomdp, s, DISC, sp) < 0  # Penalty for discharging with condition

        s_healthy = State(false, false, false, 5)
        @test reward(pomdp, s_healthy, DISC, pomdp.discharge_state) > 0  # Reward for correct discharge

        @test reward(pomdp, s, WAIT, sp) < reward(pomdp, s, COIL, sp)  # WAIT should be less rewarding than treatment
        @test reward(pomdp, State(true, true, true, 5), DISC, pomdp.discharge_state) < 0  # Severe penalty for discharging critical patient
        @test reward(pomdp, State(false, false, false, pomdp.max_duration), WAIT, pomdp.null_state) < 0  # Penalty for reaching max duration

        # Test for terminal state
        terminal_state = State(true, true, true, -1)  # This is a terminal state
        @test reward(pomdp, terminal_state, COIL, terminal_state) == 0
        @test reward(pomdp, terminal_state, DISC, terminal_state) == 0
        @test reward(pomdp, terminal_state, WAIT, terminal_state) == 0
        @test reward(pomdp, terminal_state, EMBO, terminal_state) == 0
        @test reward(pomdp, terminal_state, REVC, terminal_state) == 0
        @test reward(pomdp, terminal_state, DSA, terminal_state) == 0
        @test reward(pomdp, terminal_state, HOSP, terminal_state) == 0
    end

    @testset "Observation Model" begin
        s = State(true, false, false, 5)

        obs_wait = rand(rng, observation(pomdp, WAIT, s))
        @test typeof(obs_wait) == Vector{Int64}

        obs_dsa = rand(rng, observation(pomdp, DSA, s))
        @test typeof(obs_dsa) == Vector{Bool}

        # Test probabilities
        obs_dist_hosp = observation(pomdp, HOSP, s)
        dists = obs_dist_hosp.dists

        # Get probabilities from each distribution
        prob_dists = [dist.p for dist in dists]

        # Compute combined probabilities (Cartesian product of probabilities)
        combined_probs = [prod(prob) for prob in Iterators.product(prob_dists...)]

        @test sum(combined_probs) ≈ 1.0

        # Test edge cases
        s_all_true = State(true, true, true, 5)
        obs_all_true = rand(rng, observation(pomdp, DSA, s_all_true))
        @test obs_all_true[1] == true
        @test obs_all_true[2] == true
        @test obs_all_true[3] == true

        s_all_false = State(false, false, false, 5)
        obs_all_false = rand(rng, observation(pomdp, DSA, s_all_false))
        @test obs_all_false[1] == false
        @test obs_all_false[2] == false
        @test obs_all_false[3] == false
    end

    @testset "Belief Initialization" begin
        up = BootstrapFilter(pomdp, 100)
        initial_belief = initialize_belief(up)
        @test sum(initial_belief.probs) ≈ 1.0
        @test all(s.time == 0 for s in support(initial_belief))
    end

    @testset "POMDP Interface" begin
        @test discount(pomdp) == pomdp.discount
        @test isterminal(pomdp, pomdp.null_state)
        @test isterminal(pomdp, pomdp.discharge_state)
        @test !isterminal(pomdp, State(false, false, false, 0))
    end

    @testset "Gen Function" begin
        s = State(false, false, false, 0)
        a = WAIT

        # Test for general cases
        sp, o, r = gen(pomdp, s, a, rng)
        @test sp isa State
        @test o isa WHObs || o isa DSAObs
        @test r isa Real

        # Test for all actions
        for action in actions(pomdp)
            sp, o, r = gen(pomdp, s, action, rng)
            @test sp isa State
            @test o isa WHObs || o isa DSAObs
            @test r isa Real
        end

        # Test for terminal state
        terminal_state = State(true, true, true, -1)  # Create a terminal state
        sp, o, r = gen(pomdp, terminal_state, WAIT, rng)
        @test sp == terminal_state  # Should return the same state
        @test o isa WHObs || o isa DSAObs
        @test r isa Real

        # Test for state at maximum duration - 1
        near_max_duration_state = State(false, false, false, pomdp.max_duration - 1)
        sp, o, r = gen(pomdp, near_max_duration_state, WAIT, rng)
        @test sp == near_max_duration_state  # Should return the same state
        @test o isa WHObs || o isa DSAObs
        @test r isa Real
    end

    @testset "StateIndex and ActionIndex" begin
        # Test for state indexing
        @test stateindex(pomdp, State(true, true, true, 0)) == 1
        @test stateindex(pomdp, State(false, false, false, pomdp.max_duration)) == length(states(pomdp))

        # Test for action indexing
        @test actionindex(pomdp, WAIT) == 1
        @test actionindex(pomdp, DISC) == length(actions(pomdp))

        # Additional tests for all action types
        @test actionindex(pomdp, HOSP) == 2
        @test actionindex(pomdp, DSA) == 3
        @test actionindex(pomdp, COIL) == 4
        @test actionindex(pomdp, EMBO) == 5
        @test actionindex(pomdp, REVC) == 6
    end

    @testset "POMDPs.obsindex Function" begin
        # Test cases for WHObs
        @test POMDPs.obsindex(pomdp, WHObs(CT_POS, SIRIRAJ_LESSNEG1)) == 1
        @test POMDPs.obsindex(pomdp, WHObs(CT_POS, SIRIRAJ_AROUND0)) == 2
        @test POMDPs.obsindex(pomdp, WHObs(CT_POS, SIRIRAJ_GREATER1)) == 3
        @test POMDPs.obsindex(pomdp, WHObs(CT_NEG, SIRIRAJ_LESSNEG1)) == 4
        @test POMDPs.obsindex(pomdp, WHObs(CT_NEG, SIRIRAJ_AROUND0)) == 5
        @test POMDPs.obsindex(pomdp, WHObs(CT_NEG, SIRIRAJ_GREATER1)) == 6

        # Test cases for DSAObs
        @test POMDPs.obsindex(pomdp, DSAObs(true, true, true)) == 7
        @test POMDPs.obsindex(pomdp, DSAObs(true, true, false)) == 8
        @test POMDPs.obsindex(pomdp, DSAObs(true, false, true)) == 9
        @test POMDPs.obsindex(pomdp, DSAObs(true, false, false)) == 10
        @test POMDPs.obsindex(pomdp, DSAObs(false, true, true)) == 11
        @test POMDPs.obsindex(pomdp, DSAObs(false, true, false)) == 12
        @test POMDPs.obsindex(pomdp, DSAObs(false, false, true)) == 13
        @test POMDPs.obsindex(pomdp, DSAObs(false, false, false)) == 14
    end

    @testset "Initial State Function" begin
        # Get the deterministic initial state
        initial_state_dist = initialstate(pomdp)
        initial_state = rand(MersenneTwister(1), initial_state_dist)  # Sample from the deterministic distribution
    
        # Define the expected initial state
        expected_state = State(ane=true, avm=false, occ=false, time=0)
    
        # Test that the sampled initial state matches the expected state
        @test initial_state == expected_state
    
        # Test that the returned distribution is deterministic
        @test typeof(initial_state_dist) == Deterministic{State}
    end
    
end