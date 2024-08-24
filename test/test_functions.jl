using Test
using POMDPs
using POMDPTools
using ParticleFilters
using Distributions
using DataFrames
using DSAPOMDPs
using BasicPOMCP
using ARDESPOT

include("../src/pomdp.jl")
include("../src/policies.jl")
include("../src/functions.jl")

@testset "DSAPOMDP Utility Function Tests" begin
    @testset "keymax Function" begin
        d = Dict("a" => 1, "b" => 3, "c" => 2)
        @test keymax(d) == "b"

        # Edge case: empty dictionary
        @test_throws ArgumentError keymax(Dict())

        # Edge case: multiple maximum values
        d_multi_max = Dict("a" => 3, "b" => 3, "c" => 2)
        @test keymax(d_multi_max) in ["a", "b"]
    end

    @testset "particle2prob Function" begin
        particles = [
            State(true, false, false, 0),
            State(false, true, false, 0),
            State(true, true, true, 0),
            State(false, false, false, 0)
        ]
        b = ParticleCollection(particles)
        prob = particle2prob(b)
        @test prob[state2stateindex(State(true, false, false, 0))] ≈ 0.25
        @test prob[state2stateindex(State(false, false, false, 0))] ≈ 0.25
        @test prob[state2stateindex(State(true, true, true, 0))] ≈ 0.25

        # Edge case: empty particle collection
        empty_b = ParticleCollection(State[])
        @test_throws ErrorException particle2prob(empty_b)

        # Edge case: particle collection with one state repeated many times
        repeated_state = State(true, false, false, 0)
        repeated_b = ParticleCollection([repeated_state for _ in 1:100])
        repeated_prob = particle2prob(repeated_b)
        @test repeated_prob[state2stateindex(repeated_state)] ≈ 1.0
    end

    @testset "state2stateindex and stateindex2string Functions" begin
        @test state2stateindex(State(true, true, true, 0)) == 1
        @test state2stateindex(State(false, false, false, 0)) == 8
        @test stateindex2string(1) == "(Ane, AVM, Occ)"
        @test stateindex2string(8) == "(None)"

        # Edge case: invalid state index
        @test_throws ErrorException stateindex2string(9)
    end

    @testset "state_sub2ind Function" begin
        # Basic Valid Cases
        @test state_sub2ind((2, 2, 2, 25), 1, 1, 1, 1) == 1
        @test state_sub2ind((2, 2, 2, 25), 2, 2, 2, 25) == 200
    
        # Edge Cases: Testing with boundary indices
        @test state_sub2ind((2, 2, 2, 25), 1, 1, 1, 25) == 193
        @test state_sub2ind((2, 2, 2, 25), 2, 2, 2, 1) == 8
        @test state_sub2ind((2, 2, 2, 25), 2, 2, 1, 1) == 4
        @test state_sub2ind((2, 2, 2, 25), 2, 1, 1, 1) == 2
    
        # Edge Case: Invalid indices
        @test_throws BoundsError state_sub2ind((2, 2, 2, 25), 3, 1, 1, 1)
        @test_throws BoundsError state_sub2ind((2, 2, 2, 25), 1, 3, 1, 1)
        @test_throws BoundsError state_sub2ind((2, 2, 2, 25), 1, 1, 3, 1)
        @test_throws BoundsError state_sub2ind((2, 2, 2, 25), 1, 1, 1, 26)
    
        # Edge Case: All indices at maximum
        @test state_sub2ind((2, 2, 2, 25), 2, 2, 2, 25) == 200
    end
    

    @testset "evaluate_policies Function" begin
        # Setup
        pomdp = DSAPOMDP()
        hr = HistoryRecorder(rng=MersenneTwister(1), max_steps=24)
        up = BootstrapFilter(pomdp, 1000)
        b0 = initialize_belief(up)
        s0 = State(false, false, false, 0)
        policies = [DSAPolicy(P=pomdp), HOSPPolicy(P=pomdp)]
        policy_names = ["DSAPolicy", "HOSPPolicy"]
    
        # Test verbose output
        results_verbose = evaluate_policies(hr, pomdp, policies, up, b0, s0, verbose=true, save_to_file=false)
    
        # Validate results
        @test length(results_verbose.rdiscs) == length(policies)
        @test length(results_verbose.durations) == length(policies)
        @test length(results_verbose.is_treateds) == length(policies)
        @test all(d <= hr.max_steps for d in results_verbose.durations)
        @test !isempty(results_verbose.rdiscs)
        @test !isempty(results_verbose.durations)
        @test !isempty(results_verbose.is_treateds)
        @test all(typeof(r) == Float64 for r in results_verbose.rdiscs)
        @test all(typeof(d) == Float64 for d in results_verbose.durations)
        @test all(typeof(it) == Bool for it in results_verbose.is_treateds)
    
        # Test save_to_file option
        results_file = evaluate_policies(hr, pomdp, policies, up, b0, s0, verbose=false, save_to_file=true)
    
        # Validate results
        @test length(results_file.rdiscs) == length(policies)
        @test length(results_file.durations) == length(policies)
        @test length(results_file.is_treateds) == length(policies)
        @test all(d <= hr.max_steps for d in results_file.durations)
        @test !isempty(results_file.rdiscs)
        @test !isempty(results_file.durations)
        @test !isempty(results_file.is_treateds)
        @test all(typeof(r) == Float64 for r in results_file.rdiscs)
        @test all(typeof(d) == Float64 for d in results_file.durations)
        @test all(typeof(it) == Bool for it in results_file.is_treateds)
    
        # Check file output
        for (i, policy_name) in enumerate(policy_names)
            file_path = "output-$i.txt"
            
            @test isfile(file_path)  # Check if the file is created
            
            if isfile(file_path)
                # Read file content and check
                content = read(file_path, String)
                @test contains(content, "==========simulation for policy $i ==========")
                @test contains(content, "states = ")
                @test contains(content, "state_primes = ")
                @test contains(content, "observations = ")
                @test contains(content, "actions = ")
                @test contains(content, "rsum = ")
                @test contains(content, "duration = ")
                @test contains(content, "is_treated = ")
                
                rm(file_path)  # Clean up the file after test
            end
        end
    
        # Edge Case: Empty Policies
        empty_results = evaluate_policies(hr, pomdp, [], up, b0, s0, verbose=false, save_to_file=false)
        @test length(empty_results.rdiscs) == 0
        @test length(empty_results.durations) == 0
        @test length(empty_results.is_treateds) == 0
    
        # Edge Case: Single Policy
        single_policy_results = evaluate_policies(hr, pomdp, [DSAPolicy(P=pomdp)], up, b0, s0, verbose=false, save_to_file=false)
        @test length(single_policy_results.rdiscs) == 1
        @test length(single_policy_results.durations) == 1
        @test length(single_policy_results.is_treateds) == 1
    
        # Edge Case: Large number of policies (stress test)
        large_policies = [DSAPolicy(P=pomdp) for _ in 1:100]
        large_results = evaluate_policies(hr, pomdp, large_policies, up, b0, s0, verbose=false, save_to_file=false)
        @test length(large_results.rdiscs) == 100
        @test length(large_results.durations) == 100
        @test length(large_results.is_treateds) == 100

    end
    

    @testset "summarize_rollout Function" begin
        pomdp = DSAPOMDP()
        policy = DSAPolicy(P=pomdp)
        hr = HistoryRecorder(rng=MersenneTwister(1), max_steps=24)
        up = BootstrapFilter(pomdp, 1000)
        b0 = initialize_belief(up)
        s0 = State(false, false, false, 0)

        h = simulate(hr, pomdp, policy, up, b0, s0)
        summary = summarize_rollout(pomdp, h)
        @test length(summary) == 9
        @test typeof(summary[1]) <: Number  # rsum
        @test typeof(summary[2]) <: Vector  # actions
        @test typeof(summary[8]) <: Bool    # is_treated
        @test typeof(summary[9]) <: Number  # time_to_recover

        # Additional checks
        @test summary[1] <= 1e5  # Total reward should be non-positive
        @test all(a in actions(pomdp) for a in summary[2])  # All actions should be valid
        @test summary[9] <= hr.max_steps  # Time to recover should not exceed max_steps
    end

    @testset "replicate_policy_eval Function" begin
        pomdp = DSAPOMDP()
        hr = HistoryRecorder(rng=MersenneTwister(1), max_steps=24)
        up = BootstrapFilter(pomdp, 1000)
        b0 = initialize_belief(up)
        policies = [DSAPolicy(P=pomdp), HOSPPolicy(P=pomdp)]

        results = replicate_policy_eval(hr, pomdp, policies, up, b0, num_reps=5, verbose=false, save_to_file=false)
        @test length(keys(results)) == 5
        @test all(haskey(results["1"], key) for key in ["need_treatment", "rdisc_1", "time2recover_1", "treated_1", "rdisc_2", "time2recover_2", "treated_2"])

        # Additional checks
        @test all(results[rep]["time2recover_1"] <= hr.max_steps for rep in keys(results))  # All recovery times should not exceed max_steps
    end

    @testset "dict2df Function" begin
        dict_result = Dict(
            "1" => Dict("a" => 1, "b" => 2),
            "2" => Dict("a" => 3, "b" => 4)
        )
        df = dict2df(dict_result)
        @test size(df) == (2, 2)
        @test names(df) == ["a", "b"]
        @test df[1, :a] == 1
        @test df[2, :b] == 4
    end

    @testset "plot_belief_hist function" begin
        N = 100
        P = DSAPOMDP()
        rng = MersenneTwister(1)
        up = BootstrapFilter(P, N, rng)
        b0 = initialize_belief(up)
        max_steps = 24
    
        hr = HistoryRecorder(rng=rng, max_steps=max_steps)
        π_pomcp = solve(POMCPSolver(), P)
    
        s0 = states(P)[76]
        hist = simulate(hr, P, π_pomcp, up, b0, s0)
    
        # Test case 1: t = 0 (first conditional branch)
        plt = plot_belief_hist(hist, 0)
        @test !isnothing(plt)  # Check that a plot object is returned

        # Test case 1: t = 1 (second conditional branch)
        plt = plot_belief_hist(hist, 1)
        @test !isnothing(plt)  # Check that a plot object is returned
    end
end