using Test
using POMDPs
using POMDPTools
using Random
using Distributions
using ARDESPOT

include("../src/pomdp.jl")
include("../src/functions.jl")
include("../src/policies.jl")

@testset "DSAPOMDP Policy Tests" begin
    pomdp = DSAPOMDP()

    @testset "calc_condition_rate Function" begin
        # Create a mock particle collection
        particles = [
            State(true, false, false, 0),
            State(false, true, false, 0),
            State(true, true, true, 0),
            State(false, false, false, 0)
        ]
        b = ParticleCollection(particles)

        rates = calc_condition_rate(b)
        @test rates["ane"] ≈ 0.5
        @test rates["avm"] ≈ 0.5
        @test rates["occ"] ≈ 0.25
    end

    @testset "base_policy Function" begin
        # Test discharge scenario
        b_discharge = ParticleCollection([State(false, false, false, 0) for _ in 1:100])
        @test base_policy(pomdp, b_discharge) == DISC

        # Test dominant condition scenario
        b_ane_dominant = ParticleCollection([State(true, false, false, 0) for _ in 1:70] ∪ [State(false, false, false, 0) for _ in 1:30])
        @test base_policy(pomdp, b_ane_dominant) == COIL

        # Test no dominant condition scenario
        b_no_dominant = ParticleCollection([State(true, false, false, 0) for _ in 1:40] ∪ [State(false, true, false, 0) for _ in 1:30] ∪ [State(false, false, true, 0) for _ in 1:30])
        @test base_policy(pomdp, b_no_dominant) == HOSP
    end

    @testset "DSAPolicy" begin
        policy = DSAPolicy(P=pomdp)
        b_no_dominant = ParticleCollection([State(true, false, false, 0) for _ in 1:40] ∪ [State(false, true, false, 0) for _ in 1:30] ∪ [State(false, false, true, 0) for _ in 1:30])
        @test action(policy, b_no_dominant) == DSA
    end

    @testset "HOSPPolicy" begin
        policy = HOSPPolicy(P=pomdp)
        b_no_dominant = ParticleCollection([State(true, false, false, 0) for _ in 1:40] ∪ [State(false, true, false, 0) for _ in 1:30] ∪ [State(false, false, true, 0) for _ in 1:30])
        @test action(policy, b_no_dominant) == HOSP
    end

    @testset "Policy Edge Cases" begin
        policy = DSAPolicy(P=pomdp)

        # Test with belief near decision boundary
        b_boundary = ParticleCollection([State(true, false, false, 0) for _ in 1:59] ∪ [State(false, false, false, 0) for _ in 1:41])
        @test action(policy, b_boundary) == DSA
    end

    @testset "Different Combinations of Condition Rates" begin
        policy = DSAPolicy(P=pomdp)

        # Test with high ane, low avm and occ
        b_high_ane = ParticleCollection([State(true, false, false, 0) for _ in 1:80] ∪ [State(false, true, false, 0) for _ in 1:10] ∪ [State(false, false, true, 0) for _ in 1:10])
        @test action(policy, b_high_ane) == COIL

        # Test with high avm, low ane and occ
        b_high_avm = ParticleCollection([State(false, true, false, 0) for _ in 1:80] ∪ [State(true, false, false, 0) for _ in 1:10] ∪ [State(false, false, true, 0) for _ in 1:10])
        @test action(policy, b_high_avm) == EMBO

        # Test with equal distribution of conditions
        b_equal = ParticleCollection([State(true, false, false, 0) for _ in 1:33] ∪ [State(false, true, false, 0) for _ in 1:33] ∪ [State(false, false, true, 0) for _ in 1:34])
        @test action(policy, b_equal) == DSA
    end
end