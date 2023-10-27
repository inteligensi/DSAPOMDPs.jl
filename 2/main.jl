using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators

include("DSAPOMDP.jl")

global P = DSAPOMDP()
global rng = MersenneTwister(1)
global b0 = initialstate(P)
global s0 = rand(b0)

@with_kw struct DSAPolicy <: Policy
    policy::Policy
    p_nothing = 0.5
end

function POMDPs.action(policy::DSAPolicy, b)
    if rand() < policy.p_nothing
        return WAIT
    else
        return action(policy.policy, b)
    end
end

policy = DSAPolicy(policy=RandomPolicy(P), p_nothing=0.1)
up = NothingUpdater()

for (s, a, r, sp) in stepthrough(P, policy, up, b0, s0, "s,a,r,sp", max_steps=10)
   @show s, a, r, sp
end


# a = action(policy, b0)
# gen(P, s0, a, rng)