using Random
using POMDPPolicies
using BeliefUpdaters
using POMDPSimulators
using BasicPOMCP

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
# up = NothingUpdater()
dsaup = DSABeliefUpdater(P)

# for (s, a, r, sp) in stepthrough(P, policy, up, b0, s0, "s,a,r,sp", max_steps=10)
#    @show s, a, r, sp
# end

s = deepcopy(s0)
b = deepcopy(b0)
b1 = initialize_belief(dsaup)

# a = action(policy, b)
# (s, o, r) = gen(P, s, a, rng)
# b = update(up, b, a, o)
# b1_p = update(dsaup, b1, a, o)

function simulate(P::DSAPOMDP, up::DSABeliefUpdater, s0::State, b0::Belief, policy, rng::AbstractRNG, max_steps=25)
    r_total = 0.
    r_disc = 0.

    b = deepcopy(b0)
    s = deepcopy(s0)
    t = 0
    d = 1.

    while (!isterminal(P, s) && t < max_steps)
        t += 1
        a = action(policy, b)
        (s, o, r) = gen(P, s, a, rng)
        b = update(up, b, a, o)
        r_total += r
        r_disc += r*d
        d *= discount(P)

        @show(t=t, s=s, a=a, r=r, r_tot = r_total, r_disc = r_disc, o=o)
    end
end

# simulate(P, dsaup, s0, b1, policy, rng)

solver = POMCPSolver(c=10.0)
planner = solve(solver, P);
simulate(P, dsaup, s0, b1, planner, rng)