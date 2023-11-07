using POMDPs
using POMDPModelTools
using Parameters
using Random
using Distributions
using BeliefUpdaters
using StatsBase

export State, Action, Observation, Belief, DSAPOMDP, DSABeliefUpdater

@with_kw mutable struct State
    ane::Bool = false
    avm::Bool = false
    occ::Bool = false
    time::Int64 = 0
end

@enum Action TREAT OBSERVE WAIT
@enum NIHSS NIHSS_A NIHSS_B

@with_kw mutable struct Observation
    hp::Bool
    nihss::NIHSS
end

Observation(hp::Int64, nihss::Int64) = Observation(hp, NIHSS(nihss))


@with_kw mutable struct Belief 
    belief::Dict{State, Float64} = Dict{State, Float64}()
end


@with_kw mutable struct DSABelief     
    #belief is a distribution over states, we will represent it as a support based on the state space and a probability for each state
    support::Vector{State} = []
    probs::Vector{Float64} = []        
    t::Int64 = 0
end

@with_kw mutable struct DSAPOMDP <: POMDP{State, Action, Observation}

    p_ohp_ane_avm_occ::Float64 = 0.9
    p_ohp_ane_avm_notocc::Float64 = 0.8
    p_ohp_ane_notavm_occ::Float64 = 0.8
    p_ohp_ane_notavm_notocc::Float64 = 0.7
    p_ohp_notane_avm_occ::Float64 = 0.8
    p_ohp_notane_avm_notocc::Float64 = 0.7
    p_ohp_notane_notavm_occ::Float64 = 0.7
    p_ohp_notane_notavm_notocc::Float64 = 0.5


    p_onihssA_ane_avm_occ::Float64 = 0.95
    p_onihssA_ane_avm_notocc::Float64 = 0.8
    p_onihssA_ane_notavm_occ::Float64 = 0.8
    p_onihssA_ane_notavm_notocc::Float64 = 0.8
    p_onihssA_notane_avm_occ::Float64 = 0.7
    p_onihssA_notane_avm_notocc::Float64 = 0.7
    p_onihssA_notane_notavm_occ::Float64 = 0.7
    p_onihssA_notane_notavm_notocc::Float64 = 0.3


    p_avm::Float64 = 0.02
    p_ane::Float64 = 0.05
    p_occ::Float64 = 0.1
    max_duration::Int64 = 12
    discount::Float64 = 0.95
    null_state::State = State(
        ane = true,
        avm = true,
        occ = true,
        time = -1
    )
end


function POMDPs.states(P::DSAPOMDP)
    return [State(ane, avm, occ, time)
            for ane in [true, false] for avm in [true, false] for occ in [true, false] for time in 0:P.max_duration]
end


function POMDPs.actions(P::DSAPOMDP)
    return [TREAT, OBSERVE, WAIT]
end


function POMDPs.observations(P::DSAPOMDP)
    return [Observation(hypertension, nihss)
            for hypertension in [true, false] for nihss in [NIHSS_A, NIHSS_B]]
end

function POMDPs.transition(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s) || s.time >= (P.max_duration - 1) || (s.ane && s.avm && s.occ)
        return Deterministic(P.null_state)
    end

    if a == TREAT
        ane_dist = DiscreteNonParametric([true, false], [0., 1.])
        avm_dist = DiscreteNonParametric([true, false], [0., 1.])
        occ_dist = DiscreteNonParametric([true, false], [0., 1.])
    else
        p_ane = s.ane ? 1. : P.p_ane
        p_avm = s.avm ? 1. : P.p_avm
        p_occ = s.occ ? 1. : P.p_occ

        ane_dist = DiscreteNonParametric([true, false], [p_ane, 1-p_ane])
        avm_dist = DiscreteNonParametric([true, false], [p_avm, 1-p_avm])
        occ_dist = DiscreteNonParametric([true, false], [p_occ, 1-p_occ])
    end

    support = collect(0:1:P.max_duration)
    probs = fill(0., P.max_duration + 1)
    probs[s.time + 2] = 1.
    time_dist = DiscreteNonParametric(support, probs)

    dist = product_distribution(ane_dist, avm_dist, occ_dist, time_dist)

    return dist
end

function POMDPs.reward(P::DSAPOMDP, s::State, a::Action, sp::State)
    #:TODO update r to accumulate and separate if TREAT conditions
    # Be creative

    r = 0

    if !isterminal(P, s) && isterminal(P, sp)
        r += -100000 # Huge penalty for first time entering terminal state
    elseif isterminal(P, s)
        return 0
    end

    if a == TREAT
        r += -100 #costly procedure
    end

    if a == TREAT && !s.ane && !s.avm && !s.occ
        r += -10000 #we get sued
    elseif a == TREAT && (s.ane || s.avm || s.occ)
        r += 50000 #we heal the patient
    elseif a == OBSERVE && !s.ane && !s.avm && !s.occ
        r += -100 #oppurtunity cost
    elseif a == OBSERVE && (s.ane || s.avm || s.occ)
        r += -1000
    elseif a == WAIT && !s.ane && !s.avm && !s.occ
        r += 500
    elseif a == WAIT && (s.ane || s.avm || s.occ)
        r += -50 #we lose a patient
    end
    
    return r
end


# Observation Model: P(O|S) ---> P(S|O) = P(O|S)P(S)/P(O)
function POMDPs.observation(P::DSAPOMDP, sp::State)
    state_status = ""*string(sp.ane ? "ane" : "notane") * "_" * string(sp.avm ? "avm" : "notavm") * "_" * string(sp.occ ? "occ" : "notocc")
    hp_prob = getfield(P, Symbol("p_ohp_"*state_status))
    dist_hp = DiscreteNonParametric([true, false], [hp_prob, 1-hp_prob])
    
    nihss_prob = getfield(P, Symbol("p_onihssA_"*state_status))
    dist_nihss = DiscreteNonParametric([Int(nihss) for nihss in [NIHSS_A, NIHSS_B]], [nihss_prob, 1-nihss_prob])

    return product_distribution(dist_hp, dist_nihss)
end

function POMDPs.discount(P::DSAPOMDP)
    return P.discount
end

struct DSABeliefUpdater <: Updater
    P::DSAPOMDP
end

function POMDPs.initialize_belief(up::DSABeliefUpdater)
    #initialize belief as a uniform distribution over the state space for when t=0 and 0 for all other times
    all_support = [states(up.P)...]
    support = [state for state in all_support if state.time == 0]
    probs = fill(1/length(support), length(support))    

    return DSABelief(support, probs, 0)
end

# function POMDPs.update(up::DSABeliefUpdater, b::Belief, a::Action, o::Observation)
#     new_belief_dict = Dict{State, Float64}()
#     for (state, belief_prob) in b.belief
#         updated_belief_prob = rand()
        
#         new_belief_dict[state] = updated_belief_prob * belief_prob
#     end
#     total_probability = sum(values(new_belief_dict))
#     for (state, belief_prob) in new_belief_dict
#         new_belief_dict[state] = belief_prob / total_probability
#     end
#     return Belief(new_belief_dict)
# end

# function time_update(s::State)
#     #update the time of the state
#     return State(ane=s.ane, avm=s.avm, occ=s.occ, time=s.time+1)
# end

# function POMDPs.update(up::DSABeliefUpdater, b::DSABelief, a::Action, o::Observation)
#     #update DSABelief
#     #recompute the probability of each state in the support
#     new_support = time_update.(b.support)
#     old_probs = b.probs # prior

#     #update new_probs using bayes rule on the observation
#     new_probs = fill(0., length(new_support))


    
# end

function POMDPs.initialstate(P::DSAPOMDP)
    s0 = State(ane=false, avm=false, occ=false, time=0)

    return Deterministic(s0)
end


function POMDPs.isterminal(P::DSAPOMDP, s::State)
    return s == P.null_state
end


function POMDPs.gen(P::DSAPOMDP, s::State, a::Action, rng::AbstractRNG)
    if isterminal(P, s) || s.time >= (P.max_duration - 1) || (s.ane && s.avm && s.occ)
        sp = s
    else
        next_state = rand(rng, transition(P, s, a))
        sp = State(ane=next_state[1], avm=next_state[2], occ=next_state[3], time=next_state[4])
    end
    obs = rand(rng, observation(P, sp))
    obs = Observation(
        hp = obs[1],
        nihss = NIHSS(obs[2])
    )
    rew = reward(P, s, a, sp)
    return (sp = sp, o = obs, r = rew)
end

function Base.rand(rng::AbstractRNG, b::Belief)
    return rand(rng, b.belief)[1]
end


function Base.rand(rng::AbstractRNG, b::DSABelief)
    return sample(rng, b.support, Weights(b.probs))
end

function support(b::DSABelief)
    return b.support
end

function probs(b::DSABelief)
    return b.probs
end

# function simulate(P::DSAPOMDP, up::Union{DSABeliefUpdater, NothingUpdater}, s0::State, b0::Union{Belief, DSABelief}, policy, rng::AbstractRNG; max_steps=24, verbose=false)
#     r_total = 0.
#     r_disc = 0.

#     b = deepcopy(b0)
#     s = deepcopy(s0)
#     t = 0
#     d = 1.

#     while (!isterminal(P, s) && t < max_steps)
#         t += 1
#         a = action(policy, b)
#         (s, o, r) = gen(P, s, a, rng)
#         b = update(up, b, a, o)
#         r_total += r
#         r_disc += r*d
#         d *= discount(P)
#         if verbose
#             @show(t=t, s=s, a=a, r=r, r_tot = r_total, r_disc = r_disc, o=o)
#         end
#     end

#     return (r_total = r_total, r_disc = r_disc)
# end


function evaluate(;P, up, b0, policies, rng, num_reps, num_steps)
    results = Array{Float64}(undef, length(policies), num_runs)

    for i in 1:num_reps
        s0 = rand(b0)
        for (j, policy) in enumerate(policies)
            (r_tot, r_disc) = simulate(P, up, s0, b0, policy, rng, max_steps=12)
            results[j, i] = r_tot
        end
    end
    #compute mean and std
    means = mean(results, dims=2)
    stds = std(results, dims=2)
    return (means=means, stds=stds)    
end

# function get_obs_likelihood(P::DSAPOMDP, o::Observation)
    #get the likelihood of the state given the obs P_o_given_state
