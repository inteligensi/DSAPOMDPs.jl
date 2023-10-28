using POMDPs
using POMDPModelTools
using Parameters
using Random
using Distributions

export State, Action, Observation, Belief, DSAPOMDP, DSABeliefUpdater

@with_kw mutable struct State
    ane::Bool = false
    avm::Bool = false
    occ::Bool = false
    time::Int64 = 0
    hypertension::Bool = false
end

@enum Action TREAT OBSERVE WAIT

@with_kw mutable struct Observation
    is_ane::Bool
    is_avm::Bool
    is_occ::Bool
end

@with_kw mutable struct Belief
    belief::Dict{State, Float64} = Dict{State, Float64}()
end

@with_kw mutable struct DSAPOMDP <: POMDP{State, Action, Observation}
    p_avm::Float64 = 0.02
    p_ane::Float64 = 0.05
    p_occ::Float64 = 0.1
    max_duration::Int64 = 24
    discount::Float64 = 0.95
    null_state::State = State(
        ane = true,
        avm = true,
        occ = true,
        time = -1,
        hypertension = true
    )
end

function POMDPs.states(P::DSAPOMDP)
    return [State(ane, avm, occ, time, hypertension)
            for ane in [true, false],
                avm in [true, false],
                occ in [true, false],
                time in 0:P.max_duration,
                hypertension in [true, false]]
end

function POMDPs.actions(P::DSAPOMDP)
    return [TREAT, OBSERVE, WAIT]
end

function POMDPs.observations(P::DSAPOMDP)
    return [Observation(is_ane, is_avm, is_occ)
            for is_ane in [true, false],
                is_avm in [true, false],
                is_occ in [true, false]]
end

function POMDPs.transition(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s) || s.time >= (P.max_duration - 1) || (s.ane && s.avm && s.occ)
        return P.null_state
    end

    p_ane = s.ane ? 1. : (a == TREAT ? 0. : P.p_ane)
    p_avm = s.avm ? 1. : (a == TREAT ? 0. : P.p_avm)
    p_occ = s.occ ? 1. : (a == TREAT ? 0. : P.p_occ)

    ane_dist = DiscreteNonParametric([true, false], [p_ane, 1 - p_ane])
    avm_dist = DiscreteNonParametric([true, false], [p_avm, 1 - p_avm])
    occ_dist = DiscreteNonParametric([true, false], [p_occ, 1 - p_occ])

    support = collect(0:1:P.max_duration)
    probs = fill(0., P.max_duration + 1)
    probs[s.time + 2] = 1.
    time_dist = DiscreteNonParametric(support, probs)

    hypertension_dist = DiscreteNonParametric([s.hypertension], [1.])

    dist = product_distribution(ane_dist, avm_dist, occ_dist, time_dist, hypertension_dist)
    # sp = rand(rng, dist)

    # return State(
    #     ane = sp[1],
    #     avm = sp[2],
    #     occ = sp[3],
    #     time = sp[4],
    #     hypertension = sp[5]
    # )
    return dist
end

function POMDPs.reward(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s)
        return 0
    end
    if a == TREAT
        return -200
    elseif a == OBSERVE && (!s.ane || !s.avm || !s.occ)
        return -100
    elseif a == WAIT && (s.ane || s.avm || s.occ)
        return -50
    else
        return 0
    end
end

function POMDPs.observation(P::DSAPOMDP, sp::State)
    is_ane = sp.ane ? DiscreteNonParametric([true, false], [0.7, 0.3]) : DiscreteNonParametric([true, false], [0.4, 0.6])
    is_avm = sp.avm ? DiscreteNonParametric([true, false], [0.8, 0.2]) : DiscreteNonParametric([true, false], [0.3, 0.7])
    is_occ = sp.occ ? DiscreteNonParametric([true, false], [0.9, 0.1]) : DiscreteNonParametric([true, false], [0.2, 0.8])

    return product_distribution(is_ane, is_avm, is_occ)#Observation(rand(is_ane), rand(is_avm), rand(is_occ))
end

function POMDPs.discount(P::DSAPOMDP)
    return P.discount
end

struct DSABeliefUpdater <: Updater
    P::DSAPOMDP
end

function POMDPs.initialize_belief(up::DSABeliefUpdater)
    belief_dict = Dict{State, Float64}()
    total_states = 2^5 * (up.P.max_duration + 1)
    for ane in [true, false], avm in [true, false], occ in [true, false], time in 0:up.P.max_duration, hypertension in [true, false]
        state = State(ane=ane, avm=avm, occ=occ, time=time, hypertension=hypertension)
        belief_dict[state] = 1/total_states
    end
    return Belief(belief_dict)
end

function POMDPs.update(up::DSABeliefUpdater, b::Belief, a::Action, o::Observation)
    new_belief_dict = Dict{State, Float64}()
    for (state, belief_prob) in b.belief
        updated_belief_prob = rand()
        
        new_belief_dict[state] = updated_belief_prob * belief_prob
    end
    total_probability = sum(values(new_belief_dict))
    for (state, belief_prob) in new_belief_dict
        new_belief_dict[state] = belief_prob / total_probability
    end
    return Belief(new_belief_dict)
end

function POMDPs.initialstate(P::DSAPOMDP)
    s0 = State(ane=false, avm=false, occ=false, time=0, hypertension=false)

    return Deterministic(s0)
end

function POMDPs.isterminal(P::DSAPOMDP, s::State)
    return s == P.null_state
end

# function POMDPs.action(P::DSAPOMDP, x:Any)

function POMDPs.gen(P::DSAPOMDP, s::State, a::Action, rng::AbstractRNG)
    next_state = rand(rng, transition(P, s, a))
    sp = State(ane=next_state[1], avm=next_state[2], occ=next_state[3], time=next_state[4], hypertension=next_state[5])
    obs = rand(rng, observation(P, s))
    observ= Observation(is_ane=obs[1], is_avm=obs[2], is_occ=obs[3])
    rew = reward(P, s, a)
    return (sp = sp, o = observ, r = rew)
end
