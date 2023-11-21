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

@enum Action DSA OBSERVE COIL EMBOLIZATION AVMTREAT
@enum NIHSS NIHSS_A NIHSS_B

@with_kw mutable struct Observation
    nihss::NIHSS
    ct::Bool
    siriraj::Float64
end

@with_kw mutable struct DSAObservation
    p_ane::Bool
    p_avm::Bool
    p_occ::Bool
end

Observation(nihss::Int64, ct::Int64, siriraj::Int64) = Observation(NIHSS(nihss), ct, siriraj)


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

    p_onihssA_ane_avm_occ::Float64 = 0.9
    p_onihssA_ane_avm_notocc::Float64 = 0.8
    p_onihssA_ane_notavm_occ::Float64 = 0.8
    p_onihssA_ane_notavm_notocc::Float64 = 0.8
    p_onihssA_notane_avm_occ::Float64 = 0.7
    p_onihssA_notane_avm_notocc::Float64 = 0.7
    p_onihssA_notane_notavm_occ::Float64 = 0.7
    p_onihssA_notane_notavm_notocc::Float64 = 0.3

    # [NIHSSA, NIHSSB]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [OBSERVE]
    p_onihss_state_action::Vector{Vector{Float64}} = [
        [ 0.85, 0.75, 0.75, 0.75, 0.68, 0.68, 0.68, 0.25 ],
        [ 0.15, 0.25, 0.25, 0.25, 0.32, 0.32, 0.32, 0.75 ]
    ]

    # [TRUE, FALSE]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [OBSERVE]
    p_oct_state_action::Vector{Vector{Float64}} = [
        [ 0.85, 0.75, 0.75, 0.75, 0.68, 0.68, 0.68, 0.25 ],
        [ 0.15, 0.25, 0.25, 0.25, 0.32, 0.32, 0.32, 0.75 ]
    ]

    # [1, 2, 3]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [OBSERVE]
    p_osiriraj_state_action::Vector{Vector{Float64}} = [
        [ 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, ],
        [ 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ],
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ]
    ]

    # [TRUE, FALSE]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [DSA]
    p_ane_state_action::Vector{Vector{Float64}} = [
        [ 0.95, 0.9, 0.85, 0.8, 0.85, 0.8, 0.5, 0.5 ],
        [ 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.5, 0.5 ]
    ]

    # [TRUE, FALSE]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [DSA]
    p_avm_state_action::Vector{Vector{Float64}} = [
        [ 0.95, 0.9, 0.85, 0.8, 0.85, 0.8, 0.5, 0.5 ],
        [ 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.5, 0.5 ]
    ]

    # [TRUE, FALSE]
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    # [DSA]
    p_occ_state_action::Vector{Vector{Float64}} = [
        [ 0.95, 0.9, 0.85, 0.8, 0.85, 0.8, 0.5, 0.5 ],
        [ 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.5, 0.5 ]
    ]

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
    return [DSA, OBSERVE, COIL, EMBOLIZATION, AVMTREAT]
end


function POMDPs.observations(P::DSAPOMDP)
    return [Observation(nihss, ct, siriraj)
            for nihss in [NIHSS_A, NIHSS_B] for ct in [true, false] for siriraj in [1.0, 2.0, 3.0]]
end

function POMDPs.transition(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s) || s.time >= (P.max_duration - 1) || (s.ane && s.avm && s.occ)
        return Deterministic(P.null_state)
    end

    if a == AVMTREAT || a == COIL || a == EMBOLIZATION # divide - need to confirm / 
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

    if a == DSA || a == AVMTREAT || a == COIL || a == EMBOLIZATION
        r += -100 #costly procedure
    end

    if a == AVMTREAT && !s.avm
        r += -10000
    elseif a == AVMTREAT && s.avm
        r += 10000
    elseif a == COIL && !s.ane
        r += -5000
    elseif a == COIL && s.ane
        r += 5000
    elseif a == EMBOLIZATION && !s.occ
        r += -5000
    elseif a == EMBOLIZATION && s.occ
        r += 5000
    elseif a == DSA
        r += -100
    elseif a == OBSERVE && !s.ane && !s.avm && !s.occ
        r += 100
    elseif a == OBSERVE && (s.ane || s.avm || s.occ)
        r += -100
    end
    
    return r
end


# Observation Model: P(O|S) ---> P(S|O) = P(O|S)P(S)/P(O)
function POMDPs.observation(P::DSAPOMDP, sp::State)
    state_status = ""*string(sp.ane ? "ane" : "notane") * "_" * string(sp.avm ? "avm" : "notavm") * "_" * string(sp.occ ? "occ" : "notocc")
    
    nihss_prob = getfield(P, Symbol("p_onihssA_"*state_status))
    dist_nihss = DiscreteNonParametric([Int(nihss) for nihss in [NIHSS_A, NIHSS_B]], [nihss_prob, 1-nihss_prob])

    if sp.avm || sp.ane || sp.occ
        ct_prob = 0.4*sp.avm + 0.2*sp.ane + 0.3*sp.occ 
    else
        ct_prob = 0.1
    end
    dist_ct = DiscreteNonParametric([true, false], [ct_prob, 1-ct_prob])

    siriraj = 0.
    if sp.avm
        siriraj += 1.25
    end
    if sp.ane
        siriraj += 0.5
    end
    if sp.occ
        siriraj += 0.25
    end
    siriraj = max(min(round(siriraj, digits=1), 3.0), 1.0)
    probs = [0.4, 0.3, 0.3]
    if siriraj == 1.0
        probs[1] = 0.9
        probs[2] = 0.05
        probs[3] = 0.05
    elseif siriraj == 2.0
        probs[1] = 0.05
        probs[2] = 0.9
        probs[3] = 0.05
    elseif siriraj == 3.0
        probs[1] = 0.05
        probs[2] = 0.05
        probs[3] = 0.9
    end
    dist_siriraj = DiscreteNonParametric([1.0, 2.0, 3.0], probs)

    return product_distribution(dist_nihss, dist_ct, dist_siriraj)
end

# Observation Model: P(O|S) ---> P(S|O) = P(O|S)P(S)/P(O)
function POMDPs.observation(P::DSAPOMDP, sp::State, a::Action)
    # [ANE_AVM_OCC, ANE_AVM_NOTOCC, ANE_NOTAVM_OCC, ANE_NOTAVM_NOTOCC, NOTANE_AVM_OCC, NOTANE_AVM_NOTOCC, NOTANE_NOTAVM_OCC, NOTANE_NOTAVM_NOTOCC]
    if sp.ane && sp.avm && sp.occ
        state_index = 1
    elseif sp.ane && sp.avm && !sp.occ
        state_index = 2
    elseif sp.ane && !sp.avm && sp.occ
        state_index = 3
    elseif sp.ane && !sp.avm && !sp.occ
        state_index = 4
    elseif !sp.ane && sp.avm && sp.occ
        state_index = 5
    elseif !sp.ane && sp.avm && !sp.occ
        state_index = 6
    elseif !sp.ane && !sp.avm && sp.occ
        state_index = 7
    elseif !sp.ane && !sp.avm && !sp.occ
        state_index = 8
    end

    if a == OBSERVE
        nihssA_prob = P.p_onihss_state_action[1][state_index]
        nihssB_prob = P.p_onihss_state_action[2][state_index]
        dist_nihss = DiscreteNonParametric([Int(nihss) for nihss in [NIHSS_A, NIHSS_B]], [nihssA_prob, nihssB_prob])

        ct_prob = P.p_oct_state_action[1][state_index]
        dist_ct = DiscreteNonParametric([true, false], [ct_prob, 1-ct_prob])

        siriraj_prob = [0.0, 0.0, 0.0]
        siriraj_prob[1] = P.p_osiriraj_state_action[1][state_index]
        siriraj_prob[2] = P.p_osiriraj_state_action[2][state_index]
        siriraj_prob[3] = P.p_osiriraj_state_action[3][state_index]

        dist_siriraj = DiscreteNonParametric([1.0, 2.0, 3.0], siriraj_prob)

        return product_distribution(dist_nihss, dist_ct, dist_siriraj)
    elseif a == DSA
        ane_prob = P.p_ane_state_action[1][state_index]
        avm_prob = P.p_avm_state_action[1][state_index]
        occ_prob = P.p_occ_state_action[1][state_index]
        dist_ane = DiscreteNonParametric([true, false], [ane_prob, 1-ane_prob])
        dist_avm = DiscreteNonParametric([true, false], [avm_prob, 1-avm_prob])
        dist_occ = DiscreteNonParametric([true, false], [occ_prob, 1-occ_prob])

        return product_distribution(dist_ane, dist_avm, dist_occ)
    end
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

    if a == OBSERVE
        obs = rand(rng, observation(P, sp, a))
        obs = Observation(
            nihss = NIHSS(Int(obs[1])),
            ct = obs[2],
            siriraj = obs[3]
        )
    elseif a == DSA
        obs = rand(rng, observation(P, sp, a))
        obs = DSAObservation(
            p_ane = obs[1],
            p_avm = obs[2],
            p_occ = obs[3]
        )
    else
        obs = rand(rng, observation(P, sp))
    end
    
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
    