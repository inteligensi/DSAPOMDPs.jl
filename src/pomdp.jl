using POMDPs
using POMDPTools
using Parameters
using Random
using Distributions

@with_kw mutable struct State
    ane::Bool = false #aneurysm
    avm::Bool = false #arteriovenous malformations
    occ::Bool = false #occlusion
    time::Int64 = 0 #current time
end

#HOSP: Hospitalize, DSA: Digital Subtraction Angiography, COIL: Coiling, EMBO: Embolization, REVC: Revascularisation, DISC: Discharge patient
@enum Action WAIT HOSP DSA COIL EMBO REVC DISC
@enum CT CT_POS CT_NEG
@enum SIRIRAJ SIRIRAJ_LESSNEG1 SIRIRAJ_AROUND0 SIRIRAJ_GREATER1

#Observation struct with WAIT and HOSP action
@with_kw mutable struct WHObs
    ct::CT
    siriraj::SIRIRAJ
end

#Observation struct with DSA action
@with_kw mutable struct DSAObs
    pred_ane::Bool
    pred_avm::Bool
    pred_occ::Bool
end

const Observation = Union{WHObs, DSAObs}

@with_kw mutable struct DSAPOMDP <: POMDP{State, Action, Observation}
    #probability of each part of state became true with non-treatment action 
    p_avm::Float64 = 0.0002
    p_ane::Float64 = 0.0005
    p_occ::Float64 = 0.0002
    
    #probability of ct scan is true given state with action [HOSP, REVC, COIL, EMBO, DISC]
    p1_ct_true_given_stateindex::Vector{Float64} = [ 0.85, 0.80, 0.70, 0.75, 0.77, 0.75, 0.68, 0.25]
    #probability of siriraj score = [x<=-1, -1<x<1, x>=1] given state with action [HOSP, REVC, COIL, EMBO, DISC]
    p1_siriraj_given_stateindex::Vector{Vector{Float64}} = [
        [0.04, 0.08, 0.88], 
        [0.02, 0.08, 0.90], 
        [0.25, 0.1, 0.65], 
        [0.08, 0.07, 0.85], 
        [0.14, 0.08, 0.78], 
        [0.10, 0.05, 0.85], 
        [0.85, 0.07, 0.08], 
        [0.05, 0.90, 0.05]
    ]
    
    #probability of ct scan is true given state with action [WAIT]
    p2_ct_true_given_stateindex::Vector{Float64} = [ 0.65, 0.60, 0.55, 0.58, 0.55, 0.58, 0.58, 0.35]
    #probability of siriraj score = [x<=-1, -1<x<1, x>=1] given state with action [WAIT]
    p2_siriraj_given_stateindex::Vector{Vector{Float64}} = [
        [0.04, 0.28, 0.68], 
        [0.02, 0.28, 0.70], 
        [0.25, 0.3, 0.45], 
        [0.08, 0.27, 0.65], 
        [0.14, 0.28, 0.58], 
        [0.10, 0.25, 0.65], 
        [0.65, 0.27, 0.08], 
        [0.1, 0.8, 0.1]
    ]

    #probability of each part of state become true given state
    p_ane_true_given_stateindex::Vector{Float64} = [ 0.98, 0.98, 0.98, 0.98, 0.05, 0.05, 0.05, 0.05]
    p_avm_true_given_stateindex::Vector{Float64} = [ 0.98, 0.98, 0.05, 0.05, 0.98, 0.98, 0.05, 0.05]
    p_occ_true_given_stateindex::Vector{Float64} = [ 0.98, 0.05, 0.98, 0.05, 0.98, 0.05, 0.98, 0.05]
        
    max_duration::Int64 = 24
    discount::Float64 = 0.95

    null_state::State = State(
        ane = true,
        avm = true,
        occ = true,
        time = -1
    )

    discharge_state::State = State(
        ane = false,
        avm = false,
        occ = false,
        time = -99
    )
end

Base.:(==)(s1::State, s2::State) = s1.ane == s2.ane && s1.avm == s2.avm && s1.occ == s2.occ && s1.time == s2.time

POMDPs.isterminal(P::DSAPOMDP, s::State) = (s == P.null_state) || (s == P.discharge_state)

#State space
function POMDPs.states(P::DSAPOMDP)
    return [State(ane, avm, occ, time)
            for ane in [true, false] for avm in [true, false] for occ in [true, false] for time in 0:P.max_duration]
end

POMDPs.initialstate(P::DSAPOMDP) = Deterministic(State(ane=true, avm=false, occ=false, time=0))

#Action space
function POMDPs.actions(P::DSAPOMDP)
    return [WAIT, HOSP, DSA, COIL, EMBO, REVC, DISC]
end

#Observation space
function POMDPs.observations(P::DSAPOMDP)
    WHObs_space = [WHObs(ct, siriraj) for ct in [CT_POS, CT_NEG] for siriraj in [SIRIRAJ_LESSNEG1, SIRIRAJ_AROUND0, SIRIRAJ_GREATER1]]
    DSAObs_space = [DSAObs(pred_ane, pred_avm, pred_occ) for pred_ane in [true, false] for pred_avm in [true, false] for pred_occ in [true, false]]
    return vcat(WHObs_space, DSAObs_space)
end

function POMDPs.obsindex(P::DSAPOMDP, obs::Union{WHObs, DSAObs})
    if obs isa WHObs
        ct_base = obs.ct == CT_POS ? 1 : 4
        siriraj_idx = obs.siriraj == SIRIRAJ_LESSNEG1 ? 0 : obs.siriraj == SIRIRAJ_AROUND0 ? 1 : 2
        return ct_base + siriraj_idx
    else
        ane_base = obs.pred_ane ? 6 : 10
        avm_base = obs.pred_avm ? 0 : 1
        occ_idx = obs.pred_occ ? 1 : 2
        return ane_base + avm_base*2 + occ_idx
    end
end

#Transition function
function POMDPs.transition(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s) 
        return Deterministic(s)
    end

    #transition to the null state by the end of time P.max_duration or when the patient has all three conditions
    if s.time >= (P.max_duration - 1) || (s.ane && s.avm && s.occ)
        return Deterministic(P.null_state)
    end

    if a == DISC
        return Deterministic(P.discharge_state)
    end

    p_ane = s.ane ? 1. : P.p_ane
    p_avm = s.avm ? 1. : P.p_avm
    p_occ = s.occ ? 1. : P.p_occ

    ane_dist = DiscreteNonParametric([true, false], [p_ane, 1-p_ane])
    avm_dist = DiscreteNonParametric([true, false], [p_avm, 1-p_avm])
    occ_dist = DiscreteNonParametric([true, false], [p_occ, 1-p_occ])
    
    if a == EMBO 
        avm_dist = DiscreteNonParametric([true, false], [0., 1.])
    elseif a == COIL
        ane_dist = DiscreteNonParametric([true, false], [0., 1.])
    elseif a == REVC
        occ_dist = DiscreteNonParametric([true, false], [0., 1.])
    end

    support = collect(0:1:P.max_duration)
    probs = fill(0., P.max_duration + 1)
    probs[s.time + 2] = 1.
    time_dist = DiscreteNonParametric(support, probs)

    dist = product_distribution(ane_dist, avm_dist, occ_dist, time_dist)

    return dist
end

#Reward function
function POMDPs.reward(P::DSAPOMDP, s::State, a::Action, sp::State)
    r = 0

    if s != P.null_state && sp == P.null_state
        r += -100000 #Huge penalty for first time entering null state
    elseif isterminal(P, s)
        return 0
    end

    #Assign rewards based on the action and the current state
    if a in [REVC, COIL, EMBO]
        r += -200
    elseif a == DSA
        r += -150
    elseif a == HOSP
        r += -100
    end

    #impose penalties for mismatch and rewards for matching cases
    if a == EMBO
        r += s.avm ? 5000 : -5000
    elseif a == COIL
        r += s.ane ? 5000 : -5000
    elseif a == REVC
        r += s.occ ? 5000 : -5000
    elseif a == DSA
        r += (!s.ane && !s.avm && !s.occ) ? -750 : 250
    elseif a == HOSP
        r += (!s.ane && !s.avm && !s.occ) ? -400 : 150
    elseif a == WAIT
        r += (!s.ane && !s.avm && !s.occ) ? 100 : -1000
    elseif a == DISC
        r += (!s.ane && !s.avm && !s.occ) ? 5000 : -50000
    end

    return r
end

function POMDPs.discount(P::DSAPOMDP)
    return P.discount
end

#Observation Model
function POMDPs.observation(P::DSAPOMDP, a::Action, sp::State)
    state_index = state2stateindex(sp)

    #choose obs type for each action type
    if a in [HOSP, REVC, COIL, EMBO, DISC]
        ct_prob = P.p1_ct_true_given_stateindex[state_index]
        dist_ct = DiscreteNonParametric(
            [0, 1], 
            [ct_prob, 1-ct_prob]
            )

        siriraj_prob = P.p1_siriraj_given_stateindex[state_index]
        dist_siriraj = DiscreteNonParametric(
            [0, 1, 2], 
            [siriraj_prob[1], siriraj_prob[2], 1-siriraj_prob[1]-siriraj_prob[2]]
            )

        return product_distribution(dist_ct, dist_siriraj)

    elseif a == WAIT
        ct_prob = P.p2_ct_true_given_stateindex[state_index]
        dist_ct = DiscreteNonParametric(
            [0, 1], 
            [ct_prob, 1-ct_prob]
            )

        siriraj_prob = P.p2_siriraj_given_stateindex[state_index]
        dist_siriraj = DiscreteNonParametric(
            [0, 1, 2], 
            [siriraj_prob[1], siriraj_prob[2], 1-siriraj_prob[1]-siriraj_prob[2]]
            )

        return product_distribution(dist_ct, dist_siriraj)

    elseif a == DSA
        ane_true_prob = P.p_ane_true_given_stateindex[state_index]
        avm_true_prob = P.p_avm_true_given_stateindex[state_index]
        occ_true_prob = P.p_occ_true_given_stateindex[state_index]

        ane_true_dist = DiscreteNonParametric([true, false], [ane_true_prob, 1-ane_true_prob])
        avm_true_dist = DiscreteNonParametric([true, false], [avm_true_prob, 1-avm_true_prob])
        occ_true_dist = DiscreteNonParametric([true, false], [occ_true_prob, 1-occ_true_prob])

        return product_distribution(ane_true_dist, avm_true_dist, occ_true_dist)

    end
end


function POMDPs.initialize_belief(up)
    #initialize belief as a uniform distribution over the state space for when t=0 and 0 for all other times
    P = up.predict_model
    all_support = [states(P)...]
    support = [state for state in all_support if state.time == 0]
    probs = fill(1/length(support), length(support))
    for i in eachindex(probs)
        countTrue = 1 * support[i].ane + 1 * support[i].avm + 1 * support[i].occ
        #distribution over the state space
        #   0.785 with stroke-free conditions
        #   0.05 with only one stroke condition (either aneurysm, AVM, or occlusion
        #   0.02 with two combinations
        #   0.005 with all the conditions
        if countTrue == 0
            probs[i] = 0.785
        elseif countTrue == 1
            probs[i] = 0.05
        elseif countTrue == 2
            probs[i] = 0.02
        elseif countTrue == 3
            probs[i] = 0.005
        end
    end 
    return SparseCat(support, probs)
end


function POMDPs.gen(P::DSAPOMDP, s::State, a::Action, rng::AbstractRNG)
    if isterminal(P, s) || s.time >= (P.max_duration - 1) 
        sp = s
    else
        sp_array = rand(rng, transition(P, s, a))
        if isa(sp_array, State)
            #occurs at edge cases where sp_array is a null state
            sp = sp_array
        else             
            sp = State(ane=sp_array[1], avm=sp_array[2], occ=sp_array[3], time=sp_array[4])
        end                
    end

    obs_dist = observation(P, a, sp)  
    obs_array = rand(rng, obs_dist)
    if a in [WAIT, HOSP, REVC, COIL, EMBO, DISC]        
        o = WHObs(
            ct = CT(Int(obs_array[1])),
            siriraj = SIRIRAJ(Int(obs_array[2]))
        )
    else DSA
        o = DSAObs(
            pred_ane = obs_array[1],
            pred_avm = obs_array[2],
            pred_occ = obs_array[3]
        )
    end    
    r = reward(P, s, a, sp)

    return (sp = sp, o = o, r = r)
end


function POMDPs.stateindex(P::DSAPOMDP, s::State)
    ane_idx = s.ane ? 1 : 2
    avm_idx = s.avm ? 1 : 2
    occ_idx = s.occ ? 1 : 2
    time_idx = s.time + 1  #0-based to 1-based
 
    return state_sub2ind((2, 2, 2, P.max_duration + 1), ane_idx, avm_idx, occ_idx, time_idx)
 end

 function POMDPs.actionindex(P::DSAPOMDP, a::Action)
    if a == WAIT
        return 1
    elseif a == HOSP
        return 2
    elseif a == DSA
        return 3
    elseif a == COIL
        return 4
    elseif a == EMBO
        return 5
    elseif a == REVC
        return 6
    elseif a == DISC
        return 7
    end
 end
