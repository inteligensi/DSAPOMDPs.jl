using POMDPs
using POMDPModelTools
using Distributions
using Parameters

@with_kw mutable struct State
    avm::Bool = false # AVM
    ane::Bool = false # Aneurysm
    occ::Bool = false # Occlusion can happen if AVM or Aneurysm or other factors (hypertension, vascular damage, etc.)
    have_surgery::Bool = false # Have surgery
    have_MRA::Bool = false # Have MRA
    nihss::Int64 = 0 # NIHSS
    onset::Int64 = 0 # Onset
    t::Int64 = 0 # Time
    hypertension::Bool = false
end


# avm
# P(avm|congenital) = 0.9
# P(~avm|congenital) = 0.1
# P(congenital) = 0.001
# P(avm) = 0.009

#how does occlusion probability is affected by AVM or Aneurysm, or other factors
# P(occ) = P(occ|avm)P(avm) + P(occ|ane)P(ane|hypertension)P(hypertension) + P(occ|other)P(other)

# P(occ|avm) = 0.8
# P(~occ|avm) = 0.2

# P(occ|ane) = 0.1
# P(~occ|ane) = 0.9

# P(ane|hypertension) = 0.1
# P(~ane|hypertension) = 0.9
# P(ane|~hypertension) = 0.01
# P(~ane|~hypertension) = 0.99
# P(hypertension) = 0.7
# P(~hypertension) = 0.3
# P(ane) = 0.07

# P(occ|other) = 0.8
# P(~occ|other) = 0.2
# P(other) = 0.7

# calculate P(occ)
# P(occ) = P(occ|avm)P(avm) + P(occ|ane)P(ane|hypertension)P(hypertension) + P(occ|other)P(other)
# P(occ) = 0.8*0.009 + 0.07 + 0.8*0.7 = 0.053


@enum Action DSA MRA OBSERVE SURGERY COIL EMBOLIZATION WAIT

@with_kw mutable struct DSAPOMDP
    # POMDP Parameters
    discount_factor::Float64 = 0.95
    max_nihss::Int64 = 42
    max_duration::Int64 = 24 #months
    p_avm::Float64 = 0.02    
    p_ane::Float64 = 0.05
    p_occ::Float64 = 0.1
    null_state::State = State(
        avm=true, ane=true, occ=true, 
        have_surgery=true, have_MRA=true, 
        nihss=-1, onset=-1, t=-1, hypertension=true
        )
end

function POMDPs.states(P::DSAPOMDP)
    states = [State(avm, ane, occ, have_surgery, 
        have_MRA, nihss, onset, t, hypertension) 
        for avm in [true, false], 
            ane in [true, false], 
            occ in [true, false], 
            have_surgery in [true, false], 
            have_MRA in [true, false], 
            nihss in 0:P.max_nihss, 
            onset in 0:P.max_duration, 
            t in 0:P.max_duration, 
            hypertension in [true, false]
    return states
end

function POMDPs.actions(P::DSAPOMDP)
    return [DSA, MRA, OBSERVE, SURGERY, COIL, EMBOLIZATION, WAIT]
end

function POMDPs.reward(P::DSAPOMDP, s::State, a::Action)
    if isterminal(P, s)
        return 0
    end
    if a == SURGERY && (~s.avm)
        return -2000
    elseif a == COIL && (~s.ane)
        return -1000
    elseif a == EMBOLIZATION && (~s.occ)
        return -1000
    elseif a == DSA 
        return -200
    elseif a == MRA
        return -100
    elseif a== OBSERVE && (~s.avm || ~s.ane || ~s.occ)
        return -100
    elseif a == WAIT && (s.avm || s.ane || s.occ)
        return -50
    else
        return 0
    end
end

function POMDPs.discount(P::DSAPOMDP)
    return P.discount_factor
end

function POMDPs.observation(P::DSAPOMDP, s::State, a::Action)

    #compute obs elements
    sentinel_dist = DiscreteNonParametric([-1.], [1.])    
    siriraj = get_siriraj_score(s)
    ct = get_ct_score(s)
    
    if !isterminal(P, s) 
        return product_distribution(siriraj, ct)
    else
        return product_distribution(sentinel_dist, sentinel_dist)
    end
end
