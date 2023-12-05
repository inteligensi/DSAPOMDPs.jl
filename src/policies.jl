using ParticleFilters
using POMDPs
using Parameters

function calc_condition_rate(b::ParticleCollection{State})
    N = length(particles(b))
    rates = Dict{String, Float64}()
    rates["ane"] = sum([s.ane for s in particles(b)])/N
    rates["avm"] = sum([s.avm for s in particles(b)])/N
    rates["occ"] = sum([s.occ for s in particles(b)])/N
    return rates
end

function base_policy(P::DSAPOMDP, b; idx_a_default=2, min_rate=0.6, min_case_discharge_prob=0.9)
    #use belief with the highest prob for making accept/discharge decision
    prob = particle2prob(b)
    k = keymax(prob)    

    #if no condition with high enough prob, then DISC
    #8 is the stateindex for (no Ane, no AVM, no Occ)
    if (k == 8)  && (prob[k] ≥ min_case_discharge_prob)
        return actions(P)[7] #DISC
    else
        #perform treatment based on the condition with the dominant rate (≥ min_rate)
        #if no dominant rate, perform default action (idx_a_default)
        rates = calc_condition_rate(b) #rates is a Dict{String, Float64} with keys "ane", "avm", "occ"
        argmax_ = keymax(rates)
        max_rate = rates[argmax_]

        if max_rate < min_rate
            return actions(P)[idx_a_default] 
        else
            if argmax_ == "ane"
                return actions(P)[4] #COIL
            elseif argmax_ == "avm"
                return actions(P)[5] #EMBO
            else
                return actions(P)[6] #REVC
            end
        end
    end
end

@with_kw mutable struct DSAPolicy <: Policy P::DSAPOMDP end
POMDPs.action(policy::DSAPolicy, b::ParticleCollection{State}) = base_policy(policy.P, b, idx_a_default=3)

@with_kw mutable struct HOSPPolicy <: Policy P::DSAPOMDP end
POMDPs.action(policy::HOSPPolicy, b::ParticleCollection{State}) = base_policy(policy.P, b, idx_a_default=2)