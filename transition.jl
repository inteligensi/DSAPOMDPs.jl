#compute transition distribution
function compute_transition_function(P::DSAPOMDP, s::State, a::Action)
    #compute transition elements

    #if avm, always stays avm. if not avm, p_avm% chance of becoming afm 
    if s.avm
        p_avm = 1.
    elseif (~s.avm) && (a == SURGERY || a == COIL || a == EMBOLIZATION)
        p_avm = 0.
    else
        p_avm = P.p_avm + s.ane*P.p_ane + s.occ*P.p_occ
    end
    avm_dist = DiscreteNonParametric([true, false], [p_avm, 1-p_avm])
    
    #if ane, always stays ane. if not ane, p_ane% chance of becoming ane
    if s.ane
        p_ane = 1.
    elseif (~s.ane) && (a == SURGERY || a == COIL || a == EMBOLIZATION)
        p_ane = 0.
    else
        p_ane = P.p_ane + s.occ*P.p_occ
    end
    ane_dist = DiscreteNonParametric([true, false], [p_ane, 1-p_ane])

    #if occ, always stays occ. if not occ, p_occ% chance of becoming occ
    if s.occ
        p_occ = 1.
    elseif (~s.occ) && (a == SURGERY || a == COIL || a == EMBOLIZATION)
        p_occ = 0.
    else
        p_occ = P.p_occ
    end
    occ_dist = DiscreteNonParametric([true, false], [p_occ, 1-p_occ])

    #if have_surgery or action is SURGERY, always stays have_surgery
    if  a == SURGERY
        have_surgery_dist = DiscreteNonParametric([true], [1.])
    else
        have_surgery_dist = DiscreteNonParametric([false], [1.])
    end

    #if have_MRA or action is MRA, always stays have_MRA
    if s.have_MRA || a == MRA
        have_MRA_dist = DiscreteNonParametric([true], [1.])
    else
        have_MRA_dist = DiscreteNonParametric([false], [1.])
    end

    #increment nihss by 1 if action is SURGERY, COIL, or EMBOLIZATION
    nihss = s.nihss
    if a == SURGERY || a == COIL || a == EMBOLIZATION
        nihss += 1
    end

    # add simple rule how state affects nihss
    if p_avm > 0.5
        nihss += 2
    elseif p_ane > 0.5
        nihss += 1
    elseif p_occ > 0.5
        nihss += 0.5
    end

    #round nihss to nearest integer
    nihss = Int(round(nihss, digits=0))

    #if nihss > max_nihss, set nihss to max_nihss
    if nihss > P.max_nihss
        nihss = P.max_nihss
    end

    if s.avm && s.ane && s.occ
        nihss = P.max_nihss
    end
    
    #create nihss distribution
    support = collect(0:1:P.max_nihss)    
    probs = [0. for _ in 0:P.max_nihss]
    probs[max(1, nihss)] += 0.05
    probs[min(nihss+1, P.max_nihss+1)] += 0.9
    probs[min(nihss+2, P.max_nihss+1)] += 0.05
    nihss_dist = DiscreteNonParametric(support, probs)
    
    #increment onset by 1 if action is not WAIT
    if s.onset == 0 && a != WAIT        
        onset_dist = DiscreteNonParametric([s.onset], [1.])
    elseif s.onset == 0 && a == WAIT
        onset_dist = DiscreteNonParametric([1.], [1.])
    else
        onset_dist = DiscreteNonParametric([s.onset+1.], [1.])
    end

    #increment t by 1
    support = collect(0:1:P.max_duration)
    probs = [0. for i in 0:P.max_duration]
    probs[s.t+2] = 1.
    t_dist = DiscreteNonParametric(support, probs)

    hypertension_dist = DiscreteNonParametric([s.hypertension], [1.])

    #compute transition distribution
    dist = product_distribution(
        avm_dist, ane_dist, occ_dist, 
        have_surgery_dist, have_MRA_dist, 
        nihss_dist, onset_dist, t_dist,
        hypertension_dist)
    return dist
end


function POMDPs.transition(P::DSAPOMDP, s::State, a::Action, rng::AbstractRNG)
    #always return null state if the state is terminal
    if isterminal(P, s)
        return P.null_state
    end 

    #return to null state if time is up
    if s.t >= (P.max_duration - 1)
        return P.null_state
    end   

    #if nihss maxed, or has surgery, or has all three conditions, return to null state
    if s.nihss == P.max_nihss || s.have_surgery || s.avm && s.ane && s.occ
        return P.null_state
    end
    
    #compute transition distribution
    dist = compute_transition_function(P, s, a)

    #sample next state
    sp = rand(rng, dist)
    
    #return next state
    #avm_dist, ane_dist, occ_dist, 
    #have_surgery_dist, have_MRA_dist, 
    #nihss_dist, onset_dist, t_dist
    next_state = State(
        avm=sp[1], ane=sp[2], occ=sp[3], 
        have_surgery=sp[4], have_MRA=sp[5], 
        nihss=sp[6], onset=sp[7], t=sp[8], hypertension=sp[9]
        )
    return next_state
end
