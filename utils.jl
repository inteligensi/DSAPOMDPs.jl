#get  the index of the state
function POMDPs.stateindex(P::DSAPOMDP, s::State)
    return s.avm + 2*s.ane + 4*s.occ + 8*s.have_surgery + 16*s.have_MRA + 32*s.nihss + 64*s.onset + 128*s.t
end

#get the index of the action
function POMDPs.actionindex(P::DSAPOMDP, a::Action)
    actions = actions(P)
    return findfirst(x->x==a, actions)
end

#generate initial state
function POMDPs.initialstate(P::DSAPOMDP)
    return State(
        avm=false, ane=false, occ=true, 
        have_surgery=false, have_MRA=false, 
        nihss=10, onset=0, t=0, hypertension=false)
end

#check if the state is terminal
function POMDPs.isterminal(P::DSAPOMDP, s::State)
    return s == P.null_state
end

# Make a copy of the state
function Base.deepcopy(s::State)
    return State(deepcopy(s.avm), deepcopy(s.ane), deepcopy(s.occ), deepcopy(s.have_surgery), deepcopy(s.have_MRA), deepcopy(s.nihss), deepcopy(s.onset), deepcopy(s.t), deepcopy(s.hypertension))
end

# Make == work for states
function Base.:(==)(s1::State, s2::State)
    return (s1.avm == s2.avm && s1.ane == s2.ane && s1.occ == s2.occ && s1.have_surgery == s2.have_surgery && s1.have_MRA == s2.have_MRA && s1.nihss == s2.nihss && s1.onset == s2.onset && s1.t == s2.t)
end

#compute Siriraj score from given State
function get_siriraj_score(s::State)
    siriraj = 0.
    if s.avm
        siriraj += 1.25
    end
    if s.ane
        siriraj += 0.5
    end
    if s.occ
        siriraj += 0.25
    end
    siriraj = max(min(round(siriraj, digits=1), 3.0), 1.0)
    probs = [0.0, 0.0, 0.0]
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
    return DiscreteNonParametric([1.0, 2.0, 3.0], probs)
end

#compute binary CT scan score from given State
function get_ct_score(s::State)
    if s.avm || s.ane || s.occ
        p_true = 0.4*s.avm + 0.2*s.ane + 0.3*s.occ 
    else
        p_true = 0.1
    end
    return DiscreteNonParametric([true, false], [p_true, 1.0-p_true])
end

