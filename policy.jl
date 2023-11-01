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