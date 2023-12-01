function hello_world()
    return "Hello World!"
end


function state2stateindex(s::State)
    if s.ane && s.avm && s.occ
        state_index = 1
    elseif s.ane && s.avm && !s.occ
        state_index = 2
    elseif s.ane && !s.avm && s.occ
        state_index = 3
    elseif s.ane && !s.avm && !s.occ
        state_index = 4
    elseif !s.ane && s.avm && s.occ
        state_index = 5
    elseif !s.ane && s.avm && !s.occ
        state_index = 6
    elseif !s.ane && !s.avm && s.occ
        state_index = 7
    elseif !s.ane && !s.avm && !s.occ
        state_index = 8
    end
    return state_index
end


function Distributions.pdf(dist::Union{Distributions.ProductDistribution}, o::WHObs)
    ct_val = o.ct == CT_POS ? 0 : 1
    siriraj_val = o.siriraj == SIRIRAJ_LESSNEG1 ? 0 : o.siriraj == SIRIRAJ_AROUND0 ? 1 : 2    
    return pdf(dist, [ct_val, siriraj_val])
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}}, Discrete, Bool}, o::DSAObs)
    return pdf(dist, [o.pred_ane, o.pred_avm, o.pred_occ])
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}, DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}}, Discrete, Int64}, o::DSAObs)    
    # invalid case
    return 0.0
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}}, Discrete, Bool}, o::WHObs)
    # invalid case
    return 0.0
end

function Distributions.support(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}, DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}}, Discrete, Int64})    
    return [WHObs(CT(Int(ct)), SIRIRAJ(Int(siriraj))) for ct in [0, 1] for siriraj in [0, 1, 2]]
end

function Distributions.support(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}}, Discrete, Bool})
    return [DSAObs(pred_ane, pred_avm, pred_occ) for pred_ane in [true, false] for pred_avm in [true, false] for pred_occ in [true, false]]
end

function Distributions.support(dist::Distributions.ProductDistribution{1, 0, Tuple{DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Bool, Float64, Vector{Bool}, Vector{Float64}}, DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}}, Discrete, Int64})
    return [State(ane, avm, occ, time)
            for ane in [true, false] for avm in [true, false] for occ in [true, false] for time in 0:24]
end

function Distributions.pdf(dist::Distributions.ProductDistribution, s::State)
    return pdf(dist, [s.ane, s.avm, s.occ, s.time])
end



function compute_rdisc(P, h)
    rsum = 0.0
    disc = 1.0
    for step in eachstep(h)
        rsum += (step.r * disc)
        disc *= discount(P)
    end

    @show rsum
    return rsum
end


function state_sub2ind(dims::Tuple, i1, i2, i3, i4)
    return ((i4 - 1) * prod(dims[1:3]) +
            (i3 - 1) * prod(dims[1:2]) +
            (i2 - 1) * dims[1] +
             i1)
 end


 