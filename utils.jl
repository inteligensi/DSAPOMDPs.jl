function state_sub2ind(dims::Tuple, i1, i2, i3, i4)
    return ((i4 - 1) * prod(dims[1:3]) +
            (i3 - 1) * prod(dims[1:2]) +
            (i2 - 1) * dims[1] +
             i1)
 end

# #  function obs_sub2ind(dims::Tuple, i1, i2, i3)
# #     return ((i3 - 1) * prod(dims[1:2]) +
# #             (i2 - 1) * dims[1] +
# #              i1)
# #  end

function POMDPs.stateindex(P::DSAPOMDP, s::State)
    ane_idx = s.ane ? 1 : 2
    avm_idx = s.avm ? 1 : 2
    occ_idx = s.occ ? 1 : 2
    time_idx = s.time + 1  # 0-based to 1-based
 
    return state_sub2ind((2, 2, 2, P.max_duration + 1), ane_idx, avm_idx, occ_idx, time_idx)
 end
 
# # function POMDPs.obsindex(P::DSAPOMDP, o::Observation)
# #     is_ane_idx = o.is_ane ? 1 : 2
# #     is_avm_idx = o.is_avm ? 1 : 2
# #     is_occ_idx = o.is_occ ? 1 : 2
 
# #     return obs_sub2ind((2, 2, 2), is_ane_idx, is_avm_idx, is_occ_idx)
# # end
 
function POMDPs.actionindex(P::DSAPOMDP, a::Action)
    action_list = POMDPs.actions(P)
    return findfirst(==(a), action_list)
end

# # function Distributions.pdf(dist::Union{Distributions.ProductDistribution, Deterministic{State}}, s::State)
# #     s_values = [s.ane, s.avm, s.occ, s.time, s.hypertension]
# #     return pdf(dist, s_values)
# # end
# function obs2ind(o::Observation)
#     return 
# end

# function nihss2ind(o::Observation)
#     if o.nihss == NIHSS_A
#         return 0
#     else
#         return 1
#     end
# end

function Distributions.pdf(dist::Union{Distributions.ProductDistribution}, o::Observation)
    o_values = [Int(o.nihss == NIHSS_A), o.ct, o.siriraj]
    return pdf(dist, o_values)
end
