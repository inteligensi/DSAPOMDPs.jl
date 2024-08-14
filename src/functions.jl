using ParticleFilters
using DSAPOMDPs
using DataFrames
using Plots
using Plots.PlotMeasures

function keymax(d::Dict)
    if isempty(d)
        throw(ArgumentError("The dictionary is empty. Cannot determine the maximum key."))
    end

    return reduce((x, y) -> d[x] >= d[y] ? x : y, keys(d))
end

function particle2prob(b::ParticleCollection{State})
    try
        particles_list = particles(b)
        N = length(particles_list)

        if N == 0
            error("The particle collection is empty.")
        end

        prob = Dict{Int,Float64}()

        for ane in [true, false]
            for avm in [true, false]
                for occ in [true, false]
                    state_idx = state2stateindex(State(ane, avm, occ, 0))
                    prob[state_idx] = sum([(s.ane == ane) && (s.avm == avm) && (s.occ == occ) for s in particles_list]) / N
                end
            end
        end

        return prob

    catch e
        # println("An error occurred: ", e)
        rethrow(e)
    end
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

function stateindex2string(index)
    mapping = Dict(
        1 => "(Ane, AVM, Occ)",
        2 => "(Ane, AVM)",
        3 => "(Ane, Occ)",
        4 => "(Ane)",
        5 => "(AVM, Occ)",
        6 => "(AVM)",
        7 => "(Occ)",
        8 => "(None)"
    )

    if haskey(mapping, index)
        return mapping[index]
    else
        throw(error("Invalid state index: $index"))
    end
end


function Distributions.pdf(dist::Union{Distributions.ProductDistribution}, o::WHObs)
    ct_val = o.ct == CT_POS ? 0 : 1
    siriraj_val = o.siriraj == SIRIRAJ_LESSNEG1 ? 0 : o.siriraj == SIRIRAJ_AROUND0 ? 1 : 2
    return pdf(dist, [ct_val, siriraj_val])
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}}},Discrete,Bool}, o::DSAObs)
    return pdf(dist, [o.pred_ane, o.pred_avm, o.pred_occ])
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Int64,Float64,Vector{Int64},Vector{Float64}},DiscreteNonParametric{Int64,Float64,Vector{Int64},Vector{Float64}}},Discrete,Int64}, o::DSAObs)
    # invalid case
    return 0.0
end

function Distributions.pdf(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}}},Discrete,Bool}, o::WHObs)
    # invalid case
    return 0.0
end

function Distributions.support(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Int64,Float64,Vector{Int64},Vector{Float64}},DiscreteNonParametric{Int64,Float64,Vector{Int64},Vector{Float64}}},Discrete,Int64})
    return [WHObs(CT(Int(ct)), SIRIRAJ(Int(siriraj))) for ct in [0, 1] for siriraj in [0, 1, 2]]
end

function Distributions.support(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}}},Discrete,Bool})
    return [DSAObs(pred_ane, pred_avm, pred_occ) for pred_ane in [true, false] for pred_avm in [true, false] for pred_occ in [true, false]]
end

function Distributions.support(dist::Distributions.ProductDistribution{1,0,Tuple{DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Bool,Float64,Vector{Bool},Vector{Float64}},DiscreteNonParametric{Int64,Float64,Vector{Int64},Vector{Float64}}},Discrete,Int64})
    return [State(ane, avm, occ, time)
            for ane in [true, false] for avm in [true, false] for occ in [true, false] for time in 0:24]
end

function Distributions.pdf(dist::Distributions.ProductDistribution, s::State)
    return pdf(dist, [s.ane, s.avm, s.occ, s.time])
end

function state_sub2ind(dims::Tuple, i1, i2, i3, i4)
    dim1, dim2, dim3, dim4 = dims

    if i1 < 1 || i1 > dim1
        throw(BoundsError("Index i1 out of bounds"))
    end
    if i2 < 1 || i2 > dim2
        throw(BoundsError("Index i2 out of bounds"))
    end
    if i3 < 1 || i3 > dim3
        throw(BoundsError("Index i3 out of bounds"))
    end
    if i4 < 1 || i4 > dim4
        throw(BoundsError("Index i4 out of bounds"))
    end

    return ((i4 - 1) * prod(dims[1:3]) +
            (i3 - 1) * prod(dims[1:2]) +
            (i2 - 1) * dims[1] +
            i1)
end


function evaluate_policies(sim, pomdp, policies, up, b0, s0; pol_names=nothing, verbose=true, save_to_file=true)
    if pol_names == nothing
        pol_names = [string(i) for i in 1:length(policies)]
    end

    rdiscs = Array{Float64}(undef, length(policies))
    durations = Array{Float64}(undef, length(policies))
    is_treateds = Array{Bool}(undef, length(policies))

    # @show s0
    for (j, policy) in enumerate(policies)

        h_policy = simulate(sim, pomdp, policy, up, b0, s0)

        rdiscs[j], actions, states, state_primes, _, _, observations, is_treateds[j], durations[j] = summarize_rollout(pomdp, h_policy)

        if verbose
            println("==========simulation for policy $(pol_names[j]) ==========")
            println("states = $(join([s for s in states], ", "))")
            println("state_primes = $(join([sp for sp in state_primes], ", "))")
            println("observations = $(join([o for o in observations], ", "))")
            println("actions = $(join([a isa Symbol ? String(a)[11:end] : a for a in actions], ", "))")
            println("rsum = $(rdiscs[j])")
            println("duration = $(durations[j])")
            println("is_treated = $(is_treateds[j])")
        end

        if save_to_file
            file_path = "output-$((pol_names[j])).txt"
            if isfile(file_path)
                file = open(file_path, "a")
            else
                file = open(file_path, "w")
            end

            println(file, "==========simulation for policy $(pol_names[j]) ==========")
            println(file, "states = $(join([s for s in states], ", "))")
            println(file, "state_primes = $(join([sp for sp in state_primes], ", "))")
            println(file, "observations = $(join([o for o in observations], ", "))")
            println(file, "actions = $(join([a isa Symbol ? String(a)[11:end] : a for a in actions], ", "))")
            println(file, "rsum = $(rdiscs[j])")
            println(file, "duration = $(durations[j])")
            println(file, "is_treated = $(is_treateds[j])")

            close(file)
        end

    end

    return (rdiscs=rdiscs, durations=durations, is_treateds=is_treateds)
end


function summarize_rollout(P, h)
    rsum = 0.0
    disc = 1.0

    actions = [step.a for step in eachstep(h)]
    states = [step.s for step in eachstep(h)]
    state_primes = [step.sp for step in eachstep(h)]
    beliefs = [step.b for step in eachstep(h)]
    belief_primes = [step.bp for step in eachstep(h)]
    observations = [step.o for step in eachstep(h)]
    time_to_recover = P.max_duration
    is_treated = false

    for step in eachstep(h)
        rsum += (step.r * disc)
        disc *= discount(P)
        needs_treatment = step.s.ane || step.s.avm || step.s.occ
        still_needs_treatment = step.sp.ane || step.sp.avm || step.sp.occ
        if needs_treatment && !still_needs_treatment && step.a in [COIL, EMBO, REVC]
            time_to_recover = step.s.time + 1
            is_treated = true
        end
    end

    return rsum, actions, states, state_primes, beliefs, belief_primes, observations, is_treated, time_to_recover
end


function replicate_policy_eval(hr, P, policies, up, b0; num_reps=100, pol_names=nothing, verbose=false, save_to_file=false)
    π_names = pol_names === nothing ? ["$(i)" for i in 1:length(policies)] : pol_names
    dict_results = Dict()

    for rep in 1:num_reps
        s0 = rand(b0)
        rdiscs, time2recovers, is_treateds = evaluate_policies(hr, P, policies, up, b0, s0, pol_names=π_names, verbose=verbose, save_to_file=save_to_file)

        d_row = Dict()
        d_row["need_treatment"] = s0.ane || s0.avm || s0.occ
        for (i, name) in enumerate(π_names)
            d_row["rdisc_$(name)"] = rdiscs[i]
            d_row["time2recover_$(name)"] = time2recovers[i]
            d_row["treated_$(name)"] = is_treateds[i]
        end
        dict_results[string(rep)] = d_row
    end
    return dict_results
end


function dict2df(dict_result::Dict)
    n = length(keys(dict_result))
    df = DataFrame(dict_result["1"])
    for i in 2:n
        df = vcat(df, DataFrame(dict_result[string(i)]))
    end
    return df
end


function plot_belief_hist(hist, t=1)
    if t == 0
        prob = particle2prob(hist[1].b)
        a = hist[1].a
        text_ = " "
    else
        prob = particle2prob(hist[t].bp)
        a = hist[t].a
        text_ = "(action = $(a))"
    end

    x = [1, 2, 3, 5, 4, 6, 7, 8]
    xlabel = [stateindex2string(i) for i in x]
    y = [prob[i] for i in x]

    #plot barplot of the probability of each state
    plt = bar(
        xlabel, y, title="Belief at t=$t",
        xlabel="State", ylabel="Probability",
        ylim=(0, 1), xrotation=30,
        legend=false, size=(500, 400),
        margin_bottom=5mm,
        annotation=(1.9, 0.9, text(text_, :black, :left)))
    return plt
end
