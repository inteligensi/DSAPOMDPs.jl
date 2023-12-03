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
    actions = []
    states = []
    state_primes = []
    observations = []
    times = 0
    t = 0
    isRecover = false
    needToRecover = false
    for step in eachstep(h)
        t += 1
        rsum += (step.r * disc)
        disc *= discount(P)
        push!(actions, step.a)
        push!(states, step.s)
        push!(observations, step.o)
        push!(state_primes, step.sp)

        # Check if patient needs to be recovered
        if !needToRecover && t == 1 && (step.s.ane || step.s.avm || step.s.occ)
            needToRecover = true
        end

        # Check if patient has been recovered
        if !step.sp.ane && !step.sp.avm && !step.sp.occ && !isRecover && needToRecover
            times = step.sp.time
            isRecover = true
        end
    end

    return rsum, actions, states, state_primes, observations, times
end


function state_sub2ind(dims::Tuple, i1, i2, i3, i4)
    return ((i4 - 1) * prod(dims[1:3]) +
            (i3 - 1) * prod(dims[1:2]) +
            (i2 - 1) * dims[1] +
             i1)
 end

 function evaluate_policies_replication(sim, pomdp, policies, up, b0, num_reps, max_steps)
    results = Array{Float64}(undef, length(policies), num_reps)
    times = Array{Float64}(undef, length(policies), num_reps)

    for i in 1:num_reps
        s0 = rand(b0)
        @show s0
        for (j, policy) in enumerate(policies)
            file_path = "output-rep-$(j).txt"
            if isfile(file_path)
                file = open(file_path, "a")
            else
                file = open(file_path, "w")
            end

            h_policy = simulate(sim, pomdp, policy, up, b0, s0)

            results[j, i], actions, states, state_primes, observations, times[j, i] = compute_rdisc(pomdp, h_policy)
            println("==========simulation $i for policy $j ==========")
            println("states = $(join([s for s in states], ", "))")
            println("state_primes = $(join([sp for sp in state_primes], ", "))")
            println("observations = $(join([o for o in observations], ", "))")
            println("actions = $(join([a isa Symbol ? String(a)[11:end] : a for a in actions], ", "))")
            println("rsum = $(results[j, i])")
            println("times = $(times[j, i])")
    
            println(file, "==========simulation $i for policy $j ==========")
            println(file, "states = $(join([s for s in states], ", "))")
            println(file, "state_primes = $(join([sp for sp in state_primes], ", "))")
            println(file, "observations = $(join([o for o in observations], ", "))")
            println(file, "actions = $(join([a isa Symbol ? String(a)[11:end] : a for a in actions], ", "))")
            println(file, "rsum = $(results[j, i])")
            println(file, "times = $(times[j, i])")

            close(file)
        end
    end

    means = mean(results, dims=2)
    stds = std(results, dims=2)
    t_means = mean(times, dims=2)
    t_stds = std(times, dims=2)
    return (means=means, stds=stds, t_means=t_means, t_stds=t_stds)
end