using Random

include("DSAPOMDP.jl")
include("transition.jl")
include("utils.jl")

global P = DSAPOMDP()
global rng = MersenneTwister(1)
global s = initialstate(P)

for i in 1:30
    println(i)
    if isterminal(P, s)
        println("terminal at $i")
        break        
    end
    a = rand(actions(P))
    r = reward(P, s, a)
    obs = rand(rng, observation(P, s, a))
    sp = transition(P, s, a, rng)
    @show s, a, r, obs, sp
    s = deepcopy(sp)
end

isterminal(P, s)

