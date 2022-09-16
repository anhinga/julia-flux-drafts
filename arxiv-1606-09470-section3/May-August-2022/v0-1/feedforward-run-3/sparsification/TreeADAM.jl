# Initial sketch of ADAM optimizer

#=
A variation for dictionary-shaped data.

Looking at the following version for vector-shaped data:

https://github.com/FluxML/Flux.jl/blob/master/src/optimise/optimisers.jl

"""
    ADAM(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $EPS)
[ADAM](https://arxiv.org/abs/1412.6980) optimiser.
# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
# Examples
```julia
opt = ADAM()
opt = ADAM(0.001, (0.9, 0.8))
```
"""
mutable struct ADAM <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
ADAM(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = EPS) = ADAM(η, β, ϵ, IdDict())
ADAM(η::Real, β::Tuple, state::IdDict) = ADAM(η, β, EPS, state)

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, x) do
      (zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
  βp .= βp .* β

  return Δ
end
=#

# This is actually done for the generality of complex parameters 
# ("conj" is used in the calculation of the square of the gradient); 
# we are going to ignore that for now - reals only

# I am also going to be conservative in this first sketch
# All these overloaded operators and greek letters are very cool,
# but for the first sketch I am going to use an old-fashioned style.

# AND IN PARTICULAR I DON'T DO POLYMORPHIC TYPING EVEN WHERE IT IS
# QUITE NATURAL AND CALLED FOR (this will come in subsequent versions
# and refactor; what I am doing here is non-standard enough, so one
# problem at a time)

# We have this function for plain dictionaries:
# function add_term_to_dict!(dict_to_change::Dict{String, Float32}, multiplier::Float32, dict_as_delta::Dict{String, Float32})
#     for k in keys(dict_as_delta)
#         dict_to_change[k] = get_N(dict_to_change, k) + multiplier*dict_as_delta[k]
#     end
# end
#
# We need something like that either for arbitrary nested dictionaries,
# or, at a minimum, for the network matrices represented by dictionaries
#
# Let's ponder this a bit... Full variadic set-up has some nuances, as in
#
# https://github.com/jsa-aerial/DMM/blob/master/src/dmm/core.clj
#
# Do we want something simpler to start with?
#
# Here is a dumb way, non-variadic. Assume that we also see
#
# function get_D(dict::Dict{String, Dict{String, Float32}}, k::String)
#    if haskey(dict, k)
#        return dict[k]
#    else
#        return Dict{String, Float32}()
#    end
# end

# Note: unlikely to be differentiable in Zygote, although needs to be double-checked

function get_D!(dict::Dict{String, Dict{String, Float32}}, k::String)
    get!(dict, k, Dict{String, Float32}())
end

function get_D!(dict::Dict{String, Dict{String, Dict{String, Float32}}}, k::String)
    get!(dict, k, Dict{String, Dict{String, Float32}}())
end

function get_D!(dict::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, k::String)
    get!(dict, k, Dict{String, Dict{String, Dict{String, Float32}}}())
end

function add_term_to_dict!(dict_to_change::Dict{String, Dict{String, Float32}}, 
                           multiplier::Float32, 
                           dict_as_delta::Dict{String, Dict{String, Float32}})
    for k in keys(dict_as_delta)
        add_term_to_dict!(get_D!(dict_to_change, k), multiplier, dict_as_delta[k])
    end
end

function add_term_to_dict!(dict_to_change::Dict{String, Dict{String, Dict{String, Float32}}}, 
                           multiplier::Float32, 
                           dict_as_delta::Dict{String, Dict{String, Dict{String, Float32}}})
    for k in keys(dict_as_delta)
        add_term_to_dict!(get_D!(dict_to_change, k), multiplier, dict_as_delta[k])
    end
end

function add_term_to_dict!(dict_to_change::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, 
                           multiplier::Float32, 
                           dict_as_delta::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    for k in keys(dict_as_delta)
        add_term_to_dict!(get_D!(dict_to_change, k), multiplier, dict_as_delta[k])
    end
end

function mult_dict!(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, multiplier::Float32)
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    x[i][j][m][n] *= multiplier
    end end end end
	x
end

# NOT RELYING ON WHICH OBJECTS ARE STABLE AND WHICH ARE UNSTABLE TO RESET THE STATE.

# JUST CREATE A NEW OPTIMIZER TO SET THE NEW STATE, THESE OBJECTS DON'T COST ANYTHING.

# Doing this type-specific for version 0.0.1, not polymorphic.
# So this only works for this kind of "network_matrix"

const EPS = 1.0f-8

#=
function zerocopy(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    y = deepcopy(x)
    for i in keys(y)
        for j in keys(y[i])
            for m in keys(y[i][j])
                for n in keys(y[i][j][m])
                    y[i][j][m][n] = 0.0f0
    end end end end
	y
end
=#

function min_max_dict(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    min_v = 1.0f8
	max_v = -1.0f8
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    v = x[i][j][m][n]
                    if v < min_v
                        min_v = v
                    end
                    if v > max_v
                        max_v = v
                    end
    end end end end
	(min_v, max_v)
end

mutable struct TreeADAM
    eta::Float32
    beta::Tuple{Float32,Float32}
    epsilon::Float32
    beta_power::Vector{Float32}
    mt::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}
    vt::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}
end
TreeADAM(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
         η::Float32 = 0.001f0, β::Tuple{Float32,Float32} = (0.9f0, 0.999f0), ϵ::Float32 = EPS) = 
    TreeADAM(η, β, ϵ, [β[1], β[2]], typeof(x)(), typeof(x)())

function adam_step!(o::TreeADAM, x, Δ) 
    η, β = o.eta, o.beta

    # @. mt = β[1] * mt + (1 - β[1]) * Δ
    mult_dict!(o.mt, β[1])
    add_term_to_dict!(o.mt, (1.0f0 - β[1]), Δ)
  
    # @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    mult_dict!(o.vt, β[2])
    grad_squares = deepcopy(Δ)
    for i in keys(grad_squares)
        for j in keys(grad_squares[i])
            for m in keys(grad_squares[i][j])
                for n in keys(grad_squares[i][j][m])
                    grad_squares[i][j][m][n] *= grad_squares[i][j][m][n]
    end end end end
    add_term_to_dict!(o.vt, (1.0f0-β[2]), grad_squares)

    # @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
    # Compute this first: mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon)
    corrected_term = deepcopy(o.mt)
    for i in keys(corrected_term)
        for j in keys(corrected_term[i])
            for m in keys(corrected_term[i][j])
                for n in keys(corrected_term[i][j][m])
                    corrected_term[i][j][m][n] = 
                        o.mt[i][j][m][n] / (1.0f0 - o.beta_power[1]) /
                        (sqrt(o.vt[i][j][m][n] / (1.0f0 - o.beta_power[2])) + o.epsilon)
    end end end end
   
    o.beta_power .*= β
	
    # return Δ # do we want that, or do we want to return new "x"? or just "corrected_term"?
    # making the choice to mutate "x" right here (revisit this when we go immutable route)
    add_term_to_dict!(x, -η, corrected_term)

    printlog(io, "beta_power ", o.beta_power, " eta ", η)
    (min_grad, max_grad) = min_max_dict(Δ)
    printlog(io, "min grad ", min_grad, " max grad ", max_grad)
    (min_mt, max_mt) = min_max_dict(o.mt)
    printlog(io, "min mt ", min_mt, " max mt ", max_mt)
    (min_vt, max_vt) = min_max_dict(o.vt)
    println(io, "min vt ", min_vt, " max vt ", max_vt)
    (min_correction, max_correction) = min_max_dict(corrected_term)
    printlog(io, "min correction ", min_correction, " max correction ", max_correction)
    (min_x, max_x) = min_max_dict(x)
    printlog(io, "min weight ", min_x, " max weight ", max_x)

    x
end
