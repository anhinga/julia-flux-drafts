# Julia 1606-09470

# Variadic sketch

# We are going to have two networks in this file, and we'll try to
# train one of them to imitate another (one needs to be careful not
# cheat accidently in such a setup)

using Flux
using Zygote
using Random, Distributions

zero_value = 0.0f0
one_value = 1.0f0
end_of_text = "."

function get_N(dict::Dict{String, Float32}, k::String)
    if haskey(dict, k)
	   return dict[k]
	else
	   return zero_value
    end
end

function get_D(dict::Dict{String, Dict{String, Float32}}, k::String)
    if haskey(dict, k)
	   return dict[k]
	else
	   return Dict{String, Float32}()
    end
end

#function get_D(dict::Dict{String, Dict{String, Dict{String, Float32}}}, k::String)
#    get(dict, k, Dict{String, Dict{String, Float32}}())
#end

#function get_D(dict::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, k::String)
#    get(dict, k, Dict{String, Dict{String, Dict{String, Float32}}}())
#end

# -----------------------------------------------------

# "two-stroke engine", DMM Lite, following "pseudo-code.md"

function add_term_to_dict!(dict_to_change::Dict{String, Float32}, multiplier::Float32, dict_as_delta::Dict{String, Float32})
    for k in keys(dict_as_delta)
        dict_to_change[k] = get_N(dict_to_change, k) + multiplier*dict_as_delta[k]
    end
end

mutable struct Neuron
    f::Function
    input_dict::Dict{String, Dict{String, Float32}}
    output_dict::Dict{String, Dict{String, Float32}}
end


#struct DMM_Lite
#    neurons::Dict{String, Neuron}
#    network_matrix::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}
#end
	
Neuron(f) = Neuron(f, Dict{String, Dict{String, Float32}}(), Dict{String, Dict{String, Float32}}())

#DMM_Lite() = DMM_Lite(Dict{String, Neuron}(), 
#                      Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

# For some reason, "struct DMM_Lite" representation causes the gradient to be incorrect,
# namely to return "nothing". I don't want to try to debug this, let's just representation
# this via Dict (it is weird that "mutable struct" elsewhere was fine, but not this -
#                TODO: revisit this issue)

DMM_Lite() = Dict("neurons"=>Dict{String, Neuron}(),
                  "network_matrix"=>Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

DMM_Lite_ = Dict{String, Dict{String}} # type

function up_movement!(dmm_lite::DMM_Lite_)
    for neuron in values(dmm_lite["neurons"])
	    # println(neuron)
		# println("INPUT DICT: ", neuron.input_dict)
		# println("AFTER FUNCTION APPLICATION: ", neuron.f(neuron.input_dict))
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end

function down_movement!(dmm_lite::DMM_Lite_)
    for neuron_name in keys(dmm_lite["network_matrix"])
        if haskey(dmm_lite["neurons"], neuron_name) # correctness, and there is a problem that if a neuron disappears from a matrix
                                            # altogether, then its input is not updates at all (in reality it should become
                                            # inactive in such a situation, but that's for a subsequent version)
            neuron = dmm_lite["neurons"][neuron_name]
            neuron.input_dict = Dict{String, Dict{String, Float32}}()
            for field_name in keys(dmm_lite["network_matrix"][neuron_name])
                matrix_row = dmm_lite["network_matrix"][neuron_name][field_name]
                neuron.input_dict[field_name] = Dict{String, Float32}()
                for resulting_neuron_name in keys(matrix_row)
                    if haskey(dmm_lite["neurons"], resulting_neuron_name)
                        resulting_neuron = dmm_lite["neurons"][resulting_neuron_name]
                        for resulting_field_name in keys(matrix_row[resulting_neuron_name])
                            multiplier = matrix_row[resulting_neuron_name][resulting_field_name]
                            # neuron.input_dict[field_name] += multiplier*resulting_neuron.output_dict[resulting_field_name]
                            add_term_to_dict!(neuron.input_dict[field_name], multiplier, get_D(resulting_neuron.output_dict, resulting_field_name))
    end end end end end end # Clojure style
end

# -----------------------------------------------------

# I am sure this one can be written better
# d = d1 + d2 for dictionaries
#
function sum_dicts(x::Dict{String, Dict{String, Float32}})
    d::Dict{String, Float32} = deepcopy(get_D(x, "dict-1"))
    d2::Dict{String, Float32} = get_D(x, "dict-2")
    for k in keys(d2)
        d[k] = get_N(d, k) + d2[k]
    end
    Dict{String, Dict{String, Float32}}("dict" => d)
end

# julia> x = Dict("dict-1" => Dict("a" => 5.0f0, "b" => -3.0f0))
# Dict{String, Dict{String, Float32}} with 1 entry:
#   "dict-1" => Dict("b"=>-3.0, "a"=>5.0)
#
# julia> x["dict-2"] = Dict("b" => 3.0f0, "c" => 2.0f0)
# Dict{String, Float32} with 2 entries:
#   "c" => 2.0
#   "b" => 3.0
#
# julia> x
# Dict{String, Dict{String, Float32}} with 2 entries:
#   "dict-2" => Dict("c"=>2.0, "b"=>3.0)
#   "dict-1" => Dict("b"=>-3.0, "a"=>5.0)
#
# julia> sum_dicts(x)
# Dict{String, Dict{String, Float32}} with 1 entry:
#   "dict" => Dict("c"=>2.0, "b"=>0.0, "a"=>5.0)
#  
# It would be better if "b"=>0.0 disappear altogether, but this would do for now
#
# sum_dicts can be used as the activation function of an accumulator
		    
# max of abs of dictionary values
#
function max_norm(x::Dict{String, Dict{String, Float32}})
    d::Dict{String, Float32} = get_D(x, "dict")
    s::Float32 = zero_value
    for k in keys(d)
        s = max(s, abs(d[k]))
    end
    Dict{String, Dict{String, Float32}}("norm" => Dict(":number" => s))
end

# dot product of dictionaries
#
function dot_product_dicts(x::Dict{String, Dict{String, Float32}})
    d1::Dict{String, Float32} = get_D(x, "dict-1")
    d2::Dict{String, Float32} = get_D(x, "dict-2")
    s::Float32 = zero_value
    for k in keys(d1)
        s += d1[k] * get_N(d2, k)
    end
    if abs(s) > 100.0f0
        s = sign(s)*100.0f0
    end
    Dict{String, Dict{String, Float32}}("dot" => Dict(":number" => s))
end

# ReLU (auxiliary function)
#
function custom_relu(x::Float32)
    max(zero_value, x)
end
		
# compare scalars, two outputs, differentiable (via ReLU)
#
function compare_scalars(x::Dict{String, Dict{String, Float32}})
    d1::Dict{String, Float32} = get_D(x, "dict-1")
    d2::Dict{String, Float32} = get_D(x, "dict-2")
    n1::Float32 = get_N(d1, ":number")
    n2::Float32 = get_N(d2, ":number")
    Dict{String, Dict{String, Float32}}("true" => Dict(":number" => custom_relu(n1-n2)), "false" => Dict(":number" => custom_relu(n2-n1)))
end
		
# const number 1
#
function const_1(x::Dict{String, Dict{String, Float32}})
    Dict{String, Dict{String, Float32}}("const_1" => Dict(":number" => one_value))
end

# const end_of_string or end_of_fragment
#
function const_end(x::Dict{String, Dict{String, Float32}})
   d::Dict{String, Float32} = Dict{String, Float32}()
   d[end_of_text] = one_value
   Dict{String, Dict{String, Float32}}("char" => d)											
end
								
# plus one, for timer
#
function timer_add_one(x::Dict{String, Dict{String, Float32}})
    d::Dict{String, Float32} = get_D(x, "timer")
    n::Float32 = get_N(d, ":number")
    Dict{String, Dict{String, Float32}}("timer" => Dict(":number" => n+1))
end
									
# input_dummy (from a fixed string, depending on the timer input)
#
function input_dummy(x::Dict{String, Dict{String, Float32}})
    t::Float32 = get_N(get_D(x, "timer"), ":number")
    printlog(io, "(driving input) timer: ", t)
    t = max(t, 0)
    s::String = "test string."
    d::Dict{String, Float32} = Dict{String, Float32}()
    if t%10 == 0
        i = min(round(Int, t÷10) + 1, lastindex(s))
        Zygote.@ignore d[SubString(s, i, i)] = one_value
    end
    Dict{String, Dict{String, Float32}}("char" => d)
end
			
# output_dummy (just prints the inputs)
#
function output_dummy(x::Dict{String, Dict{String, Float32}})
    d1 = get_D(x, "dict-1")
    d2 = get_D(x, "dict-2")
    n1 = get_N(d1, ":number")
    n2 = get_N(d2, ":number")
    printlog(io, "(getting on output) left: ", n1, " right: ", n2)
    Dict{String, Dict{String, Float32}}()
end

# -------------------------------------------------------------------------

# DMM_Lite utilities

# -------------------------------------------------------------------------

function add_neuron!(dmm_lite::DMM_Lite_, neuron_name::String, activation::Function)
    dmm_lite["neurons"][neuron_name] = Neuron(activation)
end

function reset_network_dicts!(dmm_lite::DMM_Lite_)
    for v in values(dmm_lite["neurons"])
        v.input_dict = Dict{String, Dict{String, Float32}}()
        v.output_dict = Dict{String, Dict{String, Float32}}()
    end
end

function link!(dmm_lite::DMM_Lite_,
               to_neuron::String, to_field::String, from_neuron::String, from_field::String, 
               weight::Float32 = 1.0f0) # additive!
    d = dmm_lite["network_matrix"]
    for s in [to_neuron, to_field, from_neuron]
        if !haskey(d, s)
            d[s] = Dict()
        end
        d = d[s]
    end
    d[from_field] = get_N(d, from_field) + weight # additive to what was there, if anything!
end
											
# -------------------------------------------------------------------------
											
# Let's run some manually crafted networks, see that this platform works

# -------------------------------------------------------------------------

# a handcrafted DMM

handcrafted = DMM_Lite()

# input accumulator circuit

add_neuron!(handcrafted, "timer", timer_add_one)
add_neuron!(handcrafted, "input", input_dummy)
add_neuron!(handcrafted, "accum", sum_dicts)

# circuit detecting duplicate characters

add_neuron!(handcrafted, "norm", max_norm)
add_neuron!(handcrafted, "const_1", const_1)
add_neuron!(handcrafted, "compare", compare_scalars)

# circuit detecting end-of-fragment symbol

add_neuron!(handcrafted, "eos", const_end)
add_neuron!(handcrafted, "dot", dot_product_dicts)

# output neuron

add_neuron!(handcrafted, "output", output_dummy)

# input accumulator circuit

link!(handcrafted, "timer", "timer", "timer", "timer")
link!(handcrafted, "input", "timer", "timer", "timer")
link!(handcrafted, "accum", "dict-1", "accum", "dict") # this is for accumulation
link!(handcrafted, "accum", "dict-2", "input", "char") # this feeds from the input to the accumulator

# circuit detecting duplicate characters

link!(handcrafted, "norm", "dict", "accum", "dict")
link!(handcrafted, "compare", "dict-1", "norm", "norm")
link!(handcrafted, "compare", "dict-2", "const_1", "const_1")

# circuit detecting end-of-fragment symbol

link!(handcrafted, "dot", "dict-1", "accum", "dict")
link!(handcrafted, "dot", "dict-2", "eos", "char")

# output neuron

link!(handcrafted, "output", "dict-1", "compare", "true")
link!(handcrafted, "output", "dict-2", "dot", "dot")

# ------------------------------------------------------------------------------------

# a pseudo-fully-connected DMM

trainable = DMM_Lite()

# input, constants, and output

add_neuron!(trainable, "timer", timer_add_one)
add_neuron!(trainable, "input", input_dummy)

add_neuron!(trainable, "const_1", const_1)
add_neuron!(trainable, "eos", const_end)

add_neuron!(trainable, "output", output_dummy)

# N copies of each time of interneurons

n_copies = 5

# map from the name of interneuron types to their activation functions
interneuron_types = Dict("accum"=>sum_dicts, "norm"=>max_norm,
                         "compare"=>compare_scalars, "dot"=>dot_product_dicts)

for i in 1:n_copies
    for neuron_type_name in ["accum", "norm", "compare", "dot"]
        add_neuron!(trainable, neuron_type_name * "-" * string(i), 
                    interneuron_types[neuron_type_name])
    end
end

# pseudo-full connectivity

# this is still manual

link!(trainable, "timer", "timer", "timer", "timer")
link!(trainable, "input", "timer", "timer", "timer")

# we must define a set of neuron inputs and a set of neuron outputs
# and connect all outputs to all inputs in the respective sets
# with pseudo-random strength (we only need to do that occasionally,
# because generally speaking, we'd like to serialize and retrieve
# for reproducibility)

# every input and output is a pair of neuron name and field name

# for unique neurons (not connecting the timer, it's auxiliary)

set_of_inputs = [("output", "dict-1"), ("output", "dict-2")]

set_of_outputs = [("input", "char"), ("eos", "char"), ("const_1", "const_1")]

# for interneurons (we have a choice of connecting to active fields only
# or to all fields whatsoever, mentioned in this setup; let's start with
# all fields, as it is logically simpler to describe, and regularization
# during training should get rid of all the spurious connections)

for i in 1:n_copies
    for neuron_type_name in ["accum", "norm", "compare", "dot"]
        for field_name in ["dict", "dict-1", "dict-2", "norm", "dot", "true", "false"]
            append!(set_of_inputs, [(neuron_type_name * "-" * string(i), field_name)]) 
            append!(set_of_outputs, [(neuron_type_name * "-" * string(i), field_name)])
        end
    end
end

Random.seed!(123)
normal_dist_0_1 = Normal()

# actual pseudo-full topology with pseudo-random connectivity weights
#
for (input_neuron, input_field) in set_of_inputs
    for (output_neuron, output_field) in set_of_outputs
        link!(trainable, input_neuron, input_field, output_neuron, output_field, Float32(0.2*rand(normal_dist_0_1)))
    end
end

# ------------------------------------------------------------------------------------


# traditional DMM execution order is counter-intuitive: one starts with down-movement first
# (one starts with linear combination to form neuron inputs, then proceeds)
# this might not matter for DMM Lite, but I'll keep it this way for uniformity

function two_stroke_cycle!(dmm_lite::DMM_Lite_)
    down_movement!(dmm_lite)
    up_movement!(dmm_lite)
end

#=
# and to run one might wish to use sleep(1) between cycles to slow it down and see what's happening
for i in 1:150
    two_stroke_cycle!(handcrafted)
    println("timer:   ", handcrafted["neurons"]["timer"].output_dict)
    println("input:   ", handcrafted["neurons"]["input"].output_dict)
    println("accum:   ", handcrafted["neurons"]["accum"].output_dict)
    println("norm:    ", handcrafted["neurons"]["norm"].output_dict)
    println("compare: ", handcrafted["neurons"]["compare"].output_dict)
    println("dot:     ", handcrafted["neurons"]["dot"].output_dict)
    println()
    println("OUTPUT: ", handcrafted["neurons"]["output"].input_dict)
    println()
    sleep(0.1)
end

for i in 1:150
    two_stroke_cycle!(trainable)
    println("timer:   ", trainable["neurons"]["timer"].output_dict)
    println("input:   ", trainable["neurons"]["input"].output_dict)
    println()
    println("OUTPUT: ", trainable["neurons"]["output"].input_dict)
    println()
    sleep(0.1)
end
=#

square(x::Float32) = x*x

function loss(dmm_lite::DMM_Lite_)
    l = 0.0f0
    for i in 1:35 # 1:12 # 1:35 ok speed (use 0.001 reg mult), 
	              # 1:140 works, but the slowdown is spectacular (use 0.01 reg mult)
        two_stroke_cycle!(dmm_lite)
        two_stroke_cycle!(handcrafted)
        target_1 = get_N(handcrafted["neurons"]["output"].input_dict["dict-1"], ":number")
        target_2 = get_N(handcrafted["neurons"]["output"].input_dict["dict-2"], ":number")
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-1"], ":number") - target_1)
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-2"], ":number") - target_2)
    end
    regularization = 0.0f0
    for i in keys(dmm_lite["network_matrix"])
	    if i != "timer" && i != "input"
            for j in keys(dmm_lite["network_matrix"][i])
                for m in keys(dmm_lite["network_matrix"][i][j])
                    for n in keys(dmm_lite["network_matrix"][i][j][m])
                        regularization += abs(dmm_lite["network_matrix"][i][j][m][n]) + 
						                  10.0f0 * square(dmm_lite["network_matrix"][i][j][m][n])
    end end end end end
    printlog_v(io, "prereg loss ", l, " regularization ", regularization)
    l += 0.001f0 * regularization	
	printlog_v(io, "loss ", l)
    l
end

#this_loss = loss(handcrafted)
#this_loss = loss(trainable)

#println("loss: ", this_loss)

println("Computing gradient")


# try

#    this_grad = gradient(loss, handcrafted)

# this_grad = gradient(loss, trainable)

#catch e
#    println("caught!", e)
#end

# println("grad: ", this_grad[1]["network_matrix"])

include("ADAM-v0-0-1.jl")

# convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
#         this_grad[1]["network_matrix"])



