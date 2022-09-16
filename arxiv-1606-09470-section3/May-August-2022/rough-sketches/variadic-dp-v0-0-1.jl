# Julia 1606-09470

# Variadic sketch

# ***********************************************************************************
# This is the first version for which Zygote.jl produces some gradient values and
# does not break outright.
# ***********************************************************************************
# WE DON'T KNOW YET WHETHER THESE GRADIENT VALUES ARE CORRECT - this is a warning
#                                                               to the reader
# ***********************************************************************************
# If one is only doing one iteration when computing loss, 
# then only immediately adjacent weights are involved in 
# non-zero gradient components. 
# However, with more iterations this backpropagates rapidly
# ***********************************************************************************

# In the spirit of "https://github.com/jsa-aerial/DMM",
# but this is DMM Lite, following "pseudo-code.md"

# We have to use non-persistent dictionaries at the moment
# (see "../persistency.md" for details)

# We have a choice of using mutable or immutable discipline,
# as noted in "pseudo-code.md"

# I am inclined to use mutable discipline at the moment, given
# that the dictionaries are non-persistent.

# But this means the following: not only activation functions
# cannot mutate their arguments, they cannot pass mutable parts of them
# downstream, if there is a danger for them to be mutated later.

# So, no sharing, no reuse, no incremental computations at the moment.

# We'll review this choice eventually.

using Flux
using Zygote

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
	
Neuron(f) = Neuron(f, Dict{String, Dict{String, Float32}}(), Dict{String, Dict{String, Float32}}())

function up_movement!(all_neurons::Dict{String, Neuron})
    for neuron in values(all_neurons)
	    # println(neuron)
		# println("INPUT DICT: ", neuron.input_dict)
		# println("AFTER FUNCTION APPLICATION: ", neuron.f(neuron.input_dict))
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end

function down_movement!(all_neurons::Dict{String, Neuron}, 
                        network_matrix::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    for neuron_name in keys(network_matrix)
        if haskey(all_neurons, neuron_name) # correctness, and there is a problem that if a neuron disappears from a matrix
                                            # altogether, then its input is not updates at all (in reality it should become
                                            # inactive in such a situation, but that's for a subsequent version)
            neuron = all_neurons[neuron_name]
            neuron.input_dict = Dict{String, Dict{String, Float32}}()
            for field_name in keys(network_matrix[neuron_name])
                matrix_row = network_matrix[neuron_name][field_name]
                neuron.input_dict[field_name] = Dict{String, Float32}()
                for resulting_neuron_name in keys(matrix_row)
                    if haskey(all_neurons, resulting_neuron_name)
                        resulting_neuron = all_neurons[resulting_neuron_name]
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
    println("timer: ", t)
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
    println("left: ", n1, " right: ", n2)
    println()
    Dict{String, Dict{String, Float32}}()
end
											
# -------------------------------------------------------------------------
											
# Let's run some manually crafted networks, see that this platform works

# -------------------------------------------------------------------------

all_neurons = Dict{String, Neuron}()

# input accumulator circuit

all_neurons["timer"] = Neuron(timer_add_one)
all_neurons["input"] = Neuron(input_dummy)
all_neurons["accum"] = Neuron(sum_dicts)

# circuit detecting duplicate characters

all_neurons["norm"] = Neuron(max_norm)
all_neurons["const_1"] = Neuron(const_1)
all_neurons["compare"] = Neuron(compare_scalars)

# circuit detecting end-of-fragment symbol

all_neurons["eos"] = Neuron(const_end)
all_neurons["dot"] = Neuron(dot_product_dicts)

# output neuron

all_neurons["output"] = Neuron(output_dummy)

network_matrix = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()

function link!(to_neuron, to_field, from_neuron, from_field, weight = 1.0f0) # additive!
    d = network_matrix
    for s in [to_neuron, to_field, from_neuron]
        if !haskey(d, s)
            d[s] = Dict()
        end
        d = d[s]
    end
    d[from_field] = get_N(d, from_field) + weight # additive to what was there, if anything!
end

# input accumulator circuit

link!("timer", "timer", "timer", "timer")
link!("input", "timer", "timer", "timer")
link!("accum", "dict-1", "accum", "dict") # this is for accumulation
link!("accum", "dict-2", "input", "char") # this feeds from the input to the accumulator

# circuit detecting duplicate characters

link!("norm", "dict", "accum", "dict")
link!("compare", "dict-1", "norm", "norm")
link!("compare", "dict-2", "const_1", "const_1")

# circuit detecting end-of-fragment symbol

link!("dot", "dict-1", "accum", "dict")
link!("dot", "dict-2", "eos", "char")

# output neuron

link!("output", "dict-1", "compare", "true")
link!("output", "dict-2", "dot", "dot")

# traditional DMM execution order is counter-intuitive: one starts with down-movement first
# (one starts with linear combination to form neuron inputs, then proceeds)
# this might not matter for DMM Lite, but I'll keep it this way for uniformity

function two_stroke_cycle!(all_neurons::Dict{String, Neuron}, 
                           network_matrix::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    down_movement!(all_neurons, network_matrix)
    up_movement!(all_neurons)
end

# and to run one might wish to use sleep(1) between cycles to slow it down and see what's happening
for i in 1:150
    two_stroke_cycle!(all_neurons, network_matrix)
    println("timer:   ", all_neurons["timer"].output_dict)
    println("input:   ", all_neurons["input"].output_dict)
    println("accum:   ", all_neurons["accum"].output_dict)
    println("norm:    ", all_neurons["norm"].output_dict)
    println("compare: ", all_neurons["compare"].output_dict)
    println("dot:     ", all_neurons["dot"].output_dict)
    println()
    println("OUTPUT: ", all_neurons["output"].input_dict)
    println()
    sleep(0.1)
end

square(x::Float32) = x*x

function loss(network_matrix::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    l = 0.0f0
	# for "for i in 1:5" the value we are getting is
	# timer: 159.0
    # grad: (Dict{Any, Any}("dot" => Dict{Any, Any}("dict-2" => Dict{Any, Any}("eos" => Dict{Any, Any}("char" => -200.0f0)), "dict-1" => Dict{Any, Any}("accum" => Dict{Any, Any}("dict" => -200.0f0))), "output" => Dict{Any, Any}("dict-2" => Dict{Any, Any}("dot" => Dict{Any, Any}("dot" => -250.0f0)), "dict-1" => Dict{Any, Any}("compare" => Dict{Any, Any}("true" => -240.0f0))), "compare" => Dict{Any, Any}("dict-2" => Dict{Any, Any}("const_1" => Dict{Any, Any}("const_1" => 48.0f0)), "dict-1" => Dict{Any, Any}("norm" => Dict{Any, Any}("norm" => -240.0f0))), "norm" => Dict{Any, Any}("dict" => Dict{Any, Any}("accum" => Dict{Any, Any}("dict" => -180.0f0)))),)
    for i in 1:5
        two_stroke_cycle!(all_neurons, network_matrix)
        # l += square(all_neurons["compare"].output_dict["true"][":number"] - 10.0f0)
		l += square(all_neurons["output"].input_dict["dict-1"][":number"] - 10.0f0)
		l += square(all_neurons["output"].input_dict["dict-2"][":number"] - 10.0f0)
    end
    l
end

this_loss = loss(network_matrix)

println("loss: ", this_loss)

println("Computing gradient")


# try

    this_grad = gradient(loss, network_matrix)

#catch e
#    println("caught!", e)
#end

println("grad: ", this_grad)




