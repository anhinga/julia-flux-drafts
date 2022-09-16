# Julia 1606-09470

# Variadic sketch

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

zero_value = 0.0f0
one_value = 1.0f0
end_of_text = "."

function get_N(dict, k)
    get(dict, k, zero_value)
end

function get_D(dict, k)
    get(dict, k, Dict())
end

# -----------------------------------------------------

# "two-stroke engine", DMM Lite, following "pseudo-code.md"

# TODO: ADD TYPES

function add_term_to_dict!(dict_to_change, multiplier, dict_as_delta)
    for k in keys(dict_as_delta)
        dict_to_change[k] = get_N(dict_to_change, k) + multiplier*dict_as_delta[k]
    end
end

mutable struct Neuron
    f
    input_dict
    output_dict
end
	
Neuron(f) = Neuron(f, Dict(), Dict())

function up_movement!(all_neurons)
    for neuron in values(all_neurons)
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end

# old version
# function down_movement!(all_neurons, network_matrix)
#     for (neuron_name, neuron_itself) in all_neurons
#         for field in keys(neuron_itself.input_dict)
#             neuron_itself.input_dict[field] = Dict()
#             matrix_row = get_D(get_D(network_matrix, neuron_name), field)
#             for (resulting_neuron_name, resulting_neuron_itself) in all_neurons
#                for resulting_field in keys(resulting_neuron_itself.output_dict)
#                     # neuron.input_dict[field] += matrix-row[resulting_neuron][resulting_field]*resulting_neuron.output_dict[resulting_field]
#                     add_term_to_dict!(neuron_itself.input_dict[field], get_N(get_D(matrix_row, resulting_neuron_name), resulting_field), resulting_neuron_itself.output_dict[resulting_field])
#                     # here matrix_row[resulting_neuron][resulting_field] is a number
#                     # and neuron.input_dict[field] and resulting_neuron.output_dict[resulting_field]
#                     #     are vector-like objects (dictionaries of numbers)
#     end end end end # Clojure style
# end

function down_movement!(all_neurons, network_matrix)
    for neuron_name in keys(network_matrix)
        if haskey(all_neurons, neuron_name) # correctness, and there is a problem that if a neuron disappears from a matrix
                                            # altogether, then its input is not updates at all (in reality it should become
                                            # inactive in such a situation, but that's for a subsequent version)
            neuron = all_neurons[neuron_name]
            neuron.input_dict = Dict()
            for (field_name, matrix_row) in network_matrix[neuron_name]
                neuron.input_dict[field_name] = Dict()
                for resulting_neuron_name in keys(matrix_row)
                    resulting_neuron = get_D(all_neurons, resulting_neuron_name)
                    for (resulting_field_name, multiplier) in matrix_row[resulting_neuron_name]
                        # neuron.input_dict[field_name] += multiplier*resulting_neuron.output_dict[resulting_field_name]
                        add_term_to_dict!(neuron.input_dict[field_name], multiplier, get_D(resulting_neuron.output_dict, resulting_field_name))
    end end end end end # Clojure style
end

# -----------------------------------------------------

# I am sure this one can be written better
# d = d1 + d2 for dictionaries
#
function sum_dicts(x)
    d = deepcopy(get_D(x, "dict-1"))
    d2 = get_D(x, "dict-2")
    for k in keys(d2)
	    d[k] = get_N(d, k) + d2[k]
    end
    Dict("dict" => d)
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
function max_norm(x)
    d = get_D(x, "dict")
    s = zero_value
    for k in keys(d)
        s = max(s, abs(d[k]))
    end
    Dict("norm" => Dict(":number" => s))
end

# dot product of dictionaries
#
function dot_product_dicts(x)
    d1 = get_D(x, "dict-1")
    d2 = get_D(x, "dict-2")
    s = zero_value
    for k in keys(d1)
        s += d1[k] * get_N(d2, k)
    end
    Dict("dot" => Dict(":number" => s))
end

# ReLU (auxiliary function)
#
function custom_relu(x)
    max(zero_value, x)
end
		
# compare scalars, two outputs, differentiable (via ReLU)
#
function compare_scalars(x)
    d1 = get_D(x, "dict-1")
    d2 = get_D(x, "dict-2")
    n1 = get_N(d1, ":number")
    n2 = get_N(d2, ":number")
    Dict("true" => Dict(":number" => custom_relu(n1-n2)), "false" => Dict(":number" => custom_relu(n2-n1)))
end
		
# const number 1
#
function const_1(x)
    Dict("const_1" => Dict(":number" => one_value))
end

# const end_of_string or end_of_fragment
#
function const_end(x)
   d = Dict()
   d[end_of_text] = one_value
   Dict("char" => d)											
end
								
# plus one, for timer
#
function timer_add_one(x)
    d = get_D(x, "timer")
    n = get_N(d, ":number")
    Dict("timer" => Dict(":number" => n+1))
end
									
# input_dummy (from a fixed string, depending on the timer input)
#
function input_dummy(x)
    t = get_N(get_D(x, "timer"), ":number")
    println("timer: ", t)
    t = max(t, 0)
    s = "test string."
    d = Dict()
    if t%10 == 0
        i = min(round(Int, t÷10) + 1, lastindex(s))
        d[SubString(s, i, i)] = one_value
    end
    Dict("char" => d)
end
			
# output_dummy (just prints the inputs)
#
function output_dummy(x)
    d1 = get_D(x, "dict-1")
    d2 = get_D(x, "dict-2")
    n1 = get_N(d1, ":number")
    n2 = get_N(d2, ":number")
    println("left: ", n1, " right: ", n2)
    println()
end
											
# -------------------------------------------------------------------------
											
# Let's run some manually crafted networks, see that this platform works

# -------------------------------------------------------------------------

all_neurons = Dict()

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

network_matrix = Dict()

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

function two_stroke_cycle!(all_neurons, network_matrix)
    down_movement!(all_neurons, network_matrix)
    up_movement!(all_neurons)
end

# and to run one might wish to use sleep(1) between cycles to slow it down and see what's happening
for i in 1:150
    two_stroke_cycle!(all_neurons, network_matrix)
    println(all_neurons["timer"].output_dict)
    println(all_neurons["input"].output_dict)
    println(all_neurons["accum"].output_dict)
	println(all_neurons["norm"].output_dict)
	println(all_neurons["compare"].output_dict)
	println(all_neurons["dot"].output_dict)
    println()
    println("OUTPUT: ", all_neurons["output"].input_dict)
    println()
    sleep(0.5)
end




