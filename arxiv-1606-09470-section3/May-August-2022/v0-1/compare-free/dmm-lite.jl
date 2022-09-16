# DMM_Lite (a simplified version of DMM 1.0)

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
                  "network_matrix"=>Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(),
                  "fixed_matrix"=>Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

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
    for neuron_name in union(keys(dmm_lite["network_matrix"]), keys(dmm_lite["fixed_matrix"]))
        if haskey(dmm_lite["neurons"], neuron_name) # correctness, and there is a problem that if a neuron disappears from a matrix
                                            # altogether, then its input is not updates at all (in reality it should become
                                            # inactive in such a situation, but that's for a subsequent version)
            neuron = dmm_lite["neurons"][neuron_name]
            neuron.input_dict = Dict{String, Dict{String, Float32}}()
            for matrix_name in ["network_matrix", "fixed_matrix"]
                if haskey(dmm_lite[matrix_name], neuron_name)
                    for field_name in keys(dmm_lite[matrix_name][neuron_name])
                        matrix_row = dmm_lite[matrix_name][neuron_name][field_name]
                        if !haskey(neuron.input_dict, field_name)
                            neuron.input_dict[field_name] = Dict{String, Float32}()
                        end
                        for resulting_neuron_name in keys(matrix_row)
                            if haskey(dmm_lite["neurons"], resulting_neuron_name)
                                resulting_neuron = dmm_lite["neurons"][resulting_neuron_name]
                                for resulting_field_name in keys(matrix_row[resulting_neuron_name])
                                    multiplier = matrix_row[resulting_neuron_name][resulting_field_name]
                                    # neuron.input_dict[field_name] += multiplier*resulting_neuron.output_dict[resulting_field_name]
                                    add_term_to_dict!(neuron.input_dict[field_name], multiplier, get_D(resulting_neuron.output_dict, resulting_field_name))
    end end end end end end end end # Clojure style
end

# traditional DMM execution order is counter-intuitive: one starts with down-movement first
# (one starts with linear combination to form neuron inputs, then proceeds)
# this might not matter for DMM Lite, but I'll keep it this way for uniformity

function two_stroke_cycle!(dmm_lite::DMM_Lite_)
    down_movement!(dmm_lite)
    up_movement!(dmm_lite)
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
    if abs(s) > 4.0f0
        s = sign(s)*4.0f0
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

# identity; because "down_movement" recreates a new input dictionaries,
#           and because other activation functions don't mutate their arguments
#           it should be OK to just pass the input through
# function id_transform(x::Dict{String, Dict{String, Float32}})
#     x
# end
#
# however, I don't feel comfortable doing that (I wish Zygote would work
#                                               for persistent dictionaries)
# what if a subsequent refactor makes an error and modifies the output
#                                                  somewhere downstream
function id_transform(x::Dict{String, Dict{String, Float32}})
    deepcopy(x)
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

function hard_link!(dmm_lite::DMM_Lite_,
               to_neuron::String, to_field::String, from_neuron::String, from_field::String, 
               weight::Float32 = 1.0f0) # additive!
    d = dmm_lite["fixed_matrix"]
    for s in [to_neuron, to_field, from_neuron]
        if !haskey(d, s)
            d[s] = Dict()
        end
        d = d[s]
    end
    d[from_field] = get_N(d, from_field) + weight # additive to what was there, if anything!
end
											