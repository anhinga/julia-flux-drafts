# "feedforward transducer with locally recurrent elements"

# a feedforward machine with accumulators

# let's also separate unchangeable non-zero weights associated with accums
# and with the input cascade, and the changeable weights

trainable = DMM_Lite()

# input, constants, and output

add_neuron!(trainable, "timer", timer_add_one)
add_neuron!(trainable, "input", input_dummy)

add_neuron!(trainable, "const_1", const_1)
add_neuron!(trainable, "eos", const_end)

add_neuron!(trainable, "output", output_dummy)

# layers of interneurons

n_layers = 4

n_per_layer = 2 # neurons of each type per layer

# map from the name of interneuron types to their activation functions
interneuron_types = Dict("accum"=>sum_dicts, "norm"=>max_norm,
                         "compare"=>compare_scalars, "dot"=>dot_product_dicts, "id"=>id_transform)

for layer in 1:n_layers
    for k in 1:n_per_layer
        for neuron_type_name in ["accum", "norm", "compare", "dot", "id"]
            add_neuron!(trainable, neuron_type_name * "-" * string(layer) * "-" * string(k), 
                        interneuron_types[neuron_type_name])
        end
    end
end

# pseudo-full connectivity

# this is still manual

hard_link!(trainable, "timer", "timer", "timer", "timer")
hard_link!(trainable, "input", "timer", "timer", "timer")

# hard recurrent connectivity for all accumulators

for layer in 1:n_layers
    for k in 1:n_per_layer
        accum_name = "accum-" * string(layer) * "-" * string(k)
        hard_link!(trainable, accum_name, "dict-1", accum_name, "dict") 
    end
end

# activation function signatures:
#
#     sum_dict: dict-1, dict-2 => dict (we reserve "dict-1" for "accum")
#     max_norm: dict => norm
#     dot_product: dict-1, dict-2 => dot (we are pondering making "dict-1" and "dict-2" asymmetric eventually)
#     compare_scalars: dict-1, dict-2 => true, false
#
#     id_transform: * => * (can take any number of arguments, so we need to select its connectivity)
#
#     const_1: () => const_1
#     const_end: () => char
#
#     timer: irrelevant (does not participate in soft links in the current experiments)
#     input_dummy: timer => char
#     output_dummy: dict-1, dict-2 => ()

# let's define layer structure

output_of_initial_layer = [("input", "char")]

input_of_final_layer = [("output", "dict-1"), ("output", "dict-2")]

output_of_constants = [("eos", "char"), ("const_1", "const_1")]

input_of_interlayer_template = [("accum", "dict-2"), ("norm", "dict"), 
                                ("dot", "dict-1"), ("dot", "dict-2"), 
                                ("compare", "dict-1"), ("compare", "dict-2"),
                                ("id", "dict-1"), ("id", "dict-2"), ("id", "dict"), 
                                ("id", "norm"), ("id", "dot"), ("id", "true"), ("id", "false")]

output_of_interlayer_template = [("accum", "dict"), ("norm", "norm"), ("dot", "dot"),
                                 ("compare", "true"), ("compare", "false"),
                                 ("id", "dict-1"), ("id", "dict-2"), ("id", "dict"), 
                                 ("id", "norm"), ("id", "dot"), ("id", "true"), ("id", "false")]

# mylayer = 2

# myk = 3

# map(x->(x[1]*"-"*string(mylayer)*"-"*string(myk), x[2]), output_of_interlayer_template)
# 12-element Vector{Tuple{String, String}}:
# ("accum-2-3", "dict")
# ("norm-2-3", "norm")
# ("dot-2-3", "dot")
# ("compare-2-3", "true")
# ("compare-2-3", "false")
# ("id-2-3", "dict-1")
# ("id-2-3", "dict-2")
# ("id-2-3", "dict")
# ("id-2-3", "norm")
# ("id-2-3", "dot")
# ("id-2-3", "true")
# ("id-2-3", "false")

# with respect to the output of constants, the options are to make constants available at every layer,
# including the output of the initial layer, or to make sure that id_transform passes those fields through

# we'll start with making that available to every layer

function form_input_interlayer(layer)
    vcat(map(k->map(x->(x[1]*"-"*string(layer)*"-"*string(k), x[2]), input_of_interlayer_template), 1:n_per_layer)...)
end

function form_output_interlayer(layer)
    vcat(map(k->map(x->(x[1]*"-"*string(layer)*"-"*string(k), x[2]), output_of_interlayer_template), 1:n_per_layer)...)
end

# -------

outputs_this = Dict()

inputs_next = Dict()

outputs_this[1] = output_of_initial_layer

for layer in 2:n_layers
    outputs_this[layer] = vcat(output_of_constants, form_output_interlayer(layer-1))
end

outputs_this[n_layers+1] = form_output_interlayer(n_layers)

for layer in 1:n_layers
    inputs_next[layer] = form_input_interlayer(layer)
end

inputs_next[n_layers+1] = input_of_final_layer

# -------

# we must define a set of neuron inputs and a set of neuron outputs
# and connect all outputs to all inputs in the appropriate layers
# with pseudo-random strength (we only need to do that occasionally,
# because generally speaking, we'd like to serialize and retrieve
# for reproducibility)

Random.seed!(123)
normal_dist_0_1 = Normal()

for layer in 1:n_layers+1
    for (input_neuron, input_field) in inputs_next[layer]
        for (output_neuron, output_field) in outputs_this[layer]
            link!(trainable, input_neuron, input_field, output_neuron, output_field, Float32(0.01*rand(normal_dist_0_1)))
end end end

println("FEEDFORWARD with local recurrence INCLUDED")


