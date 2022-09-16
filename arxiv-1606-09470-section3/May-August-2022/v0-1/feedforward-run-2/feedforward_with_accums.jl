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
                         "compare"=>compare_scalars, "dot"=>dot_product_dicts)

for layer in 1:n_layers
    for k in 1:n_per_layer
        for neuron_type_name in ["accum", "norm", "compare", "dot"]
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
                                ("compare", "dict-1"), ("compare", "dict-2")]

output_of_interlayer_template = [("accum", "dict"), ("norm", "norm"), ("dot", "dot"),
                                 ("compare", "true"), ("compare", "false")]

function form_input_interlayer(layer)
    vcat(map(k->map(x->(x[1]*"-"*string(layer)*"-"*string(k), x[2]), input_of_interlayer_template), 1:n_per_layer)...)
end

function form_output_interlayer(layer)
    vcat(map(k->map(x->(x[1]*"-"*string(layer)*"-"*string(k), x[2]), output_of_interlayer_template), 1:n_per_layer)...)
end

# -------

# implementing with skip connections instead of id_transform

outputs_this = Dict()

inputs_next = Dict()

outputs_this[1] = vcat(output_of_initial_layer, output_of_constants)

for layer in 1:n_layers
    outputs_this[layer+1] = form_output_interlayer(layer)
end

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

# with all possible skip connections

for input_layer in 1:n_layers+1
    for (input_neuron, input_field) in inputs_next[input_layer]
        for output_layer in 1:input_layer
            for (output_neuron, output_field) in outputs_this[output_layer]
                link!(trainable, input_neuron, input_field, output_neuron, output_field, Float32(0.2*rand(normal_dist_0_1)))
end end end end

println("FEEDFORWARD with local recurrence and skip connections INCLUDED")


