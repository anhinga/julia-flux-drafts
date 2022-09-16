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

hard_link!(trainable, "timer", "timer", "timer", "timer")
hard_link!(trainable, "input", "timer", "timer", "timer")

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



