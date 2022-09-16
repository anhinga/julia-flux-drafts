# Pseudo-code for variadic DMM Lite

DMM Lite: fixed configuration, no expansion, no taking advantage of sparsity (unlike DMM 1.0 in Clojure which is the real one)

Because this is DMM Lite, we can be simple minded in organizing the "two-stroke cycle", whereas one needs to be more careful
for the real thing

`append-or-replace` - append to the stack of such objects for the immutable version, replace such object for the mutable version

the access is to the top of the stack (the latest added object) for the immutable version

## Up movement

```
For neuron in all-neurons
    append-or-replace(neuron.output-dict, neuron.activation(neuron.input-dict))    
```

## Down movement

In full-scale DMMs, down movement is controlled by the network matrix, and the network matrix
is emitted by a _Self_ neuron.

In this preliminary DMM Lite experiment, the matrix is external, and we'll let the neuron
control the use of the network matrix.

The network matrix still wants to be a tensor of at least rank 4 (unless we reshape it to flatten it)

```
For neuron in all-neurons
    append-or-replace(neuron.input-dict, new input-dict with fields) # the fields might depend on the neuron type or not
    For field in neuron.input-dict.all-fields
        matrix-row = network-matrix[neuron][field]
        For resulting-neuron in all-neurons
            For resulting-field in resulting-neuron.output-dict.all-fields
                neuron.input-dict[field] += matrix-row[resulting-neuron][resulting-field]*resulting-neuron.output-dict[resulting-field]
                # here matrix-row[resulting-neuron][resulting-field] is a number
                # and neuron.input-dict[field] and resulting-neuron.output-dict[resulting-field]
                #     are vector-like objects (dictionaries of numbers)
```

This scheme is good enough for our planned experiment, but it's not flexible enough to run sparse DMMs and such.

Let's go to a scheme of "intermediate flexibility" (we'll still have a fixed set of active neurons, unlike DMM 1.0,
but we'll let the network matrix control the process of down-movement (as it should).

We also acknowledge that we decided to work with mutable dictionaries for the time being.

:-) The pseudo-code below looks suspiciously like Julia :-)

Note that the network_matrix is a tensor of rank 4 in this implementation.

```
for neuron_name in keys(network_matrix)
    if neuron_name in all_neurons # correctness, and there is a problem that if a neuron disappears from a matrix
                                    altogether, then its input is not updates at all (in reality it should become
                                    inactive in such a situation, but that's for a subsequent version)
        neuron = all_neurons[neuron_name]
        neuron.input_dict = Dict()
        for (field_name, matrix_row) in network_matrix[neuron_name]
            neuron.input_dict[field_name] = Dict()
            for resulting_neuron_name in keys(matrix_row)
                resulting_neuron = all_neurons[resulting_neuron_name]
                for (resulting_field_name, multiplier) in matrix_row[resulting_neuron_name]
                    neuron.input_dict[field_name] += multiplier*resulting_neuron.output_dict[resulting_field_name] # replace in code                    
end end end end end        
```
