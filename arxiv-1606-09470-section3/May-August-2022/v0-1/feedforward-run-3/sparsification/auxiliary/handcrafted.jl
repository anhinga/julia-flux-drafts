# a handcrafted duplicate detector (arxiv:1606.09470 section 3)

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

