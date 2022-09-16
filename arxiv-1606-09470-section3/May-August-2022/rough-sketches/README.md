# Rough sketches

`naive-activations.jl` - this was just to help me think (keeping this file to preserve the history of my train of thought)

`pseudo-code.md` - the "two-stroke-cycle" for a "DMM Lite" engine

Differences between "DMM Lite" (here) and DMM 1.0 (2016-2017, https://github.com/jsa-aerial/DMM):

 * streams of flat dictionaries in DMM Lite instead of streams of trees (nested dictionaries) in DMM 1.0
 * the network matrix is outside the network in DMM Lite instead of being controlled by one of the neurons in DMM 1.0
 * fixed number of active neurons in DMM Lite instead of the number of active neurons being controlled by the network matrix in DMM 1.0
 * the network matrix is a tensor of rank 4 in DMM Lite vs tensor of rank 6 in DMM 1.0

`variadic-sketch.jl` - the first implementation of DMM Lite in Julia together with the first DMM Lite implementation of 
                       the DMM example described in Section 3 of arxiv:1606.09470

`variadic-dp-v0-0-1.jl` - the first version where taking gradients with respect to the network matrix does actually work
                          (Zygote.jl v0.6.40, Flux.jl v0.13.1, Julia 1.7.2);
                          type annotations have also been added just in case, 
                          although Zygote often works fine in the uptyped situation as well.

`two-networks.jl` - DMM Lite engine and two DMM Lite networks running in parallel (one handcrafted, one trainable, that is to be
                    trained for its output to imitate the output of the handcrafted one)

`ADAM-v0-0-1.jl` - ADAM optimizer for trees (to be included into `two-networks.jl` which is assumed as the context)

`train-v0-0-1.jl` - this currently just performs one `adam-step!`, and then one can run training loops interatively.

---

The sequence of training runs in the subdirectories:

`first-run` - interesting structure, need to rerun with better instrumentation

`second-run` - a smaller network, shows a blow-up effect in a recurrent machine, needs a governor

`run-2.1` - second run with the governor

`run-1.1` - first run, well instrumented and extended

`run-1.1/sparsity-exploration` - exploration of sparsity structure resulting from `run-1.1`

`run-1.1/BPTT-140` - continuing to train with sparse structures obtained in `run-1.1`

`run-1.1/post-BPTT-140`, `run-1.1/post-post-BPTT-140-bug-fix` - fixing some bugs discovered during `BPTT-140` run

`run-3` - repeating `run-1` with a new regularization scheme; works great on the initial segment

`run-3/BPTT-140` - but does not work at all for the extended segment, needs further improvements
