# Design notes for V-values

Here is what we have for v0.1:

Differences between "DMM Lite" (here) and DMM 1.0 (2016-2017, https://github.com/jsa-aerial/DMM):

 * **streams of flat dictionaries in DMM Lite instead of streams of trees (nested dictionaries) in DMM 1.0**
 * the network matrix is outside the network in DMM Lite instead of being controlled by one of the neurons in DMM 1.0
 * fixed number of active neurons in DMM Lite instead of the number of active neurons being controlled by the network matrix in DMM 1.0
 * the network matrix is a tensor of rank 4 in DMM Lite vs tensor of rank 6 in DMM 1.0

The first of these limitations is the most important one. 

In v0.2 we would like to be able to work with arbitrary V-values (at least in the sense of Section 3 of https://arxiv.org/abs/1712.07447,
but possibly also in the sense of Section 5.3).

V-values in the sense of Section 3 are implemented in Clojure as follows:

https://github.com/jsa-aerial/DMM/blob/master/src/dmm/core.clj

https://github.com/jsa-aerial/DMM/blob/master/design-notes/recurrent-maps.md

This is done purely in terms of dictionaries, and should be quite doable in `Zygote.jl`.

(A side remark: in Clojure we have used its built-in persistent dictionaries and have been
working with the streams of immutable persistent dictionaries, but we have not made
a decision on immutability or persistency for our Julia implementations (v0.1 is neither
immutable not persistent).)

---

Alternatively, in our experiments with taking gradients with respect to dictionaries in `Enzyme.jl`
we have had to enforce a stricter type discipline, and, in particular, we used the following structure:

```julia
mutable struct MD
    v::Float64
    d::Dict{Symbol, MD}
end
```

This type discipline is more expensive, due to extra indirections and extra storage for empty fields
(e.g. for empty dictionaries). But it gives extra structure, which might be particularly
helpful if we decide to implement extensions in the spirit of Section 5.3 of https://arxiv.org/abs/1712.07447.

Also, this seems to be a good time to revisit possibility of computing gradients with `Enzyme.jl` for
this problem (I just had a conversation with the creator of Enzyme, and that conversation prompted me to
revive the idea of trying to use Enzyme for this, in addition to Zygote, so that we could compare performance,
etc.)

So, I think, here is a decision: we'll use a design based on _dictionaries within mutable structs_ as
a starting point, and we might refactor later.

(A side remark: v0.1 is not fully pure-dictionary-based, it has this mutable struct for the neuron:
```julia
mutable struct Neuron
    f::Function
    input_dict::Dict{String, Dict{String, Float32}}
    output_dict::Dict{String, Dict{String, Float32}}
end
```

The overall network used to be a mutable struct as well at some point, but that had been
replaced by a dictionary with similarly named keys

```julia
#struct DMM_Lite
#    neurons::Dict{String, Neuron}
#    network_matrix::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}
#end
```
)
