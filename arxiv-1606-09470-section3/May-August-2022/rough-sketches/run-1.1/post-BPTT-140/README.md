# post-BPTT-140 focusing on the discovered performance bug

This has just been fixed, and the iterations became three times faster:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0219264), "norm-5"=>Dict("norm"=>0.25096), "accum-4"…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.199862), "accum-2"=>Dict("dict"=>0.201067), "accum-1"=>Di…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("false"=>0.0193636)), "true"=>Dict("compare-3"=>Dict("false"=>-0.01210…
  "dot-2"     => Dict("dict"=>Dict("norm-3"=>Dict("true"=>-0.00654772)), "true"=>Dict("accum-4"=>Dict("true"=>0.0083629…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.190032), "norm-5"=>Dict("norm"=>0.0691165), "accum-4"=>…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict-1"=>0.0433679)), "dot"=>Dict("accum-4"=>Dict("dot"=>0.0208008))…
  "accum-3"   => Dict("false"=>Dict("compare-4"=>Dict("norm"=>-0.00810362)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.155827), "norm-5"=>Dict("norm"=>-0.326115), "accum-4"=>…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.0902416)), "true"=>Dict("accum-5"=>Dict("dict"=>0.0888…
  "compare-4" => Dict("dot"=>Dict("accum-2"=>Dict("dot"=>0.00907417)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0643…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict-1"=>0.00865799)), "true"=>Dict("accum-2"=>Dict("norm"=>0.0028621…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("norm"=>-0.0135199)), "dot"=>Dict("accum-3"=>Dict("norm"=>0.018773…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0996243), "norm-5"=>Dict("dot"=>0.0478396, "norm"=>-…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.180737), "norm-5"=>Dict("norm"=>0.18749), "accum-4"=>D…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0231354), "norm-5"=>Dict("norm"=>-0.112434), "accum-…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.30937), "norm-5"=>Dict("norm"=>-0.0241003), "accum-4"…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.708961), "norm-5"=>Dict("norm"=>0.24083), "accum-4"=…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.164622), "norm-5"=>Dict("norm"=>0.129442), "accum-4"…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("false"=>0.0189236)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>0.21…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("false"=>0.0143029)), "false"=>Dict("norm-3"=>Dict("dict-1"=>0.1208…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.443442), "norm-5"=>Dict("norm"=>-0.0127115), "accum-4"=…

julia> count(opt.mt)
851

julia> count(opt.vt)
851

julia> steps!(16)
2022-06-05T14:35:42.263
STEP 1 ================================
prereg loss 410.80295 regularization 335.15085
loss 745.9538
STEP 2 ================================
prereg loss 400.06247 regularization 334.5425
loss 734.605
STEP 3 ================================
prereg loss 389.3122 regularization 334.02466
loss 723.33685
STEP 4 ================================
prereg loss 378.1521 regularization 333.55975
loss 711.71185
STEP 5 ================================
prereg loss 368.61615 regularization 333.0826
loss 701.6987
STEP 6 ================================
prereg loss 358.08432 regularization 332.68765
loss 690.772
STEP 7 ================================
prereg loss 347.94073 regularization 332.35764
loss 680.29834
STEP 8 ================================
prereg loss 338.7498 regularization 332.0483
loss 670.7981
STEP 9 ================================
prereg loss 327.65314 regularization 331.79083
loss 659.444
STEP 10 ================================
prereg loss 318.64127 regularization 331.48163
loss 650.1229
STEP 11 ================================
prereg loss 309.36426 regularization 331.1553
loss 640.51953
STEP 12 ================================
prereg loss 300.43744 regularization 330.81122
loss 631.24866
STEP 13 ================================
prereg loss 292.13742 regularization 330.4651
loss 622.60254
STEP 14 ================================
prereg loss 285.15756 regularization 330.08612
loss 615.24365
STEP 15 ================================
prereg loss 278.7816 regularization 329.70935
loss 608.49097
STEP 16 ================================
prereg loss 277.9474 regularization 329.3111
loss 607.2585
2022-06-05T14:41:54.795

julia> count(opt.mt)
851

julia> count(opt.vt)
851

julia> count(sparse)
851

julia>
```

However, there is no numerical equivalence with the `BPTT-140` first 16 steps, there are subtle differences related to
this strange phenomenon that some of the spurious `opt.mt` elements leaked into the network matrix (but not all of them,
while we would expect all of them to leak into the matrix because of that bug).

We would like to actually fix `TreeADAM` itself in this respect (why to initialize with zerocopy, when we can
initialize with empty dictionaries of the same type), but I'd also like to understand the nature of that bug.

(Understood, fixed in the previous commits).
