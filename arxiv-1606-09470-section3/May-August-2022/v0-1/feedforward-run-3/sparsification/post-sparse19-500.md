# Testing and building upon sparse19-after-500-steps model

Testing:

```
$ diff test.jl test-original.jl
36c36
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse19-after-500-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.653321), "input"=>Dict("char"=>-0.522006)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.26811)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6561…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.05622), "compare-5-2"=>Dict("false"=>5.18148f-5)),…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.443144)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.67114)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.66135…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.237288), "compare-1-2"=>Dict("false"=>0.454576)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-5.65419f-5)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>4.06399f-5)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.00…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.853719)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.576435)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
23

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.0068644676 reg_l1 12.581351 reg_l2 9.933794
loss 2.523135
2.523135f0

julia> # now, let's use the "tets string."
```

```
$ diff test.jl test-original.jl
36c36
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
47c47
<     s::String = "tets string."
---
>     s::String = "test string."
56c56
< trainable["network_matrix"] = deserialize("sparse19-after-500-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

```
julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.653321), "input"=>Dict("char"=>-0.522006)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.26811)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6561…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.05622), "compare-5-2"=>Dict("false"=>5.18148f-5)),…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.443144)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.67114)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.66135…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.237288), "compare-1-2"=>Dict("false"=>0.454576)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-5.65419f-5)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>4.06399f-5)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.00…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.853719)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.576435)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.0062766494 reg_l1 12.581351 reg_l2 9.933794
loss 2.522547
2.522547f0

julia> # perfect generalization for this particular example
```

Now, back to standard string. And counter-intuitively, we should
decrease regularization for better training at this point, rather than increasing it,
although it only lasts for a bit (finding a truly proper training mode this close
to optimum might turn out to be quite non-trivial): 

```
$ diff loss.jl loss-original.jl
67c67
<     l += 0.1f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
$ diff test.jl test-original.jl
36c36
<     l += 0.1f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse19-after-500-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.653321), "input"=>Dict("char"=>-0.522006)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.26811)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6561…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.05622), "compare-5-2"=>Dict("false"=>5.18148f-5)),…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.443144)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.67114)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.66135…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.237288), "compare-1-2"=>Dict("false"=>0.454576)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-5.65419f-5)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>4.06399f-5)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.00…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.853719)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.576435)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> sparse = sparsecopy(trainable["network_matrix"], 0.1f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.653321), "input"=>Dict("char"=>-0.522006)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.26811)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6561…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.05622)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.443144)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.67114)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.66135…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.237288), "compare-1-2"=>Dict("false"=>0.454576)), "d…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.853719)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.576435)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.653321), "input"=>Dict("char"=>-0.522006)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.26811)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6561…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.05622)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.443144)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.67114)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.66135…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.237288), "compare-1-2"=>Dict("false"=>0.454576)), "d…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.853719)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.576435)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
19

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-28T15:27:29.018
STEP 1 ================================
prereg loss 0.0068644676 reg_l1 12.580988 reg_l2 9.933794
loss 1.2649633
STEP 2 ================================
prereg loss 0.0076915575 reg_l1 12.597985 reg_l2 9.955561
loss 1.26749
STEP 3 ================================
prereg loss 0.018175201 reg_l1 12.595844 reg_l2 9.951397
loss 1.2777597
STEP 4 ================================
prereg loss 0.010124117 reg_l1 12.59621 reg_l2 9.950559
loss 1.2697451
STEP 5 ================================
prereg loss 0.0049293484 reg_l1 12.599379 reg_l2 9.95432
loss 1.2648672
STEP 6 ================================
prereg loss 0.006751801 reg_l1 12.603434 reg_l2 9.959784
loss 1.2670952
STEP 7 ================================
prereg loss 0.007140542 reg_l1 12.607351 reg_l2 9.965375
loss 1.2678757
STEP 8 ================================
prereg loss 0.0055982335 reg_l1 12.610378 reg_l2 9.969795
loss 1.266636
STEP 9 ================================
prereg loss 0.0040311012 reg_l1 12.612171 reg_l2 9.972391
loss 1.2652482
STEP 10 ================================
prereg loss 0.004105013 reg_l1 12.61308 reg_l2 9.973627
loss 1.265413
STEP 11 ================================
prereg loss 0.0064946655 reg_l1 12.613761 reg_l2 9.974482
loss 1.2678708
STEP 12 ================================
prereg loss 0.00720326 reg_l1 12.614657 reg_l2 9.97565
loss 1.268669
STEP 13 ================================
prereg loss 0.0054717716 reg_l1 12.616094 reg_l2 9.977695
loss 1.2670813
STEP 14 ================================
prereg loss 0.0043800636 reg_l1 12.6182575 reg_l2 9.980955
loss 1.2662059
STEP 15 ================================
prereg loss 0.0036164562 reg_l1 12.620946 reg_l2 9.98515
loss 1.2657111
STEP 16 ================================
prereg loss 0.0032682796 reg_l1 12.6238365 reg_l2 9.989761
loss 1.265652
STEP 17 ================================
prereg loss 0.002990363 reg_l1 12.6265545 reg_l2 9.994191
loss 1.2656459
STEP 18 ================================
prereg loss 0.0024574338 reg_l1 12.628805 reg_l2 9.997961
loss 1.265338
STEP 19 ================================
prereg loss 0.0017454566 reg_l1 12.6305 reg_l2 10.00089
loss 1.2647954
STEP 20 ================================
prereg loss 0.0014546724 reg_l1 12.631737 reg_l2 10.003135
loss 1.2646284
STEP 21 ================================
prereg loss 0.0022337218 reg_l1 12.6327505 reg_l2 10.005042
loss 1.2655088
STEP 22 ================================
prereg loss 0.0026247473 reg_l1 12.633833 reg_l2 10.007053
loss 1.266008
STEP 23 ================================
prereg loss 0.001972154 reg_l1 12.635131 reg_l2 10.009404
loss 1.2654853
STEP 24 ================================
prereg loss 0.0012559105 reg_l1 12.636613 reg_l2 10.012063
loss 1.2649171
STEP 25 ================================
prereg loss 0.0012509112 reg_l1 12.638112 reg_l2 10.014791
loss 1.2650621
STEP 26 ================================
prereg loss 0.0014150979 reg_l1 12.639401 reg_l2 10.017242
loss 1.2653553
STEP 27 ================================
prereg loss 0.0012037089 reg_l1 12.640326 reg_l2 10.019172
loss 1.2652363
STEP 28 ================================
prereg loss 0.0010990516 reg_l1 12.640874 reg_l2 10.020529
loss 1.2651865
STEP 29 ================================
prereg loss 0.0015046474 reg_l1 12.641169 reg_l2 10.02148
loss 1.2656215
STEP 30 ================================
prereg loss 0.0018812022 reg_l1 12.6413965 reg_l2 10.022297
loss 1.2660209
STEP 31 ================================
prereg loss 0.0017258723 reg_l1 12.64171 reg_l2 10.023227
loss 1.2658969
STEP 32 ================================
prereg loss 0.0013999959 reg_l1 12.642179 reg_l2 10.024405
loss 1.2656178
STEP 33 ================================
prereg loss 0.0014174002 reg_l1 12.642768 reg_l2 10.025804
loss 1.2656943
STEP 34 ================================
prereg loss 0.0015535399 reg_l1 12.643417 reg_l2 10.027353
loss 1.2658952
STEP 35 ================================
prereg loss 0.0015554016 reg_l1 12.644063 reg_l2 10.028941
loss 1.2659618
STEP 36 ================================
prereg loss 0.0015235043 reg_l1 12.64466 reg_l2 10.030481
loss 1.2659895
STEP 37 ================================
prereg loss 0.0018226658 reg_l1 12.6452 reg_l2 10.031946
loss 1.2663428
STEP 38 ================================
prereg loss 0.0020754018 reg_l1 12.645689 reg_l2 10.033314
loss 1.2666444
STEP 39 ================================
prereg loss 0.0018504515 reg_l1 12.64619 reg_l2 10.03469
loss 1.2664695
STEP 40 ================================
prereg loss 0.0017683082 reg_l1 12.646728 reg_l2 10.036135
loss 1.2664411
STEP 41 ================================
prereg loss 0.0018245536 reg_l1 12.647388 reg_l2 10.037804
loss 1.2665634
STEP 42 ================================
prereg loss 0.0017845114 reg_l1 12.648161 reg_l2 10.039694
loss 1.2666007
STEP 43 ================================
prereg loss 0.0016433352 reg_l1 12.649011 reg_l2 10.041739
loss 1.2665443
STEP 44 ================================
prereg loss 0.0017107062 reg_l1 12.649885 reg_l2 10.043859
loss 1.2666992
STEP 45 ================================
prereg loss 0.0018552948 reg_l1 12.650726 reg_l2 10.045923
loss 1.266928
STEP 46 ================================
prereg loss 0.0017445781 reg_l1 12.651525 reg_l2 10.047903
loss 1.2668971
STEP 47 ================================
prereg loss 0.0015668111 reg_l1 12.652258 reg_l2 10.049773
loss 1.2667925
STEP 48 ================================
prereg loss 0.0015560061 reg_l1 12.652908 reg_l2 10.0515175
loss 1.2668469
STEP 49 ================================
prereg loss 0.0016081796 reg_l1 12.653482 reg_l2 10.053172
loss 1.2669564
STEP 50 ================================
prereg loss 0.0016173399 reg_l1 12.654027 reg_l2 10.054799
loss 1.26702
STEP 51 ================================
prereg loss 0.0016939018 reg_l1 12.654596 reg_l2 10.056486
loss 1.2671535
STEP 52 ================================
prereg loss 0.0018360502 reg_l1 12.655227 reg_l2 10.058274
loss 1.2673588
STEP 53 ================================
prereg loss 0.0018683872 reg_l1 12.655922 reg_l2 10.060147
loss 1.2674606
STEP 54 ================================
prereg loss 0.0018043132 reg_l1 12.656638 reg_l2 10.062027
loss 1.2674682
STEP 55 ================================
prereg loss 0.0018550047 reg_l1 12.657393 reg_l2 10.063946
loss 1.2675943
STEP 56 ================================
prereg loss 0.0018730169 reg_l1 12.658197 reg_l2 10.065923
loss 1.2676928
STEP 57 ================================
prereg loss 0.0018653794 reg_l1 12.659016 reg_l2 10.067903
loss 1.267767
STEP 58 ================================
prereg loss 0.0018873992 reg_l1 12.659884 reg_l2 10.069957
loss 1.2678759
STEP 59 ================================
prereg loss 0.0018431348 reg_l1 12.660897 reg_l2 10.072234
loss 1.2679329
STEP 60 ================================
prereg loss 0.0018649438 reg_l1 12.662035 reg_l2 10.074702
loss 1.2680684
STEP 61 ================================
prereg loss 0.0018407911 reg_l1 12.663207 reg_l2 10.077219
loss 1.2681615
STEP 62 ================================
prereg loss 0.0017359612 reg_l1 12.664372 reg_l2 10.079715
loss 1.2681732
STEP 63 ================================
prereg loss 0.0016754093 reg_l1 12.665479 reg_l2 10.0821085
loss 1.2682233
STEP 64 ================================
prereg loss 0.0016955782 reg_l1 12.666491 reg_l2 10.084372
loss 1.2683448
STEP 65 ================================
prereg loss 0.0016940188 reg_l1 12.667489 reg_l2 10.086621
loss 1.2684429
STEP 66 ================================
prereg loss 0.0017706648 reg_l1 12.668485 reg_l2 10.088883
loss 1.2686191
STEP 67 ================================
prereg loss 0.0017928918 reg_l1 12.669532 reg_l2 10.091229
loss 1.2687461
STEP 68 ================================
prereg loss 0.0017245717 reg_l1 12.670621 reg_l2 10.093651
loss 1.2687867
STEP 69 ================================
prereg loss 0.0017640882 reg_l1 12.6717005 reg_l2 10.096067
loss 1.2689341
STEP 70 ================================
prereg loss 0.0017564184 reg_l1 12.672804 reg_l2 10.09853
loss 1.2690369
STEP 71 ================================
prereg loss 0.0017542973 reg_l1 12.67392 reg_l2 10.101022
loss 1.2691463
STEP 72 ================================
prereg loss 0.0017861078 reg_l1 12.675006 reg_l2 10.103467
loss 1.2692868
STEP 73 ================================
prereg loss 0.0016938826 reg_l1 12.676091 reg_l2 10.105916
loss 1.269303
STEP 74 ================================
prereg loss 0.0017132648 reg_l1 12.6771755 reg_l2 10.108385
loss 1.2694309
STEP 75 ================================
prereg loss 0.0016747542 reg_l1 12.678327 reg_l2 10.1109705
loss 1.2695074
STEP 76 ================================
prereg loss 0.0016638784 reg_l1 12.67952 reg_l2 10.113641
loss 1.2696159
STEP 77 ================================
prereg loss 0.0017034602 reg_l1 12.680685 reg_l2 10.116272
loss 1.269772
STEP 78 ================================
prereg loss 0.0016298346 reg_l1 12.681823 reg_l2 10.118858
loss 1.2698121
STEP 79 ================================
prereg loss 0.0015563675 reg_l1 12.682902 reg_l2 10.121365
loss 1.2698467
STEP 80 ================================
prereg loss 0.0016002795 reg_l1 12.683911 reg_l2 10.123769
loss 1.2699914
STEP 81 ================================
prereg loss 0.0015700242 reg_l1 12.684922 reg_l2 10.126193
loss 1.2700622
STEP 82 ================================
prereg loss 0.0016355689 reg_l1 12.685915 reg_l2 10.128605
loss 1.2702271
STEP 83 ================================
prereg loss 0.0016704495 reg_l1 12.686929 reg_l2 10.1310625
loss 1.2703633
STEP 84 ================================
prereg loss 0.0016572045 reg_l1 12.687962 reg_l2 10.13355
loss 1.2704535
STEP 85 ================================
prereg loss 0.0016603752 reg_l1 12.689055 reg_l2 10.136132
loss 1.2705659
STEP 86 ================================
prereg loss 0.0016642222 reg_l1 12.690115 reg_l2 10.1386595
loss 1.2706758
STEP 87 ================================
prereg loss 0.0016795554 reg_l1 12.691223 reg_l2 10.141253
loss 1.2708019
STEP 88 ================================
prereg loss 0.0016408596 reg_l1 12.692316 reg_l2 10.143828
loss 1.2708726
STEP 89 ================================
prereg loss 0.0016242652 reg_l1 12.693401 reg_l2 10.146405
loss 1.2709644
STEP 90 ================================
prereg loss 0.0015933897 reg_l1 12.694553 reg_l2 10.149091
loss 1.2710487
STEP 91 ================================
prereg loss 0.0016013952 reg_l1 12.695685 reg_l2 10.151763
loss 1.2711699
STEP 92 ================================
prereg loss 0.0015945183 reg_l1 12.696793 reg_l2 10.154406
loss 1.2712739
STEP 93 ================================
prereg loss 0.0015915085 reg_l1 12.697856 reg_l2 10.156993
loss 1.2713772
STEP 94 ================================
prereg loss 0.0016529075 reg_l1 12.698945 reg_l2 10.159616
loss 1.2715474
STEP 95 ================================
prereg loss 0.0016692209 reg_l1 12.700014 reg_l2 10.162215
loss 1.2716706
STEP 96 ================================
prereg loss 0.0016171464 reg_l1 12.701082 reg_l2 10.164818
loss 1.2717254
STEP 97 ================================
prereg loss 0.0016811672 reg_l1 12.702137 reg_l2 10.167412
loss 1.2718949
STEP 98 ================================
prereg loss 0.0016542947 reg_l1 12.703261 reg_l2 10.170127
loss 1.2719804
STEP 99 ================================
prereg loss 0.0017519649 reg_l1 12.704456 reg_l2 10.172953
loss 1.2721977
STEP 100 ================================
prereg loss 0.0017620471 reg_l1 12.705667 reg_l2 10.175798
loss 1.2723287
2022-06-28T15:32:04.130

julia> open("sparse20-after-100-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse20-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse20-after-100-steps-opt.ser", opt)

julia> steps!(100)
2022-06-28T15:38:03.917
STEP 1 ================================
prereg loss 0.0016100588 reg_l1 12.706874 reg_l2 10.178641
loss 1.2722975
STEP 2 ================================
prereg loss 0.0016167248 reg_l1 12.708033 reg_l2 10.181414
loss 1.27242
STEP 3 ================================
prereg loss 0.0015887042 reg_l1 12.709182 reg_l2 10.184182
loss 1.272507
STEP 4 ================================
prereg loss 0.0016636946 reg_l1 12.710344 reg_l2 10.18698
loss 1.2726982
STEP 5 ================================
prereg loss 0.0017646633 reg_l1 12.71153 reg_l2 10.189822
loss 1.2729176
STEP 6 ================================
prereg loss 0.0016521044 reg_l1 12.712759 reg_l2 10.192726
loss 1.272928
STEP 7 ================================
prereg loss 0.0015590142 reg_l1 12.71397 reg_l2 10.195611
loss 1.272956
STEP 8 ================================
prereg loss 0.0016293714 reg_l1 12.715094 reg_l2 10.198371
loss 1.2731388
STEP 9 ================================
prereg loss 0.0015841951 reg_l1 12.716194 reg_l2 10.201104
loss 1.2732036
STEP 10 ================================
prereg loss 0.001685051 reg_l1 12.717262 reg_l2 10.203804
loss 1.2734113
STEP 11 ================================
prereg loss 0.0017433646 reg_l1 12.718375 reg_l2 10.206585
loss 1.2735809
STEP 12 ================================
prereg loss 0.0017227035 reg_l1 12.719545 reg_l2 10.209468
loss 1.2736772
STEP 13 ================================
prereg loss 0.0017041479 reg_l1 12.720837 reg_l2 10.212523
loss 1.2737877
STEP 14 ================================
prereg loss 0.0017674891 reg_l1 12.7222 reg_l2 10.215687
loss 1.2739875
STEP 15 ================================
prereg loss 0.0016761804 reg_l1 12.723538 reg_l2 10.218804
loss 1.2740301
STEP 16 ================================
prereg loss 0.0015709258 reg_l1 12.724841 reg_l2 10.221875
loss 1.2740551
STEP 17 ================================
prereg loss 0.0015584084 reg_l1 12.72616 reg_l2 10.224976
loss 1.2741745
STEP 18 ================================
prereg loss 0.0015669791 reg_l1 12.727449 reg_l2 10.228049
loss 1.274312
STEP 19 ================================
prereg loss 0.001579511 reg_l1 12.728729 reg_l2 10.231128
loss 1.2744524
STEP 20 ================================
prereg loss 0.0015744384 reg_l1 12.729991 reg_l2 10.234188
loss 1.2745736
STEP 21 ================================
prereg loss 0.0015632268 reg_l1 12.731204 reg_l2 10.237187
loss 1.2746836
STEP 22 ================================
prereg loss 0.001583688 reg_l1 12.732353 reg_l2 10.240094
loss 1.274819
STEP 23 ================================
prereg loss 0.0016544915 reg_l1 12.733538 reg_l2 10.243048
loss 1.2750083
STEP 24 ================================
prereg loss 0.0016903552 reg_l1 12.734731 reg_l2 10.246027
loss 1.2751635
STEP 25 ================================
prereg loss 0.001665134 reg_l1 12.735957 reg_l2 10.249073
loss 1.2752608
STEP 26 ================================
prereg loss 0.0016400662 reg_l1 12.737304 reg_l2 10.252298
loss 1.2753705
STEP 27 ================================
prereg loss 0.0016343079 reg_l1 12.738628 reg_l2 10.2555065
loss 1.2754972
STEP 28 ================================
prereg loss 0.001674523 reg_l1 12.74002 reg_l2 10.258804
loss 1.2756765
STEP 29 ================================
prereg loss 0.0016297468 reg_l1 12.7413845 reg_l2 10.262073
loss 1.2757682
STEP 30 ================================
prereg loss 0.0015577518 reg_l1 12.742724 reg_l2 10.2653055
loss 1.2758301
STEP 31 ================================
prereg loss 0.0016025981 reg_l1 12.744018 reg_l2 10.268488
loss 1.2760044
STEP 32 ================================
prereg loss 0.0015697854 reg_l1 12.745359 reg_l2 10.271748
loss 1.2761058
STEP 33 ================================
prereg loss 0.0016365703 reg_l1 12.746682 reg_l2 10.274998
loss 1.2763048
STEP 34 ================================
prereg loss 0.0016248572 reg_l1 12.748011 reg_l2 10.278263
loss 1.276426
STEP 35 ================================
prereg loss 0.0015982365 reg_l1 12.749315 reg_l2 10.281502
loss 1.2765298
STEP 36 ================================
prereg loss 0.0015914197 reg_l1 12.750666 reg_l2 10.284797
loss 1.276658
STEP 37 ================================
prereg loss 0.001598907 reg_l1 12.751973 reg_l2 10.288037
loss 1.2767963
STEP 38 ================================
prereg loss 0.0016296086 reg_l1 12.753268 reg_l2 10.291272
loss 1.2769564
STEP 39 ================================
prereg loss 0.0016382843 reg_l1 12.754675 reg_l2 10.294675
loss 1.2771058
STEP 40 ================================
prereg loss 0.0016207824 reg_l1 12.756087 reg_l2 10.298101
loss 1.2772295
STEP 41 ================================
prereg loss 0.0015979016 reg_l1 12.757482 reg_l2 10.301513
loss 1.277346
STEP 42 ================================
prereg loss 0.0015781394 reg_l1 12.758926 reg_l2 10.304985
loss 1.2774707
STEP 43 ================================
prereg loss 0.001578923 reg_l1 12.760318 reg_l2 10.308388
loss 1.2776108
STEP 44 ================================
prereg loss 0.0015558746 reg_l1 12.761675 reg_l2 10.311752
loss 1.2777234
STEP 45 ================================
prereg loss 0.001600455 reg_l1 12.763013 reg_l2 10.315099
loss 1.2779018
STEP 46 ================================
prereg loss 0.0016006398 reg_l1 12.764439 reg_l2 10.318581
loss 1.2780445
STEP 47 ================================
prereg loss 0.001625712 reg_l1 12.765859 reg_l2 10.3220625
loss 1.2782116
STEP 48 ================================
prereg loss 0.0015773727 reg_l1 12.76726 reg_l2 10.325534
loss 1.2783034
STEP 49 ================================
prereg loss 0.0016078928 reg_l1 12.768719 reg_l2 10.329077
loss 1.2784798
STEP 50 ================================
prereg loss 0.0015797531 reg_l1 12.7701435 reg_l2 10.332568
loss 1.2785941
STEP 51 ================================
prereg loss 0.0015663323 reg_l1 12.771536 reg_l2 10.336034
loss 1.2787199
STEP 52 ================================
prereg loss 0.001575479 reg_l1 12.773001 reg_l2 10.339597
loss 1.2788756
STEP 53 ================================
prereg loss 0.0015896364 reg_l1 12.774457 reg_l2 10.343169
loss 1.2790353
STEP 54 ================================
prereg loss 0.0015554876 reg_l1 12.775898 reg_l2 10.346735
loss 1.2791452
STEP 55 ================================
prereg loss 0.0015746956 reg_l1 12.777295 reg_l2 10.350246
loss 1.2793043
STEP 56 ================================
prereg loss 0.0015891475 reg_l1 12.778735 reg_l2 10.35381
loss 1.2794627
STEP 57 ================================
prereg loss 0.0016249914 reg_l1 12.780149 reg_l2 10.35735
loss 1.27964
STEP 58 ================================
prereg loss 0.0015831435 reg_l1 12.78157 reg_l2 10.360917
loss 1.2797402
STEP 59 ================================
prereg loss 0.0016609009 reg_l1 12.782979 reg_l2 10.364485
loss 1.2799588
STEP 60 ================================
prereg loss 0.0016157365 reg_l1 12.784496 reg_l2 10.368214
loss 1.2800654
STEP 61 ================================
prereg loss 0.0017793806 reg_l1 12.786109 reg_l2 10.372077
loss 1.2803904
STEP 62 ================================
prereg loss 0.0016918292 reg_l1 12.787731 reg_l2 10.375974
loss 1.280465
STEP 63 ================================
prereg loss 0.0015055293 reg_l1 12.789337 reg_l2 10.379847
loss 1.2804393
STEP 64 ================================
prereg loss 0.0015828287 reg_l1 12.790845 reg_l2 10.383578
loss 1.2806674
STEP 65 ================================
prereg loss 0.0015048007 reg_l1 12.79233 reg_l2 10.3872795
loss 1.2807378
STEP 66 ================================
prereg loss 0.0016022223 reg_l1 12.79377 reg_l2 10.390937
loss 1.2809792
STEP 67 ================================
prereg loss 0.0016835751 reg_l1 12.795254 reg_l2 10.394669
loss 1.281209
STEP 68 ================================
prereg loss 0.0015714762 reg_l1 12.79676 reg_l2 10.398461
loss 1.2812474
STEP 69 ================================
prereg loss 0.0016516827 reg_l1 12.798207 reg_l2 10.402163
loss 1.2814724
STEP 70 ================================
prereg loss 0.0016467122 reg_l1 12.799693 reg_l2 10.405913
loss 1.2816161
STEP 71 ================================
prereg loss 0.0016684587 reg_l1 12.801269 reg_l2 10.409788
loss 1.2817954
STEP 72 ================================
prereg loss 0.0017044988 reg_l1 12.80289 reg_l2 10.413757
loss 1.2819935
STEP 73 ================================
prereg loss 0.0015433877 reg_l1 12.80456 reg_l2 10.417831
loss 1.2819993
STEP 74 ================================
prereg loss 0.0016088674 reg_l1 12.80618 reg_l2 10.421841
loss 1.2822269
STEP 75 ================================
prereg loss 0.0015515961 reg_l1 12.807806 reg_l2 10.425842
loss 1.2823323
STEP 76 ================================
prereg loss 0.0016456478 reg_l1 12.809443 reg_l2 10.429844
loss 1.28259
STEP 77 ================================
prereg loss 0.0017027023 reg_l1 12.811086 reg_l2 10.43387
loss 1.2828113
STEP 78 ================================
prereg loss 0.0015054148 reg_l1 12.812763 reg_l2 10.437971
loss 1.2827817
STEP 79 ================================
prereg loss 0.0015133863 reg_l1 12.81439 reg_l2 10.442004
loss 1.2829524
STEP 80 ================================
prereg loss 0.0014726102 reg_l1 12.8159685 reg_l2 10.445952
loss 1.2830695
STEP 81 ================================
prereg loss 0.0015459809 reg_l1 12.817448 reg_l2 10.449758
loss 1.2832909
STEP 82 ================================
prereg loss 0.001638091 reg_l1 12.818925 reg_l2 10.453572
loss 1.2835306
STEP 83 ================================
prereg loss 0.0015764917 reg_l1 12.820437 reg_l2 10.457467
loss 1.2836204
STEP 84 ================================
prereg loss 0.0016722841 reg_l1 12.82194 reg_l2 10.461369
loss 1.2838663
STEP 85 ================================
prereg loss 0.0016630972 reg_l1 12.82355 reg_l2 10.465419
loss 1.2840182
STEP 86 ================================
prereg loss 0.0017040942 reg_l1 12.825269 reg_l2 10.469621
loss 1.284231
STEP 87 ================================
prereg loss 0.0016857854 reg_l1 12.826999 reg_l2 10.47387
loss 1.2843857
STEP 88 ================================
prereg loss 0.0015303454 reg_l1 12.82874 reg_l2 10.478161
loss 1.2844043
STEP 89 ================================
prereg loss 0.0015060187 reg_l1 12.830532 reg_l2 10.482508
loss 1.2845592
STEP 90 ================================
prereg loss 0.0015064599 reg_l1 12.832245 reg_l2 10.486751
loss 1.2847309
STEP 91 ================================
prereg loss 0.0015149978 reg_l1 12.8339 reg_l2 10.490914
loss 1.2849051
STEP 92 ================================
prereg loss 0.0015210125 reg_l1 12.83552 reg_l2 10.495044
loss 1.285073
STEP 93 ================================
prereg loss 0.0015273809 reg_l1 12.837114 reg_l2 10.499149
loss 1.2852389
STEP 94 ================================
prereg loss 0.0015848293 reg_l1 12.838676 reg_l2 10.503222
loss 1.2854526
STEP 95 ================================
prereg loss 0.0016627334 reg_l1 12.840346 reg_l2 10.50744
loss 1.2856973
STEP 96 ================================
prereg loss 0.00165372 reg_l1 12.842018 reg_l2 10.511684
loss 1.2858555
STEP 97 ================================
prereg loss 0.0016630237 reg_l1 12.843683 reg_l2 10.515943
loss 1.2860314
STEP 98 ================================
prereg loss 0.001615554 reg_l1 12.845475 reg_l2 10.5203705
loss 1.2861631
STEP 99 ================================
prereg loss 0.0016758632 reg_l1 12.847378 reg_l2 10.524949
loss 1.2864137
STEP 100 ================================
prereg loss 0.0016154931 reg_l1 12.849282 reg_l2 10.529567
loss 1.2865437
2022-06-28T15:42:12.198

julia> open("sparse20-after-200-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse20-after-200-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse20-after-200-steps-opt.ser", opt)

julia> close(io)
```
