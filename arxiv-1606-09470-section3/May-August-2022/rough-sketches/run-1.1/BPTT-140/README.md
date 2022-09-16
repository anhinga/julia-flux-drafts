# BPTT-140 post run-1.1. June 4-5, 2022

Sparsify, then run in the 140 time step window (no curriculum).

This behaves somewhat better than `run-2.1`, although the loss
landscape is still pretty tough even for ADAM.

First, I forgot to change regularization, and even when I increased
it 100-fold (as per back-of-an-envelope calculation, and similar to
`run-2.1`) this was not enough. I had to increase it another 10-fold for
things to start making sense:

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

julia> steps!(1)
2022-06-04T12:25:40.710
STEP 1 ================================
prereg loss 410.80295 regularization 335.15085
loss 411.1381
2022-06-04T12:26:23.001

julia> steps!(15)
2022-06-04T12:27:01.626
STEP 1 ================================
prereg loss 399.66833 regularization 335.37
loss 400.0037
STEP 2 ================================
prereg loss 388.5026 regularization 335.78735
loss 388.83838
STEP 3 ================================
prereg loss 377.05835 regularization 336.24716
loss 377.3946
STEP 4 ================================
prereg loss 366.71753 regularization 336.7305
loss 367.05426
STEP 5 ================================
prereg loss 355.3816 regularization 337.23737
loss 355.71884
STEP 6 ================================
prereg loss 344.81354 regularization 337.78497
loss 345.15134
STEP 7 ================================
prereg loss 335.53833 regularization 338.34375
loss 335.87668
STEP 8 ================================
prereg loss 325.40488 regularization 338.88306
loss 325.74377
STEP 9 ================================
prereg loss 314.11148 regularization 339.4231
loss 314.4509
STEP 10 ================================
prereg loss 305.7375 regularization 339.93668
loss 306.07742
STEP 11 ================================
prereg loss 296.90866 regularization 340.43213
loss 297.24908
STEP 12 ================================
prereg loss 289.01785 regularization 340.91864
loss 289.35876
STEP 13 ================================
prereg loss 283.81927 regularization 341.39178
loss 284.16068
STEP 14 ================================
prereg loss 280.20032 regularization 341.71243
loss 280.54202
STEP 15 ================================
prereg loss 277.98358 regularization 342.02887
loss 278.32562
2022-06-04T12:46:41.466

julia>

julia> # THIS CLEARLY NEED STRONGER REGULARIZATION

julia> close(io)

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

julia> steps!(16)
2022-06-04T12:47:21.062
STEP 1 ================================
prereg loss 410.80295 regularization 335.15085
loss 444.31802
STEP 2 ================================
prereg loss 399.5847 regularization 335.27322
loss 433.112
STEP 3 ================================
prereg loss 388.38605 regularization 335.55713
loss 421.94177
ERROR: InterruptException:

[...]

julia> # NOT ENOUGH REGULARIZATION EVEN NOW

julia> close(io)
```

Now, with a really strong regularization, 1000-fold more
than in `run-1.1`:

```
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

julia> steps!(16)
2022-06-04T12:50:37.926
STEP 1 ================================
prereg loss 410.80295 regularization 335.15085
loss 745.9538
STEP 2 ================================
prereg loss 400.06247 regularization 334.5425
loss 734.605
STEP 3 ================================
prereg loss 389.2925 regularization 334.03146
loss 723.324
STEP 4 ================================
prereg loss 378.1561 regularization 333.57047
loss 711.72656
STEP 5 ================================
prereg loss 368.6304 regularization 333.09625
loss 701.7267
STEP 6 ================================
prereg loss 358.06943 regularization 332.70255
loss 690.772
STEP 7 ================================
prereg loss 347.83728 regularization 332.377
loss 680.2143
STEP 8 ================================
prereg loss 338.48807 regularization 332.0752
loss 670.56323
STEP 9 ================================
prereg loss 327.47888 regularization 331.81848
loss 659.29736
STEP 10 ================================
prereg loss 318.2593 regularization 331.50992
loss 649.7692
STEP 11 ================================
prereg loss 308.80276 regularization 331.18546
loss 639.9882
STEP 12 ================================
prereg loss 299.69827 regularization 330.84235
loss 630.54065
STEP 13 ================================
prereg loss 291.44382 regularization 330.49725
loss 621.94104
STEP 14 ================================
prereg loss 283.98483 regularization 330.12543
loss 614.1102
STEP 15 ================================
prereg loss 278.4496 regularization 329.75327
loss 608.2029
STEP 16 ================================
prereg loss 273.8518 regularization 329.31534
loss 603.1671
2022-06-04T13:10:22.177

julia> steps!(200)
2022-06-04T13:13:00.895
STEP 1 ================================
prereg loss 276.18613 regularization 328.82327
loss 605.0094
STEP 2 ================================
prereg loss 266.92227 regularization 328.31448
loss 595.23676
STEP 3 ================================
prereg loss 267.0948 regularization 327.7911
loss 594.88586
STEP 4 ================================
prereg loss 263.61032 regularization 327.2336
loss 590.84393
STEP 5 ================================
prereg loss 269.82303 regularization 326.6683
loss 596.49133
STEP 6 ================================
prereg loss 288.66248 regularization 326.48923
loss 615.15173
STEP 7 ================================
prereg loss 276.85242 regularization 326.3484
loss 603.2008
STEP 8 ================================
prereg loss 301.62766 regularization 326.16016
loss 627.78784
STEP 9 ================================
prereg loss 317.2598 regularization 325.9566
loss 643.21643
STEP 10 ================================
prereg loss 320.71875 regularization 325.71045
loss 646.4292
STEP 11 ================================
prereg loss 320.40857 regularization 325.41562
loss 645.8242
STEP 12 ================================
prereg loss 316.9353 regularization 325.08646
loss 642.0217
STEP 13 ================================
prereg loss 306.64246 regularization 324.6591
loss 631.3015
STEP 14 ================================
prereg loss 290.71045 regularization 324.18323
loss 614.8937
STEP 15 ================================
prereg loss 283.7115 regularization 323.69684
loss 607.4083
STEP 16 ================================
prereg loss 277.24997 regularization 323.20624
loss 600.4562
STEP 17 ================================
prereg loss 272.84125 regularization 322.7338
loss 595.5751
STEP 18 ================================
prereg loss 267.9326 regularization 322.2717
loss 590.2043
STEP 19 ================================
prereg loss 265.07202 regularization 321.82672
loss 586.89874
STEP 20 ================================
prereg loss 261.73688 regularization 321.38776
loss 583.12463
STEP 21 ================================
prereg loss 258.29178 regularization 320.9661
loss 579.2579
STEP 22 ================================
prereg loss 258.07843 regularization 320.54437
loss 578.6228
STEP 23 ================================
prereg loss 259.71793 regularization 320.1271
loss 579.84503
STEP 24 ================================
prereg loss 257.38092 regularization 319.75797
loss 577.1389
STEP 25 ================================
prereg loss 257.8827 regularization 319.39972
loss 577.2824
STEP 26 ================================
prereg loss 256.9792 regularization 319.0455
loss 576.02466
STEP 27 ================================
prereg loss 255.81116 regularization 318.70224
loss 574.5134
STEP 28 ================================
prereg loss 255.67271 regularization 318.36914
loss 574.0419
STEP 29 ================================
prereg loss 255.09029 regularization 318.0614
loss 573.1517
STEP 30 ================================
prereg loss 252.62595 regularization 317.7887
loss 570.4147
STEP 31 ================================
prereg loss 251.51056 regularization 317.5241
loss 569.03467
STEP 32 ================================
prereg loss 250.42836 regularization 317.26138
loss 567.68976
STEP 33 ================================
prereg loss 249.80258 regularization 317.0013
loss 566.8039
STEP 34 ================================
prereg loss 248.77623 regularization 316.74237
loss 565.5186
STEP 35 ================================
prereg loss 247.00504 regularization 316.48062
loss 563.48566
STEP 36 ================================
prereg loss 244.47464 regularization 316.2157
loss 560.6903
STEP 37 ================================
prereg loss 239.16768 regularization 315.944
loss 555.1117
STEP 38 ================================
prereg loss 240.37012 regularization 315.67935
loss 556.04944
STEP 39 ================================
prereg loss 240.18387 regularization 315.42685
loss 555.6107
STEP 40 ================================
prereg loss 240.84554 regularization 315.17462
loss 556.02014
STEP 41 ================================
prereg loss 237.24123 regularization 314.9387
loss 552.17993
STEP 42 ================================
prereg loss 232.19757 regularization 314.7186
loss 546.91614
STEP 43 ================================
prereg loss 226.5747 regularization 314.50055
loss 541.07526
STEP 44 ================================
prereg loss 226.98885 regularization 314.2657
loss 541.2545
STEP 45 ================================
prereg loss 224.75209 regularization 314.00916
loss 538.7612
STEP 46 ================================
prereg loss 216.03984 regularization 313.72772
loss 529.7676
STEP 47 ================================
prereg loss 227.346 regularization 313.41196
loss 540.75793
STEP 48 ================================
prereg loss 258.29465 regularization 313.126
loss 571.42065
STEP 49 ================================
prereg loss 244.0706 regularization 312.86148
loss 556.93207
STEP 50 ================================
prereg loss 256.65805 regularization 312.48514
loss 569.1432
STEP 51 ================================
prereg loss 231.36208 regularization 312.22028
loss 543.58234
STEP 52 ================================
prereg loss 233.5051 regularization 312.12225
loss 545.6273
STEP 53 ================================
prereg loss 192.79312 regularization 311.76035
loss 504.55347
STEP 54 ================================
prereg loss 210.02017 regularization 311.4355
loss 521.4557
STEP 55 ================================
prereg loss 224.82921 regularization 311.15417
loss 535.9834
STEP 56 ================================
prereg loss 235.52008 regularization 310.90143
loss 546.4215
STEP 57 ================================
prereg loss 217.0039 regularization 310.72086
loss 527.72473
STEP 58 ================================
prereg loss 243.85158 regularization 310.47855
loss 554.33014
STEP 59 ================================
prereg loss 241.93716 regularization 310.315
loss 552.2522
STEP 60 ================================
prereg loss 244.98697 regularization 310.1946
loss 555.1816
STEP 61 ================================
prereg loss 242.65535 regularization 310.15335
loss 552.8087
STEP 62 ================================
prereg loss 232.81949 regularization 310.07346
loss 542.89294
STEP 63 ================================
prereg loss 232.77061 regularization 309.96866
loss 542.73926
STEP 64 ================================
prereg loss 222.73425 regularization 309.9453
loss 532.67957
STEP 65 ================================
prereg loss 237.0262 regularization 309.73843
loss 546.76465
STEP 66 ================================
prereg loss 243.44229 regularization 309.6026
loss 553.0449
STEP 67 ================================
prereg loss 242.49333 regularization 309.53912
loss 552.0325
STEP 68 ================================
prereg loss 247.39752 regularization 309.4767
loss 556.87427
STEP 69 ================================
prereg loss 262.74435 regularization 309.42053
loss 572.1649
STEP 70 ================================
prereg loss 255.74355 regularization 309.374
loss 565.11755
STEP 71 ================================
prereg loss 242.95674 regularization 309.16046
loss 552.1172
STEP 72 ================================
prereg loss 229.36467 regularization 308.97794
loss 538.3426
STEP 73 ================================
prereg loss 228.83144 regularization 308.79623
loss 537.6277
STEP 74 ================================
prereg loss 225.83304 regularization 308.61118
loss 534.4442
STEP 75 ================================
prereg loss 223.3128 regularization 308.41318
loss 531.72595
STEP 76 ================================
prereg loss 220.3872 regularization 308.21136
loss 528.5986
STEP 77 ================================
prereg loss 224.61852 regularization 308.0069
loss 532.6254
STEP 78 ================================
prereg loss 224.21306 regularization 307.80396
loss 532.017
STEP 79 ================================
prereg loss 237.41164 regularization 307.6035
loss 545.01514
STEP 80 ================================
prereg loss 240.83783 regularization 307.41913
loss 548.25696
STEP 81 ================================
prereg loss 245.68784 regularization 307.2522
loss 552.94006
STEP 82 ================================
prereg loss 219.42162 regularization 307.14395
loss 526.56555
STEP 83 ================================
prereg loss 223.26125 regularization 307.02972
loss 530.29095
STEP 84 ================================
prereg loss 223.05362 regularization 306.91064
loss 529.96423
STEP 85 ================================
prereg loss 219.35846 regularization 306.78702
loss 526.1455
STEP 86 ================================
prereg loss 217.43243 regularization 306.67426
loss 524.1067
STEP 87 ================================
prereg loss 217.8465 regularization 306.54837
loss 524.3949
STEP 88 ================================
prereg loss 216.76408 regularization 306.4145
loss 523.1786
STEP 89 ================================
prereg loss 216.78554 regularization 306.2734
loss 523.05896
STEP 90 ================================
prereg loss 218.98137 regularization 306.12366
loss 525.10504
STEP 91 ================================
prereg loss 219.39177 regularization 305.96436
loss 525.35614
STEP 92 ================================
prereg loss 219.19008 regularization 305.80035
loss 524.9904
STEP 93 ================================
prereg loss 218.88756 regularization 305.63412
loss 524.52167
STEP 94 ================================
prereg loss 218.44748 regularization 305.4663
loss 523.9138
STEP 95 ================================
prereg loss 217.93399 regularization 305.2985
loss 523.2325
STEP 96 ================================
prereg loss 217.40594 regularization 305.1315
loss 522.5375
STEP 97 ================================
prereg loss 216.86269 regularization 304.96716
loss 521.82983
STEP 98 ================================
prereg loss 216.34225 regularization 304.80286
loss 521.14514
STEP 99 ================================
prereg loss 215.80899 regularization 304.6388
loss 520.44775
STEP 100 ================================
prereg loss 215.3658 regularization 304.47363
loss 519.8394
STEP 101 ================================
prereg loss 214.77939 regularization 304.30807
loss 519.08746
STEP 102 ================================
prereg loss 214.13924 regularization 304.14337
loss 518.2826
STEP 103 ================================
prereg loss 213.47307 regularization 303.97986
loss 517.45294
STEP 104 ================================
prereg loss 212.88182 regularization 303.81705
loss 516.69885
STEP 105 ================================
prereg loss 212.2843 regularization 303.65634
loss 515.9407
STEP 106 ================================
prereg loss 211.6755 regularization 303.5
loss 515.17554
STEP 107 ================================
prereg loss 211.05249 regularization 303.34467
loss 514.39716
STEP 108 ================================
prereg loss 210.42079 regularization 303.19177
loss 513.61255
STEP 109 ================================
prereg loss 209.78232 regularization 303.04202
loss 512.82434
STEP 110 ================================
prereg loss 209.13821 regularization 302.89407
loss 512.0323
STEP 111 ================================
prereg loss 208.49725 regularization 302.7473
loss 511.24457
STEP 112 ================================
prereg loss 207.8974 regularization 302.60327
loss 510.50067
STEP 113 ================================
prereg loss 207.26768 regularization 302.45895
loss 509.72662
STEP 114 ================================
prereg loss 206.60524 regularization 302.31543
loss 508.92065
STEP 115 ================================
prereg loss 205.96652 regularization 302.1719
loss 508.13843
STEP 116 ================================
prereg loss 205.3516 regularization 302.03055
loss 507.38214
STEP 117 ================================
prereg loss 204.71045 regularization 301.89145
loss 506.6019
STEP 118 ================================
prereg loss 204.05257 regularization 301.75552
loss 505.8081
STEP 119 ================================
prereg loss 203.40825 regularization 301.62
loss 505.02826
STEP 120 ================================
prereg loss 202.78993 regularization 301.48505
loss 504.27496
STEP 121 ================================
prereg loss 202.17719 regularization 301.35117
loss 503.52835
STEP 122 ================================
prereg loss 201.56297 regularization 301.21582
loss 502.7788
STEP 123 ================================
prereg loss 200.95164 regularization 301.08
loss 502.03162
STEP 124 ================================
prereg loss 200.32974 regularization 300.9453
loss 501.27505
STEP 125 ================================
prereg loss 199.68596 regularization 300.81375
loss 500.4997
STEP 126 ================================
prereg loss 198.95631 regularization 300.6813
loss 499.63763
STEP 127 ================================
prereg loss 198.20714 regularization 300.54776
loss 498.75488
STEP 128 ================================
prereg loss 197.45248 regularization 300.41446
loss 497.86694
STEP 129 ================================
prereg loss 196.69096 regularization 300.27994
loss 496.9709
STEP 130 ================================
prereg loss 195.92242 regularization 300.14435
loss 496.06677
STEP 131 ================================
prereg loss 195.146 regularization 300.0095
loss 495.1555
STEP 132 ================================
prereg loss 194.36855 regularization 299.87393
loss 494.2425
STEP 133 ================================
prereg loss 193.57745 regularization 299.7379
loss 493.31537
STEP 134 ================================
prereg loss 192.81528 regularization 299.602
loss 492.41727
STEP 135 ================================
prereg loss 192.05475 regularization 299.46512
loss 491.51987
STEP 136 ================================
prereg loss 191.27945 regularization 299.32965
loss 490.6091
STEP 137 ================================
prereg loss 190.41312 regularization 299.19495
loss 489.60806
STEP 138 ================================
prereg loss 189.56223 regularization 299.05988
loss 488.6221
STEP 139 ================================
prereg loss 188.74995 regularization 298.92578
loss 487.67572
STEP 140 ================================
prereg loss 187.9413 regularization 298.79288
loss 486.7342
STEP 141 ================================
prereg loss 187.16122 regularization 298.66043
loss 485.82166
STEP 142 ================================
prereg loss 186.38518 regularization 298.52792
loss 484.9131
STEP 143 ================================
prereg loss 185.60808 regularization 298.39624
loss 484.00433
STEP 144 ================================
prereg loss 184.82301 regularization 298.26477
loss 483.08777
STEP 145 ================================
prereg loss 184.0564 regularization 298.1328
loss 482.1892
STEP 146 ================================
prereg loss 183.3107 regularization 298.00293
loss 481.31363
STEP 147 ================================
prereg loss 182.57692 regularization 297.87363
loss 480.45056
STEP 148 ================================
prereg loss 181.8699 regularization 297.74527
loss 479.61517
STEP 149 ================================
prereg loss 181.17624 regularization 297.6184
loss 478.79465
STEP 150 ================================
prereg loss 180.5002 regularization 297.49286
loss 477.99304
STEP 151 ================================
prereg loss 179.82977 regularization 297.36942
loss 477.1992
STEP 152 ================================
prereg loss 179.15523 regularization 297.24792
loss 476.40314
STEP 153 ================================
prereg loss 178.43538 regularization 297.12762
loss 475.563
STEP 154 ================================
prereg loss 177.69136 regularization 297.0084
loss 474.69977
STEP 155 ================================
prereg loss 176.93378 regularization 296.8903
loss 473.82407
STEP 156 ================================
prereg loss 176.16147 regularization 296.77283
loss 472.9343
STEP 157 ================================
prereg loss 175.36359 regularization 296.65668
loss 472.02026
STEP 158 ================================
prereg loss 174.55229 regularization 296.54077
loss 471.09308
STEP 159 ================================
prereg loss 173.72893 regularization 296.42404
loss 470.15295
STEP 160 ================================
prereg loss 172.8545 regularization 296.3061
loss 469.16058
STEP 161 ================================
prereg loss 172.00969 regularization 296.189
loss 468.19867
STEP 162 ================================
prereg loss 171.2617 regularization 296.06995
loss 467.33167
STEP 163 ================================
prereg loss 170.57617 regularization 295.94974
loss 466.5259
STEP 164 ================================
prereg loss 169.98218 regularization 295.8295
loss 465.81168
STEP 165 ================================
prereg loss 169.23495 regularization 295.71286
loss 464.9478
STEP 166 ================================
prereg loss 168.63889 regularization 295.5964
loss 464.2353
STEP 167 ================================
prereg loss 168.04924 regularization 295.4773
loss 463.52655
STEP 168 ================================
prereg loss 167.42532 regularization 295.35593
loss 462.78125
STEP 169 ================================
prereg loss 166.99332 regularization 295.2347
loss 462.22803
STEP 170 ================================
prereg loss 166.3914 regularization 295.11905
loss 461.51044
STEP 171 ================================
prereg loss 165.73001 regularization 295.00473
loss 460.73474
STEP 172 ================================
prereg loss 165.20728 regularization 294.88947
loss 460.09674
STEP 173 ================================
prereg loss 164.66006 regularization 294.7729
loss 459.43295
STEP 174 ================================
prereg loss 164.07515 regularization 294.6534
loss 458.72858
STEP 175 ================================
prereg loss 163.46136 regularization 294.53104
loss 457.9924
STEP 176 ================================
prereg loss 162.93243 regularization 294.40906
loss 457.3415
STEP 177 ================================
prereg loss 162.39381 regularization 294.28806
loss 456.6819
STEP 178 ================================
prereg loss 161.84961 regularization 294.16907
loss 456.01868
STEP 179 ================================
prereg loss 161.30632 regularization 294.04892
loss 455.35522
STEP 180 ================================
prereg loss 160.7849 regularization 293.92825
loss 454.71313
STEP 181 ================================
prereg loss 160.2534 regularization 293.8052
loss 454.0586
STEP 182 ================================
prereg loss 159.72684 regularization 293.68216
loss 453.409
STEP 183 ================================
prereg loss 159.2019 regularization 293.55716
loss 452.75906
STEP 184 ================================
prereg loss 158.68085 regularization 293.43033
loss 452.11118
STEP 185 ================================
prereg loss 158.16136 regularization 293.30347
loss 451.46484
STEP 186 ================================
prereg loss 157.63203 regularization 293.17596
loss 450.80798
STEP 187 ================================
prereg loss 157.10336 regularization 293.04843
loss 450.1518
STEP 188 ================================
prereg loss 156.60565 regularization 292.91992
loss 449.52557
STEP 189 ================================
prereg loss 156.10864 regularization 292.79074
loss 448.89938
STEP 190 ================================
prereg loss 155.59694 regularization 292.66238
loss 448.25934
STEP 191 ================================
prereg loss 155.049 regularization 292.5318
loss 447.5808
STEP 192 ================================
prereg loss 154.4567 regularization 292.39917
loss 446.85587
STEP 193 ================================
prereg loss 153.85985 regularization 292.26657
loss 446.1264
STEP 194 ================================
prereg loss 153.27199 regularization 292.13367
loss 445.40564
STEP 195 ================================
prereg loss 152.6719 regularization 292.00055
loss 444.67245
STEP 196 ================================
prereg loss 152.05556 regularization 291.86832
loss 443.9239
STEP 197 ================================
prereg loss 151.39406 regularization 291.73984
loss 443.1339
STEP 198 ================================
prereg loss 150.68811 regularization 291.61472
loss 442.30283
STEP 199 ================================
prereg loss 150.023 regularization 291.48892
loss 441.5119
STEP 200 ================================
prereg loss 149.29582 regularization 291.36145
loss 440.6573
2022-06-04T17:25:19.459

julia> count_neg_interval(sparse, -0.001f0, 0.001f0)
775

julia> count_neg_interval(sparse, -0.01f0, 0.01f0)
690

julia> count_neg_interval(sparse, -0.1f0, 0.1f0)
358

julia> count_neg_interval(sparse, -0.2f0, 0.2f0)
152

julia> count_neg_interval(sparse, -0.3f0, 0.3f0)
60

julia> count_neg_interval(sparse, -0.4f0, 0.4f0)
16

julia> count_neg_interval(sparse, -0.5f0, 0.5f0)
9

julia> count_neg_interval(sparse, -0.6f0, 0.6f0)
5

julia> count_neg_interval(sparse, -0.7f0, 0.7f0)
2

julia> steps!(150)
2022-06-04T17:28:30.510
STEP 1 ================================
prereg loss 148.54103 regularization 291.2339
loss 439.7749
STEP 2 ================================
prereg loss 147.81905 regularization 291.1073
loss 438.92633
STEP 3 ================================
prereg loss 146.976 regularization 290.9785
loss 437.95447
STEP 4 ================================
prereg loss 146.4313 regularization 290.85278
loss 437.2841
STEP 5 ================================
prereg loss 146.51332 regularization 290.7421
loss 437.25543
STEP 6 ================================
prereg loss 146.44017 regularization 290.6254
loss 437.06555
STEP 7 ================================
prereg loss 145.65176 regularization 290.4996
loss 436.15137
STEP 8 ================================
prereg loss 144.17792 regularization 290.3665
loss 434.5444
STEP 9 ================================
prereg loss 142.84912 regularization 290.22818
loss 433.0773
STEP 10 ================================
prereg loss 142.77097 regularization 290.10196
loss 432.87292
STEP 11 ================================
prereg loss 141.57901 regularization 289.99475
loss 431.57376
STEP 12 ================================
prereg loss 141.48567 regularization 289.8824
loss 431.36804
STEP 13 ================================
prereg loss 140.75404 regularization 289.76453
loss 430.51855
STEP 14 ================================
prereg loss 139.5541 regularization 289.64392
loss 429.198
STEP 15 ================================
prereg loss 137.47737 regularization 289.51834
loss 426.99573
STEP 16 ================================
prereg loss 137.60445 regularization 289.4004
loss 427.00482
STEP 17 ================================
prereg loss 136.17288 regularization 289.29916
loss 425.47205
STEP 18 ================================
prereg loss 136.32504 regularization 289.21106
loss 425.5361
STEP 19 ================================
prereg loss 135.66595 regularization 289.11285
loss 424.7788
STEP 20 ================================
prereg loss 133.20009 regularization 289.00662
loss 422.20673
STEP 21 ================================
prereg loss 135.65611 regularization 288.90433
loss 424.56042
STEP 22 ================================
prereg loss 135.65233 regularization 288.84247
loss 424.4948
STEP 23 ================================
prereg loss 138.92256 regularization 288.76938
loss 427.69196
STEP 24 ================================
prereg loss 139.99988 regularization 288.68393
loss 428.6838
STEP 25 ================================
prereg loss 161.75858 regularization 288.5905
loss 450.3491
STEP 26 ================================
prereg loss 161.60191 regularization 288.50372
loss 450.10565
STEP 27 ================================
prereg loss 157.7279 regularization 288.40875
loss 446.13666
STEP 28 ================================
prereg loss 156.06783 regularization 288.36136
loss 444.4292
STEP 29 ================================
prereg loss 155.1949 regularization 288.29694
loss 443.49182
STEP 30 ================================
prereg loss 169.36353 regularization 288.32272
loss 457.68625
STEP 31 ================================
prereg loss 169.59818 regularization 288.3848
loss 457.98297
STEP 32 ================================
prereg loss 170.07262 regularization 288.4795
loss 458.55212
STEP 33 ================================
prereg loss 241.39027 regularization 288.4658
loss 529.8561
STEP 34 ================================
prereg loss 167.94719 regularization 288.0022
loss 455.9494
STEP 35 ================================
prereg loss 182.04005 regularization 287.60083
loss 469.64087
STEP 36 ================================
prereg loss 182.63602 regularization 287.2612
loss 469.89722
STEP 37 ================================
prereg loss 172.39188 regularization 286.9799
loss 459.37177
STEP 38 ================================
prereg loss 154.75159 regularization 286.75687
loss 441.50845
STEP 39 ================================
prereg loss 149.83966 regularization 286.56787
loss 436.40753
STEP 40 ================================
prereg loss 151.617 regularization 286.4101
loss 438.0271
STEP 41 ================================
prereg loss 166.12283 regularization 286.26685
loss 452.38968
STEP 42 ================================
prereg loss 172.55759 regularization 286.1241
loss 458.6817
STEP 43 ================================
prereg loss 181.5551 regularization 285.985
loss 467.5401
STEP 44 ================================
prereg loss 185.28284 regularization 285.84903
loss 471.13187
STEP 45 ================================
prereg loss 184.41199 regularization 285.7086
loss 470.12057
STEP 46 ================================
prereg loss 183.08333 regularization 285.56833
loss 468.65167
STEP 47 ================================
prereg loss 181.51529 regularization 285.43906
loss 466.95435
STEP 48 ================================
prereg loss 178.35124 regularization 285.31287
loss 463.66412
STEP 49 ================================
prereg loss 172.63606 regularization 285.19626
loss 457.83234
STEP 50 ================================
prereg loss 165.98961 regularization 285.08914
loss 451.07874
STEP 51 ================================
prereg loss 161.90004 regularization 284.99472
loss 446.89478
STEP 52 ================================
prereg loss 160.12497 regularization 284.91092
loss 445.0359
STEP 53 ================================
prereg loss 160.10971 regularization 284.84387
loss 444.95358
STEP 54 ================================
prereg loss 159.88152 regularization 284.78812
loss 444.66962
STEP 55 ================================
prereg loss 158.74022 regularization 284.74344
loss 443.48364
STEP 56 ================================
prereg loss 158.22365 regularization 284.70804
loss 442.9317
STEP 57 ================================
prereg loss 158.69499 regularization 284.68442
loss 443.3794
STEP 58 ================================
prereg loss 157.82814 regularization 284.6678
loss 442.4959
STEP 59 ================================
prereg loss 156.25789 regularization 284.66458
loss 440.9225
STEP 60 ================================
prereg loss 155.06053 regularization 284.66602
loss 439.72656
STEP 61 ================================
prereg loss 153.98062 regularization 284.66904
loss 438.64966
STEP 62 ================================
prereg loss 152.94577 regularization 284.67273
loss 437.6185
STEP 63 ================================
prereg loss 152.00034 regularization 284.67422
loss 436.67456
STEP 64 ================================
prereg loss 151.08708 regularization 284.67157
loss 435.75867
STEP 65 ================================
prereg loss 150.16275 regularization 284.66446
loss 434.8272
STEP 66 ================================
prereg loss 148.86287 regularization 284.64865
loss 433.51154
STEP 67 ================================
prereg loss 148.32504 regularization 284.6298
loss 432.95483
STEP 68 ================================
prereg loss 147.45528 regularization 284.61282
loss 432.0681
STEP 69 ================================
prereg loss 146.47455 regularization 284.59467
loss 431.0692
STEP 70 ================================
prereg loss 145.73943 regularization 284.57086
loss 430.3103
STEP 71 ================================
prereg loss 144.88635 regularization 284.5369
loss 429.42325
STEP 72 ================================
prereg loss 144.131 regularization 284.49612
loss 428.62714
STEP 73 ================================
prereg loss 143.32849 regularization 284.45193
loss 427.78043
STEP 74 ================================
prereg loss 142.49933 regularization 284.40442
loss 426.90375
STEP 75 ================================
prereg loss 141.52419 regularization 284.34787
loss 425.87207
STEP 76 ================================
prereg loss 140.61252 regularization 284.28842
loss 424.90094
STEP 77 ================================
prereg loss 139.61166 regularization 284.226
loss 423.83768
STEP 78 ================================
prereg loss 138.62317 regularization 284.1599
loss 422.78308
STEP 79 ================================
prereg loss 137.4068 regularization 284.093
loss 421.4998
STEP 80 ================================
prereg loss 136.17941 regularization 284.02133
loss 420.20074
STEP 81 ================================
prereg loss 134.88002 regularization 283.9501
loss 418.83014
STEP 82 ================================
prereg loss 133.42719 regularization 283.87085
loss 417.29803
STEP 83 ================================
prereg loss 132.24797 regularization 283.7969
loss 416.04486
STEP 84 ================================
prereg loss 132.03958 regularization 283.69803
loss 415.7376
STEP 85 ================================
prereg loss 130.90364 regularization 283.6134
loss 414.51703
STEP 86 ================================
prereg loss 138.15005 regularization 283.52097
loss 421.67102
STEP 87 ================================
prereg loss 150.88858 regularization 283.45364
loss 434.34222
STEP 88 ================================
prereg loss 148.78224 regularization 283.4234
loss 432.20563
STEP 89 ================================
prereg loss 132.39172 regularization 283.43292
loss 415.82465
STEP 90 ================================
prereg loss 124.455086 regularization 283.46445
loss 407.91953
STEP 91 ================================
prereg loss 132.14996 regularization 283.4827
loss 415.63266
STEP 92 ================================
prereg loss 138.12769 regularization 283.4646
loss 421.5923
STEP 93 ================================
prereg loss 120.84888 regularization 283.31476
loss 404.16364
STEP 94 ================================
prereg loss 146.77866 regularization 283.1754
loss 429.95407
STEP 95 ================================
prereg loss 125.832 regularization 283.16965
loss 409.00165
STEP 96 ================================
prereg loss 129.23032 regularization 283.15985
loss 412.39017
STEP 97 ================================
prereg loss 155.20363 regularization 283.11975
loss 438.32336
STEP 98 ================================
prereg loss 160.49759 regularization 283.0037
loss 443.50128
STEP 99 ================================
prereg loss 166.49245 regularization 282.8173
loss 449.30975
STEP 100 ================================
prereg loss 137.81523 regularization 282.5179
loss 420.33313
STEP 101 ================================
prereg loss 124.79477 regularization 282.198
loss 406.99277
STEP 102 ================================
prereg loss 142.92097 regularization 281.90277
loss 424.82373
STEP 103 ================================
prereg loss 170.25996 regularization 281.64386
loss 451.9038
STEP 104 ================================
prereg loss 147.9245 regularization 281.41458
loss 429.33908
STEP 105 ================================
prereg loss 120.143166 regularization 281.21573
loss 401.3589
STEP 106 ================================
prereg loss 118.354675 regularization 281.04028
loss 399.39496
STEP 107 ================================
prereg loss 121.49946 regularization 280.87482
loss 402.37427
STEP 108 ================================
prereg loss 125.7437 regularization 280.7194
loss 406.46307
STEP 109 ================================
prereg loss 124.96611 regularization 280.56586
loss 405.53198
STEP 110 ================================
prereg loss 126.740395 regularization 280.4209
loss 407.1613
STEP 111 ================================
prereg loss 128.1113 regularization 280.28134
loss 408.39264
STEP 112 ================================
prereg loss 129.6214 regularization 280.14856
loss 409.76996
STEP 113 ================================
prereg loss 130.40605 regularization 280.02673
loss 410.4328
STEP 114 ================================
prereg loss 130.48294 regularization 279.9128
loss 410.39575
STEP 115 ================================
prereg loss 130.2894 regularization 279.80338
loss 410.09277
STEP 116 ================================
prereg loss 129.60742 regularization 279.70087
loss 409.3083
STEP 117 ================================
prereg loss 128.80205 regularization 279.6017
loss 408.40375
STEP 118 ================================
prereg loss 127.92966 regularization 279.5062
loss 407.43585
STEP 119 ================================
prereg loss 126.9126 regularization 279.4128
loss 406.3254
STEP 120 ================================
prereg loss 125.922844 regularization 279.3191
loss 405.24194
STEP 121 ================================
prereg loss 124.76697 regularization 279.23315
loss 404.00012
STEP 122 ================================
prereg loss 123.096306 regularization 279.14615
loss 402.24246
STEP 123 ================================
prereg loss 121.14221 regularization 279.0577
loss 400.19992
STEP 124 ================================
prereg loss 119.18288 regularization 278.96643
loss 398.1493
STEP 125 ================================
prereg loss 117.37329 regularization 278.87088
loss 396.24417
STEP 126 ================================
prereg loss 115.71092 regularization 278.76755
loss 394.47845
STEP 127 ================================
prereg loss 114.7073 regularization 278.65384
loss 393.36115
STEP 128 ================================
prereg loss 113.585594 regularization 278.5266
loss 392.1122
STEP 129 ================================
prereg loss 112.379265 regularization 278.38217
loss 390.76144
STEP 130 ================================
prereg loss 110.848946 regularization 278.22418
loss 389.07312
STEP 131 ================================
prereg loss 109.70116 regularization 278.05435
loss 387.7555
STEP 132 ================================
prereg loss 110.48197 regularization 277.89236
loss 388.37433
STEP 133 ================================
prereg loss 107.704834 regularization 277.75424
loss 385.45908
STEP 134 ================================
prereg loss 107.76661 regularization 277.6247
loss 385.3913
STEP 135 ================================
prereg loss 107.77722 regularization 277.47855
loss 385.25577
STEP 136 ================================
prereg loss 105.727295 regularization 277.30917
loss 383.03647
STEP 137 ================================
prereg loss 104.757126 regularization 277.1341
loss 381.89124
STEP 138 ================================
prereg loss 105.01146 regularization 276.9765
loss 381.98798
STEP 139 ================================
prereg loss 103.448616 regularization 276.8482
loss 380.2968
STEP 140 ================================
prereg loss 103.090324 regularization 276.70212
loss 379.79245
STEP 141 ================================
prereg loss 102.191376 regularization 276.53894
loss 378.73032
STEP 142 ================================
prereg loss 100.98712 regularization 276.4032
loss 377.39032
STEP 143 ================================
prereg loss 102.087685 regularization 276.28418
loss 378.37186
STEP 144 ================================
prereg loss 101.169174 regularization 276.12344
loss 377.2926
STEP 145 ================================
prereg loss 105.13144 regularization 276.02112
loss 381.15256
STEP 146 ================================
prereg loss 103.73346 regularization 275.8795
loss 379.61295
STEP 147 ================================
prereg loss 101.91438 regularization 275.68585
loss 377.60022
STEP 148 ================================
prereg loss 99.235115 regularization 275.45602
loss 374.69113
STEP 149 ================================
prereg loss 97.585464 regularization 275.20334
loss 372.78882
STEP 150 ================================
prereg loss 97.35276 regularization 274.9397
loss 372.29245
2022-06-04T20:54:08.817

julia>
```

So, not a bad progress after 366 steps, but still far from where
we need this to be. Let's make a checkpoint (not much change
in the sparsity pattern), and do another 634 steps for the full
1000 steps, and then look again:

```
julia> a_366 = deepcopy(sparse)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0091177), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.155428, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-6.84192f-5, "dict-2"=>…
  "dot-2"     => Dict("dict"=>Dict("norm-3"=>Dict("true"=>0.000182705, "dict"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.179013), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "n…
  "accum-3"   => Dict("false"=>Dict("compare-4"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.189588), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "compare-4" => Dict("dot"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>5.50746f-5, "false"=>0.0, "dict-2"=>…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "no…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0990709), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.151016), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0341402), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.300257), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "do…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.684889), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.160623), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-0.000124237, "dict-2…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-6.465f-5, "dict-2"=…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.445548), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…

julia> count_neg_interval(sparse, -0.001f0, 0.001f0)
769

julia> count_neg_interval(sparse, -0.01f0, 0.01f0)
683

julia> count_neg_interval(sparse, -0.1f0, 0.1f0)
338

julia> count_neg_interval(sparse, -0.2f0, 0.2f0)
142

julia> count_neg_interval(sparse, -0.3f0, 0.3f0)
56

julia> count_neg_interval(sparse, -0.4f0, 0.4f0)
16

julia> count_neg_interval(sparse, -0.5f0, 0.5f0)
9

julia> count_neg_interval(sparse, -0.6f0, 0.6f0)
4

julia> count_neg_interval(sparse, -0.7f0, 0.7f0)
2

julia> serialize("366-steps-matrix.ser", sparse)

julia> open("366-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(sparse))
           println(f)
           end

julia>
```

634 more steps:

```
julia> steps!(634)
2022-06-04T21:13:29.147
STEP 1 ================================
prereg loss 100.4018 regularization 274.71423
loss 375.11603
STEP 2 ================================
prereg loss 102.46179 regularization 274.53577
loss 376.99756
STEP 3 ================================
prereg loss 102.27977 regularization 274.4031
loss 376.68286
STEP 4 ================================
prereg loss 99.51368 regularization 274.31464
loss 373.8283
STEP 5 ================================
prereg loss 96.01227 regularization 274.26544
loss 370.2777
STEP 6 ================================
prereg loss 97.34447 regularization 274.22012
loss 371.56458
STEP 7 ================================
prereg loss 96.4875 regularization 274.12698
loss 370.6145
STEP 8 ================================
prereg loss 92.9805 regularization 273.97247
loss 366.95297
STEP 9 ================================
prereg loss 92.29272 regularization 273.79498
loss 366.0877
STEP 10 ================================
prereg loss 95.143364 regularization 273.65756
loss 368.80093
STEP 11 ================================
prereg loss 94.34258 regularization 273.5659
loss 367.90848
STEP 12 ================================
prereg loss 90.1707 regularization 273.5163
loss 363.687
STEP 13 ================================
prereg loss 102.79615 regularization 273.48303
loss 376.27917
STEP 14 ================================
prereg loss 90.44287 regularization 273.2763
loss 363.71918
STEP 15 ================================
prereg loss 93.27713 regularization 273.11542
loss 366.39255
STEP 16 ================================
prereg loss 90.886154 regularization 273.00848
loss 363.89465
STEP 17 ================================
prereg loss 88.757355 regularization 272.94046
loss 361.6978
STEP 18 ================================
prereg loss 90.42285 regularization 272.8532
loss 363.27606
STEP 19 ================================
prereg loss 89.690834 regularization 272.72635
loss 362.41718
STEP 20 ================================
prereg loss 88.4124 regularization 272.5698
loss 360.98218
STEP 21 ================================
prereg loss 89.48862 regularization 272.41147
loss 361.9001
STEP 22 ================================
prereg loss 89.14221 regularization 272.30185
loss 361.44406
STEP 23 ================================
prereg loss 88.2415 regularization 272.23547
loss 360.477
STEP 24 ================================
prereg loss 88.59727 regularization 272.15247
loss 360.74973
STEP 25 ================================
prereg loss 87.609 regularization 272.0392
loss 359.64822
STEP 26 ================================
prereg loss 86.70072 regularization 271.90503
loss 358.60574
STEP 27 ================================
prereg loss 86.570816 regularization 271.79956
loss 358.37036
STEP 28 ================================
prereg loss 85.55233 regularization 271.72626
loss 357.2786
STEP 29 ================================
prereg loss 85.55849 regularization 271.64044
loss 357.1989
STEP 30 ================================
prereg loss 84.771095 regularization 271.52872
loss 356.2998
STEP 31 ================================
prereg loss 84.63506 regularization 271.41675
loss 356.05182
STEP 32 ================================
prereg loss 84.13032 regularization 271.32556
loss 355.45587
STEP 33 ================================
prereg loss 83.80952 regularization 271.24683
loss 355.05634
STEP 34 ================================
prereg loss 83.40434 regularization 271.14624
loss 354.5506
STEP 35 ================================
prereg loss 83.42616 regularization 271.0335
loss 354.45966
STEP 36 ================================
prereg loss 82.87005 regularization 270.9489
loss 353.81897
STEP 37 ================================
prereg loss 82.4682 regularization 270.84253
loss 353.31073
STEP 38 ================================
prereg loss 83.040504 regularization 270.7269
loss 353.7674
STEP 39 ================================
prereg loss 95.60797 regularization 270.6676
loss 366.27557
STEP 40 ================================
prereg loss 100.831665 regularization 270.3898
loss 371.22147
STEP 41 ================================
prereg loss 113.489456 regularization 270.18182
loss 383.67126
STEP 42 ================================
prereg loss 117.878815 regularization 270.0448
loss 387.9236
STEP 43 ================================
prereg loss 118.26915 regularization 269.96747
loss 388.23663
STEP 44 ================================
prereg loss 114.82124 regularization 269.95142
loss 384.77264
STEP 45 ================================
prereg loss 107.2926 regularization 269.98865
loss 377.28125
STEP 46 ================================
prereg loss 92.73317 regularization 270.0904
loss 362.82355
STEP 47 ================================
prereg loss 92.71209 regularization 270.24496
loss 362.95706
STEP 48 ================================
prereg loss 102.85913 regularization 270.3258
loss 373.18494
STEP 49 ================================
prereg loss 100.60712 regularization 270.3212
loss 370.9283
STEP 50 ================================
prereg loss 96.41232 regularization 270.17584
loss 366.58817
STEP 51 ================================
prereg loss 109.617 regularization 270.07895
loss 379.69595
STEP 52 ================================
prereg loss 120.73325 regularization 270.03238
loss 390.76562
STEP 53 ================================
prereg loss 108.58386 regularization 270.05023
loss 378.6341
STEP 54 ================================
prereg loss 93.78098 regularization 270.12134
loss 363.9023
STEP 55 ================================
prereg loss 113.97194 regularization 270.17792
loss 384.14984
STEP 56 ================================
prereg loss 122.49033 regularization 270.16003
loss 392.65036
STEP 57 ================================
prereg loss 115.45517 regularization 270.03445
loss 385.48962
STEP 58 ================================
prereg loss 102.72256 regularization 269.83878
loss 372.56134
STEP 59 ================================
prereg loss 97.14611 regularization 269.607
loss 366.7531
STEP 60 ================================
prereg loss 97.75265 regularization 269.37457
loss 367.12723
STEP 61 ================================
prereg loss 101.58106 regularization 269.1508
loss 370.73184
STEP 62 ================================
prereg loss 105.36563 regularization 268.94183
loss 374.30746
STEP 63 ================================
prereg loss 108.819016 regularization 268.75446
loss 377.5735
STEP 64 ================================
prereg loss 111.135666 regularization 268.59256
loss 379.7282
STEP 65 ================================
prereg loss 112.019806 regularization 268.45648
loss 380.4763
STEP 66 ================================
prereg loss 111.46241 regularization 268.34534
loss 379.80774
STEP 67 ================================
prereg loss 109.588936 regularization 268.25742
loss 377.84634
STEP 68 ================================
prereg loss 106.6731 regularization 268.19196
loss 374.86505
STEP 69 ================================
prereg loss 102.98312 regularization 268.14233
loss 371.12546
STEP 70 ================================
prereg loss 99.034004 regularization 268.106
loss 367.13998
STEP 71 ================================
prereg loss 95.55744 regularization 268.07816
loss 363.6356
STEP 72 ================================
prereg loss 93.01027 regularization 268.05112
loss 361.0614
STEP 73 ================================
prereg loss 91.39339 regularization 268.01483
loss 359.4082
STEP 74 ================================
prereg loss 90.74791 regularization 267.96298
loss 358.71088
STEP 75 ================================
prereg loss 91.0101 regularization 267.88782
loss 358.89792
STEP 76 ================================
prereg loss 91.36776 regularization 267.78543
loss 359.1532
STEP 77 ================================
prereg loss 88.13215 regularization 267.64902
loss 355.78116
STEP 78 ================================
prereg loss 94.138596 regularization 267.46356
loss 361.60217
STEP 79 ================================
prereg loss 90.25123 regularization 267.27383
loss 357.52505
STEP 80 ================================
prereg loss 89.57317 regularization 267.0771
loss 356.65027
STEP 81 ================================
prereg loss 90.8077 regularization 266.9056
loss 357.71332
STEP 82 ================================
prereg loss 89.79793 regularization 266.7864
loss 356.58435
STEP 83 ================================
prereg loss 85.43982 regularization 266.71112
loss 352.15094
STEP 84 ================================
prereg loss 83.26427 regularization 266.6573
loss 349.92157
STEP 85 ================================
prereg loss 81.3428 regularization 266.57355
loss 347.91635
STEP 86 ================================
prereg loss 81.46967 regularization 266.48474
loss 347.9544
STEP 87 ================================
prereg loss 80.69285 regularization 266.42178
loss 347.11462
STEP 88 ================================
prereg loss 80.77599 regularization 266.35223
loss 347.12823
STEP 89 ================================
prereg loss 83.5188 regularization 266.2405
loss 349.7593
STEP 90 ================================
prereg loss 86.730316 regularization 266.19766
loss 352.92798
STEP 91 ================================
prereg loss 123.524055 regularization 266.01675
loss 389.5408
STEP 92 ================================
prereg loss 125.704666 regularization 265.3349
loss 391.03955
STEP 93 ================================
prereg loss 159.87746 regularization 264.7793
loss 424.65674
STEP 94 ================================
prereg loss 176.67789 regularization 264.31323
loss 440.99112
STEP 95 ================================
prereg loss 194.14249 regularization 263.92902
loss 458.0715
STEP 96 ================================
prereg loss 203.45793 regularization 263.61774
loss 467.07568
STEP 97 ================================
prereg loss 206.40063 regularization 263.36395
loss 469.7646
STEP 98 ================================
prereg loss 206.22519 regularization 263.15616
loss 469.38135
STEP 99 ================================
prereg loss 204.31767 regularization 262.98718
loss 467.30487
STEP 100 ================================
prereg loss 202.05838 regularization 262.85446
loss 464.91284
STEP 101 ================================
prereg loss 200.00688 regularization 262.75552
loss 462.7624
STEP 102 ================================
prereg loss 198.33107 regularization 262.6874
loss 461.0185
STEP 103 ================================
prereg loss 196.89853 regularization 262.64258
loss 459.5411
STEP 104 ================================
prereg loss 195.44157 regularization 262.6177
loss 458.05927
STEP 105 ================================
prereg loss 193.76735 regularization 262.60776
loss 456.37512
STEP 106 ================================
prereg loss 192.57689 regularization 262.6125
loss 455.1894
STEP 107 ================================
prereg loss 191.55196 regularization 262.62683
loss 454.17877
STEP 108 ================================
prereg loss 190.4793 regularization 262.65012
loss 453.1294
STEP 109 ================================
prereg loss 189.24648 regularization 262.67932
loss 451.92578
STEP 110 ================================
prereg loss 187.86208 regularization 262.71417
loss 450.57623
STEP 111 ================================
prereg loss 186.3414 regularization 262.7544
loss 449.0958
STEP 112 ================================
prereg loss 184.67082 regularization 262.79803
loss 447.46887
STEP 113 ================================
prereg loss 182.91307 regularization 262.84787
loss 445.76093
STEP 114 ================================
prereg loss 181.10455 regularization 262.90015
loss 444.0047
STEP 115 ================================
prereg loss 179.18753 regularization 262.95337
loss 442.1409
STEP 116 ================================
prereg loss 177.1578 regularization 263.00824
loss 440.16605
STEP 117 ================================
prereg loss 175.0409 regularization 263.06503
loss 438.10593
STEP 118 ================================
prereg loss 172.87485 regularization 263.12137
loss 435.99622
STEP 119 ================================
prereg loss 170.6723 regularization 263.1777
loss 433.85
STEP 120 ================================
prereg loss 168.48975 regularization 263.23367
loss 431.72342
STEP 121 ================================
prereg loss 166.34273 regularization 263.28796
loss 429.63068
STEP 122 ================================
prereg loss 164.16867 regularization 263.34268
loss 427.51135
STEP 123 ================================
prereg loss 161.963 regularization 263.3973
loss 425.3603
STEP 124 ================================
prereg loss 159.73125 regularization 263.44937
loss 423.1806
STEP 125 ================================
prereg loss 157.4961 regularization 263.50046
loss 420.99655
STEP 126 ================================
prereg loss 155.3207 regularization 263.55008
loss 418.8708
STEP 127 ================================
prereg loss 153.24942 regularization 263.5975
loss 416.84692
STEP 128 ================================
prereg loss 151.27312 regularization 263.6445
loss 414.9176
STEP 129 ================================
prereg loss 149.33554 regularization 263.68826
loss 413.0238
STEP 130 ================================
prereg loss 147.41783 regularization 263.73102
loss 411.14886
STEP 131 ================================
prereg loss 145.52028 regularization 263.7743
loss 409.29456
STEP 132 ================================
prereg loss 143.67801 regularization 263.81696
loss 407.49496
STEP 133 ================================
prereg loss 141.85478 regularization 263.85797
loss 405.71277
STEP 134 ================================
prereg loss 140.03516 regularization 263.89893
loss 403.93408
STEP 135 ================================
prereg loss 138.27853 regularization 263.93826
loss 402.2168
STEP 136 ================================
prereg loss 136.55582 regularization 263.976
loss 400.53183
STEP 137 ================================
prereg loss 134.87843 regularization 264.0135
loss 398.8919
STEP 138 ================================
prereg loss 133.24008 regularization 264.05048
loss 397.29056
STEP 139 ================================
prereg loss 131.64377 regularization 264.0847
loss 395.72845
STEP 140 ================================
prereg loss 130.09535 regularization 264.11703
loss 394.2124
STEP 141 ================================
prereg loss 128.5947 regularization 264.14526
loss 392.73996
STEP 142 ================================
prereg loss 127.13441 regularization 264.17203
loss 391.30643
STEP 143 ================================
prereg loss 125.70333 regularization 264.19748
loss 389.90082
STEP 144 ================================
prereg loss 124.30016 regularization 264.22092
loss 388.5211
STEP 145 ================================
prereg loss 123.00384 regularization 264.24243
loss 387.24628
STEP 146 ================================
prereg loss 121.802956 regularization 264.261
loss 386.06393
STEP 147 ================================
prereg loss 120.644875 regularization 264.27417
loss 384.91904
STEP 148 ================================
prereg loss 119.45182 regularization 264.2826
loss 383.7344
STEP 149 ================================
prereg loss 118.18291 regularization 264.2844
loss 382.4673
STEP 150 ================================
prereg loss 116.90379 regularization 264.2807
loss 381.1845
STEP 151 ================================
prereg loss 115.749535 regularization 264.27197
loss 380.0215
STEP 152 ================================
prereg loss 114.715744 regularization 264.25952
loss 378.97528
STEP 153 ================================
prereg loss 113.81032 regularization 264.24246
loss 378.0528
STEP 154 ================================
prereg loss 112.98772 regularization 264.2209
loss 377.20862
STEP 155 ================================
prereg loss 112.24799 regularization 264.19406
loss 376.44205
STEP 156 ================================
prereg loss 111.56957 regularization 264.1624
loss 375.732
STEP 157 ================================
prereg loss 110.95263 regularization 264.12317
loss 375.0758
STEP 158 ================================
prereg loss 110.39852 regularization 264.07773
loss 374.47626
STEP 159 ================================
prereg loss 109.88073 regularization 264.02435
loss 373.9051
STEP 160 ================================
prereg loss 109.439705 regularization 263.96347
loss 373.40317
STEP 161 ================================
prereg loss 109.091156 regularization 263.8938
loss 372.98495
STEP 162 ================================
prereg loss 108.85298 regularization 263.81735
loss 372.67035
STEP 163 ================================
prereg loss 108.665955 regularization 263.73062
loss 372.39658
STEP 164 ================================
prereg loss 108.430275 regularization 263.63376
loss 372.06403
STEP 165 ================================
prereg loss 108.11203 regularization 263.52597
loss 371.638
STEP 166 ================================
prereg loss 107.74234 regularization 263.4086
loss 371.15094
STEP 167 ================================
prereg loss 107.28505 regularization 263.28314
loss 370.56818
STEP 168 ================================
prereg loss 106.771645 regularization 263.1503
loss 369.92194
STEP 169 ================================
prereg loss 106.28197 regularization 263.01105
loss 369.29303
STEP 170 ================================
prereg loss 105.82889 regularization 262.8673
loss 368.6962
STEP 171 ================================
prereg loss 105.51882 regularization 262.72305
loss 368.24188
STEP 172 ================================
prereg loss 105.2594 regularization 262.58008
loss 367.83948
STEP 173 ================================
prereg loss 105.05532 regularization 262.43793
loss 367.49326
STEP 174 ================================
prereg loss 104.75185 regularization 262.30075
loss 367.0526
STEP 175 ================================
prereg loss 104.269516 regularization 262.168
loss 366.4375
STEP 176 ================================
prereg loss 103.89561 regularization 262.03876
loss 365.93436
STEP 177 ================================
prereg loss 103.64259 regularization 261.91006
loss 365.55267
STEP 178 ================================
prereg loss 103.39587 regularization 261.78073
loss 365.1766
STEP 179 ================================
prereg loss 103.112465 regularization 261.6531
loss 364.76556
STEP 180 ================================
prereg loss 102.798485 regularization 261.52567
loss 364.32416
STEP 181 ================================
prereg loss 102.448944 regularization 261.39734
loss 363.84628
STEP 182 ================================
prereg loss 102.07712 regularization 261.2703
loss 363.3474
STEP 183 ================================
prereg loss 101.78523 regularization 261.14655
loss 362.93176
STEP 184 ================================
prereg loss 101.48332 regularization 261.02628
loss 362.50958
STEP 185 ================================
prereg loss 101.15036 regularization 260.90805
loss 362.0584
STEP 186 ================================
prereg loss 100.76078 regularization 260.7948
loss 361.55557
STEP 187 ================================
prereg loss 100.38812 regularization 260.68372
loss 361.07184
STEP 188 ================================
prereg loss 100.060356 regularization 260.5724
loss 360.63275
STEP 189 ================================
prereg loss 99.70139 regularization 260.45758
loss 360.15897
STEP 190 ================================
prereg loss 99.31536 regularization 260.34076
loss 359.65613
STEP 191 ================================
prereg loss 98.9747 regularization 260.22433
loss 359.19904
STEP 192 ================================
prereg loss 98.6313 regularization 260.1089
loss 358.74017
STEP 193 ================================
prereg loss 98.25708 regularization 259.99356
loss 358.25064
STEP 194 ================================
prereg loss 97.93241 regularization 259.87735
loss 357.80975
STEP 195 ================================
prereg loss 97.60438 regularization 259.7575
loss 357.36188
STEP 196 ================================
prereg loss 97.25452 regularization 259.63315
loss 356.88766
STEP 197 ================================
prereg loss 96.95367 regularization 259.505
loss 356.45868
STEP 198 ================================
prereg loss 96.6209 regularization 259.37793
loss 355.99884
STEP 199 ================================
prereg loss 96.24969 regularization 259.2504
loss 355.5001
STEP 200 ================================
prereg loss 95.919205 regularization 259.1237
loss 355.0429
STEP 201 ================================
prereg loss 95.59613 regularization 258.9924
loss 354.58853
STEP 202 ================================
prereg loss 95.23794 regularization 258.85794
loss 354.0959
STEP 203 ================================
prereg loss 94.887665 regularization 258.7215
loss 353.60916
STEP 204 ================================
prereg loss 94.58114 regularization 258.58676
loss 353.1679
STEP 205 ================================
prereg loss 94.17073 regularization 258.45364
loss 352.6244
STEP 206 ================================
prereg loss 93.803535 regularization 258.3188
loss 352.1223
STEP 207 ================================
prereg loss 93.476585 regularization 258.18234
loss 351.65894
STEP 208 ================================
prereg loss 93.073944 regularization 258.04883
loss 351.12277
STEP 209 ================================
prereg loss 92.69301 regularization 257.91113
loss 350.60413
STEP 210 ================================
prereg loss 92.347244 regularization 257.77527
loss 350.1225
STEP 211 ================================
prereg loss 91.9325 regularization 257.6349
loss 349.56738
STEP 212 ================================
prereg loss 91.53646 regularization 257.498
loss 349.03445
STEP 213 ================================
prereg loss 91.19358 regularization 257.35614
loss 348.5497
STEP 214 ================================
prereg loss 90.777214 regularization 257.2176
loss 347.9948
STEP 215 ================================
prereg loss 90.34481 regularization 257.07562
loss 347.42044
STEP 216 ================================
prereg loss 90.19883 regularization 256.93097
loss 347.1298
STEP 217 ================================
prereg loss 89.66408 regularization 256.78992
loss 346.45398
STEP 218 ================================
prereg loss 89.38442 regularization 256.65417
loss 346.0386
STEP 219 ================================
prereg loss 89.02678 regularization 256.5096
loss 345.53638
STEP 220 ================================
prereg loss 88.31834 regularization 256.35754
loss 344.67587
STEP 221 ================================
prereg loss 87.93313 regularization 256.21118
loss 344.14432
STEP 222 ================================
prereg loss 87.606995 regularization 256.07053
loss 343.67752
STEP 223 ================================
prereg loss 87.12644 regularization 255.92455
loss 343.051
STEP 224 ================================
prereg loss 86.73814 regularization 255.77333
loss 342.51147
STEP 225 ================================
prereg loss 86.19821 regularization 255.62901
loss 341.8272
STEP 226 ================================
prereg loss 86.08391 regularization 255.49301
loss 341.5769
STEP 227 ================================
prereg loss 85.39727 regularization 255.3452
loss 340.74246
STEP 228 ================================
prereg loss 84.981384 regularization 255.19319
loss 340.17456
STEP 229 ================================
prereg loss 84.57179 regularization 255.04825
loss 339.62006
STEP 230 ================================
prereg loss 84.01716 regularization 254.91069
loss 338.92786
STEP 231 ================================
prereg loss 83.73818 regularization 254.76802
loss 338.5062
STEP 232 ================================
prereg loss 83.10467 regularization 254.6133
loss 337.71796
STEP 233 ================================
prereg loss 82.82566 regularization 254.45424
loss 337.2799
STEP 234 ================================
prereg loss 82.24998 regularization 254.30353
loss 336.5535
STEP 235 ================================
prereg loss 82.05128 regularization 254.15912
loss 336.2104
STEP 236 ================================
prereg loss 81.42724 regularization 254.00073
loss 335.42798
STEP 237 ================================
prereg loss 81.119736 regularization 253.8319
loss 334.95163
STEP 238 ================================
prereg loss 80.75411 regularization 253.67488
loss 334.429
STEP 239 ================================
prereg loss 79.97165 regularization 253.52988
loss 333.50153
STEP 240 ================================
prereg loss 80.19989 regularization 253.38904
loss 333.58893
STEP 241 ================================
prereg loss 79.184395 regularization 253.23148
loss 332.41586
STEP 242 ================================
prereg loss 79.230316 regularization 253.0665
loss 332.2968
STEP 243 ================================
prereg loss 79.067795 regularization 252.91774
loss 331.98553
STEP 244 ================================
prereg loss 78.29883 regularization 252.78476
loss 331.0836
STEP 245 ================================
prereg loss 77.55935 regularization 252.65912
loss 330.21848
STEP 246 ================================
prereg loss 77.53511 regularization 252.52045
loss 330.05554
STEP 247 ================================
prereg loss 76.70356 regularization 252.3586
loss 329.06216
STEP 248 ================================
prereg loss 76.537506 regularization 252.18935
loss 328.72687
STEP 249 ================================
prereg loss 76.30541 regularization 252.03331
loss 328.3387
STEP 250 ================================
prereg loss 75.72168 regularization 251.89476
loss 327.61646
STEP 251 ================================
prereg loss 75.35702 regularization 251.76927
loss 327.12628
STEP 252 ================================
prereg loss 74.85759 regularization 251.63081
loss 326.4884
STEP 253 ================================
prereg loss 74.589066 regularization 251.48212
loss 326.07117
STEP 254 ================================
prereg loss 74.34892 regularization 251.34612
loss 325.69504
STEP 255 ================================
prereg loss 73.82537 regularization 251.22388
loss 325.04926
STEP 256 ================================
prereg loss 73.57508 regularization 251.10585
loss 324.68094
STEP 257 ================================
prereg loss 73.20588 regularization 250.97028
loss 324.17615
STEP 258 ================================
prereg loss 72.69309 regularization 250.8208
loss 323.5139
STEP 259 ================================
prereg loss 72.70536 regularization 250.66412
loss 323.36948
STEP 260 ================================
prereg loss 72.3368 regularization 250.53351
loss 322.8703
STEP 261 ================================
prereg loss 71.97628 regularization 250.4266
loss 322.4029
STEP 262 ================================
prereg loss 71.737564 regularization 250.30824
loss 322.0458
STEP 263 ================================
prereg loss 71.118065 regularization 250.17455
loss 321.2926
STEP 264 ================================
prereg loss 70.969604 regularization 250.03593
loss 321.00555
STEP 265 ================================
prereg loss 70.68273 regularization 249.9109
loss 320.59363
STEP 266 ================================
prereg loss 70.25119 regularization 249.80084
loss 320.05203
STEP 267 ================================
prereg loss 70.00593 regularization 249.68375
loss 319.68967
STEP 268 ================================
prereg loss 69.662224 regularization 249.54327
loss 319.2055
STEP 269 ================================
prereg loss 69.28983 regularization 249.42099
loss 318.71082
STEP 270 ================================
prereg loss 68.938416 regularization 249.31526
loss 318.25366
STEP 271 ================================
prereg loss 68.486435 regularization 249.18658
loss 317.67303
STEP 272 ================================
prereg loss 68.26778 regularization 249.05061
loss 317.3184
STEP 273 ================================
prereg loss 67.82805 regularization 248.93213
loss 316.7602
STEP 274 ================================
prereg loss 67.58779 regularization 248.82373
loss 316.41153
STEP 275 ================================
prereg loss 67.15181 regularization 248.69125
loss 315.84308
STEP 276 ================================
prereg loss 66.94012 regularization 248.55638
loss 315.4965
STEP 277 ================================
prereg loss 66.61987 regularization 248.434
loss 315.0539
STEP 278 ================================
prereg loss 66.33197 regularization 248.32245
loss 314.65442
STEP 279 ================================
prereg loss 66.029564 regularization 248.19667
loss 314.22623
STEP 280 ================================
prereg loss 65.849724 regularization 248.06047
loss 313.9102
STEP 281 ================================
prereg loss 65.514084 regularization 247.9424
loss 313.45648
STEP 282 ================================
prereg loss 65.53388 regularization 247.83449
loss 313.36838
STEP 283 ================================
prereg loss 66.44726 regularization 247.73814
loss 314.1854
STEP 284 ================================
prereg loss 65.20714 regularization 247.57353
loss 312.78067
STEP 285 ================================
prereg loss 65.70658 regularization 247.40436
loss 313.11093
STEP 286 ================================
prereg loss 65.105064 regularization 247.31566
loss 312.42072
STEP 287 ================================
prereg loss 64.12243 regularization 247.21657
loss 311.339
STEP 288 ================================
prereg loss 63.96256 regularization 247.08646
loss 311.049
STEP 289 ================================
prereg loss 63.471394 regularization 246.97627
loss 310.44766
STEP 290 ================================
prereg loss 64.28617 regularization 246.86183
loss 311.148
STEP 291 ================================
prereg loss 65.5096 regularization 246.65161
loss 312.1612
STEP 292 ================================
prereg loss 63.065826 regularization 246.54712
loss 309.61295
STEP 293 ================================
prereg loss 63.363472 regularization 246.47118
loss 309.83466
STEP 294 ================================
prereg loss 62.42757 regularization 246.30753
loss 308.7351
STEP 295 ================================
prereg loss 63.59825 regularization 246.17621
loss 309.77448
STEP 296 ================================
prereg loss 62.217163 regularization 246.12231
loss 308.33948
STEP 297 ================================
prereg loss 61.515408 regularization 245.99933
loss 307.51474
STEP 298 ================================
prereg loss 65.94825 regularization 245.9117
loss 311.85995
STEP 299 ================================
prereg loss 81.203896 regularization 245.62129
loss 326.8252
STEP 300 ================================
prereg loss 86.0304 regularization 245.48888
loss 331.5193
STEP 301 ================================
prereg loss 67.559784 regularization 245.48564
loss 313.0454
STEP 302 ================================
prereg loss 66.19523 regularization 245.53436
loss 311.72958
STEP 303 ================================
prereg loss 107.03044 regularization 245.51395
loss 352.54437
STEP 304 ================================
prereg loss 103.229126 regularization 245.02505
loss 348.25418
STEP 305 ================================
prereg loss 139.10365 regularization 244.67876
loss 383.7824
STEP 306 ================================
prereg loss 156.54222 regularization 244.42892
loss 400.97113
STEP 307 ================================
prereg loss 166.1949 regularization 244.26129
loss 410.45618
STEP 308 ================================
prereg loss 165.16888 regularization 244.16664
loss 409.3355
STEP 309 ================================
prereg loss 158.44943 regularization 244.13174
loss 402.58118
STEP 310 ================================
prereg loss 147.87085 regularization 244.14516
loss 392.016
STEP 311 ================================
prereg loss 143.47331 regularization 244.20029
loss 387.67358
STEP 312 ================================
prereg loss 138.91513 regularization 244.28918
loss 383.2043
STEP 313 ================================
prereg loss 133.01437 regularization 244.41295
loss 377.4273
STEP 314 ================================
prereg loss 127.474815 regularization 244.56374
loss 372.03854
STEP 315 ================================
prereg loss 121.384384 regularization 244.73776
loss 366.12213
STEP 316 ================================
prereg loss 114.84155 regularization 244.92757
loss 359.7691
STEP 317 ================================
prereg loss 108.090385 regularization 245.13062
loss 353.221
STEP 318 ================================
prereg loss 102.558846 regularization 245.33069
loss 347.88953
STEP 319 ================================
prereg loss 96.39724 regularization 245.52916
loss 341.9264
STEP 320 ================================
prereg loss 90.41434 regularization 245.72139
loss 336.13574
STEP 321 ================================
prereg loss 88.614914 regularization 245.90045
loss 334.51538
STEP 322 ================================
prereg loss 94.85964 regularization 246.02502
loss 340.88467
STEP 323 ================================
prereg loss 97.38978 regularization 246.08054
loss 343.4703
STEP 324 ================================
prereg loss 94.84507 regularization 246.07034
loss 340.9154
STEP 325 ================================
prereg loss 90.308586 regularization 245.97345
loss 336.28204
STEP 326 ================================
prereg loss 91.77264 regularization 245.8412
loss 337.61383
STEP 327 ================================
prereg loss 86.42812 regularization 245.66971
loss 332.09784
STEP 328 ================================
prereg loss 88.19443 regularization 245.50801
loss 333.70245
STEP 329 ================================
prereg loss 88.95287 regularization 245.35506
loss 334.30792
STEP 330 ================================
prereg loss 89.20412 regularization 245.21819
loss 334.4223
STEP 331 ================================
prereg loss 88.95729 regularization 245.09598
loss 334.05328
STEP 332 ================================
prereg loss 88.2745 regularization 244.98714
loss 333.26163
STEP 333 ================================
prereg loss 87.133575 regularization 244.89145
loss 332.02502
STEP 334 ================================
prereg loss 85.395996 regularization 244.81023
loss 330.20624
STEP 335 ================================
prereg loss 83.113106 regularization 244.7429
loss 327.85602
STEP 336 ================================
prereg loss 79.803 regularization 244.68639
loss 324.48938
STEP 337 ================================
prereg loss 80.18041 regularization 244.61484
loss 324.79526
STEP 338 ================================
prereg loss 79.004845 regularization 244.51436
loss 323.5192
STEP 339 ================================
prereg loss 77.95071 regularization 244.38004
loss 322.33075
STEP 340 ================================
prereg loss 77.85341 regularization 244.24721
loss 322.10062
STEP 341 ================================
prereg loss 77.16035 regularization 244.11069
loss 321.27103
STEP 342 ================================
prereg loss 76.40139 regularization 243.97011
loss 320.3715
STEP 343 ================================
prereg loss 75.229904 regularization 243.80594
loss 319.03583
STEP 344 ================================
prereg loss 74.16443 regularization 243.64188
loss 317.8063
STEP 345 ================================
prereg loss 73.423325 regularization 243.48346
loss 316.9068
STEP 346 ================================
prereg loss 72.570595 regularization 243.30821
loss 315.8788
STEP 347 ================================
prereg loss 72.03066 regularization 243.1552
loss 315.18585
STEP 348 ================================
prereg loss 71.50589 regularization 242.9965
loss 314.50238
STEP 349 ================================
prereg loss 70.994156 regularization 242.8607
loss 313.85486
STEP 350 ================================
prereg loss 70.28643 regularization 242.72255
loss 313.00897
STEP 351 ================================
prereg loss 69.848305 regularization 242.583
loss 312.4313
STEP 352 ================================
prereg loss 69.19534 regularization 242.46303
loss 311.6584
STEP 353 ================================
prereg loss 68.83073 regularization 242.34428
loss 311.17502
STEP 354 ================================
prereg loss 68.36978 regularization 242.21568
loss 310.58545
STEP 355 ================================
prereg loss 67.774155 regularization 242.07605
loss 309.85022
STEP 356 ================================
prereg loss 67.33228 regularization 241.9336
loss 309.26587
STEP 357 ================================
prereg loss 66.879036 regularization 241.78232
loss 308.66135
STEP 358 ================================
prereg loss 66.28213 regularization 241.64938
loss 307.93152
STEP 359 ================================
prereg loss 65.97087 regularization 241.52641
loss 307.49728
STEP 360 ================================
prereg loss 66.968155 regularization 241.37083
loss 308.339
STEP 361 ================================
prereg loss 65.876 regularization 241.28166
loss 307.15765
STEP 362 ================================
prereg loss 70.01321 regularization 241.11084
loss 311.12405
STEP 363 ================================
prereg loss 65.42333 regularization 241.03203
loss 306.45535
STEP 364 ================================
prereg loss 94.62882 regularization 241.0075
loss 335.63632
STEP 365 ================================
prereg loss 91.12283 regularization 240.58429
loss 331.70712
STEP 366 ================================
prereg loss 108.77588 regularization 240.26753
loss 349.0434
STEP 367 ================================
prereg loss 119.09883 regularization 240.03377
loss 359.1326
STEP 368 ================================
prereg loss 125.409294 regularization 239.87439
loss 365.2837
STEP 369 ================================
prereg loss 127.03905 regularization 239.77982
loss 366.81885
STEP 370 ================================
prereg loss 126.721985 regularization 239.74155
loss 366.46353
STEP 371 ================================
prereg loss 124.517296 regularization 239.75047
loss 364.26776
STEP 372 ================================
prereg loss 121.39389 regularization 239.79764
loss 361.19153
STEP 373 ================================
prereg loss 117.39517 regularization 239.87889
loss 357.27405
STEP 374 ================================
prereg loss 112.625 regularization 239.9945
loss 352.6195
STEP 375 ================================
prereg loss 106.4998 regularization 240.14168
loss 346.64148
STEP 376 ================================
prereg loss 98.529655 regularization 240.318
loss 338.84766
STEP 377 ================================
prereg loss 90.64198 regularization 240.51671
loss 331.1587
STEP 378 ================================
prereg loss 84.925224 regularization 240.71608
loss 325.6413
STEP 379 ================================
prereg loss 80.613495 regularization 240.90053
loss 321.51404
STEP 380 ================================
prereg loss 80.2888 regularization 241.07474
loss 321.36353
STEP 381 ================================
prereg loss 94.43592 regularization 241.2025
loss 335.63843
STEP 382 ================================
prereg loss 96.51995 regularization 241.20232
loss 337.72226
STEP 383 ================================
prereg loss 91.754974 regularization 241.12901
loss 332.88397
STEP 384 ================================
prereg loss 82.72597 regularization 240.96925
loss 323.69522
STEP 385 ================================
prereg loss 81.123474 regularization 240.76823
loss 321.89172
STEP 386 ================================
prereg loss 84.17473 regularization 240.586
loss 324.76074
STEP 387 ================================
prereg loss 87.0069 regularization 240.42972
loss 327.4366
STEP 388 ================================
prereg loss 88.47593 regularization 240.30562
loss 328.78156
STEP 389 ================================
prereg loss 88.44737 regularization 240.212
loss 328.65936
STEP 390 ================================
prereg loss 87.05989 regularization 240.14745
loss 327.20734
STEP 391 ================================
prereg loss 84.19355 regularization 240.11067
loss 324.30423
STEP 392 ================================
prereg loss 80.5322 regularization 240.09625
loss 320.62845
STEP 393 ================================
prereg loss 79.76305 regularization 240.09178
loss 319.85483
STEP 394 ================================
prereg loss 80.39573 regularization 240.05852
loss 320.45425
STEP 395 ================================
prereg loss 80.19155 regularization 239.99022
loss 320.18176
STEP 396 ================================
prereg loss 78.26849 regularization 239.87941
loss 318.1479
STEP 397 ================================
prereg loss 74.76586 regularization 239.72714
loss 314.493
STEP 398 ================================
prereg loss 73.95065 regularization 239.57121
loss 313.52185
STEP 399 ================================
prereg loss 73.7222 regularization 239.41145
loss 313.13367
STEP 400 ================================
prereg loss 73.14445 regularization 239.26462
loss 312.40906
STEP 401 ================================
prereg loss 71.757355 regularization 239.12515
loss 310.8825
STEP 402 ================================
prereg loss 70.68475 regularization 239.00124
loss 309.68597
STEP 403 ================================
prereg loss 69.22282 regularization 238.89029
loss 308.1131
STEP 404 ================================
prereg loss 68.49389 regularization 238.78241
loss 307.2763
STEP 405 ================================
prereg loss 68.40154 regularization 238.65375
loss 307.0553
STEP 406 ================================
prereg loss 67.347275 regularization 238.49101
loss 305.8383
STEP 407 ================================
prereg loss 66.78507 regularization 238.3124
loss 305.09747
STEP 408 ================================
prereg loss 67.27084 regularization 238.15196
loss 305.4228
STEP 409 ================================
prereg loss 66.70855 regularization 238.02573
loss 304.73428
STEP 410 ================================
prereg loss 66.16678 regularization 237.92535
loss 304.09213
STEP 411 ================================
prereg loss 66.52144 regularization 237.81749
loss 304.33893
STEP 412 ================================
prereg loss 65.99193 regularization 237.6806
loss 303.67255
STEP 413 ================================
prereg loss 65.71939 regularization 237.52629
loss 303.24567
STEP 414 ================================
prereg loss 65.57628 regularization 237.39969
loss 302.97595
STEP 415 ================================
prereg loss 65.27478 regularization 237.3002
loss 302.57498
STEP 416 ================================
prereg loss 65.01633 regularization 237.18265
loss 302.19897
STEP 417 ================================
prereg loss 64.746056 regularization 237.0488
loss 301.79486
STEP 418 ================================
prereg loss 64.20044 regularization 236.9387
loss 301.13916
STEP 419 ================================
prereg loss 63.750957 regularization 236.84224
loss 300.5932
STEP 420 ================================
prereg loss 63.35569 regularization 236.72084
loss 300.07654
STEP 421 ================================
prereg loss 63.021896 regularization 236.61404
loss 299.63593
STEP 422 ================================
prereg loss 62.580154 regularization 236.52637
loss 299.1065
STEP 423 ================================
prereg loss 62.402424 regularization 236.44699
loss 298.84943
STEP 424 ================================
prereg loss 62.31071 regularization 236.34395
loss 298.65466
STEP 425 ================================
prereg loss 62.381134 regularization 236.27443
loss 298.65558
STEP 426 ================================
prereg loss 63.777527 regularization 236.1476
loss 299.9251
STEP 427 ================================
prereg loss 62.226574 regularization 236.09628
loss 298.32285
STEP 428 ================================
prereg loss 65.02162 regularization 235.94798
loss 300.9696
STEP 429 ================================
prereg loss 60.724094 regularization 235.88815
loss 296.61224
STEP 430 ================================
prereg loss 64.76042 regularization 235.80336
loss 300.56378
STEP 431 ================================
prereg loss 75.073555 regularization 235.5641
loss 310.63766
STEP 432 ================================
prereg loss 78.39425 regularization 235.4228
loss 313.81705
STEP 433 ================================
prereg loss 70.38685 regularization 235.36174
loss 305.7486
STEP 434 ================================
prereg loss 65.349556 regularization 235.3792
loss 300.72876
STEP 435 ================================
prereg loss 75.02096 regularization 235.344
loss 310.36496
STEP 436 ================================
prereg loss 105.09858 regularization 234.88474
loss 339.9833
STEP 437 ================================
prereg loss 165.78278 regularization 234.55693
loss 400.33972
STEP 438 ================================
prereg loss 198.61276 regularization 234.35611
loss 432.96887
STEP 439 ================================
prereg loss 150.9397 regularization 234.28267
loss 385.22235
STEP 440 ================================
prereg loss 143.3326 regularization 234.27957
loss 377.61218
STEP 441 ================================
prereg loss 140.94437 regularization 234.32225
loss 375.2666
STEP 442 ================================
prereg loss 131.88737 regularization 234.40207
loss 366.28943
STEP 443 ================================
prereg loss 128.06879 regularization 234.51012
loss 362.57892
STEP 444 ================================
prereg loss 119.23793 regularization 234.64001
loss 353.87793
STEP 445 ================================
prereg loss 117.81651 regularization 234.7975
loss 352.614
STEP 446 ================================
prereg loss 114.037994 regularization 234.97652
loss 349.01453
STEP 447 ================================
prereg loss 107.31522 regularization 235.17464
loss 342.48987
STEP 448 ================================
prereg loss 104.44238 regularization 235.40149
loss 339.84387
STEP 449 ================================
prereg loss 92.98612 regularization 235.64185
loss 328.62796
STEP 450 ================================
prereg loss 81.124756 regularization 235.8964
loss 317.02115
STEP 451 ================================
prereg loss 92.505486 regularization 236.02553
loss 328.531
STEP 452 ================================
prereg loss 86.14551 regularization 235.88197
loss 322.02747
STEP 453 ================================
prereg loss 88.2398 regularization 235.70126
loss 323.94107
STEP 454 ================================
prereg loss 105.7552 regularization 235.55302
loss 341.30823
STEP 455 ================================
prereg loss 113.70315 regularization 235.4241
loss 349.12726
STEP 456 ================================
prereg loss 115.80253 regularization 235.32361
loss 351.12613
STEP 457 ================================
prereg loss 120.56726 regularization 235.2512
loss 355.81848
STEP 458 ================================
prereg loss 120.963806 regularization 235.1939
loss 356.1577
STEP 459 ================================
prereg loss 116.11221 regularization 235.15234
loss 351.26456
STEP 460 ================================
prereg loss 112.97316 regularization 235.13043
loss 348.10358
STEP 461 ================================
prereg loss 104.14834 regularization 235.12831
loss 339.27664
STEP 462 ================================
prereg loss 98.64635 regularization 235.1513
loss 333.79767
STEP 463 ================================
prereg loss 94.47487 regularization 235.20451
loss 329.67938
STEP 464 ================================
prereg loss 87.19892 regularization 235.28612
loss 322.48505
STEP 465 ================================
prereg loss 79.177 regularization 235.39546
loss 314.57245
STEP 466 ================================
prereg loss 73.92988 regularization 235.51692
loss 309.4468
STEP 467 ================================
prereg loss 78.39425 regularization 235.61685
loss 314.0111
STEP 468 ================================
prereg loss 83.39849 regularization 235.6411
loss 319.03958
STEP 469 ================================
prereg loss 75.388695 regularization 235.56096
loss 310.94965
STEP 470 ================================
prereg loss 94.85246 regularization 235.44528
loss 330.29773
STEP 471 ================================
prereg loss 72.81788 regularization 235.39384
loss 308.21173
STEP 472 ================================
prereg loss 75.998405 regularization 235.29697
loss 311.29538
STEP 473 ================================
prereg loss 72.227974 regularization 235.15007
loss 307.37805
STEP 474 ================================
prereg loss 72.073555 regularization 234.97346
loss 307.04703
STEP 475 ================================
prereg loss 75.07689 regularization 234.80907
loss 309.88596
STEP 476 ================================
prereg loss 72.7898 regularization 234.6588
loss 307.4486
STEP 477 ================================
prereg loss 70.46735 regularization 234.52103
loss 304.98837
STEP 478 ================================
prereg loss 68.46872 regularization 234.3951
loss 302.86383
STEP 479 ================================
prereg loss 68.72361 regularization 234.2786
loss 303.0022
STEP 480 ================================
prereg loss 71.49982 regularization 234.18561
loss 305.68542
STEP 481 ================================
prereg loss 69.045265 regularization 234.12248
loss 303.16776
STEP 482 ================================
prereg loss 72.79035 regularization 234.0328
loss 306.82315
STEP 483 ================================
prereg loss 70.45096 regularization 233.87338
loss 304.32434
STEP 484 ================================
prereg loss 71.826645 regularization 233.72661
loss 305.55325
STEP 485 ================================
prereg loss 70.45838 regularization 233.6081
loss 304.06647
STEP 486 ================================
prereg loss 70.3083 regularization 233.44704
loss 303.75534
STEP 487 ================================
prereg loss 67.55194 regularization 233.2362
loss 300.78815
STEP 488 ================================
prereg loss 69.60261 regularization 233.04288
loss 302.64548
STEP 489 ================================
prereg loss 68.13931 regularization 232.90013
loss 301.03943
STEP 490 ================================
prereg loss 66.40938 regularization 232.79654
loss 299.20593
STEP 491 ================================
prereg loss 65.52336 regularization 232.7116
loss 298.23495
STEP 492 ================================
prereg loss 65.74912 regularization 232.59923
loss 298.34836
STEP 493 ================================
prereg loss 64.17073 regularization 232.45561
loss 296.62634
STEP 494 ================================
prereg loss 63.824474 regularization 232.29971
loss 296.12418
STEP 495 ================================
prereg loss 64.20336 regularization 232.15025
loss 296.3536
STEP 496 ================================
prereg loss 64.50895 regularization 232.03424
loss 296.54318
STEP 497 ================================
prereg loss 63.703342 regularization 231.95285
loss 295.6562
STEP 498 ================================
prereg loss 62.947823 regularization 231.86832
loss 294.81613
STEP 499 ================================
prereg loss 62.81543 regularization 231.78001
loss 294.59546
STEP 500 ================================
prereg loss 62.194954 regularization 231.70699
loss 293.90195
STEP 501 ================================
prereg loss 62.419514 regularization 231.6386
loss 294.0581
STEP 502 ================================
prereg loss 62.372112 regularization 231.53796
loss 293.91006
STEP 503 ================================
prereg loss 64.568535 regularization 231.4135
loss 295.98203
STEP 504 ================================
prereg loss 61.77416 regularization 231.32938
loss 293.10355
STEP 505 ================================
prereg loss 61.773155 regularization 231.2215
loss 292.99466
STEP 506 ================================
prereg loss 61.270714 regularization 231.08191
loss 292.35263
STEP 507 ================================
prereg loss 61.037918 regularization 230.96997
loss 292.00787
STEP 508 ================================
prereg loss 61.193565 regularization 230.87692
loss 292.0705
STEP 509 ================================
prereg loss 60.335007 regularization 230.7612
loss 291.0962
STEP 510 ================================
prereg loss 61.3765 regularization 230.63446
loss 292.01096
STEP 511 ================================
prereg loss 60.40066 regularization 230.54329
loss 290.94394
STEP 512 ================================
prereg loss 59.39831 regularization 230.4299
loss 289.82822
STEP 513 ================================
prereg loss 60.456104 regularization 230.30547
loss 290.76157
STEP 514 ================================
prereg loss 58.694233 regularization 230.21461
loss 288.90884
STEP 515 ================================
prereg loss 60.35118 regularization 230.12799
loss 290.4792
STEP 516 ================================
prereg loss 58.914467 regularization 229.9899
loss 288.90436
STEP 517 ================================
prereg loss 59.932327 regularization 229.87224
loss 289.80457
STEP 518 ================================
prereg loss 58.970905 regularization 229.78896
loss 288.75986
STEP 519 ================================
prereg loss 59.003345 regularization 229.72429
loss 288.72763
STEP 520 ================================
prereg loss 57.999146 regularization 229.61021
loss 287.60938
STEP 521 ================================
prereg loss 58.457874 regularization 229.49022
loss 287.9481
STEP 522 ================================
prereg loss 57.65466 regularization 229.39754
loss 287.0522
STEP 523 ================================
prereg loss 58.191418 regularization 229.31801
loss 287.50943
STEP 524 ================================
prereg loss 58.473408 regularization 229.2003
loss 287.6737
STEP 525 ================================
prereg loss 59.838634 regularization 229.07227
loss 288.9109
STEP 526 ================================
prereg loss 59.52482 regularization 228.98532
loss 288.51013
STEP 527 ================================
prereg loss 58.68416 regularization 228.93149
loss 287.61566
STEP 528 ================================
prereg loss 60.504154 regularization 228.85289
loss 289.35706
STEP 529 ================================
prereg loss 60.493515 regularization 228.67152
loss 289.16504
STEP 530 ================================
prereg loss 62.577106 regularization 228.5297
loss 291.1068
STEP 531 ================================
prereg loss 66.78395 regularization 228.45708
loss 295.24103
STEP 532 ================================
prereg loss 68.07402 regularization 228.42178
loss 296.4958
STEP 533 ================================
prereg loss 68.33756 regularization 228.30135
loss 296.63892
STEP 534 ================================
prereg loss 81.15816 regularization 228.07663
loss 309.2348
STEP 535 ================================
prereg loss 62.870197 regularization 227.9944
loss 290.8646
STEP 536 ================================
prereg loss 73.17325 regularization 227.94075
loss 301.114
STEP 537 ================================
prereg loss 79.71325 regularization 227.64867
loss 307.3619
STEP 538 ================================
prereg loss 84.34143 regularization 227.50754
loss 311.84897
STEP 539 ================================
prereg loss 64.78984 regularization 227.50563
loss 292.29547
STEP 540 ================================
prereg loss 103.70151 regularization 227.56647
loss 331.26797
STEP 541 ================================
prereg loss 146.83076 regularization 227.82205
loss 374.65283
STEP 542 ================================
prereg loss 117.87206 regularization 227.58073
loss 345.4528
STEP 543 ================================
prereg loss 140.0668 regularization 227.44502
loss 367.51184
STEP 544 ================================
prereg loss 122.50483 regularization 227.36725
loss 349.87207
STEP 545 ================================
prereg loss 111.34039 regularization 227.34099
loss 338.6814
STEP 546 ================================
prereg loss 101.830345 regularization 227.33965
loss 329.16998
STEP 547 ================================
prereg loss 109.85461 regularization 227.35477
loss 337.20938
STEP 548 ================================
prereg loss 122.241974 regularization 227.37253
loss 349.6145
STEP 549 ================================
prereg loss 135.31856 regularization 227.3757
loss 362.69427
STEP 550 ================================
prereg loss 144.68336 regularization 227.37727
loss 372.06064
STEP 551 ================================
prereg loss 144.96092 regularization 227.34245
loss 372.30338
STEP 552 ================================
prereg loss 144.19151 regularization 227.31813
loss 371.50964
STEP 553 ================================
prereg loss 143.00327 regularization 227.2917
loss 370.29498
STEP 554 ================================
prereg loss 141.04654 regularization 227.26178
loss 368.30832
STEP 555 ================================
prereg loss 139.18575 regularization 227.23509
loss 366.42084
STEP 556 ================================
prereg loss 137.00882 regularization 227.20674
loss 364.21558
STEP 557 ================================
prereg loss 134.89545 regularization 227.17847
loss 362.0739
STEP 558 ================================
prereg loss 133.46397 regularization 227.1429
loss 360.60687
STEP 559 ================================
prereg loss 132.03403 regularization 227.10289
loss 359.1369
STEP 560 ================================
prereg loss 130.87349 regularization 227.05383
loss 357.9273
STEP 561 ================================
prereg loss 129.67233 regularization 226.98796
loss 356.66028
STEP 562 ================================
prereg loss 128.3883 regularization 226.90518
loss 355.2935
STEP 563 ================================
prereg loss 126.904655 regularization 226.80779
loss 353.71243
STEP 564 ================================
prereg loss 125.08666 regularization 226.69711
loss 351.78378
STEP 565 ================================
prereg loss 122.8941 regularization 226.5773
loss 349.4714
STEP 566 ================================
prereg loss 121.62105 regularization 226.44798
loss 348.06903
STEP 567 ================================
prereg loss 119.939224 regularization 226.30151
loss 346.24072
STEP 568 ================================
prereg loss 118.35957 regularization 226.15076
loss 344.5103
STEP 569 ================================
prereg loss 116.52476 regularization 225.99718
loss 342.52194
STEP 570 ================================
prereg loss 114.10303 regularization 225.8422
loss 339.94522
STEP 571 ================================
prereg loss 112.84777 regularization 225.68317
loss 338.53094
STEP 572 ================================
prereg loss 110.11877 regularization 225.50775
loss 335.62653
STEP 573 ================================
prereg loss 107.96344 regularization 225.33345
loss 333.29688
STEP 574 ================================
prereg loss 105.9942 regularization 225.16791
loss 331.1621
STEP 575 ================================
prereg loss 104.14081 regularization 225.00595
loss 329.14676
STEP 576 ================================
prereg loss 102.45902 regularization 224.84473
loss 327.30374
STEP 577 ================================
prereg loss 100.953125 regularization 224.68692
loss 325.64005
STEP 578 ================================
prereg loss 99.415726 regularization 224.54247
loss 323.9582
STEP 579 ================================
prereg loss 98.25822 regularization 224.39742
loss 322.65564
STEP 580 ================================
prereg loss 97.401054 regularization 224.25934
loss 321.6604
STEP 581 ================================
prereg loss 96.67158 regularization 224.12823
loss 320.7998
STEP 582 ================================
prereg loss 96.14447 regularization 224.00587
loss 320.15033
STEP 583 ================================
prereg loss 95.63315 regularization 223.8916
loss 319.52475
STEP 584 ================================
prereg loss 95.20339 regularization 223.78366
loss 318.98706
STEP 585 ================================
prereg loss 95.73157 regularization 223.67606
loss 319.40762
STEP 586 ================================
prereg loss 94.74807 regularization 223.54694
loss 318.295
STEP 587 ================================
prereg loss 94.47827 regularization 223.41313
loss 317.89142
STEP 588 ================================
prereg loss 94.47766 regularization 223.29137
loss 317.76904
STEP 589 ================================
prereg loss 93.901146 regularization 223.16788
loss 317.06903
STEP 590 ================================
prereg loss 93.51577 regularization 223.04034
loss 316.55612
STEP 591 ================================
prereg loss 93.087494 regularization 222.92206
loss 316.00955
STEP 592 ================================
prereg loss 92.09282 regularization 222.81459
loss 314.9074
STEP 593 ================================
prereg loss 90.85538 regularization 222.71864
loss 313.57404
STEP 594 ================================
prereg loss 91.38189 regularization 222.62529
loss 314.00717
STEP 595 ================================
prereg loss 90.1797 regularization 222.50757
loss 312.68726
STEP 596 ================================
prereg loss 89.09299 regularization 222.39198
loss 311.485
STEP 597 ================================
prereg loss 88.258224 regularization 222.28589
loss 310.54413
STEP 598 ================================
prereg loss 86.715836 regularization 222.20021
loss 308.91605
STEP 599 ================================
prereg loss 84.88659 regularization 222.13255
loss 307.01913
STEP 600 ================================
prereg loss 83.413536 regularization 222.05641
loss 305.46994
STEP 601 ================================
prereg loss 81.56967 regularization 221.96564
loss 303.5353
STEP 602 ================================
prereg loss 79.55504 regularization 221.86124
loss 301.41626
STEP 603 ================================
prereg loss 78.53684 regularization 221.7646
loss 300.30145
STEP 604 ================================
prereg loss 75.994255 regularization 221.70303
loss 297.6973
STEP 605 ================================
prereg loss 74.78997 regularization 221.6618
loss 296.45178
STEP 606 ================================
prereg loss 72.87222 regularization 221.57407
loss 294.4463
STEP 607 ================================
prereg loss 72.25521 regularization 221.44571
loss 293.70093
STEP 608 ================================
prereg loss 73.02718 regularization 221.34729
loss 294.37448
STEP 609 ================================
prereg loss 73.378456 regularization 221.31439
loss 294.69284
STEP 610 ================================
prereg loss 74.06951 regularization 221.16019
loss 295.2297
STEP 611 ================================
prereg loss 90.9088 regularization 221.11629
loss 312.0251
STEP 612 ================================
prereg loss 71.950424 regularization 220.83583
loss 292.78625
STEP 613 ================================
prereg loss 84.427 regularization 220.57382
loss 305.00082
STEP 614 ================================
prereg loss 105.87047 regularization 220.41835
loss 326.28882
STEP 615 ================================
prereg loss 111.40214 regularization 220.33832
loss 331.74045
STEP 616 ================================
prereg loss 107.98246 regularization 220.32703
loss 328.30948
STEP 617 ================================
prereg loss 98.97227 regularization 220.37761
loss 319.34988
STEP 618 ================================
prereg loss 86.87384 regularization 220.49101
loss 307.36487
STEP 619 ================================
prereg loss 69.91141 regularization 220.65717
loss 290.56857
STEP 620 ================================
prereg loss 95.17952 regularization 220.8172
loss 315.9967
STEP 621 ================================
prereg loss 71.39224 regularization 220.67369
loss 292.06592
STEP 622 ================================
prereg loss 75.162445 regularization 220.52463
loss 295.68707
STEP 623 ================================
prereg loss 84.82208 regularization 220.41966
loss 305.24176
STEP 624 ================================
prereg loss 86.72422 regularization 220.371
loss 307.0952
STEP 625 ================================
prereg loss 82.49518 regularization 220.37785
loss 302.87305
STEP 626 ================================
prereg loss 75.93713 regularization 220.43097
loss 296.3681
STEP 627 ================================
prereg loss 76.88034 regularization 220.48514
loss 297.36548
STEP 628 ================================
prereg loss 80.94567 regularization 220.48134
loss 301.427
STEP 629 ================================
prereg loss 79.801384 regularization 220.40266
loss 300.20404
STEP 630 ================================
prereg loss 76.99536 regularization 220.26646
loss 297.26184
STEP 631 ================================
prereg loss 73.67028 regularization 220.09596
loss 293.76624
STEP 632 ================================
prereg loss 75.54982 regularization 219.9333
loss 295.48312
STEP 633 ================================
prereg loss 77.23909 regularization 219.81581
loss 297.0549
STEP 634 ================================
prereg loss 74.75494 regularization 219.7515
loss 294.50644
2022-06-05T10:36:21.728

julia> a_1000 = deepcopy(sparse)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.177075), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.116525, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.000217514, "dict-2"=>…
  "dot-2"     => Dict("dict"=>Dict("norm-3"=>Dict("true"=>7.43651f-5, "dict"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.18288), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "n…
  "accum-3"   => Dict("false"=>Dict("compare-4"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.203852), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "compare-4" => Dict("dot"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>-5.52012f-5, "false"=>0.0, "dict-2"=…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "no…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0847501), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.101313), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0218848), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.29024), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.665222), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.155984), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>4.57671f-5, "dict-2"=…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-0.000125999, "dict-…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.440173), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…

julia> count_neg_interval(sparse, -0.001f0, 0.001f0)
745

julia> count_neg_interval(sparse, -0.01f0, 0.01f0)
616

julia> count_neg_interval(sparse, -0.1f0, 0.1f0)
289

julia> count_neg_interval(sparse, -0.2f0, 0.2f0)
114

julia> count_neg_interval(sparse, -0.3f0, 0.3f0)
36

julia> count_neg_interval(sparse, -0.4f0, 0.4f0)
13

julia> count_neg_interval(sparse, -0.5f0, 0.5f0)
6

julia> count_neg_interval(sparse, -0.6f0, 0.6f0)
4

julia> count_neg_interval(sparse, -0.7f0, 0.7f0)
2

julia> serialize("1000-steps-matrix.ser", sparse)

julia> open("1000-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(sparse))
           println(f)
           end

julia>
```

A very difficult loss landscape. I don't know if we have "vanishing gradient" here,
or not, but we should act as if we are and adjust regularization (and upgrade some of our activation
functions, in particular `dot`, replacing it by `x, y -> dot(softmax(x), y)` or smth similar).

Instead of the L2 component, let's aim for network matrix rows to sum to 1.

```
julia> steps!(100)
2022-06-05T10:49:18.232
STEP 1 ================================
prereg loss 70.89666 regularization 219.73558
loss 290.63223
STEP 2 ================================
prereg loss 72.89403 regularization 219.71329
loss 292.6073
STEP 3 ================================
prereg loss 73.325226 regularization 219.62979
loss 292.95502
STEP 4 ================================
prereg loss 70.44764 regularization 219.49026
loss 289.9379
STEP 5 ================================
prereg loss 69.45102 regularization 219.32516
loss 288.77618
STEP 6 ================================
prereg loss 70.96218 regularization 219.17929
loss 290.14148
STEP 7 ================================
prereg loss 69.91349 regularization 219.08752
loss 289.001
STEP 8 ================================
prereg loss 67.32017 regularization 219.0424
loss 286.36258
STEP 9 ================================
prereg loss 68.77092 regularization 218.99706
loss 287.76797
STEP 10 ================================
prereg loss 67.56218 regularization 218.89717
loss 286.45935
STEP 11 ================================
prereg loss 68.15873 regularization 218.75874
loss 286.91748
STEP 12 ================================
prereg loss 67.59523 regularization 218.67992
loss 286.27515
STEP 13 ================================
prereg loss 67.33161 regularization 218.65593
loss 285.98755
STEP 14 ================================
prereg loss 65.68369 regularization 218.55194
loss 284.23563
STEP 15 ================================
prereg loss 67.03946 regularization 218.41156
loss 285.45102
STEP 16 ================================
prereg loss 64.49115 regularization 218.33908
loss 282.83023
STEP 17 ================================
prereg loss 64.08287 regularization 218.2966
loss 282.37946
STEP 18 ================================
prereg loss 64.10936 regularization 218.24814
loss 282.35748
STEP 19 ================================
prereg loss 64.1218 regularization 218.1633
loss 282.2851
STEP 20 ================================
prereg loss 62.75373 regularization 218.02109
loss 280.7748
STEP 21 ================================
prereg loss 62.978912 regularization 217.90315
loss 280.88208
STEP 22 ================================
prereg loss 62.344364 regularization 217.83566
loss 280.18002
STEP 23 ================================
prereg loss 66.68685 regularization 217.80888
loss 284.49573
STEP 24 ================================
prereg loss 68.876755 regularization 217.626
loss 286.50275
STEP 25 ================================
prereg loss 61.777035 regularization 217.59393
loss 279.37097
STEP 26 ================================
prereg loss 69.36731 regularization 217.60307
loss 286.9704
STEP 27 ================================
prereg loss 64.49433 regularization 217.49712
loss 281.99146
STEP 28 ================================
prereg loss 72.1964 regularization 217.29216
loss 289.48856
STEP 29 ================================
prereg loss 66.02799 regularization 217.23604
loss 283.26404
STEP 30 ================================
prereg loss 63.50108 regularization 217.28517
loss 280.78625
STEP 31 ================================
prereg loss 81.48431 regularization 217.30904
loss 298.79333
STEP 32 ================================
prereg loss 69.712654 regularization 217.15695
loss 286.8696
STEP 33 ================================
prereg loss 63.27344 regularization 216.92618
loss 280.19962
STEP 34 ================================
prereg loss 69.14475 regularization 216.74464
loss 285.8894
STEP 35 ================================
prereg loss 68.72331 regularization 216.63818
loss 285.3615
STEP 36 ================================
prereg loss 66.23915 regularization 216.59775
loss 282.8369
STEP 37 ================================
prereg loss 68.94059 regularization 216.60461
loss 285.5452
STEP 38 ================================
prereg loss 77.10304 regularization 216.5607
loss 293.66376
STEP 39 ================================
prereg loss 80.55586 regularization 216.52127
loss 297.07715
STEP 40 ================================
prereg loss 75.17966 regularization 216.35164
loss 291.5313
STEP 41 ================================
prereg loss 69.306335 regularization 216.20184
loss 285.50818
STEP 42 ================================
prereg loss 69.52611 regularization 216.03392
loss 285.56003
STEP 43 ================================
prereg loss 74.237495 regularization 215.90579
loss 290.14328
STEP 44 ================================
prereg loss 71.29439 regularization 215.8565
loss 287.15088
STEP 45 ================================
prereg loss 66.90323 regularization 215.86993
loss 282.77316
STEP 46 ================================
prereg loss 68.46417 regularization 215.8771
loss 284.34128
STEP 47 ================================
prereg loss 70.05721 regularization 215.86615
loss 285.92337
STEP 48 ================================
prereg loss 71.39726 regularization 215.83867
loss 287.23593
STEP 49 ================================
prereg loss 67.428375 regularization 215.73114
loss 283.15952
STEP 50 ================================
prereg loss 66.85428 regularization 215.61198
loss 282.46625
STEP 51 ================================
prereg loss 65.11591 regularization 215.41075
loss 280.52667
STEP 52 ================================
prereg loss 72.047356 regularization 215.2759
loss 287.32324
STEP 53 ================================
prereg loss 76.33672 regularization 215.20839
loss 291.5451
STEP 54 ================================
prereg loss 63.07213 regularization 215.25006
loss 278.3222
STEP 55 ================================
prereg loss 63.384533 regularization 215.34125
loss 278.72577
STEP 56 ================================
prereg loss 67.29869 regularization 215.33803
loss 282.63672
STEP 57 ================================
prereg loss 61.268696 regularization 215.23781
loss 276.5065
STEP 58 ================================
prereg loss 66.39842 regularization 215.07887
loss 281.4773
STEP 59 ================================
prereg loss 69.26685 regularization 215.00859
loss 284.27545
STEP 60 ================================
prereg loss 62.12895 regularization 215.01973
loss 277.14868
STEP 61 ================================
prereg loss 62.143192 regularization 215.08456
loss 277.22775
STEP 62 ================================
prereg loss 65.1856 regularization 215.06061
loss 280.24622
STEP 63 ================================
prereg loss 59.87472 regularization 214.92375
loss 274.79846
STEP 64 ================================
prereg loss 66.18168 regularization 214.78723
loss 280.9689
STEP 65 ================================
prereg loss 67.868996 regularization 214.73363
loss 282.60263
STEP 66 ================================
prereg loss 59.901173 regularization 214.772
loss 274.6732
STEP 67 ================================
prereg loss 66.76372 regularization 214.838
loss 281.6017
STEP 68 ================================
prereg loss 67.478096 regularization 214.82802
loss 282.30612
STEP 69 ================================
prereg loss 58.745377 regularization 214.733
loss 273.4784
STEP 70 ================================
prereg loss 60.43436 regularization 214.60872
loss 275.0431
STEP 71 ================================
prereg loss 63.07 regularization 214.53424
loss 277.60425
STEP 72 ================================
prereg loss 60.06726 regularization 214.51962
loss 274.58688
STEP 73 ================================
prereg loss 57.28555 regularization 214.54175
loss 271.8273
STEP 74 ================================
prereg loss 55.590195 regularization 214.56544
loss 270.15564
STEP 75 ================================
prereg loss 57.544834 regularization 214.5725
loss 272.11734
STEP 76 ================================
prereg loss 57.461395 regularization 214.5157
loss 271.9771
STEP 77 ================================
prereg loss 56.13784 regularization 214.41943
loss 270.55728
STEP 78 ================================
prereg loss 57.46011 regularization 214.31644
loss 271.77655
STEP 79 ================================
prereg loss 56.799095 regularization 214.24896
loss 271.04807
STEP 80 ================================
prereg loss 56.165596 regularization 214.20059
loss 270.36618
STEP 81 ================================
prereg loss 56.690407 regularization 214.12682
loss 270.81723
STEP 82 ================================
prereg loss 55.683384 regularization 214.0167
loss 269.70007
STEP 83 ================================
prereg loss 55.652206 regularization 213.88681
loss 269.539
STEP 84 ================================
prereg loss 55.317543 regularization 213.78575
loss 269.1033
STEP 85 ================================
prereg loss 54.522743 regularization 213.7122
loss 268.23495
STEP 86 ================================
prereg loss 54.557854 regularization 213.6227
loss 268.18054
STEP 87 ================================
prereg loss 53.79464 regularization 213.50435
loss 267.29898
STEP 88 ================================
prereg loss 54.107845 regularization 213.37906
loss 267.4869
STEP 89 ================================
prereg loss 53.307796 regularization 213.2953
loss 266.6031
STEP 90 ================================
prereg loss 53.270077 regularization 213.22153
loss 266.4916
STEP 91 ================================
prereg loss 52.995888 regularization 213.12439
loss 266.12027
STEP 92 ================================
prereg loss 52.853443 regularization 213.01184
loss 265.8653
STEP 93 ================================
prereg loss 52.763477 regularization 212.92192
loss 265.6854
STEP 94 ================================
prereg loss 52.305157 regularization 212.8539
loss 265.15906
STEP 95 ================================
prereg loss 51.892315 regularization 212.78313
loss 264.67545
STEP 96 ================================
prereg loss 52.390106 regularization 212.71103
loss 265.10114
STEP 97 ================================
prereg loss 52.14381 regularization 212.58751
loss 264.73132
STEP 98 ================================
prereg loss 52.392536 regularization 212.50108
loss 264.89362
STEP 99 ================================
prereg loss 51.475555 regularization 212.45581
loss 263.93137
STEP 100 ================================
prereg loss 51.266964 regularization 212.37723
loss 263.6442
2022-06-05T12:52:23.370

julia>

julia>

julia> count(sparse)
4661

julia> count(opt.mt)
20308

julia> count(opt.vt)
20308
```

Better convergence, but we discovered a rather nasty performance bug.
It is related to the unfortunate use of `zerocopy` in initialization
of `TreeADAM` (FIX THIS, **should** be easy), and the suboptimal order of operations
in the current version of the `train` script (easy to fix).

But there is also a mystery here: with this bug, why the size of
`sparse` is still not the full 20308, although still too big
(we expected 851, but got something in between that and 20308).

**(All this is now understood and fixed)**

Another thing we discovered: one actually needs to serialize the state of the optimizer as well,
if one wants to be able to resume optimization from the same place 
(I have only been serializing the network matrix, 
and taking advantage of the fact that we are resetting neuron inputs and outputs in this particular setup, 
otherwise we would need to serialize them too; but one also needs to serialize the state of the optimizer)

Something like these (and I am also going to commit these resulting files, and notice the current version
of Julia and its packages):

```
julia> a_1100 = deepcopy(sparse)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.210515), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.107804, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-0.0002176, "dict-2"=>0…
  "dot-2"     => Dict("dict"=>Dict("norm-3"=>Dict("true"=>7.43702f-5, "dict"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.182645), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "n…
  "accum-3"   => Dict("false"=>Dict("compare-4"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.202504), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "compare-4" => Dict("dot"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>-5.52074f-5, "false"=>0.0, "dict-2"=…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0, "no…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.0, "dict-2"=>0.0,…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0821279), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0929922), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "do…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0192839), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.293267), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "do…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.66313), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "do…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.157245), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "d…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>-6.50462f-5, "dict-2"…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("dict"=>0.0, "true"=>0.0, "dot"=>0.0, "false"=>0.000163967, "dict-2…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.432969), "norm-5"=>Dict("dict"=>0.0, "true"=>0.0, "dot"…

julia> serialize("1100-steps-matrix.ser", sparse)

julia> serialize("1100-steps-treeADAM.ser", opt)

(@v1.7) pkg> status
      Status `C:\Users\anhin\.julia\environments\v1.7\Project.toml`
  [31c24e10] Distributions v0.25.61
  [7da242da] Enzyme v0.9.6
  [587475ba] Flux v0.13.3
  [de31a74c] FunctionalCollections v0.5.0
  [f67ccb44] HDF5 v0.16.9
  [7073ff75] IJulia v1.23.3
  [86fae568] ImageView v0.11.1
  [916415d5] Images v0.25.2
  [4138dd39] JLD v0.13.1
  [033835bb] JLD2 v0.4.22
  [0f8b85d8] JSON3 v1.9.5
  [37e2e3b7] ReverseDiff v1.14.0
  [5e47fb64] TestImages v1.7.0
  [e88e6eb3] Zygote v0.6.40
  [37e2e46d] LinearAlgebra
  
(Julia 1.7.3; a Windows 10 machine)
```
