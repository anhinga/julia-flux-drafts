# A smaller run with L1+L2

Fast turnaround, but does not work well.

Main lesson: I am not using `id_transform` in a reasonable fashion.

All I really want is skip connections. In the feedforward version
this does not require the `id_transform` at all.

In a recurrent version, one would need `id_transform` neurons in the
quantity proportional to how far to the back one would like to go,
or one can just use the variadic nature of neurons, and reuse more and
more inputs and outputs, but the incoming connectivity pattern for
all those things should be handcrafted and fixed, only the outcoming
pattern is learned.

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> steps!(50)
2022-06-15T17:22:24.626
STEP 1 ================================
prereg loss 0.9905278 reg_l1 16.702343 reg_l2 0.20844774
loss 1.0072302
STEP 2 ================================
prereg loss 0.9873813 reg_l1 14.790674 reg_l2 0.17773387
loss 1.002172
STEP 3 ================================
prereg loss 0.9841798 reg_l1 13.04248 reg_l2 0.15082675
loss 0.9972223
STEP 4 ================================
prereg loss 0.98092574 reg_l1 11.511187 reg_l2 0.12746413
loss 0.99243695
STEP 5 ================================
prereg loss 0.9776038 reg_l1 10.166616 reg_l2 0.107305706
loss 0.98777044
STEP 6 ================================
prereg loss 0.9742292 reg_l1 8.950021 reg_l2 0.09001353
loss 0.9831792
STEP 7 ================================
prereg loss 0.97081625 reg_l1 7.8989697 reg_l2 0.07530252
loss 0.97871524
STEP 8 ================================
prereg loss 0.9673899 reg_l1 6.985261 reg_l2 0.06288343
loss 0.9743751
STEP 9 ================================
prereg loss 0.9639354 reg_l1 6.1748366 reg_l2 0.052486166
loss 0.97011024
STEP 10 ================================
prereg loss 0.96041125 reg_l1 5.4688883 reg_l2 0.043872982
loss 0.96588016
STEP 11 ================================
prereg loss 0.95680076 reg_l1 4.869129 reg_l2 0.03681561
loss 0.96166986
STEP 12 ================================
prereg loss 0.9531082 reg_l1 4.375474 reg_l2 0.031100972
loss 0.95748365
STEP 13 ================================
prereg loss 0.9493658 reg_l1 3.9673035 reg_l2 0.026524361
loss 0.9533331
STEP 14 ================================
prereg loss 0.9456123 reg_l1 3.6414368 reg_l2 0.02289377
loss 0.94925374
STEP 15 ================================
prereg loss 0.94190717 reg_l1 3.3786695 reg_l2 0.020033032
loss 0.94528586
STEP 16 ================================
prereg loss 0.9383107 reg_l1 3.1523294 reg_l2 0.017789269
loss 0.941463
STEP 17 ================================
prereg loss 0.93488127 reg_l1 2.9635193 reg_l2 0.016029546
loss 0.9378448
STEP 18 ================================
prereg loss 0.93166906 reg_l1 2.759691 reg_l2 0.014646814
loss 0.93442875
STEP 19 ================================
prereg loss 0.92870235 reg_l1 2.6022122 reg_l2 0.01357534
loss 0.9313046
STEP 20 ================================
prereg loss 0.92600805 reg_l1 2.462444 reg_l2 0.012748881
loss 0.9284705
STEP 21 ================================
prereg loss 0.923559 reg_l1 2.3245106 reg_l2 0.01210477
loss 0.92588353
STEP 22 ================================
prereg loss 0.9214005 reg_l1 2.2112575 reg_l2 0.011594056
loss 0.92361176
STEP 23 ================================
prereg loss 0.91956884 reg_l1 2.074351 reg_l2 0.01117837
loss 0.9216432
STEP 24 ================================
prereg loss 0.9180824 reg_l1 1.9701177 reg_l2 0.010832873
loss 0.9200525
STEP 25 ================================
prereg loss 0.91692007 reg_l1 1.8596067 reg_l2 0.010525171
loss 0.9187797
STEP 26 ================================
prereg loss 0.9160411 reg_l1 1.7518232 reg_l2 0.0102414405
loss 0.9177929
STEP 27 ================================
prereg loss 0.9153947 reg_l1 1.6617287 reg_l2 0.0099772625
loss 0.91705644
STEP 28 ================================
prereg loss 0.91493267 reg_l1 1.5806972 reg_l2 0.009723939
loss 0.9165134
STEP 29 ================================
prereg loss 0.914639 reg_l1 1.5058434 reg_l2 0.009475285
loss 0.91614485
STEP 30 ================================
prereg loss 0.91446936 reg_l1 1.4308475 reg_l2 0.009228482
loss 0.91590023
STEP 31 ================================
prereg loss 0.9143933 reg_l1 1.3661312 reg_l2 0.008987668
loss 0.91575944
STEP 32 ================================
prereg loss 0.914387 reg_l1 1.2906482 reg_l2 0.008754126
loss 0.9156776
STEP 33 ================================
prereg loss 0.9144327 reg_l1 1.234718 reg_l2 0.008533986
loss 0.9156674
STEP 34 ================================
prereg loss 0.914523 reg_l1 1.1928898 reg_l2 0.008327773
loss 0.9157159
STEP 35 ================================
prereg loss 0.9146699 reg_l1 1.1483588 reg_l2 0.0081304
loss 0.9158182
STEP 36 ================================
prereg loss 0.91489035 reg_l1 1.102674 reg_l2 0.007942708
loss 0.91599303
STEP 37 ================================
prereg loss 0.9152068 reg_l1 1.0672077 reg_l2 0.007766949
loss 0.916274
STEP 38 ================================
prereg loss 0.91562617 reg_l1 1.0105904 reg_l2 0.007602006
loss 0.91663677
STEP 39 ================================
prereg loss 0.91614926 reg_l1 0.95846033 reg_l2 0.0074617765
loss 0.9171077
STEP 40 ================================
prereg loss 0.91677684 reg_l1 0.93293023 reg_l2 0.007357602
loss 0.91770977
STEP 41 ================================
prereg loss 0.91749454 reg_l1 0.91781723 reg_l2 0.0072816457
loss 0.9184123
STEP 42 ================================
prereg loss 0.91828704 reg_l1 0.89148587 reg_l2 0.0072222445
loss 0.91917855
STEP 43 ================================
prereg loss 0.91914177 reg_l1 0.8599764 reg_l2 0.0071763545
loss 0.92000175
STEP 44 ================================
prereg loss 0.9200389 reg_l1 0.82045424 reg_l2 0.0071472926
loss 0.92085934
STEP 45 ================================
prereg loss 0.9209524 reg_l1 0.7930101 reg_l2 0.0071392264
loss 0.9217454
STEP 46 ================================
prereg loss 0.9218549 reg_l1 0.77500504 reg_l2 0.0071509825
loss 0.9226299
STEP 47 ================================
prereg loss 0.9227158 reg_l1 0.7780968 reg_l2 0.0071743866
loss 0.92349386
STEP 48 ================================
prereg loss 0.92350465 reg_l1 0.7489002 reg_l2 0.0071936618
loss 0.9242535
STEP 49 ================================
prereg loss 0.9241936 reg_l1 0.7120356 reg_l2 0.0072119967
loss 0.92490566
STEP 50 ================================
prereg loss 0.9247487 reg_l1 0.69187677 reg_l2 0.007242462
loss 0.9254406
2022-06-15T17:31:06.592

julia> count(trainable["network_matrix"])
2102

julia> count(trainable["fixed_matrix"])
10

julia> trainable["fixed_matrix"]
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "accum-3-2" => Dict("dict-1"=>Dict("accum-3-2"=>Dict("dict"=>1.0)))
  "accum-4-2" => Dict("dict-1"=>Dict("accum-4-2"=>Dict("dict"=>1.0)))
  "accum-3-1" => Dict("dict-1"=>Dict("accum-3-1"=>Dict("dict"=>1.0)))
  "accum-4-1" => Dict("dict-1"=>Dict("accum-4-1"=>Dict("dict"=>1.0)))
  "accum-2-2" => Dict("dict-1"=>Dict("accum-2-2"=>Dict("dict"=>1.0)))
  "accum-1-1" => Dict("dict-1"=>Dict("accum-1-1"=>Dict("dict"=>1.0)))
  "accum-1-2" => Dict("dict-1"=>Dict("accum-1-2"=>Dict("dict"=>1.0)))
  "accum-2-1" => Dict("dict-1"=>Dict("accum-2-1"=>Dict("dict"=>1.0)))
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))

julia> close(io)
```
