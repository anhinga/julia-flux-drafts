# Redoing run-1.1 with novel regularization

Instead of L1+L2 we take L1+Novel, where novel tries to push
the sum of weights in a network matrix row to 1 (the term is
square of (the sum of weights within the given row - 1)

This is inspired in spirit by the approach of
"Overcoming the vanishing gradient problem in plain recurrent networks",
https://arxiv.org/abs/1801.06105

This seems to work much better in terms of convergence, although
there is less sparsity achieved:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
prereg loss 2.4201107 regularization 3279.3413 reg_novel 1046.6927
loss 6.7461452
DONE: adam_step!
The network is ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.185591), "norm-5"=>Dict("dict"=>-0.0243447, "true"=>…
  "norm-5"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0914422), "norm-5"=>Dict("dict"=>0.256498, "true"=>0.13…
  "accum-4"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.163911), "norm-5"=>Dict("dict"=>-0.169589, "true"=>-0.…
  "dot-2"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0923257), "norm-5"=>Dict("dict"=>-0.107769, "true"=>-0…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.166052), "norm-5"=>Dict("dict"=>0.226922, "true"=>0.035…
  "compare-5" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.175146), "norm-5"=>Dict("dict"=>-0.374431, "true"=>-0.…
  "accum-3"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.174649), "norm-5"=>Dict("dict"=>-0.105347, "true"=>0.17…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.151256), "norm-5"=>Dict("dict"=>0.55511, "true"=>-0.200…
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.48265), "norm-5"=>Dict("dict"=>-0.26362, "true"=>-0.13…
  "compare-4" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.335428), "norm-5"=>Dict("dict"=>0.119192, "true"=>-0.25…
  "compare-2" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.246845), "norm-5"=>Dict("dict"=>-0.27459, "true"=>0.002…
  "dot-1"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0684959), "norm-5"=>Dict("dict"=>0.169408, "true"=>0.1…
  "dot-3"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.231223), "norm-5"=>Dict("dict"=>0.0145549, "true"=>-0.…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.176153), "norm-5"=>Dict("dict"=>0.348762, "true"=>0.07…
  "compare-3" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.197101), "norm-5"=>Dict("dict"=>-0.0234599, "true"=>-0.…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.245517), "norm-5"=>Dict("dict"=>-0.270132, "true"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0303694), "norm-5"=>Dict("dict"=>-0.050681, "true"=>0.…
  "compare-1" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.274177), "norm-5"=>Dict("dict"=>0.213276, "true"=>0.06…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.247491), "norm-5"=>Dict("dict"=>-0.263982, "true"=>-0.0…
  "dot-5"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0598614), "norm-5"=>Dict("dict"=>-0.260952, "true"=>0.0…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.467293), "norm-5"=>Dict("dict"=>0.0331713, "true"=>-0.0…

julia> steps!(16)
2022-06-05T21:53:12.417
STEP 1 ================================
prereg loss 1.9696658 regularization 3275.0347 reg_novel 967.8846
loss 6.2125854
STEP 2 ================================
prereg loss 1.6089424 regularization 3270.124 reg_novel 894.13464
loss 5.7732015
STEP 3 ================================
prereg loss 1.3480141 regularization 3264.6677 reg_novel 825.1346
loss 5.4378166
STEP 4 ================================
prereg loss 1.1890152 regularization 3258.814 reg_novel 760.6182
loss 5.2084475
STEP 5 ================================
prereg loss 1.111058 regularization 3252.5957 reg_novel 700.3695
loss 5.0640235
STEP 6 ================================
prereg loss 1.1007833 regularization 3246.0486 reg_novel 644.1925
loss 4.9910245
STEP 7 ================================
prereg loss 1.1458588 regularization 3239.1934 reg_novel 591.929
loss 4.976981
STEP 8 ================================
prereg loss 1.2170291 regularization 3232.0234 reg_novel 543.43097
loss 4.9924836
STEP 9 ================================
prereg loss 1.281207 regularization 3224.5703 reg_novel 498.5328
loss 5.00431
STEP 10 ================================
prereg loss 1.3243695 regularization 3216.8616 reg_novel 457.03268
loss 4.998264
STEP 11 ================================
prereg loss 1.3417207 regularization 3208.9502 reg_novel 418.72858
loss 4.9693995
STEP 12 ================================
prereg loss 1.3329804 regularization 3200.8062 reg_novel 383.41687
loss 4.917204
STEP 13 ================================
prereg loss 1.3121157 regularization 3192.4255 reg_novel 350.8905
loss 4.855432
STEP 14 ================================
prereg loss 1.2646813 regularization 3183.8496 reg_novel 320.96182
loss 4.769493
STEP 15 ================================
prereg loss 1.2151241 regularization 3175.061 reg_novel 293.4462
loss 4.683632
STEP 16 ================================
prereg loss 1.1785182 regularization 3166.105 reg_novel 268.16855
loss 4.612792
2022-06-05T22:13:16.923

julia> steps!(16)
2022-06-05T22:21:02.889
STEP 1 ================================
prereg loss 1.1383766 regularization 3156.946 reg_novel 244.96555
loss 4.5402884
STEP 2 ================================
prereg loss 1.1083894 regularization 3147.6367 reg_novel 223.67264
loss 4.479699
STEP 3 ================================
prereg loss 1.0932108 regularization 3138.1587 reg_novel 204.13979
loss 4.435509
STEP 4 ================================
prereg loss 1.0907493 regularization 3128.5476 reg_novel 186.22287
loss 4.4055195
STEP 5 ================================
prereg loss 1.0945882 regularization 3118.772 reg_novel 169.79369
loss 4.383154
STEP 6 ================================
prereg loss 1.1012262 regularization 3108.8657 reg_novel 154.73268
loss 4.364825
STEP 7 ================================
prereg loss 1.1094536 regularization 3098.8313 reg_novel 140.93324
loss 4.3492184
STEP 8 ================================
prereg loss 1.1147029 regularization 3088.677 reg_novel 128.29337
loss 4.3316736
STEP 9 ================================
prereg loss 1.1163657 regularization 3078.4478 reg_novel 116.723465
loss 4.311537
STEP 10 ================================
prereg loss 1.11352 regularization 3068.1238 reg_novel 106.14008
loss 4.287784
STEP 11 ================================
prereg loss 1.1074404 regularization 3057.7195 reg_novel 96.47092
loss 4.261631
STEP 12 ================================
prereg loss 1.099214 regularization 3047.2312 reg_novel 87.65023
loss 4.2340956
STEP 13 ================================
prereg loss 1.0899849 regularization 3036.687 reg_novel 79.610756
loss 4.2062826
STEP 14 ================================
prereg loss 1.081003 regularization 3026.0964 reg_novel 72.292694
loss 4.1793923
STEP 15 ================================
prereg loss 1.0731196 regularization 3015.4187 reg_novel 65.64646
loss 4.154185
STEP 16 ================================
prereg loss 1.0670525 regularization 3004.679 reg_novel 59.618904
loss 4.1313505
2022-06-05T22:40:51.562

julia> a_32 = deepcopy(trainable["network_matrix"])
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.186714), "norm-5"=>Dict("dict"=>0.00322582, "true"=>…
  "norm-5"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0673323), "norm-5"=>Dict("dict"=>0.232388, "true"=>0.11…
  "accum-4"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.134561), "norm-5"=>Dict("dict"=>-0.140239, "true"=>0.0…
  "dot-2"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0751777), "norm-5"=>Dict("dict"=>-0.0906212, "true"=>-…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.163065), "norm-5"=>Dict("dict"=>0.252548, "true"=>0.060…
  "compare-5" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.184183), "norm-5"=>Dict("dict"=>-0.383468, "true"=>-0.…
  "accum-3"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.143145), "norm-5"=>Dict("dict"=>-0.0728871, "true"=>0.1…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.147122), "norm-5"=>Dict("dict"=>0.584413, "true"=>-0.17…
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.47357), "norm-5"=>Dict("dict"=>-0.25454, "true"=>-0.12…
  "compare-4" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.330419), "norm-5"=>Dict("dict"=>0.114183, "true"=>-0.22…
  "compare-2" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.252292), "norm-5"=>Dict("dict"=>-0.251961, "true"=>0.00…
  "dot-1"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0446866), "norm-5"=>Dict("dict"=>0.182203, "true"=>0.1…
  "dot-3"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.232219), "norm-5"=>Dict("dict"=>-0.000354288, "true"=>…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.167996), "norm-5"=>Dict("dict"=>0.321145, "true"=>0.05…
  "compare-3" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.172211), "norm-5"=>Dict("dict"=>0.00331995, "true"=>-0.…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.216746), "norm-5"=>Dict("dict"=>-0.241362, "true"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0555282), "norm-5"=>Dict("dict"=>-0.0758398, "true"=>-…
  "compare-1" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.250754), "norm-5"=>Dict("dict"=>0.205651, "true"=>0.05…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.269885), "norm-5"=>Dict("dict"=>-0.237656, "true"=>-0.0…
  "dot-5"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.027104), "norm-5"=>Dict("dict"=>-0.22901, "true"=>0.034…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.457607), "norm-5"=>Dict("dict"=>0.0456774, "true"=>-0.0…

julia> count(a_32)
20308

julia> count_neg_interval(a_32, -0.8f0, 0.8f0)
4

julia> count_neg_interval(a_32, -0.9f0, 0.9f0)
2

julia> count_neg_interval(a_32, -0.7f0, 0.7f0)
12

julia> count_neg_interval(a_32, -0.6f0, 0.6f0)
56

julia> count_neg_interval(a_32, -0.5f0, 0.5f0)
235

julia> count_neg_interval(a_32, -0.4f0, 0.4f0)
781

julia> count_neg_interval(a_32, -0.3f0, 0.3f0)
2393

julia> count_neg_interval(a_32, -0.2f0, 0.2f0)
5952

julia> count_neg_interval(a_32, -0.1f0, 0.1f0)
11688

julia> count_neg_interval(a_32, -0.01f0, 0.01f0)
18350

julia> steps!(500)
2022-06-05T22:56:38.364
STEP 1 ================================
prereg loss 1.0630492 regularization 2993.875 reg_novel 54.156006
loss 4.11108
STEP 2 ================================
prereg loss 1.0607954 regularization 2983.0444 reg_novel 49.21233
loss 4.0930524
STEP 3 ================================
prereg loss 1.0597333 regularization 2972.1821 reg_novel 44.74751
loss 4.076663
STEP 4 ================================
prereg loss 1.0591066 regularization 2961.284 reg_novel 40.721573
loss 4.0611124
STEP 5 ================================
prereg loss 1.0583683 regularization 2950.359 reg_novel 37.09081
loss 4.0458183
STEP 6 ================================
prereg loss 1.057071 regularization 2939.376 reg_novel 33.81807
loss 4.030265
STEP 7 ================================
prereg loss 1.054742 regularization 2928.3992 reg_novel 30.866285
loss 4.0140076
STEP 8 ================================
prereg loss 1.0506523 regularization 2917.3704 reg_novel 28.205296
loss 3.9962282
STEP 9 ================================
prereg loss 1.0452834 regularization 2906.3518 reg_novel 25.811777
loss 3.977447
STEP 10 ================================
prereg loss 1.0397788 regularization 2895.3137 reg_novel 23.652288
loss 3.958745
STEP 11 ================================
prereg loss 1.0345963 regularization 2884.304 reg_novel 21.703773
loss 3.9406042
STEP 12 ================================
prereg loss 1.0278486 regularization 2873.2666 reg_novel 19.94691
loss 3.921062
STEP 13 ================================
prereg loss 1.0241371 regularization 2862.2341 reg_novel 18.360767
loss 3.9047322
STEP 14 ================================
prereg loss 1.0210551 regularization 2851.172 reg_novel 16.929789
loss 3.8891573
STEP 15 ================================
prereg loss 1.018372 regularization 2840.127 reg_novel 15.64006
loss 3.874139
STEP 16 ================================
prereg loss 1.0162747 regularization 2829.0874 reg_novel 14.475371
loss 3.8598375
STEP 17 ================================
prereg loss 1.0139805 regularization 2818.0427 reg_novel 13.423317
loss 3.8454466
STEP 18 ================================
prereg loss 1.0111359 regularization 2807.0159 reg_novel 12.472335
loss 3.830624
STEP 19 ================================
prereg loss 1.0080969 regularization 2795.9858 reg_novel 11.6113205
loss 3.815694
STEP 20 ================================
prereg loss 1.0058036 regularization 2784.954 reg_novel 10.839961
loss 3.8015978
STEP 21 ================================
prereg loss 1.0034878 regularization 2773.9294 reg_novel 10.141258
loss 3.7875588
STEP 22 ================================
prereg loss 1.0001322 regularization 2762.8855 reg_novel 9.513932
loss 3.7725315
STEP 23 ================================
prereg loss 0.9967028 regularization 2751.858 reg_novel 8.953316
loss 3.757514
STEP 24 ================================
prereg loss 0.9934333 regularization 2740.8062 reg_novel 8.457705
loss 3.7426972
STEP 25 ================================
prereg loss 0.9902063 regularization 2729.7715 reg_novel 8.018115
loss 3.7279959
STEP 26 ================================
prereg loss 0.98696494 regularization 2718.7722 reg_novel 7.6252174
loss 3.7133625
STEP 27 ================================
prereg loss 0.9836667 regularization 2707.7898 reg_novel 7.2711396
loss 3.6987276
STEP 28 ================================
prereg loss 0.98035717 regularization 2696.8203 reg_novel 6.9546065
loss 3.6841323
STEP 29 ================================
prereg loss 0.9772205 regularization 2685.8777 reg_novel 6.6713643
loss 3.6697698
STEP 30 ================================
prereg loss 0.97388536 regularization 2674.9258 reg_novel 6.417145
loss 3.6552281
STEP 31 ================================
prereg loss 0.96929437 regularization 2663.987 reg_novel 6.189085
loss 3.6394706
STEP 32 ================================
prereg loss 0.96477 regularization 2653.021 reg_novel 5.9847302
loss 3.623776
STEP 33 ================================
prereg loss 0.96049786 regularization 2642.0867 reg_novel 5.806298
loss 3.6083908
STEP 34 ================================
prereg loss 0.95698774 regularization 2631.1423 reg_novel 5.651562
loss 3.5937815
STEP 35 ================================
prereg loss 0.9539639 regularization 2620.2017 reg_novel 5.5139585
loss 3.5796795
STEP 36 ================================
prereg loss 0.95137423 regularization 2609.2957 reg_novel 5.397798
loss 3.566068
STEP 37 ================================
prereg loss 0.94962794 regularization 2598.385 reg_novel 5.301446
loss 3.5533144
STEP 38 ================================
prereg loss 0.94685024 regularization 2587.5159 reg_novel 5.2220044
loss 3.5395885
STEP 39 ================================
prereg loss 0.9435972 regularization 2576.679 reg_novel 5.1595316
loss 3.525436
STEP 40 ================================
prereg loss 0.93988574 regularization 2565.8557 reg_novel 5.1059837
loss 3.5108476
STEP 41 ================================
prereg loss 0.9363752 regularization 2555.03 reg_novel 5.0557613
loss 3.496461
STEP 42 ================================
prereg loss 0.93339473 regularization 2544.2358 reg_novel 5.0068374
loss 3.4826374
STEP 43 ================================
prereg loss 0.9306068 regularization 2533.4517 reg_novel 4.957269
loss 3.4690158
STEP 44 ================================
prereg loss 0.9280918 regularization 2522.6948 reg_novel 4.908794
loss 3.4556954
STEP 45 ================================
prereg loss 0.925042 regularization 2511.9438 reg_novel 4.868365
loss 3.4418542
STEP 46 ================================
prereg loss 0.92133254 regularization 2501.217 reg_novel 4.8357925
loss 3.4273856
STEP 47 ================================
prereg loss 0.9178374 regularization 2490.5007 reg_novel 4.8066487
loss 3.413145
STEP 48 ================================
prereg loss 0.91498417 regularization 2479.81 reg_novel 4.7778106
loss 3.3995724
STEP 49 ================================
prereg loss 0.9121609 regularization 2469.1575 reg_novel 4.750599
loss 3.386069
STEP 50 ================================
prereg loss 0.90997934 regularization 2458.5317 reg_novel 4.7307806
loss 3.373242
STEP 51 ================================
prereg loss 0.90718454 regularization 2447.9326 reg_novel 4.719849
loss 3.359837
STEP 52 ================================
prereg loss 0.9039854 regularization 2437.35 reg_novel 4.71279
loss 3.3460484
STEP 53 ================================
prereg loss 0.90090525 regularization 2426.775 reg_novel 4.709688
loss 3.3323898
STEP 54 ================================
prereg loss 0.8978424 regularization 2416.2234 reg_novel 4.709382
loss 3.3187754
STEP 55 ================================
prereg loss 0.8949679 regularization 2405.6936 reg_novel 4.7065225
loss 3.3053684
STEP 56 ================================
prereg loss 0.89234245 regularization 2395.175 reg_novel 4.706185
loss 3.292224
STEP 57 ================================
prereg loss 0.890075 regularization 2384.6855 reg_novel 4.706644
loss 3.2794673
STEP 58 ================================
prereg loss 0.8881162 regularization 2374.2188 reg_novel 4.7094607
loss 3.2670445
STEP 59 ================================
prereg loss 0.88589036 regularization 2363.805 reg_novel 4.7172503
loss 3.2544127
STEP 60 ================================
prereg loss 0.8827341 regularization 2353.417 reg_novel 4.724915
loss 3.2408762
STEP 61 ================================
prereg loss 0.8793625 regularization 2343.0662 reg_novel 4.7386866
loss 3.2271674
STEP 62 ================================
prereg loss 0.87627894 regularization 2332.7085 reg_novel 4.7499676
loss 3.2137375
STEP 63 ================================
prereg loss 0.87339526 regularization 2322.3992 reg_novel 4.7661037
loss 3.2005606
STEP 64 ================================
prereg loss 0.87047946 regularization 2312.0845 reg_novel 4.781217
loss 3.1873455
STEP 65 ================================
prereg loss 0.8677803 regularization 2301.8093 reg_novel 4.7945256
loss 3.1743846
STEP 66 ================================
prereg loss 0.8643682 regularization 2291.5574 reg_novel 4.808561
loss 3.1607344
STEP 67 ================================
prereg loss 0.8614402 regularization 2281.341 reg_novel 4.825023
loss 3.1476064
STEP 68 ================================
prereg loss 0.8591498 regularization 2271.1553 reg_novel 4.840366
loss 3.1351457
STEP 69 ================================
prereg loss 0.8573003 regularization 2260.9812 reg_novel 4.8482347
loss 3.1231298
STEP 70 ================================
prereg loss 0.8557931 regularization 2250.8472 reg_novel 4.8454857
loss 3.111486
STEP 71 ================================
prereg loss 0.85424095 regularization 2240.7112 reg_novel 4.842574
loss 3.0997946
STEP 72 ================================
prereg loss 0.85096306 regularization 2230.6118 reg_novel 4.834822
loss 3.08641
STEP 73 ================================
prereg loss 0.84831107 regularization 2220.5378 reg_novel 4.830155
loss 3.073679
STEP 74 ================================
prereg loss 0.84581536 regularization 2210.4944 reg_novel 4.8253875
loss 3.0611353
STEP 75 ================================
prereg loss 0.84432954 regularization 2200.4849 reg_novel 4.820083
loss 3.0496347
STEP 76 ================================
prereg loss 0.84242636 regularization 2190.4763 reg_novel 4.814469
loss 3.037717
STEP 77 ================================
prereg loss 0.84023964 regularization 2180.5 reg_novel 4.8131003
loss 3.0255527
STEP 78 ================================
prereg loss 0.83745 regularization 2170.5479 reg_novel 4.8135567
loss 3.0128117
STEP 79 ================================
prereg loss 0.8348546 regularization 2160.6445 reg_novel 4.8129125
loss 3.000312
STEP 80 ================================
prereg loss 0.8331761 regularization 2150.749 reg_novel 4.80987
loss 2.9887352
STEP 81 ================================
prereg loss 0.83092874 regularization 2140.9243 reg_novel 4.8070765
loss 2.9766603
STEP 82 ================================
prereg loss 0.82880586 regularization 2131.11 reg_novel 4.8052444
loss 2.9647214
STEP 83 ================================
prereg loss 0.82679933 regularization 2121.324 reg_novel 4.803554
loss 2.952927
STEP 84 ================================
prereg loss 0.825631 regularization 2111.5486 reg_novel 4.804473
loss 2.9419842
STEP 85 ================================
prereg loss 0.8240727 regularization 2101.7925 reg_novel 4.8098316
loss 2.930675
STEP 86 ================================
prereg loss 0.8220776 regularization 2092.0557 reg_novel 4.818118
loss 2.9189515
STEP 87 ================================
prereg loss 0.8206135 regularization 2082.3423 reg_novel 4.830634
loss 2.9077864
STEP 88 ================================
prereg loss 0.8185231 regularization 2072.652 reg_novel 4.837455
loss 2.8960128
STEP 89 ================================
prereg loss 0.81716365 regularization 2062.984 reg_novel 4.8428617
loss 2.8849905
STEP 90 ================================
prereg loss 0.81612176 regularization 2053.3562 reg_novel 4.841968
loss 2.8743203
STEP 91 ================================
prereg loss 0.81470305 regularization 2043.7435 reg_novel 4.839893
loss 2.8632865
STEP 92 ================================
prereg loss 0.8127768 regularization 2034.1808 reg_novel 4.8374376
loss 2.8517952
STEP 93 ================================
prereg loss 0.8113449 regularization 2024.6342 reg_novel 4.8409934
loss 2.8408203
STEP 94 ================================
prereg loss 0.8092614 regularization 2015.1317 reg_novel 4.845232
loss 2.8292382
STEP 95 ================================
prereg loss 0.8094318 regularization 2005.6282 reg_novel 4.8482056
loss 2.8199084
STEP 96 ================================
prereg loss 0.8061305 regularization 1996.1539 reg_novel 4.8450756
loss 2.8071299
STEP 97 ================================
prereg loss 0.80542415 regularization 1986.7041 reg_novel 4.8399286
loss 2.7969682
STEP 98 ================================
prereg loss 0.8079738 regularization 1977.2865 reg_novel 4.8368435
loss 2.7900972
STEP 99 ================================
prereg loss 0.8079473 regularization 1967.914 reg_novel 4.834639
loss 2.780696
STEP 100 ================================
prereg loss 0.8047427 regularization 1958.5684 reg_novel 4.833165
loss 2.7681444
STEP 101 ================================
prereg loss 0.79997677 regularization 1949.2784 reg_novel 4.824668
loss 2.75408
STEP 102 ================================
prereg loss 0.79790395 regularization 1940.0145 reg_novel 4.8199368
loss 2.7427385
STEP 103 ================================
prereg loss 0.8005641 regularization 1930.7858 reg_novel 4.819889
loss 2.7361698
STEP 104 ================================
prereg loss 0.7986145 regularization 1921.5819 reg_novel 4.8144174
loss 2.7250109
STEP 105 ================================
prereg loss 0.79393286 regularization 1912.4138 reg_novel 4.8071437
loss 2.711154
STEP 106 ================================
prereg loss 0.7924754 regularization 1903.2441 reg_novel 4.799865
loss 2.7005196
STEP 107 ================================
prereg loss 0.79322773 regularization 1894.1003 reg_novel 4.7988806
loss 2.692127
STEP 108 ================================
prereg loss 0.791838 regularization 1885.0228 reg_novel 4.800738
loss 2.6816616
STEP 109 ================================
prereg loss 0.7891125 regularization 1875.9895 reg_novel 4.8078537
loss 2.66991
STEP 110 ================================
prereg loss 0.7881902 regularization 1866.9991 reg_novel 4.817854
loss 2.6600072
STEP 111 ================================
prereg loss 0.78688544 regularization 1857.9991 reg_novel 4.8302245
loss 2.649715
STEP 112 ================================
prereg loss 0.78541803 regularization 1849.0404 reg_novel 4.8432374
loss 2.6393018
STEP 113 ================================
prereg loss 0.7846228 regularization 1840.1346 reg_novel 4.856416
loss 2.6296139
STEP 114 ================================
prereg loss 0.7837732 regularization 1831.2573 reg_novel 4.8675604
loss 2.6198983
STEP 115 ================================
prereg loss 0.78160304 regularization 1822.3848 reg_novel 4.8796864
loss 2.6088676
STEP 116 ================================
prereg loss 0.780321 regularization 1813.552 reg_novel 4.8957653
loss 2.598769
STEP 117 ================================
prereg loss 0.7789256 regularization 1804.7555 reg_novel 4.9101243
loss 2.5885913
STEP 118 ================================
prereg loss 0.77886456 regularization 1796.0112 reg_novel 4.920769
loss 2.5797966
STEP 119 ================================
prereg loss 0.7768977 regularization 1787.2906 reg_novel 4.932781
loss 2.5691211
STEP 120 ================================
prereg loss 0.773726 regularization 1778.5983 reg_novel 4.94652
loss 2.5572708
STEP 121 ================================
prereg loss 0.7730302 regularization 1769.9564 reg_novel 4.9576707
loss 2.5479443
STEP 122 ================================
prereg loss 0.7722001 regularization 1761.3292 reg_novel 4.9619403
loss 2.5384912
STEP 123 ================================
prereg loss 0.77022004 regularization 1752.7279 reg_novel 4.9658523
loss 2.527914
STEP 124 ================================
prereg loss 0.7689711 regularization 1744.1605 reg_novel 4.9707813
loss 2.5181026
STEP 125 ================================
prereg loss 0.76788425 regularization 1735.6444 reg_novel 4.9761047
loss 2.5085049
STEP 126 ================================
prereg loss 0.7668725 regularization 1727.1675 reg_novel 4.9718165
loss 2.499012
STEP 127 ================================
prereg loss 0.7665585 regularization 1718.734 reg_novel 4.965351
loss 2.490258
STEP 128 ================================
prereg loss 0.76502883 regularization 1710.3376 reg_novel 4.962066
loss 2.4803286
STEP 129 ================================
prereg loss 0.76337117 regularization 1701.9559 reg_novel 4.965582
loss 2.4702928
STEP 130 ================================
prereg loss 0.7622182 regularization 1693.6057 reg_novel 4.969861
loss 2.4607937
STEP 131 ================================
prereg loss 0.75997096 regularization 1685.279 reg_novel 4.9775834
loss 2.4502277
STEP 132 ================================
prereg loss 0.7606162 regularization 1676.9929 reg_novel 4.9944553
loss 2.4426036
STEP 133 ================================
prereg loss 0.7601092 regularization 1668.739 reg_novel 5.00382
loss 2.4338522
STEP 134 ================================
prereg loss 0.75816965 regularization 1660.4865 reg_novel 5.0082703
loss 2.4236646
STEP 135 ================================
prereg loss 0.75517327 regularization 1652.2866 reg_novel 5.0140324
loss 2.412474
STEP 136 ================================
prereg loss 0.752458 regularization 1644.1105 reg_novel 5.0209103
loss 2.4015894
STEP 137 ================================
prereg loss 0.7537416 regularization 1635.9694 reg_novel 5.0247545
loss 2.3947358
STEP 138 ================================
prereg loss 0.7513976 regularization 1627.8702 reg_novel 5.0326004
loss 2.3843007
STEP 139 ================================
prereg loss 0.746819 regularization 1619.7982 reg_novel 5.0454254
loss 2.3716626
STEP 140 ================================
prereg loss 0.7462997 regularization 1611.7551 reg_novel 5.05695
loss 2.363112
STEP 141 ================================
prereg loss 0.74624187 regularization 1603.7538 reg_novel 5.0591846
loss 2.3550549
STEP 142 ================================
prereg loss 0.7430703 regularization 1595.7819 reg_novel 5.0503597
loss 2.3439026
STEP 143 ================================
prereg loss 0.7384415 regularization 1587.8564 reg_novel 5.039854
loss 2.331338
STEP 144 ================================
prereg loss 0.73887014 regularization 1579.9658 reg_novel 5.0319223
loss 2.323868
STEP 145 ================================
prereg loss 0.7382213 regularization 1572.1057 reg_novel 5.0335164
loss 2.3153605
STEP 146 ================================
prereg loss 0.73370963 regularization 1564.2717 reg_novel 5.040873
loss 2.3030224
STEP 147 ================================
prereg loss 0.72834355 regularization 1556.4982 reg_novel 5.0491896
loss 2.289891
STEP 148 ================================
prereg loss 0.73042446 regularization 1548.7617 reg_novel 5.060954
loss 2.2842472
STEP 149 ================================
prereg loss 0.73082876 regularization 1541.0541 reg_novel 5.0722895
loss 2.2769551
STEP 150 ================================
prereg loss 0.7265023 regularization 1533.3584 reg_novel 5.0786366
loss 2.2649393
STEP 151 ================================
prereg loss 0.7189368 regularization 1525.6791 reg_novel 5.0766864
loss 2.2496924
STEP 152 ================================
prereg loss 0.7163683 regularization 1518.0254 reg_novel 5.0759025
loss 2.2394698
STEP 153 ================================
prereg loss 0.7147841 regularization 1510.433 reg_novel 5.077296
loss 2.2302945
STEP 154 ================================
prereg loss 0.710285 regularization 1502.8917 reg_novel 5.085195
loss 2.218262
STEP 155 ================================
prereg loss 0.7034129 regularization 1495.3784 reg_novel 5.092826
loss 2.2038841
STEP 156 ================================
prereg loss 0.6993726 regularization 1487.916 reg_novel 5.0997863
loss 2.1923885
STEP 157 ================================
prereg loss 0.6966292 regularization 1480.5101 reg_novel 5.100436
loss 2.18224
STEP 158 ================================
prereg loss 0.6886041 regularization 1473.1353 reg_novel 5.092476
loss 2.166832
STEP 159 ================================
prereg loss 0.682611 regularization 1465.8215 reg_novel 5.0773215
loss 2.15351
STEP 160 ================================
prereg loss 0.67796415 regularization 1458.5385 reg_novel 5.0650897
loss 2.1415677
STEP 161 ================================
prereg loss 0.67083895 regularization 1451.2932 reg_novel 5.0576124
loss 2.1271896
STEP 162 ================================
prereg loss 0.66390556 regularization 1444.0857 reg_novel 5.06086
loss 2.1130521
STEP 163 ================================
prereg loss 0.6595749 regularization 1436.9049 reg_novel 5.06609
loss 2.101546
STEP 164 ================================
prereg loss 0.65282106 regularization 1429.7572 reg_novel 5.068793
loss 2.087647
STEP 165 ================================
prereg loss 0.6460246 regularization 1422.6373 reg_novel 5.080703
loss 2.0737426
STEP 166 ================================
prereg loss 0.6396831 regularization 1415.5293 reg_novel 5.0973563
loss 2.06031
STEP 167 ================================
prereg loss 0.63333356 regularization 1408.4554 reg_novel 5.120587
loss 2.0469098
STEP 168 ================================
prereg loss 0.6255153 regularization 1401.3984 reg_novel 5.14195
loss 2.0320559
STEP 169 ================================
prereg loss 0.61954886 regularization 1394.3804 reg_novel 5.1647305
loss 2.019094
STEP 170 ================================
prereg loss 0.6084354 regularization 1387.4071 reg_novel 5.1772475
loss 2.00102
STEP 171 ================================
prereg loss 0.5997355 regularization 1380.473 reg_novel 5.1930118
loss 1.9854016
STEP 172 ================================
prereg loss 0.5897628 regularization 1373.5753 reg_novel 5.209087
loss 1.9685472
STEP 173 ================================
prereg loss 0.578942 regularization 1366.7131 reg_novel 5.224669
loss 1.9508798
STEP 174 ================================
prereg loss 0.57022774 regularization 1359.8794 reg_novel 5.2439
loss 1.9353511
STEP 175 ================================
prereg loss 0.558423 regularization 1353.0957 reg_novel 5.2536983
loss 1.9167724
STEP 176 ================================
prereg loss 0.549863 regularization 1346.3329 reg_novel 5.2576137
loss 1.9014535
STEP 177 ================================
prereg loss 0.53769714 regularization 1339.622 reg_novel 5.2611637
loss 1.8825803
STEP 178 ================================
prereg loss 0.5305283 regularization 1332.9172 reg_novel 5.2690268
loss 1.8687147
STEP 179 ================================
prereg loss 0.5203098 regularization 1326.2312 reg_novel 5.276138
loss 1.8518171
STEP 180 ================================
prereg loss 0.50854903 regularization 1319.6025 reg_novel 5.2813964
loss 1.8334332
STEP 181 ================================
prereg loss 0.4993167 regularization 1312.9851 reg_novel 5.2884493
loss 1.8175904
STEP 182 ================================
prereg loss 0.48504174 regularization 1306.3829 reg_novel 5.299801
loss 1.7967246
STEP 183 ================================
prereg loss 0.46866542 regularization 1299.8324 reg_novel 5.3112936
loss 1.7738092
STEP 184 ================================
prereg loss 0.45054683 regularization 1293.3273 reg_novel 5.32525
loss 1.7491994
STEP 185 ================================
prereg loss 0.4352439 regularization 1286.8223 reg_novel 5.3331137
loss 1.7273993
STEP 186 ================================
prereg loss 0.41979596 regularization 1280.3599 reg_novel 5.340987
loss 1.7054969
STEP 187 ================================
prereg loss 0.40410346 regularization 1273.9559 reg_novel 5.350153
loss 1.6834095
STEP 188 ================================
prereg loss 0.38979742 regularization 1267.5828 reg_novel 5.348105
loss 1.6627283
STEP 189 ================================
prereg loss 0.37615922 regularization 1261.2422 reg_novel 5.346255
loss 1.6427478
STEP 190 ================================
prereg loss 0.3688775 regularization 1254.9364 reg_novel 5.342296
loss 1.6291562
STEP 191 ================================
prereg loss 0.35152197 regularization 1248.7249 reg_novel 5.3410835
loss 1.605588
STEP 192 ================================
prereg loss 0.3535887 regularization 1242.5583 reg_novel 5.3432784
loss 1.6014904
STEP 193 ================================
prereg loss 0.3468404 regularization 1236.3739 reg_novel 5.3353467
loss 1.5885496
STEP 194 ================================
prereg loss 0.3332866 regularization 1230.2014 reg_novel 5.3203053
loss 1.5688084
STEP 195 ================================
prereg loss 0.32508308 regularization 1224.0372 reg_novel 5.3096366
loss 1.55443
STEP 196 ================================
prereg loss 0.32267058 regularization 1217.8868 reg_novel 5.3053374
loss 1.5458628
STEP 197 ================================
prereg loss 0.31038612 regularization 1211.7803 reg_novel 5.3113146
loss 1.5274777
STEP 198 ================================
prereg loss 0.30343243 regularization 1205.6838 reg_novel 5.3239717
loss 1.5144404
STEP 199 ================================
prereg loss 0.28666615 regularization 1199.6426 reg_novel 5.3377686
loss 1.4916465
STEP 200 ================================
prereg loss 0.2774884 regularization 1193.5968 reg_novel 5.358243
loss 1.4764435
STEP 201 ================================
prereg loss 0.27269384 regularization 1187.5605 reg_novel 5.389131
loss 1.4656435
STEP 202 ================================
prereg loss 0.2720489 regularization 1181.5177 reg_novel 5.4041734
loss 1.4589708
STEP 203 ================================
prereg loss 0.24774265 regularization 1175.5103 reg_novel 5.4266305
loss 1.4286796
STEP 204 ================================
prereg loss 0.24950962 regularization 1169.5447 reg_novel 5.4522977
loss 1.4245065
STEP 205 ================================
prereg loss 0.23876277 regularization 1163.5638 reg_novel 5.473375
loss 1.4078
STEP 206 ================================
prereg loss 0.23813695 regularization 1157.6426 reg_novel 5.492594
loss 1.401272
STEP 207 ================================
prereg loss 0.22969751 regularization 1151.7656 reg_novel 5.518646
loss 1.3869818
STEP 208 ================================
prereg loss 0.2107085 regularization 1145.909 reg_novel 5.532265
loss 1.3621498
STEP 209 ================================
prereg loss 0.20456894 regularization 1140.1006 reg_novel 5.544884
loss 1.3502145
STEP 210 ================================
prereg loss 0.19906184 regularization 1134.3303 reg_novel 5.5692153
loss 1.3389615
STEP 211 ================================
prereg loss 0.18886396 regularization 1128.5847 reg_novel 5.597602
loss 1.3230463
STEP 212 ================================
prereg loss 0.26380232 regularization 1122.8674 reg_novel 5.609805
loss 1.3922795
STEP 213 ================================
prereg loss 0.2074193 regularization 1117.2125 reg_novel 5.6331344
loss 1.3302649
STEP 214 ================================
prereg loss 0.2158285 regularization 1111.55 reg_novel 5.63676
loss 1.3330154
STEP 215 ================================
prereg loss 0.18110491 regularization 1105.8893 reg_novel 5.631367
loss 1.2926255
STEP 216 ================================
prereg loss 0.20357911 regularization 1100.2788 reg_novel 5.6337833
loss 1.3094918
STEP 217 ================================
prereg loss 0.21270171 regularization 1094.7635 reg_novel 5.6509075
loss 1.3131162
STEP 218 ================================
prereg loss 0.3038137 regularization 1089.2567 reg_novel 5.6597176
loss 1.3987302
STEP 219 ================================
prereg loss 0.18598755 regularization 1083.7096 reg_novel 5.6632648
loss 1.2753605
STEP 220 ================================
prereg loss 0.16078798 regularization 1078.2247 reg_novel 5.674337
loss 1.2446871
STEP 221 ================================
prereg loss 0.18871197 regularization 1072.7397 reg_novel 5.6905794
loss 1.2671424
STEP 222 ================================
prereg loss 0.17577894 regularization 1067.2764 reg_novel 5.704202
loss 1.2487595
STEP 223 ================================
prereg loss 0.21760319 regularization 1061.8055 reg_novel 5.7193284
loss 1.2851281
STEP 224 ================================
prereg loss 0.17007361 regularization 1056.3818 reg_novel 5.7429743
loss 1.2321986
STEP 225 ================================
prereg loss 0.17093055 regularization 1051.0011 reg_novel 5.770763
loss 1.2277025
STEP 226 ================================
prereg loss 0.21363558 regularization 1045.6615 reg_novel 5.8029375
loss 1.2651001
STEP 227 ================================
prereg loss 0.15956447 regularization 1040.3315 reg_novel 5.822938
loss 1.205719
STEP 228 ================================
prereg loss 0.19612978 regularization 1035.0065 reg_novel 5.841025
loss 1.2369773
STEP 229 ================================
prereg loss 0.21671395 regularization 1029.7247 reg_novel 5.847502
loss 1.2522862
STEP 230 ================================
prereg loss 0.2708256 regularization 1024.4862 reg_novel 5.8477354
loss 1.3011596
STEP 231 ================================
prereg loss 0.19444829 regularization 1019.2894 reg_novel 5.84938
loss 1.2195871
STEP 232 ================================
prereg loss 0.18969752 regularization 1014.0997 reg_novel 5.8548117
loss 1.2096521
STEP 233 ================================
prereg loss 0.18248445 regularization 1008.91486 reg_novel 5.864162
loss 1.1972635
STEP 234 ================================
prereg loss 0.17585877 regularization 1003.7412 reg_novel 5.8763666
loss 1.1854764
STEP 235 ================================
prereg loss 0.17906572 regularization 998.60944 reg_novel 5.8926096
loss 1.1835678
STEP 236 ================================
prereg loss 0.17729187 regularization 993.51855 reg_novel 5.9057755
loss 1.1767162
STEP 237 ================================
prereg loss 0.17067043 regularization 988.4583 reg_novel 5.9117737
loss 1.1650405
STEP 238 ================================
prereg loss 0.16042863 regularization 983.39905 reg_novel 5.9090214
loss 1.1497368
STEP 239 ================================
prereg loss 0.15668267 regularization 978.3573 reg_novel 5.9146156
loss 1.1409547
STEP 240 ================================
prereg loss 0.15157104 regularization 973.3555 reg_novel 5.9260683
loss 1.1308527
STEP 241 ================================
prereg loss 0.15168002 regularization 968.40063 reg_novel 5.940578
loss 1.1260213
STEP 242 ================================
prereg loss 0.15515755 regularization 963.48395 reg_novel 5.9605155
loss 1.1246021
STEP 243 ================================
prereg loss 0.15375309 regularization 958.5729 reg_novel 5.977831
loss 1.1183038
STEP 244 ================================
prereg loss 0.14943588 regularization 953.68463 reg_novel 5.993072
loss 1.1091137
STEP 245 ================================
prereg loss 0.15031423 regularization 948.8212 reg_novel 6.0224476
loss 1.105158
STEP 246 ================================
prereg loss 0.14037256 regularization 943.99414 reg_novel 6.046859
loss 1.0904136
STEP 247 ================================
prereg loss 0.14552231 regularization 939.2053 reg_novel 6.078497
loss 1.0908061
STEP 248 ================================
prereg loss 0.13414013 regularization 934.42114 reg_novel 6.1154876
loss 1.0746768
STEP 249 ================================
prereg loss 0.1407902 regularization 929.6517 reg_novel 6.1580105
loss 1.0766
STEP 250 ================================
prereg loss 0.1362925 regularization 924.9187 reg_novel 6.1955743
loss 1.0674068
STEP 251 ================================
prereg loss 0.11985656 regularization 920.24963 reg_novel 6.2334814
loss 1.0463398
STEP 252 ================================
prereg loss 0.12156455 regularization 915.58417 reg_novel 6.271374
loss 1.0434201
STEP 253 ================================
prereg loss 0.11898239 regularization 910.9332 reg_novel 6.3074074
loss 1.036223
STEP 254 ================================
prereg loss 0.11518111 regularization 906.316 reg_novel 6.3495636
loss 1.0278467
STEP 255 ================================
prereg loss 0.11701217 regularization 901.713 reg_novel 6.3822923
loss 1.0251075
STEP 256 ================================
prereg loss 0.121446125 regularization 897.1521 reg_novel 6.403647
loss 1.0250019
STEP 257 ================================
prereg loss 0.11564793 regularization 892.6579 reg_novel 6.4206085
loss 1.0147265
STEP 258 ================================
prereg loss 0.11720622 regularization 888.17773 reg_novel 6.442778
loss 1.0118268
STEP 259 ================================
prereg loss 0.11192047 regularization 883.68945 reg_novel 6.4607234
loss 1.0020707
STEP 260 ================================
prereg loss 0.10573143 regularization 879.1996 reg_novel 6.485351
loss 0.9914164
STEP 261 ================================
prereg loss 0.1028541 regularization 874.7766 reg_novel 6.5078697
loss 0.9841386
STEP 262 ================================
prereg loss 0.10930897 regularization 870.40094 reg_novel 6.537089
loss 0.986247
STEP 263 ================================
prereg loss 0.103237934 regularization 866.0391 reg_novel 6.5643525
loss 0.97584146
STEP 264 ================================
prereg loss 0.09985296 regularization 861.7052 reg_novel 6.5912194
loss 0.9681494
STEP 265 ================================
prereg loss 0.10031922 regularization 857.3907 reg_novel 6.617307
loss 0.9643272
STEP 266 ================================
prereg loss 0.097384505 regularization 853.0958 reg_novel 6.643638
loss 0.95712405
STEP 267 ================================
prereg loss 0.09798171 regularization 848.8407 reg_novel 6.6733007
loss 0.9534957
STEP 268 ================================
prereg loss 0.09753654 regularization 844.5981 reg_novel 6.705902
loss 0.94884056
STEP 269 ================================
prereg loss 0.09237622 regularization 840.3563 reg_novel 6.7427006
loss 0.9394753
STEP 270 ================================
prereg loss 0.0920299 regularization 836.138 reg_novel 6.77417
loss 0.9349421
STEP 271 ================================
prereg loss 0.08925906 regularization 831.94305 reg_novel 6.812172
loss 0.9280143
STEP 272 ================================
prereg loss 0.08965787 regularization 827.7768 reg_novel 6.849974
loss 0.9242847
STEP 273 ================================
prereg loss 0.08856101 regularization 823.6397 reg_novel 6.882277
loss 0.919083
STEP 274 ================================
prereg loss 0.086470425 regularization 819.5261 reg_novel 6.912624
loss 0.9129092
STEP 275 ================================
prereg loss 0.08659565 regularization 815.45 reg_novel 6.941236
loss 0.908987
STEP 276 ================================
prereg loss 0.08503168 regularization 811.4124 reg_novel 6.9703903
loss 0.90341455
STEP 277 ================================
prereg loss 0.085242465 regularization 807.40234 reg_novel 6.998546
loss 0.89964336
STEP 278 ================================
prereg loss 0.085740186 regularization 803.41754 reg_novel 7.0263386
loss 0.8961841
STEP 279 ================================
prereg loss 0.081781276 regularization 799.4462 reg_novel 7.0568886
loss 0.88828444
STEP 280 ================================
prereg loss 0.08113867 regularization 795.5029 reg_novel 7.080216
loss 0.8837218
STEP 281 ================================
prereg loss 0.080514364 regularization 791.5935 reg_novel 7.1122355
loss 0.8792202
STEP 282 ================================
prereg loss 0.08496951 regularization 787.67676 reg_novel 7.150594
loss 0.8797969
STEP 283 ================================
prereg loss 0.07809925 regularization 783.7879 reg_novel 7.198293
loss 0.8690855
STEP 284 ================================
prereg loss 0.08026158 regularization 779.9391 reg_novel 7.2475085
loss 0.8674482
STEP 285 ================================
prereg loss 0.076816805 regularization 776.0893 reg_novel 7.290728
loss 0.8601968
STEP 286 ================================
prereg loss 0.07715632 regularization 772.2656 reg_novel 7.32985
loss 0.8567518
STEP 287 ================================
prereg loss 0.07817941 regularization 768.4753 reg_novel 7.3732595
loss 0.854028
STEP 288 ================================
prereg loss 0.07564701 regularization 764.708 reg_novel 7.4224133
loss 0.8477774
STEP 289 ================================
prereg loss 0.0743176 regularization 760.92554 reg_novel 7.474798
loss 0.84271795
STEP 290 ================================
prereg loss 0.074625485 regularization 757.18274 reg_novel 7.521353
loss 0.8393296
STEP 291 ================================
prereg loss 0.07272566 regularization 753.4847 reg_novel 7.5722704
loss 0.8337827
STEP 292 ================================
prereg loss 0.07246966 regularization 749.82385 reg_novel 7.616001
loss 0.8299095
STEP 293 ================================
prereg loss 0.07224019 regularization 746.14905 reg_novel 7.6432467
loss 0.82603246
STEP 294 ================================
prereg loss 0.07061135 regularization 742.48474 reg_novel 7.6693697
loss 0.8207655
STEP 295 ================================
prereg loss 0.07002729 regularization 738.86847 reg_novel 7.6954393
loss 0.8165912
STEP 296 ================================
prereg loss 0.06902856 regularization 735.2833 reg_novel 7.728552
loss 0.8120405
STEP 297 ================================
prereg loss 0.06882097 regularization 731.68494 reg_novel 7.7634473
loss 0.8082694
STEP 298 ================================
prereg loss 0.06858583 regularization 728.1049 reg_novel 7.798311
loss 0.8044891
STEP 299 ================================
prereg loss 0.06744973 regularization 724.5576 reg_novel 7.8337946
loss 0.79984117
STEP 300 ================================
prereg loss 0.066682614 regularization 721.024 reg_novel 7.873508
loss 0.7955802
STEP 301 ================================
prereg loss 0.065999374 regularization 717.51227 reg_novel 7.919456
loss 0.7914311
STEP 302 ================================
prereg loss 0.065661825 regularization 714.0338 reg_novel 7.9719505
loss 0.78766763
STEP 303 ================================
prereg loss 0.06554232 regularization 710.6034 reg_novel 8.025449
loss 0.7841712
STEP 304 ================================
prereg loss 0.06419313 regularization 707.1809 reg_novel 8.078157
loss 0.7794522
STEP 305 ================================
prereg loss 0.06355973 regularization 703.7684 reg_novel 8.126112
loss 0.7754543
STEP 306 ================================
prereg loss 0.06292837 regularization 700.37396 reg_novel 8.172163
loss 0.77147454
STEP 307 ================================
prereg loss 0.06251715 regularization 696.9898 reg_novel 8.225317
loss 0.7677323
STEP 308 ================================
prereg loss 0.06221358 regularization 693.6086 reg_novel 8.281132
loss 0.76410335
STEP 309 ================================
prereg loss 0.06126751 regularization 690.23334 reg_novel 8.336506
loss 0.7598373
STEP 310 ================================
prereg loss 0.06095212 regularization 686.9216 reg_novel 8.39088
loss 0.7562646
STEP 311 ================================
prereg loss 0.060246527 regularization 683.6487 reg_novel 8.436651
loss 0.7523319
STEP 312 ================================
prereg loss 0.059880037 regularization 680.4076 reg_novel 8.463778
loss 0.74875146
STEP 313 ================================
prereg loss 0.05925126 regularization 677.1984 reg_novel 8.479648
loss 0.7449294
STEP 314 ================================
prereg loss 0.058841344 regularization 673.9989 reg_novel 8.500321
loss 0.74134064
STEP 315 ================================
prereg loss 0.05832746 regularization 670.8169 reg_novel 8.524259
loss 0.7376686
STEP 316 ================================
prereg loss 0.057978004 regularization 667.6461 reg_novel 8.550597
loss 0.7341747
STEP 317 ================================
prereg loss 0.05747017 regularization 664.4783 reg_novel 8.577357
loss 0.7305258
STEP 318 ================================
prereg loss 0.056912426 regularization 661.34985 reg_novel 8.615387
loss 0.7268777
STEP 319 ================================
prereg loss 0.056527518 regularization 658.2155 reg_novel 8.655015
loss 0.72339803
STEP 320 ================================
prereg loss 0.056118496 regularization 655.10785 reg_novel 8.699265
loss 0.7199256
STEP 321 ================================
prereg loss 0.055782408 regularization 651.99243 reg_novel 8.747379
loss 0.7165222
STEP 322 ================================
prereg loss 0.0552426 regularization 648.8828 reg_novel 8.797347
loss 0.7129228
STEP 323 ================================
prereg loss 0.05483197 regularization 645.8003 reg_novel 8.8548
loss 0.70948714
STEP 324 ================================
prereg loss 0.05490384 regularization 642.73816 reg_novel 8.9195385
loss 0.70656157
STEP 325 ================================
prereg loss 0.05396855 regularization 639.6951 reg_novel 8.980504
loss 0.7026442
STEP 326 ================================
prereg loss 0.05357813 regularization 636.68024 reg_novel 9.0285
loss 0.6992869
STEP 327 ================================
prereg loss 0.053800527 regularization 633.6665 reg_novel 9.076879
loss 0.69654393
STEP 328 ================================
prereg loss 0.05267422 regularization 630.68304 reg_novel 9.130874
loss 0.6924882
STEP 329 ================================
prereg loss 0.05275004 regularization 627.7185 reg_novel 9.183384
loss 0.68965197
STEP 330 ================================
prereg loss 0.05375611 regularization 624.7524 reg_novel 9.235804
loss 0.6877443
STEP 331 ================================
prereg loss 0.052691482 regularization 621.8365 reg_novel 9.299371
loss 0.6838274
STEP 332 ================================
prereg loss 0.0525829 regularization 618.9495 reg_novel 9.360805
loss 0.68089324
STEP 333 ================================
prereg loss 0.05089509 regularization 616.06274 reg_novel 9.41644
loss 0.6763743
STEP 334 ================================
prereg loss 0.05123698 regularization 613.20746 reg_novel 9.468665
loss 0.6739131
STEP 335 ================================
prereg loss 0.050372686 regularization 610.3971 reg_novel 9.516207
loss 0.67028594
STEP 336 ================================
prereg loss 0.050307635 regularization 607.58594 reg_novel 9.564094
loss 0.6674577
STEP 337 ================================
prereg loss 0.04962044 regularization 604.79114 reg_novel 9.622842
loss 0.6640344
STEP 338 ================================
prereg loss 0.0491347 regularization 602.0315 reg_novel 9.678704
loss 0.6608449
STEP 339 ================================
prereg loss 0.048875246 regularization 599.2887 reg_novel 9.738044
loss 0.657902
STEP 340 ================================
prereg loss 0.04842029 regularization 596.56757 reg_novel 9.795149
loss 0.654783
STEP 341 ================================
prereg loss 0.048200283 regularization 593.87756 reg_novel 9.840422
loss 0.65191835
STEP 342 ================================
prereg loss 0.04768072 regularization 591.2239 reg_novel 9.884353
loss 0.648789
STEP 343 ================================
prereg loss 0.047513206 regularization 588.5608 reg_novel 9.931054
loss 0.64600503
STEP 344 ================================
prereg loss 0.04700277 regularization 585.9197 reg_novel 9.989463
loss 0.6429119
STEP 345 ================================
prereg loss 0.046614323 regularization 583.28894 reg_novel 10.060635
loss 0.6399639
STEP 346 ================================
prereg loss 0.04677686 regularization 580.6649 reg_novel 10.132501
loss 0.6375743
STEP 347 ================================
prereg loss 0.046075746 regularization 578.04047 reg_novel 10.202136
loss 0.63431835
STEP 348 ================================
prereg loss 0.04605219 regularization 575.44 reg_novel 10.265382
loss 0.6317577
STEP 349 ================================
prereg loss 0.04807276 regularization 572.8747 reg_novel 10.329425
loss 0.6312769
STEP 350 ================================
prereg loss 0.045867592 regularization 570.33527 reg_novel 10.391313
loss 0.6265942
STEP 351 ================================
prereg loss 0.046547547 regularization 567.8307 reg_novel 10.46028
loss 0.62483853
STEP 352 ================================
prereg loss 0.04614135 regularization 565.3381 reg_novel 10.530072
loss 0.62200946
STEP 353 ================================
prereg loss 0.04653058 regularization 562.8477 reg_novel 10.593406
loss 0.61997175
STEP 354 ================================
prereg loss 0.04425483 regularization 560.387 reg_novel 10.646215
loss 0.61528814
STEP 355 ================================
prereg loss 0.04965404 regularization 557.93445 reg_novel 10.697955
loss 0.6182865
STEP 356 ================================
prereg loss 0.11867152 regularization 555.48114 reg_novel 10.753869
loss 0.6849066
STEP 357 ================================
prereg loss 0.33247527 regularization 553.08636 reg_novel 10.828338
loss 0.89638996
STEP 358 ================================
prereg loss 0.05015805 regularization 550.6714 reg_novel 10.873413
loss 0.61170286
STEP 359 ================================
prereg loss 0.27223223 regularization 548.25024 reg_novel 10.920015
loss 0.8314025
STEP 360 ================================
prereg loss 0.124140605 regularization 545.87573 reg_novel 10.967655
loss 0.6809841
STEP 361 ================================
prereg loss 0.08864092 regularization 543.4969 reg_novel 11.011999
loss 0.64314985
STEP 362 ================================
prereg loss 0.09719055 regularization 541.1186 reg_novel 11.061483
loss 0.6493707
STEP 363 ================================
prereg loss 0.1126342 regularization 538.77216 reg_novel 11.12822
loss 0.6625346
STEP 364 ================================
prereg loss 0.10896586 regularization 536.4497 reg_novel 11.208959
loss 0.65662456
STEP 365 ================================
prereg loss 0.08640585 regularization 534.1722 reg_novel 11.28692
loss 0.63186496
STEP 366 ================================
prereg loss 0.08861754 regularization 531.9031 reg_novel 11.366176
loss 0.63188684
STEP 367 ================================
prereg loss 0.08680021 regularization 529.6634 reg_novel 11.4443035
loss 0.627908
STEP 368 ================================
prereg loss 0.09144212 regularization 527.43726 reg_novel 11.513407
loss 0.6303928
STEP 369 ================================
prereg loss 0.095351666 regularization 525.2125 reg_novel 11.582008
loss 0.63214624
STEP 370 ================================
prereg loss 0.09495012 regularization 523.01996 reg_novel 11.649396
loss 0.6296195
STEP 371 ================================
prereg loss 0.10529319 regularization 520.8386 reg_novel 11.714091
loss 0.63784593
STEP 372 ================================
prereg loss 0.09379506 regularization 518.6537 reg_novel 11.780337
loss 0.6242291
STEP 373 ================================
prereg loss 0.08044214 regularization 516.4739 reg_novel 11.847593
loss 0.60876364
STEP 374 ================================
prereg loss 0.07578107 regularization 514.3169 reg_novel 11.920735
loss 0.6020187
STEP 375 ================================
prereg loss 0.06942714 regularization 512.1611 reg_novel 11.994357
loss 0.5935826
STEP 376 ================================
prereg loss 0.06528675 regularization 510.00107 reg_novel 12.057583
loss 0.5873454
STEP 377 ================================
prereg loss 0.06134917 regularization 507.8735 reg_novel 12.129327
loss 0.581352
STEP 378 ================================
prereg loss 0.05690587 regularization 505.77777 reg_novel 12.191378
loss 0.574875
STEP 379 ================================
prereg loss 0.05424101 regularization 503.68668 reg_novel 12.254589
loss 0.5701823
STEP 380 ================================
prereg loss 0.054481335 regularization 501.62515 reg_novel 12.318526
loss 0.56842506
STEP 381 ================================
prereg loss 0.054708175 regularization 499.58722 reg_novel 12.385963
loss 0.5666814
STEP 382 ================================
prereg loss 0.05261457 regularization 497.5566 reg_novel 12.461226
loss 0.56263244
STEP 383 ================================
prereg loss 0.049090754 regularization 495.53696 reg_novel 12.528275
loss 0.557156
STEP 384 ================================
prereg loss 0.04608237 regularization 493.53198 reg_novel 12.591163
loss 0.55220556
STEP 385 ================================
prereg loss 0.04551163 regularization 491.55316 reg_novel 12.663332
loss 0.5497281
STEP 386 ================================
prereg loss 0.04543713 regularization 489.5849 reg_novel 12.734552
loss 0.5477566
STEP 387 ================================
prereg loss 0.04628237 regularization 487.61096 reg_novel 12.799954
loss 0.5466933
STEP 388 ================================
prereg loss 0.044588048 regularization 485.64536 reg_novel 12.871825
loss 0.54310524
STEP 389 ================================
prereg loss 0.044031203 regularization 483.69427 reg_novel 12.952337
loss 0.5406778
STEP 390 ================================
prereg loss 0.043822907 regularization 481.7505 reg_novel 13.03919
loss 0.5386126
STEP 391 ================================
prereg loss 0.04322859 regularization 479.80365 reg_novel 13.117995
loss 0.5361503
STEP 392 ================================
prereg loss 0.043129317 regularization 477.88135 reg_novel 13.194924
loss 0.5342056
STEP 393 ================================
prereg loss 0.04321099 regularization 475.9644 reg_novel 13.269125
loss 0.53244454
STEP 394 ================================
prereg loss 0.04276993 regularization 474.0883 reg_novel 13.336936
loss 0.5301952
STEP 395 ================================
prereg loss 0.042448387 regularization 472.21976 reg_novel 13.404592
loss 0.5280727
STEP 396 ================================
prereg loss 0.042545203 regularization 470.35687 reg_novel 13.46974
loss 0.52637184
STEP 397 ================================
prereg loss 0.04182287 regularization 468.50742 reg_novel 13.531866
loss 0.5238622
STEP 398 ================================
prereg loss 0.04086457 regularization 466.68173 reg_novel 13.594285
loss 0.52114064
STEP 399 ================================
prereg loss 0.04051293 regularization 464.88843 reg_novel 13.64929
loss 0.51905066
STEP 400 ================================
prereg loss 0.039856065 regularization 463.08685 reg_novel 13.698308
loss 0.51664126
STEP 401 ================================
prereg loss 0.039185032 regularization 461.30875 reg_novel 13.759447
loss 0.51425326
STEP 402 ================================
prereg loss 0.038930338 regularization 459.50757 reg_novel 13.818287
loss 0.5122562
STEP 403 ================================
prereg loss 0.038212396 regularization 457.73212 reg_novel 13.880619
loss 0.50982517
STEP 404 ================================
prereg loss 0.037624355 regularization 455.962 reg_novel 13.95067
loss 0.50753707
STEP 405 ================================
prereg loss 0.037245464 regularization 454.18347 reg_novel 14.024632
loss 0.5054536
STEP 406 ================================
prereg loss 0.03678771 regularization 452.4104 reg_novel 14.1
loss 0.50329816
STEP 407 ================================
prereg loss 0.03610578 regularization 450.65988 reg_novel 14.179657
loss 0.5009453
STEP 408 ================================
prereg loss 0.035896428 regularization 448.90747 reg_novel 14.254539
loss 0.49905846
STEP 409 ================================
prereg loss 0.03568354 regularization 447.17905 reg_novel 14.319635
loss 0.49718225
STEP 410 ================================
prereg loss 0.035043586 regularization 445.4993 reg_novel 14.387961
loss 0.4949309
STEP 411 ================================
prereg loss 0.03482174 regularization 443.81543 reg_novel 14.450721
loss 0.49308792
STEP 412 ================================
prereg loss 0.03463432 regularization 442.15836 reg_novel 14.512008
loss 0.4913047
STEP 413 ================================
prereg loss 0.034309436 regularization 440.5132 reg_novel 14.582965
loss 0.48940563
STEP 414 ================================
prereg loss 0.033963874 regularization 438.84692 reg_novel 14.658714
loss 0.48746955
STEP 415 ================================
prereg loss 0.03407725 regularization 437.22287 reg_novel 14.729177
loss 0.48602933
STEP 416 ================================
prereg loss 0.03363705 regularization 435.61407 reg_novel 14.79371
loss 0.48404488
STEP 417 ================================
prereg loss 0.033314973 regularization 434.00998 reg_novel 14.854069
loss 0.48217905
STEP 418 ================================
prereg loss 0.033243153 regularization 432.41003 reg_novel 14.924116
loss 0.48057732
STEP 419 ================================
prereg loss 0.033090442 regularization 430.8205 reg_novel 15.002657
loss 0.47891364
STEP 420 ================================
prereg loss 0.032709442 regularization 429.25238 reg_novel 15.079197
loss 0.47704104
STEP 421 ================================
prereg loss 0.03248231 regularization 427.7081 reg_novel 15.155678
loss 0.4753461
STEP 422 ================================
prereg loss 0.032121383 regularization 426.162 reg_novel 15.230044
loss 0.47351345
STEP 423 ================================
prereg loss 0.031593468 regularization 424.6317 reg_novel 15.303006
loss 0.47152823
STEP 424 ================================
prereg loss 0.031421892 regularization 423.12033 reg_novel 15.3691845
loss 0.46991143
STEP 425 ================================
prereg loss 0.031177433 regularization 421.62427 reg_novel 15.434423
loss 0.46823612
STEP 426 ================================
prereg loss 0.030872991 regularization 420.12723 reg_novel 15.504457
loss 0.4665047
STEP 427 ================================
prereg loss 0.030619988 regularization 418.62558 reg_novel 15.570608
loss 0.46481618
STEP 428 ================================
prereg loss 0.030454874 regularization 417.14285 reg_novel 15.639612
loss 0.46323735
STEP 429 ================================
prereg loss 0.030201891 regularization 415.65903 reg_novel 15.711011
loss 0.46157193
STEP 430 ================================
prereg loss 0.029979309 regularization 414.18054 reg_novel 15.781806
loss 0.4599417
STEP 431 ================================
prereg loss 0.02976696 regularization 412.7363 reg_novel 15.849572
loss 0.45835283
STEP 432 ================================
prereg loss 0.029505426 regularization 411.29663 reg_novel 15.910243
loss 0.4567123
STEP 433 ================================
prereg loss 0.029303605 regularization 409.89267 reg_novel 15.9619055
loss 0.4551582
STEP 434 ================================
prereg loss 0.02907375 regularization 408.49207 reg_novel 16.018072
loss 0.4535839
STEP 435 ================================
prereg loss 0.028844291 regularization 407.09863 reg_novel 16.07714
loss 0.45202008
STEP 436 ================================
prereg loss 0.028654972 regularization 405.70712 reg_novel 16.138432
loss 0.45050055
STEP 437 ================================
prereg loss 0.02844628 regularization 404.31918 reg_novel 16.20228
loss 0.44896775
STEP 438 ================================
prereg loss 0.028276555 regularization 402.93518 reg_novel 16.272379
loss 0.44748414
STEP 439 ================================
prereg loss 0.028112167 regularization 401.53986 reg_novel 16.340494
loss 0.44599253
STEP 440 ================================
prereg loss 0.02790187 regularization 400.15228 reg_novel 16.409126
loss 0.44446328
STEP 441 ================================
prereg loss 0.0276978 regularization 398.78754 reg_novel 16.483534
loss 0.4429689
STEP 442 ================================
prereg loss 0.027538871 regularization 397.42297 reg_novel 16.557102
loss 0.44151896
STEP 443 ================================
prereg loss 0.02737549 regularization 396.07846 reg_novel 16.622517
loss 0.44007647
STEP 444 ================================
prereg loss 0.027185857 regularization 394.74307 reg_novel 16.687378
loss 0.43861634
STEP 445 ================================
prereg loss 0.02697637 regularization 393.43045 reg_novel 16.74684
loss 0.4371537
STEP 446 ================================
prereg loss 0.02677827 regularization 392.12247 reg_novel 16.8015
loss 0.43570226
STEP 447 ================================
prereg loss 0.026559493 regularization 390.83212 reg_novel 16.855366
loss 0.43424702
STEP 448 ================================
prereg loss 0.026382327 regularization 389.5453 reg_novel 16.905615
loss 0.43283325
STEP 449 ================================
prereg loss 0.026220733 regularization 388.26486 reg_novel 16.965242
loss 0.43145087
STEP 450 ================================
prereg loss 0.02607309 regularization 386.99268 reg_novel 17.024704
loss 0.4300905
STEP 451 ================================
prereg loss 0.025924811 regularization 385.706 reg_novel 17.095827
loss 0.42872664
STEP 452 ================================
prereg loss 0.025768368 regularization 384.41058 reg_novel 17.167366
loss 0.42734632
STEP 453 ================================
prereg loss 0.025592301 regularization 383.15012 reg_novel 17.239536
loss 0.42598197
STEP 454 ================================
prereg loss 0.025441144 regularization 381.90048 reg_novel 17.314962
loss 0.4246566
STEP 455 ================================
prereg loss 0.025322204 regularization 380.6635 reg_novel 17.3812
loss 0.42336693
STEP 456 ================================
prereg loss 0.02526392 regularization 379.43985 reg_novel 17.432621
loss 0.42213643
STEP 457 ================================
prereg loss 0.025133481 regularization 378.2249 reg_novel 17.486069
loss 0.4208445
STEP 458 ================================
prereg loss 0.024929302 regularization 377.02155 reg_novel 17.530403
loss 0.41948128
STEP 459 ================================
prereg loss 0.024698721 regularization 375.84705 reg_novel 17.56949
loss 0.4181153
STEP 460 ================================
prereg loss 0.024486838 regularization 374.67816 reg_novel 17.616142
loss 0.41678116
STEP 461 ================================
prereg loss 0.024324894 regularization 373.50317 reg_novel 17.663565
loss 0.41549164
STEP 462 ================================
prereg loss 0.024268212 regularization 372.33218 reg_novel 17.717495
loss 0.4143179
STEP 463 ================================
prereg loss 0.024073996 regularization 371.15344 reg_novel 17.782324
loss 0.4130098
STEP 464 ================================
prereg loss 0.023913253 regularization 369.96216 reg_novel 17.852125
loss 0.41172758
STEP 465 ================================
prereg loss 0.023801234 regularization 368.80292 reg_novel 17.920902
loss 0.41052508
STEP 466 ================================
prereg loss 0.02360856 regularization 367.6504 reg_novel 17.988894
loss 0.40924788
STEP 467 ================================
prereg loss 0.023480693 regularization 366.5039 reg_novel 18.053501
loss 0.4080381
STEP 468 ================================
prereg loss 0.023356348 regularization 365.36423 reg_novel 18.110945
loss 0.40683153
STEP 469 ================================
prereg loss 0.023214195 regularization 364.2366 reg_novel 18.164621
loss 0.40561545
STEP 470 ================================
prereg loss 0.023065718 regularization 363.11462 reg_novel 18.223125
loss 0.4044035
STEP 471 ================================
prereg loss 0.022938058 regularization 362.00085 reg_novel 18.275814
loss 0.40321475
STEP 472 ================================
prereg loss 0.022783268 regularization 360.91513 reg_novel 18.315754
loss 0.4020142
STEP 473 ================================
prereg loss 0.022632813 regularization 359.83307 reg_novel 18.365288
loss 0.4008312
STEP 474 ================================
prereg loss 0.02250169 regularization 358.75067 reg_novel 18.413473
loss 0.39966586
STEP 475 ================================
prereg loss 0.022368101 regularization 357.6707 reg_novel 18.459167
loss 0.39849797
STEP 476 ================================
prereg loss 0.022221291 regularization 356.59616 reg_novel 18.50967
loss 0.39732715
STEP 477 ================================
prereg loss 0.022046769 regularization 355.53485 reg_novel 18.563112
loss 0.39614478
STEP 478 ================================
prereg loss 0.021901052 regularization 354.46674 reg_novel 18.613678
loss 0.39498147
STEP 479 ================================
prereg loss 0.021813719 regularization 353.40186 reg_novel 18.67048
loss 0.39388606
STEP 480 ================================
prereg loss 0.021629477 regularization 352.35095 reg_novel 18.7257
loss 0.39270616
STEP 481 ================================
prereg loss 0.021505376 regularization 351.31668 reg_novel 18.775513
loss 0.3915976
STEP 482 ================================
prereg loss 0.0213567 regularization 350.2861 reg_novel 18.826227
loss 0.39046904
STEP 483 ================================
prereg loss 0.021228848 regularization 349.27792 reg_novel 18.87369
loss 0.38938048
STEP 484 ================================
prereg loss 0.021126822 regularization 348.26813 reg_novel 18.925613
loss 0.38832057
STEP 485 ================================
prereg loss 0.021000572 regularization 347.2437 reg_novel 18.981972
loss 0.38722625
STEP 486 ================================
prereg loss 0.020864582 regularization 346.23038 reg_novel 19.046349
loss 0.3861413
STEP 487 ================================
prereg loss 0.020754155 regularization 345.2123 reg_novel 19.11044
loss 0.38507694
STEP 488 ================================
prereg loss 0.020645788 regularization 344.21075 reg_novel 19.176205
loss 0.3840328
STEP 489 ================================
prereg loss 0.020503808 regularization 343.2121 reg_novel 19.239077
loss 0.382955
STEP 490 ================================
prereg loss 0.020457627 regularization 342.2213 reg_novel 19.297283
loss 0.38197622
STEP 491 ================================
prereg loss 0.020248024 regularization 341.24405 reg_novel 19.362053
loss 0.38085416
STEP 492 ================================
prereg loss 0.020128602 regularization 340.27985 reg_novel 19.422638
loss 0.37983114
STEP 493 ================================
prereg loss 0.020027896 regularization 339.31656 reg_novel 19.46983
loss 0.3788143
STEP 494 ================================
prereg loss 0.019941809 regularization 338.37146 reg_novel 19.516584
loss 0.37782988
STEP 495 ================================
prereg loss 0.01981903 regularization 337.43246 reg_novel 19.562445
loss 0.37681395
STEP 496 ================================
prereg loss 0.019659642 regularization 336.5072 reg_novel 19.604727
loss 0.37577158
STEP 497 ================================
prereg loss 0.019519357 regularization 335.5969 reg_novel 19.647955
loss 0.37476423
STEP 498 ================================
prereg loss 0.019393714 regularization 334.68723 reg_novel 19.682064
loss 0.373763
STEP 499 ================================
prereg loss 0.019277139 regularization 333.80035 reg_novel 19.71513
loss 0.37279263
STEP 500 ================================
prereg loss 0.019155677 regularization 332.9237 reg_novel 19.744024
loss 0.37182343
2022-06-06T09:28:14.009

julia> close(io)

julia> a_532 = deepcopy(trainable["network_matrix"])
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("dict"=>-3.13597f-5, "true"…
  "norm-5"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000606863), "norm-5"=>Dict("dict"=>0.000248746, "true"=…
  "accum-4"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000220673), "norm-5"=>Dict("dict"=>0.000552185, "true"=…
  "dot-2"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000386233), "norm-5"=>Dict("dict"=>-7.2999f-5, "true"=>…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("dict"=>0.00107631, "true"=>-0.…
  "compare-5" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000149236), "norm-5"=>Dict("dict"=>0.00023654, "true"=>…
  "accum-3"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00133074), "norm-5"=>Dict("dict"=>0.000994761, "true"=>…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "true"=>0.000…
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00110128), "norm-5"=>Dict("dict"=>0.000357958, "true"=>…
  "compare-4" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0338154), "norm-5"=>Dict("dict"=>0.00148583, "true"=>0.…
  "compare-2" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00046), "norm-5"=>Dict("dict"=>0.000199041, "true"=>0.0…
  "dot-1"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>7.75446f-5), "norm-5"=>Dict("dict"=>0.000589918, "true"=>…
  "dot-3"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>1.26586f-5), "norm-5"=>Dict("dict"=>0.00031829, "true"=>0…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dict"=>-6.08445f-5, "true"=>-…
  "compare-3" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000523074), "norm-5"=>Dict("dict"=>0.000427456, "true"=…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "true"=>-0…
  "accum-2"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>2.3784f-5), "norm-5"=>Dict("dict"=>8.75653f-5, "true"=>-2…
  "compare-1" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00017185), "norm-5"=>Dict("dict"=>0.000180495, "true"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("dict"=>0.000267972, "true"=>3…
  "dot-5"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.000345763), "norm-5"=>Dict("dict"=>0.00158707, "true"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("dict"=>-0.000140536, "true"=>0…

julia> count_neg_interval(a_532, -0.8f0, 0.8f0)
2

julia> count_neg_interval(a_532, -0.7f0, 0.7f0)
6

julia> count_neg_interval(a_532, -0.6f0, 0.6f0)
14

julia> count_neg_interval(a_532, -0.5f0, 0.5f0)
31

julia> count_neg_interval(a_532, -0.4f0, 0.4f0)
72

julia> count_neg_interval(a_532, -0.3f0, 0.3f0)
187

julia> count_neg_interval(a_532, -0.2f0, 0.2f0)
514

julia> count_neg_interval(a_532, -0.1f0, 0.1f0)
1223

julia> count_neg_interval(a_532, -0.01f0, 0.01f0)
2588

julia> count_neg_interval(a_532, -0.001f0, 0.001f0)
4589

julia> count_neg_interval(a_532, -0.0001f0, 0.0001f0)
15559
```

Let's serialize the state in case we want to continue this line,
and let's see how this generalizes beyond (not much):

```
julia> serialize("532-steps-matrix.ser", a_532)

julia> serialize("532-steps-treeADAM.ser", opt)

julia> function loss_k(dmm_lite::DMM_Lite_, k_steps)
                  l = 0.0f0
                  for i in 1:k_steps
                          two_stroke_cycle!(dmm_lite)
                      two_stroke_cycle!(handcrafted)
                      target_1 = get_N(handcrafted["neurons"]["output"].input_dict["dict-1"], ":number")
                      target_2 = get_N(handcrafted["neurons"]["output"].input_dict["dict-2"], ":number")
                      l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-1"], ":number") - target_1)
                      l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-2"], ":number") - target_2)
                  end
                  regularization = 0.0f0
                  for i in keys(dmm_lite["network_matrix"])
                      if i != "timer" && i != "input"
                          for j in keys(dmm_lite["network_matrix"][i])
                              for m in keys(dmm_lite["network_matrix"][i][j])
                                  for n in keys(dmm_lite["network_matrix"][i][j][m])
                                      regularization += abs(dmm_lite["network_matrix"][i][j][m][n]) +
                                                        10.0f0 * square(dmm_lite["network_matrix"][i][j][m][n])
                  end end end end end
                  println("prereg loss ", l, " regularization ", regularization)
                  l += 0.001f0 * regularization
                  println("loss ", l)
                  l
              end
loss_k (generic function with 1 method)

julia> function input_dummy(x::Dict{String, Dict{String, Float32}})
           t::Float32 = get_N(get_D(x, "timer"), ":number")
           println("(driving input) timer: ", t)
           t = max(t, 0)
           s::String = "test string."
           d::Dict{String, Float32} = Dict{String, Float32}()
           if t%10 == 0
               i = min(round(Int, t÷10) + 1, lastindex(s))
               Zygote.@ignore d[SubString(s, i, i)] = one_value
           end
           Dict{String, Dict{String, Float32}}("char" => d)
       end
input_dummy (generic function with 1 method)

julia> function output_dummy(x::Dict{String, Dict{String, Float32}})
           d1 = get_D(x, "dict-1")
           d2 = get_D(x, "dict-2")
           n1 = get_N(d1, ":number")
           n2 = get_N(d2, ":number")
           println("(getting on output) left: ", n1, " right: ", n2)
           Dict{String, Dict{String, Float32}}()
       end
output_dummy (generic function with 1 method)

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0023547523 right: -0.0035653375
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.013933016 right: 0.001065556
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.0073140096 right: -0.002692571
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.004812545 right: 0.0016065624
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: -0.0033839084 right: 0.006341746
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.0040186066 right: 0.0075021572
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.0119924545 right: 0.0007108543
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: -0.0007235557 right: -0.020627433
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0049314555 right: -0.014546376
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.014756707 right: -0.004519049
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: 0.00060490146 right: -0.0109394975
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.004106693 right: 0.01444013
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.0102882 right: 0.0025288984
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.0065983906 right: -0.004046656
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.014273774 right: -0.020314993
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.0043557044 right: 0.011949873
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.016356021 right: 0.007991366
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.014053417 right: -0.0034093726
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: 0.0016101254 right: -0.005057825
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.00997846 right: -0.013803562
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: 0.0017756261 right: 0.0016488284
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.0083333505 right: 0.0010057595
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.0015799934 right: 0.024445947
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.012374217 right: 0.012805346
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.0017918106 right: 0.0061049704
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.0011648107 right: 0.003087666
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.008426305 right: -0.0077740783
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.0041429643 right: -0.0068685506
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.022050854 right: -0.0051865466
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.0031171693 right: -0.006166309
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: 0.001095593 right: 0.029823171
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: -0.008725647 right: -0.00057294965
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.99836713 right: 0.0033960268
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 0.019027296 regularization 1039.3572
loss 1.0583845
1.0583845f0

julia> reset_dicts!()

julia> loss_k(trainable, 40)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0023547523 right: -0.0035653375
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.013933016 right: 0.001065556
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.0073140096 right: -0.002692571
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.004812545 right: 0.0016065624
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: -0.0033839084 right: 0.006341746
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.0040186066 right: 0.0075021572
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.0119924545 right: 0.0007108543
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: -0.0007235557 right: -0.020627433
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0049314555 right: -0.014546376
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.014756707 right: -0.004519049
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: 0.00060490146 right: -0.0109394975
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.004106693 right: 0.01444013
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.0102882 right: 0.0025288984
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.0065983906 right: -0.004046656
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.014273774 right: -0.020314993
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.0043557044 right: 0.011949873
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.016356021 right: 0.007991366
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.014053417 right: -0.0034093726
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: 0.0016101254 right: -0.005057825
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.00997846 right: -0.013803562
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: 0.0017756261 right: 0.0016488284
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.0083333505 right: 0.0010057595
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.0015799934 right: 0.024445947
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.012374217 right: 0.012805346
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.0017918106 right: 0.0061049704
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.0011648107 right: 0.003087666
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.008426305 right: -0.0077740783
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.0041429643 right: -0.0068685506
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.022050854 right: -0.0051865466
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.0031171693 right: -0.006166309
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: 0.001095593 right: 0.029823171
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: -0.008725647 right: -0.00057294965
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.99836713 right: 0.0033960268
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
(getting on output) left: -0.07885009 right: 0.0118732015
(driving input) timer: 35.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 35.0
(getting on output) left: 0.032482415 right: 0.19054505
(driving input) timer: 36.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 36.0
(getting on output) left: 0.7160764 right: 0.09423425
(driving input) timer: 37.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 37.0
(getting on output) left: 0.31137174 right: -0.16997111
(driving input) timer: 38.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 38.0
(getting on output) left: 0.37271163 right: 0.08357367
(driving input) timer: 39.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 39.0
prereg loss 3.1485503 regularization 1039.3572
loss 4.187907
4.187907f0
```

Next thing, let's see how robust the `loss_k(..., 35)` is to
sparsification.

```
julia> sparse = sparsecopy(a_532, 0.001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.00221501), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "compare-2"=…
  "dot-2"     => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>0.00102535), "compare-5"=>Dict("norm"=>0.189665), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("dict"=>0.00107631, "norm"=>0.0…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00133074), "norm-5"=>Dict("false"=>0.00114423, "dict-2"…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00110128), "compare-1"=>Dict("dict-1"=>0.39461), "dot-4…
  "compare-4" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0338154), "norm-5"=>Dict("dict"=>0.00148583, "dot"=>0.0…
  "compare-2" => Dict("dict"=>Dict("norm-5"=>Dict("false"=>-0.0758803), "dot-2"=>Dict("dict"=>0.0400329), "norm-1"=>Dic…
  "dot-1"     => Dict("dict"=>Dict("accum-5"=>Dict("true"=>0.00872398), "accum-1"=>Dict("norm"=>0.0704986), "compare-4"…
  "dot-3"     => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.00364167, "norm"=>0.00115309), "accum-4"=>Dict("dict"=>0.…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("norm-5"=>Dict("true"=>0.00108923), "accum-4"=>Dict("false"=>0.0011344), "dot-2"=>Di…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "true"=>-0…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("accum-2"=>Dict("norm"=>0.00302132), "accum-1"=>Dict("dot"=>0.0011829, "dict-2"=>0.0…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("norm-5"=>Dict("dict"=>0.00158707, "true"=>0.00313214, "dot"=>0.00118942, "dict-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
4589

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.00221501), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "compare-2"=…
  "dot-2"     => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>0.00102535), "compare-5"=>Dict("norm"=>0.189665), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("dict"=>0.00107631, "norm"=>0.0…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00133074), "norm-5"=>Dict("false"=>0.00114423, "dict-2"…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.00110128), "compare-1"=>Dict("dict-1"=>0.39461), "dot-4…
  "compare-4" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0338154), "norm-5"=>Dict("dict"=>0.00148583, "dot"=>0.0…
  "compare-2" => Dict("dict"=>Dict("norm-5"=>Dict("false"=>-0.0758803), "dot-2"=>Dict("dict"=>0.0400329), "norm-1"=>Dic…
  "dot-1"     => Dict("dict"=>Dict("accum-5"=>Dict("true"=>0.00872398), "accum-1"=>Dict("norm"=>0.0704986), "compare-4"…
  "dot-3"     => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.00364167, "norm"=>0.00115309), "accum-4"=>Dict("dict"=>0.…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("norm-5"=>Dict("true"=>0.00108923), "accum-4"=>Dict("false"=>0.0011344), "dot-2"=>Di…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "true"=>-0…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("accum-2"=>Dict("norm"=>0.00302132), "accum-1"=>Dict("dot"=>0.0011829, "dict-2"=>0.0…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("norm-5"=>Dict("dict"=>0.00158707, "true"=>0.00313214, "dot"=>0.00118942, "dict-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.002373822 right: -0.003660243
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.014151368 right: 0.0010185726
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.007722049 right: -0.0032244269
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.0051751714 right: 0.0010419637
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: -0.003255562 right: 0.0059377374
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.004056953 right: 0.00738059
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.011806903 right: 0.0004539676
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: -0.00056600105 right: -0.021071833
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0048089605 right: -0.014666554
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.015032924 right: -0.0047098696
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: 0.0015747622 right: -0.011816595
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.0045903586 right: 0.013938053
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.009007315 right: 0.0022214055
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.0071526673 right: -0.0043425057
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.014413421 right: -0.020979354
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.0044655036 right: 0.011684783
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.016047522 right: 0.007718878
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.014071625 right: -0.0036497358
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: 0.0017938279 right: -0.005243605
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.010472456 right: -0.014528418
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: 0.0019288696 right: 0.00094915554
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.0076932274 right: 0.00063233636
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.0019419934 right: 0.024002358
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.012099417 right: 0.0121611785
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.002228357 right: 0.0055974796
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.00090327393 right: 0.0024432568
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.008398242 right: -0.0077555226
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.004408719 right: -0.007207427
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.021459084 right: -0.0056738257
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.0029699272 right: -0.006487783
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: 0.0012554489 right: 0.029474054
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: -0.008622602 right: -0.0011736453
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.9938187 right: 0.0030936003
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 0.01904542 regularization 1034.8179
loss 1.0538634
1.0538634f0

julia> sparse = sparsecopy(a_532, 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "dot-2"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-3"=>Dict("false"=>0.0390518), "compare-5"=>Dict("norm"=>0.189665), "accum-2…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0121724, "dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.1…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "dot-4"=>Dict("norm"=>0.0365663), "dot-2"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("dot"=>0.0443886, "norm"=>0.0548995), "const_1"=>Dict("const_1"=>0…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "compare-1"…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "accum-3…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348), "dot-4"=>Dict("dict"=>0.02573…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "dot"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "dot-2"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-3"=>Dict("false"=>0.0390518), "compare-5"=>Dict("norm"=>0.189665), "accum-2…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0121724, "dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.1…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "dot-4"=>Dict("norm"=>0.0365663), "dot-2"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("dot"=>0.0443886, "norm"=>0.0548995), "const_1"=>Dict("const_1"=>0…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "compare-1"…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "accum-3…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348), "dot-4"=>Dict("dict"=>0.02573…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "dot"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0025854334 right: -0.0040870495
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.014784977 right: 0.0015749075
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.004937441 right: -0.002119245
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.0023581143 right: 0.005331792
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: -0.0023302315 right: 0.009084168
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.0038969554 right: 0.010767791
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.011909682 right: 0.0021034535
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: -0.0020710174 right: -0.016364802
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0061527025 right: -0.012204161
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.014810285 right: -0.0035186745
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: -0.0004776083 right: -0.005676478
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.0066056345 right: 0.017790465
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.016846301 right: 0.00374024
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.002660891 right: -0.0015560742
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.006383029 right: -0.01750104
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.0046238173 right: 0.012805872
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.017575987 right: 0.008376522
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.0150852725 right: 0.001131909
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: -0.0005294066 right: -0.0026244596
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.004866928 right: -0.0076985657
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: -0.004813928 right: 0.0048491172
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.017990459 right: 0.0034891851
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: -0.0007797638 right: 0.024875298
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.0180104 right: 0.019747376
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.0016971752 right: 0.010639632
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: -0.0070472416 right: 0.0053745536
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.007805234 right: -0.0077077346
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.0028446224 right: -0.012864886
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.021159187 right: -0.0045717284
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.011453448 right: -0.004547801
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: -0.0027418286 right: 0.031400964
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: -0.011587277 right: 0.010218881
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.9585904 right: 0.00626212
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 0.021726422 regularization 1029.6827
loss 1.0514091
1.0514091f0

julia> count(sparse)
2588

julia> sparse = sparsecopy(a_532, 0.1f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("norm-5"=>Dict("norm"=>0.262948), "accum-4"=>Dict("dict"=>-0.170075), "dot-2"=>Dic…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.221367), "accum-2"=>Dict("dict"=>0.268936), "accum-1"=>Di…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-2"=>Dict("false"=>0.250145)), "true"=>Dict…
  "dot-2"     => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.189665), "accum-2"=>Dict("dict-2"=>0.104874), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "accum-4"=>Dict("dict"=>-0.223017), "dot-2"=>D…
  "compare-5" => Dict("dict"=>Dict("accum-2"=>Dict("dict-1"=>0.244779), "norm-2"=>Dict("true"=>0.146654, "false"=>0.120…
  "accum-3"   => Dict("dict"=>Dict("accum-3"=>Dict("norm"=>0.104668), "norm-4"=>Dict("dict"=>0.145186)), "true"=>Dict("…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "norm-1"=>Dict("norm"=>0.117353)), "true"=>Dic…
  "compare-4" => Dict("dict"=>Dict("dot-2"=>Dict("norm"=>0.139621)), "true"=>Dict("accum-4"=>Dict("true"=>-0.138839, "n…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict-1"=>0.321897)), "true"=>Dict("compare-5"=>Dict("false"=>0.154482…
  "dot-1"     => Dict("dict"=>Dict("dot-2"=>Dict("true"=>0.130827)), "true"=>Dict("accum-4"=>Dict("true"=>-0.117763), "…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "dot-1"=>Dict("dot"=>0.136361)), "true"=>Dict("…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("norm"=>0.277711), "accum-4"=>…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348)), "true"=>Dict("accum-4"=>Dict…
  "accum-5"   => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.146956, "dot"=>0.353909, "false"=>0.170694), "dot-2"=>Dic…
  "accum-2"   => Dict("dict"=>Dict("compare-3"=>Dict("norm"=>-0.118362, "dict-1"=>0.123013), "compare-1"=>Dict("dict-1"…
  "compare-1" => Dict("dict"=>Dict("accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>Dict("dict"=>0.228134)), "true"=>Dict(…
  "dot-4"     => Dict("dict"=>Dict("compare-3"=>Dict("dict-2"=>0.141754), "accum-2"=>Dict("false"=>0.441112), "accum-3"…
  "dot-5"     => Dict("true"=>Dict("accum-5"=>Dict("dot"=>-0.111619), "accum-3"=>Dict("dot"=>0.136276), "accum-4"=>Dict…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-1"=>Dict("norm"=>-0.216055), "compare-5"…

julia> count(sparse)
1223

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("norm-5"=>Dict("norm"=>0.262948), "accum-4"=>Dict("dict"=>-0.170075), "dot-2"=>Dic…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.221367), "accum-2"=>Dict("dict"=>0.268936), "accum-1"=>Di…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-2"=>Dict("false"=>0.250145)), "true"=>Dict…
  "dot-2"     => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.189665), "accum-2"=>Dict("dict-2"=>0.104874), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "accum-4"=>Dict("dict"=>-0.223017), "dot-2"=>D…
  "compare-5" => Dict("dict"=>Dict("accum-2"=>Dict("dict-1"=>0.244779), "norm-2"=>Dict("true"=>0.146654, "false"=>0.120…
  "accum-3"   => Dict("dict"=>Dict("accum-3"=>Dict("norm"=>0.104668), "norm-4"=>Dict("dict"=>0.145186)), "true"=>Dict("…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "norm-1"=>Dict("norm"=>0.117353)), "true"=>Dic…
  "compare-4" => Dict("dict"=>Dict("dot-2"=>Dict("norm"=>0.139621)), "true"=>Dict("accum-4"=>Dict("true"=>-0.138839, "n…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict-1"=>0.321897)), "true"=>Dict("compare-5"=>Dict("false"=>0.154482…
  "dot-1"     => Dict("dict"=>Dict("dot-2"=>Dict("true"=>0.130827)), "true"=>Dict("accum-4"=>Dict("true"=>-0.117763), "…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "dot-1"=>Dict("dot"=>0.136361)), "true"=>Dict("…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("norm"=>0.277711), "accum-4"=>…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348)), "true"=>Dict("accum-4"=>Dict…
  "accum-5"   => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.146956, "dot"=>0.353909, "false"=>0.170694), "dot-2"=>Dic…
  "accum-2"   => Dict("dict"=>Dict("compare-3"=>Dict("norm"=>-0.118362, "dict-1"=>0.123013), "compare-1"=>Dict("dict-1"…
  "compare-1" => Dict("dict"=>Dict("accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>Dict("dict"=>0.228134)), "true"=>Dict(…
  "dot-4"     => Dict("dict"=>Dict("compare-3"=>Dict("dict-2"=>0.141754), "accum-2"=>Dict("false"=>0.441112), "accum-3"…
  "dot-5"     => Dict("true"=>Dict("accum-5"=>Dict("dot"=>-0.111619), "accum-3"=>Dict("dot"=>0.136276), "accum-4"=>Dict…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-1"=>Dict("norm"=>-0.216055), "compare-5"…

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.141516 right: 0.07544956
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.06880695 right: 0.094836846
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: 0.11875404 right: 0.065184414
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.012895968 right: 0.16011678
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: 0.1059659 right: 0.12915418
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.10962721 right: 0.06706034
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: 0.010715719 right: 0.15060903
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: 0.09344293 right: 0.1087348
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.011091409 right: 0.09048484
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.13815592 right: 0.13035345
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: 0.026986573 right: 0.12203619
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: 0.0308281 right: 0.087274164
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.33058387 right: 0.1631203
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.05932286 right: 0.13233848
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.06847666 right: 0.10426043
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.1643493 right: 0.17550343
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: 0.13664335 right: 0.10745197
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: 0.08736404 right: 0.11058037
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: 0.110713795 right: 0.059933838
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.004830962 right: 0.1864679
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: 0.0985447 right: 0.13117893
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: -0.029888768 right: 0.07182501
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.5635829 right: 0.1875678
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: 0.09404168 right: 0.10198064
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.040486608 right: 0.10762982
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.5109365 right: 0.2690079
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: 0.2959274 right: 0.083456434
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.31818315 right: 0.12426644
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: 0.16417158 right: -0.016707364
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.34123302 right: 0.21077305
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: 0.22738023 right: 0.14902923
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: 0.019628093 right: 0.09468837
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.6617071 right: 0.14618415
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 1.9410367 regularization 920.47473
loss 2.8615115
2.8615115f0

julia> sparse = sparsecopy(a_532, 0.05f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("norm"=>0.262948), "accum-4…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-2"=>Dict("false"=>0.250145), "norm-1"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.189665), "accum-2"=>Dict("dict-2"=>0.104874), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("accum-2"=>Dict("dict-1"=>0.244779), "norm-2"=>Dict("true"=>0.146654, "false"=>0.120…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.104668), "norm-4"=>D…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "norm-1"=>Dict("norm"=>0.117353)), "true"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.0548995), "compare-2"=>Dict("dot"=>0.0784272), "dot-2"=>…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "dot-1"=>Di…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "dot-1"=…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("norm"=>0.277711), "accum-4"=>…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348)), "true"=>Dict("dot-5"=>Dict("…
  "accum-5"   => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.146956, "true"=>0.0826538, "dot"=>0.353909, "false"=>0.17…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>Dict("dict"=>0.228134), "dot-3"=>Dict(…
  "dot-4"     => Dict("dict"=>Dict("compare-3"=>Dict("dict-2"=>0.141754), "accum-2"=>Dict("false"=>0.441112), "compare-…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812)), "true"=>D…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "accum-4"=>Dict("dict"=>-0.0929324, "dict-2"=>…

julia>

julia> count(sparse)
1844

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("norm"=>0.262948), "accum-4…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-2"=>Dict("false"=>0.250145), "norm-1"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.189665), "accum-2"=>Dict("dict-2"=>0.104874), "accum-3"=…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("accum-2"=>Dict("dict-1"=>0.244779), "norm-2"=>Dict("true"=>0.146654, "false"=>0.120…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.104668), "norm-4"=>D…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "norm-1"=>Dict("norm"=>0.117353)), "true"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("norm"=>0.0548995), "compare-2"=>Dict("dot"=>0.0784272), "dot-2"=>…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "dot-1"=>Di…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "dot-1"=…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("norm"=>0.277711), "accum-4"=>…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348)), "true"=>Dict("dot-5"=>Dict("…
  "accum-5"   => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.146956, "true"=>0.0826538, "dot"=>0.353909, "false"=>0.17…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>Dict("dict"=>0.228134), "dot-3"=>Dict(…
  "dot-4"     => Dict("dict"=>Dict("compare-3"=>Dict("dict-2"=>0.141754), "accum-2"=>Dict("false"=>0.441112), "compare-…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812)), "true"=>D…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "accum-4"=>Dict("dict"=>-0.0929324, "dict-2"=>…

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.019692075 right: -0.0246117
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.03112723 right: 0.0051356144
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.0039826697 right: 0.0038726144
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.013648692 right: 0.009064555
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: 0.014920648 right: 0.005641144
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.0012444891 right: 0.0054450035
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.024803558 right: 0.027010765
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: 0.030099265 right: -0.0055728815
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0149314515 right: -0.024024144
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.08653532 right: -0.0029884428
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: -0.014190067 right: 0.014002621
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.009527808 right: 0.04101161
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.121474035 right: -0.010668155
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: -0.034456957 right: 0.0088666305
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: -0.009436674 right: 0.0264544
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.014255779 right: 0.004929496
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.0040508322 right: -0.0014396869
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.018206995 right: 0.019384125
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: 0.04819471 right: -0.013425883
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: -0.01908178 right: 0.0046448186
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: 0.0050906055 right: 0.044164434
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.045675386 right: -0.010115501
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.24515471 right: 0.024234325
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.05322724 right: 0.025965344
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.0149860885 right: 0.05470884
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.08490448 right: 0.032853235
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: 0.071456686 right: -0.03734459
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.092805296 right: -0.046970055
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.040063225 right: -0.02061266
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.0723934 right: 0.06258515
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: -0.013180431 right: 0.062654294
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: 0.004437441 right: 0.03311919
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.99317896 right: -0.052938327
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 0.16514926 regularization 1001.1163
loss 1.1662655
1.1662655f0
```

Let's settle on 0.01f0 threshold, like the last time.

```
julia> sparse = sparsecopy(a_532, 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "dot-2"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-3"=>Dict("false"=>0.0390518), "compare-5"=>Dict("norm"=>0.189665), "accum-2…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0121724, "dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.1…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "dot-4"=>Dict("norm"=>0.0365663), "dot-2"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("dot"=>0.0443886, "norm"=>0.0548995), "const_1"=>Dict("const_1"=>0…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "compare-1"…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "accum-3…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348), "dot-4"=>Dict("dict"=>0.02573…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "dot"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Dict("norm"=>0.0957387), "norm-1"=>Dic…
  "accum-4"   => Dict("dict"=>Dict("accum-2"=>Dict("dict-2"=>0.113267), "dot-4"=>Dict("false"=>0.0161288), "dot-2"=>Dic…
  "dot-2"     => Dict("dict"=>Dict("compare-3"=>Dict("false"=>0.0390518), "compare-5"=>Dict("norm"=>0.189665), "accum-2…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.188803), "norm-5"=>Dict("norm"=>0.0976651), "accum-4"=>…
  "compare-5" => Dict("dict"=>Dict("compare-5"=>Dict("dict"=>0.0436601), "accum-1"=>Dict("false"=>0.0312751), "accum-2"…
  "accum-3"   => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0121724, "dict-1"=>0.05483), "accum-3"=>Dict("norm"=>0.1…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.133503), "norm-5"=>Dict("dict"=>0.484826, "norm"=>-0.36…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.39461), "dot-4"=>Dict("norm"=>0.0365663), "dot-2"=>Dic…
  "compare-4" => Dict("dict"=>Dict("compare-5"=>Dict("dot"=>0.0443886, "norm"=>0.0548995), "const_1"=>Dict("const_1"=>0…
  "compare-2" => Dict("dict"=>Dict("accum-1"=>Dict("true"=>0.0843061), "norm-5"=>Dict("false"=>-0.0758803), "compare-1"…
  "dot-1"     => Dict("dict"=>Dict("accum-1"=>Dict("norm"=>0.0704986), "compare-2"=>Dict("dict"=>0.0818895, "dict-2"=>0…
  "dot-3"     => Dict("dict"=>Dict("compare-3"=>Dict("dict"=>0.123418), "compare-5"=>Dict("false"=>0.0920493), "accum-3…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.168965), "norm-5"=>Dict("dot"=>0.0157298, "norm"=>0.27…
  "compare-3" => Dict("dict"=>Dict("compare-1"=>Dict("dot"=>0.221851, "dict-2"=>0.10348), "dot-4"=>Dict("dict"=>0.02573…
  "accum-5"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0134213), "norm-5"=>Dict("dict"=>-0.038037, "dot"=>-0.…
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "dot-2"=>Dict("dot"=>0.0616065), "compare-5"=>Di…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-4"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> reset_dicts!()

julia> loss_k(trainable, 35)
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.07165938 right: -0.086523235
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0025854334 right: -0.0040870495
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.014784977 right: 0.0015749075
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.004937441 right: -0.002119245
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: 0.0023581143 right: 0.005331792
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: -0.0023302315 right: 0.009084168
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: 0.0038969554 right: 0.010767791
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: -0.011909682 right: 0.0021034535
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: -0.0020710174 right: -0.016364802
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.0061527025 right: -0.012204161
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: 0.014810285 right: -0.0035186745
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: -0.0004776083 right: -0.005676478
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.0066056345 right: 0.017790465
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: 0.016846301 right: 0.00374024
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: 0.002660891 right: -0.0015560742
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: 0.006383029 right: -0.01750104
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: 0.0046238173 right: 0.012805872
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.017575987 right: 0.008376522
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.0150852725 right: 0.001131909
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: -0.0005294066 right: -0.0026244596
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: 0.004866928 right: -0.0076985657
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: -0.004813928 right: 0.0048491172
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: 0.017990459 right: 0.0034891851
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: -0.0007797638 right: 0.024875298
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: -0.0180104 right: 0.019747376
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: 0.0016971752 right: 0.010639632
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: -0.0070472416 right: 0.0053745536
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.007805234 right: -0.0077077346
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: 0.0028446224 right: -0.012864886
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.021159187 right: -0.0045717284
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: 0.011453448 right: -0.004547801
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: -0.0027418286 right: 0.031400964
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: -0.011587277 right: 0.010218881
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.9585904 right: 0.00626212
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
prereg loss 0.021726422 regularization 1029.6827
loss 1.0514091
1.0514091f0

julia> count(sparse)
2588

julia> serialize("sparse_2588.ser", sparse)
```
