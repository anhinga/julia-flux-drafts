# Decided to sparsify even further post `sparse20-after-100-steps-matrix`

This is very optional (no strong reasons to rush in this direction), but let's try.

Looks interesting, but can't quite figure out yet how to make the training runs of interest converge

**(Note: training in this directory is accidently done on a different "training string", so the loss numbers are not directly comparable.**

**Eventually, we might want to re-run this experiment, it's currently an optional item on my todo list related to this project.)**

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.679991), "input"=>Dict("char"=>-0.503598)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.28953)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6847…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.08638)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.438594)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.689225)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.6794…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.250608), "compare-1-2"=>Dict("false"=>0.449961)), "d…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.761408)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.604257)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
19

julia> count_interval(trainable["network_matrix"], -0.3f0, 0.3f0)
2

julia> sparse = sparsecopy(trainable["network_matrix"], 0.3f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.679991), "input"=>Dict("char"=>-0.503598)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.28953)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6847…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.08638)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.438594)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.689225)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.6794…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.449961)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.761408)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.604257)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.679991), "input"=>Dict("char"=>-0.503598)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.28953)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.6847…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.08638)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.438594)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.689225)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.6794…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.449961)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.761408)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.604257)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
17

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-28T20:07:40.758
STEP 1 ================================
prereg loss 95.66268 reg_l1 12.198566 reg_l2 10.049428
loss 96.88254
STEP 2 ================================
prereg loss 92.51267 reg_l1 12.201568 reg_l2 10.057621
loss 93.73283
STEP 3 ================================
prereg loss 89.40215 reg_l1 12.195962 reg_l2 10.050567
loss 90.62175
STEP 4 ================================
prereg loss 86.345665 reg_l1 12.189941 reg_l2 10.042774
loss 87.56466
STEP 5 ================================
prereg loss 83.345024 reg_l1 12.184435 reg_l2 10.035758
loss 84.56347
STEP 6 ================================
prereg loss 80.39939 reg_l1 12.180362 reg_l2 10.031005
loss 81.617424
STEP 7 ================================
prereg loss 77.507614 reg_l1 12.1779785 reg_l2 10.028987
loss 78.72541
STEP 8 ================================
prereg loss 74.67066 reg_l1 12.176862 reg_l2 10.029041
loss 75.88835
STEP 9 ================================
prereg loss 71.89084 reg_l1 12.176423 reg_l2 10.030202
loss 73.10848
STEP 10 ================================
prereg loss 69.16956 reg_l1 12.176105 reg_l2 10.031523
loss 70.38718
STEP 11 ================================
prereg loss 66.50694 reg_l1 12.175435 reg_l2 10.03221
loss 67.72449
STEP 12 ================================
prereg loss 63.90327 reg_l1 12.174163 reg_l2 10.031834
loss 65.12069
STEP 13 ================================
prereg loss 61.35827 reg_l1 12.172283 reg_l2 10.030398
loss 62.575497
STEP 14 ================================
prereg loss 58.87303 reg_l1 12.169947 reg_l2 10.028158
loss 60.090027
STEP 15 ================================
prereg loss 56.44834 reg_l1 12.167349 reg_l2 10.025472
loss 57.665077
STEP 16 ================================
prereg loss 54.084904 reg_l1 12.164732 reg_l2 10.022743
loss 55.301376
STEP 17 ================================
prereg loss 51.78282 reg_l1 12.16233 reg_l2 10.020388
loss 52.999054
STEP 18 ================================
prereg loss 49.54229 reg_l1 12.160336 reg_l2 10.018733
loss 50.758324
STEP 19 ================================
prereg loss 47.362923 reg_l1 12.158831 reg_l2 10.017929
loss 48.578804
STEP 20 ================================
prereg loss 45.244987 reg_l1 12.157771 reg_l2 10.0179
loss 46.460766
STEP 21 ================================
prereg loss 43.18862 reg_l1 12.157007 reg_l2 10.018395
loss 44.404324
STEP 22 ================================
prereg loss 41.194046 reg_l1 12.156331 reg_l2 10.019054
loss 42.40968
STEP 23 ================================
prereg loss 39.261086 reg_l1 12.155529 reg_l2 10.019522
loss 40.47664
STEP 24 ================================
prereg loss 37.389465 reg_l1 12.15444 reg_l2 10.019523
loss 38.604908
STEP 25 ================================
prereg loss 35.57868 reg_l1 12.1529875 reg_l2 10.018932
loss 36.793976
STEP 26 ================================
prereg loss 33.828247 reg_l1 12.151198 reg_l2 10.017792
loss 35.043365
STEP 27 ================================
prereg loss 32.138042 reg_l1 12.149174 reg_l2 10.016267
loss 33.35296
STEP 28 ================================
prereg loss 30.507555 reg_l1 12.1470585 reg_l2 10.014613
loss 31.722261
STEP 29 ================================
prereg loss 28.93623 reg_l1 12.145019 reg_l2 10.0131
loss 30.150732
STEP 30 ================================
prereg loss 27.423449 reg_l1 12.143188 reg_l2 10.011953
loss 28.637768
STEP 31 ================================
prereg loss 25.968237 reg_l1 12.141642 reg_l2 10.011298
loss 27.182402
STEP 32 ================================
prereg loss 24.569876 reg_l1 12.140374 reg_l2 10.011115
loss 25.783913
STEP 33 ================================
prereg loss 23.227655 reg_l1 12.139303 reg_l2 10.011265
loss 24.441586
STEP 34 ================================
prereg loss 21.94075 reg_l1 12.138305 reg_l2 10.011535
loss 23.154581
STEP 35 ================================
prereg loss 20.708336 reg_l1 12.137249 reg_l2 10.0117
loss 21.92206
STEP 36 ================================
prereg loss 19.529184 reg_l1 12.13604 reg_l2 10.011593
loss 20.742788
STEP 37 ================================
prereg loss 18.402313 reg_l1 12.134644 reg_l2 10.011149
loss 19.615778
STEP 38 ================================
prereg loss 17.326656 reg_l1 12.133087 reg_l2 10.010418
loss 18.539965
STEP 39 ================================
prereg loss 16.301092 reg_l1 12.131455 reg_l2 10.009539
loss 17.514238
STEP 40 ================================
prereg loss 15.324504 reg_l1 12.129855 reg_l2 10.008692
loss 16.537489
STEP 41 ================================
prereg loss 14.395652 reg_l1 12.128397 reg_l2 10.008055
loss 15.608492
STEP 42 ================================
prereg loss 13.513235 reg_l1 12.127147 reg_l2 10.007744
loss 14.72595
STEP 43 ================================
prereg loss 12.675973 reg_l1 12.126117 reg_l2 10.00778
loss 13.888584
STEP 44 ================================
prereg loss 11.882566 reg_l1 12.125265 reg_l2 10.008089
loss 13.095093
STEP 45 ================================
prereg loss 11.131788 reg_l1 12.124505 reg_l2 10.008525
loss 12.344238
STEP 46 ================================
prereg loss 10.422289 reg_l1 12.123743 reg_l2 10.008932
loss 11.634664
STEP 47 ================================
prereg loss 9.752706 reg_l1 12.122903 reg_l2 10.009179
loss 10.964996
STEP 48 ================================
prereg loss 9.121658 reg_l1 12.121948 reg_l2 10.009212
loss 10.333853
STEP 49 ================================
prereg loss 8.52777 reg_l1 12.120895 reg_l2 10.009053
loss 9.73986
STEP 50 ================================
prereg loss 7.969647 reg_l1 12.119799 reg_l2 10.0088
loss 9.181627
STEP 51 ================================
prereg loss 7.4459352 reg_l1 12.118733 reg_l2 10.0085745
loss 8.657808
STEP 52 ================================
prereg loss 6.9552393 reg_l1 12.117763 reg_l2 10.008488
loss 8.167015
STEP 53 ================================
prereg loss 6.496127 reg_l1 12.1169195 reg_l2 10.008605
loss 7.707819
STEP 54 ================================
prereg loss 6.067278 reg_l1 12.116204 reg_l2 10.008911
loss 7.2788982
STEP 55 ================================
prereg loss 5.6673107 reg_l1 12.115567 reg_l2 10.009333
loss 6.8788676
STEP 56 ================================
prereg loss 5.294912 reg_l1 12.114949 reg_l2 10.009768
loss 6.506407
STEP 57 ================================
prereg loss 4.9487715 reg_l1 12.1142845 reg_l2 10.010106
loss 6.1602
STEP 58 ================================
prereg loss 4.6275377 reg_l1 12.11354 reg_l2 10.010291
loss 5.838892
STEP 59 ================================
prereg loss 4.3299513 reg_l1 12.112715 reg_l2 10.010321
loss 5.5412226
STEP 60 ================================
prereg loss 4.054759 reg_l1 12.111839 reg_l2 10.010245
loss 5.265943
STEP 61 ================================
prereg loss 3.8007145 reg_l1 12.110966 reg_l2 10.010152
loss 5.0118113
STEP 62 ================================
prereg loss 3.566619 reg_l1 12.1101465 reg_l2 10.010128
loss 4.7776337
STEP 63 ================================
prereg loss 3.351308 reg_l1 12.109411 reg_l2 10.010222
loss 4.562249
STEP 64 ================================
prereg loss 3.153619 reg_l1 12.108764 reg_l2 10.010443
loss 4.3644953
STEP 65 ================================
prereg loss 2.972478 reg_l1 12.108178 reg_l2 10.010739
loss 4.1832957
STEP 66 ================================
prereg loss 2.8068273 reg_l1 12.107611 reg_l2 10.011044
loss 4.0175886
STEP 67 ================================
prereg loss 2.655631 reg_l1 12.107021 reg_l2 10.011288
loss 3.8663332
STEP 68 ================================
prereg loss 2.5179224 reg_l1 12.106391 reg_l2 10.01143
loss 3.7285614
STEP 69 ================================
prereg loss 2.3927474 reg_l1 12.105714 reg_l2 10.011472
loss 3.6033187
STEP 70 ================================
prereg loss 2.2792034 reg_l1 12.105022 reg_l2 10.011458
loss 3.4897056
STEP 71 ================================
prereg loss 2.176419 reg_l1 12.104347 reg_l2 10.011448
loss 3.3868537
STEP 72 ================================
prereg loss 2.0835855 reg_l1 12.103722 reg_l2 10.011493
loss 3.2939577
STEP 73 ================================
prereg loss 1.9998982 reg_l1 12.103158 reg_l2 10.011621
loss 3.2102141
STEP 74 ================================
prereg loss 1.9246197 reg_l1 12.1026535 reg_l2 10.011822
loss 3.134885
STEP 75 ================================
prereg loss 1.8570721 reg_l1 12.102181 reg_l2 10.012051
loss 3.0672903
STEP 76 ================================
prereg loss 1.7965827 reg_l1 12.101712 reg_l2 10.012261
loss 3.006754
STEP 77 ================================
prereg loss 1.7425439 reg_l1 12.101215 reg_l2 10.012408
loss 2.9526653
STEP 78 ================================
prereg loss 1.6943862 reg_l1 12.100686 reg_l2 10.012476
loss 2.9044547
STEP 79 ================================
prereg loss 1.6515367 reg_l1 12.100129 reg_l2 10.012482
loss 2.8615496
STEP 80 ================================
prereg loss 1.6135049 reg_l1 12.09957 reg_l2 10.012461
loss 2.823462
STEP 81 ================================
prereg loss 1.5798241 reg_l1 12.09903 reg_l2 10.012454
loss 2.7897272
STEP 82 ================================
prereg loss 1.5500493 reg_l1 12.098524 reg_l2 10.012486
loss 2.7599018
STEP 83 ================================
prereg loss 1.5237731 reg_l1 12.098052 reg_l2 10.012564
loss 2.7335782
STEP 84 ================================
prereg loss 1.5006335 reg_l1 12.097602 reg_l2 10.012658
loss 2.7103937
STEP 85 ================================
prereg loss 1.4802929 reg_l1 12.097154 reg_l2 10.012739
loss 2.6900082
STEP 86 ================================
prereg loss 1.4624403 reg_l1 12.096689 reg_l2 10.012779
loss 2.6721091
STEP 87 ================================
prereg loss 1.4467862 reg_l1 12.096199 reg_l2 10.012761
loss 2.656406
STEP 88 ================================
prereg loss 1.4330721 reg_l1 12.095689 reg_l2 10.012694
loss 2.642641
STEP 89 ================================
prereg loss 1.4210734 reg_l1 12.095172 reg_l2 10.012603
loss 2.6305907
STEP 90 ================================
prereg loss 1.4105674 reg_l1 12.094668 reg_l2 10.012517
loss 2.6200342
STEP 91 ================================
prereg loss 1.4013603 reg_l1 12.094186 reg_l2 10.012458
loss 2.6107788
STEP 92 ================================
prereg loss 1.393279 reg_l1 12.093732 reg_l2 10.012427
loss 2.602652
STEP 93 ================================
prereg loss 1.3861839 reg_l1 12.093293 reg_l2 10.012415
loss 2.5955133
STEP 94 ================================
prereg loss 1.3799291 reg_l1 12.092859 reg_l2 10.012395
loss 2.589215
STEP 95 ================================
prereg loss 1.3743954 reg_l1 12.092416 reg_l2 10.012345
loss 2.583637
STEP 96 ================================
prereg loss 1.3694838 reg_l1 12.091961 reg_l2 10.012263
loss 2.57868
STEP 97 ================================
prereg loss 1.3651087 reg_l1 12.091495 reg_l2 10.012151
loss 2.5742583
STEP 98 ================================
prereg loss 1.3611653 reg_l1 12.091027 reg_l2 10.01203
loss 2.5702682
STEP 99 ================================
prereg loss 1.3575981 reg_l1 12.09057 reg_l2 10.011913
loss 2.5666552
STEP 100 ================================
prereg loss 1.3543403 reg_l1 12.090134 reg_l2 10.011821
loss 2.5633535
2022-06-28T20:12:06.799

julia> open("sparse21-after-100-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse21-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse21-after-100-steps-opt.ser", opt)

julia> steps!(100)
2022-06-28T20:13:39.876
STEP 1 ================================
prereg loss 1.3513434 reg_l1 12.089712 reg_l2 10.011746
loss 2.5603147
STEP 2 ================================
prereg loss 1.3485581 reg_l1 12.089301 reg_l2 10.01168
loss 2.5574882
STEP 3 ================================
prereg loss 1.3459502 reg_l1 12.088889 reg_l2 10.011604
loss 2.5548391
STEP 4 ================================
prereg loss 1.3434967 reg_l1 12.0884695 reg_l2 10.011507
loss 2.5523436
STEP 5 ================================
prereg loss 1.3411548 reg_l1 12.08804 reg_l2 10.011389
loss 2.5499587
STEP 6 ================================
prereg loss 1.338913 reg_l1 12.087604 reg_l2 10.011253
loss 2.5476732
STEP 7 ================================
prereg loss 1.3367501 reg_l1 12.087169 reg_l2 10.011115
loss 2.545467
STEP 8 ================================
prereg loss 1.3346543 reg_l1 12.0867405 reg_l2 10.010983
loss 2.5433283
STEP 9 ================================
prereg loss 1.3326105 reg_l1 12.086325 reg_l2 10.010869
loss 2.541243
STEP 10 ================================
prereg loss 1.3306075 reg_l1 12.08592 reg_l2 10.010766
loss 2.5391996
STEP 11 ================================
prereg loss 1.3286364 reg_l1 12.085516 reg_l2 10.010662
loss 2.537188
STEP 12 ================================
prereg loss 1.3267006 reg_l1 12.08511 reg_l2 10.010552
loss 2.5352116
STEP 13 ================================
prereg loss 1.3247936 reg_l1 12.084697 reg_l2 10.010427
loss 2.5332632
STEP 14 ================================
prereg loss 1.3229105 reg_l1 12.084279 reg_l2 10.010289
loss 2.5313385
STEP 15 ================================
prereg loss 1.3210459 reg_l1 12.08386 reg_l2 10.010147
loss 2.5294318
STEP 16 ================================
prereg loss 1.3191997 reg_l1 12.083447 reg_l2 10.010012
loss 2.5275445
STEP 17 ================================
prereg loss 1.3173755 reg_l1 12.083043 reg_l2 10.00989
loss 2.5256798
STEP 18 ================================
prereg loss 1.3155696 reg_l1 12.082646 reg_l2 10.009779
loss 2.5238342
STEP 19 ================================
prereg loss 1.3137813 reg_l1 12.082258 reg_l2 10.009673
loss 2.522007
STEP 20 ================================
prereg loss 1.3120071 reg_l1 12.081867 reg_l2 10.009571
loss 2.5201938
STEP 21 ================================
prereg loss 1.3102535 reg_l1 12.081476 reg_l2 10.00946
loss 2.5184011
STEP 22 ================================
prereg loss 1.3085234 reg_l1 12.08108 reg_l2 10.009346
loss 2.5166316
STEP 23 ================================
prereg loss 1.3068008 reg_l1 12.080687 reg_l2 10.009225
loss 2.5148697
STEP 24 ================================
prereg loss 1.3050989 reg_l1 12.080294 reg_l2 10.009111
loss 2.5131283
STEP 25 ================================
prereg loss 1.3034152 reg_l1 12.079909 reg_l2 10.009008
loss 2.5114062
STEP 26 ================================
prereg loss 1.3017437 reg_l1 12.07953 reg_l2 10.008914
loss 2.5096967
STEP 27 ================================
prereg loss 1.3000888 reg_l1 12.079157 reg_l2 10.008828
loss 2.5080044
STEP 28 ================================
prereg loss 1.2984471 reg_l1 12.078783 reg_l2 10.008745
loss 2.5063255
STEP 29 ================================
prereg loss 1.2968233 reg_l1 12.078411 reg_l2 10.008659
loss 2.5046644
STEP 30 ================================
prereg loss 1.2952013 reg_l1 12.078033 reg_l2 10.008568
loss 2.5030046
STEP 31 ================================
prereg loss 1.2936041 reg_l1 12.077661 reg_l2 10.008479
loss 2.5013702
STEP 32 ================================
prereg loss 1.2920113 reg_l1 12.077285 reg_l2 10.008394
loss 2.4997396
STEP 33 ================================
prereg loss 1.2904267 reg_l1 12.076917 reg_l2 10.008317
loss 2.4981184
STEP 34 ================================
prereg loss 1.2888479 reg_l1 12.076551 reg_l2 10.008246
loss 2.496503
STEP 35 ================================
prereg loss 1.2872813 reg_l1 12.076192 reg_l2 10.0081835
loss 2.4949005
STEP 36 ================================
prereg loss 1.2857147 reg_l1 12.075834 reg_l2 10.008126
loss 2.493298
STEP 37 ================================
prereg loss 1.2841542 reg_l1 12.075474 reg_l2 10.008066
loss 2.4917016
STEP 38 ================================
prereg loss 1.2825983 reg_l1 12.075115 reg_l2 10.008008
loss 2.49011
STEP 39 ================================
prereg loss 1.2810438 reg_l1 12.074757 reg_l2 10.00795
loss 2.4885194
STEP 40 ================================
prereg loss 1.2794937 reg_l1 12.074402 reg_l2 10.007898
loss 2.486934
STEP 41 ================================
prereg loss 1.2779411 reg_l1 12.074048 reg_l2 10.007853
loss 2.4853458
STEP 42 ================================
prereg loss 1.2763792 reg_l1 12.073702 reg_l2 10.007817
loss 2.4837494
STEP 43 ================================
prereg loss 1.2748275 reg_l1 12.073359 reg_l2 10.007787
loss 2.4821634
STEP 44 ================================
prereg loss 1.2732687 reg_l1 12.073016 reg_l2 10.007758
loss 2.4805703
STEP 45 ================================
prereg loss 1.2717013 reg_l1 12.072676 reg_l2 10.007731
loss 2.478969
STEP 46 ================================
prereg loss 1.2701373 reg_l1 12.072333 reg_l2 10.0077095
loss 2.4773707
STEP 47 ================================
prereg loss 1.2685647 reg_l1 12.071996 reg_l2 10.007686
loss 2.4757643
STEP 48 ================================
prereg loss 1.2669883 reg_l1 12.071659 reg_l2 10.00767
loss 2.4741542
STEP 49 ================================
prereg loss 1.2654021 reg_l1 12.071326 reg_l2 10.007661
loss 2.4725347
STEP 50 ================================
prereg loss 1.2638086 reg_l1 12.070997 reg_l2 10.007658
loss 2.4709084
STEP 51 ================================
prereg loss 1.2622156 reg_l1 12.070671 reg_l2 10.007662
loss 2.4692826
STEP 52 ================================
prereg loss 1.260614 reg_l1 12.070346 reg_l2 10.007666
loss 2.4676485
STEP 53 ================================
prereg loss 1.2590058 reg_l1 12.070022 reg_l2 10.007671
loss 2.466008
STEP 54 ================================
prereg loss 1.2573891 reg_l1 12.069699 reg_l2 10.00768
loss 2.464359
STEP 55 ================================
prereg loss 1.2557703 reg_l1 12.069379 reg_l2 10.007693
loss 2.4627082
STEP 56 ================================
prereg loss 1.2541404 reg_l1 12.069061 reg_l2 10.00771
loss 2.4610467
STEP 57 ================================
prereg loss 1.2525055 reg_l1 12.068746 reg_l2 10.007734
loss 2.4593801
STEP 58 ================================
prereg loss 1.2508703 reg_l1 12.068434 reg_l2 10.007763
loss 2.4577136
STEP 59 ================================
prereg loss 1.2492251 reg_l1 12.068124 reg_l2 10.007795
loss 2.4560375
STEP 60 ================================
prereg loss 1.2475755 reg_l1 12.067817 reg_l2 10.007829
loss 2.4543571
STEP 61 ================================
prereg loss 1.2459246 reg_l1 12.067509 reg_l2 10.007868
loss 2.4526753
STEP 62 ================================
prereg loss 1.2442644 reg_l1 12.0672035 reg_l2 10.007908
loss 2.4509847
STEP 63 ================================
prereg loss 1.242599 reg_l1 12.066901 reg_l2 10.007955
loss 2.4492893
STEP 64 ================================
prereg loss 1.2409341 reg_l1 12.066603 reg_l2 10.008003
loss 2.4475944
STEP 65 ================================
prereg loss 1.2392656 reg_l1 12.066306 reg_l2 10.008058
loss 2.4458961
STEP 66 ================================
prereg loss 1.2375972 reg_l1 12.066011 reg_l2 10.008116
loss 2.4441984
STEP 67 ================================
prereg loss 1.2359215 reg_l1 12.06572 reg_l2 10.008179
loss 2.4424934
STEP 68 ================================
prereg loss 1.2342463 reg_l1 12.065429 reg_l2 10.008244
loss 2.4407892
STEP 69 ================================
prereg loss 1.2325683 reg_l1 12.065142 reg_l2 10.008311
loss 2.4390826
STEP 70 ================================
prereg loss 1.2308886 reg_l1 12.064856 reg_l2 10.008384
loss 2.437374
STEP 71 ================================
prereg loss 1.2292093 reg_l1 12.064572 reg_l2 10.008459
loss 2.4356666
STEP 72 ================================
prereg loss 1.2275289 reg_l1 12.06429 reg_l2 10.008537
loss 2.433958
STEP 73 ================================
prereg loss 1.2258538 reg_l1 12.064013 reg_l2 10.00862
loss 2.432255
STEP 74 ================================
prereg loss 1.2241786 reg_l1 12.063736 reg_l2 10.008707
loss 2.430552
STEP 75 ================================
prereg loss 1.222499 reg_l1 12.063462 reg_l2 10.008798
loss 2.4288454
STEP 76 ================================
prereg loss 1.2208266 reg_l1 12.063191 reg_l2 10.008891
loss 2.427146
STEP 77 ================================
prereg loss 1.2191491 reg_l1 12.06292 reg_l2 10.008986
loss 2.4254413
STEP 78 ================================
prereg loss 1.2174773 reg_l1 12.062652 reg_l2 10.009085
loss 2.4237425
STEP 79 ================================
prereg loss 1.215805 reg_l1 12.062387 reg_l2 10.009189
loss 2.4220438
STEP 80 ================================
prereg loss 1.2141407 reg_l1 12.062124 reg_l2 10.009295
loss 2.420353
STEP 81 ================================
prereg loss 1.212476 reg_l1 12.061865 reg_l2 10.009405
loss 2.4186625
STEP 82 ================================
prereg loss 1.2108186 reg_l1 12.061607 reg_l2 10.009519
loss 2.4169793
STEP 83 ================================
prereg loss 1.2091575 reg_l1 12.061351 reg_l2 10.009637
loss 2.4152927
STEP 84 ================================
prereg loss 1.2075065 reg_l1 12.061095 reg_l2 10.009754
loss 2.4136162
STEP 85 ================================
prereg loss 1.2058547 reg_l1 12.060844 reg_l2 10.009877
loss 2.4119391
STEP 86 ================================
prereg loss 1.2042103 reg_l1 12.060596 reg_l2 10.010003
loss 2.41027
STEP 87 ================================
prereg loss 1.2025666 reg_l1 12.0603485 reg_l2 10.010136
loss 2.4086015
STEP 88 ================================
prereg loss 1.2009292 reg_l1 12.060105 reg_l2 10.010267
loss 2.4069397
STEP 89 ================================
prereg loss 1.1992915 reg_l1 12.059864 reg_l2 10.010403
loss 2.4052777
STEP 90 ================================
prereg loss 1.1976659 reg_l1 12.059624 reg_l2 10.010544
loss 2.4036283
STEP 91 ================================
prereg loss 1.196043 reg_l1 12.059387 reg_l2 10.010686
loss 2.4019818
STEP 92 ================================
prereg loss 1.1944191 reg_l1 12.059151 reg_l2 10.010831
loss 2.4003344
STEP 93 ================================
prereg loss 1.192806 reg_l1 12.058917 reg_l2 10.010979
loss 2.3986979
STEP 94 ================================
prereg loss 1.1911973 reg_l1 12.058688 reg_l2 10.011128
loss 2.397066
STEP 95 ================================
prereg loss 1.189593 reg_l1 12.05846 reg_l2 10.011285
loss 2.3954391
STEP 96 ================================
prereg loss 1.1879928 reg_l1 12.058235 reg_l2 10.011443
loss 2.3938165
STEP 97 ================================
prereg loss 1.186395 reg_l1 12.058011 reg_l2 10.011604
loss 2.3921962
STEP 98 ================================
prereg loss 1.1848063 reg_l1 12.057791 reg_l2 10.011768
loss 2.3905854
STEP 99 ================================
prereg loss 1.1832213 reg_l1 12.057572 reg_l2 10.011936
loss 2.3889785
STEP 100 ================================
prereg loss 1.1816385 reg_l1 12.057355 reg_l2 10.012105
loss 2.387374
2022-06-28T20:17:36.659

julia> open("sparse21-after-200-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse21-after-200-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse21-after-200-steps-opt.ser", opt)

julia> steps!(300)
2022-06-28T20:19:06.289
STEP 1 ================================
prereg loss 1.1800652 reg_l1 12.05714 reg_l2 10.012277
loss 2.3857794
STEP 2 ================================
prereg loss 1.1784956 reg_l1 12.056929 reg_l2 10.012455
loss 2.3841887
STEP 3 ================================
prereg loss 1.1769319 reg_l1 12.05672 reg_l2 10.012634
loss 2.382604
STEP 4 ================================
prereg loss 1.1753703 reg_l1 12.056513 reg_l2 10.012816
loss 2.3810215
STEP 5 ================================
prereg loss 1.1738131 reg_l1 12.056308 reg_l2 10.013001
loss 2.379444
STEP 6 ================================
prereg loss 1.1722668 reg_l1 12.056104 reg_l2 10.013191
loss 2.3778772
STEP 7 ================================
prereg loss 1.1707201 reg_l1 12.0559025 reg_l2 10.01338
loss 2.3763103
STEP 8 ================================
prereg loss 1.1691761 reg_l1 12.055706 reg_l2 10.013576
loss 2.3747468
STEP 9 ================================
prereg loss 1.1676434 reg_l1 12.055511 reg_l2 10.013773
loss 2.3731947
STEP 10 ================================
prereg loss 1.1661136 reg_l1 12.055317 reg_l2 10.013975
loss 2.3716455
STEP 11 ================================
prereg loss 1.1645887 reg_l1 12.055126 reg_l2 10.014179
loss 2.3701015
STEP 12 ================================
prereg loss 1.1630652 reg_l1 12.054939 reg_l2 10.014384
loss 2.3685591
STEP 13 ================================
prereg loss 1.161551 reg_l1 12.054751 reg_l2 10.014592
loss 2.3670263
STEP 14 ================================
prereg loss 1.1600368 reg_l1 12.054566 reg_l2 10.014806
loss 2.3654933
STEP 15 ================================
prereg loss 1.1585306 reg_l1 12.054385 reg_l2 10.015021
loss 2.363969
STEP 16 ================================
prereg loss 1.1570301 reg_l1 12.054207 reg_l2 10.01524
loss 2.3624508
STEP 17 ================================
prereg loss 1.1555274 reg_l1 12.05403 reg_l2 10.015463
loss 2.3609304
STEP 18 ================================
prereg loss 1.1540354 reg_l1 12.053855 reg_l2 10.015686
loss 2.359421
STEP 19 ================================
prereg loss 1.152545 reg_l1 12.053683 reg_l2 10.015913
loss 2.3579135
STEP 20 ================================
prereg loss 1.1510621 reg_l1 12.053514 reg_l2 10.016142
loss 2.3564134
STEP 21 ================================
prereg loss 1.1495844 reg_l1 12.053344 reg_l2 10.016376
loss 2.354919
STEP 22 ================================
prereg loss 1.1481065 reg_l1 12.053178 reg_l2 10.016611
loss 2.3534243
STEP 23 ================================
prereg loss 1.1466341 reg_l1 12.053015 reg_l2 10.016851
loss 2.3519356
STEP 24 ================================
prereg loss 1.1451657 reg_l1 12.052855 reg_l2 10.017092
loss 2.350451
STEP 25 ================================
prereg loss 1.1437036 reg_l1 12.052697 reg_l2 10.017337
loss 2.3489733
STEP 26 ================================
prereg loss 1.1422431 reg_l1 12.05254 reg_l2 10.017586
loss 2.347497
STEP 27 ================================
prereg loss 1.1407886 reg_l1 12.052385 reg_l2 10.017836
loss 2.3460271
STEP 28 ================================
prereg loss 1.1393383 reg_l1 12.052235 reg_l2 10.018088
loss 2.3445616
STEP 29 ================================
prereg loss 1.1378934 reg_l1 12.052086 reg_l2 10.018344
loss 2.343102
STEP 30 ================================
prereg loss 1.1364484 reg_l1 12.051939 reg_l2 10.018603
loss 2.3416424
STEP 31 ================================
prereg loss 1.135009 reg_l1 12.051794 reg_l2 10.018867
loss 2.3401885
STEP 32 ================================
prereg loss 1.1335741 reg_l1 12.051652 reg_l2 10.019132
loss 2.3387394
STEP 33 ================================
prereg loss 1.1321453 reg_l1 12.051512 reg_l2 10.0194
loss 2.3372965
STEP 34 ================================
prereg loss 1.1307144 reg_l1 12.051374 reg_l2 10.0196705
loss 2.335852
STEP 35 ================================
prereg loss 1.1292909 reg_l1 12.051239 reg_l2 10.019943
loss 2.334415
STEP 36 ================================
prereg loss 1.1278712 reg_l1 12.051106 reg_l2 10.020222
loss 2.3329818
STEP 37 ================================
prereg loss 1.1264536 reg_l1 12.050975 reg_l2 10.0205
loss 2.331551
STEP 38 ================================
prereg loss 1.1250451 reg_l1 12.050847 reg_l2 10.0207815
loss 2.3301296
STEP 39 ================================
prereg loss 1.1236341 reg_l1 12.05072 reg_l2 10.021066
loss 2.3287063
STEP 40 ================================
prereg loss 1.122231 reg_l1 12.050597 reg_l2 10.021353
loss 2.3272908
STEP 41 ================================
prereg loss 1.1208357 reg_l1 12.050476 reg_l2 10.021642
loss 2.3258834
STEP 42 ================================
prereg loss 1.1194367 reg_l1 12.050356 reg_l2 10.021936
loss 2.3244724
STEP 43 ================================
prereg loss 1.1180401 reg_l1 12.05024 reg_l2 10.022233
loss 2.323064
STEP 44 ================================
prereg loss 1.1166523 reg_l1 12.050126 reg_l2 10.022531
loss 2.3216648
STEP 45 ================================
prereg loss 1.1152625 reg_l1 12.050013 reg_l2 10.022833
loss 2.3202639
STEP 46 ================================
prereg loss 1.1138858 reg_l1 12.049904 reg_l2 10.023137
loss 2.3188763
STEP 47 ================================
prereg loss 1.1125062 reg_l1 12.049795 reg_l2 10.023444
loss 2.3174858
STEP 48 ================================
prereg loss 1.1111329 reg_l1 12.049689 reg_l2 10.023754
loss 2.3161018
STEP 49 ================================
prereg loss 1.1097605 reg_l1 12.049586 reg_l2 10.024065
loss 2.3147192
STEP 50 ================================
prereg loss 1.1083916 reg_l1 12.049484 reg_l2 10.024381
loss 2.3133402
STEP 51 ================================
prereg loss 1.1070299 reg_l1 12.049388 reg_l2 10.024698
loss 2.3119688
STEP 52 ================================
prereg loss 1.1056693 reg_l1 12.049293 reg_l2 10.02502
loss 2.3105984
STEP 53 ================================
prereg loss 1.1043105 reg_l1 12.049198 reg_l2 10.025344
loss 2.3092303
STEP 54 ================================
prereg loss 1.1029563 reg_l1 12.049107 reg_l2 10.025669
loss 2.307867
STEP 55 ================================
prereg loss 1.10161 reg_l1 12.049015 reg_l2 10.025998
loss 2.3065114
STEP 56 ================================
prereg loss 1.1002631 reg_l1 12.048929 reg_l2 10.026329
loss 2.3051562
STEP 57 ================================
prereg loss 1.0989212 reg_l1 12.048843 reg_l2 10.026663
loss 2.3038056
STEP 58 ================================
prereg loss 1.0975838 reg_l1 12.048762 reg_l2 10.027001
loss 2.3024602
STEP 59 ================================
prereg loss 1.0962491 reg_l1 12.048682 reg_l2 10.02734
loss 2.3011174
STEP 60 ================================
prereg loss 1.0949169 reg_l1 12.048604 reg_l2 10.027681
loss 2.2997775
STEP 61 ================================
prereg loss 1.0935844 reg_l1 12.048529 reg_l2 10.028026
loss 2.2984374
STEP 62 ================================
prereg loss 1.0922627 reg_l1 12.048453 reg_l2 10.028374
loss 2.2971082
STEP 63 ================================
prereg loss 1.0909414 reg_l1 12.048383 reg_l2 10.028723
loss 2.2957797
STEP 64 ================================
prereg loss 1.0896266 reg_l1 12.048315 reg_l2 10.029076
loss 2.294458
STEP 65 ================================
prereg loss 1.0883068 reg_l1 12.048247 reg_l2 10.029431
loss 2.2931316
STEP 66 ================================
prereg loss 1.0869992 reg_l1 12.048184 reg_l2 10.029789
loss 2.2918177
STEP 67 ================================
prereg loss 1.0856929 reg_l1 12.048121 reg_l2 10.030149
loss 2.290505
STEP 68 ================================
prereg loss 1.0843923 reg_l1 12.048061 reg_l2 10.030512
loss 2.2891984
STEP 69 ================================
prereg loss 1.083093 reg_l1 12.048002 reg_l2 10.030876
loss 2.2878933
STEP 70 ================================
prereg loss 1.0817957 reg_l1 12.047948 reg_l2 10.031244
loss 2.2865906
STEP 71 ================================
prereg loss 1.0804988 reg_l1 12.0478945 reg_l2 10.031614
loss 2.2852883
STEP 72 ================================
prereg loss 1.0792139 reg_l1 12.047845 reg_l2 10.031988
loss 2.2839985
STEP 73 ================================
prereg loss 1.0779229 reg_l1 12.047796 reg_l2 10.032363
loss 2.2827024
STEP 74 ================================
prereg loss 1.0766438 reg_l1 12.047749 reg_l2 10.032742
loss 2.2814188
STEP 75 ================================
prereg loss 1.0753623 reg_l1 12.047707 reg_l2 10.033121
loss 2.280133
STEP 76 ================================
prereg loss 1.0740901 reg_l1 12.047665 reg_l2 10.033505
loss 2.2788568
STEP 77 ================================
prereg loss 1.072817 reg_l1 12.047623 reg_l2 10.033891
loss 2.2775793
STEP 78 ================================
prereg loss 1.0715477 reg_l1 12.047586 reg_l2 10.034278
loss 2.2763064
STEP 79 ================================
prereg loss 1.0702825 reg_l1 12.047552 reg_l2 10.03467
loss 2.2750378
STEP 80 ================================
prereg loss 1.069021 reg_l1 12.04752 reg_l2 10.035062
loss 2.273773
STEP 81 ================================
prereg loss 1.0677618 reg_l1 12.047488 reg_l2 10.035458
loss 2.2725105
STEP 82 ================================
prereg loss 1.0665058 reg_l1 12.047461 reg_l2 10.035855
loss 2.271252
STEP 83 ================================
prereg loss 1.0652565 reg_l1 12.047434 reg_l2 10.036255
loss 2.27
STEP 84 ================================
prereg loss 1.0640091 reg_l1 12.047409 reg_l2 10.036657
loss 2.26875
STEP 85 ================================
prereg loss 1.0627635 reg_l1 12.047387 reg_l2 10.037065
loss 2.2675023
STEP 86 ================================
prereg loss 1.0615247 reg_l1 12.047369 reg_l2 10.037472
loss 2.2662616
STEP 87 ================================
prereg loss 1.0602859 reg_l1 12.047351 reg_l2 10.037881
loss 2.265021
STEP 88 ================================
prereg loss 1.0590496 reg_l1 12.047335 reg_l2 10.038293
loss 2.263783
STEP 89 ================================
prereg loss 1.05782 reg_l1 12.047322 reg_l2 10.038712
loss 2.2625523
STEP 90 ================================
prereg loss 1.056588 reg_l1 12.047313 reg_l2 10.039126
loss 2.2613194
STEP 91 ================================
prereg loss 1.0553623 reg_l1 12.047303 reg_l2 10.039545
loss 2.2600927
STEP 92 ================================
prereg loss 1.0541422 reg_l1 12.0472975 reg_l2 10.039969
loss 2.258872
STEP 93 ================================
prereg loss 1.052926 reg_l1 12.047292 reg_l2 10.040393
loss 2.2576551
STEP 94 ================================
prereg loss 1.0517099 reg_l1 12.04729 reg_l2 10.040819
loss 2.2564387
STEP 95 ================================
prereg loss 1.050499 reg_l1 12.04729 reg_l2 10.041249
loss 2.255228
STEP 96 ================================
prereg loss 1.0492927 reg_l1 12.047293 reg_l2 10.041679
loss 2.2540221
STEP 97 ================================
prereg loss 1.0480857 reg_l1 12.0472975 reg_l2 10.042113
loss 2.2528155
STEP 98 ================================
prereg loss 1.0468849 reg_l1 12.047304 reg_l2 10.042549
loss 2.2516153
STEP 99 ================================
prereg loss 1.0456805 reg_l1 12.047314 reg_l2 10.042986
loss 2.250412
STEP 100 ================================
prereg loss 1.0444871 reg_l1 12.047323 reg_l2 10.043427
loss 2.2492194
STEP 101 ================================
prereg loss 1.0432951 reg_l1 12.047337 reg_l2 10.043871
loss 2.2480288
STEP 102 ================================
prereg loss 1.0421072 reg_l1 12.047354 reg_l2 10.044317
loss 2.2468426
STEP 103 ================================
prereg loss 1.0409214 reg_l1 12.047372 reg_l2 10.0447645
loss 2.2456586
STEP 104 ================================
prereg loss 1.0397385 reg_l1 12.047391 reg_l2 10.045213
loss 2.2444777
STEP 105 ================================
prereg loss 1.0385559 reg_l1 12.047412 reg_l2 10.045667
loss 2.243297
STEP 106 ================================
prereg loss 1.0373794 reg_l1 12.047436 reg_l2 10.046121
loss 2.2421231
STEP 107 ================================
prereg loss 1.0362076 reg_l1 12.047463 reg_l2 10.046577
loss 2.240954
STEP 108 ================================
prereg loss 1.0350366 reg_l1 12.047491 reg_l2 10.047036
loss 2.2397857
STEP 109 ================================
prereg loss 1.0338683 reg_l1 12.047523 reg_l2 10.0475
loss 2.2386208
STEP 110 ================================
prereg loss 1.032704 reg_l1 12.047554 reg_l2 10.04796
loss 2.2374594
STEP 111 ================================
prereg loss 1.0315422 reg_l1 12.047589 reg_l2 10.048427
loss 2.2363012
STEP 112 ================================
prereg loss 1.0303831 reg_l1 12.0476265 reg_l2 10.048893
loss 2.2351458
STEP 113 ================================
prereg loss 1.0292253 reg_l1 12.047666 reg_l2 10.049366
loss 2.233992
STEP 114 ================================
prereg loss 1.028076 reg_l1 12.047708 reg_l2 10.049837
loss 2.2328467
STEP 115 ================================
prereg loss 1.0269284 reg_l1 12.0477495 reg_l2 10.050312
loss 2.2317033
STEP 116 ================================
prereg loss 1.0257773 reg_l1 12.047795 reg_l2 10.05079
loss 2.230557
STEP 117 ================================
prereg loss 1.0246356 reg_l1 12.047842 reg_l2 10.051267
loss 2.2294197
STEP 118 ================================
prereg loss 1.0234971 reg_l1 12.047892 reg_l2 10.051748
loss 2.2282863
STEP 119 ================================
prereg loss 1.022359 reg_l1 12.047944 reg_l2 10.052231
loss 2.2271533
STEP 120 ================================
prereg loss 1.0212233 reg_l1 12.047997 reg_l2 10.052717
loss 2.2260232
STEP 121 ================================
prereg loss 1.0200897 reg_l1 12.048053 reg_l2 10.0532055
loss 2.224895
STEP 122 ================================
prereg loss 1.0189655 reg_l1 12.048112 reg_l2 10.053694
loss 2.2237768
STEP 123 ================================
prereg loss 1.0178385 reg_l1 12.048171 reg_l2 10.054187
loss 2.2226558
STEP 124 ================================
prereg loss 1.0167156 reg_l1 12.048234 reg_l2 10.05468
loss 2.221539
STEP 125 ================================
prereg loss 1.0155976 reg_l1 12.048299 reg_l2 10.055177
loss 2.2204275
STEP 126 ================================
prereg loss 1.0144782 reg_l1 12.0483675 reg_l2 10.0556755
loss 2.219315
STEP 127 ================================
prereg loss 1.0133647 reg_l1 12.048433 reg_l2 10.056175
loss 2.218208
STEP 128 ================================
prereg loss 1.0122527 reg_l1 12.048505 reg_l2 10.056677
loss 2.2171032
STEP 129 ================================
prereg loss 1.0111439 reg_l1 12.048576 reg_l2 10.057181
loss 2.2160015
STEP 130 ================================
prereg loss 1.0100391 reg_l1 12.048651 reg_l2 10.057687
loss 2.2149043
STEP 131 ================================
prereg loss 1.0089364 reg_l1 12.048728 reg_l2 10.058196
loss 2.2138093
STEP 132 ================================
prereg loss 1.0078353 reg_l1 12.048808 reg_l2 10.058709
loss 2.212716
STEP 133 ================================
prereg loss 1.0067395 reg_l1 12.048889 reg_l2 10.05922
loss 2.2116284
STEP 134 ================================
prereg loss 1.0056452 reg_l1 12.048972 reg_l2 10.059734
loss 2.2105424
STEP 135 ================================
prereg loss 1.0045522 reg_l1 12.049056 reg_l2 10.060251
loss 2.2094579
STEP 136 ================================
prereg loss 1.0034659 reg_l1 12.049144 reg_l2 10.060769
loss 2.2083802
STEP 137 ================================
prereg loss 1.0023776 reg_l1 12.0492325 reg_l2 10.061291
loss 2.207301
STEP 138 ================================
prereg loss 1.0012953 reg_l1 12.049324 reg_l2 10.061814
loss 2.2062278
STEP 139 ================================
prereg loss 1.0002135 reg_l1 12.049418 reg_l2 10.06234
loss 2.2051554
STEP 140 ================================
prereg loss 0.9991357 reg_l1 12.049512 reg_l2 10.062867
loss 2.2040868
STEP 141 ================================
prereg loss 0.99805874 reg_l1 12.04961 reg_l2 10.063395
loss 2.2030199
STEP 142 ================================
prereg loss 0.9969899 reg_l1 12.049709 reg_l2 10.063926
loss 2.2019608
STEP 143 ================================
prereg loss 0.9959173 reg_l1 12.049809 reg_l2 10.064459
loss 2.2008982
STEP 144 ================================
prereg loss 0.99484974 reg_l1 12.049914 reg_l2 10.064996
loss 2.1998413
STEP 145 ================================
prereg loss 0.9937835 reg_l1 12.050019 reg_l2 10.065531
loss 2.1987853
STEP 146 ================================
prereg loss 0.992721 reg_l1 12.050126 reg_l2 10.0660715
loss 2.1977336
STEP 147 ================================
prereg loss 0.99166137 reg_l1 12.050237 reg_l2 10.066611
loss 2.196685
STEP 148 ================================
prereg loss 0.9906069 reg_l1 12.050346 reg_l2 10.067152
loss 2.1956415
STEP 149 ================================
prereg loss 0.9895548 reg_l1 12.050459 reg_l2 10.067698
loss 2.1946008
STEP 150 ================================
prereg loss 0.9885023 reg_l1 12.050575 reg_l2 10.068244
loss 2.19356
STEP 151 ================================
prereg loss 0.9874552 reg_l1 12.050693 reg_l2 10.068795
loss 2.1925244
STEP 152 ================================
prereg loss 0.98640513 reg_l1 12.050811 reg_l2 10.069344
loss 2.1914864
STEP 153 ================================
prereg loss 0.98536676 reg_l1 12.050933 reg_l2 10.069897
loss 2.19046
STEP 154 ================================
prereg loss 0.9843239 reg_l1 12.051054 reg_l2 10.070451
loss 2.1894293
STEP 155 ================================
prereg loss 0.9832856 reg_l1 12.05118 reg_l2 10.071008
loss 2.1884036
STEP 156 ================================
prereg loss 0.9822473 reg_l1 12.051308 reg_l2 10.071565
loss 2.1873782
STEP 157 ================================
prereg loss 0.9812172 reg_l1 12.051436 reg_l2 10.072126
loss 2.1863608
STEP 158 ================================
prereg loss 0.98018295 reg_l1 12.051568 reg_l2 10.072689
loss 2.1853397
STEP 159 ================================
prereg loss 0.9791541 reg_l1 12.051701 reg_l2 10.073251
loss 2.1843243
STEP 160 ================================
prereg loss 0.97813267 reg_l1 12.051834 reg_l2 10.073817
loss 2.183316
STEP 161 ================================
prereg loss 0.9771087 reg_l1 12.0519705 reg_l2 10.074384
loss 2.1823058
STEP 162 ================================
prereg loss 0.97608715 reg_l1 12.05211 reg_l2 10.074953
loss 2.1812983
STEP 163 ================================
prereg loss 0.9750679 reg_l1 12.052252 reg_l2 10.075524
loss 2.180293
STEP 164 ================================
prereg loss 0.97405404 reg_l1 12.052393 reg_l2 10.0760975
loss 2.1792934
STEP 165 ================================
prereg loss 0.97303915 reg_l1 12.052538 reg_l2 10.076673
loss 2.178293
STEP 166 ================================
prereg loss 0.97202975 reg_l1 12.052683 reg_l2 10.07725
loss 2.177298
STEP 167 ================================
prereg loss 0.9710199 reg_l1 12.052833 reg_l2 10.077826
loss 2.1763031
STEP 168 ================================
prereg loss 0.9700163 reg_l1 12.052982 reg_l2 10.078406
loss 2.1753147
STEP 169 ================================
prereg loss 0.9690123 reg_l1 12.053134 reg_l2 10.07899
loss 2.1743257
STEP 170 ================================
prereg loss 0.9680141 reg_l1 12.0532875 reg_l2 10.079573
loss 2.173343
STEP 171 ================================
prereg loss 0.96701473 reg_l1 12.053444 reg_l2 10.080157
loss 2.1723592
STEP 172 ================================
prereg loss 0.9660164 reg_l1 12.0536 reg_l2 10.080747
loss 2.1713765
STEP 173 ================================
prereg loss 0.96502393 reg_l1 12.0537615 reg_l2 10.081334
loss 2.1704001
STEP 174 ================================
prereg loss 0.9640321 reg_l1 12.053923 reg_l2 10.081927
loss 2.1694243
STEP 175 ================================
prereg loss 0.963044 reg_l1 12.054085 reg_l2 10.082521
loss 2.1684525
STEP 176 ================================
prereg loss 0.96205884 reg_l1 12.05425 reg_l2 10.083113
loss 2.1674838
STEP 177 ================================
prereg loss 0.96107054 reg_l1 12.054418 reg_l2 10.08371
loss 2.1665125
STEP 178 ================================
prereg loss 0.9600897 reg_l1 12.054586 reg_l2 10.084309
loss 2.1655483
STEP 179 ================================
prereg loss 0.95911044 reg_l1 12.054755 reg_l2 10.0849085
loss 2.164586
STEP 180 ================================
prereg loss 0.95813227 reg_l1 12.05493 reg_l2 10.08551
loss 2.1636252
STEP 181 ================================
prereg loss 0.9571598 reg_l1 12.055104 reg_l2 10.086113
loss 2.1626704
STEP 182 ================================
prereg loss 0.9561829 reg_l1 12.05528 reg_l2 10.086719
loss 2.161711
STEP 183 ================================
prereg loss 0.9552152 reg_l1 12.055457 reg_l2 10.087323
loss 2.1607609
STEP 184 ================================
prereg loss 0.95424503 reg_l1 12.055637 reg_l2 10.087933
loss 2.1598089
STEP 185 ================================
prereg loss 0.95328206 reg_l1 12.055818 reg_l2 10.088543
loss 2.1588638
STEP 186 ================================
prereg loss 0.95231974 reg_l1 12.056003 reg_l2 10.089154
loss 2.15792
STEP 187 ================================
prereg loss 0.9513572 reg_l1 12.056187 reg_l2 10.0897665
loss 2.156976
STEP 188 ================================
prereg loss 0.9503971 reg_l1 12.056375 reg_l2 10.090384
loss 2.1560345
STEP 189 ================================
prereg loss 0.94944257 reg_l1 12.056563 reg_l2 10.090998
loss 2.155099
STEP 190 ================================
prereg loss 0.94848335 reg_l1 12.056753 reg_l2 10.091617
loss 2.1541586
STEP 191 ================================
prereg loss 0.9475341 reg_l1 12.056946 reg_l2 10.0922365
loss 2.1532288
STEP 192 ================================
prereg loss 0.9465846 reg_l1 12.05714 reg_l2 10.092861
loss 2.1522987
STEP 193 ================================
prereg loss 0.945638 reg_l1 12.057335 reg_l2 10.093484
loss 2.1513715
STEP 194 ================================
prereg loss 0.94469196 reg_l1 12.057534 reg_l2 10.09411
loss 2.1504455
STEP 195 ================================
prereg loss 0.9437433 reg_l1 12.057732 reg_l2 10.094733
loss 2.1495166
STEP 196 ================================
prereg loss 0.94280475 reg_l1 12.057933 reg_l2 10.095363
loss 2.148598
STEP 197 ================================
prereg loss 0.941867 reg_l1 12.058136 reg_l2 10.095994
loss 2.1476808
STEP 198 ================================
prereg loss 0.94092774 reg_l1 12.058341 reg_l2 10.096625
loss 2.146762
STEP 199 ================================
prereg loss 0.9399938 reg_l1 12.058548 reg_l2 10.097257
loss 2.1458485
STEP 200 ================================
prereg loss 0.9390627 reg_l1 12.058757 reg_l2 10.097892
loss 2.1449385
STEP 201 ================================
prereg loss 0.9381317 reg_l1 12.058967 reg_l2 10.098528
loss 2.1440284
STEP 202 ================================
prereg loss 0.9372024 reg_l1 12.059177 reg_l2 10.099167
loss 2.14312
STEP 203 ================================
prereg loss 0.93627554 reg_l1 12.059392 reg_l2 10.099808
loss 2.1422148
STEP 204 ================================
prereg loss 0.9353497 reg_l1 12.059607 reg_l2 10.100449
loss 2.1413102
STEP 205 ================================
prereg loss 0.93442726 reg_l1 12.059822 reg_l2 10.10109
loss 2.1404095
STEP 206 ================================
prereg loss 0.9335087 reg_l1 12.0600395 reg_l2 10.101736
loss 2.1395128
STEP 207 ================================
prereg loss 0.93258923 reg_l1 12.060261 reg_l2 10.102381
loss 2.1386154
STEP 208 ================================
prereg loss 0.93167335 reg_l1 12.060483 reg_l2 10.103027
loss 2.1377218
STEP 209 ================================
prereg loss 0.93076104 reg_l1 12.060706 reg_l2 10.103677
loss 2.1368318
STEP 210 ================================
prereg loss 0.92984957 reg_l1 12.06093 reg_l2 10.104328
loss 2.1359427
STEP 211 ================================
prereg loss 0.92893946 reg_l1 12.061158 reg_l2 10.104979
loss 2.1350553
STEP 212 ================================
prereg loss 0.9280325 reg_l1 12.061387 reg_l2 10.105634
loss 2.1341712
STEP 213 ================================
prereg loss 0.9271255 reg_l1 12.061617 reg_l2 10.106288
loss 2.1332872
STEP 214 ================================
prereg loss 0.9262215 reg_l1 12.061849 reg_l2 10.106944
loss 2.1324062
STEP 215 ================================
prereg loss 0.92532194 reg_l1 12.06208 reg_l2 10.107601
loss 2.13153
STEP 216 ================================
prereg loss 0.9244241 reg_l1 12.062316 reg_l2 10.108261
loss 2.1306558
STEP 217 ================================
prereg loss 0.92352825 reg_l1 12.062553 reg_l2 10.108923
loss 2.1297836
STEP 218 ================================
prereg loss 0.9226295 reg_l1 12.062792 reg_l2 10.109586
loss 2.1289086
STEP 219 ================================
prereg loss 0.9217358 reg_l1 12.063031 reg_l2 10.1102495
loss 2.128039
STEP 220 ================================
prereg loss 0.92084455 reg_l1 12.063272 reg_l2 10.110916
loss 2.127172
STEP 221 ================================
prereg loss 0.91995114 reg_l1 12.063516 reg_l2 10.111581
loss 2.1263027
STEP 222 ================================
prereg loss 0.91906947 reg_l1 12.063761 reg_l2 10.112251
loss 2.1254456
STEP 223 ================================
prereg loss 0.9181805 reg_l1 12.064008 reg_l2 10.11292
loss 2.1245813
STEP 224 ================================
prereg loss 0.9172974 reg_l1 12.064256 reg_l2 10.113593
loss 2.123723
STEP 225 ================================
prereg loss 0.91641563 reg_l1 12.064507 reg_l2 10.114266
loss 2.1228664
STEP 226 ================================
prereg loss 0.91553617 reg_l1 12.064757 reg_l2 10.114941
loss 2.122012
STEP 227 ================================
prereg loss 0.91465586 reg_l1 12.065008 reg_l2 10.115615
loss 2.1211567
STEP 228 ================================
prereg loss 0.9137839 reg_l1 12.065263 reg_l2 10.116292
loss 2.1203103
STEP 229 ================================
prereg loss 0.912906 reg_l1 12.065519 reg_l2 10.116972
loss 2.119458
STEP 230 ================================
prereg loss 0.912038 reg_l1 12.065779 reg_l2 10.117653
loss 2.1186159
STEP 231 ================================
prereg loss 0.9111688 reg_l1 12.066036 reg_l2 10.118335
loss 2.1177726
STEP 232 ================================
prereg loss 0.91030043 reg_l1 12.066296 reg_l2 10.119016
loss 2.11693
STEP 233 ================================
prereg loss 0.909436 reg_l1 12.066558 reg_l2 10.1197
loss 2.1160917
STEP 234 ================================
prereg loss 0.9085672 reg_l1 12.066823 reg_l2 10.120386
loss 2.1152496
STEP 235 ================================
prereg loss 0.9077062 reg_l1 12.067088 reg_l2 10.121075
loss 2.114415
STEP 236 ================================
prereg loss 0.90684813 reg_l1 12.067353 reg_l2 10.121764
loss 2.1135836
STEP 237 ================================
prereg loss 0.9059872 reg_l1 12.067622 reg_l2 10.122453
loss 2.1127493
STEP 238 ================================
prereg loss 0.90512896 reg_l1 12.067891 reg_l2 10.123143
loss 2.111918
STEP 239 ================================
prereg loss 0.90427417 reg_l1 12.068163 reg_l2 10.123836
loss 2.1110904
STEP 240 ================================
prereg loss 0.9034228 reg_l1 12.068434 reg_l2 10.124531
loss 2.1102662
STEP 241 ================================
prereg loss 0.90257037 reg_l1 12.068711 reg_l2 10.125226
loss 2.1094415
STEP 242 ================================
prereg loss 0.9017225 reg_l1 12.068986 reg_l2 10.125924
loss 2.1086211
STEP 243 ================================
prereg loss 0.90087235 reg_l1 12.069263 reg_l2 10.126622
loss 2.1077986
STEP 244 ================================
prereg loss 0.9000324 reg_l1 12.069541 reg_l2 10.127319
loss 2.1069865
STEP 245 ================================
prereg loss 0.89918524 reg_l1 12.069821 reg_l2 10.12802
loss 2.1061673
STEP 246 ================================
prereg loss 0.8983428 reg_l1 12.070106 reg_l2 10.128723
loss 2.1053534
STEP 247 ================================
prereg loss 0.89750284 reg_l1 12.070386 reg_l2 10.129428
loss 2.1045415
STEP 248 ================================
prereg loss 0.8966633 reg_l1 12.070672 reg_l2 10.130133
loss 2.1037307
STEP 249 ================================
prereg loss 0.8958312 reg_l1 12.070956 reg_l2 10.130839
loss 2.1029267
STEP 250 ================================
prereg loss 0.89499253 reg_l1 12.071244 reg_l2 10.131547
loss 2.102117
STEP 251 ================================
prereg loss 0.89416075 reg_l1 12.071532 reg_l2 10.132256
loss 2.101314
STEP 252 ================================
prereg loss 0.8933302 reg_l1 12.071822 reg_l2 10.132963
loss 2.1005125
STEP 253 ================================
prereg loss 0.89250153 reg_l1 12.072116 reg_l2 10.133676
loss 2.099713
STEP 254 ================================
prereg loss 0.8916742 reg_l1 12.072408 reg_l2 10.134389
loss 2.098915
STEP 255 ================================
prereg loss 0.89084804 reg_l1 12.072702 reg_l2 10.135102
loss 2.0981183
STEP 256 ================================
prereg loss 0.8900257 reg_l1 12.072999 reg_l2 10.135818
loss 2.0973256
STEP 257 ================================
prereg loss 0.88920265 reg_l1 12.073297 reg_l2 10.136533
loss 2.0965323
STEP 258 ================================
prereg loss 0.888385 reg_l1 12.073594 reg_l2 10.137253
loss 2.0957444
STEP 259 ================================
prereg loss 0.88756394 reg_l1 12.073894 reg_l2 10.137971
loss 2.0949533
STEP 260 ================================
prereg loss 0.8867486 reg_l1 12.074196 reg_l2 10.138693
loss 2.0941682
STEP 261 ================================
prereg loss 0.8859357 reg_l1 12.074499 reg_l2 10.139413
loss 2.0933857
STEP 262 ================================
prereg loss 0.8851175 reg_l1 12.074803 reg_l2 10.140138
loss 2.0925977
STEP 263 ================================
prereg loss 0.8843083 reg_l1 12.075109 reg_l2 10.140861
loss 2.091819
STEP 264 ================================
prereg loss 0.8834981 reg_l1 12.075416 reg_l2 10.141587
loss 2.0910397
STEP 265 ================================
prereg loss 0.88268864 reg_l1 12.075724 reg_l2 10.142313
loss 2.090261
STEP 266 ================================
prereg loss 0.8818845 reg_l1 12.0760355 reg_l2 10.143041
loss 2.089488
STEP 267 ================================
prereg loss 0.88107866 reg_l1 12.076344 reg_l2 10.143772
loss 2.088713
STEP 268 ================================
prereg loss 0.880275 reg_l1 12.076656 reg_l2 10.144502
loss 2.0879407
STEP 269 ================================
prereg loss 0.8794748 reg_l1 12.076971 reg_l2 10.145233
loss 2.087172
STEP 270 ================================
prereg loss 0.8786748 reg_l1 12.077286 reg_l2 10.145966
loss 2.0864034
STEP 271 ================================
prereg loss 0.87787515 reg_l1 12.0776005 reg_l2 10.1467
loss 2.0856352
STEP 272 ================================
prereg loss 0.87708026 reg_l1 12.077919 reg_l2 10.147436
loss 2.0848722
STEP 273 ================================
prereg loss 0.87628394 reg_l1 12.078238 reg_l2 10.148172
loss 2.0841076
STEP 274 ================================
prereg loss 0.875493 reg_l1 12.07856 reg_l2 10.148909
loss 2.083349
STEP 275 ================================
prereg loss 0.8747002 reg_l1 12.07888 reg_l2 10.149647
loss 2.0825882
STEP 276 ================================
prereg loss 0.87391204 reg_l1 12.079202 reg_l2 10.150387
loss 2.0818322
STEP 277 ================================
prereg loss 0.87312 reg_l1 12.079528 reg_l2 10.151131
loss 2.0810728
STEP 278 ================================
prereg loss 0.87233543 reg_l1 12.079852 reg_l2 10.151872
loss 2.0803208
STEP 279 ================================
prereg loss 0.871552 reg_l1 12.080178 reg_l2 10.152614
loss 2.0795698
STEP 280 ================================
prereg loss 0.870769 reg_l1 12.080507 reg_l2 10.153362
loss 2.0788198
STEP 281 ================================
prereg loss 0.86998755 reg_l1 12.080835 reg_l2 10.154107
loss 2.078071
STEP 282 ================================
prereg loss 0.8692086 reg_l1 12.081164 reg_l2 10.154855
loss 2.0773249
STEP 283 ================================
prereg loss 0.86843324 reg_l1 12.081497 reg_l2 10.1556015
loss 2.076583
STEP 284 ================================
prereg loss 0.86765414 reg_l1 12.08183 reg_l2 10.15635
loss 2.0758371
STEP 285 ================================
prereg loss 0.8668825 reg_l1 12.082164 reg_l2 10.157102
loss 2.075099
STEP 286 ================================
prereg loss 0.8661116 reg_l1 12.0824995 reg_l2 10.157853
loss 2.0743616
STEP 287 ================================
prereg loss 0.86533946 reg_l1 12.082836 reg_l2 10.158606
loss 2.0736232
STEP 288 ================================
prereg loss 0.8645732 reg_l1 12.083172 reg_l2 10.159359
loss 2.0728903
STEP 289 ================================
prereg loss 0.8638031 reg_l1 12.083512 reg_l2 10.160115
loss 2.0721543
STEP 290 ================================
prereg loss 0.8630359 reg_l1 12.083853 reg_l2 10.16087
loss 2.0714211
STEP 291 ================================
prereg loss 0.8622698 reg_l1 12.084194 reg_l2 10.161628
loss 2.0706892
STEP 292 ================================
prereg loss 0.86150926 reg_l1 12.084536 reg_l2 10.162388
loss 2.0699627
STEP 293 ================================
prereg loss 0.8607468 reg_l1 12.084879 reg_l2 10.163145
loss 2.0692346
STEP 294 ================================
prereg loss 0.8599913 reg_l1 12.085224 reg_l2 10.163908
loss 2.0685139
STEP 295 ================================
prereg loss 0.85923165 reg_l1 12.085568 reg_l2 10.164668
loss 2.0677886
STEP 296 ================================
prereg loss 0.85847473 reg_l1 12.0859165 reg_l2 10.165431
loss 2.0670664
STEP 297 ================================
prereg loss 0.85772085 reg_l1 12.086265 reg_l2 10.166195
loss 2.0663474
STEP 298 ================================
prereg loss 0.8569688 reg_l1 12.086615 reg_l2 10.166961
loss 2.0656302
STEP 299 ================================
prereg loss 0.85621685 reg_l1 12.086965 reg_l2 10.167728
loss 2.0649133
STEP 300 ================================
prereg loss 0.85546356 reg_l1 12.0873165 reg_l2 10.168495
loss 2.0641952
2022-06-28T20:30:55.404

julia> open("sparse21-after-500-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse21-after-500-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse21-after-500-steps-opt.ser", opt)

julia> steps!(500)
2022-06-28T20:33:06.705
STEP 1 ================================
prereg loss 0.854716 reg_l1 12.087669 reg_l2 10.169261
loss 2.063483
STEP 2 ================================
prereg loss 0.85396904 reg_l1 12.088022 reg_l2 10.170032
loss 2.0627713
STEP 3 ================================
prereg loss 0.85322505 reg_l1 12.088378 reg_l2 10.170803
loss 2.062063
STEP 4 ================================
prereg loss 0.8524765 reg_l1 12.088737 reg_l2 10.171576
loss 2.06135
STEP 5 ================================
prereg loss 0.8517338 reg_l1 12.089092 reg_l2 10.172348
loss 2.0606432
STEP 6 ================================
prereg loss 0.85099345 reg_l1 12.089452 reg_l2 10.173122
loss 2.0599387
STEP 7 ================================
prereg loss 0.85025746 reg_l1 12.08981 reg_l2 10.173897
loss 2.0592384
STEP 8 ================================
prereg loss 0.84951895 reg_l1 12.090172 reg_l2 10.174673
loss 2.058536
STEP 9 ================================
prereg loss 0.84878343 reg_l1 12.090533 reg_l2 10.17545
loss 2.0578368
STEP 10 ================================
prereg loss 0.8480473 reg_l1 12.090896 reg_l2 10.176228
loss 2.057137
STEP 11 ================================
prereg loss 0.8473124 reg_l1 12.091258 reg_l2 10.177006
loss 2.0564382
STEP 12 ================================
prereg loss 0.8465849 reg_l1 12.091623 reg_l2 10.177786
loss 2.0557473
STEP 13 ================================
prereg loss 0.8458556 reg_l1 12.09199 reg_l2 10.178569
loss 2.0550547
STEP 14 ================================
prereg loss 0.8451217 reg_l1 12.092358 reg_l2 10.179351
loss 2.0543575
STEP 15 ================================
prereg loss 0.84440035 reg_l1 12.092725 reg_l2 10.180133
loss 2.0536728
STEP 16 ================================
prereg loss 0.8436724 reg_l1 12.093094 reg_l2 10.180916
loss 2.0529819
STEP 17 ================================
prereg loss 0.8429506 reg_l1 12.093463 reg_l2 10.181702
loss 2.0522969
STEP 18 ================================
prereg loss 0.84222525 reg_l1 12.093836 reg_l2 10.1824875
loss 2.0516088
STEP 19 ================================
prereg loss 0.84150976 reg_l1 12.094209 reg_l2 10.183275
loss 2.0509307
STEP 20 ================================
prereg loss 0.84078705 reg_l1 12.094582 reg_l2 10.184062
loss 2.0502453
STEP 21 ================================
prereg loss 0.8400697 reg_l1 12.094956 reg_l2 10.184851
loss 2.0495653
STEP 22 ================================
prereg loss 0.83935696 reg_l1 12.095331 reg_l2 10.185643
loss 2.04889
STEP 23 ================================
prereg loss 0.8386389 reg_l1 12.095707 reg_l2 10.186433
loss 2.0482097
STEP 24 ================================
prereg loss 0.8379295 reg_l1 12.096084 reg_l2 10.187226
loss 2.0475378
STEP 25 ================================
prereg loss 0.8372143 reg_l1 12.096461 reg_l2 10.188019
loss 2.0468605
STEP 26 ================================
prereg loss 0.8365064 reg_l1 12.096841 reg_l2 10.188814
loss 2.0461905
STEP 27 ================================
prereg loss 0.83579546 reg_l1 12.097222 reg_l2 10.189608
loss 2.0455177
STEP 28 ================================
prereg loss 0.8350912 reg_l1 12.097603 reg_l2 10.190405
loss 2.0448515
STEP 29 ================================
prereg loss 0.83438176 reg_l1 12.097985 reg_l2 10.191201
loss 2.0441804
STEP 30 ================================
prereg loss 0.83368045 reg_l1 12.098369 reg_l2 10.191999
loss 2.0435174
STEP 31 ================================
prereg loss 0.8329779 reg_l1 12.098751 reg_l2 10.1928
loss 2.0428529
STEP 32 ================================
prereg loss 0.8322757 reg_l1 12.099135 reg_l2 10.193599
loss 2.0421894
STEP 33 ================================
prereg loss 0.8315757 reg_l1 12.099522 reg_l2 10.1944
loss 2.041528
STEP 34 ================================
prereg loss 0.8308768 reg_l1 12.099911 reg_l2 10.195204
loss 2.0408678
STEP 35 ================================
prereg loss 0.8301815 reg_l1 12.100297 reg_l2 10.196006
loss 2.0402112
STEP 36 ================================
prereg loss 0.82948864 reg_l1 12.100684 reg_l2 10.196808
loss 2.039557
STEP 37 ================================
prereg loss 0.8287925 reg_l1 12.101075 reg_l2 10.197614
loss 2.0389001
STEP 38 ================================
prereg loss 0.82810116 reg_l1 12.101465 reg_l2 10.19842
loss 2.0382476
STEP 39 ================================
prereg loss 0.8274118 reg_l1 12.101858 reg_l2 10.199225
loss 2.0375977
STEP 40 ================================
prereg loss 0.82672524 reg_l1 12.102249 reg_l2 10.200034
loss 2.03695
STEP 41 ================================
prereg loss 0.82603455 reg_l1 12.102642 reg_l2 10.200843
loss 2.0362988
STEP 42 ================================
prereg loss 0.8253489 reg_l1 12.103038 reg_l2 10.201652
loss 2.0356526
STEP 43 ================================
prereg loss 0.8246637 reg_l1 12.103432 reg_l2 10.202461
loss 2.035007
STEP 44 ================================
prereg loss 0.8239806 reg_l1 12.103827 reg_l2 10.203272
loss 2.0343635
STEP 45 ================================
prereg loss 0.8232995 reg_l1 12.104224 reg_l2 10.204085
loss 2.033722
STEP 46 ================================
prereg loss 0.8226188 reg_l1 12.104624 reg_l2 10.204898
loss 2.0330813
STEP 47 ================================
prereg loss 0.8219399 reg_l1 12.105021 reg_l2 10.205712
loss 2.032442
STEP 48 ================================
prereg loss 0.82126576 reg_l1 12.10542 reg_l2 10.206527
loss 2.031808
STEP 49 ================================
prereg loss 0.8205885 reg_l1 12.105819 reg_l2 10.207343
loss 2.0311704
STEP 50 ================================
prereg loss 0.8199124 reg_l1 12.106221 reg_l2 10.208159
loss 2.0305345
STEP 51 ================================
prereg loss 0.81924284 reg_l1 12.106624 reg_l2 10.20898
loss 2.0299053
STEP 52 ================================
prereg loss 0.81856966 reg_l1 12.107024 reg_l2 10.209796
loss 2.029272
STEP 53 ================================
prereg loss 0.81790334 reg_l1 12.1074295 reg_l2 10.210614
loss 2.0286462
STEP 54 ================================
prereg loss 0.8172313 reg_l1 12.107836 reg_l2 10.211435
loss 2.028015
STEP 55 ================================
prereg loss 0.81656736 reg_l1 12.108238 reg_l2 10.212256
loss 2.0273912
STEP 56 ================================
prereg loss 0.8158986 reg_l1 12.108645 reg_l2 10.2130785
loss 2.0267632
STEP 57 ================================
prereg loss 0.815236 reg_l1 12.109053 reg_l2 10.213903
loss 2.0261412
STEP 58 ================================
prereg loss 0.8145743 reg_l1 12.10946 reg_l2 10.214725
loss 2.0255203
STEP 59 ================================
prereg loss 0.81391406 reg_l1 12.109867 reg_l2 10.215549
loss 2.024901
STEP 60 ================================
prereg loss 0.81325203 reg_l1 12.110277 reg_l2 10.216374
loss 2.0242798
STEP 61 ================================
prereg loss 0.8125942 reg_l1 12.110687 reg_l2 10.217203
loss 2.023663
STEP 62 ================================
prereg loss 0.81193835 reg_l1 12.111097 reg_l2 10.218031
loss 2.0230482
STEP 63 ================================
prereg loss 0.8112834 reg_l1 12.11151 reg_l2 10.218857
loss 2.0224345
STEP 64 ================================
prereg loss 0.81062746 reg_l1 12.111921 reg_l2 10.2196865
loss 2.0218196
STEP 65 ================================
prereg loss 0.8099744 reg_l1 12.112335 reg_l2 10.220518
loss 2.0212078
STEP 66 ================================
prereg loss 0.8093233 reg_l1 12.112749 reg_l2 10.221347
loss 2.0205984
STEP 67 ================================
prereg loss 0.8086745 reg_l1 12.113163 reg_l2 10.222178
loss 2.019991
STEP 68 ================================
prereg loss 0.80802506 reg_l1 12.113577 reg_l2 10.223011
loss 2.0193827
STEP 69 ================================
prereg loss 0.80737954 reg_l1 12.113996 reg_l2 10.223845
loss 2.018779
STEP 70 ================================
prereg loss 0.80673295 reg_l1 12.114412 reg_l2 10.224678
loss 2.0181742
STEP 71 ================================
prereg loss 0.8060908 reg_l1 12.114828 reg_l2 10.2255125
loss 2.0175736
STEP 72 ================================
prereg loss 0.8054511 reg_l1 12.115246 reg_l2 10.226347
loss 2.0169756
STEP 73 ================================
prereg loss 0.8048071 reg_l1 12.115665 reg_l2 10.227185
loss 2.0163736
STEP 74 ================================
prereg loss 0.80416983 reg_l1 12.116085 reg_l2 10.228022
loss 2.0157783
STEP 75 ================================
prereg loss 0.80353266 reg_l1 12.116506 reg_l2 10.22886
loss 2.0151832
STEP 76 ================================
prereg loss 0.80289644 reg_l1 12.116926 reg_l2 10.229701
loss 2.014589
STEP 77 ================================
prereg loss 0.80225855 reg_l1 12.117348 reg_l2 10.230539
loss 2.0139933
STEP 78 ================================
prereg loss 0.80162555 reg_l1 12.11777 reg_l2 10.2313795
loss 2.0134027
STEP 79 ================================
prereg loss 0.8009947 reg_l1 12.118194 reg_l2 10.232222
loss 2.012814
STEP 80 ================================
prereg loss 0.8003623 reg_l1 12.118619 reg_l2 10.233066
loss 2.0122242
STEP 81 ================================
prereg loss 0.79973227 reg_l1 12.119041 reg_l2 10.233906
loss 2.0116365
STEP 82 ================================
prereg loss 0.7991063 reg_l1 12.119467 reg_l2 10.234751
loss 2.011053
STEP 83 ================================
prereg loss 0.79847753 reg_l1 12.119893 reg_l2 10.235595
loss 2.0104668
STEP 84 ================================
prereg loss 0.7978537 reg_l1 12.120319 reg_l2 10.236442
loss 2.0098858
STEP 85 ================================
prereg loss 0.7972282 reg_l1 12.120748 reg_l2 10.2372875
loss 2.009303
STEP 86 ================================
prereg loss 0.796606 reg_l1 12.121176 reg_l2 10.238134
loss 2.0087235
STEP 87 ================================
prereg loss 0.795988 reg_l1 12.121602 reg_l2 10.238981
loss 2.0081482
STEP 88 ================================
prereg loss 0.7953668 reg_l1 12.122033 reg_l2 10.239831
loss 2.0075703
STEP 89 ================================
prereg loss 0.79474735 reg_l1 12.122463 reg_l2 10.240679
loss 2.0069938
STEP 90 ================================
prereg loss 0.79412866 reg_l1 12.122892 reg_l2 10.24153
loss 2.0064178
STEP 91 ================================
prereg loss 0.79351497 reg_l1 12.123324 reg_l2 10.242382
loss 2.0058475
STEP 92 ================================
prereg loss 0.7928992 reg_l1 12.123755 reg_l2 10.243233
loss 2.0052748
STEP 93 ================================
prereg loss 0.7922877 reg_l1 12.1241865 reg_l2 10.244084
loss 2.0047064
STEP 94 ================================
prereg loss 0.79167795 reg_l1 12.1246195 reg_l2 10.244939
loss 2.00414
STEP 95 ================================
prereg loss 0.7910659 reg_l1 12.125053 reg_l2 10.245792
loss 2.0035713
STEP 96 ================================
prereg loss 0.79045755 reg_l1 12.125488 reg_l2 10.246647
loss 2.0030065
STEP 97 ================================
prereg loss 0.7898476 reg_l1 12.125923 reg_l2 10.247502
loss 2.00244
STEP 98 ================================
prereg loss 0.78924674 reg_l1 12.126358 reg_l2 10.248361
loss 2.0018826
STEP 99 ================================
prereg loss 0.788642 reg_l1 12.126794 reg_l2 10.249216
loss 2.0013213
STEP 100 ================================
prereg loss 0.78803486 reg_l1 12.12723 reg_l2 10.250074
loss 2.000758
STEP 101 ================================
prereg loss 0.7874353 reg_l1 12.127667 reg_l2 10.250933
loss 2.0002022
STEP 102 ================================
prereg loss 0.7868376 reg_l1 12.128104 reg_l2 10.251794
loss 1.999648
STEP 103 ================================
prereg loss 0.7862344 reg_l1 12.128542 reg_l2 10.252653
loss 1.9990886
STEP 104 ================================
prereg loss 0.78563935 reg_l1 12.128982 reg_l2 10.253512
loss 1.9985375
STEP 105 ================================
prereg loss 0.7850418 reg_l1 12.129421 reg_l2 10.2543745
loss 1.9979839
STEP 106 ================================
prereg loss 0.7844495 reg_l1 12.129862 reg_l2 10.255237
loss 1.9974358
STEP 107 ================================
prereg loss 0.78385633 reg_l1 12.130301 reg_l2 10.2561
loss 1.9968865
STEP 108 ================================
prereg loss 0.78326607 reg_l1 12.130743 reg_l2 10.256963
loss 1.9963404
STEP 109 ================================
prereg loss 0.78267485 reg_l1 12.131184 reg_l2 10.25783
loss 1.9957933
STEP 110 ================================
prereg loss 0.78208333 reg_l1 12.131626 reg_l2 10.258695
loss 1.9952459
STEP 111 ================================
prereg loss 0.7814993 reg_l1 12.132068 reg_l2 10.259561
loss 1.9947062
STEP 112 ================================
prereg loss 0.7809106 reg_l1 12.132512 reg_l2 10.2604265
loss 1.9941618
STEP 113 ================================
prereg loss 0.78032714 reg_l1 12.132956 reg_l2 10.261293
loss 1.9936228
STEP 114 ================================
prereg loss 0.77974284 reg_l1 12.133401 reg_l2 10.262164
loss 1.993083
STEP 115 ================================
prereg loss 0.77915937 reg_l1 12.133844 reg_l2 10.263032
loss 1.9925439
STEP 116 ================================
prereg loss 0.778578 reg_l1 12.134289 reg_l2 10.2639
loss 1.9920068
STEP 117 ================================
prereg loss 0.77799976 reg_l1 12.134735 reg_l2 10.264771
loss 1.9914733
STEP 118 ================================
prereg loss 0.7774189 reg_l1 12.135181 reg_l2 10.265641
loss 1.990937
STEP 119 ================================
prereg loss 0.7768462 reg_l1 12.135628 reg_l2 10.266514
loss 1.990409
STEP 120 ================================
prereg loss 0.7762691 reg_l1 12.136076 reg_l2 10.267387
loss 1.9898767
STEP 121 ================================
prereg loss 0.7756956 reg_l1 12.136523 reg_l2 10.268261
loss 1.9893479
STEP 122 ================================
prereg loss 0.7751231 reg_l1 12.136971 reg_l2 10.2691345
loss 1.9888203
STEP 123 ================================
prereg loss 0.77455235 reg_l1 12.137421 reg_l2 10.27001
loss 1.9882945
STEP 124 ================================
prereg loss 0.77397877 reg_l1 12.137869 reg_l2 10.270884
loss 1.9877658
STEP 125 ================================
prereg loss 0.7734149 reg_l1 12.138319 reg_l2 10.271762
loss 1.9872468
STEP 126 ================================
prereg loss 0.7728469 reg_l1 12.138769 reg_l2 10.272637
loss 1.9867239
STEP 127 ================================
prereg loss 0.77228135 reg_l1 12.139219 reg_l2 10.273516
loss 1.9862032
STEP 128 ================================
prereg loss 0.77171516 reg_l1 12.139669 reg_l2 10.274395
loss 1.9856821
STEP 129 ================================
prereg loss 0.77115303 reg_l1 12.1401205 reg_l2 10.275271
loss 1.9851651
STEP 130 ================================
prereg loss 0.77059233 reg_l1 12.140573 reg_l2 10.276152
loss 1.9846497
STEP 131 ================================
prereg loss 0.77003336 reg_l1 12.141026 reg_l2 10.277033
loss 1.984136
STEP 132 ================================
prereg loss 0.7694731 reg_l1 12.1414795 reg_l2 10.277913
loss 1.983621
STEP 133 ================================
prereg loss 0.7689163 reg_l1 12.141932 reg_l2 10.278796
loss 1.9831095
STEP 134 ================================
prereg loss 0.7683576 reg_l1 12.142385 reg_l2 10.279677
loss 1.9825962
STEP 135 ================================
prereg loss 0.7678028 reg_l1 12.1428385 reg_l2 10.28056
loss 1.9820867
STEP 136 ================================
prereg loss 0.76725054 reg_l1 12.143294 reg_l2 10.281444
loss 1.98158
STEP 137 ================================
prereg loss 0.76669997 reg_l1 12.14375 reg_l2 10.282329
loss 1.981075
STEP 138 ================================
prereg loss 0.76614803 reg_l1 12.144204 reg_l2 10.283215
loss 1.9805684
STEP 139 ================================
prereg loss 0.76559705 reg_l1 12.144659 reg_l2 10.284099
loss 1.980063
STEP 140 ================================
prereg loss 0.7650491 reg_l1 12.145116 reg_l2 10.284985
loss 1.9795607
STEP 141 ================================
prereg loss 0.7645002 reg_l1 12.145572 reg_l2 10.285873
loss 1.9790573
STEP 142 ================================
prereg loss 0.76395744 reg_l1 12.14603 reg_l2 10.286761
loss 1.9785604
STEP 143 ================================
prereg loss 0.76341194 reg_l1 12.146488 reg_l2 10.28765
loss 1.9780607
STEP 144 ================================
prereg loss 0.7628699 reg_l1 12.146944 reg_l2 10.288537
loss 1.9775643
STEP 145 ================================
prereg loss 0.76232725 reg_l1 12.147403 reg_l2 10.289427
loss 1.9770675
STEP 146 ================================
prereg loss 0.7617881 reg_l1 12.147862 reg_l2 10.2903185
loss 1.9765744
STEP 147 ================================
prereg loss 0.76125026 reg_l1 12.14832 reg_l2 10.291212
loss 1.9760823
STEP 148 ================================
prereg loss 0.7607135 reg_l1 12.148781 reg_l2 10.292103
loss 1.9755917
STEP 149 ================================
prereg loss 0.7601755 reg_l1 12.14924 reg_l2 10.2929945
loss 1.9750996
STEP 150 ================================
prereg loss 0.75964415 reg_l1 12.149699 reg_l2 10.293886
loss 1.9746141
STEP 151 ================================
prereg loss 0.75910854 reg_l1 12.15016 reg_l2 10.294781
loss 1.9741246
STEP 152 ================================
prereg loss 0.75857526 reg_l1 12.150622 reg_l2 10.295676
loss 1.9736376
STEP 153 ================================
prereg loss 0.7580426 reg_l1 12.151083 reg_l2 10.29657
loss 1.9731508
STEP 154 ================================
prereg loss 0.75751597 reg_l1 12.151543 reg_l2 10.297464
loss 1.9726703
STEP 155 ================================
prereg loss 0.756988 reg_l1 12.152003 reg_l2 10.298362
loss 1.9721882
STEP 156 ================================
prereg loss 0.7564613 reg_l1 12.152468 reg_l2 10.299258
loss 1.971708
STEP 157 ================================
prereg loss 0.7559349 reg_l1 12.15293 reg_l2 10.30016
loss 1.9712279
STEP 158 ================================
prereg loss 0.7554114 reg_l1 12.153394 reg_l2 10.301057
loss 1.9707508
STEP 159 ================================
prereg loss 0.7548893 reg_l1 12.153855 reg_l2 10.301952
loss 1.9702749
STEP 160 ================================
prereg loss 0.7543679 reg_l1 12.154319 reg_l2 10.302852
loss 1.9697998
STEP 161 ================================
prereg loss 0.7538486 reg_l1 12.154784 reg_l2 10.303755
loss 1.969327
STEP 162 ================================
prereg loss 0.7533268 reg_l1 12.15525 reg_l2 10.304657
loss 1.9688518
STEP 163 ================================
prereg loss 0.75281143 reg_l1 12.155714 reg_l2 10.305557
loss 1.9683828
STEP 164 ================================
prereg loss 0.7522957 reg_l1 12.156176 reg_l2 10.306458
loss 1.9679132
STEP 165 ================================
prereg loss 0.75178087 reg_l1 12.156642 reg_l2 10.30736
loss 1.9674451
STEP 166 ================================
prereg loss 0.75126785 reg_l1 12.157109 reg_l2 10.308267
loss 1.9669788
STEP 167 ================================
prereg loss 0.7507556 reg_l1 12.157575 reg_l2 10.30917
loss 1.9665132
STEP 168 ================================
prereg loss 0.7502432 reg_l1 12.15804 reg_l2 10.310074
loss 1.9660472
STEP 169 ================================
prereg loss 0.7497341 reg_l1 12.158506 reg_l2 10.31098
loss 1.9655848
STEP 170 ================================
prereg loss 0.7492275 reg_l1 12.158973 reg_l2 10.311885
loss 1.9651248
STEP 171 ================================
prereg loss 0.74871904 reg_l1 12.15944 reg_l2 10.312792
loss 1.964663
STEP 172 ================================
prereg loss 0.7482138 reg_l1 12.159908 reg_l2 10.313698
loss 1.9642048
STEP 173 ================================
prereg loss 0.7477105 reg_l1 12.160375 reg_l2 10.314607
loss 1.963748
STEP 174 ================================
prereg loss 0.7472061 reg_l1 12.160844 reg_l2 10.3155155
loss 1.9632905
STEP 175 ================================
prereg loss 0.7467049 reg_l1 12.161312 reg_l2 10.316424
loss 1.962836
STEP 176 ================================
prereg loss 0.7462037 reg_l1 12.161779 reg_l2 10.317333
loss 1.9623816
STEP 177 ================================
prereg loss 0.74570626 reg_l1 12.162249 reg_l2 10.318246
loss 1.9619312
STEP 178 ================================
prereg loss 0.74520606 reg_l1 12.162719 reg_l2 10.319156
loss 1.961478
STEP 179 ================================
prereg loss 0.74471074 reg_l1 12.163187 reg_l2 10.320067
loss 1.9610295
STEP 180 ================================
prereg loss 0.7442154 reg_l1 12.163658 reg_l2 10.320982
loss 1.9605813
STEP 181 ================================
prereg loss 0.7437236 reg_l1 12.164125 reg_l2 10.321893
loss 1.9601362
STEP 182 ================================
prereg loss 0.7432286 reg_l1 12.164597 reg_l2 10.322806
loss 1.9596882
STEP 183 ================================
prereg loss 0.74273753 reg_l1 12.165068 reg_l2 10.32372
loss 1.9592444
STEP 184 ================================
prereg loss 0.7422477 reg_l1 12.165539 reg_l2 10.324635
loss 1.9588016
STEP 185 ================================
prereg loss 0.7417601 reg_l1 12.166007 reg_l2 10.325549
loss 1.9583609
STEP 186 ================================
prereg loss 0.7412743 reg_l1 12.166479 reg_l2 10.326465
loss 1.9579222
STEP 187 ================================
prereg loss 0.7407885 reg_l1 12.166951 reg_l2 10.32738
loss 1.9574838
STEP 188 ================================
prereg loss 0.7403057 reg_l1 12.167422 reg_l2 10.328299
loss 1.9570479
STEP 189 ================================
prereg loss 0.73981935 reg_l1 12.167894 reg_l2 10.329219
loss 1.9566088
STEP 190 ================================
prereg loss 0.7393366 reg_l1 12.168368 reg_l2 10.330135
loss 1.9561734
STEP 191 ================================
prereg loss 0.73885477 reg_l1 12.168839 reg_l2 10.331056
loss 1.9557388
STEP 192 ================================
prereg loss 0.7383778 reg_l1 12.169311 reg_l2 10.331973
loss 1.9553089
STEP 193 ================================
prereg loss 0.7378996 reg_l1 12.169784 reg_l2 10.332893
loss 1.9548781
STEP 194 ================================
prereg loss 0.73742276 reg_l1 12.170258 reg_l2 10.333815
loss 1.9544485
STEP 195 ================================
prereg loss 0.7369436 reg_l1 12.170731 reg_l2 10.334737
loss 1.9540167
STEP 196 ================================
prereg loss 0.73647237 reg_l1 12.171204 reg_l2 10.335659
loss 1.9535928
STEP 197 ================================
prereg loss 0.7359982 reg_l1 12.171678 reg_l2 10.33658
loss 1.953166
STEP 198 ================================
prereg loss 0.7355277 reg_l1 12.172151 reg_l2 10.337503
loss 1.9527428
STEP 199 ================================
prereg loss 0.735057 reg_l1 12.172626 reg_l2 10.3384285
loss 1.9523196
STEP 200 ================================
prereg loss 0.7345875 reg_l1 12.1730995 reg_l2 10.339353
loss 1.9518974
STEP 201 ================================
prereg loss 0.73411965 reg_l1 12.173574 reg_l2 10.340276
loss 1.9514772
STEP 202 ================================
prereg loss 0.733652 reg_l1 12.17405 reg_l2 10.341203
loss 1.9510571
STEP 203 ================================
prereg loss 0.733188 reg_l1 12.174524 reg_l2 10.342131
loss 1.9506404
STEP 204 ================================
prereg loss 0.7327246 reg_l1 12.174999 reg_l2 10.343056
loss 1.9502246
STEP 205 ================================
prereg loss 0.7322605 reg_l1 12.175473 reg_l2 10.343985
loss 1.9498079
STEP 206 ================================
prereg loss 0.7318009 reg_l1 12.17595 reg_l2 10.34491
loss 1.9493959
STEP 207 ================================
prereg loss 0.73134166 reg_l1 12.176425 reg_l2 10.3458395
loss 1.9489841
STEP 208 ================================
prereg loss 0.73088366 reg_l1 12.176902 reg_l2 10.34677
loss 1.9485738
STEP 209 ================================
prereg loss 0.7304236 reg_l1 12.177377 reg_l2 10.347699
loss 1.9481614
STEP 210 ================================
prereg loss 0.7299688 reg_l1 12.177854 reg_l2 10.34863
loss 1.9477541
STEP 211 ================================
prereg loss 0.7295131 reg_l1 12.178329 reg_l2 10.349559
loss 1.947346
STEP 212 ================================
prereg loss 0.72905785 reg_l1 12.178808 reg_l2 10.350491
loss 1.9469388
STEP 213 ================================
prereg loss 0.72860736 reg_l1 12.179286 reg_l2 10.351424
loss 1.9465361
STEP 214 ================================
prereg loss 0.7281541 reg_l1 12.179761 reg_l2 10.352356
loss 1.9461303
STEP 215 ================================
prereg loss 0.72770387 reg_l1 12.180237 reg_l2 10.353291
loss 1.9457276
STEP 216 ================================
prereg loss 0.727257 reg_l1 12.1807165 reg_l2 10.354224
loss 1.9453287
STEP 217 ================================
prereg loss 0.7268079 reg_l1 12.181193 reg_l2 10.355159
loss 1.9449272
STEP 218 ================================
prereg loss 0.72636247 reg_l1 12.18167 reg_l2 10.356093
loss 1.9445295
STEP 219 ================================
prereg loss 0.7259171 reg_l1 12.182149 reg_l2 10.357029
loss 1.944132
STEP 220 ================================
prereg loss 0.7254747 reg_l1 12.182626 reg_l2 10.357965
loss 1.9437373
STEP 221 ================================
prereg loss 0.7250324 reg_l1 12.183104 reg_l2 10.358902
loss 1.9433427
STEP 222 ================================
prereg loss 0.7245898 reg_l1 12.183583 reg_l2 10.35984
loss 1.9429482
STEP 223 ================================
prereg loss 0.72415024 reg_l1 12.184062 reg_l2 10.360779
loss 1.9425564
STEP 224 ================================
prereg loss 0.72371316 reg_l1 12.18454 reg_l2 10.361716
loss 1.9421672
STEP 225 ================================
prereg loss 0.72327536 reg_l1 12.185019 reg_l2 10.362657
loss 1.9417772
STEP 226 ================================
prereg loss 0.7228375 reg_l1 12.185498 reg_l2 10.363596
loss 1.9413874
STEP 227 ================================
prereg loss 0.7224019 reg_l1 12.185978 reg_l2 10.364536
loss 1.9409997
STEP 228 ================================
prereg loss 0.72196764 reg_l1 12.186457 reg_l2 10.365477
loss 1.9406133
STEP 229 ================================
prereg loss 0.72153616 reg_l1 12.186936 reg_l2 10.366419
loss 1.9402298
STEP 230 ================================
prereg loss 0.72110593 reg_l1 12.187415 reg_l2 10.367361
loss 1.9398475
STEP 231 ================================
prereg loss 0.72067827 reg_l1 12.187894 reg_l2 10.368303
loss 1.9394677
STEP 232 ================================
prereg loss 0.72024745 reg_l1 12.1883745 reg_l2 10.3692465
loss 1.939085
STEP 233 ================================
prereg loss 0.7198202 reg_l1 12.188854 reg_l2 10.370192
loss 1.9387057
STEP 234 ================================
prereg loss 0.71939534 reg_l1 12.189334 reg_l2 10.371136
loss 1.9383287
STEP 235 ================================
prereg loss 0.7189705 reg_l1 12.189815 reg_l2 10.372081
loss 1.937952
STEP 236 ================================
prereg loss 0.7185464 reg_l1 12.190294 reg_l2 10.373027
loss 1.9375758
STEP 237 ================================
prereg loss 0.71812475 reg_l1 12.190776 reg_l2 10.373975
loss 1.9372023
STEP 238 ================================
prereg loss 0.7177014 reg_l1 12.191256 reg_l2 10.37492
loss 1.936827
STEP 239 ================================
prereg loss 0.71728104 reg_l1 12.191737 reg_l2 10.375865
loss 1.9364548
STEP 240 ================================
prereg loss 0.71686286 reg_l1 12.192218 reg_l2 10.376814
loss 1.9360847
STEP 241 ================================
prereg loss 0.7164473 reg_l1 12.192698 reg_l2 10.377763
loss 1.9357171
STEP 242 ================================
prereg loss 0.7160292 reg_l1 12.193179 reg_l2 10.378711
loss 1.9353471
STEP 243 ================================
prereg loss 0.7156149 reg_l1 12.193662 reg_l2 10.379661
loss 1.9349811
STEP 244 ================================
prereg loss 0.71520084 reg_l1 12.194141 reg_l2 10.380613
loss 1.9346149
STEP 245 ================================
prereg loss 0.7147863 reg_l1 12.194624 reg_l2 10.381563
loss 1.9342487
STEP 246 ================================
prereg loss 0.71437794 reg_l1 12.195105 reg_l2 10.382515
loss 1.9338884
STEP 247 ================================
prereg loss 0.71396726 reg_l1 12.195585 reg_l2 10.383465
loss 1.9335258
STEP 248 ================================
prereg loss 0.7135584 reg_l1 12.196068 reg_l2 10.384419
loss 1.9331651
STEP 249 ================================
prereg loss 0.7131495 reg_l1 12.196549 reg_l2 10.385373
loss 1.9328043
STEP 250 ================================
prereg loss 0.7127438 reg_l1 12.197032 reg_l2 10.386327
loss 1.932447
STEP 251 ================================
prereg loss 0.7123377 reg_l1 12.197512 reg_l2 10.387279
loss 1.9320889
STEP 252 ================================
prereg loss 0.71193427 reg_l1 12.197995 reg_l2 10.388235
loss 1.9317338
STEP 253 ================================
prereg loss 0.7115309 reg_l1 12.198478 reg_l2 10.38919
loss 1.9313787
STEP 254 ================================
prereg loss 0.71112925 reg_l1 12.198961 reg_l2 10.390146
loss 1.9310255
STEP 255 ================================
prereg loss 0.71072894 reg_l1 12.199442 reg_l2 10.391102
loss 1.9306731
STEP 256 ================================
prereg loss 0.7103304 reg_l1 12.199925 reg_l2 10.39206
loss 1.9303229
STEP 257 ================================
prereg loss 0.7099337 reg_l1 12.200406 reg_l2 10.393018
loss 1.9299743
STEP 258 ================================
prereg loss 0.7095362 reg_l1 12.20089 reg_l2 10.393978
loss 1.9296252
STEP 259 ================================
prereg loss 0.70914096 reg_l1 12.201372 reg_l2 10.394937
loss 1.9292781
STEP 260 ================================
prereg loss 0.70874476 reg_l1 12.201855 reg_l2 10.395896
loss 1.9289303
STEP 261 ================================
prereg loss 0.70835114 reg_l1 12.202338 reg_l2 10.396855
loss 1.9285849
STEP 262 ================================
prereg loss 0.7079615 reg_l1 12.202822 reg_l2 10.397817
loss 1.9282436
STEP 263 ================================
prereg loss 0.70756966 reg_l1 12.203304 reg_l2 10.398778
loss 1.9279001
STEP 264 ================================
prereg loss 0.7071817 reg_l1 12.203788 reg_l2 10.399738
loss 1.9275604
STEP 265 ================================
prereg loss 0.7067916 reg_l1 12.20427 reg_l2 10.4007015
loss 1.9272187
STEP 266 ================================
prereg loss 0.706405 reg_l1 12.204753 reg_l2 10.401665
loss 1.9268804
STEP 267 ================================
prereg loss 0.7060195 reg_l1 12.205236 reg_l2 10.402628
loss 1.9265432
STEP 268 ================================
prereg loss 0.70563424 reg_l1 12.205722 reg_l2 10.403594
loss 1.9262065
STEP 269 ================================
prereg loss 0.70524967 reg_l1 12.206205 reg_l2 10.404557
loss 1.9258702
STEP 270 ================================
prereg loss 0.70486665 reg_l1 12.206688 reg_l2 10.405524
loss 1.9255354
STEP 271 ================================
prereg loss 0.704487 reg_l1 12.2071705 reg_l2 10.40649
loss 1.925204
STEP 272 ================================
prereg loss 0.704107 reg_l1 12.207655 reg_l2 10.407455
loss 1.9248724
STEP 273 ================================
prereg loss 0.7037272 reg_l1 12.208139 reg_l2 10.408422
loss 1.9245412
STEP 274 ================================
prereg loss 0.70335084 reg_l1 12.208622 reg_l2 10.40939
loss 1.9242132
STEP 275 ================================
prereg loss 0.702976 reg_l1 12.209106 reg_l2 10.41036
loss 1.9238867
STEP 276 ================================
prereg loss 0.70259696 reg_l1 12.20959 reg_l2 10.411327
loss 1.9235561
STEP 277 ================================
prereg loss 0.70222723 reg_l1 12.210076 reg_l2 10.412298
loss 1.9232349
STEP 278 ================================
prereg loss 0.70185155 reg_l1 12.210559 reg_l2 10.413267
loss 1.9229074
STEP 279 ================================
prereg loss 0.70147985 reg_l1 12.211043 reg_l2 10.414239
loss 1.9225843
STEP 280 ================================
prereg loss 0.7011103 reg_l1 12.211528 reg_l2 10.41521
loss 1.9222631
STEP 281 ================================
prereg loss 0.7007406 reg_l1 12.212012 reg_l2 10.416181
loss 1.9219419
STEP 282 ================================
prereg loss 0.7003726 reg_l1 12.212495 reg_l2 10.417152
loss 1.921622
STEP 283 ================================
prereg loss 0.7000048 reg_l1 12.21298 reg_l2 10.418128
loss 1.9213029
STEP 284 ================================
prereg loss 0.69964087 reg_l1 12.213465 reg_l2 10.4191
loss 1.9209874
STEP 285 ================================
prereg loss 0.6992754 reg_l1 12.21395 reg_l2 10.420074
loss 1.9206704
STEP 286 ================================
prereg loss 0.6989124 reg_l1 12.214434 reg_l2 10.421049
loss 1.9203558
STEP 287 ================================
prereg loss 0.6985485 reg_l1 12.214919 reg_l2 10.422026
loss 1.9200404
STEP 288 ================================
prereg loss 0.69819075 reg_l1 12.2154045 reg_l2 10.423001
loss 1.9197311
STEP 289 ================================
prereg loss 0.6978309 reg_l1 12.215888 reg_l2 10.423979
loss 1.9194198
STEP 290 ================================
prereg loss 0.69747114 reg_l1 12.216373 reg_l2 10.424953
loss 1.9191085
STEP 291 ================================
prereg loss 0.6971144 reg_l1 12.216858 reg_l2 10.425931
loss 1.9188001
STEP 292 ================================
prereg loss 0.69675535 reg_l1 12.217343 reg_l2 10.426911
loss 1.9184897
STEP 293 ================================
prereg loss 0.69640046 reg_l1 12.217829 reg_l2 10.427891
loss 1.9181833
STEP 294 ================================
prereg loss 0.6960467 reg_l1 12.218312 reg_l2 10.428869
loss 1.9178779
STEP 295 ================================
prereg loss 0.6956975 reg_l1 12.218797 reg_l2 10.429849
loss 1.9175773
STEP 296 ================================
prereg loss 0.69534445 reg_l1 12.219282 reg_l2 10.430828
loss 1.9172727
STEP 297 ================================
prereg loss 0.6949935 reg_l1 12.219768 reg_l2 10.431808
loss 1.9169703
STEP 298 ================================
prereg loss 0.6946438 reg_l1 12.220254 reg_l2 10.432793
loss 1.9166691
STEP 299 ================================
prereg loss 0.6942976 reg_l1 12.220738 reg_l2 10.433775
loss 1.9163716
STEP 300 ================================
prereg loss 0.6939475 reg_l1 12.221223 reg_l2 10.434755
loss 1.9160697
STEP 301 ================================
prereg loss 0.69360393 reg_l1 12.221708 reg_l2 10.4357395
loss 1.9157748
STEP 302 ================================
prereg loss 0.6932569 reg_l1 12.222195 reg_l2 10.436726
loss 1.9154763
STEP 303 ================================
prereg loss 0.69291604 reg_l1 12.222679 reg_l2 10.43771
loss 1.915184
STEP 304 ================================
prereg loss 0.6925721 reg_l1 12.223164 reg_l2 10.438693
loss 1.9148885
STEP 305 ================================
prereg loss 0.6922312 reg_l1 12.22365 reg_l2 10.439681
loss 1.9145962
STEP 306 ================================
prereg loss 0.69189084 reg_l1 12.224135 reg_l2 10.440668
loss 1.9143044
STEP 307 ================================
prereg loss 0.69154996 reg_l1 12.22462 reg_l2 10.441655
loss 1.914012
STEP 308 ================================
prereg loss 0.6912124 reg_l1 12.225105 reg_l2 10.442642
loss 1.913723
STEP 309 ================================
prereg loss 0.69087434 reg_l1 12.225591 reg_l2 10.44363
loss 1.9134334
STEP 310 ================================
prereg loss 0.6905396 reg_l1 12.226076 reg_l2 10.444619
loss 1.9131472
STEP 311 ================================
prereg loss 0.69020444 reg_l1 12.226563 reg_l2 10.445608
loss 1.9128609
STEP 312 ================================
prereg loss 0.68987185 reg_l1 12.227049 reg_l2 10.4466
loss 1.9125767
STEP 313 ================================
prereg loss 0.68953794 reg_l1 12.227534 reg_l2 10.447592
loss 1.9122913
STEP 314 ================================
prereg loss 0.68920606 reg_l1 12.22802 reg_l2 10.448582
loss 1.912008
STEP 315 ================================
prereg loss 0.6888769 reg_l1 12.228504 reg_l2 10.4495735
loss 1.9117274
STEP 316 ================================
prereg loss 0.6885496 reg_l1 12.228991 reg_l2 10.450566
loss 1.9114487
STEP 317 ================================
prereg loss 0.68822056 reg_l1 12.229477 reg_l2 10.451559
loss 1.9111683
STEP 318 ================================
prereg loss 0.6878934 reg_l1 12.229963 reg_l2 10.452553
loss 1.9108897
STEP 319 ================================
prereg loss 0.6875692 reg_l1 12.23045 reg_l2 10.4535475
loss 1.9106143
STEP 320 ================================
prereg loss 0.6872454 reg_l1 12.230934 reg_l2 10.454543
loss 1.9103389
STEP 321 ================================
prereg loss 0.68692064 reg_l1 12.2314205 reg_l2 10.45554
loss 1.9100627
STEP 322 ================================
prereg loss 0.6865995 reg_l1 12.231907 reg_l2 10.456535
loss 1.9097902
STEP 323 ================================
prereg loss 0.6862758 reg_l1 12.232391 reg_l2 10.457531
loss 1.9095149
STEP 324 ================================
prereg loss 0.68595845 reg_l1 12.232876 reg_l2 10.4585285
loss 1.909246
STEP 325 ================================
prereg loss 0.68563795 reg_l1 12.233362 reg_l2 10.459525
loss 1.9089742
STEP 326 ================================
prereg loss 0.6853183 reg_l1 12.2338505 reg_l2 10.4605255
loss 1.9087033
STEP 327 ================================
prereg loss 0.6850026 reg_l1 12.234337 reg_l2 10.461525
loss 1.9084363
STEP 328 ================================
prereg loss 0.68468964 reg_l1 12.234823 reg_l2 10.462526
loss 1.908172
STEP 329 ================================
prereg loss 0.6843732 reg_l1 12.235308 reg_l2 10.463526
loss 1.9079039
STEP 330 ================================
prereg loss 0.6840603 reg_l1 12.235793 reg_l2 10.464525
loss 1.9076395
STEP 331 ================================
prereg loss 0.68374777 reg_l1 12.23628 reg_l2 10.465528
loss 1.9073758
STEP 332 ================================
prereg loss 0.68343896 reg_l1 12.236765 reg_l2 10.466532
loss 1.9071155
STEP 333 ================================
prereg loss 0.6831275 reg_l1 12.237251 reg_l2 10.467535
loss 1.9068527
STEP 334 ================================
prereg loss 0.68281686 reg_l1 12.237738 reg_l2 10.468537
loss 1.9065907
STEP 335 ================================
prereg loss 0.68251127 reg_l1 12.238223 reg_l2 10.469542
loss 1.9063337
STEP 336 ================================
prereg loss 0.68220335 reg_l1 12.23871 reg_l2 10.470547
loss 1.9060745
STEP 337 ================================
prereg loss 0.68189883 reg_l1 12.239197 reg_l2 10.471551
loss 1.9058186
STEP 338 ================================
prereg loss 0.6815924 reg_l1 12.239682 reg_l2 10.472556
loss 1.9055607
STEP 339 ================================
prereg loss 0.681292 reg_l1 12.24017 reg_l2 10.473564
loss 1.905309
STEP 340 ================================
prereg loss 0.6809859 reg_l1 12.240656 reg_l2 10.474572
loss 1.9050516
STEP 341 ================================
prereg loss 0.6806891 reg_l1 12.241141 reg_l2 10.47558
loss 1.9048033
STEP 342 ================================
prereg loss 0.6803851 reg_l1 12.241627 reg_l2 10.476587
loss 1.9045478
STEP 343 ================================
prereg loss 0.6800874 reg_l1 12.242115 reg_l2 10.477597
loss 1.904299
STEP 344 ================================
prereg loss 0.67979 reg_l1 12.2426 reg_l2 10.478608
loss 1.9040501
STEP 345 ================================
prereg loss 0.67949384 reg_l1 12.243087 reg_l2 10.479617
loss 1.9038026
STEP 346 ================================
prereg loss 0.67919636 reg_l1 12.243571 reg_l2 10.480629
loss 1.9035535
STEP 347 ================================
prereg loss 0.67890364 reg_l1 12.244059 reg_l2 10.481639
loss 1.9033096
STEP 348 ================================
prereg loss 0.6786098 reg_l1 12.244545 reg_l2 10.482654
loss 1.9030643
STEP 349 ================================
prereg loss 0.6783163 reg_l1 12.245032 reg_l2 10.483665
loss 1.9028196
STEP 350 ================================
prereg loss 0.6780245 reg_l1 12.24552 reg_l2 10.484678
loss 1.9025764
STEP 351 ================================
prereg loss 0.67773366 reg_l1 12.246005 reg_l2 10.485692
loss 1.9023342
STEP 352 ================================
prereg loss 0.6774446 reg_l1 12.246491 reg_l2 10.486709
loss 1.9020938
STEP 353 ================================
prereg loss 0.67715853 reg_l1 12.246978 reg_l2 10.487724
loss 1.9018564
STEP 354 ================================
prereg loss 0.6768693 reg_l1 12.247463 reg_l2 10.488741
loss 1.9016156
STEP 355 ================================
prereg loss 0.6765826 reg_l1 12.247949 reg_l2 10.489755
loss 1.9013774
STEP 356 ================================
prereg loss 0.67629844 reg_l1 12.248437 reg_l2 10.490776
loss 1.9011421
STEP 357 ================================
prereg loss 0.67601323 reg_l1 12.248923 reg_l2 10.491793
loss 1.9009056
STEP 358 ================================
prereg loss 0.6757317 reg_l1 12.249411 reg_l2 10.492813
loss 1.9006729
STEP 359 ================================
prereg loss 0.67544734 reg_l1 12.249895 reg_l2 10.493831
loss 1.9004369
STEP 360 ================================
prereg loss 0.6751655 reg_l1 12.250382 reg_l2 10.49485
loss 1.9002037
STEP 361 ================================
prereg loss 0.67488754 reg_l1 12.25087 reg_l2 10.4958725
loss 1.8999746
STEP 362 ================================
prereg loss 0.67460656 reg_l1 12.251356 reg_l2 10.496894
loss 1.8997422
STEP 363 ================================
prereg loss 0.6743304 reg_l1 12.2518425 reg_l2 10.497914
loss 1.8995147
STEP 364 ================================
prereg loss 0.6740514 reg_l1 12.252329 reg_l2 10.498938
loss 1.8992844
STEP 365 ================================
prereg loss 0.6737764 reg_l1 12.252815 reg_l2 10.499962
loss 1.899058
STEP 366 ================================
prereg loss 0.67350274 reg_l1 12.253304 reg_l2 10.500987
loss 1.898833
STEP 367 ================================
prereg loss 0.67322844 reg_l1 12.25379 reg_l2 10.502011
loss 1.8986075
STEP 368 ================================
prereg loss 0.6729594 reg_l1 12.254276 reg_l2 10.5030365
loss 1.898387
STEP 369 ================================
prereg loss 0.6726852 reg_l1 12.254762 reg_l2 10.504061
loss 1.8981614
STEP 370 ================================
prereg loss 0.67241687 reg_l1 12.255247 reg_l2 10.505089
loss 1.8979416
STEP 371 ================================
prereg loss 0.67214555 reg_l1 12.255734 reg_l2 10.506116
loss 1.8977189
STEP 372 ================================
prereg loss 0.6718794 reg_l1 12.256223 reg_l2 10.507144
loss 1.8975017
STEP 373 ================================
prereg loss 0.6716106 reg_l1 12.256708 reg_l2 10.508172
loss 1.8972814
STEP 374 ================================
prereg loss 0.6713439 reg_l1 12.257196 reg_l2 10.509203
loss 1.8970636
STEP 375 ================================
prereg loss 0.6710784 reg_l1 12.257681 reg_l2 10.510233
loss 1.8968465
STEP 376 ================================
prereg loss 0.67081434 reg_l1 12.258169 reg_l2 10.511262
loss 1.8966312
STEP 377 ================================
prereg loss 0.67055047 reg_l1 12.258656 reg_l2 10.512294
loss 1.8964161
STEP 378 ================================
prereg loss 0.6702882 reg_l1 12.259141 reg_l2 10.513326
loss 1.8962023
STEP 379 ================================
prereg loss 0.67002773 reg_l1 12.259629 reg_l2 10.514359
loss 1.8959907
STEP 380 ================================
prereg loss 0.6697677 reg_l1 12.260115 reg_l2 10.515392
loss 1.8957791
STEP 381 ================================
prereg loss 0.66950935 reg_l1 12.260603 reg_l2 10.516427
loss 1.8955696
STEP 382 ================================
prereg loss 0.6692502 reg_l1 12.261089 reg_l2 10.517461
loss 1.895359
STEP 383 ================================
prereg loss 0.6689943 reg_l1 12.261575 reg_l2 10.518497
loss 1.8951519
STEP 384 ================================
prereg loss 0.66873723 reg_l1 12.262064 reg_l2 10.519533
loss 1.8949437
STEP 385 ================================
prereg loss 0.6684828 reg_l1 12.262549 reg_l2 10.520569
loss 1.8947377
STEP 386 ================================
prereg loss 0.6682297 reg_l1 12.263036 reg_l2 10.5216055
loss 1.8945333
STEP 387 ================================
prereg loss 0.6679756 reg_l1 12.263523 reg_l2 10.522643
loss 1.8943279
STEP 388 ================================
prereg loss 0.6677241 reg_l1 12.264009 reg_l2 10.523684
loss 1.894125
STEP 389 ================================
prereg loss 0.6674749 reg_l1 12.264497 reg_l2 10.524723
loss 1.8939247
STEP 390 ================================
prereg loss 0.667224 reg_l1 12.264984 reg_l2 10.525764
loss 1.8937225
STEP 391 ================================
prereg loss 0.66697484 reg_l1 12.265471 reg_l2 10.526803
loss 1.893522
STEP 392 ================================
prereg loss 0.6667271 reg_l1 12.265957 reg_l2 10.527846
loss 1.893323
STEP 393 ================================
prereg loss 0.6664789 reg_l1 12.266445 reg_l2 10.52889
loss 1.8931234
STEP 394 ================================
prereg loss 0.6662331 reg_l1 12.2669325 reg_l2 10.529931
loss 1.8929265
STEP 395 ================================
prereg loss 0.6659853 reg_l1 12.26742 reg_l2 10.530973
loss 1.8927274
STEP 396 ================================
prereg loss 0.66574293 reg_l1 12.267906 reg_l2 10.532018
loss 1.8925335
STEP 397 ================================
prereg loss 0.6654979 reg_l1 12.268394 reg_l2 10.533067
loss 1.8923373
STEP 398 ================================
prereg loss 0.66525644 reg_l1 12.268881 reg_l2 10.534112
loss 1.8921444
STEP 399 ================================
prereg loss 0.66501606 reg_l1 12.269367 reg_l2 10.535158
loss 1.8919528
STEP 400 ================================
prereg loss 0.66477615 reg_l1 12.269855 reg_l2 10.536204
loss 1.8917615
STEP 401 ================================
prereg loss 0.66453564 reg_l1 12.270342 reg_l2 10.537255
loss 1.8915699
STEP 402 ================================
prereg loss 0.66429764 reg_l1 12.27083 reg_l2 10.538302
loss 1.8913808
STEP 403 ================================
prereg loss 0.66406196 reg_l1 12.271316 reg_l2 10.539351
loss 1.8911936
STEP 404 ================================
prereg loss 0.6638264 reg_l1 12.271803 reg_l2 10.540401
loss 1.8910067
STEP 405 ================================
prereg loss 0.6635918 reg_l1 12.272291 reg_l2 10.541452
loss 1.890821
STEP 406 ================================
prereg loss 0.6633554 reg_l1 12.272779 reg_l2 10.542505
loss 1.8906333
STEP 407 ================================
prereg loss 0.6631251 reg_l1 12.273265 reg_l2 10.543556
loss 1.8904517
STEP 408 ================================
prereg loss 0.6628919 reg_l1 12.273753 reg_l2 10.54461
loss 1.8902674
STEP 409 ================================
prereg loss 0.66265935 reg_l1 12.2742405 reg_l2 10.545665
loss 1.8900833
STEP 410 ================================
prereg loss 0.6624289 reg_l1 12.274729 reg_l2 10.54672
loss 1.8899019
STEP 411 ================================
prereg loss 0.66220105 reg_l1 12.275216 reg_l2 10.547774
loss 1.8897227
STEP 412 ================================
prereg loss 0.66197413 reg_l1 12.275702 reg_l2 10.548828
loss 1.8895445
STEP 413 ================================
prereg loss 0.66174406 reg_l1 12.276192 reg_l2 10.549886
loss 1.8893633
STEP 414 ================================
prereg loss 0.66152 reg_l1 12.276678 reg_l2 10.550944
loss 1.8891878
STEP 415 ================================
prereg loss 0.66129506 reg_l1 12.277166 reg_l2 10.552002
loss 1.8890117
STEP 416 ================================
prereg loss 0.6610704 reg_l1 12.277654 reg_l2 10.5530615
loss 1.8888359
STEP 417 ================================
prereg loss 0.6608468 reg_l1 12.278141 reg_l2 10.554121
loss 1.8886609
STEP 418 ================================
prereg loss 0.6606226 reg_l1 12.278629 reg_l2 10.555183
loss 1.8884856
STEP 419 ================================
prereg loss 0.66040295 reg_l1 12.279116 reg_l2 10.556243
loss 1.8883145
STEP 420 ================================
prereg loss 0.6601818 reg_l1 12.279604 reg_l2 10.557305
loss 1.8881423
STEP 421 ================================
prereg loss 0.65996337 reg_l1 12.280092 reg_l2 10.558367
loss 1.8879726
STEP 422 ================================
prereg loss 0.6597457 reg_l1 12.280579 reg_l2 10.55943
loss 1.8878036
STEP 423 ================================
prereg loss 0.6595264 reg_l1 12.281067 reg_l2 10.560496
loss 1.8876331
STEP 424 ================================
prereg loss 0.65931135 reg_l1 12.281555 reg_l2 10.561561
loss 1.8874669
STEP 425 ================================
prereg loss 0.65909475 reg_l1 12.2820425 reg_l2 10.562627
loss 1.8872991
STEP 426 ================================
prereg loss 0.6588802 reg_l1 12.282531 reg_l2 10.563692
loss 1.8871334
STEP 427 ================================
prereg loss 0.6586655 reg_l1 12.283018 reg_l2 10.564759
loss 1.8869674
STEP 428 ================================
prereg loss 0.6584539 reg_l1 12.283505 reg_l2 10.565826
loss 1.8868043
STEP 429 ================================
prereg loss 0.65824026 reg_l1 12.283996 reg_l2 10.5668955
loss 1.8866398
STEP 430 ================================
prereg loss 0.6580319 reg_l1 12.284483 reg_l2 10.5679655
loss 1.8864801
STEP 431 ================================
prereg loss 0.657819 reg_l1 12.28497 reg_l2 10.569035
loss 1.8863161
STEP 432 ================================
prereg loss 0.65761125 reg_l1 12.285458 reg_l2 10.570107
loss 1.886157
STEP 433 ================================
prereg loss 0.6574009 reg_l1 12.285948 reg_l2 10.571179
loss 1.8859956
STEP 434 ================================
prereg loss 0.65719503 reg_l1 12.286436 reg_l2 10.57225
loss 1.8858387
STEP 435 ================================
prereg loss 0.6569898 reg_l1 12.286923 reg_l2 10.573324
loss 1.8856822
STEP 436 ================================
prereg loss 0.65678436 reg_l1 12.287412 reg_l2 10.574398
loss 1.8855255
STEP 437 ================================
prereg loss 0.65658015 reg_l1 12.2879 reg_l2 10.575474
loss 1.8853703
STEP 438 ================================
prereg loss 0.6563766 reg_l1 12.288387 reg_l2 10.57655
loss 1.8852154
STEP 439 ================================
prereg loss 0.656173 reg_l1 12.288876 reg_l2 10.577624
loss 1.8850605
STEP 440 ================================
prereg loss 0.65597403 reg_l1 12.289364 reg_l2 10.578703
loss 1.8849105
STEP 441 ================================
prereg loss 0.6557734 reg_l1 12.289853 reg_l2 10.579782
loss 1.8847587
STEP 442 ================================
prereg loss 0.6555715 reg_l1 12.290341 reg_l2 10.58086
loss 1.8846056
STEP 443 ================================
prereg loss 0.6553741 reg_l1 12.290831 reg_l2 10.581938
loss 1.8844571
STEP 444 ================================
prereg loss 0.65517485 reg_l1 12.291318 reg_l2 10.583019
loss 1.8843067
STEP 445 ================================
prereg loss 0.6549787 reg_l1 12.291808 reg_l2 10.584101
loss 1.8841596
STEP 446 ================================
prereg loss 0.6547832 reg_l1 12.292296 reg_l2 10.585182
loss 1.8840129
STEP 447 ================================
prereg loss 0.6545878 reg_l1 12.292784 reg_l2 10.5862665
loss 1.8838663
STEP 448 ================================
prereg loss 0.6543929 reg_l1 12.293274 reg_l2 10.587351
loss 1.8837204
STEP 449 ================================
prereg loss 0.6541986 reg_l1 12.293764 reg_l2 10.588436
loss 1.883575
STEP 450 ================================
prereg loss 0.6540081 reg_l1 12.2942505 reg_l2 10.589519
loss 1.8834331
STEP 451 ================================
prereg loss 0.6538134 reg_l1 12.294741 reg_l2 10.590606
loss 1.8832874
STEP 452 ================================
prereg loss 0.6536258 reg_l1 12.29523 reg_l2 10.591694
loss 1.8831489
STEP 453 ================================
prereg loss 0.65343595 reg_l1 12.295718 reg_l2 10.592781
loss 1.8830078
STEP 454 ================================
prereg loss 0.6532461 reg_l1 12.296208 reg_l2 10.593869
loss 1.8828669
STEP 455 ================================
prereg loss 0.6530578 reg_l1 12.296697 reg_l2 10.594959
loss 1.8827275
STEP 456 ================================
prereg loss 0.652873 reg_l1 12.297186 reg_l2 10.596049
loss 1.8825915
STEP 457 ================================
prereg loss 0.65268636 reg_l1 12.297675 reg_l2 10.597138
loss 1.8824539
STEP 458 ================================
prereg loss 0.65250015 reg_l1 12.298165 reg_l2 10.598232
loss 1.8823167
STEP 459 ================================
prereg loss 0.6523174 reg_l1 12.298655 reg_l2 10.599324
loss 1.8821828
STEP 460 ================================
prereg loss 0.65213275 reg_l1 12.299144 reg_l2 10.600419
loss 1.8820472
STEP 461 ================================
prereg loss 0.651949 reg_l1 12.299635 reg_l2 10.601514
loss 1.8819125
STEP 462 ================================
prereg loss 0.6517714 reg_l1 12.300123 reg_l2 10.602607
loss 1.8817837
STEP 463 ================================
prereg loss 0.6515895 reg_l1 12.300613 reg_l2 10.603705
loss 1.8816509
STEP 464 ================================
prereg loss 0.65141094 reg_l1 12.301102 reg_l2 10.604801
loss 1.8815211
STEP 465 ================================
prereg loss 0.651227 reg_l1 12.301592 reg_l2 10.6059
loss 1.8813862
STEP 466 ================================
prereg loss 0.6510521 reg_l1 12.302082 reg_l2 10.606996
loss 1.8812604
STEP 467 ================================
prereg loss 0.65087783 reg_l1 12.30257 reg_l2 10.608097
loss 1.8811349
STEP 468 ================================
prereg loss 0.6507009 reg_l1 12.303062 reg_l2 10.609197
loss 1.8810072
STEP 469 ================================
prereg loss 0.65052736 reg_l1 12.303553 reg_l2 10.6103
loss 1.8808826
STEP 470 ================================
prereg loss 0.6503498 reg_l1 12.304041 reg_l2 10.6114
loss 1.880754
STEP 471 ================================
prereg loss 0.6501775 reg_l1 12.304532 reg_l2 10.612503
loss 1.8806307
STEP 472 ================================
prereg loss 0.65000623 reg_l1 12.305022 reg_l2 10.613606
loss 1.8805084
STEP 473 ================================
prereg loss 0.6498336 reg_l1 12.305512 reg_l2 10.614711
loss 1.8803849
STEP 474 ================================
prereg loss 0.6496608 reg_l1 12.306005 reg_l2 10.615817
loss 1.8802613
STEP 475 ================================
prereg loss 0.6494914 reg_l1 12.306494 reg_l2 10.616924
loss 1.8801408
STEP 476 ================================
prereg loss 0.64932364 reg_l1 12.306985 reg_l2 10.61803
loss 1.880022
STEP 477 ================================
prereg loss 0.6491552 reg_l1 12.307474 reg_l2 10.619139
loss 1.8799026
STEP 478 ================================
prereg loss 0.648988 reg_l1 12.307965 reg_l2 10.620248
loss 1.8797846
STEP 479 ================================
prereg loss 0.6488204 reg_l1 12.308455 reg_l2 10.621357
loss 1.879666
STEP 480 ================================
prereg loss 0.6486569 reg_l1 12.308947 reg_l2 10.622468
loss 1.8795516
STEP 481 ================================
prereg loss 0.648492 reg_l1 12.309438 reg_l2 10.623581
loss 1.8794358
STEP 482 ================================
prereg loss 0.6483309 reg_l1 12.309927 reg_l2 10.624693
loss 1.8793236
STEP 483 ================================
prereg loss 0.6481663 reg_l1 12.31042 reg_l2 10.625804
loss 1.8792083
STEP 484 ================================
prereg loss 0.6480045 reg_l1 12.310911 reg_l2 10.626921
loss 1.8790956
STEP 485 ================================
prereg loss 0.64783937 reg_l1 12.311402 reg_l2 10.628034
loss 1.8789797
STEP 486 ================================
prereg loss 0.6476817 reg_l1 12.311893 reg_l2 10.62915
loss 1.8788711
STEP 487 ================================
prereg loss 0.6475223 reg_l1 12.312386 reg_l2 10.630268
loss 1.8787608
STEP 488 ================================
prereg loss 0.64736414 reg_l1 12.312877 reg_l2 10.631386
loss 1.8786519
STEP 489 ================================
prereg loss 0.64720577 reg_l1 12.313369 reg_l2 10.632503
loss 1.8785427
STEP 490 ================================
prereg loss 0.6470477 reg_l1 12.31386 reg_l2 10.633624
loss 1.8784337
STEP 491 ================================
prereg loss 0.6468926 reg_l1 12.314353 reg_l2 10.634744
loss 1.8783278
STEP 492 ================================
prereg loss 0.64673775 reg_l1 12.314844 reg_l2 10.635866
loss 1.8782222
STEP 493 ================================
prereg loss 0.64658207 reg_l1 12.315335 reg_l2 10.636988
loss 1.8781157
STEP 494 ================================
prereg loss 0.64642894 reg_l1 12.315828 reg_l2 10.638109
loss 1.8780118
STEP 495 ================================
prereg loss 0.6462782 reg_l1 12.316319 reg_l2 10.639235
loss 1.8779101
STEP 496 ================================
prereg loss 0.64612424 reg_l1 12.3168125 reg_l2 10.640362
loss 1.8778055
STEP 497 ================================
prereg loss 0.6459743 reg_l1 12.317304 reg_l2 10.641488
loss 1.8777046
STEP 498 ================================
prereg loss 0.64582276 reg_l1 12.317796 reg_l2 10.642612
loss 1.8776023
STEP 499 ================================
prereg loss 0.6456752 reg_l1 12.31829 reg_l2 10.643741
loss 1.8775042
STEP 500 ================================
prereg loss 0.6455269 reg_l1 12.318782 reg_l2 10.644872
loss 1.877405
2022-06-28T20:53:19.060

julia> open("sparse21-after-1000-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse21-after-1000-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse21-after-1000-steps-opt.ser", opt)

julia> steps!(1000)
2022-06-28T21:01:52.369
STEP 1 ================================
prereg loss 0.64537865 reg_l1 12.319275 reg_l2 10.646
loss 1.8773062
STEP 2 ================================
prereg loss 0.64523315 reg_l1 12.319767 reg_l2 10.64713
loss 1.8772099
STEP 3 ================================
prereg loss 0.6450879 reg_l1 12.320261 reg_l2 10.648262
loss 1.877114
STEP 4 ================================
prereg loss 0.64494175 reg_l1 12.320754 reg_l2 10.649396
loss 1.8770173
STEP 5 ================================
prereg loss 0.6447986 reg_l1 12.321246 reg_l2 10.650528
loss 1.8769233
STEP 6 ================================
prereg loss 0.6446534 reg_l1 12.321739 reg_l2 10.651663
loss 1.8768272
STEP 7 ================================
prereg loss 0.64451116 reg_l1 12.322233 reg_l2 10.652799
loss 1.8767345
STEP 8 ================================
prereg loss 0.64437073 reg_l1 12.322726 reg_l2 10.653934
loss 1.8766434
STEP 9 ================================
prereg loss 0.64422905 reg_l1 12.32322 reg_l2 10.655072
loss 1.8765512
STEP 10 ================================
prereg loss 0.64408827 reg_l1 12.323714 reg_l2 10.656208
loss 1.8764597
STEP 11 ================================
prereg loss 0.6439502 reg_l1 12.324208 reg_l2 10.65735
loss 1.876371
STEP 12 ================================
prereg loss 0.6438108 reg_l1 12.324701 reg_l2 10.658488
loss 1.876281
STEP 13 ================================
prereg loss 0.6436753 reg_l1 12.325195 reg_l2 10.659632
loss 1.8761948
STEP 14 ================================
prereg loss 0.6435377 reg_l1 12.325689 reg_l2 10.66077
loss 1.8761067
STEP 15 ================================
prereg loss 0.64340216 reg_l1 12.326183 reg_l2 10.661915
loss 1.8760204
STEP 16 ================================
prereg loss 0.6432677 reg_l1 12.326678 reg_l2 10.66306
loss 1.8759356
STEP 17 ================================
prereg loss 0.64313304 reg_l1 12.327172 reg_l2 10.664206
loss 1.8758503
STEP 18 ================================
prereg loss 0.6430002 reg_l1 12.327667 reg_l2 10.665351
loss 1.875767
STEP 19 ================================
prereg loss 0.6428677 reg_l1 12.328162 reg_l2 10.666499
loss 1.8756839
STEP 20 ================================
prereg loss 0.6427361 reg_l1 12.328655 reg_l2 10.6676445
loss 1.8756016
STEP 21 ================================
prereg loss 0.642605 reg_l1 12.32915 reg_l2 10.668794
loss 1.87552
STEP 22 ================================
prereg loss 0.6424758 reg_l1 12.329646 reg_l2 10.669943
loss 1.8754404
STEP 23 ================================
prereg loss 0.6423457 reg_l1 12.330143 reg_l2 10.671097
loss 1.87536
STEP 24 ================================
prereg loss 0.64221805 reg_l1 12.330637 reg_l2 10.672247
loss 1.8752818
STEP 25 ================================
prereg loss 0.64208966 reg_l1 12.331134 reg_l2 10.673399
loss 1.8752031
STEP 26 ================================
prereg loss 0.6419634 reg_l1 12.331629 reg_l2 10.674556
loss 1.8751264
STEP 27 ================================
prereg loss 0.64183646 reg_l1 12.332126 reg_l2 10.675712
loss 1.8750491
STEP 28 ================================
prereg loss 0.64171326 reg_l1 12.33262 reg_l2 10.676864
loss 1.8749752
STEP 29 ================================
prereg loss 0.6415872 reg_l1 12.333117 reg_l2 10.678022
loss 1.8748989
STEP 30 ================================
prereg loss 0.641462 reg_l1 12.333613 reg_l2 10.679182
loss 1.8748233
STEP 31 ================================
prereg loss 0.6413392 reg_l1 12.334109 reg_l2 10.680341
loss 1.8747501
STEP 32 ================================
prereg loss 0.6412169 reg_l1 12.334605 reg_l2 10.6814995
loss 1.8746774
STEP 33 ================================
prereg loss 0.64109755 reg_l1 12.3351 reg_l2 10.682659
loss 1.8746076
STEP 34 ================================
prereg loss 0.640976 reg_l1 12.335598 reg_l2 10.683825
loss 1.8745358
STEP 35 ================================
prereg loss 0.64085823 reg_l1 12.336095 reg_l2 10.684989
loss 1.8744678
STEP 36 ================================
prereg loss 0.6407375 reg_l1 12.336593 reg_l2 10.6861515
loss 1.8743968
STEP 37 ================================
prereg loss 0.6406206 reg_l1 12.337089 reg_l2 10.687315
loss 1.8743294
STEP 38 ================================
prereg loss 0.64050275 reg_l1 12.337586 reg_l2 10.68848
loss 1.8742614
STEP 39 ================================
prereg loss 0.64038706 reg_l1 12.338083 reg_l2 10.689649
loss 1.8741955
STEP 40 ================================
prereg loss 0.6402698 reg_l1 12.33858 reg_l2 10.690819
loss 1.8741279
STEP 41 ================================
prereg loss 0.6401556 reg_l1 12.339079 reg_l2 10.691988
loss 1.8740635
STEP 42 ================================
prereg loss 0.6400424 reg_l1 12.339577 reg_l2 10.693158
loss 1.8740001
STEP 43 ================================
prereg loss 0.63992745 reg_l1 12.340074 reg_l2 10.69433
loss 1.8739347
STEP 44 ================================
prereg loss 0.6398169 reg_l1 12.340572 reg_l2 10.695502
loss 1.8738742
STEP 45 ================================
prereg loss 0.6397056 reg_l1 12.341073 reg_l2 10.696674
loss 1.8738129
STEP 46 ================================
prereg loss 0.63959473 reg_l1 12.341569 reg_l2 10.697849
loss 1.8737516
STEP 47 ================================
prereg loss 0.6394839 reg_l1 12.342067 reg_l2 10.699025
loss 1.8736906
STEP 48 ================================
prereg loss 0.6393743 reg_l1 12.342566 reg_l2 10.700201
loss 1.873631
STEP 49 ================================
prereg loss 0.6392671 reg_l1 12.343064 reg_l2 10.701379
loss 1.8735735
STEP 50 ================================
prereg loss 0.6391619 reg_l1 12.343564 reg_l2 10.702558
loss 1.8735182
STEP 51 ================================
prereg loss 0.6390533 reg_l1 12.344065 reg_l2 10.703739
loss 1.8734598
STEP 52 ================================
prereg loss 0.63894886 reg_l1 12.3445635 reg_l2 10.70492
loss 1.8734052
STEP 53 ================================
prereg loss 0.63884264 reg_l1 12.345062 reg_l2 10.706101
loss 1.873349
STEP 54 ================================
prereg loss 0.6387376 reg_l1 12.345563 reg_l2 10.707287
loss 1.8732939
STEP 55 ================================
prereg loss 0.6386347 reg_l1 12.346063 reg_l2 10.708468
loss 1.873241
STEP 56 ================================
prereg loss 0.63852966 reg_l1 12.346562 reg_l2 10.709655
loss 1.8731859
STEP 57 ================================
prereg loss 0.6384291 reg_l1 12.347063 reg_l2 10.710843
loss 1.8731353
STEP 58 ================================
prereg loss 0.6383269 reg_l1 12.347565 reg_l2 10.712031
loss 1.8730834
STEP 59 ================================
prereg loss 0.6382255 reg_l1 12.3480625 reg_l2 10.713217
loss 1.8730319
STEP 60 ================================
prereg loss 0.6381262 reg_l1 12.348564 reg_l2 10.714407
loss 1.8729827
STEP 61 ================================
prereg loss 0.6380279 reg_l1 12.349064 reg_l2 10.715599
loss 1.8729343
STEP 62 ================================
prereg loss 0.63792956 reg_l1 12.349566 reg_l2 10.716793
loss 1.8728862
STEP 63 ================================
prereg loss 0.63783133 reg_l1 12.350066 reg_l2 10.717984
loss 1.872838
STEP 64 ================================
prereg loss 0.6377346 reg_l1 12.350568 reg_l2 10.719176
loss 1.8727913
STEP 65 ================================
prereg loss 0.63763934 reg_l1 12.351071 reg_l2 10.720374
loss 1.8727465
STEP 66 ================================
prereg loss 0.6375445 reg_l1 12.351573 reg_l2 10.72157
loss 1.8727019
STEP 67 ================================
prereg loss 0.6374503 reg_l1 12.352074 reg_l2 10.722769
loss 1.8726578
STEP 68 ================================
prereg loss 0.63735664 reg_l1 12.352574 reg_l2 10.723964
loss 1.8726141
STEP 69 ================================
prereg loss 0.637263 reg_l1 12.353078 reg_l2 10.725164
loss 1.8725708
STEP 70 ================================
prereg loss 0.63717127 reg_l1 12.35358 reg_l2 10.726369
loss 1.8725294
STEP 71 ================================
prereg loss 0.6370789 reg_l1 12.354084 reg_l2 10.727569
loss 1.8724873
STEP 72 ================================
prereg loss 0.6369868 reg_l1 12.354586 reg_l2 10.72877
loss 1.8724453
STEP 73 ================================
prereg loss 0.6368996 reg_l1 12.35509 reg_l2 10.729978
loss 1.8724086
STEP 74 ================================
prereg loss 0.6368089 reg_l1 12.355595 reg_l2 10.731185
loss 1.8723683
STEP 75 ================================
prereg loss 0.6367215 reg_l1 12.356096 reg_l2 10.732389
loss 1.8723311
STEP 76 ================================
prereg loss 0.6366343 reg_l1 12.3566 reg_l2 10.733595
loss 1.8722942
STEP 77 ================================
prereg loss 0.6365469 reg_l1 12.357104 reg_l2 10.734806
loss 1.8722575
STEP 78 ================================
prereg loss 0.6364611 reg_l1 12.35761 reg_l2 10.736018
loss 1.8722222
STEP 79 ================================
prereg loss 0.6363757 reg_l1 12.358112 reg_l2 10.737226
loss 1.8721869
STEP 80 ================================
prereg loss 0.6362906 reg_l1 12.358617 reg_l2 10.738437
loss 1.8721523
STEP 81 ================================
prereg loss 0.63620704 reg_l1 12.359122 reg_l2 10.739654
loss 1.8721192
STEP 82 ================================
prereg loss 0.6361248 reg_l1 12.35963 reg_l2 10.740871
loss 1.8720877
STEP 83 ================================
prereg loss 0.6360429 reg_l1 12.360133 reg_l2 10.742085
loss 1.8720562
STEP 84 ================================
prereg loss 0.6359615 reg_l1 12.360637 reg_l2 10.7432995
loss 1.8720253
STEP 85 ================================
prereg loss 0.63587946 reg_l1 12.361142 reg_l2 10.74452
loss 1.8719938
STEP 86 ================================
prereg loss 0.6357996 reg_l1 12.3616495 reg_l2 10.745742
loss 1.8719645
STEP 87 ================================
prereg loss 0.63572073 reg_l1 12.362155 reg_l2 10.746961
loss 1.8719362
STEP 88 ================================
prereg loss 0.63564193 reg_l1 12.362659 reg_l2 10.748183
loss 1.871908
STEP 89 ================================
prereg loss 0.63556427 reg_l1 12.363167 reg_l2 10.749408
loss 1.871881
STEP 90 ================================
prereg loss 0.63548774 reg_l1 12.363675 reg_l2 10.750632
loss 1.8718553
STEP 91 ================================
prereg loss 0.63541096 reg_l1 12.3641815 reg_l2 10.751859
loss 1.871829
STEP 92 ================================
prereg loss 0.63533455 reg_l1 12.364687 reg_l2 10.753083
loss 1.8718033
STEP 93 ================================
prereg loss 0.6352597 reg_l1 12.365194 reg_l2 10.754311
loss 1.8717792
STEP 94 ================================
prereg loss 0.63518417 reg_l1 12.365704 reg_l2 10.755543
loss 1.8717545
STEP 95 ================================
prereg loss 0.6351113 reg_l1 12.366211 reg_l2 10.756773
loss 1.8717325
STEP 96 ================================
prereg loss 0.6350381 reg_l1 12.366716 reg_l2 10.758003
loss 1.8717098
STEP 97 ================================
prereg loss 0.6349678 reg_l1 12.367225 reg_l2 10.759235
loss 1.8716903
STEP 98 ================================
prereg loss 0.6348954 reg_l1 12.367735 reg_l2 10.76047
loss 1.8716688
STEP 99 ================================
prereg loss 0.6348272 reg_l1 12.368243 reg_l2 10.761706
loss 1.8716516
STEP 100 ================================
prereg loss 0.6347583 reg_l1 12.3687525 reg_l2 10.762942
loss 1.8716335
STEP 101 ================================
prereg loss 0.63468754 reg_l1 12.36926 reg_l2 10.76418
loss 1.8716135
STEP 102 ================================
prereg loss 0.63462067 reg_l1 12.36977 reg_l2 10.765419
loss 1.8715976
STEP 103 ================================
prereg loss 0.6345528 reg_l1 12.370281 reg_l2 10.766661
loss 1.8715808
STEP 104 ================================
prereg loss 0.6344882 reg_l1 12.370789 reg_l2 10.767901
loss 1.8715671
STEP 105 ================================
prereg loss 0.63441926 reg_l1 12.371299 reg_l2 10.769143
loss 1.8715491
STEP 106 ================================
prereg loss 0.6343561 reg_l1 12.371809 reg_l2 10.770388
loss 1.871537
STEP 107 ================================
prereg loss 0.6342897 reg_l1 12.372319 reg_l2 10.771635
loss 1.8715217
STEP 108 ================================
prereg loss 0.63422686 reg_l1 12.37283 reg_l2 10.7728815
loss 1.87151
STEP 109 ================================
prereg loss 0.6341648 reg_l1 12.373342 reg_l2 10.774128
loss 1.871499
STEP 110 ================================
prereg loss 0.63410336 reg_l1 12.373853 reg_l2 10.775379
loss 1.8714886
STEP 111 ================================
prereg loss 0.6340404 reg_l1 12.374364 reg_l2 10.776629
loss 1.8714769
STEP 112 ================================
prereg loss 0.6339795 reg_l1 12.374876 reg_l2 10.777879
loss 1.8714671
STEP 113 ================================
prereg loss 0.6339204 reg_l1 12.375388 reg_l2 10.779134
loss 1.8714592
STEP 114 ================================
prereg loss 0.63385886 reg_l1 12.3759 reg_l2 10.780387
loss 1.871449
STEP 115 ================================
prereg loss 0.6338001 reg_l1 12.376412 reg_l2 10.781644
loss 1.8714414
STEP 116 ================================
prereg loss 0.6337427 reg_l1 12.376926 reg_l2 10.782901
loss 1.8714354
STEP 117 ================================
prereg loss 0.633686 reg_l1 12.377439 reg_l2 10.784159
loss 1.8714299
STEP 118 ================================
prereg loss 0.6336296 reg_l1 12.377952 reg_l2 10.7854185
loss 1.8714249
STEP 119 ================================
prereg loss 0.6335731 reg_l1 12.378467 reg_l2 10.786679
loss 1.8714199
STEP 120 ================================
prereg loss 0.633519 reg_l1 12.378979 reg_l2 10.787943
loss 1.8714168
STEP 121 ================================
prereg loss 0.6334633 reg_l1 12.379491 reg_l2 10.789205
loss 1.8714125
STEP 122 ================================
prereg loss 0.63341063 reg_l1 12.380007 reg_l2 10.790468
loss 1.8714113
STEP 123 ================================
prereg loss 0.63335675 reg_l1 12.380524 reg_l2 10.791737
loss 1.8714092
STEP 124 ================================
prereg loss 0.6333048 reg_l1 12.381038 reg_l2 10.793003
loss 1.8714085
STEP 125 ================================
prereg loss 0.6332525 reg_l1 12.381552 reg_l2 10.794271
loss 1.8714077
STEP 126 ================================
prereg loss 0.63320196 reg_l1 12.382068 reg_l2 10.795544
loss 1.8714087
STEP 127 ================================
prereg loss 0.6331497 reg_l1 12.382583 reg_l2 10.796815
loss 1.871408
STEP 128 ================================
prereg loss 0.6331025 reg_l1 12.3831 reg_l2 10.798087
loss 1.8714125
STEP 129 ================================
prereg loss 0.6330537 reg_l1 12.383615 reg_l2 10.799359
loss 1.8714151
STEP 130 ================================
prereg loss 0.633006 reg_l1 12.384131 reg_l2 10.800636
loss 1.8714192
STEP 131 ================================
prereg loss 0.63295937 reg_l1 12.384648 reg_l2 10.801912
loss 1.8714242
STEP 132 ================================
prereg loss 0.63291395 reg_l1 12.385165 reg_l2 10.803191
loss 1.8714305
STEP 133 ================================
prereg loss 0.6328669 reg_l1 12.385681 reg_l2 10.80447
loss 1.8714352
STEP 134 ================================
prereg loss 0.6328233 reg_l1 12.386198 reg_l2 10.805749
loss 1.871443
STEP 135 ================================
prereg loss 0.63277894 reg_l1 12.386718 reg_l2 10.807034
loss 1.8714507
STEP 136 ================================
prereg loss 0.6327345 reg_l1 12.387238 reg_l2 10.808317
loss 1.8714583
STEP 137 ================================
prereg loss 0.63269156 reg_l1 12.387755 reg_l2 10.809601
loss 1.8714671
STEP 138 ================================
prereg loss 0.63265014 reg_l1 12.388273 reg_l2 10.810888
loss 1.8714775
STEP 139 ================================
prereg loss 0.6326091 reg_l1 12.388794 reg_l2 10.812176
loss 1.8714886
STEP 140 ================================
prereg loss 0.6325669 reg_l1 12.389313 reg_l2 10.813465
loss 1.8714982
STEP 141 ================================
prereg loss 0.6325285 reg_l1 12.389832 reg_l2 10.814755
loss 1.8715117
STEP 142 ================================
prereg loss 0.63248867 reg_l1 12.39035 reg_l2 10.816047
loss 1.8715236
STEP 143 ================================
prereg loss 0.6324497 reg_l1 12.390872 reg_l2 10.817342
loss 1.871537
STEP 144 ================================
prereg loss 0.6324093 reg_l1 12.391392 reg_l2 10.818637
loss 1.8715484
STEP 145 ================================
prereg loss 0.6323722 reg_l1 12.391912 reg_l2 10.819933
loss 1.8715634
STEP 146 ================================
prereg loss 0.6323354 reg_l1 12.392434 reg_l2 10.821229
loss 1.8715788
STEP 147 ================================
prereg loss 0.63229984 reg_l1 12.392956 reg_l2 10.822529
loss 1.8715954
STEP 148 ================================
prereg loss 0.6322653 reg_l1 12.393478 reg_l2 10.823828
loss 1.8716131
STEP 149 ================================
prereg loss 0.63222957 reg_l1 12.393999 reg_l2 10.82513
loss 1.8716295
STEP 150 ================================
prereg loss 0.63219535 reg_l1 12.394522 reg_l2 10.826432
loss 1.8716476
STEP 151 ================================
prereg loss 0.6321623 reg_l1 12.395044 reg_l2 10.827741
loss 1.8716667
STEP 152 ================================
prereg loss 0.6321299 reg_l1 12.395569 reg_l2 10.829046
loss 1.8716868
STEP 153 ================================
prereg loss 0.63209796 reg_l1 12.396091 reg_l2 10.830351
loss 1.8717071
STEP 154 ================================
prereg loss 0.63206667 reg_l1 12.396615 reg_l2 10.831661
loss 1.8717282
STEP 155 ================================
prereg loss 0.6320379 reg_l1 12.397139 reg_l2 10.832972
loss 1.8717518
STEP 156 ================================
prereg loss 0.63200736 reg_l1 12.397663 reg_l2 10.834284
loss 1.8717737
STEP 157 ================================
prereg loss 0.6319784 reg_l1 12.398188 reg_l2 10.835595
loss 1.8717972
STEP 158 ================================
prereg loss 0.6319506 reg_l1 12.398711 reg_l2 10.83691
loss 1.8718218
STEP 159 ================================
prereg loss 0.6319251 reg_l1 12.399236 reg_l2 10.838225
loss 1.8718487
STEP 160 ================================
prereg loss 0.63189936 reg_l1 12.399764 reg_l2 10.839543
loss 1.8718758
STEP 161 ================================
prereg loss 0.6318715 reg_l1 12.400289 reg_l2 10.840863
loss 1.8719003
STEP 162 ================================
prereg loss 0.63184685 reg_l1 12.400814 reg_l2 10.842181
loss 1.8719282
STEP 163 ================================
prereg loss 0.63182193 reg_l1 12.4013405 reg_l2 10.843504
loss 1.8719561
STEP 164 ================================
prereg loss 0.6317979 reg_l1 12.401868 reg_l2 10.844828
loss 1.8719847
STEP 165 ================================
prereg loss 0.6317752 reg_l1 12.402395 reg_l2 10.846154
loss 1.8720148
STEP 166 ================================
prereg loss 0.63175315 reg_l1 12.402925 reg_l2 10.847479
loss 1.8720455
STEP 167 ================================
prereg loss 0.63173026 reg_l1 12.403451 reg_l2 10.848805
loss 1.8720753
STEP 168 ================================
prereg loss 0.63170964 reg_l1 12.40398 reg_l2 10.850137
loss 1.8721077
STEP 169 ================================
prereg loss 0.63168824 reg_l1 12.404508 reg_l2 10.851468
loss 1.872139
STEP 170 ================================
prereg loss 0.63166696 reg_l1 12.405037 reg_l2 10.8528
loss 1.8721707
STEP 171 ================================
prereg loss 0.6316472 reg_l1 12.405567 reg_l2 10.8541355
loss 1.872204
STEP 172 ================================
prereg loss 0.631628 reg_l1 12.406097 reg_l2 10.855471
loss 1.8722377
STEP 173 ================================
prereg loss 0.6316126 reg_l1 12.406626 reg_l2 10.856808
loss 1.8722751
STEP 174 ================================
prereg loss 0.63159233 reg_l1 12.407158 reg_l2 10.858146
loss 1.8723083
STEP 175 ================================
prereg loss 0.63157743 reg_l1 12.407688 reg_l2 10.859487
loss 1.8723462
STEP 176 ================================
prereg loss 0.6315605 reg_l1 12.408217 reg_l2 10.860828
loss 1.8723822
STEP 177 ================================
prereg loss 0.63154477 reg_l1 12.408751 reg_l2 10.862169
loss 1.8724198
STEP 178 ================================
prereg loss 0.63153166 reg_l1 12.409281 reg_l2 10.863515
loss 1.8724597
STEP 179 ================================
prereg loss 0.63151723 reg_l1 12.409814 reg_l2 10.864862
loss 1.8724988
STEP 180 ================================
prereg loss 0.63150495 reg_l1 12.410346 reg_l2 10.866208
loss 1.8725395
STEP 181 ================================
prereg loss 0.6314927 reg_l1 12.41088 reg_l2 10.867557
loss 1.8725808
STEP 182 ================================
prereg loss 0.6314811 reg_l1 12.411413 reg_l2 10.86891
loss 1.8726225
STEP 183 ================================
prereg loss 0.63146937 reg_l1 12.411948 reg_l2 10.870261
loss 1.8726642
STEP 184 ================================
prereg loss 0.6314586 reg_l1 12.412479 reg_l2 10.871614
loss 1.8727067
STEP 185 ================================
prereg loss 0.63144845 reg_l1 12.413016 reg_l2 10.872971
loss 1.87275
STEP 186 ================================
prereg loss 0.6314406 reg_l1 12.413549 reg_l2 10.874326
loss 1.8727956
STEP 187 ================================
prereg loss 0.6314323 reg_l1 12.414085 reg_l2 10.875686
loss 1.8728409
STEP 188 ================================
prereg loss 0.6314248 reg_l1 12.414619 reg_l2 10.877044
loss 1.8728868
STEP 189 ================================
prereg loss 0.63141984 reg_l1 12.415154 reg_l2 10.878406
loss 1.8729353
STEP 190 ================================
prereg loss 0.6314133 reg_l1 12.415692 reg_l2 10.87977
loss 1.8729825
STEP 191 ================================
prereg loss 0.6314074 reg_l1 12.416229 reg_l2 10.881136
loss 1.8730303
STEP 192 ================================
prereg loss 0.6314034 reg_l1 12.416765 reg_l2 10.882502
loss 1.87308
STEP 193 ================================
prereg loss 0.63139945 reg_l1 12.417302 reg_l2 10.883867
loss 1.8731296
STEP 194 ================================
prereg loss 0.63139397 reg_l1 12.417842 reg_l2 10.88524
loss 1.8731782
STEP 195 ================================
prereg loss 0.631392 reg_l1 12.418381 reg_l2 10.886612
loss 1.8732301
STEP 196 ================================
prereg loss 0.6313894 reg_l1 12.418919 reg_l2 10.887983
loss 1.8732812
STEP 197 ================================
prereg loss 0.6313889 reg_l1 12.419457 reg_l2 10.889358
loss 1.8733346
STEP 198 ================================
prereg loss 0.6313845 reg_l1 12.419995 reg_l2 10.890735
loss 1.873384
STEP 199 ================================
prereg loss 0.6313841 reg_l1 12.420537 reg_l2 10.89211
loss 1.8734379
STEP 200 ================================
prereg loss 0.6313876 reg_l1 12.421075 reg_l2 10.89349
loss 1.8734951
STEP 201 ================================
prereg loss 0.6313883 reg_l1 12.4216175 reg_l2 10.894872
loss 1.87355
STEP 202 ================================
prereg loss 0.631391 reg_l1 12.422156 reg_l2 10.896255
loss 1.8736067
STEP 203 ================================
prereg loss 0.6313939 reg_l1 12.422699 reg_l2 10.897638
loss 1.8736638
STEP 204 ================================
prereg loss 0.63139796 reg_l1 12.423241 reg_l2 10.899021
loss 1.8737221
STEP 205 ================================
prereg loss 0.63140005 reg_l1 12.423782 reg_l2 10.900409
loss 1.8737783
STEP 206 ================================
prereg loss 0.6314045 reg_l1 12.424326 reg_l2 10.901801
loss 1.8738371
STEP 207 ================================
prereg loss 0.6314077 reg_l1 12.424869 reg_l2 10.903192
loss 1.8738945
STEP 208 ================================
prereg loss 0.63141656 reg_l1 12.42541 reg_l2 10.904581
loss 1.8739576
STEP 209 ================================
prereg loss 0.6314243 reg_l1 12.425955 reg_l2 10.905977
loss 1.8740199
STEP 210 ================================
prereg loss 0.63143015 reg_l1 12.4265 reg_l2 10.907374
loss 1.8740802
STEP 211 ================================
prereg loss 0.631439 reg_l1 12.427045 reg_l2 10.908772
loss 1.8741435
STEP 212 ================================
prereg loss 0.63144815 reg_l1 12.427588 reg_l2 10.910169
loss 1.874207
STEP 213 ================================
prereg loss 0.6314554 reg_l1 12.428133 reg_l2 10.911569
loss 1.8742688
STEP 214 ================================
prereg loss 0.6314687 reg_l1 12.4286785 reg_l2 10.912974
loss 1.8743365
STEP 215 ================================
prereg loss 0.6314763 reg_l1 12.429227 reg_l2 10.914378
loss 1.874399
STEP 216 ================================
prereg loss 0.6314889 reg_l1 12.429773 reg_l2 10.91578
loss 1.8744663
STEP 217 ================================
prereg loss 0.63150036 reg_l1 12.430322 reg_l2 10.917191
loss 1.8745326
STEP 218 ================================
prereg loss 0.6315147 reg_l1 12.430869 reg_l2 10.9186
loss 1.8746016
STEP 219 ================================
prereg loss 0.63152635 reg_l1 12.431416 reg_l2 10.92001
loss 1.8746679
STEP 220 ================================
prereg loss 0.63154155 reg_l1 12.431964 reg_l2 10.921422
loss 1.874738
STEP 221 ================================
prereg loss 0.631557 reg_l1 12.432515 reg_l2 10.922837
loss 1.8748085
STEP 222 ================================
prereg loss 0.63156915 reg_l1 12.433064 reg_l2 10.924255
loss 1.8748757
STEP 223 ================================
prereg loss 0.6315849 reg_l1 12.433612 reg_l2 10.925673
loss 1.8749461
STEP 224 ================================
prereg loss 0.6316004 reg_l1 12.434162 reg_l2 10.927092
loss 1.8750166
STEP 225 ================================
prereg loss 0.63161653 reg_l1 12.434714 reg_l2 10.9285145
loss 1.875088
STEP 226 ================================
prereg loss 0.6316358 reg_l1 12.435266 reg_l2 10.929937
loss 1.8751624
STEP 227 ================================
prereg loss 0.63165504 reg_l1 12.435819 reg_l2 10.931361
loss 1.875237
STEP 228 ================================
prereg loss 0.6316737 reg_l1 12.436368 reg_l2 10.932788
loss 1.8753105
STEP 229 ================================
prereg loss 0.63169396 reg_l1 12.436921 reg_l2 10.93422
loss 1.8753861
STEP 230 ================================
prereg loss 0.6317149 reg_l1 12.437474 reg_l2 10.935648
loss 1.8754623
STEP 231 ================================
prereg loss 0.63173604 reg_l1 12.438027 reg_l2 10.937078
loss 1.8755388
STEP 232 ================================
prereg loss 0.63175577 reg_l1 12.438583 reg_l2 10.938512
loss 1.8756142
STEP 233 ================================
prereg loss 0.6317796 reg_l1 12.439136 reg_l2 10.939949
loss 1.8756931
STEP 234 ================================
prereg loss 0.6318035 reg_l1 12.439691 reg_l2 10.941385
loss 1.8757726
STEP 235 ================================
prereg loss 0.6318291 reg_l1 12.440246 reg_l2 10.942823
loss 1.8758538
STEP 236 ================================
prereg loss 0.6318513 reg_l1 12.440803 reg_l2 10.944265
loss 1.8759316
STEP 237 ================================
prereg loss 0.631878 reg_l1 12.4413595 reg_l2 10.945708
loss 1.876014
STEP 238 ================================
prereg loss 0.63190347 reg_l1 12.441916 reg_l2 10.947153
loss 1.876095
STEP 239 ================================
prereg loss 0.63192976 reg_l1 12.4424715 reg_l2 10.948599
loss 1.876177
STEP 240 ================================
prereg loss 0.6319577 reg_l1 12.443029 reg_l2 10.950047
loss 1.8762608
STEP 241 ================================
prereg loss 0.6319843 reg_l1 12.443589 reg_l2 10.951497
loss 1.8763433
STEP 242 ================================
prereg loss 0.6320141 reg_l1 12.444147 reg_l2 10.952949
loss 1.8764288
STEP 243 ================================
prereg loss 0.63204515 reg_l1 12.444706 reg_l2 10.954402
loss 1.8765157
STEP 244 ================================
prereg loss 0.6320751 reg_l1 12.445266 reg_l2 10.955854
loss 1.8766017
STEP 245 ================================
prereg loss 0.632107 reg_l1 12.445825 reg_l2 10.957314
loss 1.8766896
STEP 246 ================================
prereg loss 0.632138 reg_l1 12.446386 reg_l2 10.958773
loss 1.8767767
STEP 247 ================================
prereg loss 0.63216865 reg_l1 12.446947 reg_l2 10.960233
loss 1.8768634
STEP 248 ================================
prereg loss 0.63220406 reg_l1 12.447508 reg_l2 10.961697
loss 1.8769549
STEP 249 ================================
prereg loss 0.6322369 reg_l1 12.448071 reg_l2 10.963161
loss 1.877044
STEP 250 ================================
prereg loss 0.6322704 reg_l1 12.448633 reg_l2 10.964627
loss 1.8771338
STEP 251 ================================
prereg loss 0.63230467 reg_l1 12.449196 reg_l2 10.966094
loss 1.8772243
STEP 252 ================================
prereg loss 0.6323398 reg_l1 12.449758 reg_l2 10.967566
loss 1.8773155
STEP 253 ================================
prereg loss 0.63237476 reg_l1 12.450322 reg_l2 10.969035
loss 1.877407
STEP 254 ================================
prereg loss 0.6324102 reg_l1 12.450888 reg_l2 10.970511
loss 1.8774991
STEP 255 ================================
prereg loss 0.63244927 reg_l1 12.451452 reg_l2 10.971986
loss 1.8775945
STEP 256 ================================
prereg loss 0.63248724 reg_l1 12.452017 reg_l2 10.973462
loss 1.8776889
STEP 257 ================================
prereg loss 0.63252646 reg_l1 12.452582 reg_l2 10.974942
loss 1.8777847
STEP 258 ================================
prereg loss 0.6325649 reg_l1 12.453151 reg_l2 10.976424
loss 1.87788
STEP 259 ================================
prereg loss 0.63260406 reg_l1 12.453715 reg_l2 10.977906
loss 1.8779757
STEP 260 ================================
prereg loss 0.63264525 reg_l1 12.454283 reg_l2 10.97939
loss 1.8780736
STEP 261 ================================
prereg loss 0.63268787 reg_l1 12.454853 reg_l2 10.98088
loss 1.8781731
STEP 262 ================================
prereg loss 0.632729 reg_l1 12.4554205 reg_l2 10.9823675
loss 1.8782711
STEP 263 ================================
prereg loss 0.63277113 reg_l1 12.45599 reg_l2 10.983857
loss 1.8783702
STEP 264 ================================
prereg loss 0.6328137 reg_l1 12.456559 reg_l2 10.9853525
loss 1.8784696
STEP 265 ================================
prereg loss 0.6328589 reg_l1 12.4571295 reg_l2 10.986846
loss 1.8785719
STEP 266 ================================
prereg loss 0.6329012 reg_l1 12.4577 reg_l2 10.988342
loss 1.8786712
STEP 267 ================================
prereg loss 0.6329482 reg_l1 12.458271 reg_l2 10.9898405
loss 1.8787754
STEP 268 ================================
prereg loss 0.6329926 reg_l1 12.458843 reg_l2 10.991341
loss 1.8788769
STEP 269 ================================
prereg loss 0.6330409 reg_l1 12.459415 reg_l2 10.992845
loss 1.8789824
STEP 270 ================================
prereg loss 0.63308567 reg_l1 12.459987 reg_l2 10.994348
loss 1.8790843
STEP 271 ================================
prereg loss 0.6331358 reg_l1 12.460561 reg_l2 10.995852
loss 1.8791919
STEP 272 ================================
prereg loss 0.6331848 reg_l1 12.461135 reg_l2 10.997362
loss 1.8792983
STEP 273 ================================
prereg loss 0.63323617 reg_l1 12.461709 reg_l2 10.998873
loss 1.879407
STEP 274 ================================
prereg loss 0.6332838 reg_l1 12.462284 reg_l2 11.000383
loss 1.8795123
STEP 275 ================================
prereg loss 0.6333342 reg_l1 12.462858 reg_l2 11.001899
loss 1.8796201
STEP 276 ================================
prereg loss 0.6333864 reg_l1 12.463436 reg_l2 11.003416
loss 1.87973
STEP 277 ================================
prereg loss 0.63343924 reg_l1 12.464011 reg_l2 11.004931
loss 1.8798404
STEP 278 ================================
prereg loss 0.6334883 reg_l1 12.464588 reg_l2 11.006452
loss 1.8799472
STEP 279 ================================
prereg loss 0.633542 reg_l1 12.465165 reg_l2 11.007975
loss 1.8800585
STEP 280 ================================
prereg loss 0.63359594 reg_l1 12.465745 reg_l2 11.009499
loss 1.8801705
STEP 281 ================================
prereg loss 0.6336503 reg_l1 12.466324 reg_l2 11.011025
loss 1.8802828
STEP 282 ================================
prereg loss 0.6337067 reg_l1 12.466903 reg_l2 11.012551
loss 1.880397
STEP 283 ================================
prereg loss 0.63376325 reg_l1 12.4674835 reg_l2 11.014082
loss 1.8805115
STEP 284 ================================
prereg loss 0.6338186 reg_l1 12.468062 reg_l2 11.0156145
loss 1.8806249
STEP 285 ================================
prereg loss 0.6338757 reg_l1 12.468644 reg_l2 11.017148
loss 1.8807402
STEP 286 ================================
prereg loss 0.63393414 reg_l1 12.469224 reg_l2 11.018684
loss 1.8808565
STEP 287 ================================
prereg loss 0.63399374 reg_l1 12.469807 reg_l2 11.020224
loss 1.8809744
STEP 288 ================================
prereg loss 0.6340523 reg_l1 12.470391 reg_l2 11.021765
loss 1.8810915
STEP 289 ================================
prereg loss 0.6341131 reg_l1 12.470973 reg_l2 11.023307
loss 1.8812104
STEP 290 ================================
prereg loss 0.63417244 reg_l1 12.471557 reg_l2 11.024848
loss 1.8813281
STEP 291 ================================
prereg loss 0.63423526 reg_l1 12.47214 reg_l2 11.026397
loss 1.8814493
STEP 292 ================================
prereg loss 0.6342972 reg_l1 12.472727 reg_l2 11.027945
loss 1.8815699
STEP 293 ================================
prereg loss 0.6343595 reg_l1 12.473311 reg_l2 11.029495
loss 1.8816906
STEP 294 ================================
prereg loss 0.6344219 reg_l1 12.473898 reg_l2 11.031048
loss 1.8818116
STEP 295 ================================
prereg loss 0.6344862 reg_l1 12.474484 reg_l2 11.032603
loss 1.8819346
STEP 296 ================================
prereg loss 0.63455236 reg_l1 12.475072 reg_l2 11.034161
loss 1.8820596
STEP 297 ================================
prereg loss 0.63461363 reg_l1 12.475659 reg_l2 11.035719
loss 1.8821796
STEP 298 ================================
prereg loss 0.63468325 reg_l1 12.476248 reg_l2 11.037278
loss 1.882308
STEP 299 ================================
prereg loss 0.63474774 reg_l1 12.476836 reg_l2 11.03884
loss 1.8824314
STEP 300 ================================
prereg loss 0.63481814 reg_l1 12.477427 reg_l2 11.040408
loss 1.8825607
STEP 301 ================================
prereg loss 0.63488555 reg_l1 12.478018 reg_l2 11.041975
loss 1.8826873
STEP 302 ================================
prereg loss 0.6349522 reg_l1 12.478608 reg_l2 11.043542
loss 1.882813
STEP 303 ================================
prereg loss 0.6350225 reg_l1 12.479199 reg_l2 11.0451145
loss 1.8829424
STEP 304 ================================
prereg loss 0.6350907 reg_l1 12.479792 reg_l2 11.046689
loss 1.8830699
STEP 305 ================================
prereg loss 0.63516223 reg_l1 12.480384 reg_l2 11.048264
loss 1.8832006
STEP 306 ================================
prereg loss 0.63523346 reg_l1 12.480975 reg_l2 11.049843
loss 1.8833311
STEP 307 ================================
prereg loss 0.63530415 reg_l1 12.481572 reg_l2 11.05142
loss 1.8834615
STEP 308 ================================
prereg loss 0.6353791 reg_l1 12.482165 reg_l2 11.053004
loss 1.8835956
STEP 309 ================================
prereg loss 0.6354532 reg_l1 12.48276 reg_l2 11.054586
loss 1.8837293
STEP 310 ================================
prereg loss 0.63552743 reg_l1 12.483356 reg_l2 11.056171
loss 1.8838632
STEP 311 ================================
prereg loss 0.63559973 reg_l1 12.483953 reg_l2 11.057762
loss 1.883995
STEP 312 ================================
prereg loss 0.6356767 reg_l1 12.4845495 reg_l2 11.059354
loss 1.8841317
STEP 313 ================================
prereg loss 0.6357509 reg_l1 12.4851465 reg_l2 11.060946
loss 1.8842655
STEP 314 ================================
prereg loss 0.6358291 reg_l1 12.485746 reg_l2 11.062541
loss 1.8844037
STEP 315 ================================
prereg loss 0.6359042 reg_l1 12.486343 reg_l2 11.064139
loss 1.8845385
STEP 316 ================================
prereg loss 0.63598204 reg_l1 12.486943 reg_l2 11.065739
loss 1.8846763
STEP 317 ================================
prereg loss 0.6360622 reg_l1 12.487544 reg_l2 11.067338
loss 1.8848166
STEP 318 ================================
prereg loss 0.63614076 reg_l1 12.488144 reg_l2 11.068941
loss 1.8849552
STEP 319 ================================
prereg loss 0.63622165 reg_l1 12.488745 reg_l2 11.070549
loss 1.8850962
STEP 320 ================================
prereg loss 0.6363015 reg_l1 12.489348 reg_l2 11.072157
loss 1.8852364
STEP 321 ================================
prereg loss 0.6363834 reg_l1 12.489951 reg_l2 11.073768
loss 1.8853786
STEP 322 ================================
prereg loss 0.6364655 reg_l1 12.490554 reg_l2 11.07538
loss 1.8855209
STEP 323 ================================
prereg loss 0.6365478 reg_l1 12.491158 reg_l2 11.076992
loss 1.8856636
STEP 324 ================================
prereg loss 0.6366308 reg_l1 12.491762 reg_l2 11.078612
loss 1.885807
STEP 325 ================================
prereg loss 0.63671565 reg_l1 12.492368 reg_l2 11.080232
loss 1.8859525
STEP 326 ================================
prereg loss 0.6368009 reg_l1 12.492973 reg_l2 11.081853
loss 1.8860983
STEP 327 ================================
prereg loss 0.63688546 reg_l1 12.493579 reg_l2 11.083475
loss 1.8862433
STEP 328 ================================
prereg loss 0.63697153 reg_l1 12.494186 reg_l2 11.085102
loss 1.8863902
STEP 329 ================================
prereg loss 0.63705987 reg_l1 12.494795 reg_l2 11.086732
loss 1.8865395
STEP 330 ================================
prereg loss 0.6371455 reg_l1 12.495401 reg_l2 11.08836
loss 1.8866857
STEP 331 ================================
prereg loss 0.6372358 reg_l1 12.496012 reg_l2 11.089994
loss 1.886837
STEP 332 ================================
prereg loss 0.6373238 reg_l1 12.496623 reg_l2 11.091632
loss 1.886986
STEP 333 ================================
prereg loss 0.6374115 reg_l1 12.497233 reg_l2 11.093266
loss 1.8871348
STEP 334 ================================
prereg loss 0.6375015 reg_l1 12.497843 reg_l2 11.094905
loss 1.8872858
STEP 335 ================================
prereg loss 0.6375941 reg_l1 12.498455 reg_l2 11.096548
loss 1.8874396
STEP 336 ================================
prereg loss 0.63768494 reg_l1 12.499069 reg_l2 11.098195
loss 1.8875918
STEP 337 ================================
prereg loss 0.63777643 reg_l1 12.499681 reg_l2 11.099838
loss 1.8877447
STEP 338 ================================
prereg loss 0.6378704 reg_l1 12.500293 reg_l2 11.101487
loss 1.8878996
STEP 339 ================================
prereg loss 0.63796294 reg_l1 12.500909 reg_l2 11.10314
loss 1.8880539
STEP 340 ================================
prereg loss 0.6380583 reg_l1 12.501524 reg_l2 11.1047945
loss 1.8882108
STEP 341 ================================
prereg loss 0.63815296 reg_l1 12.502141 reg_l2 11.106448
loss 1.888367
STEP 342 ================================
prereg loss 0.6382483 reg_l1 12.502757 reg_l2 11.108107
loss 1.888524
STEP 343 ================================
prereg loss 0.6383457 reg_l1 12.503373 reg_l2 11.109769
loss 1.8886831
STEP 344 ================================
prereg loss 0.63844156 reg_l1 12.503991 reg_l2 11.111432
loss 1.8888407
STEP 345 ================================
prereg loss 0.6385411 reg_l1 12.504611 reg_l2 11.113096
loss 1.8890022
STEP 346 ================================
prereg loss 0.6386405 reg_l1 12.505228 reg_l2 11.114762
loss 1.8891634
STEP 347 ================================
prereg loss 0.6387388 reg_l1 12.50585 reg_l2 11.116434
loss 1.8893237
STEP 348 ================================
prereg loss 0.6388376 reg_l1 12.506471 reg_l2 11.118106
loss 1.8894846
STEP 349 ================================
prereg loss 0.638939 reg_l1 12.507092 reg_l2 11.119778
loss 1.8896483
STEP 350 ================================
prereg loss 0.63903874 reg_l1 12.507712 reg_l2 11.121456
loss 1.8898101
STEP 351 ================================
prereg loss 0.6391408 reg_l1 12.508337 reg_l2 11.123136
loss 1.8899746
STEP 352 ================================
prereg loss 0.6392412 reg_l1 12.508961 reg_l2 11.12482
loss 1.8901373
STEP 353 ================================
prereg loss 0.63934636 reg_l1 12.5095825 reg_l2 11.126501
loss 1.8903047
STEP 354 ================================
prereg loss 0.6394514 reg_l1 12.510209 reg_l2 11.128187
loss 1.8904723
STEP 355 ================================
prereg loss 0.63955534 reg_l1 12.510834 reg_l2 11.129877
loss 1.8906387
STEP 356 ================================
prereg loss 0.63965917 reg_l1 12.511461 reg_l2 11.131567
loss 1.8908054
STEP 357 ================================
prereg loss 0.6397661 reg_l1 12.512087 reg_l2 11.133263
loss 1.8909748
STEP 358 ================================
prereg loss 0.6398739 reg_l1 12.512715 reg_l2 11.134958
loss 1.8911455
STEP 359 ================================
prereg loss 0.6399831 reg_l1 12.513345 reg_l2 11.136658
loss 1.8913176
STEP 360 ================================
prereg loss 0.6400903 reg_l1 12.513973 reg_l2 11.138357
loss 1.8914876
STEP 361 ================================
prereg loss 0.64019823 reg_l1 12.514603 reg_l2 11.140062
loss 1.8916585
STEP 362 ================================
prereg loss 0.64030725 reg_l1 12.515234 reg_l2 11.141767
loss 1.8918307
STEP 363 ================================
prereg loss 0.64041674 reg_l1 12.515864 reg_l2 11.143476
loss 1.8920032
STEP 364 ================================
prereg loss 0.64053106 reg_l1 12.516498 reg_l2 11.145186
loss 1.8921808
STEP 365 ================================
prereg loss 0.64064175 reg_l1 12.517132 reg_l2 11.1469
loss 1.892355
STEP 366 ================================
prereg loss 0.6407523 reg_l1 12.517765 reg_l2 11.148616
loss 1.8925289
STEP 367 ================================
prereg loss 0.6408671 reg_l1 12.5184 reg_l2 11.150333
loss 1.8927071
STEP 368 ================================
prereg loss 0.64097977 reg_l1 12.519033 reg_l2 11.152053
loss 1.8928832
STEP 369 ================================
prereg loss 0.6410944 reg_l1 12.51967 reg_l2 11.153776
loss 1.8930614
STEP 370 ================================
prereg loss 0.64120924 reg_l1 12.520305 reg_l2 11.155503
loss 1.8932397
STEP 371 ================================
prereg loss 0.6413265 reg_l1 12.520943 reg_l2 11.157232
loss 1.8934207
STEP 372 ================================
prereg loss 0.6414421 reg_l1 12.521581 reg_l2 11.158962
loss 1.8936002
STEP 373 ================================
prereg loss 0.64156 reg_l1 12.522221 reg_l2 11.160696
loss 1.8937821
STEP 374 ================================
prereg loss 0.6416764 reg_l1 12.522861 reg_l2 11.162432
loss 1.8939625
STEP 375 ================================
prereg loss 0.64179343 reg_l1 12.523501 reg_l2 11.16417
loss 1.8941436
STEP 376 ================================
prereg loss 0.6419138 reg_l1 12.524142 reg_l2 11.16591
loss 1.894328
STEP 377 ================================
prereg loss 0.6420334 reg_l1 12.524784 reg_l2 11.167654
loss 1.8945119
STEP 378 ================================
prereg loss 0.64215344 reg_l1 12.525426 reg_l2 11.169401
loss 1.894696
STEP 379 ================================
prereg loss 0.6422752 reg_l1 12.52607 reg_l2 11.171147
loss 1.8948822
STEP 380 ================================
prereg loss 0.6423984 reg_l1 12.526715 reg_l2 11.172899
loss 1.8950701
STEP 381 ================================
prereg loss 0.6425211 reg_l1 12.527359 reg_l2 11.174654
loss 1.895257
STEP 382 ================================
prereg loss 0.6426444 reg_l1 12.528006 reg_l2 11.17641
loss 1.895445
STEP 383 ================================
prereg loss 0.64276767 reg_l1 12.528651 reg_l2 11.178167
loss 1.8956329
STEP 384 ================================
prereg loss 0.64289194 reg_l1 12.529299 reg_l2 11.179928
loss 1.8958218
STEP 385 ================================
prereg loss 0.6430179 reg_l1 12.529946 reg_l2 11.181692
loss 1.8960125
STEP 386 ================================
prereg loss 0.6431445 reg_l1 12.530596 reg_l2 11.18346
loss 1.8962041
STEP 387 ================================
prereg loss 0.6432716 reg_l1 12.531245 reg_l2 11.185227
loss 1.8963962
STEP 388 ================================
prereg loss 0.6433975 reg_l1 12.531896 reg_l2 11.186996
loss 1.8965871
STEP 389 ================================
prereg loss 0.6435268 reg_l1 12.532548 reg_l2 11.188773
loss 1.8967816
STEP 390 ================================
prereg loss 0.64365697 reg_l1 12.533199 reg_l2 11.19055
loss 1.896977
STEP 391 ================================
prereg loss 0.6437857 reg_l1 12.533853 reg_l2 11.192329
loss 1.897171
STEP 392 ================================
prereg loss 0.6439167 reg_l1 12.534506 reg_l2 11.194112
loss 1.8973674
STEP 393 ================================
prereg loss 0.6440475 reg_l1 12.53516 reg_l2 11.195896
loss 1.8975636
STEP 394 ================================
prereg loss 0.6441809 reg_l1 12.535814 reg_l2 11.197683
loss 1.8977623
STEP 395 ================================
prereg loss 0.64431083 reg_l1 12.536471 reg_l2 11.199472
loss 1.897958
STEP 396 ================================
prereg loss 0.6444453 reg_l1 12.537129 reg_l2 11.201267
loss 1.8981583
STEP 397 ================================
prereg loss 0.64457834 reg_l1 12.537786 reg_l2 11.20306
loss 1.8983569
STEP 398 ================================
prereg loss 0.64471304 reg_l1 12.5384445 reg_l2 11.204858
loss 1.8985575
STEP 399 ================================
prereg loss 0.6448521 reg_l1 12.5391035 reg_l2 11.20666
loss 1.8987625
STEP 400 ================================
prereg loss 0.6449881 reg_l1 12.5397625 reg_l2 11.208463
loss 1.8989644
STEP 401 ================================
prereg loss 0.6451229 reg_l1 12.540425 reg_l2 11.210268
loss 1.8991654
STEP 402 ================================
prereg loss 0.64526343 reg_l1 12.541088 reg_l2 11.212077
loss 1.8993722
STEP 403 ================================
prereg loss 0.6453988 reg_l1 12.541748 reg_l2 11.213887
loss 1.8995736
STEP 404 ================================
prereg loss 0.64554137 reg_l1 12.542413 reg_l2 11.215702
loss 1.8997827
STEP 405 ================================
prereg loss 0.64567953 reg_l1 12.543077 reg_l2 11.21752
loss 1.8999872
STEP 406 ================================
prereg loss 0.6458194 reg_l1 12.543742 reg_l2 11.219339
loss 1.9001937
STEP 407 ================================
prereg loss 0.64596164 reg_l1 12.544407 reg_l2 11.221161
loss 1.9004023
STEP 408 ================================
prereg loss 0.6461018 reg_l1 12.545075 reg_l2 11.222988
loss 1.9006093
STEP 409 ================================
prereg loss 0.64624584 reg_l1 12.545743 reg_l2 11.224811
loss 1.9008201
STEP 410 ================================
prereg loss 0.6463895 reg_l1 12.5464115 reg_l2 11.226645
loss 1.9010307
STEP 411 ================================
prereg loss 0.64653265 reg_l1 12.547079 reg_l2 11.228478
loss 1.9012406
STEP 412 ================================
prereg loss 0.6466794 reg_l1 12.54775 reg_l2 11.230312
loss 1.9014544
STEP 413 ================================
prereg loss 0.6468243 reg_l1 12.548422 reg_l2 11.232153
loss 1.9016664
STEP 414 ================================
prereg loss 0.6469714 reg_l1 12.549092 reg_l2 11.2339945
loss 1.9018807
STEP 415 ================================
prereg loss 0.64712095 reg_l1 12.549767 reg_l2 11.235836
loss 1.9020976
STEP 416 ================================
prereg loss 0.6472664 reg_l1 12.55044 reg_l2 11.237683
loss 1.9023104
STEP 417 ================================
prereg loss 0.64741635 reg_l1 12.551115 reg_l2 11.239536
loss 1.9025279
STEP 418 ================================
prereg loss 0.6475659 reg_l1 12.551788 reg_l2 11.241385
loss 1.9027448
STEP 419 ================================
prereg loss 0.6477142 reg_l1 12.552466 reg_l2 11.24324
loss 1.9029608
STEP 420 ================================
prereg loss 0.64786744 reg_l1 12.5531435 reg_l2 11.245098
loss 1.9031818
STEP 421 ================================
prereg loss 0.6480172 reg_l1 12.553822 reg_l2 11.246962
loss 1.9033995
STEP 422 ================================
prereg loss 0.6481712 reg_l1 12.5545 reg_l2 11.248825
loss 1.9036212
STEP 423 ================================
prereg loss 0.6483222 reg_l1 12.555177 reg_l2 11.2506895
loss 1.90384
STEP 424 ================================
prereg loss 0.64847726 reg_l1 12.555859 reg_l2 11.252561
loss 1.9040632
STEP 425 ================================
prereg loss 0.64863235 reg_l1 12.5565405 reg_l2 11.254434
loss 1.9042864
STEP 426 ================================
prereg loss 0.64878774 reg_l1 12.557223 reg_l2 11.256309
loss 1.9045101
STEP 427 ================================
prereg loss 0.6489442 reg_l1 12.557903 reg_l2 11.2581835
loss 1.9047346
STEP 428 ================================
prereg loss 0.6491006 reg_l1 12.558591 reg_l2 11.260066
loss 1.9049597
STEP 429 ================================
prereg loss 0.6492592 reg_l1 12.559275 reg_l2 11.26195
loss 1.9051867
STEP 430 ================================
prereg loss 0.6494202 reg_l1 12.559961 reg_l2 11.263835
loss 1.9054163
STEP 431 ================================
prereg loss 0.6495759 reg_l1 12.560649 reg_l2 11.265727
loss 1.9056408
STEP 432 ================================
prereg loss 0.64973444 reg_l1 12.561335 reg_l2 11.267617
loss 1.9058678
STEP 433 ================================
prereg loss 0.6498959 reg_l1 12.562022 reg_l2 11.269513
loss 1.9060981
STEP 434 ================================
prereg loss 0.6500582 reg_l1 12.562715 reg_l2 11.271412
loss 1.9063296
STEP 435 ================================
prereg loss 0.6502178 reg_l1 12.563403 reg_l2 11.273312
loss 1.9065582
STEP 436 ================================
prereg loss 0.65038246 reg_l1 12.564094 reg_l2 11.275216
loss 1.9067919
STEP 437 ================================
prereg loss 0.6505447 reg_l1 12.564787 reg_l2 11.2771225
loss 1.9070234
STEP 438 ================================
prereg loss 0.65070987 reg_l1 12.565479 reg_l2 11.279034
loss 1.9072578
STEP 439 ================================
prereg loss 0.65087616 reg_l1 12.566173 reg_l2 11.280943
loss 1.9074935
STEP 440 ================================
prereg loss 0.65104306 reg_l1 12.566867 reg_l2 11.28286
loss 1.9077297
STEP 441 ================================
prereg loss 0.6512111 reg_l1 12.567566 reg_l2 11.2847805
loss 1.9079678
STEP 442 ================================
prereg loss 0.6513771 reg_l1 12.568261 reg_l2 11.286698
loss 1.9082032
STEP 443 ================================
prereg loss 0.65154594 reg_l1 12.568957 reg_l2 11.288624
loss 1.9084418
STEP 444 ================================
prereg loss 0.6517139 reg_l1 12.569656 reg_l2 11.290551
loss 1.9086795
STEP 445 ================================
prereg loss 0.6518844 reg_l1 12.570358 reg_l2 11.292481
loss 1.9089203
STEP 446 ================================
prereg loss 0.65205646 reg_l1 12.571055 reg_l2 11.294413
loss 1.909162
STEP 447 ================================
prereg loss 0.65222603 reg_l1 12.571757 reg_l2 11.296351
loss 1.9094019
STEP 448 ================================
prereg loss 0.65239966 reg_l1 12.572461 reg_l2 11.298292
loss 1.9096458
STEP 449 ================================
prereg loss 0.6525713 reg_l1 12.573164 reg_l2 11.300233
loss 1.9098878
STEP 450 ================================
prereg loss 0.6527441 reg_l1 12.573869 reg_l2 11.302179
loss 1.910131
STEP 451 ================================
prereg loss 0.65291965 reg_l1 12.5745735 reg_l2 11.30413
loss 1.910377
STEP 452 ================================
prereg loss 0.65309346 reg_l1 12.575279 reg_l2 11.306079
loss 1.9106214
STEP 453 ================================
prereg loss 0.65327156 reg_l1 12.575987 reg_l2 11.308031
loss 1.9108703
STEP 454 ================================
prereg loss 0.65344626 reg_l1 12.5766945 reg_l2 11.309991
loss 1.9111156
STEP 455 ================================
prereg loss 0.65362567 reg_l1 12.577403 reg_l2 11.311952
loss 1.911366
STEP 456 ================================
prereg loss 0.6538029 reg_l1 12.578114 reg_l2 11.313914
loss 1.9116143
STEP 457 ================================
prereg loss 0.6539809 reg_l1 12.578823 reg_l2 11.315879
loss 1.9118633
STEP 458 ================================
prereg loss 0.6541607 reg_l1 12.5795355 reg_l2 11.317853
loss 1.9121141
STEP 459 ================================
prereg loss 0.65434146 reg_l1 12.580248 reg_l2 11.319824
loss 1.9123663
STEP 460 ================================
prereg loss 0.6545241 reg_l1 12.580962 reg_l2 11.321799
loss 1.9126203
STEP 461 ================================
prereg loss 0.65470433 reg_l1 12.581678 reg_l2 11.323778
loss 1.9128722
STEP 462 ================================
prereg loss 0.6548889 reg_l1 12.582393 reg_l2 11.325761
loss 1.9131281
STEP 463 ================================
prereg loss 0.6550696 reg_l1 12.583109 reg_l2 11.3277445
loss 1.9133805
STEP 464 ================================
prereg loss 0.65525645 reg_l1 12.583828 reg_l2 11.329732
loss 1.9136393
STEP 465 ================================
prereg loss 0.65544343 reg_l1 12.584547 reg_l2 11.331725
loss 1.9138981
STEP 466 ================================
prereg loss 0.6556293 reg_l1 12.585266 reg_l2 11.333719
loss 1.914156
STEP 467 ================================
prereg loss 0.6558167 reg_l1 12.585985 reg_l2 11.335716
loss 1.9144152
STEP 468 ================================
prereg loss 0.6560026 reg_l1 12.586708 reg_l2 11.337719
loss 1.9146733
STEP 469 ================================
prereg loss 0.65619177 reg_l1 12.587431 reg_l2 11.339723
loss 1.9149349
STEP 470 ================================
prereg loss 0.6563813 reg_l1 12.588154 reg_l2 11.341727
loss 1.9151967
STEP 471 ================================
prereg loss 0.65657276 reg_l1 12.588879 reg_l2 11.343737
loss 1.9154606
STEP 472 ================================
prereg loss 0.65676284 reg_l1 12.589603 reg_l2 11.345751
loss 1.9157232
STEP 473 ================================
prereg loss 0.65695465 reg_l1 12.590331 reg_l2 11.347767
loss 1.9159877
STEP 474 ================================
prereg loss 0.6571462 reg_l1 12.591057 reg_l2 11.349784
loss 1.9162519
STEP 475 ================================
prereg loss 0.6573409 reg_l1 12.591787 reg_l2 11.3518095
loss 1.9165196
STEP 476 ================================
prereg loss 0.6575348 reg_l1 12.592517 reg_l2 11.353837
loss 1.9167864
STEP 477 ================================
prereg loss 0.65772927 reg_l1 12.593246 reg_l2 11.355864
loss 1.9170539
STEP 478 ================================
prereg loss 0.6579257 reg_l1 12.593977 reg_l2 11.357897
loss 1.9173235
STEP 479 ================================
prereg loss 0.65812016 reg_l1 12.594712 reg_l2 11.359933
loss 1.9175915
STEP 480 ================================
prereg loss 0.65832084 reg_l1 12.595444 reg_l2 11.361972
loss 1.9178653
STEP 481 ================================
prereg loss 0.6585173 reg_l1 12.59618 reg_l2 11.364013
loss 1.9181354
STEP 482 ================================
prereg loss 0.6587157 reg_l1 12.596914 reg_l2 11.366058
loss 1.9184072
STEP 483 ================================
prereg loss 0.65891504 reg_l1 12.597652 reg_l2 11.368107
loss 1.9186803
STEP 484 ================================
prereg loss 0.6591166 reg_l1 12.59839 reg_l2 11.370159
loss 1.9189556
STEP 485 ================================
prereg loss 0.6593186 reg_l1 12.599128 reg_l2 11.372211
loss 1.9192314
STEP 486 ================================
prereg loss 0.6595184 reg_l1 12.599868 reg_l2 11.37427
loss 1.9195051
STEP 487 ================================
prereg loss 0.659724 reg_l1 12.600609 reg_l2 11.376335
loss 1.9197849
STEP 488 ================================
prereg loss 0.6599268 reg_l1 12.60135 reg_l2 11.378395
loss 1.9200618
STEP 489 ================================
prereg loss 0.6601321 reg_l1 12.602093 reg_l2 11.380466
loss 1.9203415
STEP 490 ================================
prereg loss 0.6603373 reg_l1 12.6028385 reg_l2 11.382539
loss 1.9206212
STEP 491 ================================
prereg loss 0.66054195 reg_l1 12.603583 reg_l2 11.384613
loss 1.9209003
STEP 492 ================================
prereg loss 0.66075146 reg_l1 12.604331 reg_l2 11.38669
loss 1.9211845
STEP 493 ================================
prereg loss 0.6609611 reg_l1 12.605077 reg_l2 11.388771
loss 1.9214687
STEP 494 ================================
prereg loss 0.6611687 reg_l1 12.605824 reg_l2 11.390855
loss 1.9217511
STEP 495 ================================
prereg loss 0.6613777 reg_l1 12.606572 reg_l2 11.392942
loss 1.922035
STEP 496 ================================
prereg loss 0.66158843 reg_l1 12.607324 reg_l2 11.395034
loss 1.9223208
STEP 497 ================================
prereg loss 0.6618025 reg_l1 12.608077 reg_l2 11.397129
loss 1.9226103
STEP 498 ================================
prereg loss 0.6620138 reg_l1 12.608829 reg_l2 11.399228
loss 1.9228966
STEP 499 ================================
prereg loss 0.66222537 reg_l1 12.609581 reg_l2 11.401331
loss 1.9231834
STEP 500 ================================
prereg loss 0.66244113 reg_l1 12.610335 reg_l2 11.403434
loss 1.9234747
STEP 501 ================================
prereg loss 0.6626558 reg_l1 12.611092 reg_l2 11.405542
loss 1.923765
STEP 502 ================================
prereg loss 0.6628704 reg_l1 12.611848 reg_l2 11.407654
loss 1.9240552
STEP 503 ================================
prereg loss 0.66308725 reg_l1 12.612605 reg_l2 11.409766
loss 1.9243478
STEP 504 ================================
prereg loss 0.6633053 reg_l1 12.613366 reg_l2 11.411886
loss 1.924642
STEP 505 ================================
prereg loss 0.6635212 reg_l1 12.614127 reg_l2 11.414009
loss 1.9249339
STEP 506 ================================
prereg loss 0.66374063 reg_l1 12.614887 reg_l2 11.416134
loss 1.9252294
STEP 507 ================================
prereg loss 0.66396 reg_l1 12.615648 reg_l2 11.418262
loss 1.9255248
STEP 508 ================================
prereg loss 0.66417956 reg_l1 12.616413 reg_l2 11.420395
loss 1.925821
STEP 509 ================================
prereg loss 0.66440344 reg_l1 12.617177 reg_l2 11.422527
loss 1.9261211
STEP 510 ================================
prereg loss 0.6646252 reg_l1 12.617941 reg_l2 11.424666
loss 1.9264193
STEP 511 ================================
prereg loss 0.6648489 reg_l1 12.618708 reg_l2 11.426809
loss 1.9267197
STEP 512 ================================
prereg loss 0.6650732 reg_l1 12.619477 reg_l2 11.428955
loss 1.927021
STEP 513 ================================
prereg loss 0.6652993 reg_l1 12.620246 reg_l2 11.431105
loss 1.9273239
STEP 514 ================================
prereg loss 0.66552407 reg_l1 12.621015 reg_l2 11.433255
loss 1.9276257
STEP 515 ================================
prereg loss 0.6657522 reg_l1 12.621787 reg_l2 11.435415
loss 1.927931
STEP 516 ================================
prereg loss 0.6659793 reg_l1 12.622559 reg_l2 11.437573
loss 1.9282353
STEP 517 ================================
prereg loss 0.66620886 reg_l1 12.623331 reg_l2 11.439734
loss 1.928542
STEP 518 ================================
prereg loss 0.66643906 reg_l1 12.624105 reg_l2 11.441902
loss 1.9288496
STEP 519 ================================
prereg loss 0.66666585 reg_l1 12.624883 reg_l2 11.444073
loss 1.9291542
STEP 520 ================================
prereg loss 0.6669004 reg_l1 12.625656 reg_l2 11.446246
loss 1.929466
STEP 521 ================================
prereg loss 0.66713023 reg_l1 12.626435 reg_l2 11.448422
loss 1.9297738
STEP 522 ================================
prereg loss 0.66736716 reg_l1 12.627216 reg_l2 11.4506035
loss 1.9300888
STEP 523 ================================
prereg loss 0.66759974 reg_l1 12.627995 reg_l2 11.4527855
loss 1.9303992
STEP 524 ================================
prereg loss 0.6678326 reg_l1 12.628774 reg_l2 11.454972
loss 1.93071
STEP 525 ================================
prereg loss 0.66806966 reg_l1 12.629558 reg_l2 11.457164
loss 1.9310255
STEP 526 ================================
prereg loss 0.668303 reg_l1 12.6303425 reg_l2 11.459361
loss 1.9313372
STEP 527 ================================
prereg loss 0.6685404 reg_l1 12.631125 reg_l2 11.461558
loss 1.931653
STEP 528 ================================
prereg loss 0.6687789 reg_l1 12.631911 reg_l2 11.463759
loss 1.93197
STEP 529 ================================
prereg loss 0.669018 reg_l1 12.6327 reg_l2 11.465968
loss 1.9322879
STEP 530 ================================
prereg loss 0.6692571 reg_l1 12.633488 reg_l2 11.468176
loss 1.932606
STEP 531 ================================
prereg loss 0.669497 reg_l1 12.634277 reg_l2 11.4703865
loss 1.9329247
STEP 532 ================================
prereg loss 0.6697386 reg_l1 12.635066 reg_l2 11.472607
loss 1.9332452
STEP 533 ================================
prereg loss 0.6699819 reg_l1 12.63586 reg_l2 11.474826
loss 1.933568
STEP 534 ================================
prereg loss 0.6702267 reg_l1 12.636653 reg_l2 11.477049
loss 1.933892
STEP 535 ================================
prereg loss 0.67046905 reg_l1 12.637447 reg_l2 11.479278
loss 1.9342138
STEP 536 ================================
prereg loss 0.67071563 reg_l1 12.638243 reg_l2 11.481509
loss 1.93454
STEP 537 ================================
prereg loss 0.67096186 reg_l1 12.639038 reg_l2 11.483743
loss 1.9348657
STEP 538 ================================
prereg loss 0.6712108 reg_l1 12.639834 reg_l2 11.485982
loss 1.9351943
STEP 539 ================================
prereg loss 0.67145705 reg_l1 12.640635 reg_l2 11.488224
loss 1.9355205
STEP 540 ================================
prereg loss 0.67170835 reg_l1 12.641435 reg_l2 11.49047
loss 1.9358518
STEP 541 ================================
prereg loss 0.67195743 reg_l1 12.642236 reg_l2 11.49272
loss 1.9361811
STEP 542 ================================
prereg loss 0.672207 reg_l1 12.643039 reg_l2 11.494972
loss 1.9365109
STEP 543 ================================
prereg loss 0.67246014 reg_l1 12.643842 reg_l2 11.497231
loss 1.9368443
STEP 544 ================================
prereg loss 0.67271394 reg_l1 12.644648 reg_l2 11.499492
loss 1.9371786
STEP 545 ================================
prereg loss 0.6729652 reg_l1 12.645452 reg_l2 11.501756
loss 1.9375105
STEP 546 ================================
prereg loss 0.67322147 reg_l1 12.646257 reg_l2 11.504023
loss 1.9378473
STEP 547 ================================
prereg loss 0.6734752 reg_l1 12.647068 reg_l2 11.506295
loss 1.9381821
STEP 548 ================================
prereg loss 0.67373073 reg_l1 12.647878 reg_l2 11.508568
loss 1.9385185
STEP 549 ================================
prereg loss 0.6739899 reg_l1 12.648687 reg_l2 11.510847
loss 1.9388586
STEP 550 ================================
prereg loss 0.6742469 reg_l1 12.649498 reg_l2 11.513133
loss 1.9391967
STEP 551 ================================
prereg loss 0.67450565 reg_l1 12.650314 reg_l2 11.51542
loss 1.939537
STEP 552 ================================
prereg loss 0.6747655 reg_l1 12.651126 reg_l2 11.51771
loss 1.9398782
STEP 553 ================================
prereg loss 0.6750274 reg_l1 12.651943 reg_l2 11.520004
loss 1.9402217
STEP 554 ================================
prereg loss 0.6752888 reg_l1 12.6527605 reg_l2 11.522303
loss 1.9405649
STEP 555 ================================
prereg loss 0.6755528 reg_l1 12.653578 reg_l2 11.524605
loss 1.9409106
STEP 556 ================================
prereg loss 0.6758134 reg_l1 12.654397 reg_l2 11.526909
loss 1.9412532
STEP 557 ================================
prereg loss 0.67608106 reg_l1 12.655217 reg_l2 11.529221
loss 1.9416028
STEP 558 ================================
prereg loss 0.6763469 reg_l1 12.65604 reg_l2 11.531536
loss 1.9419509
STEP 559 ================================
prereg loss 0.676612 reg_l1 12.656862 reg_l2 11.533847
loss 1.9422983
STEP 560 ================================
prereg loss 0.6768796 reg_l1 12.657685 reg_l2 11.536173
loss 1.9426482
STEP 561 ================================
prereg loss 0.6771462 reg_l1 12.658512 reg_l2 11.538498
loss 1.9429975
STEP 562 ================================
prereg loss 0.6774167 reg_l1 12.659339 reg_l2 11.540825
loss 1.9433506
STEP 563 ================================
prereg loss 0.67768884 reg_l1 12.660169 reg_l2 11.543159
loss 1.9437057
STEP 564 ================================
prereg loss 0.67796123 reg_l1 12.660995 reg_l2 11.545495
loss 1.9440607
STEP 565 ================================
prereg loss 0.6782328 reg_l1 12.661827 reg_l2 11.547836
loss 1.9444156
STEP 566 ================================
prereg loss 0.6785073 reg_l1 12.662658 reg_l2 11.550181
loss 1.9447731
STEP 567 ================================
prereg loss 0.6787816 reg_l1 12.663491 reg_l2 11.552529
loss 1.9451308
STEP 568 ================================
prereg loss 0.679054 reg_l1 12.664326 reg_l2 11.55488
loss 1.9454867
STEP 569 ================================
prereg loss 0.67933273 reg_l1 12.665161 reg_l2 11.557237
loss 1.9458488
STEP 570 ================================
prereg loss 0.67960864 reg_l1 12.665998 reg_l2 11.559599
loss 1.9462085
STEP 571 ================================
prereg loss 0.6798862 reg_l1 12.666838 reg_l2 11.561964
loss 1.94657
STEP 572 ================================
prereg loss 0.6801636 reg_l1 12.667678 reg_l2 11.564331
loss 1.9469315
STEP 573 ================================
prereg loss 0.6804446 reg_l1 12.668518 reg_l2 11.566701
loss 1.9472964
STEP 574 ================================
prereg loss 0.68072593 reg_l1 12.669359 reg_l2 11.569078
loss 1.9476619
STEP 575 ================================
prereg loss 0.6810091 reg_l1 12.670203 reg_l2 11.571459
loss 1.9480295
STEP 576 ================================
prereg loss 0.68129236 reg_l1 12.671048 reg_l2 11.573842
loss 1.9483972
STEP 577 ================================
prereg loss 0.6815753 reg_l1 12.671894 reg_l2 11.576232
loss 1.9487647
STEP 578 ================================
prereg loss 0.6818601 reg_l1 12.672742 reg_l2 11.578622
loss 1.9491343
STEP 579 ================================
prereg loss 0.6821472 reg_l1 12.67359 reg_l2 11.5810175
loss 1.9495063
STEP 580 ================================
prereg loss 0.68243384 reg_l1 12.674438 reg_l2 11.583418
loss 1.9498777
STEP 581 ================================
prereg loss 0.6827198 reg_l1 12.675292 reg_l2 11.585821
loss 1.9502491
STEP 582 ================================
prereg loss 0.68301046 reg_l1 12.676142 reg_l2 11.58823
loss 1.9506247
STEP 583 ================================
prereg loss 0.6833006 reg_l1 12.676997 reg_l2 11.590643
loss 1.9510003
STEP 584 ================================
prereg loss 0.6835922 reg_l1 12.677853 reg_l2 11.593058
loss 1.9513775
STEP 585 ================================
prereg loss 0.68388367 reg_l1 12.678709 reg_l2 11.595479
loss 1.9517546
STEP 586 ================================
prereg loss 0.6841769 reg_l1 12.679568 reg_l2 11.597903
loss 1.9521338
STEP 587 ================================
prereg loss 0.68447137 reg_l1 12.680427 reg_l2 11.600331
loss 1.952514
STEP 588 ================================
prereg loss 0.6847663 reg_l1 12.681287 reg_l2 11.602766
loss 1.952895
STEP 589 ================================
prereg loss 0.6850631 reg_l1 12.682148 reg_l2 11.605201
loss 1.953278
STEP 590 ================================
prereg loss 0.68535846 reg_l1 12.683011 reg_l2 11.607642
loss 1.9536595
STEP 591 ================================
prereg loss 0.68565476 reg_l1 12.683875 reg_l2 11.6100855
loss 1.9540423
STEP 592 ================================
prereg loss 0.6859566 reg_l1 12.684743 reg_l2 11.612536
loss 1.9544309
STEP 593 ================================
prereg loss 0.686256 reg_l1 12.685608 reg_l2 11.614988
loss 1.9548168
STEP 594 ================================
prereg loss 0.68655854 reg_l1 12.686477 reg_l2 11.617444
loss 1.9552062
STEP 595 ================================
prereg loss 0.68686223 reg_l1 12.687347 reg_l2 11.619908
loss 1.955597
STEP 596 ================================
prereg loss 0.6871637 reg_l1 12.688218 reg_l2 11.622374
loss 1.9559855
STEP 597 ================================
prereg loss 0.6874697 reg_l1 12.68909 reg_l2 11.624841
loss 1.9563787
STEP 598 ================================
prereg loss 0.68777454 reg_l1 12.689964 reg_l2 11.627314
loss 1.956771
STEP 599 ================================
prereg loss 0.68808085 reg_l1 12.690842 reg_l2 11.629795
loss 1.957165
STEP 600 ================================
prereg loss 0.6883869 reg_l1 12.691717 reg_l2 11.632277
loss 1.9575586
STEP 601 ================================
prereg loss 0.68869436 reg_l1 12.692594 reg_l2 11.634762
loss 1.9579537
STEP 602 ================================
prereg loss 0.68900216 reg_l1 12.693473 reg_l2 11.637253
loss 1.9583495
STEP 603 ================================
prereg loss 0.68931645 reg_l1 12.694357 reg_l2 11.639749
loss 1.9587522
STEP 604 ================================
prereg loss 0.6896236 reg_l1 12.695236 reg_l2 11.642246
loss 1.9591472
STEP 605 ================================
prereg loss 0.68993896 reg_l1 12.69612 reg_l2 11.64475
loss 1.9595511
STEP 606 ================================
prereg loss 0.6902527 reg_l1 12.697006 reg_l2 11.647262
loss 1.9599533
STEP 607 ================================
prereg loss 0.6905648 reg_l1 12.697893 reg_l2 11.649773
loss 1.9603541
STEP 608 ================================
prereg loss 0.6908819 reg_l1 12.69878 reg_l2 11.6522875
loss 1.9607599
STEP 609 ================================
prereg loss 0.69119924 reg_l1 12.699669 reg_l2 11.654808
loss 1.9611661
STEP 610 ================================
prereg loss 0.69151765 reg_l1 12.700562 reg_l2 11.657334
loss 1.9615738
STEP 611 ================================
prereg loss 0.6918365 reg_l1 12.701451 reg_l2 11.6598625
loss 1.9619817
STEP 612 ================================
prereg loss 0.6921559 reg_l1 12.702345 reg_l2 11.662395
loss 1.9623904
STEP 613 ================================
prereg loss 0.69247746 reg_l1 12.70324 reg_l2 11.664932
loss 1.9628016
STEP 614 ================================
prereg loss 0.6927999 reg_l1 12.704138 reg_l2 11.667476
loss 1.9632137
STEP 615 ================================
prereg loss 0.69312066 reg_l1 12.705033 reg_l2 11.670022
loss 1.963624
STEP 616 ================================
prereg loss 0.6934477 reg_l1 12.705934 reg_l2 11.672569
loss 1.9640411
STEP 617 ================================
prereg loss 0.6937754 reg_l1 12.706834 reg_l2 11.675128
loss 1.9644588
STEP 618 ================================
prereg loss 0.6940981 reg_l1 12.707736 reg_l2 11.677689
loss 1.9648718
STEP 619 ================================
prereg loss 0.6944271 reg_l1 12.708637 reg_l2 11.68025
loss 1.9652908
STEP 620 ================================
prereg loss 0.6947553 reg_l1 12.709543 reg_l2 11.682818
loss 1.9657097
STEP 621 ================================
prereg loss 0.6950853 reg_l1 12.710448 reg_l2 11.685392
loss 1.9661301
STEP 622 ================================
prereg loss 0.69541395 reg_l1 12.711358 reg_l2 11.687971
loss 1.9665498
STEP 623 ================================
prereg loss 0.69574654 reg_l1 12.712266 reg_l2 11.690548
loss 1.9669732
STEP 624 ================================
prereg loss 0.6960799 reg_l1 12.713175 reg_l2 11.693136
loss 1.9673975
STEP 625 ================================
prereg loss 0.6964147 reg_l1 12.71409 reg_l2 11.69573
loss 1.9678237
STEP 626 ================================
prereg loss 0.6967481 reg_l1 12.715 reg_l2 11.698321
loss 1.9682481
STEP 627 ================================
prereg loss 0.69708407 reg_l1 12.715917 reg_l2 11.700918
loss 1.9686757
STEP 628 ================================
prereg loss 0.69742244 reg_l1 12.716834 reg_l2 11.703525
loss 1.969106
STEP 629 ================================
prereg loss 0.6977609 reg_l1 12.717752 reg_l2 11.706133
loss 1.9695361
STEP 630 ================================
prereg loss 0.69810104 reg_l1 12.718669 reg_l2 11.708744
loss 1.969968
STEP 631 ================================
prereg loss 0.6984412 reg_l1 12.719591 reg_l2 11.711362
loss 1.9704003
STEP 632 ================================
prereg loss 0.698786 reg_l1 12.720515 reg_l2 11.713985
loss 1.9708376
STEP 633 ================================
prereg loss 0.69912577 reg_l1 12.721438 reg_l2 11.71661
loss 1.9712696
STEP 634 ================================
prereg loss 0.6994722 reg_l1 12.722362 reg_l2 11.719238
loss 1.9717084
STEP 635 ================================
prereg loss 0.6998162 reg_l1 12.723289 reg_l2 11.721875
loss 1.9721451
STEP 636 ================================
prereg loss 0.7001636 reg_l1 12.724218 reg_l2 11.724517
loss 1.9725854
STEP 637 ================================
prereg loss 0.7005112 reg_l1 12.725147 reg_l2 11.727158
loss 1.9730259
STEP 638 ================================
prereg loss 0.70085835 reg_l1 12.726078 reg_l2 11.729805
loss 1.9734662
STEP 639 ================================
prereg loss 0.70121014 reg_l1 12.727011 reg_l2 11.732465
loss 1.9739113
STEP 640 ================================
prereg loss 0.70156074 reg_l1 12.727945 reg_l2 11.735121
loss 1.9743553
STEP 641 ================================
prereg loss 0.7019133 reg_l1 12.728879 reg_l2 11.7377825
loss 1.9748013
STEP 642 ================================
prereg loss 0.70226777 reg_l1 12.729817 reg_l2 11.740451
loss 1.9752495
STEP 643 ================================
prereg loss 0.7026208 reg_l1 12.730756 reg_l2 11.743123
loss 1.9756963
STEP 644 ================================
prereg loss 0.7029757 reg_l1 12.731696 reg_l2 11.745798
loss 1.9761453
STEP 645 ================================
prereg loss 0.7033312 reg_l1 12.732637 reg_l2 11.748479
loss 1.9765949
STEP 646 ================================
prereg loss 0.7036912 reg_l1 12.733581 reg_l2 11.751166
loss 1.9770494
STEP 647 ================================
prereg loss 0.70404893 reg_l1 12.734527 reg_l2 11.753858
loss 1.9775016
STEP 648 ================================
prereg loss 0.70440936 reg_l1 12.735474 reg_l2 11.756551
loss 1.9779568
STEP 649 ================================
prereg loss 0.7047724 reg_l1 12.73642 reg_l2 11.759252
loss 1.9784143
STEP 650 ================================
prereg loss 0.70513403 reg_l1 12.7373705 reg_l2 11.761956
loss 1.9788711
STEP 651 ================================
prereg loss 0.7054987 reg_l1 12.738321 reg_l2 11.764667
loss 1.9793309
STEP 652 ================================
prereg loss 0.7058641 reg_l1 12.739273 reg_l2 11.767377
loss 1.9797914
STEP 653 ================================
prereg loss 0.7062279 reg_l1 12.740227 reg_l2 11.770096
loss 1.9802506
STEP 654 ================================
prereg loss 0.70659727 reg_l1 12.741183 reg_l2 11.77282
loss 1.9807155
STEP 655 ================================
prereg loss 0.7069646 reg_l1 12.74214 reg_l2 11.775548
loss 1.9811786
STEP 656 ================================
prereg loss 0.7073355 reg_l1 12.743098 reg_l2 11.778281
loss 1.9816453
STEP 657 ================================
prereg loss 0.7077062 reg_l1 12.744058 reg_l2 11.781017
loss 1.982112
STEP 658 ================================
prereg loss 0.70807695 reg_l1 12.745021 reg_l2 11.78376
loss 1.9825791
STEP 659 ================================
prereg loss 0.70845026 reg_l1 12.745983 reg_l2 11.786505
loss 1.9830487
STEP 660 ================================
prereg loss 0.70882404 reg_l1 12.746949 reg_l2 11.78926
loss 1.983519
STEP 661 ================================
prereg loss 0.7092007 reg_l1 12.747916 reg_l2 11.792017
loss 1.9839923
STEP 662 ================================
prereg loss 0.7095758 reg_l1 12.748881 reg_l2 11.794775
loss 1.9844639
STEP 663 ================================
prereg loss 0.70995593 reg_l1 12.749851 reg_l2 11.797542
loss 1.9849411
STEP 664 ================================
prereg loss 0.7103358 reg_l1 12.750824 reg_l2 11.800315
loss 1.9854183
STEP 665 ================================
prereg loss 0.7107141 reg_l1 12.751796 reg_l2 11.803091
loss 1.9858937
STEP 666 ================================
prereg loss 0.71109694 reg_l1 12.752769 reg_l2 11.80587
loss 1.9863739
STEP 667 ================================
prereg loss 0.7114788 reg_l1 12.753747 reg_l2 11.808656
loss 1.9868536
STEP 668 ================================
prereg loss 0.71186537 reg_l1 12.754724 reg_l2 11.811448
loss 1.9873378
STEP 669 ================================
prereg loss 0.7122513 reg_l1 12.755702 reg_l2 11.814241
loss 1.9878216
STEP 670 ================================
prereg loss 0.7126378 reg_l1 12.756683 reg_l2 11.81704
loss 1.9883062
STEP 671 ================================
prereg loss 0.7130238 reg_l1 12.757668 reg_l2 11.819847
loss 1.9887905
STEP 672 ================================
prereg loss 0.7134129 reg_l1 12.75865 reg_l2 11.822658
loss 1.9892778
STEP 673 ================================
prereg loss 0.7138054 reg_l1 12.759634 reg_l2 11.825469
loss 1.9897687
STEP 674 ================================
prereg loss 0.71419597 reg_l1 12.760622 reg_l2 11.828291
loss 1.9902582
STEP 675 ================================
prereg loss 0.7145885 reg_l1 12.761613 reg_l2 11.831119
loss 1.9907498
STEP 676 ================================
prereg loss 0.7149829 reg_l1 12.7626 reg_l2 11.833945
loss 1.9912429
STEP 677 ================================
prereg loss 0.71537536 reg_l1 12.763594 reg_l2 11.83678
loss 1.9917347
STEP 678 ================================
prereg loss 0.7157731 reg_l1 12.764588 reg_l2 11.839622
loss 1.992232
STEP 679 ================================
prereg loss 0.7161733 reg_l1 12.765583 reg_l2 11.842465
loss 1.9927316
STEP 680 ================================
prereg loss 0.716571 reg_l1 12.766579 reg_l2 11.845313
loss 1.9932289
STEP 681 ================================
prereg loss 0.7169709 reg_l1 12.767579 reg_l2 11.848167
loss 1.9937289
STEP 682 ================================
prereg loss 0.717373 reg_l1 12.7685795 reg_l2 11.85103
loss 1.994231
STEP 683 ================================
prereg loss 0.71777433 reg_l1 12.76958 reg_l2 11.853891
loss 1.9947324
STEP 684 ================================
prereg loss 0.7181787 reg_l1 12.770585 reg_l2 11.856761
loss 1.9952371
STEP 685 ================================
prereg loss 0.71858674 reg_l1 12.771589 reg_l2 11.859638
loss 1.9957457
STEP 686 ================================
prereg loss 0.71899366 reg_l1 12.772595 reg_l2 11.862516
loss 1.9962533
STEP 687 ================================
prereg loss 0.7194014 reg_l1 12.773605 reg_l2 11.865401
loss 1.996762
STEP 688 ================================
prereg loss 0.7198087 reg_l1 12.774616 reg_l2 11.86829
loss 1.9972703
STEP 689 ================================
prereg loss 0.7202219 reg_l1 12.775627 reg_l2 11.871186
loss 1.9977846
STEP 690 ================================
prereg loss 0.72063535 reg_l1 12.77664 reg_l2 11.874085
loss 1.9982994
STEP 691 ================================
prereg loss 0.72104686 reg_l1 12.777657 reg_l2 11.876989
loss 1.9988124
STEP 692 ================================
prereg loss 0.72146034 reg_l1 12.778672 reg_l2 11.8799
loss 1.9993275
STEP 693 ================================
prereg loss 0.72187793 reg_l1 12.779691 reg_l2 11.882815
loss 1.999847
STEP 694 ================================
prereg loss 0.72229785 reg_l1 12.780713 reg_l2 11.885737
loss 2.000369
STEP 695 ================================
prereg loss 0.72271407 reg_l1 12.781733 reg_l2 11.88866
loss 2.0008874
STEP 696 ================================
prereg loss 0.7231337 reg_l1 12.782756 reg_l2 11.891594
loss 2.0014093
STEP 697 ================================
prereg loss 0.7235546 reg_l1 12.783783 reg_l2 11.894529
loss 2.001933
STEP 698 ================================
prereg loss 0.72397876 reg_l1 12.78481 reg_l2 11.89747
loss 2.0024598
STEP 699 ================================
prereg loss 0.7244023 reg_l1 12.785838 reg_l2 11.900416
loss 2.0029862
STEP 700 ================================
prereg loss 0.72482705 reg_l1 12.786868 reg_l2 11.903365
loss 2.0035138
STEP 701 ================================
prereg loss 0.725253 reg_l1 12.787901 reg_l2 11.906325
loss 2.004043
STEP 702 ================================
prereg loss 0.72567993 reg_l1 12.788938 reg_l2 11.909289
loss 2.0045738
STEP 703 ================================
prereg loss 0.72611004 reg_l1 12.789969 reg_l2 11.912252
loss 2.005107
STEP 704 ================================
prereg loss 0.7265397 reg_l1 12.791006 reg_l2 11.915225
loss 2.0056403
STEP 705 ================================
prereg loss 0.72697204 reg_l1 12.7920475 reg_l2 11.918204
loss 2.0061767
STEP 706 ================================
prereg loss 0.7274052 reg_l1 12.793089 reg_l2 11.921186
loss 2.006714
STEP 707 ================================
prereg loss 0.72783905 reg_l1 12.794129 reg_l2 11.924173
loss 2.007252
STEP 708 ================================
prereg loss 0.72827554 reg_l1 12.795176 reg_l2 11.927169
loss 2.007793
STEP 709 ================================
prereg loss 0.728711 reg_l1 12.796221 reg_l2 11.930167
loss 2.0083332
STEP 710 ================================
prereg loss 0.72915286 reg_l1 12.797266 reg_l2 11.933169
loss 2.0088794
STEP 711 ================================
prereg loss 0.72959065 reg_l1 12.798319 reg_l2 11.936182
loss 2.0094225
STEP 712 ================================
prereg loss 0.73002917 reg_l1 12.799369 reg_l2 11.939197
loss 2.0099661
STEP 713 ================================
prereg loss 0.73047453 reg_l1 12.800422 reg_l2 11.942214
loss 2.0105166
STEP 714 ================================
prereg loss 0.73091835 reg_l1 12.801476 reg_l2 11.945242
loss 2.011066
STEP 715 ================================
prereg loss 0.731364 reg_l1 12.802533 reg_l2 11.948274
loss 2.0116172
STEP 716 ================================
prereg loss 0.731811 reg_l1 12.803591 reg_l2 11.951307
loss 2.01217
STEP 717 ================================
prereg loss 0.7322621 reg_l1 12.804651 reg_l2 11.954349
loss 2.0127273
STEP 718 ================================
prereg loss 0.7327097 reg_l1 12.805715 reg_l2 11.957397
loss 2.013281
STEP 719 ================================
prereg loss 0.73316264 reg_l1 12.806777 reg_l2 11.960447
loss 2.0138402
STEP 720 ================================
prereg loss 0.733614 reg_l1 12.807842 reg_l2 11.963507
loss 2.0143983
STEP 721 ================================
prereg loss 0.73406684 reg_l1 12.808909 reg_l2 11.96657
loss 2.014958
STEP 722 ================================
prereg loss 0.73452365 reg_l1 12.8099785 reg_l2 11.969639
loss 2.0155215
STEP 723 ================================
prereg loss 0.73497975 reg_l1 12.811049 reg_l2 11.972711
loss 2.0160847
STEP 724 ================================
prereg loss 0.73543817 reg_l1 12.812122 reg_l2 11.975792
loss 2.0166504
STEP 725 ================================
prereg loss 0.7358973 reg_l1 12.813198 reg_l2 11.978876
loss 2.0172172
STEP 726 ================================
prereg loss 0.7363605 reg_l1 12.814272 reg_l2 11.981964
loss 2.0177877
STEP 727 ================================
prereg loss 0.7368212 reg_l1 12.8153515 reg_l2 11.985064
loss 2.0183563
STEP 728 ================================
prereg loss 0.7372845 reg_l1 12.816431 reg_l2 11.988167
loss 2.0189276
STEP 729 ================================
prereg loss 0.7377487 reg_l1 12.817512 reg_l2 11.991271
loss 2.0194998
STEP 730 ================================
prereg loss 0.73821604 reg_l1 12.818594 reg_l2 11.994383
loss 2.0200753
STEP 731 ================================
prereg loss 0.7386854 reg_l1 12.819683 reg_l2 11.997502
loss 2.0206537
STEP 732 ================================
prereg loss 0.73915577 reg_l1 12.820768 reg_l2 12.000624
loss 2.0212326
STEP 733 ================================
prereg loss 0.739625 reg_l1 12.821857 reg_l2 12.003753
loss 2.0218108
STEP 734 ================================
prereg loss 0.7400956 reg_l1 12.8229475 reg_l2 12.00689
loss 2.0223904
STEP 735 ================================
prereg loss 0.74057025 reg_l1 12.824042 reg_l2 12.010033
loss 2.0229745
STEP 736 ================================
prereg loss 0.74104434 reg_l1 12.825134 reg_l2 12.013175
loss 2.023558
STEP 737 ================================
prereg loss 0.74152213 reg_l1 12.826231 reg_l2 12.016326
loss 2.0241454
STEP 738 ================================
prereg loss 0.7420011 reg_l1 12.827332 reg_l2 12.019486
loss 2.0247343
STEP 739 ================================
prereg loss 0.74248075 reg_l1 12.82843 reg_l2 12.022649
loss 2.0253239
STEP 740 ================================
prereg loss 0.7429655 reg_l1 12.829531 reg_l2 12.025817
loss 2.0259187
STEP 741 ================================
prereg loss 0.74344593 reg_l1 12.830634 reg_l2 12.028991
loss 2.0265093
STEP 742 ================================
prereg loss 0.7439325 reg_l1 12.831741 reg_l2 12.03217
loss 2.0271068
STEP 743 ================================
prereg loss 0.7444181 reg_l1 12.832848 reg_l2 12.035354
loss 2.0277028
STEP 744 ================================
prereg loss 0.7449065 reg_l1 12.833956 reg_l2 12.038547
loss 2.0283022
STEP 745 ================================
prereg loss 0.74539495 reg_l1 12.83507 reg_l2 12.041744
loss 2.028902
STEP 746 ================================
prereg loss 0.7458844 reg_l1 12.836182 reg_l2 12.044947
loss 2.0295026
STEP 747 ================================
prereg loss 0.7463768 reg_l1 12.8372965 reg_l2 12.048153
loss 2.0301065
STEP 748 ================================
prereg loss 0.74686754 reg_l1 12.838414 reg_l2 12.051371
loss 2.030709
STEP 749 ================================
prereg loss 0.7473635 reg_l1 12.839534 reg_l2 12.05459
loss 2.031317
STEP 750 ================================
prereg loss 0.74786097 reg_l1 12.8406515 reg_l2 12.057813
loss 2.0319262
STEP 751 ================================
prereg loss 0.7483571 reg_l1 12.841777 reg_l2 12.061048
loss 2.0325348
STEP 752 ================================
prereg loss 0.7488578 reg_l1 12.842901 reg_l2 12.064285
loss 2.033148
STEP 753 ================================
prereg loss 0.7493598 reg_l1 12.844027 reg_l2 12.067525
loss 2.0337625
STEP 754 ================================
prereg loss 0.74985844 reg_l1 12.845154 reg_l2 12.070774
loss 2.0343738
STEP 755 ================================
prereg loss 0.75036323 reg_l1 12.846286 reg_l2 12.074031
loss 2.0349917
STEP 756 ================================
prereg loss 0.75086755 reg_l1 12.84742 reg_l2 12.077293
loss 2.0356095
STEP 757 ================================
prereg loss 0.7513747 reg_l1 12.848552 reg_l2 12.080558
loss 2.03623
STEP 758 ================================
prereg loss 0.75188226 reg_l1 12.849689 reg_l2 12.083829
loss 2.0368512
STEP 759 ================================
prereg loss 0.75239456 reg_l1 12.850827 reg_l2 12.087109
loss 2.0374773
STEP 760 ================================
prereg loss 0.7529054 reg_l1 12.851967 reg_l2 12.090391
loss 2.0381021
STEP 761 ================================
prereg loss 0.7534209 reg_l1 12.853107 reg_l2 12.093682
loss 2.0387316
STEP 762 ================================
prereg loss 0.7539354 reg_l1 12.854253 reg_l2 12.096978
loss 2.0393608
STEP 763 ================================
prereg loss 0.7544496 reg_l1 12.855397 reg_l2 12.100279
loss 2.0399895
STEP 764 ================================
prereg loss 0.7549699 reg_l1 12.8565445 reg_l2 12.103587
loss 2.0406244
STEP 765 ================================
prereg loss 0.75548834 reg_l1 12.857696 reg_l2 12.106899
loss 2.0412579
STEP 766 ================================
prereg loss 0.75601083 reg_l1 12.858846 reg_l2 12.11022
loss 2.0418954
STEP 767 ================================
prereg loss 0.7565339 reg_l1 12.859999 reg_l2 12.1135435
loss 2.0425339
STEP 768 ================================
prereg loss 0.757058 reg_l1 12.861155 reg_l2 12.116876
loss 2.0431736
STEP 769 ================================
prereg loss 0.75758225 reg_l1 12.862313 reg_l2 12.120216
loss 2.0438137
STEP 770 ================================
prereg loss 0.75810933 reg_l1 12.863472 reg_l2 12.123558
loss 2.0444565
STEP 771 ================================
prereg loss 0.75863993 reg_l1 12.864631 reg_l2 12.126904
loss 2.045103
STEP 772 ================================
prereg loss 0.7591731 reg_l1 12.865797 reg_l2 12.130262
loss 2.0457528
STEP 773 ================================
prereg loss 0.7597036 reg_l1 12.866962 reg_l2 12.133625
loss 2.0463998
STEP 774 ================================
prereg loss 0.76023763 reg_l1 12.868129 reg_l2 12.13699
loss 2.0470505
STEP 775 ================================
prereg loss 0.7607715 reg_l1 12.869299 reg_l2 12.140366
loss 2.0477014
STEP 776 ================================
prereg loss 0.76130694 reg_l1 12.870472 reg_l2 12.143747
loss 2.0483541
STEP 777 ================================
prereg loss 0.761846 reg_l1 12.871643 reg_l2 12.14713
loss 2.0490103
STEP 778 ================================
prereg loss 0.76238805 reg_l1 12.872819 reg_l2 12.150522
loss 2.04967
STEP 779 ================================
prereg loss 0.7629294 reg_l1 12.873998 reg_l2 12.15392
loss 2.0503292
STEP 780 ================================
prereg loss 0.76347345 reg_l1 12.875175 reg_l2 12.157324
loss 2.050991
STEP 781 ================================
prereg loss 0.7640198 reg_l1 12.876356 reg_l2 12.160733
loss 2.0516555
STEP 782 ================================
prereg loss 0.7645686 reg_l1 12.877541 reg_l2 12.164149
loss 2.0523226
STEP 783 ================================
prereg loss 0.7651128 reg_l1 12.878727 reg_l2 12.167573
loss 2.0529854
STEP 784 ================================
prereg loss 0.7656651 reg_l1 12.879912 reg_l2 12.1710005
loss 2.0536563
STEP 785 ================================
prereg loss 0.76621735 reg_l1 12.881102 reg_l2 12.174437
loss 2.0543275
STEP 786 ================================
prereg loss 0.76677245 reg_l1 12.882295 reg_l2 12.177877
loss 2.055002
STEP 787 ================================
prereg loss 0.7673256 reg_l1 12.883487 reg_l2 12.181323
loss 2.0556743
STEP 788 ================================
prereg loss 0.76788265 reg_l1 12.884683 reg_l2 12.184776
loss 2.056351
STEP 789 ================================
prereg loss 0.768443 reg_l1 12.885881 reg_l2 12.188238
loss 2.0570312
STEP 790 ================================
prereg loss 0.7690016 reg_l1 12.887081 reg_l2 12.191704
loss 2.0577097
STEP 791 ================================
prereg loss 0.7695648 reg_l1 12.888282 reg_l2 12.195174
loss 2.058393
STEP 792 ================================
prereg loss 0.7701296 reg_l1 12.889487 reg_l2 12.198656
loss 2.0590785
STEP 793 ================================
prereg loss 0.7706943 reg_l1 12.890693 reg_l2 12.20214
loss 2.0597637
STEP 794 ================================
prereg loss 0.7712618 reg_l1 12.891901 reg_l2 12.20563
loss 2.060452
STEP 795 ================================
prereg loss 0.7718299 reg_l1 12.893112 reg_l2 12.20913
loss 2.0611413
STEP 796 ================================
prereg loss 0.7724007 reg_l1 12.894322 reg_l2 12.212634
loss 2.061833
STEP 797 ================================
prereg loss 0.77297413 reg_l1 12.895537 reg_l2 12.21614
loss 2.062528
STEP 798 ================================
prereg loss 0.7735469 reg_l1 12.896753 reg_l2 12.219658
loss 2.0632222
STEP 799 ================================
prereg loss 0.7741231 reg_l1 12.897974 reg_l2 12.223184
loss 2.0639205
STEP 800 ================================
prereg loss 0.77470225 reg_l1 12.899192 reg_l2 12.226709
loss 2.0646214
STEP 801 ================================
prereg loss 0.77527887 reg_l1 12.900415 reg_l2 12.230245
loss 2.0653205
STEP 802 ================================
prereg loss 0.7758607 reg_l1 12.901642 reg_l2 12.23379
loss 2.066025
STEP 803 ================================
prereg loss 0.7764433 reg_l1 12.902866 reg_l2 12.237336
loss 2.06673
STEP 804 ================================
prereg loss 0.77702725 reg_l1 12.904095 reg_l2 12.240887
loss 2.0674367
STEP 805 ================================
prereg loss 0.777615 reg_l1 12.905327 reg_l2 12.244453
loss 2.0681477
STEP 806 ================================
prereg loss 0.7782013 reg_l1 12.906561 reg_l2 12.24802
loss 2.0688574
STEP 807 ================================
prereg loss 0.77879536 reg_l1 12.907795 reg_l2 12.251591
loss 2.0695748
STEP 808 ================================
prereg loss 0.7793832 reg_l1 12.909032 reg_l2 12.255174
loss 2.0702863
STEP 809 ================================
prereg loss 0.77997667 reg_l1 12.910273 reg_l2 12.258764
loss 2.071004
STEP 810 ================================
prereg loss 0.78057104 reg_l1 12.911515 reg_l2 12.262355
loss 2.0717225
STEP 811 ================================
prereg loss 0.7811683 reg_l1 12.912758 reg_l2 12.265953
loss 2.0724442
STEP 812 ================================
prereg loss 0.7817656 reg_l1 12.914004 reg_l2 12.269562
loss 2.073166
STEP 813 ================================
prereg loss 0.7823668 reg_l1 12.915253 reg_l2 12.273176
loss 2.073892
STEP 814 ================================
prereg loss 0.7829694 reg_l1 12.916501 reg_l2 12.276793
loss 2.0746195
STEP 815 ================================
prereg loss 0.78357047 reg_l1 12.917753 reg_l2 12.280419
loss 2.0753458
STEP 816 ================================
prereg loss 0.7841757 reg_l1 12.91901 reg_l2 12.284054
loss 2.0760767
STEP 817 ================================
prereg loss 0.7847851 reg_l1 12.920268 reg_l2 12.287693
loss 2.076812
STEP 818 ================================
prereg loss 0.78539586 reg_l1 12.921524 reg_l2 12.291336
loss 2.0775483
STEP 819 ================================
prereg loss 0.7860079 reg_l1 12.922787 reg_l2 12.2949915
loss 2.0782866
STEP 820 ================================
prereg loss 0.78661764 reg_l1 12.924051 reg_l2 12.298652
loss 2.079023
STEP 821 ================================
prereg loss 0.78723425 reg_l1 12.925314 reg_l2 12.302312
loss 2.0797656
STEP 822 ================================
prereg loss 0.78785455 reg_l1 12.926582 reg_l2 12.305986
loss 2.0805128
STEP 823 ================================
prereg loss 0.78846866 reg_l1 12.927854 reg_l2 12.309671
loss 2.081254
STEP 824 ================================
prereg loss 0.78909093 reg_l1 12.929124 reg_l2 12.313353
loss 2.0820034
STEP 825 ================================
prereg loss 0.7897108 reg_l1 12.930397 reg_l2 12.317043
loss 2.0827506
STEP 826 ================================
prereg loss 0.7903358 reg_l1 12.931677 reg_l2 12.320749
loss 2.0835035
STEP 827 ================================
prereg loss 0.79096085 reg_l1 12.932956 reg_l2 12.324454
loss 2.0842564
STEP 828 ================================
prereg loss 0.79159194 reg_l1 12.934234 reg_l2 12.328163
loss 2.0850153
STEP 829 ================================
prereg loss 0.79221946 reg_l1 12.935518 reg_l2 12.331883
loss 2.0857713
STEP 830 ================================
prereg loss 0.79285073 reg_l1 12.936805 reg_l2 12.335615
loss 2.0865312
STEP 831 ================================
prereg loss 0.7934859 reg_l1 12.938089 reg_l2 12.339343
loss 2.0872948
STEP 832 ================================
prereg loss 0.794119 reg_l1 12.93938 reg_l2 12.343079
loss 2.088057
STEP 833 ================================
prereg loss 0.79475695 reg_l1 12.940674 reg_l2 12.346833
loss 2.0888243
STEP 834 ================================
prereg loss 0.79539156 reg_l1 12.941968 reg_l2 12.350586
loss 2.0895884
STEP 835 ================================
prereg loss 0.796034 reg_l1 12.943261 reg_l2 12.35434
loss 2.0903602
STEP 836 ================================
prereg loss 0.796677 reg_l1 12.944561 reg_l2 12.358109
loss 2.091133
STEP 837 ================================
prereg loss 0.79732144 reg_l1 12.945867 reg_l2 12.361889
loss 2.091908
STEP 838 ================================
prereg loss 0.79796857 reg_l1 12.947166 reg_l2 12.3656645
loss 2.0926852
STEP 839 ================================
prereg loss 0.79861915 reg_l1 12.948471 reg_l2 12.369451
loss 2.0934663
STEP 840 ================================
prereg loss 0.7992683 reg_l1 12.9497795 reg_l2 12.37325
loss 2.0942464
STEP 841 ================================
prereg loss 0.799921 reg_l1 12.95109 reg_l2 12.377051
loss 2.09503
STEP 842 ================================
prereg loss 0.8005759 reg_l1 12.952401 reg_l2 12.380855
loss 2.0958161
STEP 843 ================================
prereg loss 0.8012309 reg_l1 12.953716 reg_l2 12.384671
loss 2.0966024
STEP 844 ================================
prereg loss 0.80188906 reg_l1 12.955036 reg_l2 12.388498
loss 2.0973926
STEP 845 ================================
prereg loss 0.80254894 reg_l1 12.95635 reg_l2 12.392322
loss 2.098184
STEP 846 ================================
prereg loss 0.803211 reg_l1 12.957673 reg_l2 12.396159
loss 2.0989783
STEP 847 ================================
prereg loss 0.80387276 reg_l1 12.958998 reg_l2 12.400005
loss 2.0997725
STEP 848 ================================
prereg loss 0.8045407 reg_l1 12.960323 reg_l2 12.403853
loss 2.100573
STEP 849 ================================
prereg loss 0.8052097 reg_l1 12.96165 reg_l2 12.407709
loss 2.1013746
STEP 850 ================================
prereg loss 0.8058772 reg_l1 12.962984 reg_l2 12.411577
loss 2.1021757
STEP 851 ================================
prereg loss 0.8065479 reg_l1 12.964317 reg_l2 12.415447
loss 2.1029797
STEP 852 ================================
prereg loss 0.8072236 reg_l1 12.965649 reg_l2 12.419321
loss 2.1037886
STEP 853 ================================
prereg loss 0.8079006 reg_l1 12.966988 reg_l2 12.42321
loss 2.1045995
STEP 854 ================================
prereg loss 0.80857867 reg_l1 12.96833 reg_l2 12.427106
loss 2.1054118
STEP 855 ================================
prereg loss 0.8092569 reg_l1 12.96967 reg_l2 12.431002
loss 2.106224
STEP 856 ================================
prereg loss 0.8099354 reg_l1 12.971013 reg_l2 12.434907
loss 2.1070368
STEP 857 ================================
prereg loss 0.81062096 reg_l1 12.972363 reg_l2 12.438825
loss 2.1078572
STEP 858 ================================
prereg loss 0.8113082 reg_l1 12.973711 reg_l2 12.442744
loss 2.1086793
STEP 859 ================================
prereg loss 0.8119984 reg_l1 12.97506 reg_l2 12.446669
loss 2.1095045
STEP 860 ================================
prereg loss 0.81268793 reg_l1 12.976417 reg_l2 12.450606
loss 2.1103296
STEP 861 ================================
prereg loss 0.8133797 reg_l1 12.977773 reg_l2 12.454548
loss 2.111157
STEP 862 ================================
prereg loss 0.8140733 reg_l1 12.97913 reg_l2 12.458496
loss 2.1119862
STEP 863 ================================
prereg loss 0.8147689 reg_l1 12.980492 reg_l2 12.462453
loss 2.1128182
STEP 864 ================================
prereg loss 0.81546706 reg_l1 12.981855 reg_l2 12.466419
loss 2.1136527
STEP 865 ================================
prereg loss 0.816166 reg_l1 12.98322 reg_l2 12.470388
loss 2.1144881
STEP 866 ================================
prereg loss 0.8168655 reg_l1 12.984588 reg_l2 12.474363
loss 2.1153243
STEP 867 ================================
prereg loss 0.81756943 reg_l1 12.98596 reg_l2 12.4783535
loss 2.1161654
STEP 868 ================================
prereg loss 0.8182754 reg_l1 12.987333 reg_l2 12.482346
loss 2.1170087
STEP 869 ================================
prereg loss 0.81898326 reg_l1 12.988708 reg_l2 12.486346
loss 2.117854
STEP 870 ================================
prereg loss 0.8196928 reg_l1 12.990084 reg_l2 12.490353
loss 2.1187012
STEP 871 ================================
prereg loss 0.8204056 reg_l1 12.991466 reg_l2 12.4943695
loss 2.1195521
STEP 872 ================================
prereg loss 0.82112056 reg_l1 12.992847 reg_l2 12.498388
loss 2.1204054
STEP 873 ================================
prereg loss 0.8218377 reg_l1 12.994231 reg_l2 12.5024185
loss 2.121261
STEP 874 ================================
prereg loss 0.8225601 reg_l1 12.995621 reg_l2 12.506455
loss 2.1221223
STEP 875 ================================
prereg loss 0.82327974 reg_l1 12.997008 reg_l2 12.510498
loss 2.1229806
STEP 876 ================================
prereg loss 0.8240033 reg_l1 12.9984 reg_l2 12.51455
loss 2.1238432
STEP 877 ================================
prereg loss 0.8247261 reg_l1 12.999796 reg_l2 12.51861
loss 2.1247058
STEP 878 ================================
prereg loss 0.8254524 reg_l1 13.001192 reg_l2 12.5226755
loss 2.1255717
STEP 879 ================================
prereg loss 0.8261848 reg_l1 13.00259 reg_l2 12.526747
loss 2.1264439
STEP 880 ================================
prereg loss 0.82691336 reg_l1 13.003992 reg_l2 12.530828
loss 2.1273127
STEP 881 ================================
prereg loss 0.82764494 reg_l1 13.005396 reg_l2 12.53492
loss 2.1281846
STEP 882 ================================
prereg loss 0.82838434 reg_l1 13.006802 reg_l2 12.539015
loss 2.1290646
STEP 883 ================================
prereg loss 0.82911754 reg_l1 13.008211 reg_l2 12.5431185
loss 2.1299386
STEP 884 ================================
prereg loss 0.8298609 reg_l1 13.009625 reg_l2 12.547231
loss 2.1308236
STEP 885 ================================
prereg loss 0.8306017 reg_l1 13.011039 reg_l2 12.551348
loss 2.1317058
STEP 886 ================================
prereg loss 0.831345 reg_l1 13.012454 reg_l2 12.555473
loss 2.1325905
STEP 887 ================================
prereg loss 0.83208764 reg_l1 13.013873 reg_l2 12.5596075
loss 2.1334748
STEP 888 ================================
prereg loss 0.8328371 reg_l1 13.015293 reg_l2 12.563749
loss 2.1343665
STEP 889 ================================
prereg loss 0.8335879 reg_l1 13.016717 reg_l2 12.567897
loss 2.1352596
STEP 890 ================================
prereg loss 0.83434117 reg_l1 13.018143 reg_l2 12.572051
loss 2.1361556
STEP 891 ================================
prereg loss 0.8350943 reg_l1 13.019571 reg_l2 12.576216
loss 2.1370513
STEP 892 ================================
prereg loss 0.8358536 reg_l1 13.021004 reg_l2 12.580388
loss 2.137954
STEP 893 ================================
prereg loss 0.83661157 reg_l1 13.022439 reg_l2 12.584565
loss 2.1388555
STEP 894 ================================
prereg loss 0.83737224 reg_l1 13.023873 reg_l2 12.588751
loss 2.1397595
STEP 895 ================================
prereg loss 0.83813846 reg_l1 13.025313 reg_l2 12.592947
loss 2.1406698
STEP 896 ================================
prereg loss 0.8389038 reg_l1 13.026753 reg_l2 12.597148
loss 2.1415792
STEP 897 ================================
prereg loss 0.8396706 reg_l1 13.028196 reg_l2 12.6013565
loss 2.1424901
STEP 898 ================================
prereg loss 0.84044176 reg_l1 13.029643 reg_l2 12.605574
loss 2.1434062
STEP 899 ================================
prereg loss 0.8412149 reg_l1 13.031091 reg_l2 12.6098
loss 2.1443238
STEP 900 ================================
prereg loss 0.84198606 reg_l1 13.032543 reg_l2 12.61403
loss 2.1452403
STEP 901 ================================
prereg loss 0.84276575 reg_l1 13.033997 reg_l2 12.618271
loss 2.1461654
STEP 902 ================================
prereg loss 0.84354705 reg_l1 13.035454 reg_l2 12.62252
loss 2.1470923
STEP 903 ================================
prereg loss 0.84432983 reg_l1 13.036912 reg_l2 12.626776
loss 2.1480212
STEP 904 ================================
prereg loss 0.8451076 reg_l1 13.038374 reg_l2 12.631039
loss 2.148945
STEP 905 ================================
prereg loss 0.8458933 reg_l1 13.039837 reg_l2 12.63531
loss 2.149877
STEP 906 ================================
prereg loss 0.8466844 reg_l1 13.041301 reg_l2 12.639589
loss 2.1508145
STEP 907 ================================
prereg loss 0.8474732 reg_l1 13.042773 reg_l2 12.643877
loss 2.1517506
STEP 908 ================================
prereg loss 0.8482652 reg_l1 13.044244 reg_l2 12.648172
loss 2.1526895
STEP 909 ================================
prereg loss 0.8490609 reg_l1 13.045717 reg_l2 12.652471
loss 2.1536326
STEP 910 ================================
prereg loss 0.84986174 reg_l1 13.0471945 reg_l2 12.656783
loss 2.154581
STEP 911 ================================
prereg loss 0.85066074 reg_l1 13.048676 reg_l2 12.661103
loss 2.1555283
STEP 912 ================================
prereg loss 0.85145885 reg_l1 13.050154 reg_l2 12.665427
loss 2.1564744
STEP 913 ================================
prereg loss 0.8522639 reg_l1 13.05164 reg_l2 12.669762
loss 2.1574278
STEP 914 ================================
prereg loss 0.853072 reg_l1 13.053127 reg_l2 12.674105
loss 2.1583848
STEP 915 ================================
prereg loss 0.85388154 reg_l1 13.054615 reg_l2 12.678453
loss 2.159343
STEP 916 ================================
prereg loss 0.85469145 reg_l1 13.056108 reg_l2 12.68281
loss 2.1603024
STEP 917 ================================
prereg loss 0.85550493 reg_l1 13.057604 reg_l2 12.687177
loss 2.1612654
STEP 918 ================================
prereg loss 0.8563183 reg_l1 13.059101 reg_l2 12.691548
loss 2.1622283
STEP 919 ================================
prereg loss 0.85713816 reg_l1 13.0606 reg_l2 12.695931
loss 2.1631982
STEP 920 ================================
prereg loss 0.85795933 reg_l1 13.062103 reg_l2 12.700324
loss 2.1641698
STEP 921 ================================
prereg loss 0.85878134 reg_l1 13.063608 reg_l2 12.70472
loss 2.165142
STEP 922 ================================
prereg loss 0.85960525 reg_l1 13.065115 reg_l2 12.709122
loss 2.1661167
STEP 923 ================================
prereg loss 0.86043555 reg_l1 13.066625 reg_l2 12.713535
loss 2.167098
STEP 924 ================================
prereg loss 0.86126214 reg_l1 13.06814 reg_l2 12.71796
loss 2.1680763
STEP 925 ================================
prereg loss 0.8620952 reg_l1 13.069654 reg_l2 12.722388
loss 2.1690607
STEP 926 ================================
prereg loss 0.86292976 reg_l1 13.071172 reg_l2 12.726824
loss 2.170047
STEP 927 ================================
prereg loss 0.86376524 reg_l1 13.072695 reg_l2 12.731272
loss 2.1710348
STEP 928 ================================
prereg loss 0.8646073 reg_l1 13.074219 reg_l2 12.735728
loss 2.1720293
STEP 929 ================================
prereg loss 0.86544895 reg_l1 13.075743 reg_l2 12.740186
loss 2.1730232
STEP 930 ================================
prereg loss 0.86629355 reg_l1 13.0772705 reg_l2 12.744657
loss 2.1740208
STEP 931 ================================
prereg loss 0.8671375 reg_l1 13.078804 reg_l2 12.749136
loss 2.1750178
STEP 932 ================================
prereg loss 0.86798763 reg_l1 13.080339 reg_l2 12.753619
loss 2.1760216
STEP 933 ================================
prereg loss 0.86884093 reg_l1 13.081873 reg_l2 12.758116
loss 2.1770282
STEP 934 ================================
prereg loss 0.86969244 reg_l1 13.083414 reg_l2 12.762619
loss 2.1780338
STEP 935 ================================
prereg loss 0.87054944 reg_l1 13.084956 reg_l2 12.767132
loss 2.1790452
STEP 936 ================================
prereg loss 0.8714124 reg_l1 13.0865 reg_l2 12.7716465
loss 2.1800623
STEP 937 ================================
prereg loss 0.872267 reg_l1 13.088047 reg_l2 12.776176
loss 2.1810718
STEP 938 ================================
prereg loss 0.8731325 reg_l1 13.089599 reg_l2 12.780713
loss 2.1820924
STEP 939 ================================
prereg loss 0.8739984 reg_l1 13.09115 reg_l2 12.7852545
loss 2.1831136
STEP 940 ================================
prereg loss 0.8748644 reg_l1 13.092706 reg_l2 12.789808
loss 2.184135
STEP 941 ================================
prereg loss 0.8757386 reg_l1 13.094263 reg_l2 12.794368
loss 2.185165
STEP 942 ================================
prereg loss 0.8766104 reg_l1 13.095824 reg_l2 12.798937
loss 2.186193
STEP 943 ================================
prereg loss 0.8774841 reg_l1 13.097389 reg_l2 12.803514
loss 2.187223
STEP 944 ================================
prereg loss 0.87836224 reg_l1 13.098955 reg_l2 12.8081
loss 2.1882577
STEP 945 ================================
prereg loss 0.87924457 reg_l1 13.100525 reg_l2 12.812694
loss 2.1892972
STEP 946 ================================
prereg loss 0.88012683 reg_l1 13.102097 reg_l2 12.817294
loss 2.1903365
STEP 947 ================================
prereg loss 0.88101196 reg_l1 13.103669 reg_l2 12.821906
loss 2.191379
STEP 948 ================================
prereg loss 0.88190156 reg_l1 13.105248 reg_l2 12.826523
loss 2.1924264
STEP 949 ================================
prereg loss 0.8827919 reg_l1 13.106829 reg_l2 12.831153
loss 2.1934748
STEP 950 ================================
prereg loss 0.8836823 reg_l1 13.108409 reg_l2 12.835785
loss 2.1945233
STEP 951 ================================
prereg loss 0.884577 reg_l1 13.109997 reg_l2 12.84043
loss 2.1955767
STEP 952 ================================
prereg loss 0.8854776 reg_l1 13.111585 reg_l2 12.845082
loss 2.1966362
STEP 953 ================================
prereg loss 0.8863766 reg_l1 13.113174 reg_l2 12.849744
loss 2.197694
STEP 954 ================================
prereg loss 0.8872808 reg_l1 13.114769 reg_l2 12.854412
loss 2.1987576
STEP 955 ================================
prereg loss 0.888185 reg_l1 13.116365 reg_l2 12.859092
loss 2.1998215
STEP 956 ================================
prereg loss 0.88909537 reg_l1 13.117965 reg_l2 12.863778
loss 2.200892
STEP 957 ================================
prereg loss 0.8900044 reg_l1 13.119567 reg_l2 12.868473
loss 2.201961
STEP 958 ================================
prereg loss 0.8909204 reg_l1 13.12117 reg_l2 12.873175
loss 2.2030373
STEP 959 ================================
prereg loss 0.89183575 reg_l1 13.122777 reg_l2 12.877887
loss 2.2041135
STEP 960 ================================
prereg loss 0.8927539 reg_l1 13.124391 reg_l2 12.88261
loss 2.205193
STEP 961 ================================
prereg loss 0.8936734 reg_l1 13.126001 reg_l2 12.887338
loss 2.2062736
STEP 962 ================================
prereg loss 0.8945972 reg_l1 13.127617 reg_l2 12.892072
loss 2.2073588
STEP 963 ================================
prereg loss 0.89552265 reg_l1 13.129236 reg_l2 12.896822
loss 2.2084463
STEP 964 ================================
prereg loss 0.8964514 reg_l1 13.130859 reg_l2 12.901576
loss 2.2095373
STEP 965 ================================
prereg loss 0.8973813 reg_l1 13.132481 reg_l2 12.906336
loss 2.2106295
STEP 966 ================================
prereg loss 0.8983172 reg_l1 13.134108 reg_l2 12.91111
loss 2.211728
STEP 967 ================================
prereg loss 0.89925236 reg_l1 13.13574 reg_l2 12.915894
loss 2.2128265
STEP 968 ================================
prereg loss 0.9001914 reg_l1 13.137371 reg_l2 12.920679
loss 2.2139287
STEP 969 ================================
prereg loss 0.90113306 reg_l1 13.139008 reg_l2 12.925478
loss 2.215034
STEP 970 ================================
prereg loss 0.9020755 reg_l1 13.140647 reg_l2 12.930287
loss 2.2161403
STEP 971 ================================
prereg loss 0.90302235 reg_l1 13.142286 reg_l2 12.9351015
loss 2.217251
STEP 972 ================================
prereg loss 0.90397346 reg_l1 13.143929 reg_l2 12.939924
loss 2.2183664
STEP 973 ================================
prereg loss 0.9049256 reg_l1 13.145578 reg_l2 12.944757
loss 2.2194834
STEP 974 ================================
prereg loss 0.90588206 reg_l1 13.147228 reg_l2 12.949601
loss 2.220605
STEP 975 ================================
prereg loss 0.9068364 reg_l1 13.148877 reg_l2 12.954449
loss 2.221724
STEP 976 ================================
prereg loss 0.9078008 reg_l1 13.150534 reg_l2 12.959308
loss 2.2228541
STEP 977 ================================
prereg loss 0.90876025 reg_l1 13.152194 reg_l2 12.964178
loss 2.2239797
STEP 978 ================================
prereg loss 0.9097248 reg_l1 13.153854 reg_l2 12.96905
loss 2.2251103
STEP 979 ================================
prereg loss 0.9106968 reg_l1 13.155517 reg_l2 12.973936
loss 2.2262485
STEP 980 ================================
prereg loss 0.9116654 reg_l1 13.157186 reg_l2 12.978835
loss 2.2273839
STEP 981 ================================
prereg loss 0.91263974 reg_l1 13.158855 reg_l2 12.983739
loss 2.2285252
STEP 982 ================================
prereg loss 0.91361296 reg_l1 13.160525 reg_l2 12.988646
loss 2.2296655
STEP 983 ================================
prereg loss 0.91459084 reg_l1 13.162203 reg_l2 12.99357
loss 2.230811
STEP 984 ================================
prereg loss 0.91557384 reg_l1 13.163882 reg_l2 12.998504
loss 2.2319622
STEP 985 ================================
prereg loss 0.9165569 reg_l1 13.16556 reg_l2 13.003436
loss 2.2331128
STEP 986 ================================
prereg loss 0.9175441 reg_l1 13.167245 reg_l2 13.008384
loss 2.2342687
STEP 987 ================================
prereg loss 0.9185348 reg_l1 13.168934 reg_l2 13.013346
loss 2.2354283
STEP 988 ================================
prereg loss 0.9195306 reg_l1 13.170622 reg_l2 13.018311
loss 2.2365928
STEP 989 ================================
prereg loss 0.9205214 reg_l1 13.172316 reg_l2 13.023281
loss 2.237753
STEP 990 ================================
prereg loss 0.9215193 reg_l1 13.174012 reg_l2 13.028269
loss 2.2389207
STEP 991 ================================
prereg loss 0.9225218 reg_l1 13.17571 reg_l2 13.033262
loss 2.2400928
STEP 992 ================================
prereg loss 0.92352355 reg_l1 13.177412 reg_l2 13.038261
loss 2.2412648
STEP 993 ================================
prereg loss 0.9245286 reg_l1 13.179114 reg_l2 13.043272
loss 2.2424402
STEP 994 ================================
prereg loss 0.92553455 reg_l1 13.180824 reg_l2 13.048295
loss 2.243617
STEP 995 ================================
prereg loss 0.92655015 reg_l1 13.182534 reg_l2 13.053324
loss 2.2448034
STEP 996 ================================
prereg loss 0.92756057 reg_l1 13.184246 reg_l2 13.058359
loss 2.245985
STEP 997 ================================
prereg loss 0.9285793 reg_l1 13.185962 reg_l2 13.063408
loss 2.2471755
STEP 998 ================================
prereg loss 0.9296011 reg_l1 13.187683 reg_l2 13.068465
loss 2.2483695
STEP 999 ================================
prereg loss 0.9306208 reg_l1 13.189403 reg_l2 13.073528
loss 2.249561
STEP 1000 ================================
prereg loss 0.93164575 reg_l1 13.191128 reg_l2 13.078602
loss 2.2507586
2022-06-28T21:44:09.718

julia> open("sparse21-after-2000-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse21-after-2000-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse21-after-2000-steps-opt.ser", opt)

julia> count_interval(trainable["network_matrix"], -0.3f0, 0.3f0)
1

julia> count_interval(trainable["network_matrix"], -0.34f0, 0.34f0)
2

julia> sparse1 = sparsecopy(trainable["network_matrix"], 0.3f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse1
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
16

julia> steps!(100)
2022-06-28T21:49:23.284
STEP 1 ================================
prereg loss 1.215429 reg_l1 13.049709 reg_l2 13.063195
loss 2.5204
STEP 2 ================================
prereg loss 1.2159975 reg_l1 13.051675 reg_l2 13.068277
loss 2.521165
STEP 3 ================================
prereg loss 1.213847 reg_l1 13.054424 reg_l2 13.073324
loss 2.5192895
STEP 4 ================================
prereg loss 1.2095808 reg_l1 13.05782 reg_l2 13.078344
loss 2.5153627
STEP 5 ================================
prereg loss 1.2037376 reg_l1 13.061732 reg_l2 13.083342
loss 2.5099108
STEP 6 ================================
prereg loss 1.1968219 reg_l1 13.066061 reg_l2 13.088322
loss 2.503428
STEP 7 ================================
prereg loss 1.1892521 reg_l1 13.070717 reg_l2 13.093292
loss 2.4963238
STEP 8 ================================
prereg loss 1.18139 reg_l1 13.075619 reg_l2 13.098253
loss 2.488952
STEP 9 ================================
prereg loss 1.1735281 reg_l1 13.080706 reg_l2 13.103213
loss 2.4815986
STEP 10 ================================
prereg loss 1.1658908 reg_l1 13.085918 reg_l2 13.108165
loss 2.4744825
STEP 11 ================================
prereg loss 1.1586584 reg_l1 13.091206 reg_l2 13.113115
loss 2.467779
STEP 12 ================================
prereg loss 1.1519685 reg_l1 13.096525 reg_l2 13.118065
loss 2.461621
STEP 13 ================================
prereg loss 1.1459113 reg_l1 13.101843 reg_l2 13.123008
loss 2.4560957
STEP 14 ================================
prereg loss 1.1405377 reg_l1 13.107116 reg_l2 13.127944
loss 2.4512494
STEP 15 ================================
prereg loss 1.135877 reg_l1 13.112324 reg_l2 13.132872
loss 2.4471095
STEP 16 ================================
prereg loss 1.1319324 reg_l1 13.117439 reg_l2 13.137793
loss 2.4436765
STEP 17 ================================
prereg loss 1.1286756 reg_l1 13.122435 reg_l2 13.142694
loss 2.440919
STEP 18 ================================
prereg loss 1.1260773 reg_l1 13.127295 reg_l2 13.147584
loss 2.4388068
STEP 19 ================================
prereg loss 1.1240768 reg_l1 13.132002 reg_l2 13.152456
loss 2.437277
STEP 20 ================================
prereg loss 1.1226254 reg_l1 13.136544 reg_l2 13.157305
loss 2.4362798
STEP 21 ================================
prereg loss 1.1216451 reg_l1 13.140907 reg_l2 13.162132
loss 2.4357357
STEP 22 ================================
prereg loss 1.121069 reg_l1 13.145084 reg_l2 13.166928
loss 2.4355774
STEP 23 ================================
prereg loss 1.1208197 reg_l1 13.149069 reg_l2 13.1717005
loss 2.4357266
STEP 24 ================================
prereg loss 1.1208367 reg_l1 13.152857 reg_l2 13.176437
loss 2.4361224
STEP 25 ================================
prereg loss 1.1210408 reg_l1 13.1564455 reg_l2 13.181143
loss 2.4366856
STEP 26 ================================
prereg loss 1.1213824 reg_l1 13.159836 reg_l2 13.185815
loss 2.437366
STEP 27 ================================
prereg loss 1.1217952 reg_l1 13.16303 reg_l2 13.190449
loss 2.4380982
STEP 28 ================================
prereg loss 1.1222336 reg_l1 13.166031 reg_l2 13.195046
loss 2.4388366
STEP 29 ================================
prereg loss 1.1226555 reg_l1 13.168844 reg_l2 13.199611
loss 2.43954
STEP 30 ================================
prereg loss 1.12303 reg_l1 13.171478 reg_l2 13.204136
loss 2.440178
STEP 31 ================================
prereg loss 1.1233286 reg_l1 13.173939 reg_l2 13.20862
loss 2.4407225
STEP 32 ================================
prereg loss 1.123536 reg_l1 13.176235 reg_l2 13.213074
loss 2.4411595
STEP 33 ================================
prereg loss 1.1236395 reg_l1 13.17838 reg_l2 13.217494
loss 2.4414775
STEP 34 ================================
prereg loss 1.1236268 reg_l1 13.180374 reg_l2 13.221876
loss 2.4416642
STEP 35 ================================
prereg loss 1.1235018 reg_l1 13.18224 reg_l2 13.22623
loss 2.4417257
STEP 36 ================================
prereg loss 1.1232705 reg_l1 13.183986 reg_l2 13.230558
loss 2.441669
STEP 37 ================================
prereg loss 1.1229368 reg_l1 13.185619 reg_l2 13.234856
loss 2.4414988
STEP 38 ================================
prereg loss 1.1225073 reg_l1 13.187152 reg_l2 13.23913
loss 2.4412227
STEP 39 ================================
prereg loss 1.1220053 reg_l1 13.188601 reg_l2 13.243384
loss 2.4408655
STEP 40 ================================
prereg loss 1.1214272 reg_l1 13.189974 reg_l2 13.2476225
loss 2.4404244
STEP 41 ================================
prereg loss 1.1208019 reg_l1 13.1912775 reg_l2 13.251839
loss 2.4399297
STEP 42 ================================
prereg loss 1.1201342 reg_l1 13.19253 reg_l2 13.256048
loss 2.4393873
STEP 43 ================================
prereg loss 1.119436 reg_l1 13.193741 reg_l2 13.26025
loss 2.43881
STEP 44 ================================
prereg loss 1.118726 reg_l1 13.194912 reg_l2 13.2644415
loss 2.4382172
STEP 45 ================================
prereg loss 1.1180166 reg_l1 13.196058 reg_l2 13.268627
loss 2.4376225
STEP 46 ================================
prereg loss 1.1173075 reg_l1 13.19719 reg_l2 13.272813
loss 2.4370265
STEP 47 ================================
prereg loss 1.1166197 reg_l1 13.198311 reg_l2 13.276999
loss 2.436451
STEP 48 ================================
prereg loss 1.1159457 reg_l1 13.199428 reg_l2 13.28119
loss 2.4358885
STEP 49 ================================
prereg loss 1.1153033 reg_l1 13.200548 reg_l2 13.285377
loss 2.435358
STEP 50 ================================
prereg loss 1.1146897 reg_l1 13.201681 reg_l2 13.289573
loss 2.4348578
STEP 51 ================================
prereg loss 1.1141078 reg_l1 13.202827 reg_l2 13.293779
loss 2.4343905
STEP 52 ================================
prereg loss 1.1135557 reg_l1 13.203991 reg_l2 13.297987
loss 2.4339547
STEP 53 ================================
prereg loss 1.113039 reg_l1 13.205179 reg_l2 13.302201
loss 2.433557
STEP 54 ================================
prereg loss 1.1125523 reg_l1 13.206398 reg_l2 13.306432
loss 2.4331923
STEP 55 ================================
prereg loss 1.1120942 reg_l1 13.207643 reg_l2 13.310665
loss 2.4328585
STEP 56 ================================
prereg loss 1.1116595 reg_l1 13.208916 reg_l2 13.314907
loss 2.4325511
STEP 57 ================================
prereg loss 1.1112497 reg_l1 13.210222 reg_l2 13.319159
loss 2.432272
STEP 58 ================================
prereg loss 1.110861 reg_l1 13.211565 reg_l2 13.323425
loss 2.4320173
STEP 59 ================================
prereg loss 1.1104908 reg_l1 13.212936 reg_l2 13.327693
loss 2.4317846
STEP 60 ================================
prereg loss 1.1101373 reg_l1 13.214344 reg_l2 13.331972
loss 2.4315717
STEP 61 ================================
prereg loss 1.1097884 reg_l1 13.215791 reg_l2 13.336264
loss 2.4313674
STEP 62 ================================
prereg loss 1.109454 reg_l1 13.217264 reg_l2 13.340559
loss 2.4311805
STEP 63 ================================
prereg loss 1.1091298 reg_l1 13.218771 reg_l2 13.344861
loss 2.431007
STEP 64 ================================
prereg loss 1.1088091 reg_l1 13.22031 reg_l2 13.349173
loss 2.43084
STEP 65 ================================
prereg loss 1.1084964 reg_l1 13.22188 reg_l2 13.353495
loss 2.4306846
STEP 66 ================================
prereg loss 1.108185 reg_l1 13.223477 reg_l2 13.357818
loss 2.430533
STEP 67 ================================
prereg loss 1.1078796 reg_l1 13.2251005 reg_l2 13.362151
loss 2.43039
STEP 68 ================================
prereg loss 1.1075754 reg_l1 13.226751 reg_l2 13.366495
loss 2.4302506
STEP 69 ================================
prereg loss 1.1072779 reg_l1 13.228424 reg_l2 13.370837
loss 2.4301205
STEP 70 ================================
prereg loss 1.1069875 reg_l1 13.230116 reg_l2 13.375191
loss 2.429999
STEP 71 ================================
prereg loss 1.1066982 reg_l1 13.231832 reg_l2 13.37955
loss 2.4298813
STEP 72 ================================
prereg loss 1.1064123 reg_l1 13.2335615 reg_l2 13.38391
loss 2.4297686
STEP 73 ================================
prereg loss 1.1061296 reg_l1 13.235307 reg_l2 13.388278
loss 2.4296603
STEP 74 ================================
prereg loss 1.1058592 reg_l1 13.237067 reg_l2 13.392653
loss 2.429566
STEP 75 ================================
prereg loss 1.1055862 reg_l1 13.238838 reg_l2 13.39703
loss 2.42947
STEP 76 ================================
prereg loss 1.105328 reg_l1 13.240617 reg_l2 13.40141
loss 2.4293897
STEP 77 ================================
prereg loss 1.1050702 reg_l1 13.242408 reg_l2 13.405798
loss 2.429311
STEP 78 ================================
prereg loss 1.104815 reg_l1 13.244202 reg_l2 13.410186
loss 2.4292352
STEP 79 ================================
prereg loss 1.1045651 reg_l1 13.245998 reg_l2 13.4145775
loss 2.429165
STEP 80 ================================
prereg loss 1.1043187 reg_l1 13.247804 reg_l2 13.418977
loss 2.429099
STEP 81 ================================
prereg loss 1.1040794 reg_l1 13.249609 reg_l2 13.42337
loss 2.4290404
STEP 82 ================================
prereg loss 1.1038477 reg_l1 13.251412 reg_l2 13.427771
loss 2.428989
STEP 83 ================================
prereg loss 1.1036105 reg_l1 13.253217 reg_l2 13.432172
loss 2.4289322
STEP 84 ================================
prereg loss 1.1033788 reg_l1 13.255019 reg_l2 13.436575
loss 2.4288807
STEP 85 ================================
prereg loss 1.1031423 reg_l1 13.256816 reg_l2 13.440981
loss 2.428824
STEP 86 ================================
prereg loss 1.1029135 reg_l1 13.258615 reg_l2 13.445388
loss 2.4287748
STEP 87 ================================
prereg loss 1.1026828 reg_l1 13.260407 reg_l2 13.449793
loss 2.4287236
STEP 88 ================================
prereg loss 1.1024534 reg_l1 13.262194 reg_l2 13.4542
loss 2.4286728
STEP 89 ================================
prereg loss 1.1022282 reg_l1 13.263976 reg_l2 13.45861
loss 2.4286258
STEP 90 ================================
prereg loss 1.1019981 reg_l1 13.265755 reg_l2 13.463019
loss 2.4285736
STEP 91 ================================
prereg loss 1.1017653 reg_l1 13.267525 reg_l2 13.467428
loss 2.4285178
STEP 92 ================================
prereg loss 1.1015365 reg_l1 13.269292 reg_l2 13.471841
loss 2.4284658
STEP 93 ================================
prereg loss 1.1013081 reg_l1 13.27105 reg_l2 13.476252
loss 2.4284132
STEP 94 ================================
prereg loss 1.1010761 reg_l1 13.272805 reg_l2 13.480664
loss 2.4283566
STEP 95 ================================
prereg loss 1.1008496 reg_l1 13.274551 reg_l2 13.48508
loss 2.4283047
STEP 96 ================================
prereg loss 1.1006209 reg_l1 13.276295 reg_l2 13.489492
loss 2.4282503
STEP 97 ================================
prereg loss 1.1003896 reg_l1 13.278033 reg_l2 13.493906
loss 2.428193
STEP 98 ================================
prereg loss 1.1001633 reg_l1 13.279763 reg_l2 13.498325
loss 2.4281397
STEP 99 ================================
prereg loss 1.0999361 reg_l1 13.281489 reg_l2 13.50274
loss 2.428085
STEP 100 ================================
prereg loss 1.0997108 reg_l1 13.283211 reg_l2 13.50716
loss 2.428032
2022-06-28T21:53:22.576

julia> open("sparse22-after-100-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> serialize("sparse22-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse22-after-100-steps-opt.ser", opt)

julia> # no, we needed to reinitialize the opt

julia> sparse1 = sparsecopy(sparse, 0.3f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse1
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
16

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-28T21:56:33.454
STEP 1 ================================
prereg loss 1.215429 reg_l1 13.049709 reg_l2 13.063195
loss 2.5204
STEP 2 ================================
prereg loss 1.2166803 reg_l1 13.05171 reg_l2 13.075222
loss 2.5218513
STEP 3 ================================
prereg loss 1.2222769 reg_l1 13.050731 reg_l2 13.075281
loss 2.52735
STEP 4 ================================
prereg loss 1.2238262 reg_l1 13.058777 reg_l2 13.0909
loss 2.5297039
STEP 5 ================================
prereg loss 1.2230977 reg_l1 13.067585 reg_l2 13.108296
loss 2.5298562
STEP 6 ================================
prereg loss 1.2199863 reg_l1 13.075574 reg_l2 13.125292
loss 2.5275438
STEP 7 ================================
prereg loss 1.2161016 reg_l1 13.082665 reg_l2 13.141725
loss 2.5243683
STEP 8 ================================
prereg loss 1.2130735 reg_l1 13.089442 reg_l2 13.158113
loss 2.5220177
STEP 9 ================================
prereg loss 1.2111576 reg_l1 13.096484 reg_l2 13.174938
loss 2.5208058
STEP 10 ================================
prereg loss 1.2101507 reg_l1 13.10384 reg_l2 13.192133
loss 2.5205348
STEP 11 ================================
prereg loss 1.2100503 reg_l1 13.1111555 reg_l2 13.209216
loss 2.5211658
STEP 12 ================================
prereg loss 1.2105631 reg_l1 13.11794 reg_l2 13.225582
loss 2.522357
STEP 13 ================================
prereg loss 1.2112445 reg_l1 13.1238785 reg_l2 13.240865
loss 2.5236323
STEP 14 ================================
prereg loss 1.2120111 reg_l1 13.129032 reg_l2 13.255134
loss 2.5249143
STEP 15 ================================
prereg loss 1.2131307 reg_l1 13.133717 reg_l2 13.268774
loss 2.5265024
STEP 16 ================================
prereg loss 1.21481 reg_l1 13.138339 reg_l2 13.282271
loss 2.528644
STEP 17 ================================
prereg loss 1.2169782 reg_l1 13.143222 reg_l2 13.296044
loss 2.5313005
STEP 18 ================================
prereg loss 1.2194979 reg_l1 13.148462 reg_l2 13.310265
loss 2.5343442
STEP 19 ================================
prereg loss 1.2221905 reg_l1 13.153944 reg_l2 13.324849
loss 2.5375848
STEP 20 ================================
prereg loss 1.2247788 reg_l1 13.159454 reg_l2 13.339594
loss 2.5407243
STEP 21 ================================
prereg loss 1.226991 reg_l1 13.16485 reg_l2 13.354359
loss 2.543476
STEP 22 ================================
prereg loss 1.2288178 reg_l1 13.17015 reg_l2 13.369155
loss 2.5458329
STEP 23 ================================
prereg loss 1.2305281 reg_l1 13.175504 reg_l2 13.384125
loss 2.5480785
STEP 24 ================================
prereg loss 1.2323393 reg_l1 13.181086 reg_l2 13.399439
loss 2.550448
STEP 25 ================================
prereg loss 1.2343869 reg_l1 13.186991 reg_l2 13.41516
loss 2.553086
STEP 26 ================================
prereg loss 1.2367078 reg_l1 13.193142 reg_l2 13.431182
loss 2.5560222
STEP 27 ================================
prereg loss 1.2392455 reg_l1 13.199331 reg_l2 13.447259
loss 2.5591788
STEP 28 ================================
prereg loss 1.2418872 reg_l1 13.205349 reg_l2 13.4631405
loss 2.5624223
STEP 29 ================================
prereg loss 1.2445552 reg_l1 13.211097 reg_l2 13.4787035
loss 2.5656648
STEP 30 ================================
prereg loss 1.2473415 reg_l1 13.216619 reg_l2 13.493988
loss 2.5690033
STEP 31 ================================
prereg loss 1.2503822 reg_l1 13.222055 reg_l2 13.509152
loss 2.5725877
STEP 32 ================================
prereg loss 1.2537761 reg_l1 13.227561 reg_l2 13.524371
loss 2.5765324
STEP 33 ================================
prereg loss 1.257481 reg_l1 13.233192 reg_l2 13.539721
loss 2.5808003
STEP 34 ================================
prereg loss 1.2614119 reg_l1 13.238897 reg_l2 13.555169
loss 2.5853016
STEP 35 ================================
prereg loss 1.26541 reg_l1 13.244567 reg_l2 13.570621
loss 2.5898666
STEP 36 ================================
prereg loss 1.2693334 reg_l1 13.250131 reg_l2 13.586018
loss 2.5943465
STEP 37 ================================
prereg loss 1.2731599 reg_l1 13.255608 reg_l2 13.6013975
loss 2.5987206
STEP 38 ================================
prereg loss 1.2769401 reg_l1 13.261102 reg_l2 13.6168785
loss 2.6030502
STEP 39 ================================
prereg loss 1.2807962 reg_l1 13.266726 reg_l2 13.632584
loss 2.6074686
STEP 40 ================================
prereg loss 1.2847673 reg_l1 13.272529 reg_l2 13.648562
loss 2.61202
STEP 41 ================================
prereg loss 1.2888291 reg_l1 13.278456 reg_l2 13.664742
loss 2.6166747
STEP 42 ================================
prereg loss 1.2929331 reg_l1 13.284388 reg_l2 13.680994
loss 2.6213717
STEP 43 ================================
prereg loss 1.2969924 reg_l1 13.29023 reg_l2 13.697203
loss 2.6260154
STEP 44 ================================
prereg loss 1.3010534 reg_l1 13.295957 reg_l2 13.713331
loss 2.630649
STEP 45 ================================
prereg loss 1.3051999 reg_l1 13.301626 reg_l2 13.72943
loss 2.6353626
STEP 46 ================================
prereg loss 1.309536 reg_l1 13.30732 reg_l2 13.745582
loss 2.6402678
STEP 47 ================================
prereg loss 1.3140869 reg_l1 13.313078 reg_l2 13.761823
loss 2.6453948
STEP 48 ================================
prereg loss 1.3188323 reg_l1 13.318869 reg_l2 13.778122
loss 2.6507192
STEP 49 ================================
prereg loss 1.323669 reg_l1 13.32463 reg_l2 13.794418
loss 2.656132
STEP 50 ================================
prereg loss 1.328539 reg_l1 13.330313 reg_l2 13.810666
loss 2.6615703
STEP 51 ================================
prereg loss 1.3334199 reg_l1 13.335935 reg_l2 13.82689
loss 2.6670134
STEP 52 ================================
prereg loss 1.3383528 reg_l1 13.341569 reg_l2 13.843174
loss 2.6725097
STEP 53 ================================
prereg loss 1.3433758 reg_l1 13.347288 reg_l2 13.859601
loss 2.6781046
STEP 54 ================================
prereg loss 1.3485107 reg_l1 13.353117 reg_l2 13.876198
loss 2.6838226
STEP 55 ================================
prereg loss 1.353706 reg_l1 13.359026 reg_l2 13.892939
loss 2.6896086
STEP 56 ================================
prereg loss 1.3588973 reg_l1 13.364955 reg_l2 13.909757
loss 2.6953928
STEP 57 ================================
prereg loss 1.364065 reg_l1 13.370863 reg_l2 13.926608
loss 2.7011514
STEP 58 ================================
prereg loss 1.3692368 reg_l1 13.376763 reg_l2 13.943502
loss 2.7069132
STEP 59 ================================
prereg loss 1.3744825 reg_l1 13.382701 reg_l2 13.960478
loss 2.7127526
STEP 60 ================================
prereg loss 1.3798394 reg_l1 13.3887005 reg_l2 13.977562
loss 2.7187095
STEP 61 ================================
prereg loss 1.3853351 reg_l1 13.394755 reg_l2 13.994745
loss 2.7248106
STEP 62 ================================
prereg loss 1.3909079 reg_l1 13.400822 reg_l2 14.011976
loss 2.73099
STEP 63 ================================
prereg loss 1.3965448 reg_l1 13.406856 reg_l2 14.02921
loss 2.7372303
STEP 64 ================================
prereg loss 1.4022259 reg_l1 13.412849 reg_l2 14.046439
loss 2.7435107
STEP 65 ================================
prereg loss 1.4079758 reg_l1 13.418832 reg_l2 14.063697
loss 2.7498589
STEP 66 ================================
prereg loss 1.413826 reg_l1 13.424848 reg_l2 14.081034
loss 2.756311
STEP 67 ================================
prereg loss 1.4197924 reg_l1 13.430914 reg_l2 14.09847
loss 2.7628837
STEP 68 ================================
prereg loss 1.4258353 reg_l1 13.437015 reg_l2 14.115991
loss 2.7695367
STEP 69 ================================
prereg loss 1.4319016 reg_l1 13.44312 reg_l2 14.133572
loss 2.7762136
STEP 70 ================================
prereg loss 1.4379826 reg_l1 13.449219 reg_l2 14.151195
loss 2.7829046
STEP 71 ================================
prereg loss 1.444078 reg_l1 13.455322 reg_l2 14.168879
loss 2.7896104
STEP 72 ================================
prereg loss 1.4502299 reg_l1 13.461459 reg_l2 14.186654
loss 2.7963758
STEP 73 ================================
prereg loss 1.4564637 reg_l1 13.4676485 reg_l2 14.204534
loss 2.8032286
STEP 74 ================================
prereg loss 1.462772 reg_l1 13.473877 reg_l2 14.222508
loss 2.8101597
STEP 75 ================================
prereg loss 1.4691347 reg_l1 13.480124 reg_l2 14.240545
loss 2.8171473
STEP 76 ================================
prereg loss 1.4755348 reg_l1 13.486362 reg_l2 14.25862
loss 2.824171
STEP 77 ================================
prereg loss 1.4819775 reg_l1 13.492597 reg_l2 14.276734
loss 2.8312373
STEP 78 ================================
prereg loss 1.4884918 reg_l1 13.498844 reg_l2 14.294907
loss 2.838376
STEP 79 ================================
prereg loss 1.4951125 reg_l1 13.505126 reg_l2 14.31315
loss 2.8456252
STEP 80 ================================
prereg loss 1.5018225 reg_l1 13.511437 reg_l2 14.331474
loss 2.8529663
STEP 81 ================================
prereg loss 1.5086069 reg_l1 13.517766 reg_l2 14.349856
loss 2.8603835
STEP 82 ================================
prereg loss 1.5154365 reg_l1 13.524097 reg_l2 14.368283
loss 2.8678463
STEP 83 ================================
prereg loss 1.5223051 reg_l1 13.53043 reg_l2 14.386758
loss 2.875348
STEP 84 ================================
prereg loss 1.529241 reg_l1 13.536782 reg_l2 14.405299
loss 2.8829193
STEP 85 ================================
prereg loss 1.5362425 reg_l1 13.543168 reg_l2 14.423922
loss 2.8905592
STEP 86 ================================
prereg loss 1.5433103 reg_l1 13.549593 reg_l2 14.442631
loss 2.8982697
STEP 87 ================================
prereg loss 1.5504425 reg_l1 13.556042 reg_l2 14.461415
loss 2.9060466
STEP 88 ================================
prereg loss 1.557603 reg_l1 13.562499 reg_l2 14.480258
loss 2.913853
STEP 89 ================================
prereg loss 1.5647947 reg_l1 13.568967 reg_l2 14.499154
loss 2.9216914
STEP 90 ================================
prereg loss 1.5720514 reg_l1 13.57545 reg_l2 14.518115
loss 2.9295964
STEP 91 ================================
prereg loss 1.5793747 reg_l1 13.581958 reg_l2 14.5371475
loss 2.9375706
STEP 92 ================================
prereg loss 1.5867682 reg_l1 13.588492 reg_l2 14.556254
loss 2.9456174
STEP 93 ================================
prereg loss 1.5942323 reg_l1 13.595044 reg_l2 14.575422
loss 2.9537368
STEP 94 ================================
prereg loss 1.601753 reg_l1 13.6016 reg_l2 14.594641
loss 2.961913
STEP 95 ================================
prereg loss 1.6093115 reg_l1 13.608161 reg_l2 14.613903
loss 2.9701276
STEP 96 ================================
prereg loss 1.6169477 reg_l1 13.614733 reg_l2 14.6332245
loss 2.978421
STEP 97 ================================
prereg loss 1.6246387 reg_l1 13.621329 reg_l2 14.652613
loss 2.9867716
STEP 98 ================================
prereg loss 1.6324012 reg_l1 13.627948 reg_l2 14.67207
loss 2.9951959
STEP 99 ================================
prereg loss 1.6402243 reg_l1 13.634587 reg_l2 14.691591
loss 3.003683
STEP 100 ================================
prereg loss 1.6480904 reg_l1 13.641237 reg_l2 14.711167
loss 3.0122142
2022-06-28T22:00:45.717

julia> open("version2-sparse22-after-100-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> sparse1 = sparsecopy(sparse, 0.34f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse1
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 8 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>1.00458)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.43396)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.9578…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>1.37824)), "dict-1"=>Dict("compare-3-1"=>Dict("false…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.399825)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.755745)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7459…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.411152)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.351671)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.922213)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
15

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-28T22:03:00.790
STEP 1 ================================
prereg loss 3.2157693 reg_l1 12.715551 reg_l2 12.951533
loss 4.4873247
STEP 2 ================================
prereg loss 3.2151177 reg_l1 12.71855 reg_l2 12.964228
loss 4.486973
STEP 3 ================================
prereg loss 3.2138348 reg_l1 12.726911 reg_l2 12.982751
loss 4.486526
STEP 4 ================================
prereg loss 3.2132905 reg_l1 12.736492 reg_l2 13.00253
loss 4.4869394
STEP 5 ================================
prereg loss 3.2139988 reg_l1 12.745349 reg_l2 13.021273
loss 4.488534
STEP 6 ================================
prereg loss 3.2145715 reg_l1 12.752442 reg_l2 13.037743
loss 4.4898157
STEP 7 ================================
prereg loss 3.214943 reg_l1 12.758178 reg_l2 13.05239
loss 4.490761
STEP 8 ================================
prereg loss 3.215849 reg_l1 12.763289 reg_l2 13.066075
loss 4.492178
STEP 9 ================================
prereg loss 3.2174356 reg_l1 12.768534 reg_l2 13.079745
loss 4.494289
STEP 10 ================================
prereg loss 3.2193613 reg_l1 12.774422 reg_l2 13.094098
loss 4.4968033
STEP 11 ================================
prereg loss 3.22146 reg_l1 12.7809725 reg_l2 13.10928
loss 4.4995575
STEP 12 ================================
prereg loss 3.2237072 reg_l1 12.787884 reg_l2 13.125028
loss 4.502496
STEP 13 ================================
prereg loss 3.2258937 reg_l1 12.794759 reg_l2 13.140927
loss 4.5053697
STEP 14 ================================
prereg loss 3.2276454 reg_l1 12.801321 reg_l2 13.156667
loss 4.5077777
STEP 15 ================================
prereg loss 3.2290416 reg_l1 12.807564 reg_l2 13.172205
loss 4.509798
STEP 16 ================================
prereg loss 3.2304497 reg_l1 12.813666 reg_l2 13.187687
loss 4.5118165
STEP 17 ================================
prereg loss 3.2321315 reg_l1 12.819873 reg_l2 13.203324
loss 4.5141187
STEP 18 ================================
prereg loss 3.234189 reg_l1 12.82636 reg_l2 13.219269
loss 4.516825
STEP 19 ================================
prereg loss 3.236561 reg_l1 12.833129 reg_l2 13.235484
loss 4.519874
STEP 20 ================================
prereg loss 3.2392702 reg_l1 12.84 reg_l2 13.251763
loss 4.52327
STEP 21 ================================
prereg loss 3.2421734 reg_l1 12.846736 reg_l2 13.267842
loss 4.526847
STEP 22 ================================
prereg loss 3.2450895 reg_l1 12.853148 reg_l2 13.28354
loss 4.5304046
STEP 23 ================================
prereg loss 3.2479167 reg_l1 12.859213 reg_l2 13.298858
loss 4.5338383
STEP 24 ================================
prereg loss 3.2507603 reg_l1 12.865066 reg_l2 13.313969
loss 4.5372667
STEP 25 ================================
prereg loss 3.25376 reg_l1 12.870928 reg_l2 13.329133
loss 4.540853
STEP 26 ================================
prereg loss 3.2569673 reg_l1 12.877 reg_l2 13.344591
loss 4.5446672
STEP 27 ================================
prereg loss 3.2603846 reg_l1 12.883368 reg_l2 13.360449
loss 4.5487213
STEP 28 ================================
prereg loss 3.2639358 reg_l1 12.88997 reg_l2 13.376648
loss 4.5529327
STEP 29 ================================
prereg loss 3.2675858 reg_l1 12.896631 reg_l2 13.392997
loss 4.557249
STEP 30 ================================
prereg loss 3.271184 reg_l1 12.903192 reg_l2 13.4093075
loss 4.5615034
STEP 31 ================================
prereg loss 3.2747266 reg_l1 12.909569 reg_l2 13.425487
loss 4.5656834
STEP 32 ================================
prereg loss 3.2782636 reg_l1 12.915801 reg_l2 13.4415455
loss 4.569844
STEP 33 ================================
prereg loss 3.2819474 reg_l1 12.922005 reg_l2 13.457591
loss 4.5741477
STEP 34 ================================
prereg loss 3.28586 reg_l1 12.928292 reg_l2 13.473736
loss 4.5786896
STEP 35 ================================
prereg loss 3.290034 reg_l1 12.934713 reg_l2 13.490028
loss 4.5835056
STEP 36 ================================
prereg loss 3.2944412 reg_l1 12.941218 reg_l2 13.506417
loss 4.588563
STEP 37 ================================
prereg loss 3.2989764 reg_l1 12.947703 reg_l2 13.522803
loss 4.5937467
STEP 38 ================================
prereg loss 3.3035326 reg_l1 12.954075 reg_l2 13.539107
loss 4.59894
STEP 39 ================================
prereg loss 3.3080914 reg_l1 12.960319 reg_l2 13.555325
loss 4.604123
STEP 40 ================================
prereg loss 3.3126698 reg_l1 12.9665 reg_l2 13.571541
loss 4.6093197
STEP 41 ================================
prereg loss 3.3173265 reg_l1 12.972734 reg_l2 13.587874
loss 4.6146
STEP 42 ================================
prereg loss 3.3221533 reg_l1 12.979103 reg_l2 13.604421
loss 4.620064
STEP 43 ================================
prereg loss 3.3271575 reg_l1 12.9856205 reg_l2 13.621191
loss 4.6257195
STEP 44 ================================
prereg loss 3.332275 reg_l1 12.992221 reg_l2 13.638107
loss 4.631497
STEP 45 ================================
prereg loss 3.3374894 reg_l1 12.998799 reg_l2 13.655055
loss 4.637369
STEP 46 ================================
prereg loss 3.3426948 reg_l1 13.005293 reg_l2 13.671956
loss 4.6432242
STEP 47 ================================
prereg loss 3.347974 reg_l1 13.011692 reg_l2 13.6888
loss 4.649143
STEP 48 ================================
prereg loss 3.3533435 reg_l1 13.018063 reg_l2 13.705647
loss 4.65515
STEP 49 ================================
prereg loss 3.3588715 reg_l1 13.024474 reg_l2 13.7225685
loss 4.661319
STEP 50 ================================
prereg loss 3.3645916 reg_l1 13.030972 reg_l2 13.7396145
loss 4.667689
STEP 51 ================================
prereg loss 3.3704789 reg_l1 13.037536 reg_l2 13.756773
loss 4.6742325
STEP 52 ================================
prereg loss 3.3764877 reg_l1 13.044108 reg_l2 13.773993
loss 4.6808987
STEP 53 ================================
prereg loss 3.3825488 reg_l1 13.050644 reg_l2 13.791221
loss 4.6876135
STEP 54 ================================
prereg loss 3.3886344 reg_l1 13.057129 reg_l2 13.808458
loss 4.6943474
STEP 55 ================================
prereg loss 3.3947759 reg_l1 13.063602 reg_l2 13.825748
loss 4.701136
STEP 56 ================================
prereg loss 3.4010477 reg_l1 13.070124 reg_l2 13.843151
loss 4.7080603
STEP 57 ================================
prereg loss 3.407455 reg_l1 13.076726 reg_l2 13.860697
loss 4.7151275
STEP 58 ================================
prereg loss 3.4140122 reg_l1 13.083404 reg_l2 13.878379
loss 4.7223525
STEP 59 ================================
prereg loss 3.4207013 reg_l1 13.090108 reg_l2 13.896144
loss 4.729712
STEP 60 ================================
prereg loss 3.4274645 reg_l1 13.096789 reg_l2 13.913933
loss 4.7371435
STEP 61 ================================
prereg loss 3.4342844 reg_l1 13.103425 reg_l2 13.931728
loss 4.744627
STEP 62 ================================
prereg loss 3.441206 reg_l1 13.110037 reg_l2 13.949544
loss 4.7522097
STEP 63 ================================
prereg loss 3.4482632 reg_l1 13.116672 reg_l2 13.967429
loss 4.7599306
STEP 64 ================================
prereg loss 3.4554863 reg_l1 13.123355 reg_l2 13.985418
loss 4.767822
STEP 65 ================================
prereg loss 3.4628625 reg_l1 13.130094 reg_l2 14.003511
loss 4.7758718
STEP 66 ================================
prereg loss 3.47035 reg_l1 13.136854 reg_l2 14.02169
loss 4.7840357
STEP 67 ================================
prereg loss 3.477923 reg_l1 13.143612 reg_l2 14.039918
loss 4.792284
STEP 68 ================================
prereg loss 3.485574 reg_l1 13.150349 reg_l2 14.05819
loss 4.8006086
STEP 69 ================================
prereg loss 3.4933124 reg_l1 13.157088 reg_l2 14.076524
loss 4.809021
STEP 70 ================================
prereg loss 3.501191 reg_l1 13.163858 reg_l2 14.09495
loss 4.817577
STEP 71 ================================
prereg loss 3.509208 reg_l1 13.170678 reg_l2 14.113488
loss 4.826276
STEP 72 ================================
prereg loss 3.5173593 reg_l1 13.177544 reg_l2 14.132132
loss 4.8351135
STEP 73 ================================
prereg loss 3.5256517 reg_l1 13.18443 reg_l2 14.150853
loss 4.8440948
STEP 74 ================================
prereg loss 3.5340338 reg_l1 13.19131 reg_l2 14.169624
loss 4.8531647
STEP 75 ================================
prereg loss 3.542522 reg_l1 13.198177 reg_l2 14.188439
loss 4.86234
STEP 76 ================================
prereg loss 3.5511353 reg_l1 13.205046 reg_l2 14.20731
loss 4.8716397
STEP 77 ================================
prereg loss 3.5598934 reg_l1 13.211936 reg_l2 14.226257
loss 4.881087
STEP 78 ================================
prereg loss 3.5688026 reg_l1 13.218863 reg_l2 14.245302
loss 4.890689
STEP 79 ================================
prereg loss 3.5778513 reg_l1 13.225825 reg_l2 14.264439
loss 4.900434
STEP 80 ================================
prereg loss 3.5870395 reg_l1 13.232801 reg_l2 14.283651
loss 4.91032
STEP 81 ================================
prereg loss 3.5963337 reg_l1 13.239782 reg_l2 14.30293
loss 4.920312
STEP 82 ================================
prereg loss 3.6057284 reg_l1 13.246763 reg_l2 14.3222685
loss 4.9304047
STEP 83 ================================
prereg loss 3.6152563 reg_l1 13.253759 reg_l2 14.341686
loss 4.9406323
STEP 84 ================================
prereg loss 3.6249228 reg_l1 13.260782 reg_l2 14.361196
loss 4.951001
STEP 85 ================================
prereg loss 3.6347547 reg_l1 13.267843 reg_l2 14.3808
loss 4.9615393
STEP 86 ================================
prereg loss 3.644716 reg_l1 13.274927 reg_l2 14.400493
loss 4.972209
STEP 87 ================================
prereg loss 3.6548257 reg_l1 13.282023 reg_l2 14.420257
loss 4.983028
STEP 88 ================================
prereg loss 3.6650543 reg_l1 13.289122 reg_l2 14.440084
loss 4.9939666
STEP 89 ================================
prereg loss 3.6754165 reg_l1 13.296225 reg_l2 14.459972
loss 5.005039
STEP 90 ================================
prereg loss 3.68591 reg_l1 13.303342 reg_l2 14.479939
loss 5.016244
STEP 91 ================================
prereg loss 3.6965835 reg_l1 13.310487 reg_l2 14.4999895
loss 5.027632
STEP 92 ================================
prereg loss 3.7074032 reg_l1 13.317656 reg_l2 14.52013
loss 5.039169
STEP 93 ================================
prereg loss 3.7183816 reg_l1 13.324846 reg_l2 14.540351
loss 5.050866
STEP 94 ================================
prereg loss 3.7294793 reg_l1 13.3320465 reg_l2 14.560644
loss 5.062684
STEP 95 ================================
prereg loss 3.7407181 reg_l1 13.339252 reg_l2 14.581005
loss 5.074643
STEP 96 ================================
prereg loss 3.7520986 reg_l1 13.346472 reg_l2 14.601447
loss 5.0867457
STEP 97 ================================
prereg loss 3.7636402 reg_l1 13.353714 reg_l2 14.621969
loss 5.0990114
STEP 98 ================================
prereg loss 3.7753372 reg_l1 13.36098 reg_l2 14.642583
loss 5.1114354
STEP 99 ================================
prereg loss 3.7871804 reg_l1 13.368269 reg_l2 14.66328
loss 5.124007
STEP 100 ================================
prereg loss 3.7991836 reg_l1 13.375572 reg_l2 14.684053
loss 5.1367407
2022-06-28T22:07:02.825
```
