# A recurrent run with added "id-transform" neurons to the mix.

The results are so-so, the initial loss is very high, the run shows signs of convergence,
but I do not want to investigate further at this time.

I am committing the serialization of the end state, in case we decide to continue.

This has 4 copies of each type of interneuron; with the original 5 copies the initial loss
is higher (over `1e11`) and the convergence in the initial 16 steps is more erratic than what we
see below (I have not pursued that one further).

Overall, this illustrates that we don't understand the dynamic properties of recurrent systems
of this kind well yet. Before we return to that, let's investigate a simpler setup of
"feedforward transducers with local recurrencies"

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1")

julia> include("prepare.jl")
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> steps!(16)
2022-06-11T12:38:11.981
STEP 1 ================================
prereg loss 5.665264e8 regularization 3279.3345 reg_novel 1046.693
loss 5.665264e8
STEP 2 ================================
prereg loss 4.6507757e8 regularization 3275.0854 reg_novel 971.08826
loss 4.6507757e8
STEP 3 ================================
prereg loss 3.567716e8 regularization 3270.2598 reg_novel 900.4526
loss 3.567716e8
STEP 4 ================================
prereg loss 2.8811037e8 regularization 3264.9395 reg_novel 834.32544
loss 2.8811037e8
STEP 5 ================================
prereg loss 2.0121722e8 regularization 3259.1995 reg_novel 772.4676
loss 2.0121722e8
STEP 6 ================================
prereg loss 1.5250685e8 regularization 3253.101 reg_novel 714.6491
loss 1.5250685e8
STEP 7 ================================
prereg loss 1.1976628e8 regularization 3246.6917 reg_novel 660.68567
loss 1.1976628e8
STEP 8 ================================
prereg loss 9.982667e7 regularization 3239.9685 reg_novel 610.4116
loss 9.982667e7
STEP 9 ================================
prereg loss 8.7506216e7 regularization 3232.9731 reg_novel 563.65515
loss 8.7506216e7
STEP 10 ================================
prereg loss 6.807926e7 regularization 3225.6956 reg_novel 520.22876
loss 6.807926e7
STEP 11 ================================
prereg loss 5.0716676e7 regularization 3218.1838 reg_novel 479.96667
loss 5.071668e7
STEP 12 ================================
prereg loss 3.7045296e7 regularization 3210.4521 reg_novel 442.67636
loss 3.70453e7
STEP 13 ================================
prereg loss 2.9446272e7 regularization 3202.505 reg_novel 408.20917
loss 2.9446276e7
STEP 14 ================================
prereg loss 2.42827e7 regularization 3194.3457 reg_novel 376.33743
loss 2.4282704e7
STEP 15 ================================
prereg loss 1.87187e7 regularization 3185.9443 reg_novel 346.88092
loss 1.8718704e7
STEP 16 ================================
prereg loss 1.5715315e7 regularization 3177.3633 reg_novel 319.6563
loss 1.5715318e7
2022-06-11T13:05:54.809

julia> steps!(16)
2022-06-11T22:25:48.546
STEP 1 ================================
prereg loss 1.3229834e7 regularization 3168.5884 reg_novel 294.51157
loss 1.3229837e7
STEP 2 ================================
prereg loss 1.1477025e7 regularization 3159.626 reg_novel 271.29242
loss 1.1477028e7
STEP 3 ================================
prereg loss 1.0114952e7 regularization 3150.503 reg_novel 249.85739
loss 1.0114955e7
STEP 4 ================================
prereg loss 8.940634e6 regularization 3141.2283 reg_novel 230.08025
loss 8.940637e6
STEP 5 ================================
prereg loss 7.5460685e6 regularization 3131.797 reg_novel 211.8236
loss 7.546072e6
STEP 6 ================================
prereg loss 6.5514775e6 regularization 3122.25 reg_novel 194.97453
loss 6.551481e6
STEP 7 ================================
prereg loss 5.9000705e6 regularization 3112.5596 reg_novel 179.42484
loss 5.900074e6
STEP 8 ================================
prereg loss 4.888531e6 regularization 3102.7341 reg_novel 165.07755
loss 4.8885345e6
STEP 9 ================================
prereg loss 4.34186e6 regularization 3092.818 reg_novel 151.85548
loss 4.341863e6
STEP 10 ================================
prereg loss 3.814033e6 regularization 3082.8213 reg_novel 139.65862
loss 3.8140362e6
STEP 11 ================================
prereg loss 3.7792725e6 regularization 3072.7532 reg_novel 128.42783
loss 3.7792758e6
STEP 12 ================================
prereg loss 3.1974278e6 regularization 3062.6094 reg_novel 118.08087
loss 3.197431e6
STEP 13 ================================
prereg loss 2.8611775e6 regularization 3052.415 reg_novel 108.54796
loss 2.8611808e6
STEP 14 ================================
prereg loss 2.6000582e6 regularization 3042.1592 reg_novel 99.76921
loss 2.6000615e6
STEP 15 ================================
prereg loss 2.443562e6 regularization 3031.8665 reg_novel 91.69201
loss 2.443565e6
STEP 16 ================================
prereg loss 2.4479988e6 regularization 3021.4775 reg_novel 84.277565
loss 2.4480018e6
2022-06-11T22:52:33.467

julia> steps!(500)
2022-06-11T23:00:16.813
STEP 1 ================================
prereg loss 2.3667545e6 regularization 3011.048 reg_novel 77.468285
loss 2.3667575e6
STEP 2 ================================
prereg loss 2.3618948e6 regularization 3000.537 reg_novel 71.21432
loss 2.3618978e6
STEP 3 ================================
prereg loss 2.490999e6 regularization 2990.0054 reg_novel 65.476295
loss 2.491002e6
STEP 4 ================================
prereg loss 2.4387535e6 regularization 2979.4448 reg_novel 60.216343
loss 2.4387565e6
STEP 5 ================================
prereg loss 2.4621325e6 regularization 2968.8567 reg_novel 55.39796
loss 2.4621355e6
STEP 6 ================================
prereg loss 2.5029365e6 regularization 2958.218 reg_novel 50.981606
loss 2.5029395e6
STEP 7 ================================
prereg loss 2.357915e6 regularization 2947.5664 reg_novel 46.933628
loss 2.357918e6
STEP 8 ================================
prereg loss 2.176677e6 regularization 2936.8833 reg_novel 43.221386
loss 2.17668e6
STEP 9 ================================
prereg loss 2.0256698e6 regularization 2926.2021 reg_novel 39.81762
loss 2.0256728e6
STEP 10 ================================
prereg loss 2.0012405e6 regularization 2915.4897 reg_novel 36.700127
loss 2.0012435e6
STEP 11 ================================
prereg loss 1.7586086e6 regularization 2904.8103 reg_novel 33.840153
loss 1.7586116e6
STEP 12 ================================
prereg loss 1.5681045e6 regularization 2894.14 reg_novel 31.218596
loss 1.5681074e6
STEP 13 ================================
prereg loss 1.4262331e6 regularization 2883.4685 reg_novel 28.818167
loss 1.426236e6
STEP 14 ================================
prereg loss 1.3451828e6 regularization 2872.7808 reg_novel 26.620903
loss 1.3451856e6
STEP 15 ================================
prereg loss 1.2523664e6 regularization 2862.1062 reg_novel 24.613718
loss 1.2523692e6
STEP 16 ================================
prereg loss 1.2140148e6 regularization 2851.4087 reg_novel 22.781048
loss 1.2140176e6
STEP 17 ================================
prereg loss 1.1279805e6 regularization 2840.7136 reg_novel 21.105154
loss 1.1279834e6
STEP 18 ================================
prereg loss 1.04068844e6 regularization 2829.989 reg_novel 19.57644
loss 1.0406913e6
STEP 19 ================================
prereg loss 957602.6 regularization 2819.3176 reg_novel 18.182625
loss 957605.44
STEP 20 ================================
prereg loss 879821.1 regularization 2808.6033 reg_novel 16.90993
loss 879823.94
STEP 21 ================================
prereg loss 805876.1 regularization 2797.9128 reg_novel 15.754377
loss 805878.94
STEP 22 ================================
prereg loss 737320.6 regularization 2787.2134 reg_novel 14.696344
loss 737323.44
STEP 23 ================================
prereg loss 694941.5 regularization 2776.5225 reg_novel 13.733783
loss 694944.3
STEP 24 ================================
prereg loss 651525.44 regularization 2765.8206 reg_novel 12.8601675
loss 651528.2
STEP 25 ================================
prereg loss 600818.6 regularization 2755.0906 reg_novel 12.074039
loss 600821.4
STEP 26 ================================
prereg loss 584702.5 regularization 2744.3867 reg_novel 11.365459
loss 584705.25
STEP 27 ================================
prereg loss 516067.0 regularization 2733.7246 reg_novel 10.722398
loss 516069.75
STEP 28 ================================
prereg loss 474231.94 regularization 2723.0652 reg_novel 10.137696
loss 474234.66
STEP 29 ================================
prereg loss 414296.8 regularization 2712.4258 reg_novel 9.609708
loss 414299.53
STEP 30 ================================
prereg loss 356013.8 regularization 2701.8142 reg_novel 9.133111
loss 356016.53
STEP 31 ================================
prereg loss 302106.2 regularization 2691.1685 reg_novel 8.70145
loss 302108.88
STEP 32 ================================
prereg loss 260493.69 regularization 2680.5422 reg_novel 8.308668
loss 260496.38
STEP 33 ================================
prereg loss 219064.94 regularization 2669.9065 reg_novel 7.952128
loss 219067.61
STEP 34 ================================
prereg loss 198741.47 regularization 2659.2983 reg_novel 7.6338882
loss 198744.14
STEP 35 ================================
prereg loss 178304.1 regularization 2648.6917 reg_novel 7.352732
loss 178306.75
STEP 36 ================================
prereg loss 159104.92 regularization 2638.0898 reg_novel 7.1000557
loss 159107.56
STEP 37 ================================
prereg loss 130435.3 regularization 2627.4976 reg_novel 6.876316
loss 130437.93
STEP 38 ================================
prereg loss 108540.78 regularization 2616.9392 reg_novel 6.682742
loss 108543.41
STEP 39 ================================
prereg loss 92863.19 regularization 2606.3745 reg_novel 6.5141187
loss 92865.8
STEP 40 ================================
prereg loss 79006.31 regularization 2595.8677 reg_novel 6.3693705
loss 79008.914
STEP 41 ================================
prereg loss 75554.47 regularization 2585.3506 reg_novel 6.2387376
loss 75557.06
STEP 42 ================================
prereg loss 78899.266 regularization 2574.8716 reg_novel 6.119057
loss 78901.84
STEP 43 ================================
prereg loss 71878.27 regularization 2564.3823 reg_novel 6.0102262
loss 71880.84
STEP 44 ================================
prereg loss 65013.387 regularization 2553.9246 reg_novel 5.907063
loss 65015.945
STEP 45 ================================
prereg loss 50485.555 regularization 2543.461 reg_novel 5.8080225
loss 50488.105
STEP 46 ================================
prereg loss 47594.875 regularization 2533.0303 reg_novel 5.7193456
loss 47597.414
STEP 47 ================================
prereg loss 43053.406 regularization 2522.5977 reg_novel 5.6422925
loss 43055.934
STEP 48 ================================
prereg loss 40005.246 regularization 2512.1987 reg_novel 5.5727835
loss 40007.766
STEP 49 ================================
prereg loss 36648.555 regularization 2501.8135 reg_novel 5.5127993
loss 36651.062
STEP 50 ================================
prereg loss 33136.082 regularization 2491.472 reg_novel 5.4604588
loss 33138.58
STEP 51 ================================
prereg loss 29706.484 regularization 2481.1558 reg_novel 5.4181333
loss 29708.97
STEP 52 ================================
prereg loss 26863.342 regularization 2470.8518 reg_novel 5.3836803
loss 26865.818
STEP 53 ================================
prereg loss 22921.244 regularization 2460.5627 reg_novel 5.3525224
loss 22923.71
STEP 54 ================================
prereg loss 19524.557 regularization 2450.2903 reg_novel 5.3237076
loss 19527.012
STEP 55 ================================
prereg loss 13247.427 regularization 2440.0356 reg_novel 5.2990956
loss 13249.872
STEP 56 ================================
prereg loss 12669.927 regularization 2429.819 reg_novel 5.274824
loss 12672.362
STEP 57 ================================
prereg loss 10877.667 regularization 2419.607 reg_novel 5.256706
loss 10880.092
STEP 58 ================================
prereg loss 9960.653 regularization 2409.4294 reg_novel 5.244142
loss 9963.068
STEP 59 ================================
prereg loss 13707.978 regularization 2399.2576 reg_novel 5.235363
loss 13710.382
STEP 60 ================================
prereg loss 9465.891 regularization 2389.145 reg_novel 5.226967
loss 9468.285
STEP 61 ================================
prereg loss 12397.854 regularization 2379.0483 reg_novel 5.2159834
loss 12400.237
STEP 62 ================================
prereg loss 11739.9375 regularization 2368.981 reg_novel 5.2074175
loss 11742.312
STEP 63 ================================
prereg loss 10094.114 regularization 2358.9316 reg_novel 5.1961045
loss 10096.479
STEP 64 ================================
prereg loss 9213.068 regularization 2348.92 reg_novel 5.187771
loss 9215.423
STEP 65 ================================
prereg loss 7413.051 regularization 2338.9268 reg_novel 5.1822863
loss 7415.395
STEP 66 ================================
prereg loss 6183.788 regularization 2328.939 reg_novel 5.1788235
loss 6186.122
STEP 67 ================================
prereg loss 4414.1484 regularization 2318.9604 reg_novel 5.180251
loss 4416.4727
STEP 68 ================================
prereg loss 4858.955 regularization 2309.0156 reg_novel 5.18526
loss 4861.269
STEP 69 ================================
prereg loss 4517.5327 regularization 2299.1108 reg_novel 5.1893806
loss 4519.837
STEP 70 ================================
prereg loss 2382.904 regularization 2289.238 reg_novel 5.1906385
loss 2385.1985
STEP 71 ================================
prereg loss 2956.7754 regularization 2279.363 reg_novel 5.1891646
loss 2959.06
STEP 72 ================================
prereg loss 2294.0251 regularization 2269.513 reg_novel 5.187947
loss 2296.2998
STEP 73 ================================
prereg loss 764.98254 regularization 2259.6868 reg_novel 5.187216
loss 767.24744
STEP 74 ================================
prereg loss 191.32407 regularization 2249.8828 reg_novel 5.189204
loss 193.57913
STEP 75 ================================
prereg loss 331.4065 regularization 2240.1235 reg_novel 5.1906466
loss 333.6518
STEP 76 ================================
prereg loss 307.02832 regularization 2230.3848 reg_novel 5.18875
loss 309.2639
STEP 77 ================================
prereg loss 383.4735 regularization 2220.6702 reg_novel 5.1845427
loss 385.69937
STEP 78 ================================
prereg loss 501.27496 regularization 2210.9668 reg_novel 5.1825414
loss 503.49112
STEP 79 ================================
prereg loss 488.18408 regularization 2201.3064 reg_novel 5.181348
loss 490.39056
STEP 80 ================================
prereg loss 551.68823 regularization 2191.7031 reg_novel 5.1810875
loss 553.88513
STEP 81 ================================
prereg loss 518.944 regularization 2182.0842 reg_novel 5.179647
loss 521.1312
STEP 82 ================================
prereg loss 371.33273 regularization 2172.534 reg_novel 5.181607
loss 373.51044
STEP 83 ================================
prereg loss 294.09988 regularization 2163.0142 reg_novel 5.18367
loss 296.2681
STEP 84 ================================
prereg loss 305.03546 regularization 2153.5115 reg_novel 5.1875267
loss 307.19415
STEP 85 ================================
prereg loss 328.01407 regularization 2144.0076 reg_novel 5.194464
loss 330.16327
STEP 86 ================================
prereg loss 188.48608 regularization 2134.5215 reg_novel 5.204005
loss 190.62581
STEP 87 ================================
prereg loss 161.69156 regularization 2125.0542 reg_novel 5.217304
loss 163.82182
STEP 88 ================================
prereg loss 130.5569 regularization 2115.6167 reg_novel 5.231346
loss 132.67775
STEP 89 ================================
prereg loss 113.15852 regularization 2106.1829 reg_novel 5.2396884
loss 115.26994
STEP 90 ================================
prereg loss 137.1158 regularization 2096.7817 reg_novel 5.2494807
loss 139.21783
STEP 91 ================================
prereg loss 99.09516 regularization 2087.4182 reg_novel 5.255909
loss 101.187836
STEP 92 ================================
prereg loss 69.14614 regularization 2078.0796 reg_novel 5.2598953
loss 71.22948
STEP 93 ================================
prereg loss 52.267982 regularization 2068.7698 reg_novel 5.261362
loss 54.342014
STEP 94 ================================
prereg loss 45.246468 regularization 2059.4785 reg_novel 5.2685304
loss 47.311214
STEP 95 ================================
prereg loss 43.01982 regularization 2050.2214 reg_novel 5.2779846
loss 45.07532
STEP 96 ================================
prereg loss 36.422478 regularization 2040.9814 reg_novel 5.2837515
loss 38.468742
STEP 97 ================================
prereg loss 30.560932 regularization 2031.7706 reg_novel 5.2856293
loss 32.59799
STEP 98 ================================
prereg loss 22.200514 regularization 2022.5778 reg_novel 5.2844634
loss 24.228376
STEP 99 ================================
prereg loss 17.192118 regularization 2013.4375 reg_novel 5.281684
loss 19.210836
STEP 100 ================================
prereg loss 13.814012 regularization 2004.346 reg_novel 5.279823
loss 15.823637
STEP 101 ================================
prereg loss 11.66654 regularization 1995.2903 reg_novel 5.2757854
loss 13.667107
STEP 102 ================================
prereg loss 10.244408 regularization 1986.2678 reg_novel 5.2624216
loss 12.235938
STEP 103 ================================
prereg loss 9.302633 regularization 1977.2793 reg_novel 5.250416
loss 11.285163
STEP 104 ================================
prereg loss 8.542319 regularization 1968.3202 reg_novel 5.242621
loss 10.5158825
STEP 105 ================================
prereg loss 8.047823 regularization 1959.3855 reg_novel 5.2342076
loss 10.012443
STEP 106 ================================
prereg loss 7.6743245 regularization 1950.4678 reg_novel 5.231188
loss 9.630024
STEP 107 ================================
prereg loss 7.3848205 regularization 1941.5579 reg_novel 5.230159
loss 9.331609
STEP 108 ================================
prereg loss 7.154657 regularization 1932.6661 reg_novel 5.233848
loss 9.092557
STEP 109 ================================
prereg loss 6.885985 regularization 1923.8226 reg_novel 5.239519
loss 8.815047
STEP 110 ================================
prereg loss 6.6122203 regularization 1915.0265 reg_novel 5.2504215
loss 8.532497
STEP 111 ================================
prereg loss 6.38035 regularization 1906.2878 reg_novel 5.2638016
loss 8.291902
STEP 112 ================================
prereg loss 6.2047977 regularization 1897.5533 reg_novel 5.2811847
loss 8.107633
STEP 113 ================================
prereg loss 6.0829253 regularization 1888.8594 reg_novel 5.293486
loss 7.9770784
STEP 114 ================================
prereg loss 5.9975986 regularization 1880.2183 reg_novel 5.303783
loss 7.8831205
STEP 115 ================================
prereg loss 5.9508467 regularization 1871.5924 reg_novel 5.315292
loss 7.8277545
STEP 116 ================================
prereg loss 5.911604 regularization 1862.9996 reg_novel 5.3311973
loss 7.779935
STEP 117 ================================
prereg loss 5.8828526 regularization 1854.442 reg_novel 5.3510203
loss 7.7426457
STEP 118 ================================
prereg loss 5.869539 regularization 1845.9067 reg_novel 5.3668222
loss 7.7208123
STEP 119 ================================
prereg loss 5.8561172 regularization 1837.4186 reg_novel 5.3784533
loss 7.6989145
STEP 120 ================================
prereg loss 5.8459396 regularization 1828.94 reg_novel 5.3891363
loss 7.680269
STEP 121 ================================
prereg loss 5.842074 regularization 1820.5015 reg_novel 5.3992457
loss 7.6679745
STEP 122 ================================
prereg loss 5.835828 regularization 1812.1207 reg_novel 5.404622
loss 7.653353
STEP 123 ================================
prereg loss 5.8321958 regularization 1803.7507 reg_novel 5.4059687
loss 7.6413527
STEP 124 ================================
prereg loss 5.81881 regularization 1795.4202 reg_novel 5.407746
loss 7.619638
STEP 125 ================================
prereg loss 5.8191743 regularization 1787.1239 reg_novel 5.41294
loss 7.6117115
STEP 126 ================================
prereg loss 5.8167386 regularization 1778.8671 reg_novel 5.417193
loss 7.6010227
STEP 127 ================================
prereg loss 5.812215 regularization 1770.6384 reg_novel 5.413238
loss 7.5882664
STEP 128 ================================
prereg loss 5.8013134 regularization 1762.4406 reg_novel 5.409584
loss 7.569164
STEP 129 ================================
prereg loss 5.788499 regularization 1754.2771 reg_novel 5.4117036
loss 7.5481877
STEP 130 ================================
prereg loss 5.769098 regularization 1746.1427 reg_novel 5.422646
loss 7.5206633
STEP 131 ================================
prereg loss 5.7472053 regularization 1738.014 reg_novel 5.436957
loss 7.4906564
STEP 132 ================================
prereg loss 5.717533 regularization 1729.9236 reg_novel 5.4534388
loss 7.4529104
STEP 133 ================================
prereg loss 5.686352 regularization 1721.868 reg_novel 5.478174
loss 7.413698
STEP 134 ================================
prereg loss 5.6567397 regularization 1713.85 reg_novel 5.499426
loss 7.376089
STEP 135 ================================
prereg loss 5.6140676 regularization 1705.8407 reg_novel 5.5170484
loss 7.325425
STEP 136 ================================
prereg loss 5.569239 regularization 1697.8987 reg_novel 5.5328174
loss 7.2726707
STEP 137 ================================
prereg loss 5.507525 regularization 1689.9873 reg_novel 5.543636
loss 7.203056
STEP 138 ================================
prereg loss 5.4445724 regularization 1682.0938 reg_novel 5.5483522
loss 7.1322145
STEP 139 ================================
prereg loss 5.394703 regularization 1674.2413 reg_novel 5.5532207
loss 7.0744977
STEP 140 ================================
prereg loss 5.3492594 regularization 1666.3965 reg_novel 5.5570507
loss 7.021213
STEP 141 ================================
prereg loss 5.304954 regularization 1658.5901 reg_novel 5.5578423
loss 6.969102
STEP 142 ================================
prereg loss 5.2546654 regularization 1650.8179 reg_novel 5.5532565
loss 6.9110365
STEP 143 ================================
prereg loss 5.196145 regularization 1643.0865 reg_novel 5.5425477
loss 6.8447742
STEP 144 ================================
prereg loss 5.1372337 regularization 1635.3998 reg_novel 5.53467
loss 6.778168
STEP 145 ================================
prereg loss 5.0705047 regularization 1627.7302 reg_novel 5.533521
loss 6.7037687
STEP 146 ================================
prereg loss 4.994669 regularization 1620.1241 reg_novel 5.5388184
loss 6.620332
STEP 147 ================================
prereg loss 4.915277 regularization 1612.5298 reg_novel 5.5453353
loss 6.5333524
STEP 148 ================================
prereg loss 4.8382297 regularization 1604.9926 reg_novel 5.551247
loss 6.4487734
STEP 149 ================================
prereg loss 4.7521567 regularization 1597.4634 reg_novel 5.562053
loss 6.355182
STEP 150 ================================
prereg loss 4.658306 regularization 1589.9778 reg_novel 5.5769873
loss 6.253861
STEP 151 ================================
prereg loss 4.5504866 regularization 1582.4913 reg_novel 5.5893703
loss 6.1385674
STEP 152 ================================
prereg loss 4.4394207 regularization 1575.0189 reg_novel 5.595818
loss 6.0200357
STEP 153 ================================
prereg loss 4.3443875 regularization 1567.5609 reg_novel 5.599678
loss 5.917548
STEP 154 ================================
prereg loss 4.2619414 regularization 1560.1833 reg_novel 5.597622
loss 5.8277225
STEP 155 ================================
prereg loss 4.1946254 regularization 1552.8508 reg_novel 5.6010523
loss 5.7530775
STEP 156 ================================
prereg loss 4.1215057 regularization 1545.5532 reg_novel 5.601664
loss 5.672661
STEP 157 ================================
prereg loss 4.046093 regularization 1538.2776 reg_novel 5.597383
loss 5.5899677
STEP 158 ================================
prereg loss 3.966608 regularization 1531.0494 reg_novel 5.5870748
loss 5.5032444
STEP 159 ================================
prereg loss 3.9012573 regularization 1523.8494 reg_novel 5.572836
loss 5.4306793
STEP 160 ================================
prereg loss 3.8372817 regularization 1516.6887 reg_novel 5.5565033
loss 5.359527
STEP 161 ================================
prereg loss 3.766207 regularization 1509.5554 reg_novel 5.5407505
loss 5.2813034
STEP 162 ================================
prereg loss 3.6950862 regularization 1502.4562 reg_novel 5.528447
loss 5.203071
STEP 163 ================================
prereg loss 3.6252947 regularization 1495.3907 reg_novel 5.529655
loss 5.126215
STEP 164 ================================
prereg loss 3.5438075 regularization 1488.3607 reg_novel 5.5377197
loss 5.037706
STEP 165 ================================
prereg loss 3.4548976 regularization 1481.3683 reg_novel 5.548866
loss 4.941815
STEP 166 ================================
prereg loss 3.372316 regularization 1474.3849 reg_novel 5.56858
loss 4.852269
STEP 167 ================================
prereg loss 3.2958925 regularization 1467.4177 reg_novel 5.5880265
loss 4.7688985
STEP 168 ================================
prereg loss 3.2253857 regularization 1460.4895 reg_novel 5.6124105
loss 4.691488
STEP 169 ================================
prereg loss 3.1658647 regularization 1453.5765 reg_novel 5.634031
loss 4.6250753
STEP 170 ================================
prereg loss 3.1063066 regularization 1446.6873 reg_novel 5.6555266
loss 4.5586495
STEP 171 ================================
prereg loss 3.0409727 regularization 1439.8402 reg_novel 5.675049
loss 4.4864883
STEP 172 ================================
prereg loss 2.978391 regularization 1433.024 reg_novel 5.693042
loss 4.417108
STEP 173 ================================
prereg loss 2.9123037 regularization 1426.2588 reg_novel 5.707856
loss 4.34427
STEP 174 ================================
prereg loss 2.848775 regularization 1419.5149 reg_novel 5.714144
loss 4.274004
STEP 175 ================================
prereg loss 2.7975676 regularization 1412.798 reg_novel 5.719579
loss 4.2160854
STEP 176 ================================
prereg loss 2.7473145 regularization 1406.1282 reg_novel 5.7250013
loss 4.159168
STEP 177 ================================
prereg loss 2.7093706 regularization 1399.4604 reg_novel 5.722653
loss 4.114554
STEP 178 ================================
prereg loss 2.6742775 regularization 1392.8348 reg_novel 5.7195673
loss 4.072832
STEP 179 ================================
prereg loss 2.637728 regularization 1386.2106 reg_novel 5.71753
loss 4.0296564
STEP 180 ================================
prereg loss 2.600512 regularization 1379.6111 reg_novel 5.7198257
loss 3.985843
STEP 181 ================================
prereg loss 2.5644937 regularization 1373.0638 reg_novel 5.723091
loss 3.9432807
STEP 182 ================================
prereg loss 2.53639 regularization 1366.5366 reg_novel 5.724813
loss 3.9086516
STEP 183 ================================
prereg loss 2.511023 regularization 1360.0222 reg_novel 5.727438
loss 3.8767729
STEP 184 ================================
prereg loss 2.4882026 regularization 1353.5343 reg_novel 5.7344656
loss 3.8474712
STEP 185 ================================
prereg loss 2.464928 regularization 1347.0902 reg_novel 5.750272
loss 3.8177686
STEP 186 ================================
prereg loss 2.4428413 regularization 1340.6694 reg_novel 5.7660623
loss 3.7892768
STEP 187 ================================
prereg loss 2.4247913 regularization 1334.269 reg_novel 5.7811
loss 3.7648416
STEP 188 ================================
prereg loss 2.406208 regularization 1327.9154 reg_novel 5.7851844
loss 3.7399087
STEP 189 ================================
prereg loss 2.3845918 regularization 1321.5901 reg_novel 5.7849026
loss 3.711967
STEP 190 ================================
prereg loss 2.361101 regularization 1315.3016 reg_novel 5.7845554
loss 3.682187
STEP 191 ================================
prereg loss 2.3344069 regularization 1309.0367 reg_novel 5.7870517
loss 3.6492307
STEP 192 ================================
prereg loss 2.3045897 regularization 1302.8258 reg_novel 5.784434
loss 3.6132
STEP 193 ================================
prereg loss 2.280632 regularization 1296.6567 reg_novel 5.7761316
loss 3.583065
STEP 194 ================================
prereg loss 2.2547863 regularization 1290.5209 reg_novel 5.7690296
loss 3.5510762
STEP 195 ================================
prereg loss 2.2331247 regularization 1284.414 reg_novel 5.7680073
loss 3.5233068
STEP 196 ================================
prereg loss 2.2144158 regularization 1278.3353 reg_novel 5.7744164
loss 3.4985256
STEP 197 ================================
prereg loss 2.200366 regularization 1272.2861 reg_novel 5.783852
loss 3.478436
STEP 198 ================================
prereg loss 2.1904397 regularization 1266.2554 reg_novel 5.7945485
loss 3.4624896
STEP 199 ================================
prereg loss 2.1843095 regularization 1260.2377 reg_novel 5.8109264
loss 3.4503582
STEP 200 ================================
prereg loss 2.1762187 regularization 1254.251 reg_novel 5.825159
loss 3.436295
STEP 201 ================================
prereg loss 2.1689653 regularization 1248.2773 reg_novel 5.8455567
loss 3.4230883
STEP 202 ================================
prereg loss 2.1662037 regularization 1242.3547 reg_novel 5.86996
loss 3.4144285
STEP 203 ================================
prereg loss 2.1677167 regularization 1236.4614 reg_novel 5.89669
loss 3.410075
STEP 204 ================================
prereg loss 2.1716821 regularization 1230.6052 reg_novel 5.9194913
loss 3.408207
STEP 205 ================================
prereg loss 2.1741073 regularization 1224.7781 reg_novel 5.941656
loss 3.404827
STEP 206 ================================
prereg loss 2.175848 regularization 1218.957 reg_novel 5.963285
loss 3.4007683
STEP 207 ================================
prereg loss 2.1756945 regularization 1213.1936 reg_novel 5.9816008
loss 3.3948698
STEP 208 ================================
prereg loss 2.1717665 regularization 1207.4402 reg_novel 5.997249
loss 3.385204
STEP 209 ================================
prereg loss 2.169819 regularization 1201.6965 reg_novel 6.0104113
loss 3.3775263
STEP 210 ================================
prereg loss 2.1715465 regularization 1196.0092 reg_novel 6.027354
loss 3.3735828
STEP 211 ================================
prereg loss 2.1723206 regularization 1190.3446 reg_novel 6.0539846
loss 3.3687193
STEP 212 ================================
prereg loss 2.1708457 regularization 1184.6744 reg_novel 6.078758
loss 3.361599
STEP 213 ================================
prereg loss 2.1685612 regularization 1179.0679 reg_novel 6.1046352
loss 3.3537338
STEP 214 ================================
prereg loss 2.166207 regularization 1173.4904 reg_novel 6.1232653
loss 3.345821
STEP 215 ================================
prereg loss 2.1670578 regularization 1167.9293 reg_novel 6.137734
loss 3.341125
STEP 216 ================================
prereg loss 2.171509 regularization 1162.4054 reg_novel 6.1486697
loss 3.340063
STEP 217 ================================
prereg loss 2.1724474 regularization 1156.9005 reg_novel 6.1567955
loss 3.3355048
STEP 218 ================================
prereg loss 2.1708677 regularization 1151.4811 reg_novel 6.159892
loss 3.3285089
STEP 219 ================================
prereg loss 2.1694775 regularization 1146.0731 reg_novel 6.1579423
loss 3.3217087
STEP 220 ================================
prereg loss 2.1679146 regularization 1140.6989 reg_novel 6.1641135
loss 3.3147776
STEP 221 ================================
prereg loss 2.1684449 regularization 1135.3496 reg_novel 6.1693306
loss 3.3099637
STEP 222 ================================
prereg loss 2.16966 regularization 1129.9998 reg_novel 6.1793923
loss 3.3058393
STEP 223 ================================
prereg loss 2.1709137 regularization 1124.6991 reg_novel 6.193922
loss 3.3018067
STEP 224 ================================
prereg loss 2.1724923 regularization 1119.4124 reg_novel 6.212539
loss 3.2981172
STEP 225 ================================
prereg loss 2.1716347 regularization 1114.1409 reg_novel 6.2330303
loss 3.2920084
STEP 226 ================================
prereg loss 2.171905 regularization 1108.8865 reg_novel 6.2539616
loss 3.2870455
STEP 227 ================================
prereg loss 2.170753 regularization 1103.652 reg_novel 6.2788153
loss 3.280684
STEP 228 ================================
prereg loss 2.171653 regularization 1098.4309 reg_novel 6.306571
loss 3.2763906
STEP 229 ================================
prereg loss 2.173588 regularization 1093.2448 reg_novel 6.3309903
loss 3.2731638
STEP 230 ================================
prereg loss 2.1771207 regularization 1088.1194 reg_novel 6.3461504
loss 3.2715862
STEP 231 ================================
prereg loss 2.1789296 regularization 1083.0464 reg_novel 6.360231
loss 3.2683363
STEP 232 ================================
prereg loss 2.1781545 regularization 1078.0018 reg_novel 6.375172
loss 3.2625315
STEP 233 ================================
prereg loss 2.1743937 regularization 1072.9644 reg_novel 6.395109
loss 3.2537532
STEP 234 ================================
prereg loss 2.1726458 regularization 1067.923 reg_novel 6.4143724
loss 3.2469833
STEP 235 ================================
prereg loss 2.175081 regularization 1062.9159 reg_novel 6.436106
loss 3.244433
STEP 236 ================================
prereg loss 2.1830678 regularization 1057.9517 reg_novel 6.4560547
loss 3.2474756
STEP 237 ================================
prereg loss 2.193747 regularization 1053.0151 reg_novel 6.479913
loss 3.253242
STEP 238 ================================
prereg loss 2.2040706 regularization 1048.1119 reg_novel 6.4989886
loss 3.2586815
STEP 239 ================================
prereg loss 2.213494 regularization 1043.2197 reg_novel 6.5100255
loss 3.263224
STEP 240 ================================
prereg loss 2.2247555 regularization 1038.3525 reg_novel 6.528448
loss 3.2696366
STEP 241 ================================
prereg loss 2.2365768 regularization 1033.5116 reg_novel 6.5498805
loss 3.2766383
STEP 242 ================================
prereg loss 2.247567 regularization 1028.7026 reg_novel 6.571692
loss 3.2828412
STEP 243 ================================
prereg loss 2.25624 regularization 1023.922 reg_novel 6.601068
loss 3.2867632
STEP 244 ================================
prereg loss 2.2636573 regularization 1019.14417 reg_novel 6.6310196
loss 3.2894325
STEP 245 ================================
prereg loss 2.2722075 regularization 1014.4063 reg_novel 6.654884
loss 3.2932687
STEP 246 ================================
prereg loss 2.2815151 regularization 1009.6995 reg_novel 6.6894765
loss 3.297904
STEP 247 ================================
prereg loss 2.2892482 regularization 1005.01697 reg_novel 6.7220187
loss 3.3009872
STEP 248 ================================
prereg loss 2.2971702 regularization 1000.36456 reg_novel 6.758359
loss 3.3042932
STEP 249 ================================
prereg loss 2.303251 regularization 995.7373 reg_novel 6.8009653
loss 3.3057895
STEP 250 ================================
prereg loss 2.308983 regularization 991.14056 reg_novel 6.848751
loss 3.3069725
STEP 251 ================================
prereg loss 2.3143272 regularization 986.5645 reg_novel 6.892722
loss 3.3077846
STEP 252 ================================
prereg loss 2.3239095 regularization 982.03485 reg_novel 6.932063
loss 3.3128765
STEP 253 ================================
prereg loss 2.3339531 regularization 977.50525 reg_novel 6.971334
loss 3.3184297
STEP 254 ================================
prereg loss 2.344945 regularization 972.9873 reg_novel 7.00971
loss 3.324942
STEP 255 ================================
prereg loss 2.355987 regularization 968.5048 reg_novel 7.051739
loss 3.3315437
STEP 256 ================================
prereg loss 2.3633094 regularization 964.05066 reg_novel 7.0833035
loss 3.3344433
STEP 257 ================================
prereg loss 2.365622 regularization 959.6344 reg_novel 7.104828
loss 3.3323612
STEP 258 ================================
prereg loss 2.3652494 regularization 955.2608 reg_novel 7.1224694
loss 3.3276327
STEP 259 ================================
prereg loss 2.3655634 regularization 950.91614 reg_novel 7.1455555
loss 3.323625
STEP 260 ================================
prereg loss 2.3692987 regularization 946.5541 reg_novel 7.173248
loss 3.3230262
STEP 261 ================================
prereg loss 2.3749022 regularization 942.2128 reg_novel 7.2062645
loss 3.3243213
STEP 262 ================================
prereg loss 2.3797543 regularization 937.93085 reg_novel 7.2424936
loss 3.3249278
STEP 263 ================================
prereg loss 2.3856778 regularization 933.6995 reg_novel 7.2805905
loss 3.326658
STEP 264 ================================
prereg loss 2.3947248 regularization 929.4951 reg_novel 7.313953
loss 3.331534
STEP 265 ================================
prereg loss 2.4029365 regularization 925.3028 reg_novel 7.343813
loss 3.3355832
STEP 266 ================================
prereg loss 2.4139743 regularization 921.1444 reg_novel 7.3714166
loss 3.3424902
STEP 267 ================================
prereg loss 2.4243793 regularization 917.01587 reg_novel 7.4047966
loss 3.3488002
STEP 268 ================================
prereg loss 2.435163 regularization 912.90955 reg_novel 7.4433117
loss 3.355516
STEP 269 ================================
prereg loss 2.4456315 regularization 908.81323 reg_novel 7.478849
loss 3.3619237
STEP 270 ================================
prereg loss 2.4562497 regularization 904.6922 reg_novel 7.5189605
loss 3.368461
STEP 271 ================================
prereg loss 2.4664414 regularization 900.6104 reg_novel 7.5601134
loss 3.3746119
STEP 272 ================================
prereg loss 2.476857 regularization 896.5509 reg_novel 7.6048326
loss 3.3810127
STEP 273 ================================
prereg loss 2.486916 regularization 892.5224 reg_novel 7.6426806
loss 3.3870811
STEP 274 ================================
prereg loss 2.4962156 regularization 888.50903 reg_novel 7.673767
loss 3.3923984
STEP 275 ================================
prereg loss 2.5077178 regularization 884.5267 reg_novel 7.703041
loss 3.3999476
STEP 276 ================================
prereg loss 2.5220346 regularization 880.5857 reg_novel 7.729949
loss 3.4103503
STEP 277 ================================
prereg loss 2.5343976 regularization 876.6872 reg_novel 7.7525935
loss 3.4188375
STEP 278 ================================
prereg loss 2.5456114 regularization 872.8001 reg_novel 7.772416
loss 3.426184
STEP 279 ================================
prereg loss 2.5584157 regularization 868.9467 reg_novel 7.7872934
loss 3.4351497
STEP 280 ================================
prereg loss 2.5751965 regularization 865.11176 reg_novel 7.811734
loss 3.44812
STEP 281 ================================
prereg loss 2.5880973 regularization 861.31 reg_novel 7.8385105
loss 3.4572458
STEP 282 ================================
prereg loss 2.5993574 regularization 857.54565 reg_novel 7.8735023
loss 3.4647765
STEP 283 ================================
prereg loss 2.6107323 regularization 853.7709 reg_novel 7.915062
loss 3.4724183
STEP 284 ================================
prereg loss 2.624599 regularization 850.0166 reg_novel 7.957265
loss 3.482573
STEP 285 ================================
prereg loss 2.640629 regularization 846.291 reg_novel 8.002339
loss 3.4949224
STEP 286 ================================
prereg loss 2.654165 regularization 842.56976 reg_novel 8.04595
loss 3.5047808
STEP 287 ================================
prereg loss 2.6652713 regularization 838.87683 reg_novel 8.089751
loss 3.512238
STEP 288 ================================
prereg loss 2.6748111 regularization 835.20667 reg_novel 8.142732
loss 3.5181606
STEP 289 ================================
prereg loss 2.6841097 regularization 831.5455 reg_novel 8.196848
loss 3.523852
STEP 290 ================================
prereg loss 2.692444 regularization 827.8907 reg_novel 8.249518
loss 3.5285845
STEP 291 ================================
prereg loss 2.6998549 regularization 824.2713 reg_novel 8.300808
loss 3.5324268
STEP 292 ================================
prereg loss 2.7054663 regularization 820.70435 reg_novel 8.356596
loss 3.5345273
STEP 293 ================================
prereg loss 2.709093 regularization 817.1681 reg_novel 8.39988
loss 3.534661
STEP 294 ================================
prereg loss 2.7114525 regularization 813.6254 reg_novel 8.427297
loss 3.5335052
STEP 295 ================================
prereg loss 2.712452 regularization 810.0998 reg_novel 8.449058
loss 3.5310009
STEP 296 ================================
prereg loss 2.714697 regularization 806.60297 reg_novel 8.472202
loss 3.529772
STEP 297 ================================
prereg loss 2.7155344 regularization 803.1406 reg_novel 8.500714
loss 3.527176
STEP 298 ================================
prereg loss 2.7126324 regularization 799.6719 reg_novel 8.539698
loss 3.520844
STEP 299 ================================
prereg loss 2.709095 regularization 796.1973 reg_novel 8.583141
loss 3.5138755
STEP 300 ================================
prereg loss 2.7066982 regularization 792.7329 reg_novel 8.624597
loss 3.5080557
STEP 301 ================================
prereg loss 2.7052562 regularization 789.3062 reg_novel 8.666398
loss 3.503229
STEP 302 ================================
prereg loss 2.702933 regularization 785.9054 reg_novel 8.713533
loss 3.497552
STEP 303 ================================
prereg loss 2.7045429 regularization 782.54266 reg_novel 8.762457
loss 3.4958482
STEP 304 ================================
prereg loss 2.7046282 regularization 779.2204 reg_novel 8.81144
loss 3.49266
STEP 305 ================================
prereg loss 2.7045665 regularization 775.89374 reg_novel 8.857312
loss 3.4893174
STEP 306 ================================
prereg loss 2.7061121 regularization 772.5906 reg_novel 8.904711
loss 3.4876075
STEP 307 ================================
prereg loss 2.7045622 regularization 769.3117 reg_novel 8.955998
loss 3.48283
STEP 308 ================================
prereg loss 2.7038836 regularization 766.0411 reg_novel 9.014557
loss 3.4789393
STEP 309 ================================
prereg loss 2.7018611 regularization 762.78345 reg_novel 9.072616
loss 3.4737172
STEP 310 ================================
prereg loss 2.692131 regularization 759.5399 reg_novel 9.129341
loss 3.4608004
STEP 311 ================================
prereg loss 2.6829712 regularization 756.3286 reg_novel 9.179574
loss 3.4484794
STEP 312 ================================
prereg loss 2.6743484 regularization 753.15967 reg_novel 9.222378
loss 3.4367304
STEP 313 ================================
prereg loss 2.669218 regularization 750.02765 reg_novel 9.25252
loss 3.4284983
STEP 314 ================================
prereg loss 2.6627793 regularization 746.9241 reg_novel 9.279229
loss 3.4189827
STEP 315 ================================
prereg loss 2.6549628 regularization 743.83307 reg_novel 9.316652
loss 3.4081125
STEP 316 ================================
prereg loss 2.6428623 regularization 740.75305 reg_novel 9.35121
loss 3.3929667
STEP 317 ================================
prereg loss 2.6277907 regularization 737.6983 reg_novel 9.378999
loss 3.374868
STEP 318 ================================
prereg loss 2.6121955 regularization 734.6664 reg_novel 9.398868
loss 3.3562608
STEP 319 ================================
prereg loss 2.5990036 regularization 731.6618 reg_novel 9.428726
loss 3.340094
STEP 320 ================================
prereg loss 2.5910614 regularization 728.6674 reg_novel 9.46282
loss 3.3291917
STEP 321 ================================
prereg loss 2.5876186 regularization 725.6773 reg_novel 9.504588
loss 3.3228006
STEP 322 ================================
prereg loss 2.5864472 regularization 722.6879 reg_novel 9.551915
loss 3.3186872
STEP 323 ================================
prereg loss 2.5902598 regularization 719.70184 reg_novel 9.608563
loss 3.3195703
STEP 324 ================================
prereg loss 2.5962124 regularization 716.7263 reg_novel 9.669859
loss 3.3226085
STEP 325 ================================
prereg loss 2.5982685 regularization 713.7643 reg_novel 9.7442255
loss 3.321777
STEP 326 ================================
prereg loss 2.5982802 regularization 710.80475 reg_novel 9.814466
loss 3.3188994
STEP 327 ================================
prereg loss 2.5999234 regularization 707.8755 reg_novel 9.877408
loss 3.3176763
STEP 328 ================================
prereg loss 2.600173 regularization 704.96515 reg_novel 9.94257
loss 3.3150806
STEP 329 ================================
prereg loss 2.5991762 regularization 702.0787 reg_novel 10.013306
loss 3.311268
STEP 330 ================================
prereg loss 2.599227 regularization 699.21075 reg_novel 10.07346
loss 3.3085113
STEP 331 ================================
prereg loss 2.5992696 regularization 696.35785 reg_novel 10.131678
loss 3.3057592
STEP 332 ================================
prereg loss 2.596548 regularization 693.5625 reg_novel 10.199502
loss 3.3003101
STEP 333 ================================
prereg loss 2.589431 regularization 690.7772 reg_novel 10.264889
loss 3.2904732
STEP 334 ================================
prereg loss 2.584026 regularization 687.994 reg_novel 10.327414
loss 3.2823477
STEP 335 ================================
prereg loss 2.5769813 regularization 685.2383 reg_novel 10.388005
loss 3.2726076
STEP 336 ================================
prereg loss 2.5696137 regularization 682.51373 reg_novel 10.447913
loss 3.2625754
STEP 337 ================================
prereg loss 2.5610487 regularization 679.8086 reg_novel 10.5076275
loss 3.251365
STEP 338 ================================
prereg loss 2.5535815 regularization 677.1324 reg_novel 10.5750675
loss 3.241289
STEP 339 ================================
prereg loss 2.5516958 regularization 674.47034 reg_novel 10.637972
loss 3.2368042
STEP 340 ================================
prereg loss 2.5523124 regularization 671.8221 reg_novel 10.697268
loss 3.2348318
STEP 341 ================================
prereg loss 2.5502372 regularization 669.1862 reg_novel 10.755636
loss 3.230179
STEP 342 ================================
prereg loss 2.5452204 regularization 666.575 reg_novel 10.812081
loss 3.2226076
STEP 343 ================================
prereg loss 2.5376916 regularization 664.00354 reg_novel 10.869456
loss 3.2125645
STEP 344 ================================
prereg loss 2.5269973 regularization 661.43005 reg_novel 10.927417
loss 3.199355
STEP 345 ================================
prereg loss 2.5156357 regularization 658.8767 reg_novel 10.993435
loss 3.1855059
STEP 346 ================================
prereg loss 2.5012743 regularization 656.34015 reg_novel 11.066049
loss 3.1686807
STEP 347 ================================
prereg loss 2.485311 regularization 653.8467 reg_novel 11.129957
loss 3.1502876
STEP 348 ================================
prereg loss 2.469679 regularization 651.34863 reg_novel 11.191048
loss 3.1322188
STEP 349 ================================
prereg loss 2.4540915 regularization 648.8698 reg_novel 11.250314
loss 3.1142118
STEP 350 ================================
prereg loss 2.4391658 regularization 646.4172 reg_novel 11.314249
loss 3.0968974
STEP 351 ================================
prereg loss 2.4272087 regularization 643.97205 reg_novel 11.383176
loss 3.0825639
STEP 352 ================================
prereg loss 2.418974 regularization 641.54956 reg_novel 11.446149
loss 3.0719697
STEP 353 ================================
prereg loss 2.4121037 regularization 639.14966 reg_novel 11.516438
loss 3.06277
STEP 354 ================================
prereg loss 2.402228 regularization 636.7538 reg_novel 11.583376
loss 3.0505652
STEP 355 ================================
prereg loss 2.3913789 regularization 634.38293 reg_novel 11.641024
loss 3.0374029
STEP 356 ================================
prereg loss 2.386071 regularization 632.0206 reg_novel 11.706082
loss 3.0297976
STEP 357 ================================
prereg loss 2.3841186 regularization 629.6644 reg_novel 11.771062
loss 3.0255542
STEP 358 ================================
prereg loss 2.3820183 regularization 627.32684 reg_novel 11.847678
loss 3.0211928
STEP 359 ================================
prereg loss 2.375613 regularization 624.9987 reg_novel 11.915704
loss 3.0125275
STEP 360 ================================
prereg loss 2.3697739 regularization 622.6713 reg_novel 11.978776
loss 3.004424
STEP 361 ================================
prereg loss 2.3644013 regularization 620.37805 reg_novel 12.043878
loss 2.9968233
STEP 362 ================================
prereg loss 2.3616488 regularization 618.10815 reg_novel 12.10613
loss 2.9918633
STEP 363 ================================
prereg loss 2.3613694 regularization 615.8571 reg_novel 12.1694145
loss 2.9893959
STEP 364 ================================
prereg loss 2.3567421 regularization 613.59827 reg_novel 12.237064
loss 2.9825776
STEP 365 ================================
prereg loss 2.346918 regularization 611.34314 reg_novel 12.3057375
loss 2.970567
STEP 366 ================================
prereg loss 2.336989 regularization 609.1174 reg_novel 12.372812
loss 2.9584792
STEP 367 ================================
prereg loss 2.331765 regularization 606.908 reg_novel 12.4418545
loss 2.951115
STEP 368 ================================
prereg loss 2.3287826 regularization 604.7167 reg_novel 12.503322
loss 2.9460025
STEP 369 ================================
prereg loss 2.3236244 regularization 602.5465 reg_novel 12.5577755
loss 2.9387288
STEP 370 ================================
prereg loss 2.3185427 regularization 600.3681 reg_novel 12.619715
loss 2.9315305
STEP 371 ================================
prereg loss 2.3160691 regularization 598.2098 reg_novel 12.673442
loss 2.9269524
STEP 372 ================================
prereg loss 2.312556 regularization 596.0694 reg_novel 12.722432
loss 2.9213479
STEP 373 ================================
prereg loss 2.307451 regularization 593.9436 reg_novel 12.772801
loss 2.9141674
STEP 374 ================================
prereg loss 2.2995043 regularization 591.8284 reg_novel 12.828151
loss 2.904161
STEP 375 ================================
prereg loss 2.2916443 regularization 589.7223 reg_novel 12.895039
loss 2.8942618
STEP 376 ================================
prereg loss 2.2846828 regularization 587.63226 reg_novel 12.962994
loss 2.885278
STEP 377 ================================
prereg loss 2.2751644 regularization 585.5484 reg_novel 13.022372
loss 2.8737352
STEP 378 ================================
prereg loss 2.2673342 regularization 583.49207 reg_novel 13.089
loss 2.8639154
STEP 379 ================================
prereg loss 2.2582886 regularization 581.4567 reg_novel 13.152686
loss 2.8528981
STEP 380 ================================
prereg loss 2.2497935 regularization 579.432 reg_novel 13.218835
loss 2.8424444
STEP 381 ================================
prereg loss 2.239751 regularization 577.4186 reg_novel 13.280831
loss 2.8304505
STEP 382 ================================
prereg loss 2.2280526 regularization 575.4281 reg_novel 13.3495455
loss 2.8168302
STEP 383 ================================
prereg loss 2.2177625 regularization 573.4477 reg_novel 13.414881
loss 2.804625
STEP 384 ================================
prereg loss 2.2110667 regularization 571.48193 reg_novel 13.47892
loss 2.7960277
STEP 385 ================================
prereg loss 2.2087734 regularization 569.528 reg_novel 13.534801
loss 2.7918363
STEP 386 ================================
prereg loss 2.203268 regularization 567.60046 reg_novel 13.5933895
loss 2.784462
STEP 387 ================================
prereg loss 2.1969378 regularization 565.6821 reg_novel 13.6578045
loss 2.7762778
STEP 388 ================================
prereg loss 2.1873674 regularization 563.7787 reg_novel 13.720095
loss 2.7648664
STEP 389 ================================
prereg loss 2.176332 regularization 561.8739 reg_novel 13.785775
loss 2.7519917
STEP 390 ================================
prereg loss 2.1615105 regularization 559.9906 reg_novel 13.864072
loss 2.7353652
STEP 391 ================================
prereg loss 2.1499565 regularization 558.11237 reg_novel 13.947888
loss 2.7220168
STEP 392 ================================
prereg loss 2.1424313 regularization 556.2415 reg_novel 14.021853
loss 2.7126946
STEP 393 ================================
prereg loss 2.1366656 regularization 554.38403 reg_novel 14.099477
loss 2.7051492
STEP 394 ================================
prereg loss 2.1259491 regularization 552.54803 reg_novel 14.175438
loss 2.6926727
STEP 395 ================================
prereg loss 2.1161535 regularization 550.7424 reg_novel 14.245424
loss 2.6811414
STEP 396 ================================
prereg loss 2.1080098 regularization 548.94836 reg_novel 14.312712
loss 2.6712708
STEP 397 ================================
prereg loss 2.1027873 regularization 547.1556 reg_novel 14.370246
loss 2.664313
STEP 398 ================================
prereg loss 2.101533 regularization 545.38837 reg_novel 14.432315
loss 2.6613536
STEP 399 ================================
prereg loss 2.0983357 regularization 543.63544 reg_novel 14.491676
loss 2.656463
STEP 400 ================================
prereg loss 2.0921922 regularization 541.916 reg_novel 14.55747
loss 2.6486657
STEP 401 ================================
prereg loss 2.0851018 regularization 540.18494 reg_novel 14.612346
loss 2.6398993
STEP 402 ================================
prereg loss 2.078949 regularization 538.47125 reg_novel 14.669636
loss 2.6320899
STEP 403 ================================
prereg loss 2.0706227 regularization 536.7678 reg_novel 14.731632
loss 2.6221223
STEP 404 ================================
prereg loss 2.0637016 regularization 535.0649 reg_novel 14.802416
loss 2.6135688
STEP 405 ================================
prereg loss 2.0610003 regularization 533.3735 reg_novel 14.88096
loss 2.6092548
STEP 406 ================================
prereg loss 2.0552032 regularization 531.66925 reg_novel 14.96462
loss 2.6018372
STEP 407 ================================
prereg loss 2.051429 regularization 529.97754 reg_novel 15.052224
loss 2.596459
STEP 408 ================================
prereg loss 2.0432506 regularization 528.30695 reg_novel 15.136743
loss 2.5866942
STEP 409 ================================
prereg loss 2.0342288 regularization 526.64984 reg_novel 15.217407
loss 2.576096
STEP 410 ================================
prereg loss 2.0262115 regularization 525.0132 reg_novel 15.290596
loss 2.5665154
STEP 411 ================================
prereg loss 2.0194602 regularization 523.40234 reg_novel 15.370831
loss 2.5582335
STEP 412 ================================
prereg loss 2.016376 regularization 521.7984 reg_novel 15.453238
loss 2.5536277
STEP 413 ================================
prereg loss 2.012834 regularization 520.21356 reg_novel 15.52822
loss 2.5485759
STEP 414 ================================
prereg loss 2.0067146 regularization 518.64996 reg_novel 15.599311
loss 2.540964
STEP 415 ================================
prereg loss 1.9988735 regularization 517.07434 reg_novel 15.683009
loss 2.5316308
STEP 416 ================================
prereg loss 1.9889351 regularization 515.51855 reg_novel 15.766535
loss 2.5202203
STEP 417 ================================
prereg loss 1.9788107 regularization 513.98987 reg_novel 15.845449
loss 2.508646
STEP 418 ================================
prereg loss 1.9679633 regularization 512.4622 reg_novel 15.918418
loss 2.496344
STEP 419 ================================
prereg loss 1.9613091 regularization 510.93936 reg_novel 15.989247
loss 2.4882376
STEP 420 ================================
prereg loss 1.9547207 regularization 509.4341 reg_novel 16.062786
loss 2.4802177
STEP 421 ================================
prereg loss 1.9483341 regularization 507.94092 reg_novel 16.132792
loss 2.4724078
STEP 422 ================================
prereg loss 1.9449602 regularization 506.4624 reg_novel 16.208858
loss 2.4676316
STEP 423 ================================
prereg loss 1.9419795 regularization 504.98758 reg_novel 16.292812
loss 2.46326
STEP 424 ================================
prereg loss 1.9393109 regularization 503.5165 reg_novel 16.37059
loss 2.459198
STEP 425 ================================
prereg loss 1.9375517 regularization 502.07654 reg_novel 16.452599
loss 2.456081
STEP 426 ================================
prereg loss 1.9331313 regularization 500.64255 reg_novel 16.518978
loss 2.4502928
STEP 427 ================================
prereg loss 1.9304254 regularization 499.21912 reg_novel 16.582575
loss 2.446227
STEP 428 ================================
prereg loss 1.9297051 regularization 497.80762 reg_novel 16.640862
loss 2.4441538
STEP 429 ================================
prereg loss 1.927342 regularization 496.39032 reg_novel 16.714378
loss 2.4404469
STEP 430 ================================
prereg loss 1.9244019 regularization 494.9827 reg_novel 16.795736
loss 2.4361804
STEP 431 ================================
prereg loss 1.9197198 regularization 493.57388 reg_novel 16.871292
loss 2.430165
STEP 432 ================================
prereg loss 1.914632 regularization 492.1909 reg_novel 16.940876
loss 2.4237638
STEP 433 ================================
prereg loss 1.9130573 regularization 490.83228 reg_novel 17.002695
loss 2.4208922
STEP 434 ================================
prereg loss 1.9108742 regularization 489.49026 reg_novel 17.063303
loss 2.4174278
STEP 435 ================================
prereg loss 1.9077384 regularization 488.16287 reg_novel 17.125322
loss 2.4130268
STEP 436 ================================
prereg loss 1.9049159 regularization 486.83798 reg_novel 17.188519
loss 2.4089425
STEP 437 ================================
prereg loss 1.8983256 regularization 485.52472 reg_novel 17.246767
loss 2.401097
STEP 438 ================================
prereg loss 1.8896258 regularization 484.2119 reg_novel 17.31613
loss 2.3911538
STEP 439 ================================
prereg loss 1.8810093 regularization 482.88165 reg_novel 17.39833
loss 2.3812895
STEP 440 ================================
prereg loss 1.8737844 regularization 481.55338 reg_novel 17.486986
loss 2.372825
STEP 441 ================================
prereg loss 1.8688713 regularization 480.23636 reg_novel 17.568014
loss 2.3666759
STEP 442 ================================
prereg loss 1.8624109 regularization 478.93954 reg_novel 17.63819
loss 2.3589888
STEP 443 ================================
prereg loss 1.8586097 regularization 477.66098 reg_novel 17.712076
loss 2.3539827
STEP 444 ================================
prereg loss 1.8521132 regularization 476.37726 reg_novel 17.78367
loss 2.3462741
STEP 445 ================================
prereg loss 1.8474884 regularization 475.11496 reg_novel 17.850636
loss 2.340454
STEP 446 ================================
prereg loss 1.8426399 regularization 473.86652 reg_novel 17.906208
loss 2.3344126
STEP 447 ================================
prereg loss 1.8386601 regularization 472.63925 reg_novel 17.960098
loss 2.3292594
STEP 448 ================================
prereg loss 1.8354225 regularization 471.41956 reg_novel 18.020437
loss 2.3248625
STEP 449 ================================
prereg loss 1.8314912 regularization 470.1865 reg_novel 18.066628
loss 2.3197443
STEP 450 ================================
prereg loss 1.8256487 regularization 468.96686 reg_novel 18.114386
loss 2.3127298
STEP 451 ================================
prereg loss 1.8181709 regularization 467.76526 reg_novel 18.168869
loss 2.304105
STEP 452 ================================
prereg loss 1.8070946 regularization 466.5416 reg_novel 18.229027
loss 2.2918653
STEP 453 ================================
prereg loss 1.7976068 regularization 465.3259 reg_novel 18.299515
loss 2.2812324
STEP 454 ================================
prereg loss 1.7910559 regularization 464.12033 reg_novel 18.368032
loss 2.2735443
STEP 455 ================================
prereg loss 1.7863691 regularization 462.92096 reg_novel 18.44117
loss 2.2677312
STEP 456 ================================
prereg loss 1.7847422 regularization 461.74194 reg_novel 18.512554
loss 2.2649968
STEP 457 ================================
prereg loss 1.7809093 regularization 460.58603 reg_novel 18.576796
loss 2.2600722
STEP 458 ================================
prereg loss 1.7768997 regularization 459.4185 reg_novel 18.632988
loss 2.2549512
STEP 459 ================================
prereg loss 1.7737818 regularization 458.27615 reg_novel 18.67159
loss 2.2507296
STEP 460 ================================
prereg loss 1.7697783 regularization 457.1604 reg_novel 18.711903
loss 2.2456505
STEP 461 ================================
prereg loss 1.7660873 regularization 456.0429 reg_novel 18.75854
loss 2.2408888
STEP 462 ================================
prereg loss 1.7616909 regularization 454.9244 reg_novel 18.811739
loss 2.235427
STEP 463 ================================
prereg loss 1.7557751 regularization 453.81967 reg_novel 18.864801
loss 2.2284596
STEP 464 ================================
prereg loss 1.7495445 regularization 452.70892 reg_novel 18.922499
loss 2.221176
STEP 465 ================================
prereg loss 1.7443237 regularization 451.59143 reg_novel 18.986351
loss 2.2149014
STEP 466 ================================
prereg loss 1.7409003 regularization 450.48587 reg_novel 19.050575
loss 2.2104368
STEP 467 ================================
prereg loss 1.7422754 regularization 449.39603 reg_novel 19.110243
loss 2.2107816
STEP 468 ================================
prereg loss 1.744669 regularization 448.30646 reg_novel 19.172913
loss 2.2121484
STEP 469 ================================
prereg loss 1.744497 regularization 447.22632 reg_novel 19.225082
loss 2.2109485
STEP 470 ================================
prereg loss 1.7426897 regularization 446.1572 reg_novel 19.269526
loss 2.2081165
STEP 471 ================================
prereg loss 1.7418394 regularization 445.09375 reg_novel 19.321678
loss 2.206255
STEP 472 ================================
prereg loss 1.7388506 regularization 444.03534 reg_novel 19.37214
loss 2.202258
STEP 473 ================================
prereg loss 1.7363925 regularization 442.9914 reg_novel 19.421385
loss 2.1988053
STEP 474 ================================
prereg loss 1.7358904 regularization 441.94894 reg_novel 19.4769
loss 2.1973162
STEP 475 ================================
prereg loss 1.737453 regularization 440.9048 reg_novel 19.534737
loss 2.1978924
STEP 476 ================================
prereg loss 1.7393699 regularization 439.868 reg_novel 19.5894
loss 2.1988273
STEP 477 ================================
prereg loss 1.7440587 regularization 438.84354 reg_novel 19.645498
loss 2.2025478
STEP 478 ================================
prereg loss 1.7467259 regularization 437.82336 reg_novel 19.701815
loss 2.204251
STEP 479 ================================
prereg loss 1.7462758 regularization 436.80972 reg_novel 19.760464
loss 2.202846
STEP 480 ================================
prereg loss 1.744468 regularization 435.79694 reg_novel 19.818068
loss 2.200083
STEP 481 ================================
prereg loss 1.7417389 regularization 434.80002 reg_novel 19.874155
loss 2.196413
STEP 482 ================================
prereg loss 1.7391759 regularization 433.81564 reg_novel 19.926609
loss 2.1929183
STEP 483 ================================
prereg loss 1.7361745 regularization 432.8412 reg_novel 19.975842
loss 2.1889915
STEP 484 ================================
prereg loss 1.7329577 regularization 431.88065 reg_novel 20.023643
loss 2.1848621
STEP 485 ================================
prereg loss 1.7302679 regularization 430.90573 reg_novel 20.08216
loss 2.1812558
STEP 486 ================================
prereg loss 1.7279042 regularization 429.9291 reg_novel 20.14397
loss 2.1779773
STEP 487 ================================
prereg loss 1.7272654 regularization 428.96396 reg_novel 20.21082
loss 2.1764402
STEP 488 ================================
prereg loss 1.7256359 regularization 427.99545 reg_novel 20.27671
loss 2.173908
STEP 489 ================================
prereg loss 1.7253563 regularization 427.04156 reg_novel 20.348368
loss 2.1727462
STEP 490 ================================
prereg loss 1.7236098 regularization 426.09644 reg_novel 20.410868
loss 2.1701171
STEP 491 ================================
prereg loss 1.7216959 regularization 425.15485 reg_novel 20.464819
loss 2.1673155
STEP 492 ================================
prereg loss 1.7199224 regularization 424.23477 reg_novel 20.522041
loss 2.1646793
STEP 493 ================================
prereg loss 1.7193553 regularization 423.31927 reg_novel 20.58028
loss 2.163255
STEP 494 ================================
prereg loss 1.7172517 regularization 422.40305 reg_novel 20.638933
loss 2.1602936
STEP 495 ================================
prereg loss 1.713264 regularization 421.49966 reg_novel 20.694023
loss 2.1554577
STEP 496 ================================
prereg loss 1.7085445 regularization 420.59982 reg_novel 20.7402
loss 2.1498845
STEP 497 ================================
prereg loss 1.7055947 regularization 419.71707 reg_novel 20.789522
loss 2.1461012
STEP 498 ================================
prereg loss 1.7044008 regularization 418.84726 reg_novel 20.837181
loss 2.1440852
STEP 499 ================================
prereg loss 1.7056189 regularization 417.9736 reg_novel 20.87286
loss 2.1444654
STEP 500 ================================
prereg loss 1.7070739 regularization 417.12497 reg_novel 20.911917
loss 2.1451108
2022-06-12T12:53:25.710

julia> after_532_steps = deepcopy(trainable["network_matrix"])
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 21 entries:
  "id-2"      => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.0005039, "true"=>0.000286836, "dot"=>0.000117107, "false"=>0.…
  "output"    => Dict("dict-2"=>Dict("id-2"=>Dict("dict"=>-0.113698, "true"=>0.113455, "dot"=>0.215947, "false"=>0.3234…
  "accum-4"   => Dict("dict"=>Dict("id-2"=>Dict("dict"=>2.9685f-5, "true"=>0.114913, "dot"=>0.00145751, "false"=>3.1290…
  "dot-2"     => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000671144, "true"=>0.00127245, "dot"=>0.00122243, "false"=>0.…
  "norm-1"    => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.0165836, "true"=>-0.311343, "dot"=>0.157257, "false"=>-0.0131…
  "id-1"      => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000138099, "true"=>-0.00017642, "dot"=>-7.13765f-5, "false"=>…
  "accum-3"   => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000477225, "true"=>0.000645986, "dot"=>-4.20968f-5, "false"=>…
  "norm-4"    => Dict("dict"=>Dict("id-2"=>Dict("dict"=>-0.0117977, "true"=>-0.0317267, "dot"=>-0.316072, "false"=>0.09…
  "id-4"      => Dict("dict"=>Dict("id-2"=>Dict("dict"=>9.25775f-5, "true"=>0.00901038, "dot"=>0.00435005, "false"=>0.0…
  "accum-1"   => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000565337, "true"=>0.000272644, "dot"=>0.000159473, "false"=>…
  "compare-4" => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000438848, "true"=>0.00634237, "dot"=>0.000381375, "false"=>0…
  "compare-2" => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.013285, "true"=>0.0108362, "dot"=>0.000107019, "false"=>0.000…
  "dot-1"     => Dict("dict"=>Dict("id-2"=>Dict("dict"=>4.15578f-5, "true"=>0.0492862, "dot"=>0.000427196, "false"=>0.0…
  "dot-3"     => Dict("dict"=>Dict("id-2"=>Dict("dict"=>-2.73908f-6, "true"=>2.08535f-5, "dot"=>-0.0689736, "false"=>5.…
  "norm-3"    => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.203043, "true"=>0.171873, "dot"=>0.0857674, "false"=>-0.2575,…
  "compare-3" => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.00058093, "true"=>0.000488873, "dot"=>0.000364526, "false"=>0…
  "accum-2"   => Dict("dict"=>Dict("id-2"=>Dict("dict"=>-2.8449f-5, "true"=>8.27379f-5, "dot"=>0.000661398, "false"=>0.…
  "compare-1" => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.00045049, "true"=>0.000329064, "dot"=>0.000323557, "false"=>0…
  "dot-4"     => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.000234399, "true"=>4.94201f-6, "dot"=>0.000449174, "false"=>8…
  "norm-2"    => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.290335, "true"=>-0.105024, "dot"=>-0.194988, "false"=>-0.3986…
  "id-3"      => Dict("dict"=>Dict("id-2"=>Dict("dict"=>0.00131249, "true"=>0.000660721, "dot"=>0.000738066, "false"=>0…

julia> trainable["fixed_matrix"]
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 2 entries:
  "timer" => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "input" => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  
julia> serialize("after-532-steps.ser", after_532_steps)

julia> serialize("opt_after-532_steps.ser", opt)

julia> close(io)
```
