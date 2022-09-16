# Results of the first run

Initialization needs to be adjusted down with all these `id_transforms`. 

We should return to a more classic L1+L2 regularization in this setup, as this setup is "almost feedforward".

We have 2000 steps here, and for a while there was an impression that there might be slow convergence,
but at the end it turned out not to be the case. 

The `log-file.txt` printout contains plenty of interesting data.

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> steps!(2)
2022-06-13T10:50:55.757
STEP 1 ================================
prereg loss 6.2903684e7 regularization 1215.8446 reg_novel 523.26056
loss 6.2903684e7
STEP 2 ================================
prereg loss 5.9157652e7 regularization 1214.7319 reg_novel 512.5788
loss 5.9157652e7
2022-06-13T10:53:02.465

julia> steps!(30)
2022-06-13T10:55:43.839
STEP 1 ================================
prereg loss 5.537667e7 regularization 1213.5269 reg_novel 502.36807
loss 5.537667e7
STEP 2 ================================
prereg loss 5.2062412e7 regularization 1212.2894 reg_novel 492.42978
loss 5.2062412e7
STEP 3 ================================
prereg loss 4.8986828e7 regularization 1211.0258 reg_novel 482.8436
loss 4.8986828e7
STEP 4 ================================
prereg loss 4.6100636e7 regularization 1209.7349 reg_novel 473.58124
loss 4.6100636e7
STEP 5 ================================
prereg loss 4.339643e7 regularization 1208.4187 reg_novel 464.64633
loss 4.339643e7
STEP 6 ================================
prereg loss 4.081405e7 regularization 1207.0828 reg_novel 456.03763
loss 4.081405e7
STEP 7 ================================
prereg loss 3.827527e7 regularization 1205.7172 reg_novel 447.7444
loss 3.827527e7
STEP 8 ================================
prereg loss 3.5926572e7 regularization 1204.3285 reg_novel 439.75952
loss 3.5926572e7
STEP 9 ================================
prereg loss 3.3595868e7 regularization 1202.9192 reg_novel 432.07376
loss 3.3595868e7
STEP 10 ================================
prereg loss 3.1448358e7 regularization 1201.5042 reg_novel 424.6754
loss 3.144836e7
STEP 11 ================================
prereg loss 2.9322888e7 regularization 1200.0881 reg_novel 417.56696
loss 2.932289e7
STEP 12 ================================
prereg loss 2.734798e7 regularization 1198.6635 reg_novel 410.74292
loss 2.7347982e7
STEP 13 ================================
prereg loss 2.538355e7 regularization 1197.2142 reg_novel 404.18994
loss 2.5383552e7
STEP 14 ================================
prereg loss 2.3507524e7 regularization 1195.7561 reg_novel 397.89606
loss 2.3507526e7
STEP 15 ================================
prereg loss 2.1736924e7 regularization 1194.2809 reg_novel 391.85455
loss 2.1736926e7
STEP 16 ================================
prereg loss 2.0023624e7 regularization 1192.7926 reg_novel 386.04688
loss 2.0023626e7
STEP 17 ================================
prereg loss 1.8375496e7 regularization 1191.2832 reg_novel 380.494
loss 1.8375498e7
STEP 18 ================================
prereg loss 1.683376e7 regularization 1189.7635 reg_novel 375.1895
loss 1.6833762e7
STEP 19 ================================
prereg loss 1.5363679e7 regularization 1188.22 reg_novel 370.1075
loss 1.5363681e7
STEP 20 ================================
prereg loss 1.3942704e7 regularization 1186.6533 reg_novel 365.25793
loss 1.3942706e7
STEP 21 ================================
prereg loss 1.2616909e7 regularization 1185.0658 reg_novel 360.63385
loss 1.2616911e7
STEP 22 ================================
prereg loss 1.1342845e7 regularization 1183.4463 reg_novel 356.2209
loss 1.1342847e7
STEP 23 ================================
prereg loss 1.0161048e7 regularization 1181.7969 reg_novel 352.01376
loss 1.016105e7
STEP 24 ================================
prereg loss 9.01951e6 regularization 1180.1306 reg_novel 348.01654
loss 9.019512e6
STEP 25 ================================
prereg loss 7.9845115e6 regularization 1178.4403 reg_novel 344.2412
loss 7.984513e6
STEP 26 ================================
prereg loss 7.0064835e6 regularization 1176.7245 reg_novel 340.64978
loss 7.006485e6
STEP 27 ================================
prereg loss 6.1012145e6 regularization 1174.987 reg_novel 337.2345
loss 6.101216e6
STEP 28 ================================
prereg loss 5.277653e6 regularization 1173.2399 reg_novel 333.98026
loss 5.2776545e6
STEP 29 ================================
prereg loss 4.5123925e6 regularization 1171.463 reg_novel 330.88223
loss 4.512394e6
STEP 30 ================================
prereg loss 3.8173895e6 regularization 1169.6561 reg_novel 327.9285
loss 3.817391e6
2022-06-13T11:19:15.571

julia> steps!(200)
2022-06-13T13:07:56.828
STEP 1 ================================
prereg loss 3.2031362e6 regularization 1167.8208 reg_novel 325.1478
loss 3.2031378e6
STEP 2 ================================
prereg loss 2.655687e6 regularization 1165.9701 reg_novel 322.49927
loss 2.6556885e6
STEP 3 ================================
prereg loss 2.1743582e6 regularization 1164.0885 reg_novel 319.9714
loss 2.1743598e6
STEP 4 ================================
prereg loss 1.7421419e6 regularization 1162.18 reg_novel 317.54724
loss 1.7421434e6
STEP 5 ================================
prereg loss 1.3754042e6 regularization 1160.238 reg_novel 315.22308
loss 1.3754058e6
STEP 6 ================================
prereg loss 1.0747565e6 regularization 1158.2861 reg_novel 312.99994
loss 1.074758e6
STEP 7 ================================
prereg loss 837049.4 regularization 1156.2991 reg_novel 310.8656
loss 837050.8
STEP 8 ================================
prereg loss 657749.2 regularization 1154.2847 reg_novel 308.8153
loss 657750.6
STEP 9 ================================
prereg loss 529450.25 regularization 1152.2383 reg_novel 306.84454
loss 529451.7
STEP 10 ================================
prereg loss 441709.94 regularization 1150.1537 reg_novel 304.946
loss 441711.4
STEP 11 ================================
prereg loss 386487.88 regularization 1148.0476 reg_novel 303.11404
loss 386489.3
STEP 12 ================================
prereg loss 355531.12 regularization 1145.9081 reg_novel 301.34222
loss 355532.56
STEP 13 ================================
prereg loss 341909.5 regularization 1143.759 reg_novel 299.62238
loss 341910.94
STEP 14 ================================
prereg loss 337420.12 regularization 1141.581 reg_novel 297.9506
loss 337421.56
STEP 15 ================================
prereg loss 335838.0 regularization 1139.375 reg_novel 296.3239
loss 335839.44
STEP 16 ================================
prereg loss 333460.7 regularization 1137.1285 reg_novel 294.7413
loss 333462.12
STEP 17 ================================
prereg loss 326598.03 regularization 1134.8584 reg_novel 293.19894
loss 326599.47
STEP 18 ================================
prereg loss 314713.53 regularization 1132.5549 reg_novel 291.69504
loss 314714.97
STEP 19 ================================
prereg loss 299085.12 regularization 1130.2279 reg_novel 290.22925
loss 299086.53
STEP 20 ================================
prereg loss 280670.47 regularization 1127.8813 reg_novel 288.80164
loss 280671.88
STEP 21 ================================
prereg loss 260580.34 regularization 1125.5215 reg_novel 287.4108
loss 260581.75
STEP 22 ================================
prereg loss 239755.1 regularization 1123.1534 reg_novel 286.05347
loss 239756.5
STEP 23 ================================
prereg loss 219328.7 regularization 1120.7644 reg_novel 284.7293
loss 219330.11
STEP 24 ================================
prereg loss 200577.33 regularization 1118.3698 reg_novel 283.43988
loss 200578.73
STEP 25 ================================
prereg loss 184183.73 regularization 1115.9712 reg_novel 282.17853
loss 184185.12
STEP 26 ================================
prereg loss 171206.92 regularization 1113.5565 reg_novel 280.9446
loss 171208.31
STEP 27 ================================
prereg loss 160449.7 regularization 1111.1454 reg_novel 279.73834
loss 160451.1
STEP 28 ================================
prereg loss 151947.05 regularization 1108.7258 reg_novel 278.55566
loss 151948.44
STEP 29 ================================
prereg loss 144116.7 regularization 1106.3042 reg_novel 277.39917
loss 144118.1
STEP 30 ================================
prereg loss 136513.48 regularization 1103.8776 reg_novel 276.27112
loss 136514.86
STEP 31 ================================
prereg loss 129228.164 regularization 1101.449 reg_novel 275.17285
loss 129229.54
STEP 32 ================================
prereg loss 122497.29 regularization 1099.0186 reg_novel 274.10492
loss 122498.664
STEP 33 ================================
prereg loss 115913.09 regularization 1096.5928 reg_novel 273.06226
loss 115914.46
STEP 34 ================================
prereg loss 108932.17 regularization 1094.1766 reg_novel 272.04416
loss 108933.54
STEP 35 ================================
prereg loss 101747.54 regularization 1091.7559 reg_novel 271.04962
loss 101748.9
STEP 36 ================================
prereg loss 94544.19 regularization 1089.3412 reg_novel 270.07935
loss 94545.55
STEP 37 ================================
prereg loss 87120.695 regularization 1086.9246 reg_novel 269.13
loss 87122.055
STEP 38 ================================
prereg loss 79904.164 regularization 1084.5175 reg_novel 268.2015
loss 79905.516
STEP 39 ================================
prereg loss 76442.66 regularization 1082.1051 reg_novel 267.29123
loss 76444.01
STEP 40 ================================
prereg loss 67089.63 regularization 1079.691 reg_novel 266.3336
loss 67090.98
STEP 41 ================================
prereg loss 60082.227 regularization 1077.2731 reg_novel 265.37106
loss 60083.57
STEP 42 ================================
prereg loss 54634.3 regularization 1074.8685 reg_novel 264.43555
loss 54635.64
STEP 43 ================================
prereg loss 49800.883 regularization 1072.4634 reg_novel 263.52408
loss 49802.22
STEP 44 ================================
prereg loss 45370.46 regularization 1070.0636 reg_novel 262.64392
loss 45371.793
STEP 45 ================================
prereg loss 41172.383 regularization 1067.6624 reg_novel 261.78455
loss 41173.71
STEP 46 ================================
prereg loss 37288.48 regularization 1065.2607 reg_novel 260.9515
loss 37289.81
STEP 47 ================================
prereg loss 33953.61 regularization 1062.8634 reg_novel 260.138
loss 33954.934
STEP 48 ================================
prereg loss 31342.344 regularization 1060.4656 reg_novel 259.34436
loss 31343.664
STEP 49 ================================
prereg loss 29347.44 regularization 1058.0767 reg_novel 258.56815
loss 29348.756
STEP 50 ================================
prereg loss 27776.076 regularization 1055.6929 reg_novel 257.80362
loss 27777.39
STEP 51 ================================
prereg loss 27015.871 regularization 1053.3091 reg_novel 257.1192
loss 27017.182
STEP 52 ================================
prereg loss 25840.957 regularization 1050.9436 reg_novel 256.43863
loss 25842.264
STEP 53 ================================
prereg loss 25881.97 regularization 1048.5883 reg_novel 255.76645
loss 25883.275
STEP 54 ================================
prereg loss 26066.139 regularization 1046.2301 reg_novel 255.10051
loss 26067.44
STEP 55 ================================
prereg loss 26341.904 regularization 1043.8737 reg_novel 254.43987
loss 26343.203
STEP 56 ================================
prereg loss 26585.553 regularization 1041.5278 reg_novel 253.78777
loss 26586.848
STEP 57 ================================
prereg loss 26818.467 regularization 1039.1744 reg_novel 253.14474
loss 26819.76
STEP 58 ================================
prereg loss 27027.447 regularization 1036.8224 reg_novel 252.5119
loss 27028.736
STEP 59 ================================
prereg loss 27234.99 regularization 1034.471 reg_novel 251.88812
loss 27236.277
STEP 60 ================================
prereg loss 27411.23 regularization 1032.1302 reg_novel 251.2714
loss 27412.514
STEP 61 ================================
prereg loss 27558.877 regularization 1029.8007 reg_novel 250.65558
loss 27560.158
STEP 62 ================================
prereg loss 27563.383 regularization 1027.4747 reg_novel 250.04466
loss 27564.66
STEP 63 ================================
prereg loss 27047.686 regularization 1025.1707 reg_novel 249.43375
loss 27048.96
STEP 64 ================================
prereg loss 26287.629 regularization 1022.8856 reg_novel 248.82617
loss 26288.9
STEP 65 ================================
prereg loss 26464.574 regularization 1020.6049 reg_novel 248.23436
loss 26465.844
STEP 66 ================================
prereg loss 26575.36 regularization 1018.3171 reg_novel 247.65744
loss 26576.625
STEP 67 ================================
prereg loss 26536.688 regularization 1016.0332 reg_novel 247.0951
loss 26537.951
STEP 68 ================================
prereg loss 26383.84 regularization 1013.75574 reg_novel 246.54683
loss 26385.1
STEP 69 ================================
prereg loss 26136.191 regularization 1011.4699 reg_novel 246.01476
loss 26137.45
STEP 70 ================================
prereg loss 25861.469 regularization 1009.178 reg_novel 245.49449
loss 25862.723
STEP 71 ================================
prereg loss 25502.459 regularization 1006.88684 reg_novel 244.9845
loss 25503.71
STEP 72 ================================
prereg loss 25079.51 regularization 1004.61554 reg_novel 244.48558
loss 25080.76
STEP 73 ================================
prereg loss 24651.266 regularization 1002.3416 reg_novel 243.99449
loss 24652.512
STEP 74 ================================
prereg loss 24322.732 regularization 1000.07697 reg_novel 243.5123
loss 24323.977
STEP 75 ================================
prereg loss 24182.826 regularization 997.8082 reg_novel 243.03996
loss 24184.066
STEP 76 ================================
prereg loss 24126.754 regularization 995.551 reg_novel 242.57765
loss 24127.992
STEP 77 ================================
prereg loss 24169.785 regularization 993.29535 reg_novel 242.12361
loss 24171.021
STEP 78 ================================
prereg loss 24242.578 regularization 991.03534 reg_novel 241.67966
loss 24243.81
STEP 79 ================================
prereg loss 24346.602 regularization 988.7862 reg_novel 241.24506
loss 24347.832
STEP 80 ================================
prereg loss 24529.39 regularization 986.5453 reg_novel 240.81764
loss 24530.617
STEP 81 ================================
prereg loss 24789.607 regularization 984.3006 reg_novel 240.39761
loss 24790.832
STEP 82 ================================
prereg loss 25099.303 regularization 982.0651 reg_novel 239.98308
loss 25100.525
STEP 83 ================================
prereg loss 25312.508 regularization 979.8344 reg_novel 239.57138
loss 25313.727
STEP 84 ================================
prereg loss 25438.799 regularization 977.6113 reg_novel 239.16827
loss 25440.016
STEP 85 ================================
prereg loss 25442.676 regularization 975.39703 reg_novel 238.77184
loss 25443.89
STEP 86 ================================
prereg loss 25360.729 regularization 973.188 reg_novel 238.37259
loss 25361.94
STEP 87 ================================
prereg loss 25163.193 regularization 970.9933 reg_novel 237.98038
loss 25164.402
STEP 88 ================================
prereg loss 25009.768 regularization 968.8028 reg_novel 237.5949
loss 25010.975
STEP 89 ================================
prereg loss 24928.893 regularization 966.6082 reg_novel 237.21907
loss 24930.096
STEP 90 ================================
prereg loss 24849.742 regularization 964.41473 reg_novel 236.84882
loss 24850.943
STEP 91 ================================
prereg loss 24756.025 regularization 962.237 reg_novel 236.48328
loss 24757.225
STEP 92 ================================
prereg loss 24614.78 regularization 960.068 reg_novel 236.1222
loss 24615.975
STEP 93 ================================
prereg loss 24423.602 regularization 957.90063 reg_novel 235.76302
loss 24424.795
STEP 94 ================================
prereg loss 24137.975 regularization 955.7538 reg_novel 235.4072
loss 24139.166
STEP 95 ================================
prereg loss 23705.33 regularization 953.59894 reg_novel 235.05656
loss 23706.52
STEP 96 ================================
prereg loss 23361.58 regularization 951.45465 reg_novel 234.7095
loss 23362.766
STEP 97 ================================
prereg loss 23071.156 regularization 949.3199 reg_novel 234.36769
loss 23072.34
STEP 98 ================================
prereg loss 22831.873 regularization 947.20245 reg_novel 234.0323
loss 22833.055
STEP 99 ================================
prereg loss 22671.922 regularization 945.0951 reg_novel 233.69893
loss 22673.102
STEP 100 ================================
prereg loss 22596.314 regularization 943.0011 reg_novel 233.3699
loss 22597.49
STEP 101 ================================
prereg loss 22616.373 regularization 940.9165 reg_novel 233.0492
loss 22617.547
STEP 102 ================================
prereg loss 22650.709 regularization 938.83264 reg_novel 232.73819
loss 22651.88
STEP 103 ================================
prereg loss 22656.508 regularization 936.75916 reg_novel 232.43468
loss 22657.678
STEP 104 ================================
prereg loss 22579.861 regularization 934.69196 reg_novel 232.12993
loss 22581.027
STEP 105 ================================
prereg loss 22398.176 regularization 932.6358 reg_novel 231.83067
loss 22399.34
STEP 106 ================================
prereg loss 22112.67 regularization 930.5823 reg_novel 231.53821
loss 22113.832
STEP 107 ================================
prereg loss 21793.457 regularization 928.5416 reg_novel 231.25372
loss 21794.617
STEP 108 ================================
prereg loss 21527.178 regularization 926.50525 reg_novel 230.97963
loss 21528.336
STEP 109 ================================
prereg loss 21292.346 regularization 924.49036 reg_novel 230.7154
loss 21293.5
STEP 110 ================================
prereg loss 21080.842 regularization 922.4753 reg_novel 230.45695
loss 21081.994
STEP 111 ================================
prereg loss 20937.48 regularization 920.4741 reg_novel 230.20195
loss 20938.63
STEP 112 ================================
prereg loss 20836.03 regularization 918.4898 reg_novel 229.94917
loss 20837.178
STEP 113 ================================
prereg loss 20757.11 regularization 916.5128 reg_novel 229.70155
loss 20758.256
STEP 114 ================================
prereg loss 20705.324 regularization 914.54596 reg_novel 229.45815
loss 20706.469
STEP 115 ================================
prereg loss 20659.156 regularization 912.58813 reg_novel 229.22008
loss 20660.299
STEP 116 ================================
prereg loss 20415.893 regularization 910.6472 reg_novel 228.99059
loss 20417.031
STEP 117 ================================
prereg loss 20246.521 regularization 908.72644 reg_novel 228.76688
loss 20247.658
STEP 118 ================================
prereg loss 20140.973 regularization 906.81213 reg_novel 228.55023
loss 20142.107
STEP 119 ================================
prereg loss 20086.01 regularization 904.89905 reg_novel 228.3429
loss 20087.143
STEP 120 ================================
prereg loss 20046.695 regularization 902.9793 reg_novel 228.14265
loss 20047.826
STEP 121 ================================
prereg loss 20015.766 regularization 901.06366 reg_novel 227.94885
loss 20016.895
STEP 122 ================================
prereg loss 19964.398 regularization 899.1657 reg_novel 227.7628
loss 19965.525
STEP 123 ================================
prereg loss 19848.707 regularization 897.27124 reg_novel 227.58217
loss 19849.832
STEP 124 ================================
prereg loss 19737.238 regularization 895.3757 reg_novel 227.40562
loss 19738.361
STEP 125 ================================
prereg loss 19641.895 regularization 893.49023 reg_novel 227.22816
loss 19643.016
STEP 126 ================================
prereg loss 19556.943 regularization 891.6105 reg_novel 227.05385
loss 19558.062
STEP 127 ================================
prereg loss 19460.469 regularization 889.7375 reg_novel 226.88393
loss 19461.586
STEP 128 ================================
prereg loss 19368.033 regularization 887.8809 reg_novel 226.71785
loss 19369.148
STEP 129 ================================
prereg loss 19287.17 regularization 886.0264 reg_novel 226.55312
loss 19288.283
STEP 130 ================================
prereg loss 19226.834 regularization 884.1891 reg_novel 226.39247
loss 19227.945
STEP 131 ================================
prereg loss 19168.504 regularization 882.3701 reg_novel 226.23456
loss 19169.613
STEP 132 ================================
prereg loss 19136.488 regularization 880.5666 reg_novel 226.07903
loss 19137.596
STEP 133 ================================
prereg loss 19071.504 regularization 878.77014 reg_novel 225.93593
loss 19072.61
STEP 134 ================================
prereg loss 18996.375 regularization 876.99585 reg_novel 225.80766
loss 18997.479
STEP 135 ================================
prereg loss 18931.883 regularization 875.2329 reg_novel 225.69345
loss 18932.984
STEP 136 ================================
prereg loss 18880.719 regularization 873.4767 reg_novel 225.59283
loss 18881.818
STEP 137 ================================
prereg loss 18851.348 regularization 871.7179 reg_novel 225.50533
loss 18852.445
STEP 138 ================================
prereg loss 18820.389 regularization 869.96234 reg_novel 225.42781
loss 18821.484
STEP 139 ================================
prereg loss 18803.826 regularization 868.22186 reg_novel 225.35483
loss 18804.92
STEP 140 ================================
prereg loss 18809.463 regularization 866.50354 reg_novel 225.28781
loss 18810.555
STEP 141 ================================
prereg loss 18793.615 regularization 864.7945 reg_novel 225.22533
loss 18794.705
STEP 142 ================================
prereg loss 18752.307 regularization 863.09625 reg_novel 225.17067
loss 18753.395
STEP 143 ================================
prereg loss 18708.074 regularization 861.39795 reg_novel 225.12291
loss 18709.16
STEP 144 ================================
prereg loss 18637.861 regularization 859.71326 reg_novel 225.08566
loss 18638.945
STEP 145 ================================
prereg loss 18567.707 regularization 858.0409 reg_novel 225.05463
loss 18568.791
STEP 146 ================================
prereg loss 18494.207 regularization 856.3886 reg_novel 225.03027
loss 18495.29
STEP 147 ================================
prereg loss 18435.062 regularization 854.74963 reg_novel 225.0125
loss 18436.143
STEP 148 ================================
prereg loss 18397.676 regularization 853.1163 reg_novel 224.9992
loss 18398.754
STEP 149 ================================
prereg loss 18354.73 regularization 851.49243 reg_novel 224.98897
loss 18355.807
STEP 150 ================================
prereg loss 18309.318 regularization 849.9004 reg_novel 224.9849
loss 18310.393
STEP 151 ================================
prereg loss 18277.424 regularization 848.3128 reg_novel 224.98651
loss 18278.498
STEP 152 ================================
prereg loss 18232.803 regularization 846.728 reg_novel 224.99544
loss 18233.875
STEP 153 ================================
prereg loss 18175.031 regularization 845.14484 reg_novel 225.01494
loss 18176.102
STEP 154 ================================
prereg loss 18095.68 regularization 843.5691 reg_novel 225.04164
loss 18096.748
STEP 155 ================================
prereg loss 17997.469 regularization 842.00397 reg_novel 225.07585
loss 17998.535
STEP 156 ================================
prereg loss 17927.295 regularization 840.4496 reg_novel 225.11426
loss 17928.361
STEP 157 ================================
prereg loss 17845.295 regularization 838.8995 reg_novel 225.16167
loss 17846.36
STEP 158 ================================
prereg loss 17740.209 regularization 837.35785 reg_novel 225.21255
loss 17741.271
STEP 159 ================================
prereg loss 17629.19 regularization 835.8305 reg_novel 225.2754
loss 17630.25
STEP 160 ================================
prereg loss 17511.047 regularization 834.3063 reg_novel 225.3472
loss 17512.107
STEP 161 ================================
prereg loss 17391.799 regularization 832.78345 reg_novel 225.42986
loss 17392.857
STEP 162 ================================
prereg loss 17259.45 regularization 831.272 reg_novel 225.51836
loss 17260.506
STEP 163 ================================
prereg loss 17120.418 regularization 829.7734 reg_novel 225.61484
loss 17121.473
STEP 164 ================================
prereg loss 17007.455 regularization 828.28503 reg_novel 225.71365
loss 17008.51
STEP 165 ================================
prereg loss 16943.045 regularization 826.81195 reg_novel 225.81783
loss 16944.098
STEP 166 ================================
prereg loss 16865.598 regularization 825.3585 reg_novel 225.92598
loss 16866.648
STEP 167 ================================
prereg loss 16762.812 regularization 823.9217 reg_novel 226.04076
loss 16763.863
STEP 168 ================================
prereg loss 16679.984 regularization 822.48376 reg_novel 226.16405
loss 16681.033
STEP 169 ================================
prereg loss 16656.041 regularization 821.04016 reg_novel 226.29749
loss 16657.088
STEP 170 ================================
prereg loss 16667.533 regularization 819.5951 reg_novel 226.43924
loss 16668.58
STEP 171 ================================
prereg loss 16698.541 regularization 818.1532 reg_novel 226.59056
loss 16699.586
STEP 172 ================================
prereg loss 16739.404 regularization 816.7327 reg_novel 226.74924
loss 16740.447
STEP 173 ================================
prereg loss 16798.137 regularization 815.3172 reg_novel 226.9132
loss 16799.18
STEP 174 ================================
prereg loss 16847.857 regularization 813.92474 reg_novel 227.08154
loss 16848.898
STEP 175 ================================
prereg loss 16870.984 regularization 812.5469 reg_novel 227.25632
loss 16872.023
STEP 176 ================================
prereg loss 16895.285 regularization 811.17175 reg_novel 227.43918
loss 16896.324
STEP 177 ================================
prereg loss 16953.479 regularization 809.8136 reg_novel 227.62985
loss 16954.516
STEP 178 ================================
prereg loss 16977.947 regularization 808.46277 reg_novel 227.82498
loss 16978.984
STEP 179 ================================
prereg loss 17020.479 regularization 807.12885 reg_novel 228.02632
loss 17021.514
STEP 180 ================================
prereg loss 17074.627 regularization 805.81494 reg_novel 228.23329
loss 17075.66
STEP 181 ================================
prereg loss 17078.932 regularization 804.5041 reg_novel 228.44524
loss 17079.965
STEP 182 ================================
prereg loss 17041.71 regularization 803.19684 reg_novel 228.66727
loss 17042.742
STEP 183 ================================
prereg loss 17024.836 regularization 801.89233 reg_novel 228.8998
loss 17025.867
STEP 184 ================================
prereg loss 16942.414 regularization 800.598 reg_novel 229.13069
loss 16943.443
STEP 185 ================================
prereg loss 16727.377 regularization 799.31055 reg_novel 229.35385
loss 16728.406
STEP 186 ================================
prereg loss 16522.727 regularization 798.0375 reg_novel 229.5725
loss 16523.754
STEP 187 ================================
prereg loss 16345.277 regularization 796.7607 reg_novel 229.79466
loss 16346.304
STEP 188 ================================
prereg loss 16096.97 regularization 795.4908 reg_novel 230.018
loss 16097.995
STEP 189 ================================
prereg loss 15857.296 regularization 794.2287 reg_novel 230.2449
loss 15858.32
STEP 190 ================================
prereg loss 15661.672 regularization 792.97656 reg_novel 230.47374
loss 15662.695
STEP 191 ================================
prereg loss 15391.59 regularization 791.7355 reg_novel 230.70105
loss 15392.612
STEP 192 ================================
prereg loss 15176.024 regularization 790.5205 reg_novel 230.93394
loss 15177.046
STEP 193 ================================
prereg loss 14991.874 regularization 789.30316 reg_novel 231.16827
loss 14992.895
STEP 194 ================================
prereg loss 14793.309 regularization 788.0896 reg_novel 231.40808
loss 14794.328
STEP 195 ================================
prereg loss 14554.046 regularization 786.8837 reg_novel 231.65262
loss 14555.064
STEP 196 ================================
prereg loss 14326.73 regularization 785.6819 reg_novel 231.90498
loss 14327.748
STEP 197 ================================
prereg loss 14126.128 regularization 784.49255 reg_novel 232.1657
loss 14127.145
STEP 198 ================================
prereg loss 13941.982 regularization 783.3124 reg_novel 232.43443
loss 13942.998
STEP 199 ================================
prereg loss 13682.96 regularization 782.1431 reg_novel 232.70949
loss 13683.975
STEP 200 ================================
prereg loss 13507.176 regularization 780.9853 reg_novel 232.9945
loss 13508.189
2022-06-13T15:54:06.111

julia> steps!(100)
2022-06-13T16:32:22.260
STEP 1 ================================
prereg loss 13418.346 regularization 779.8354 reg_novel 233.29192
loss 13419.358
STEP 2 ================================
prereg loss 13363.128 regularization 778.6859 reg_novel 233.6005
loss 13364.141
STEP 3 ================================
prereg loss 13310.014 regularization 777.5507 reg_novel 233.9194
loss 13311.025
STEP 4 ================================
prereg loss 13244.559 regularization 776.42426 reg_novel 234.24983
loss 13245.569
STEP 5 ================================
prereg loss 13173.726 regularization 775.30347 reg_novel 234.5917
loss 13174.735
STEP 6 ================================
prereg loss 13114.088 regularization 774.19214 reg_novel 234.94405
loss 13115.097
STEP 7 ================================
prereg loss 13038.387 regularization 773.07684 reg_novel 235.30704
loss 13039.3955
STEP 8 ================================
prereg loss 12983.707 regularization 771.9702 reg_novel 235.67978
loss 12984.715
STEP 9 ================================
prereg loss 12978.168 regularization 770.87994 reg_novel 236.0655
loss 12979.175
STEP 10 ================================
prereg loss 13024.171 regularization 769.7997 reg_novel 236.46252
loss 13025.177
STEP 11 ================================
prereg loss 13054.383 regularization 768.73553 reg_novel 236.87283
loss 13055.389
STEP 12 ================================
prereg loss 13112.957 regularization 767.68335 reg_novel 237.29568
loss 13113.962
STEP 13 ================================
prereg loss 13093.694 regularization 766.63544 reg_novel 237.7319
loss 13094.698
STEP 14 ================================
prereg loss 13076.458 regularization 765.5968 reg_novel 238.18294
loss 13077.462
STEP 15 ================================
prereg loss 13029.243 regularization 764.5701 reg_novel 238.64828
loss 13030.246
STEP 16 ================================
prereg loss 12926.8 regularization 763.5505 reg_novel 239.12627
loss 12927.803
STEP 17 ================================
prereg loss 12804.62 regularization 762.53754 reg_novel 239.62141
loss 12805.622
STEP 18 ================================
prereg loss 12691.694 regularization 761.5249 reg_novel 240.13295
loss 12692.696
STEP 19 ================================
prereg loss 12579.944 regularization 760.5191 reg_novel 240.65814
loss 12580.945
STEP 20 ================================
prereg loss 12489.869 regularization 759.52 reg_novel 241.18236
loss 12490.87
STEP 21 ================================
prereg loss 12421.037 regularization 758.5275 reg_novel 241.7071
loss 12422.037
STEP 22 ================================
prereg loss 12381.381 regularization 757.5476 reg_novel 242.22993
loss 12382.381
STEP 23 ================================
prereg loss 12356.305 regularization 756.59717 reg_novel 242.81445
loss 12357.304
STEP 24 ================================
prereg loss 12321.845 regularization 755.6674 reg_novel 243.44682
loss 12322.844
STEP 25 ================================
prereg loss 12268.017 regularization 754.7583 reg_novel 244.11882
loss 12269.016
STEP 26 ================================
prereg loss 12248.501 regularization 753.8507 reg_novel 244.82785
loss 12249.5
STEP 27 ================================
prereg loss 12214.357 regularization 752.9448 reg_novel 245.57039
loss 12215.355
STEP 28 ================================
prereg loss 12187.672 regularization 752.0522 reg_novel 246.3428
loss 12188.67
STEP 29 ================================
prereg loss 12176.941 regularization 751.1576 reg_novel 247.14705
loss 12177.939
STEP 30 ================================
prereg loss 12160.914 regularization 750.27155 reg_novel 247.97562
loss 12161.912
STEP 31 ================================
prereg loss 12120.068 regularization 749.3919 reg_novel 248.82411
loss 12121.066
STEP 32 ================================
prereg loss 12059.084 regularization 748.52045 reg_novel 249.69864
loss 12060.082
STEP 33 ================================
prereg loss 12018.345 regularization 747.66425 reg_novel 250.5926
loss 12019.343
STEP 34 ================================
prereg loss 11960.102 regularization 746.8283 reg_novel 251.50182
loss 11961.1
STEP 35 ================================
prereg loss 11908.095 regularization 746.00885 reg_novel 252.42076
loss 11909.093
STEP 36 ================================
prereg loss 11761.077 regularization 745.19916 reg_novel 253.35327
loss 11762.076
STEP 37 ================================
prereg loss 11599.807 regularization 744.39734 reg_novel 254.30252
loss 11600.806
STEP 38 ================================
prereg loss 11436.7705 regularization 743.5884 reg_novel 255.22131
loss 11437.77
STEP 39 ================================
prereg loss 11277.591 regularization 742.7765 reg_novel 256.11612
loss 11278.59
STEP 40 ================================
prereg loss 11135.663 regularization 741.95685 reg_novel 256.99536
loss 11136.662
STEP 41 ================================
prereg loss 11026.631 regularization 741.1343 reg_novel 257.86298
loss 11027.63
STEP 42 ================================
prereg loss 10924.422 regularization 740.3079 reg_novel 258.72165
loss 10925.421
STEP 43 ================================
prereg loss 10794.564 regularization 739.49286 reg_novel 259.57068
loss 10795.563
STEP 44 ================================
prereg loss 10632.078 regularization 738.6863 reg_novel 260.4109
loss 10633.077
STEP 45 ================================
prereg loss 10486.807 regularization 737.9018 reg_novel 261.25345
loss 10487.806
STEP 46 ================================
prereg loss 10380.894 regularization 737.1313 reg_novel 262.09277
loss 10381.893
STEP 47 ================================
prereg loss 10331.541 regularization 736.3757 reg_novel 262.9433
loss 10332.54
STEP 48 ================================
prereg loss 10281.259 regularization 735.6443 reg_novel 263.8077
loss 10282.258
STEP 49 ================================
prereg loss 10232.701 regularization 734.90894 reg_novel 264.67862
loss 10233.701
STEP 50 ================================
prereg loss 10177.4795 regularization 734.1862 reg_novel 265.5597
loss 10178.4795
STEP 51 ================================
prereg loss 10105.018 regularization 733.46277 reg_novel 266.45157
loss 10106.018
STEP 52 ================================
prereg loss 10026.968 regularization 732.7496 reg_novel 267.35867
loss 10027.968
STEP 53 ================================
prereg loss 9970.604 regularization 732.0401 reg_novel 268.28143
loss 9971.604
STEP 54 ================================
prereg loss 9927.253 regularization 731.33154 reg_novel 269.21695
loss 9928.254
STEP 55 ================================
prereg loss 9908.323 regularization 730.62317 reg_novel 270.1581
loss 9909.324
STEP 56 ================================
prereg loss 9885.646 regularization 729.91364 reg_novel 271.10892
loss 9886.647
STEP 57 ================================
prereg loss 9870.139 regularization 729.22205 reg_novel 272.06088
loss 9871.14
STEP 58 ================================
prereg loss 9861.311 regularization 728.5362 reg_novel 272.99554
loss 9862.3125
STEP 59 ================================
prereg loss 9831.27 regularization 727.8551 reg_novel 273.92145
loss 9832.271
STEP 60 ================================
prereg loss 9796.14 regularization 727.17847 reg_novel 274.8127
loss 9797.142
STEP 61 ================================
prereg loss 9762.821 regularization 726.5079 reg_novel 275.67267
loss 9763.823
STEP 62 ================================
prereg loss 9707.229 regularization 725.83453 reg_novel 276.50732
loss 9708.23
STEP 63 ================================
prereg loss 9643.538 regularization 725.16785 reg_novel 277.3155
loss 9644.541
STEP 64 ================================
prereg loss 9537.184 regularization 724.5035 reg_novel 278.10867
loss 9538.187
STEP 65 ================================
prereg loss 9391.564 regularization 723.83826 reg_novel 278.89252
loss 9392.567
STEP 66 ================================
prereg loss 9289.467 regularization 723.18823 reg_novel 279.6834
loss 9290.47
STEP 67 ================================
prereg loss 9202.334 regularization 722.5298 reg_novel 280.49158
loss 9203.337
STEP 68 ================================
prereg loss 9143.705 regularization 721.8768 reg_novel 281.3276
loss 9144.708
STEP 69 ================================
prereg loss 9115.544 regularization 721.19727 reg_novel 281.9586
loss 9116.547
STEP 70 ================================
prereg loss 9095.48 regularization 720.5011 reg_novel 282.408
loss 9096.483
STEP 71 ================================
prereg loss 9074.672 regularization 719.7837 reg_novel 282.70474
loss 9075.675
STEP 72 ================================
prereg loss 9050.133 regularization 719.0612 reg_novel 282.86624
loss 9051.135
STEP 73 ================================
prereg loss 9014.799 regularization 718.3212 reg_novel 282.9133
loss 9015.8
STEP 74 ================================
prereg loss 8975.285 regularization 717.58093 reg_novel 282.87073
loss 8976.285
STEP 75 ================================
prereg loss 8947.655 regularization 716.8336 reg_novel 282.76944
loss 8948.655
STEP 76 ================================
prereg loss 8958.517 regularization 716.09326 reg_novel 282.61578
loss 8959.516
STEP 77 ================================
prereg loss 8979.195 regularization 715.3471 reg_novel 282.4187
loss 8980.193
STEP 78 ================================
prereg loss 9012.274 regularization 714.60565 reg_novel 282.1861
loss 9013.271
STEP 79 ================================
prereg loss 9039.704 regularization 713.8693 reg_novel 281.9225
loss 9040.7
STEP 80 ================================
prereg loss 9004.175 regularization 713.1486 reg_novel 281.63364
loss 9005.17
STEP 81 ================================
prereg loss 8982.971 regularization 712.4289 reg_novel 281.32513
loss 8983.965
STEP 82 ================================
prereg loss 8952.322 regularization 711.70624 reg_novel 281.0022
loss 8953.315
STEP 83 ================================
prereg loss 8932.948 regularization 710.9957 reg_novel 280.67007
loss 8933.939
STEP 84 ================================
prereg loss 8914.186 regularization 710.2886 reg_novel 280.32645
loss 8915.176
STEP 85 ================================
prereg loss 8893.3125 regularization 709.5883 reg_novel 279.97998
loss 8894.302
STEP 86 ================================
prereg loss 8899.916 regularization 708.8896 reg_novel 279.63416
loss 8900.904
STEP 87 ================================
prereg loss 8921.238 regularization 708.1907 reg_novel 279.29694
loss 8922.226
STEP 88 ================================
prereg loss 8939.171 regularization 707.50775 reg_novel 278.95724
loss 8940.157
STEP 89 ================================
prereg loss 8853.246 regularization 706.82983 reg_novel 278.62344
loss 8854.231
STEP 90 ================================
prereg loss 8733.391 regularization 706.16113 reg_novel 278.29968
loss 8734.375
STEP 91 ================================
prereg loss 8628.3125 regularization 705.49634 reg_novel 277.9886
loss 8629.296
STEP 92 ================================
prereg loss 8593.225 regularization 704.83356 reg_novel 277.68808
loss 8594.207
STEP 93 ================================
prereg loss 8568.359 regularization 704.18823 reg_novel 277.4506
loss 8569.341
STEP 94 ================================
prereg loss 8547.454 regularization 703.5596 reg_novel 277.26053
loss 8548.435
STEP 95 ================================
prereg loss 8498.697 regularization 702.959 reg_novel 277.11597
loss 8499.678
STEP 96 ================================
prereg loss 8459.714 regularization 702.3751 reg_novel 277.00894
loss 8460.693
STEP 97 ================================
prereg loss 8397.684 regularization 701.80554 reg_novel 276.94162
loss 8398.662
STEP 98 ================================
prereg loss 8352.506 regularization 701.26337 reg_novel 276.90677
loss 8353.484
STEP 99 ================================
prereg loss 8285.953 regularization 700.7226 reg_novel 276.8608
loss 8286.931
STEP 100 ================================
prereg loss 8223.256 regularization 700.1878 reg_novel 276.8276
loss 8224.232
2022-06-13T17:53:26.583

julia> steps!(200)
2022-06-13T18:47:48.006
STEP 1 ================================
prereg loss 8143.09 regularization 699.6676 reg_novel 276.8082
loss 8144.0664
STEP 2 ================================
prereg loss 8082.842 regularization 699.15356 reg_novel 276.8081
loss 8083.818
STEP 3 ================================
prereg loss 8018.0117 regularization 698.6377 reg_novel 276.76697
loss 8018.9873
STEP 4 ================================
prereg loss 7947.926 regularization 698.123 reg_novel 276.6909
loss 7948.9004
STEP 5 ================================
prereg loss 7888.335 regularization 697.6044 reg_novel 276.58243
loss 7889.309
STEP 6 ================================
prereg loss 7845.958 regularization 697.08746 reg_novel 276.4457
loss 7846.9316
STEP 7 ================================
prereg loss 7818.646 regularization 696.57635 reg_novel 276.28558
loss 7819.6187
STEP 8 ================================
prereg loss 7796.437 regularization 696.0662 reg_novel 276.10483
loss 7797.409
STEP 9 ================================
prereg loss 7784.03 regularization 695.56104 reg_novel 275.90866
loss 7785.0015
STEP 10 ================================
prereg loss 7793.8643 regularization 695.0669 reg_novel 275.6977
loss 7794.835
STEP 11 ================================
prereg loss 7811.7295 regularization 694.5659 reg_novel 275.4727
loss 7812.6997
STEP 12 ================================
prereg loss 7793.2173 regularization 694.0721 reg_novel 275.2321
loss 7794.1865
STEP 13 ================================
prereg loss 7761.615 regularization 693.5845 reg_novel 274.982
loss 7762.584
STEP 14 ================================
prereg loss 7752.6694 regularization 693.1002 reg_novel 274.72568
loss 7753.637
STEP 15 ================================
prereg loss 7750.698 regularization 692.617 reg_novel 274.4701
loss 7751.6655
STEP 16 ================================
prereg loss 7747.03 regularization 692.1464 reg_novel 274.21063
loss 7747.996
STEP 17 ================================
prereg loss 7750.0986 regularization 691.6815 reg_novel 273.95197
loss 7751.0645
STEP 18 ================================
prereg loss 7790.7734 regularization 691.2265 reg_novel 273.69043
loss 7791.7383
STEP 19 ================================
prereg loss 7796.214 regularization 690.7856 reg_novel 273.4237
loss 7797.178
STEP 20 ================================
prereg loss 7801.2935 regularization 690.3575 reg_novel 273.1505
loss 7802.257
STEP 21 ================================
prereg loss 7760.047 regularization 689.9372 reg_novel 272.88193
loss 7761.01
STEP 22 ================================
prereg loss 7722.241 regularization 689.52295 reg_novel 272.62625
loss 7723.203
STEP 23 ================================
prereg loss 7702.184 regularization 689.1162 reg_novel 272.38583
loss 7703.1455
STEP 24 ================================
prereg loss 7692.6396 regularization 688.7143 reg_novel 272.16196
loss 7693.6006
STEP 25 ================================
prereg loss 7655.147 regularization 688.32446 reg_novel 271.9599
loss 7656.1074
STEP 26 ================================
prereg loss 7628.742 regularization 687.9431 reg_novel 271.77698
loss 7629.702
STEP 27 ================================
prereg loss 7557.4634 regularization 687.5731 reg_novel 271.6093
loss 7558.4224
STEP 28 ================================
prereg loss 7489.1665 regularization 687.2065 reg_novel 271.46268
loss 7490.125
STEP 29 ================================
prereg loss 7434.517 regularization 686.84357 reg_novel 271.34372
loss 7435.475
STEP 30 ================================
prereg loss 7375.215 regularization 686.4907 reg_novel 271.2473
loss 7376.1724
STEP 31 ================================
prereg loss 7331.997 regularization 686.1573 reg_novel 271.17575
loss 7332.9546
STEP 32 ================================
prereg loss 7295.8247 regularization 685.8476 reg_novel 271.1307
loss 7296.7817
STEP 33 ================================
prereg loss 7267.056 regularization 685.53015 reg_novel 271.14438
loss 7268.0127
STEP 34 ================================
prereg loss 7198.7417 regularization 685.2253 reg_novel 271.21072
loss 7199.698
STEP 35 ================================
prereg loss 7143.3237 regularization 684.925 reg_novel 271.33047
loss 7144.28
STEP 36 ================================
prereg loss 7097.462 regularization 684.63696 reg_novel 271.49786
loss 7098.418
STEP 37 ================================
prereg loss 7078.598 regularization 684.3552 reg_novel 271.7082
loss 7079.554
STEP 38 ================================
prereg loss 7058.6753 regularization 684.0867 reg_novel 271.97202
loss 7059.6313
STEP 39 ================================
prereg loss 7042.2305 regularization 683.8112 reg_novel 272.2145
loss 7043.1865
STEP 40 ================================
prereg loss 7017.0947 regularization 683.53326 reg_novel 272.43018
loss 7018.051
STEP 41 ================================
prereg loss 6983.5723 regularization 683.267 reg_novel 272.6265
loss 6984.5283
STEP 42 ================================
prereg loss 6928.793 regularization 683.00073 reg_novel 272.8094
loss 6929.7485
STEP 43 ================================
prereg loss 6884.29 regularization 682.74133 reg_novel 272.98218
loss 6885.2456
STEP 44 ================================
prereg loss 6836.0645 regularization 682.49396 reg_novel 273.14227
loss 6837.02
STEP 45 ================================
prereg loss 6774.874 regularization 682.2487 reg_novel 273.29913
loss 6775.8296
STEP 46 ================================
prereg loss 6710.087 regularization 682.0067 reg_novel 273.45898
loss 6711.0425
STEP 47 ================================
prereg loss 6635.4414 regularization 681.7764 reg_novel 273.62753
loss 6636.397
STEP 48 ================================
prereg loss 6557.3604 regularization 681.5535 reg_novel 273.8116
loss 6558.316
STEP 49 ================================
prereg loss 6488.9106 regularization 681.3427 reg_novel 274.0016
loss 6489.866
STEP 50 ================================
prereg loss 6421.437 regularization 681.143 reg_novel 274.2042
loss 6422.3926
STEP 51 ================================
prereg loss 6368.622 regularization 680.9628 reg_novel 274.42456
loss 6369.5776
STEP 52 ================================
prereg loss 6316.8228 regularization 680.7883 reg_novel 274.6637
loss 6317.7783
STEP 53 ================================
prereg loss 6258.536 regularization 680.6143 reg_novel 274.9233
loss 6259.4917
STEP 54 ================================
prereg loss 6198.533 regularization 680.4537 reg_novel 275.2112
loss 6199.489
STEP 55 ================================
prereg loss 6138.5986 regularization 680.30457 reg_novel 275.52026
loss 6139.5547
STEP 56 ================================
prereg loss 6080.619 regularization 680.16003 reg_novel 275.84842
loss 6081.575
STEP 57 ================================
prereg loss 6033.2437 regularization 680.03186 reg_novel 276.1938
loss 6034.1997
STEP 58 ================================
prereg loss 5973.5493 regularization 679.91895 reg_novel 276.56723
loss 5974.506
STEP 59 ================================
prereg loss 5892.932 regularization 679.82135 reg_novel 276.9698
loss 5893.889
STEP 60 ================================
prereg loss 5808.2676 regularization 679.72815 reg_novel 277.39047
loss 5809.2246
STEP 61 ================================
prereg loss 5725.518 regularization 679.6437 reg_novel 277.82953
loss 5726.4756
STEP 62 ================================
prereg loss 5672.673 regularization 679.5588 reg_novel 278.26724
loss 5673.631
STEP 63 ================================
prereg loss 5635.075 regularization 679.4852 reg_novel 278.70822
loss 5636.033
STEP 64 ================================
prereg loss 5585.0273 regularization 679.4157 reg_novel 279.15332
loss 5585.986
STEP 65 ================================
prereg loss 5520.889 regularization 679.34546 reg_novel 279.59766
loss 5521.848
STEP 66 ================================
prereg loss 5438.4243 regularization 679.27545 reg_novel 280.00854
loss 5439.384
STEP 67 ================================
prereg loss 5339.528 regularization 679.19684 reg_novel 280.39532
loss 5340.4873
STEP 68 ================================
prereg loss 5199.1353 regularization 679.1186 reg_novel 280.76282
loss 5200.095
STEP 69 ================================
prereg loss 5066.705 regularization 679.03766 reg_novel 281.11804
loss 5067.665
STEP 70 ================================
prereg loss 4945.1074 regularization 678.9557 reg_novel 281.4596
loss 4946.068
STEP 71 ================================
prereg loss 4850.372 regularization 678.86584 reg_novel 281.79373
loss 4851.3325
STEP 72 ================================
prereg loss 4773.308 regularization 678.77716 reg_novel 282.10754
loss 4774.269
STEP 73 ================================
prereg loss 4698.4717 regularization 678.6779 reg_novel 282.39435
loss 4699.4326
STEP 74 ================================
prereg loss 4625.9937 regularization 678.565 reg_novel 282.65176
loss 4626.955
STEP 75 ================================
prereg loss 4554.571 regularization 678.4587 reg_novel 282.91925
loss 4555.532
STEP 76 ================================
prereg loss 4484.7417 regularization 678.361 reg_novel 283.1986
loss 4485.703
STEP 77 ================================
prereg loss 4413.1543 regularization 678.26996 reg_novel 283.48276
loss 4414.116
STEP 78 ================================
prereg loss 4347.447 regularization 678.18774 reg_novel 283.78152
loss 4348.4087
STEP 79 ================================
prereg loss 4294.0903 regularization 678.105 reg_novel 284.0939
loss 4295.0527
STEP 80 ================================
prereg loss 4250.5176 regularization 678.01855 reg_novel 284.4212
loss 4251.48
STEP 81 ================================
prereg loss 4178.8545 regularization 677.93756 reg_novel 284.75607
loss 4179.8174
STEP 82 ================================
prereg loss 4110.1484 regularization 677.8702 reg_novel 285.1019
loss 4111.1113
STEP 83 ================================
prereg loss 4022.62 regularization 677.8033 reg_novel 285.45004
loss 4023.5833
STEP 84 ================================
prereg loss 3958.528 regularization 677.7243 reg_novel 285.79987
loss 3959.4917
STEP 85 ================================
prereg loss 3899.6633 regularization 677.6414 reg_novel 286.1665
loss 3900.6272
STEP 86 ================================
prereg loss 3883.2415 regularization 677.56116 reg_novel 286.54752
loss 3884.2056
STEP 87 ================================
prereg loss 3894.7698 regularization 677.4692 reg_novel 286.93314
loss 3895.7341
STEP 88 ================================
prereg loss 3906.67 regularization 677.36633 reg_novel 287.32092
loss 3907.6345
STEP 89 ================================
prereg loss 3944.7073 regularization 677.26544 reg_novel 287.7096
loss 3945.6724
STEP 90 ================================
prereg loss 3986.7515 regularization 677.1649 reg_novel 288.1021
loss 3987.7168
STEP 91 ================================
prereg loss 4076.7256 regularization 677.06354 reg_novel 288.49585
loss 4077.6912
STEP 92 ================================
prereg loss 4185.725 regularization 676.9622 reg_novel 288.88562
loss 4186.691
STEP 93 ================================
prereg loss 4188.5034 regularization 676.85486 reg_novel 289.26828
loss 4189.4697
STEP 94 ================================
prereg loss 4288.799 regularization 676.74945 reg_novel 289.5763
loss 4289.765
STEP 95 ================================
prereg loss 4375.3174 regularization 676.6394 reg_novel 289.81168
loss 4376.2837
STEP 96 ================================
prereg loss 4403.1484 regularization 676.5213 reg_novel 289.97986
loss 4404.1147
STEP 97 ================================
prereg loss 4269.054 regularization 676.4011 reg_novel 290.08423
loss 4270.0205
STEP 98 ================================
prereg loss 4113.541 regularization 676.2798 reg_novel 290.132
loss 4114.5073
STEP 99 ================================
prereg loss 3953.8071 regularization 676.1536 reg_novel 290.12567
loss 3954.7734
STEP 100 ================================
prereg loss 3804.433 regularization 676.0223 reg_novel 290.07245
loss 3805.3992
STEP 101 ================================
prereg loss 3656.6694 regularization 675.89136 reg_novel 289.97647
loss 3657.6353
STEP 102 ================================
prereg loss 3553.2551 regularization 675.7614 reg_novel 289.84793
loss 3554.2207
STEP 103 ================================
prereg loss 3440.6772 regularization 675.63727 reg_novel 289.68384
loss 3441.6426
STEP 104 ================================
prereg loss 3338.8972 regularization 675.5095 reg_novel 289.48798
loss 3339.8623
STEP 105 ================================
prereg loss 3221.8408 regularization 675.38385 reg_novel 289.26978
loss 3222.8054
STEP 106 ================================
prereg loss 3134.8376 regularization 675.2658 reg_novel 289.01733
loss 3135.802
STEP 107 ================================
prereg loss 3046.7783 regularization 675.1485 reg_novel 288.73083
loss 3047.7422
STEP 108 ================================
prereg loss 2965.1714 regularization 675.03754 reg_novel 288.42114
loss 2966.1348
STEP 109 ================================
prereg loss 2899.9385 regularization 674.9208 reg_novel 288.0885
loss 2900.9014
STEP 110 ================================
prereg loss 2849.8074 regularization 674.8054 reg_novel 287.73547
loss 2850.77
STEP 111 ================================
prereg loss 2800.817 regularization 674.69824 reg_novel 287.36957
loss 2801.779
STEP 112 ================================
prereg loss 2773.857 regularization 674.5978 reg_novel 286.9894
loss 2774.8186
STEP 113 ================================
prereg loss 2754.9824 regularization 674.4975 reg_novel 286.59802
loss 2755.9436
STEP 114 ================================
prereg loss 2743.5579 regularization 674.4086 reg_novel 286.20224
loss 2744.5186
STEP 115 ================================
prereg loss 2734.8628 regularization 674.32574 reg_novel 285.80032
loss 2735.823
STEP 116 ================================
prereg loss 2725.9192 regularization 674.2552 reg_novel 285.42023
loss 2726.879
STEP 117 ================================
prereg loss 2713.8062 regularization 674.188 reg_novel 285.05844
loss 2714.7654
STEP 118 ================================
prereg loss 2701.0205 regularization 674.1307 reg_novel 284.7093
loss 2701.9792
STEP 119 ================================
prereg loss 2692.5457 regularization 674.0793 reg_novel 284.3636
loss 2693.5042
STEP 120 ================================
prereg loss 2698.4487 regularization 674.032 reg_novel 284.02066
loss 2699.4067
STEP 121 ================================
prereg loss 2718.38 regularization 673.9814 reg_novel 283.68115
loss 2719.3376
STEP 122 ================================
prereg loss 2738.4487 regularization 673.9326 reg_novel 283.34235
loss 2739.406
STEP 123 ================================
prereg loss 2749.8657 regularization 673.88367 reg_novel 283.00348
loss 2750.8225
STEP 124 ================================
prereg loss 2763.1904 regularization 673.83636 reg_novel 282.6644
loss 2764.147
STEP 125 ================================
prereg loss 2783.79 regularization 673.7941 reg_novel 282.32422
loss 2784.746
STEP 126 ================================
prereg loss 2800.217 regularization 673.7453 reg_novel 281.98676
loss 2801.1729
STEP 127 ================================
prereg loss 2814.8179 regularization 673.69556 reg_novel 281.6457
loss 2815.7732
STEP 128 ================================
prereg loss 2829.5352 regularization 673.64813 reg_novel 281.29495
loss 2830.49
STEP 129 ================================
prereg loss 2846.9692 regularization 673.60175 reg_novel 280.93393
loss 2847.9238
STEP 130 ================================
prereg loss 2861.0334 regularization 673.5527 reg_novel 280.56696
loss 2861.9875
STEP 131 ================================
prereg loss 2884.7444 regularization 673.50476 reg_novel 280.19745
loss 2885.698
STEP 132 ================================
prereg loss 2901.1975 regularization 673.4537 reg_novel 279.82257
loss 2902.151
STEP 133 ================================
prereg loss 2911.435 regularization 673.40625 reg_novel 279.44092
loss 2912.388
STEP 134 ================================
prereg loss 2922.4858 regularization 673.3597 reg_novel 279.05408
loss 2923.4382
STEP 135 ================================
prereg loss 2928.9158 regularization 673.3149 reg_novel 278.66385
loss 2929.8677
STEP 136 ================================
prereg loss 2928.3416 regularization 673.26874 reg_novel 278.27505
loss 2929.2932
STEP 137 ================================
prereg loss 2943.4895 regularization 673.22076 reg_novel 277.8797
loss 2944.4407
STEP 138 ================================
prereg loss 2956.2544 regularization 673.1689 reg_novel 277.4815
loss 2957.205
STEP 139 ================================
prereg loss 2975.9336 regularization 673.13226 reg_novel 277.0792
loss 2976.8838
STEP 140 ================================
prereg loss 2996.6494 regularization 673.0927 reg_novel 276.6749
loss 2997.599
STEP 141 ================================
prereg loss 3014.4937 regularization 673.0563 reg_novel 276.26572
loss 3015.4429
STEP 142 ================================
prereg loss 3026.3777 regularization 673.0208 reg_novel 275.85455
loss 3027.3267
STEP 143 ================================
prereg loss 3036.3418 regularization 672.97943 reg_novel 275.43973
loss 3037.2903
STEP 144 ================================
prereg loss 3043.7742 regularization 672.93756 reg_novel 275.02432
loss 3044.7222
STEP 145 ================================
prereg loss 3061.4966 regularization 672.89215 reg_novel 274.6056
loss 3062.444
STEP 146 ================================
prereg loss 3075.8704 regularization 672.8452 reg_novel 274.1849
loss 3076.8174
STEP 147 ================================
prereg loss 3092.325 regularization 672.7948 reg_novel 273.7613
loss 3093.2715
STEP 148 ================================
prereg loss 3107.356 regularization 672.73895 reg_novel 273.33173
loss 3108.302
STEP 149 ================================
prereg loss 3121.8938 regularization 672.6918 reg_novel 272.89398
loss 3122.8394
STEP 150 ================================
prereg loss 3135.3962 regularization 672.6435 reg_novel 272.4459
loss 3136.3413
STEP 151 ================================
prereg loss 3151.373 regularization 672.5981 reg_novel 271.99664
loss 3152.3176
STEP 152 ================================
prereg loss 3167.8755 regularization 672.54395 reg_novel 271.54523
loss 3168.8196
STEP 153 ================================
prereg loss 3182.749 regularization 672.4893 reg_novel 271.09152
loss 3183.6926
STEP 154 ================================
prereg loss 3193.4187 regularization 672.4351 reg_novel 270.63596
loss 3194.3618
STEP 155 ================================
prereg loss 3198.122 regularization 672.3834 reg_novel 270.17703
loss 3199.0647
STEP 156 ================================
prereg loss 3196.6685 regularization 672.3322 reg_novel 269.71652
loss 3197.6106
STEP 157 ================================
prereg loss 3206.7021 regularization 672.28516 reg_novel 269.25064
loss 3207.6438
STEP 158 ================================
prereg loss 3215.0464 regularization 672.23834 reg_novel 268.78366
loss 3215.9873
STEP 159 ================================
prereg loss 3218.852 regularization 672.1903 reg_novel 268.31097
loss 3219.7925
STEP 160 ================================
prereg loss 3205.942 regularization 672.14813 reg_novel 267.8337
loss 3206.8818
STEP 161 ================================
prereg loss 3191.7742 regularization 672.1061 reg_novel 267.3949
loss 3192.7136
STEP 162 ================================
prereg loss 3172.7998 regularization 672.084 reg_novel 266.94766
loss 3173.7388
STEP 163 ================================
prereg loss 3157.5674 regularization 672.06934 reg_novel 266.49146
loss 3158.5059
STEP 164 ================================
prereg loss 3153.3672 regularization 672.0586 reg_novel 266.02814
loss 3154.3052
STEP 165 ================================
prereg loss 3142.3862 regularization 672.04193 reg_novel 265.55933
loss 3143.3237
STEP 166 ================================
prereg loss 3142.64 regularization 672.0311 reg_novel 265.08572
loss 3143.577
STEP 167 ================================
prereg loss 3131.6846 regularization 672.02 reg_novel 264.611
loss 3132.621
STEP 168 ================================
prereg loss 3126.7708 regularization 672.015 reg_novel 264.13028
loss 3127.7068
STEP 169 ================================
prereg loss 3109.5312 regularization 672.0095 reg_novel 263.64087
loss 3110.4668
STEP 170 ================================
prereg loss 3099.9365 regularization 671.9676 reg_novel 263.07733
loss 3100.8716
STEP 171 ================================
prereg loss 3059.7695 regularization 671.9257 reg_novel 262.5186
loss 3060.7039
STEP 172 ================================
prereg loss 3026.9216 regularization 671.8929 reg_novel 261.95758
loss 3027.8555
STEP 173 ================================
prereg loss 2999.116 regularization 671.8625 reg_novel 261.39786
loss 3000.0493
STEP 174 ================================
prereg loss 2977.4028 regularization 671.84033 reg_novel 260.84576
loss 2978.3354
STEP 175 ================================
prereg loss 2955.4019 regularization 671.816 reg_novel 260.29913
loss 2956.334
STEP 176 ================================
prereg loss 2934.497 regularization 671.79626 reg_novel 259.75476
loss 2935.4287
STEP 177 ================================
prereg loss 2912.0332 regularization 671.7786 reg_novel 259.21747
loss 2912.964
STEP 178 ================================
prereg loss 2889.1604 regularization 671.762 reg_novel 258.69244
loss 2890.0908
STEP 179 ================================
prereg loss 2876.8362 regularization 671.75055 reg_novel 258.17377
loss 2877.766
STEP 180 ================================
prereg loss 2856.2512 regularization 671.74744 reg_novel 257.64508
loss 2857.1807
STEP 181 ================================
prereg loss 2840.3472 regularization 671.7471 reg_novel 257.12195
loss 2841.2761
STEP 182 ================================
prereg loss 2825.7234 regularization 671.7516 reg_novel 256.6095
loss 2826.6519
STEP 183 ================================
prereg loss 2806.4448 regularization 671.75507 reg_novel 256.1045
loss 2807.3728
STEP 184 ================================
prereg loss 2794.901 regularization 671.7614 reg_novel 255.58353
loss 2795.8281
STEP 185 ================================
prereg loss 2787.3862 regularization 671.7654 reg_novel 255.0847
loss 2788.313
STEP 186 ================================
prereg loss 2772.9888 regularization 671.7722 reg_novel 254.60204
loss 2773.915
STEP 187 ================================
prereg loss 2764.0137 regularization 671.7753 reg_novel 254.12624
loss 2764.9395
STEP 188 ================================
prereg loss 2757.139 regularization 671.78204 reg_novel 253.65823
loss 2758.0645
STEP 189 ================================
prereg loss 2746.5422 regularization 671.78723 reg_novel 253.19672
loss 2747.4673
STEP 190 ================================
prereg loss 2743.8088 regularization 671.7962 reg_novel 252.73775
loss 2744.7334
STEP 191 ================================
prereg loss 2739.8994 regularization 671.8109 reg_novel 252.28261
loss 2740.8235
STEP 192 ================================
prereg loss 2742.8486 regularization 671.82965 reg_novel 251.82896
loss 2743.7722
STEP 193 ================================
prereg loss 2742.2114 regularization 671.85254 reg_novel 251.33536
loss 2743.1345
STEP 194 ================================
prereg loss 2740.3254 regularization 671.8766 reg_novel 250.85449
loss 2741.2483
STEP 195 ================================
prereg loss 2748.5742 regularization 671.9012 reg_novel 250.3955
loss 2749.4966
STEP 196 ================================
prereg loss 2753.933 regularization 671.9305 reg_novel 249.95303
loss 2754.855
STEP 197 ================================
prereg loss 2756.6055 regularization 671.95245 reg_novel 249.52725
loss 2757.5269
STEP 198 ================================
prereg loss 2760.663 regularization 671.9803 reg_novel 249.11768
loss 2761.5842
STEP 199 ================================
prereg loss 2761.8096 regularization 672.0122 reg_novel 248.74597
loss 2762.7302
STEP 200 ================================
prereg loss 2755.901 regularization 672.0441 reg_novel 248.37737
loss 2756.8213
2022-06-13T21:38:57.434

julia> steps!(1000)
2022-06-13T22:10:03.221
STEP 1 ================================
prereg loss 2751.2861 regularization 672.082 reg_novel 248.00783
loss 2752.2063
STEP 2 ================================
prereg loss 2744.609 regularization 672.1205 reg_novel 247.6426
loss 2745.5286
STEP 3 ================================
prereg loss 2742.8516 regularization 672.16034 reg_novel 247.27618
loss 2743.771
STEP 4 ================================
prereg loss 2744.0137 regularization 672.2003 reg_novel 246.9038
loss 2744.9329
STEP 5 ================================
prereg loss 2754.022 regularization 672.24493 reg_novel 246.52002
loss 2754.9407
STEP 6 ================================
prereg loss 2765.213 regularization 672.30524 reg_novel 246.14052
loss 2766.1313
STEP 7 ================================
prereg loss 2784.4617 regularization 672.3827 reg_novel 245.76617
loss 2785.38
STEP 8 ================================
prereg loss 2806.1372 regularization 672.4726 reg_novel 245.39613
loss 2807.0552
STEP 9 ================================
prereg loss 2817.845 regularization 672.57837 reg_novel 245.02942
loss 2818.7627
STEP 10 ================================
prereg loss 2815.1833 regularization 672.693 reg_novel 244.66988
loss 2816.1008
STEP 11 ================================
prereg loss 2810.7808 regularization 672.8118 reg_novel 244.3229
loss 2811.698
STEP 12 ================================
prereg loss 2804.3914 regularization 672.9345 reg_novel 243.98366
loss 2805.3083
STEP 13 ================================
prereg loss 2798.5122 regularization 673.06976 reg_novel 243.65617
loss 2799.429
STEP 14 ================================
prereg loss 2790.2004 regularization 673.2046 reg_novel 243.33972
loss 2791.117
STEP 15 ================================
prereg loss 2794.2324 regularization 673.3041 reg_novel 243.02806
loss 2795.1487
STEP 16 ================================
prereg loss 2807.2478 regularization 673.3758 reg_novel 242.71124
loss 2808.1638
STEP 17 ================================
prereg loss 2824.1284 regularization 673.4347 reg_novel 242.39429
loss 2825.0442
STEP 18 ================================
prereg loss 2840.275 regularization 673.4835 reg_novel 242.07698
loss 2841.1904
STEP 19 ================================
prereg loss 2835.193 regularization 673.52136 reg_novel 241.76091
loss 2836.1084
STEP 20 ================================
prereg loss 2828.5474 regularization 673.55505 reg_novel 241.45564
loss 2829.4624
STEP 21 ================================
prereg loss 2815.5413 regularization 673.5829 reg_novel 241.16263
loss 2816.456
STEP 22 ================================
prereg loss 2793.121 regularization 673.61096 reg_novel 240.88182
loss 2794.0356
STEP 23 ================================
prereg loss 2772.404 regularization 673.6393 reg_novel 240.61815
loss 2773.3184
STEP 24 ================================
prereg loss 2752.457 regularization 673.6693 reg_novel 240.37341
loss 2753.371
STEP 25 ================================
prereg loss 2741.959 regularization 673.70746 reg_novel 240.14359
loss 2742.8728
STEP 26 ================================
prereg loss 2739.1846 regularization 673.7557 reg_novel 239.9287
loss 2740.0981
STEP 27 ================================
prereg loss 2730.1733 regularization 673.85596 reg_novel 239.73683
loss 2731.087
STEP 28 ================================
prereg loss 2736.637 regularization 674.00964 reg_novel 239.57094
loss 2737.5505
STEP 29 ================================
prereg loss 2757.2957 regularization 674.1826 reg_novel 239.41122
loss 2758.2092
STEP 30 ================================
prereg loss 2781.0007 regularization 674.3607 reg_novel 239.25249
loss 2781.9143
STEP 31 ================================
prereg loss 2807.6543 regularization 674.5317 reg_novel 239.09186
loss 2808.5679
STEP 32 ================================
prereg loss 2839.7903 regularization 674.7064 reg_novel 238.92822
loss 2840.7039
STEP 33 ================================
prereg loss 2872.6448 regularization 674.8766 reg_novel 238.75812
loss 2873.5583
STEP 34 ================================
prereg loss 2895.1353 regularization 675.0351 reg_novel 238.55377
loss 2896.0488
STEP 35 ================================
prereg loss 2914.2222 regularization 675.18085 reg_novel 238.31659
loss 2915.1357
STEP 36 ================================
prereg loss 2920.0107 regularization 675.31586 reg_novel 238.05444
loss 2920.924
STEP 37 ================================
prereg loss 2918.947 regularization 675.4385 reg_novel 237.77176
loss 2919.8604
STEP 38 ================================
prereg loss 2910.5757 regularization 675.554 reg_novel 237.47723
loss 2911.4888
STEP 39 ================================
prereg loss 2904.0337 regularization 675.66724 reg_novel 237.1901
loss 2904.9465
STEP 40 ================================
prereg loss 2895.3745 regularization 675.78186 reg_novel 236.93417
loss 2896.287
STEP 41 ================================
prereg loss 2890.5032 regularization 675.902 reg_novel 236.69757
loss 2891.4158
STEP 42 ================================
prereg loss 2901.5007 regularization 676.02527 reg_novel 236.46407
loss 2902.4133
STEP 43 ================================
prereg loss 2920.467 regularization 676.148 reg_novel 236.23137
loss 2921.3794
STEP 44 ================================
prereg loss 2944.068 regularization 676.2802 reg_novel 235.99898
loss 2944.9805
STEP 45 ================================
prereg loss 2956.64 regularization 676.4232 reg_novel 235.76234
loss 2957.552
STEP 46 ================================
prereg loss 2971.9062 regularization 676.57184 reg_novel 235.5236
loss 2972.8184
STEP 47 ================================
prereg loss 2983.338 regularization 676.7237 reg_novel 235.28691
loss 2984.25
STEP 48 ================================
prereg loss 3004.895 regularization 676.8834 reg_novel 235.06279
loss 3005.807
STEP 49 ================================
prereg loss 3026.2424 regularization 677.05164 reg_novel 234.85146
loss 3027.1543
STEP 50 ================================
prereg loss 3041.3035 regularization 677.2299 reg_novel 234.64867
loss 3042.2153
STEP 51 ================================
prereg loss 3054.897 regularization 677.41724 reg_novel 234.45714
loss 3055.8088
STEP 52 ================================
prereg loss 3068.0613 regularization 677.6095 reg_novel 234.27728
loss 3068.9731
STEP 53 ================================
prereg loss 3089.1006 regularization 677.8009 reg_novel 234.10045
loss 3090.0125
STEP 54 ================================
prereg loss 3108.2761 regularization 677.99005 reg_novel 233.94836
loss 3109.188
STEP 55 ================================
prereg loss 3124.5962 regularization 678.17804 reg_novel 233.79301
loss 3125.508
STEP 56 ================================
prereg loss 3137.2808 regularization 678.36633 reg_novel 233.63683
loss 3138.1929
STEP 57 ================================
prereg loss 3150.1953 regularization 678.551 reg_novel 233.45766
loss 3151.1074
STEP 58 ================================
prereg loss 3160.608 regularization 678.73444 reg_novel 233.28625
loss 3161.52
STEP 59 ================================
prereg loss 3162.8281 regularization 678.9099 reg_novel 233.09773
loss 3163.7402
STEP 60 ================================
prereg loss 3076.8774 regularization 679.09985 reg_novel 232.94519
loss 3077.7896
STEP 61 ================================
prereg loss 3053.437 regularization 679.29443 reg_novel 232.8141
loss 3054.349
STEP 62 ================================
prereg loss 3036.1729 regularization 679.49316 reg_novel 232.67839
loss 3037.085
STEP 63 ================================
prereg loss 3021.2795 regularization 679.69855 reg_novel 232.5543
loss 3022.192
STEP 64 ================================
prereg loss 3011.1514 regularization 679.90186 reg_novel 232.4352
loss 3012.0637
STEP 65 ================================
prereg loss 3001.5403 regularization 680.1063 reg_novel 232.31458
loss 3002.4526
STEP 66 ================================
prereg loss 2992.3284 regularization 680.3104 reg_novel 232.18796
loss 2993.241
STEP 67 ================================
prereg loss 2987.4966 regularization 680.5126 reg_novel 232.05838
loss 2988.4092
STEP 68 ================================
prereg loss 2985.1501 regularization 680.7163 reg_novel 231.92857
loss 2986.0627
STEP 69 ================================
prereg loss 2988.105 regularization 680.92053 reg_novel 231.79536
loss 2989.0176
STEP 70 ================================
prereg loss 2981.6274 regularization 681.1215 reg_novel 231.65941
loss 2982.5403
STEP 71 ================================
prereg loss 2975.775 regularization 681.3236 reg_novel 231.52197
loss 2976.6877
STEP 72 ================================
prereg loss 2973.5645 regularization 681.5374 reg_novel 231.38104
loss 2974.4773
STEP 73 ================================
prereg loss 2972.4626 regularization 681.7478 reg_novel 231.2369
loss 2973.3757
STEP 74 ================================
prereg loss 2972.9058 regularization 681.95807 reg_novel 231.08952
loss 2973.8188
STEP 75 ================================
prereg loss 2974.9858 regularization 682.1647 reg_novel 230.94427
loss 2975.899
STEP 76 ================================
prereg loss 2978.7444 regularization 682.36993 reg_novel 230.80399
loss 2979.6575
STEP 77 ================================
prereg loss 2984.749 regularization 682.57574 reg_novel 230.69449
loss 2985.6624
STEP 78 ================================
prereg loss 2991.0657 regularization 682.78296 reg_novel 230.60966
loss 2991.979
STEP 79 ================================
prereg loss 2997.428 regularization 682.9901 reg_novel 230.52377
loss 2998.3416
STEP 80 ================================
prereg loss 3013.419 regularization 683.19965 reg_novel 230.43471
loss 3014.3325
STEP 81 ================================
prereg loss 3025.3882 regularization 683.3997 reg_novel 230.34427
loss 3026.302
STEP 82 ================================
prereg loss 3033.2234 regularization 683.59283 reg_novel 230.24973
loss 3034.1372
STEP 83 ================================
prereg loss 3039.9204 regularization 683.7863 reg_novel 230.15385
loss 3040.8342
STEP 84 ================================
prereg loss 3043.665 regularization 683.9799 reg_novel 230.05211
loss 3044.579
STEP 85 ================================
prereg loss 3041.2134 regularization 684.16864 reg_novel 229.94795
loss 3042.1274
STEP 86 ================================
prereg loss 3042.9395 regularization 684.35425 reg_novel 229.84526
loss 3043.8538
STEP 87 ================================
prereg loss 3052.0461 regularization 684.5387 reg_novel 229.73853
loss 3052.9604
STEP 88 ================================
prereg loss 3076.1807 regularization 684.73364 reg_novel 229.63123
loss 3077.095
STEP 89 ================================
prereg loss 3090.204 regularization 684.92474 reg_novel 229.52621
loss 3091.1187
STEP 90 ================================
prereg loss 3087.629 regularization 685.1168 reg_novel 229.42047
loss 3088.5435
STEP 91 ================================
prereg loss 3072.112 regularization 685.31494 reg_novel 229.31813
loss 3073.0266
STEP 92 ================================
prereg loss 3056.434 regularization 685.51056 reg_novel 229.2217
loss 3057.3489
STEP 93 ================================
prereg loss 3043.6106 regularization 685.70465 reg_novel 229.15694
loss 3044.5254
STEP 94 ================================
prereg loss 3020.6328 regularization 685.9021 reg_novel 229.09335
loss 3021.5479
STEP 95 ================================
prereg loss 3002.2124 regularization 686.10364 reg_novel 229.09659
loss 3003.1277
STEP 96 ================================
prereg loss 2998.272 regularization 686.3069 reg_novel 229.13231
loss 2999.1875
STEP 97 ================================
prereg loss 3006.5342 regularization 686.5098 reg_novel 229.18953
loss 3007.45
STEP 98 ================================
prereg loss 3023.481 regularization 686.7144 reg_novel 229.2656
loss 3024.397
STEP 99 ================================
prereg loss 3036.0571 regularization 686.9224 reg_novel 229.36089
loss 3036.9734
STEP 100 ================================
prereg loss 3052.0764 regularization 687.1384 reg_novel 229.4693
loss 3052.993
STEP 101 ================================
prereg loss 3073.966 regularization 687.3551 reg_novel 229.55379
loss 3074.883
STEP 102 ================================
prereg loss 3088.4124 regularization 687.5698 reg_novel 229.61754
loss 3089.3296
STEP 103 ================================
prereg loss 3082.9941 regularization 687.7778 reg_novel 229.65738
loss 3083.9116
STEP 104 ================================
prereg loss 3073.463 regularization 687.97485 reg_novel 229.67456
loss 3074.3806
STEP 105 ================================
prereg loss 3061.1704 regularization 688.17084 reg_novel 229.6894
loss 3062.0884
STEP 106 ================================
prereg loss 3052.9575 regularization 688.3594 reg_novel 229.68758
loss 3053.8755
STEP 107 ================================
prereg loss 3044.4185 regularization 688.54126 reg_novel 229.67278
loss 3045.3367
STEP 108 ================================
prereg loss 3026.2754 regularization 688.7115 reg_novel 229.64728
loss 3027.1938
STEP 109 ================================
prereg loss 3000.5537 regularization 688.8704 reg_novel 229.59422
loss 3001.4722
STEP 110 ================================
prereg loss 2963.2861 regularization 689.02386 reg_novel 229.52061
loss 2964.2046
STEP 111 ================================
prereg loss 2919.0667 regularization 689.1798 reg_novel 229.44128
loss 2919.9854
STEP 112 ================================
prereg loss 2871.2546 regularization 689.3409 reg_novel 229.36621
loss 2872.1733
STEP 113 ================================
prereg loss 2825.3813 regularization 689.5095 reg_novel 229.31699
loss 2826.3003
STEP 114 ================================
prereg loss 2789.0852 regularization 689.67236 reg_novel 229.28584
loss 2790.0042
STEP 115 ================================
prereg loss 2758.7422 regularization 689.8331 reg_novel 229.2718
loss 2759.6614
STEP 116 ================================
prereg loss 2735.6987 regularization 689.9947 reg_novel 229.27005
loss 2736.618
STEP 117 ================================
prereg loss 2717.357 regularization 690.152 reg_novel 229.28221
loss 2718.2764
STEP 118 ================================
prereg loss 2703.4653 regularization 690.29785 reg_novel 229.2739
loss 2704.385
STEP 119 ================================
prereg loss 2696.5444 regularization 690.4407 reg_novel 229.25093
loss 2697.464
STEP 120 ================================
prereg loss 2682.014 regularization 690.5742 reg_novel 229.20944
loss 2682.9336
STEP 121 ================================
prereg loss 2663.0583 regularization 690.6994 reg_novel 229.15163
loss 2663.9783
STEP 122 ================================
prereg loss 2639.0159 regularization 690.81604 reg_novel 229.07933
loss 2639.9358
STEP 123 ================================
prereg loss 2601.545 regularization 690.92584 reg_novel 228.99373
loss 2602.4648
STEP 124 ================================
prereg loss 2560.7617 regularization 691.0382 reg_novel 228.89726
loss 2561.6816
STEP 125 ================================
prereg loss 2502.1343 regularization 691.1425 reg_novel 228.79234
loss 2503.0542
STEP 126 ================================
prereg loss 2440.6692 regularization 691.2435 reg_novel 228.67905
loss 2441.589
STEP 127 ================================
prereg loss 2375.6826 regularization 691.3413 reg_novel 228.55835
loss 2376.6025
STEP 128 ================================
prereg loss 2302.9111 regularization 691.43475 reg_novel 228.43129
loss 2303.831
STEP 129 ================================
prereg loss 2232.3467 regularization 691.5246 reg_novel 228.29564
loss 2233.2666
STEP 130 ================================
prereg loss 2171.5342 regularization 691.611 reg_novel 228.15038
loss 2172.4539
STEP 131 ================================
prereg loss 2119.4668 regularization 691.6937 reg_novel 227.99594
loss 2120.3865
STEP 132 ================================
prereg loss 2073.072 regularization 691.7718 reg_novel 227.83427
loss 2073.9917
STEP 133 ================================
prereg loss 2029.59 regularization 691.8397 reg_novel 227.66782
loss 2030.5095
STEP 134 ================================
prereg loss 1985.8862 regularization 691.8956 reg_novel 227.50085
loss 1986.8057
STEP 135 ================================
prereg loss 1938.7305 regularization 691.9443 reg_novel 227.32736
loss 1939.6498
STEP 136 ================================
prereg loss 1889.0033 regularization 691.98566 reg_novel 227.15071
loss 1889.9225
STEP 137 ================================
prereg loss 1839.4093 regularization 692.0218 reg_novel 226.97113
loss 1840.3282
STEP 138 ================================
prereg loss 1789.5729 regularization 692.0523 reg_novel 226.78957
loss 1790.4917
STEP 139 ================================
prereg loss 1745.698 regularization 692.0826 reg_novel 226.6047
loss 1746.6167
STEP 140 ================================
prereg loss 1702.8892 regularization 692.1135 reg_novel 226.41496
loss 1703.8077
STEP 141 ================================
prereg loss 1663.6685 regularization 692.14166 reg_novel 226.22015
loss 1664.5868
STEP 142 ================================
prereg loss 1635.4878 regularization 692.166 reg_novel 226.02359
loss 1636.406
STEP 143 ================================
prereg loss 1614.5076 regularization 692.19977 reg_novel 225.82132
loss 1615.4255
STEP 144 ================================
prereg loss 1595.2223 regularization 692.2342 reg_novel 225.61693
loss 1596.1401
STEP 145 ================================
prereg loss 1575.8013 regularization 692.2687 reg_novel 225.41422
loss 1576.719
STEP 146 ================================
prereg loss 1560.262 regularization 692.3031 reg_novel 225.21066
loss 1561.1794
STEP 147 ================================
prereg loss 1538.2888 regularization 692.3414 reg_novel 225.00842
loss 1539.2062
STEP 148 ================================
prereg loss 1512.9337 regularization 692.3796 reg_novel 224.80803
loss 1513.851
STEP 149 ================================
prereg loss 1483.1501 regularization 692.4166 reg_novel 224.6068
loss 1484.0671
STEP 150 ================================
prereg loss 1450.5676 regularization 692.45245 reg_novel 224.40283
loss 1451.4845
STEP 151 ================================
prereg loss 1417.8711 regularization 692.4853 reg_novel 224.20079
loss 1418.7877
STEP 152 ================================
prereg loss 1386.4973 regularization 692.521 reg_novel 223.99591
loss 1387.4138
STEP 153 ================================
prereg loss 1358.2615 regularization 692.556 reg_novel 223.79216
loss 1359.1779
STEP 154 ================================
prereg loss 1331.9932 regularization 692.58954 reg_novel 223.58691
loss 1332.9093
STEP 155 ================================
prereg loss 1311.2518 regularization 692.6191 reg_novel 223.38019
loss 1312.1678
STEP 156 ================================
prereg loss 1296.824 regularization 692.64655 reg_novel 223.17397
loss 1297.7397
STEP 157 ================================
prereg loss 1284.8413 regularization 692.6758 reg_novel 222.96672
loss 1285.757
STEP 158 ================================
prereg loss 1273.1604 regularization 692.6962 reg_novel 222.76134
loss 1274.0758
STEP 159 ================================
prereg loss 1260.0349 regularization 692.7139 reg_novel 222.55725
loss 1260.9502
STEP 160 ================================
prereg loss 1248.0732 regularization 692.72644 reg_novel 222.3549
loss 1248.9883
STEP 161 ================================
prereg loss 1233.5815 regularization 692.7365 reg_novel 222.15228
loss 1234.4965
STEP 162 ================================
prereg loss 1217.3335 regularization 692.74274 reg_novel 221.95128
loss 1218.2482
STEP 163 ================================
prereg loss 1198.3539 regularization 692.74524 reg_novel 221.75015
loss 1199.2684
STEP 164 ================================
prereg loss 1178.5055 regularization 692.7472 reg_novel 221.55154
loss 1179.4198
STEP 165 ================================
prereg loss 1157.3613 regularization 692.74774 reg_novel 221.35243
loss 1158.2754
STEP 166 ================================
prereg loss 1135.9 regularization 692.75006 reg_novel 221.15265
loss 1136.814
STEP 167 ================================
prereg loss 1117.044 regularization 692.75244 reg_novel 220.95525
loss 1117.9576
STEP 168 ================================
prereg loss 1101.1854 regularization 692.7563 reg_novel 220.75946
loss 1102.099
STEP 169 ================================
prereg loss 1086.5974 regularization 692.7613 reg_novel 220.56342
loss 1087.5107
STEP 170 ================================
prereg loss 1069.8472 regularization 692.7668 reg_novel 220.37097
loss 1070.7603
STEP 171 ================================
prereg loss 1050.7332 regularization 692.77765 reg_novel 220.18243
loss 1051.6461
STEP 172 ================================
prereg loss 1029.561 regularization 692.7878 reg_novel 219.9976
loss 1030.4739
STEP 173 ================================
prereg loss 1009.62415 regularization 692.802 reg_novel 219.81822
loss 1010.53674
STEP 174 ================================
prereg loss 1011.655 regularization 692.82556 reg_novel 219.65527
loss 1012.5675
STEP 175 ================================
prereg loss 1010.3433 regularization 692.84985 reg_novel 219.49266
loss 1011.2557
STEP 176 ================================
prereg loss 995.3742 regularization 692.879 reg_novel 219.33655
loss 996.28644
STEP 177 ================================
prereg loss 979.65515 regularization 692.90576 reg_novel 219.1983
loss 980.56726
STEP 178 ================================
prereg loss 966.6449 regularization 692.9396 reg_novel 219.06424
loss 967.5569
STEP 179 ================================
prereg loss 954.3109 regularization 692.97565 reg_novel 218.93402
loss 955.22284
STEP 180 ================================
prereg loss 943.2646 regularization 693.02106 reg_novel 218.80785
loss 944.1764
STEP 181 ================================
prereg loss 932.2795 regularization 693.07043 reg_novel 218.68379
loss 933.1912
STEP 182 ================================
prereg loss 926.125 regularization 693.12805 reg_novel 218.56535
loss 927.0367
STEP 183 ================================
prereg loss 942.0223 regularization 693.19446 reg_novel 218.44398
loss 942.9339
STEP 184 ================================
prereg loss 952.8803 regularization 693.26166 reg_novel 218.32678
loss 953.7919
STEP 185 ================================
prereg loss 951.90814 regularization 693.33124 reg_novel 218.21844
loss 952.8197
STEP 186 ================================
prereg loss 938.5848 regularization 693.40765 reg_novel 218.11069
loss 939.4963
STEP 187 ================================
prereg loss 920.7507 regularization 693.4851 reg_novel 218.00179
loss 921.6622
STEP 188 ================================
prereg loss 909.48254 regularization 693.56836 reg_novel 217.88652
loss 910.394
STEP 189 ================================
prereg loss 903.2814 regularization 693.6506 reg_novel 217.76646
loss 904.1928
STEP 190 ================================
prereg loss 903.95166 regularization 693.7307 reg_novel 217.64188
loss 904.86304
STEP 191 ================================
prereg loss 890.70575 regularization 693.8058 reg_novel 217.50949
loss 891.61707
STEP 192 ================================
prereg loss 882.86206 regularization 693.87305 reg_novel 217.37358
loss 883.7733
STEP 193 ================================
prereg loss 876.66943 regularization 693.93353 reg_novel 217.24937
loss 877.5806
STEP 194 ================================
prereg loss 864.531 regularization 693.9891 reg_novel 217.11565
loss 865.44214
STEP 195 ================================
prereg loss 846.9133 regularization 694.03644 reg_novel 216.9686
loss 847.82434
STEP 196 ================================
prereg loss 824.4779 regularization 694.07715 reg_novel 216.81006
loss 825.3888
STEP 197 ================================
prereg loss 807.2979 regularization 694.1134 reg_novel 216.64397
loss 808.2087
STEP 198 ================================
prereg loss 801.6831 regularization 694.143 reg_novel 216.47179
loss 802.59375
STEP 199 ================================
prereg loss 805.8554 regularization 694.1721 reg_novel 216.29074
loss 806.76587
STEP 200 ================================
prereg loss 811.33264 regularization 694.192 reg_novel 216.10301
loss 812.2429
STEP 201 ================================
prereg loss 810.19727 regularization 694.215 reg_novel 215.91402
loss 811.1074
STEP 202 ================================
prereg loss 815.5001 regularization 694.2506 reg_novel 215.73888
loss 816.4101
STEP 203 ================================
prereg loss 805.4168 regularization 694.2912 reg_novel 215.56512
loss 806.32666
STEP 204 ================================
prereg loss 783.6955 regularization 694.33154 reg_novel 215.39285
loss 784.6052
STEP 205 ================================
prereg loss 757.20386 regularization 694.37 reg_novel 215.22372
loss 758.11346
STEP 206 ================================
prereg loss 736.794 regularization 694.41284 reg_novel 215.05592
loss 737.7035
STEP 207 ================================
prereg loss 725.568 regularization 694.4517 reg_novel 214.88945
loss 726.47736
STEP 208 ================================
prereg loss 719.7591 regularization 694.4895 reg_novel 214.72491
loss 720.66833
STEP 209 ================================
prereg loss 722.8369 regularization 694.52637 reg_novel 214.56364
loss 723.74603
STEP 210 ================================
prereg loss 716.94775 regularization 694.557 reg_novel 214.40831
loss 717.8567
STEP 211 ================================
prereg loss 711.2112 regularization 694.58496 reg_novel 214.25179
loss 712.12
STEP 212 ================================
prereg loss 704.1165 regularization 694.61224 reg_novel 214.09352
loss 705.0252
STEP 213 ================================
prereg loss 697.6199 regularization 694.6406 reg_novel 213.93195
loss 698.52844
STEP 214 ================================
prereg loss 692.62537 regularization 694.6658 reg_novel 213.76787
loss 693.5338
STEP 215 ================================
prereg loss 689.1108 regularization 694.6832 reg_novel 213.60419
loss 690.01904
STEP 216 ================================
prereg loss 684.7208 regularization 694.6965 reg_novel 213.43956
loss 685.62897
STEP 217 ================================
prereg loss 678.90027 regularization 694.70715 reg_novel 213.27695
loss 679.8082
STEP 218 ================================
prereg loss 671.41003 regularization 694.7178 reg_novel 213.11801
loss 672.3179
STEP 219 ================================
prereg loss 662.6911 regularization 694.7298 reg_novel 212.96362
loss 663.5988
STEP 220 ================================
prereg loss 654.9291 regularization 694.73926 reg_novel 212.81125
loss 655.8366
STEP 221 ================================
prereg loss 647.29535 regularization 694.7498 reg_novel 212.66402
loss 648.20276
STEP 222 ================================
prereg loss 640.05054 regularization 694.755 reg_novel 212.51761
loss 640.9578
STEP 223 ================================
prereg loss 633.6393 regularization 694.7597 reg_novel 212.36848
loss 634.5464
STEP 224 ================================
prereg loss 627.68646 regularization 694.7642 reg_novel 212.21684
loss 628.59344
STEP 225 ================================
prereg loss 623.22595 regularization 694.768 reg_novel 212.06155
loss 624.13275
STEP 226 ================================
prereg loss 619.0006 regularization 694.7721 reg_novel 211.90257
loss 619.9073
STEP 227 ================================
prereg loss 614.9563 regularization 694.7756 reg_novel 211.74234
loss 615.8628
STEP 228 ================================
prereg loss 611.5996 regularization 694.7753 reg_novel 211.58435
loss 612.506
STEP 229 ================================
prereg loss 608.4086 regularization 694.78 reg_novel 211.42738
loss 609.3148
STEP 230 ================================
prereg loss 604.7595 regularization 694.7882 reg_novel 211.2719
loss 605.6656
STEP 231 ================================
prereg loss 600.7832 regularization 694.7955 reg_novel 211.1203
loss 601.68915
STEP 232 ================================
prereg loss 596.7058 regularization 694.799 reg_novel 210.96873
loss 597.6116
STEP 233 ================================
prereg loss 592.60205 regularization 694.79895 reg_novel 210.81398
loss 593.5077
STEP 234 ================================
prereg loss 586.259 regularization 694.80084 reg_novel 210.65752
loss 587.1644
STEP 235 ================================
prereg loss 580.3412 regularization 694.7993 reg_novel 210.49997
loss 581.24646
STEP 236 ================================
prereg loss 572.04926 regularization 694.79974 reg_novel 210.35129
loss 572.9544
STEP 237 ================================
prereg loss 564.5612 regularization 694.79596 reg_novel 210.22206
loss 565.46625
STEP 238 ================================
prereg loss 557.929 regularization 694.7911 reg_novel 210.11212
loss 558.8339
STEP 239 ================================
prereg loss 552.00885 regularization 694.7834 reg_novel 210.0142
loss 552.91364
STEP 240 ================================
prereg loss 547.0208 regularization 694.7741 reg_novel 209.92868
loss 547.92554
STEP 241 ================================
prereg loss 542.7358 regularization 694.7632 reg_novel 209.85255
loss 543.6404
STEP 242 ================================
prereg loss 538.9639 regularization 694.75 reg_novel 209.78622
loss 539.86847
STEP 243 ================================
prereg loss 535.64954 regularization 694.73706 reg_novel 209.73155
loss 536.554
STEP 244 ================================
prereg loss 531.855 regularization 694.7289 reg_novel 209.68738
loss 532.7594
STEP 245 ================================
prereg loss 532.1919 regularization 694.7409 reg_novel 209.66187
loss 533.0963
STEP 246 ================================
prereg loss 535.3345 regularization 694.7515 reg_novel 209.63376
loss 536.23883
STEP 247 ================================
prereg loss 535.146 regularization 694.7579 reg_novel 209.60371
loss 536.05035
STEP 248 ================================
prereg loss 530.22644 regularization 694.7648 reg_novel 209.56885
loss 531.1308
STEP 249 ================================
prereg loss 522.00793 regularization 694.7701 reg_novel 209.52753
loss 522.91223
STEP 250 ================================
prereg loss 511.87396 regularization 694.7742 reg_novel 209.48381
loss 512.7782
STEP 251 ================================
prereg loss 503.09534 regularization 694.77637 reg_novel 209.43858
loss 503.99954
STEP 252 ================================
prereg loss 498.72968 regularization 694.7782 reg_novel 209.39029
loss 499.63385
STEP 253 ================================
prereg loss 499.73676 regularization 694.7803 reg_novel 209.34158
loss 500.64087
STEP 254 ================================
prereg loss 503.1405 regularization 694.78314 reg_novel 209.29147
loss 504.0446
STEP 255 ================================
prereg loss 505.0047 regularization 694.7878 reg_novel 209.24045
loss 505.90872
STEP 256 ================================
prereg loss 502.64743 regularization 694.794 reg_novel 209.18526
loss 503.55142
STEP 257 ================================
prereg loss 495.93716 regularization 694.801 reg_novel 209.12729
loss 496.8411
STEP 258 ================================
prereg loss 490.07623 regularization 694.81116 reg_novel 209.06549
loss 490.9801
STEP 259 ================================
prereg loss 484.60486 regularization 694.82 reg_novel 209.0034
loss 485.50867
STEP 260 ================================
prereg loss 478.54126 regularization 694.83026 reg_novel 208.9467
loss 479.44504
STEP 261 ================================
prereg loss 473.56262 regularization 694.8391 reg_novel 208.89557
loss 474.46637
STEP 262 ================================
prereg loss 470.01813 regularization 694.8552 reg_novel 208.84406
loss 470.9218
STEP 263 ================================
prereg loss 468.27106 regularization 694.8741 reg_novel 208.79161
loss 469.1747
STEP 264 ================================
prereg loss 467.3027 regularization 694.89404 reg_novel 208.74251
loss 468.20633
STEP 265 ================================
prereg loss 466.27966 regularization 694.91534 reg_novel 208.69508
loss 467.1833
STEP 266 ================================
prereg loss 467.28116 regularization 694.93604 reg_novel 208.6508
loss 468.18475
STEP 267 ================================
prereg loss 466.7864 regularization 694.9562 reg_novel 208.60875
loss 467.68997
STEP 268 ================================
prereg loss 463.83966 regularization 694.9761 reg_novel 208.5662
loss 464.7432
STEP 269 ================================
prereg loss 459.07468 regularization 694.9968 reg_novel 208.52441
loss 459.9782
STEP 270 ================================
prereg loss 454.76154 regularization 695.01196 reg_novel 208.48175
loss 455.66504
STEP 271 ================================
prereg loss 450.8017 regularization 695.0266 reg_novel 208.43666
loss 451.70517
STEP 272 ================================
prereg loss 447.75818 regularization 695.03784 reg_novel 208.38971
loss 448.66162
STEP 273 ================================
prereg loss 446.0827 regularization 695.0477 reg_novel 208.34215
loss 446.98608
STEP 274 ================================
prereg loss 445.00485 regularization 695.057 reg_novel 208.29489
loss 445.9082
STEP 275 ================================
prereg loss 445.7903 regularization 695.0631 reg_novel 208.24724
loss 446.69363
STEP 276 ================================
prereg loss 446.35648 regularization 695.07117 reg_novel 208.19637
loss 447.25974
STEP 277 ================================
prereg loss 444.28656 regularization 695.0767 reg_novel 208.14214
loss 445.1898
STEP 278 ================================
prereg loss 443.9583 regularization 695.07983 reg_novel 208.08676
loss 444.86148
STEP 279 ================================
prereg loss 444.85883 regularization 695.0807 reg_novel 208.02881
loss 445.76193
STEP 280 ================================
prereg loss 446.42032 regularization 695.08124 reg_novel 207.96552
loss 447.32336
STEP 281 ================================
prereg loss 447.5265 regularization 695.07904 reg_novel 207.89877
loss 448.42947
STEP 282 ================================
prereg loss 447.5055 regularization 695.07404 reg_novel 207.86151
loss 448.40842
STEP 283 ================================
prereg loss 446.655 regularization 695.0781 reg_novel 207.80609
loss 447.5579
STEP 284 ================================
prereg loss 444.59924 regularization 695.0787 reg_novel 207.74588
loss 445.50208
STEP 285 ================================
prereg loss 442.4769 regularization 695.0796 reg_novel 207.68024
loss 443.37967
STEP 286 ================================
prereg loss 440.25348 regularization 695.0784 reg_novel 207.6093
loss 441.15616
STEP 287 ================================
prereg loss 436.62357 regularization 695.0795 reg_novel 207.53436
loss 437.52618
STEP 288 ================================
prereg loss 438.41144 regularization 695.09875 reg_novel 207.46811
loss 439.314
STEP 289 ================================
prereg loss 440.5288 regularization 695.1186 reg_novel 207.398
loss 441.43134
STEP 290 ================================
prereg loss 442.1577 regularization 695.13293 reg_novel 207.32454
loss 443.06018
STEP 291 ================================
prereg loss 441.1981 regularization 695.1476 reg_novel 207.24673
loss 442.1005
STEP 292 ================================
prereg loss 438.02435 regularization 695.16125 reg_novel 207.16896
loss 438.9267
STEP 293 ================================
prereg loss 433.32797 regularization 695.1741 reg_novel 207.09131
loss 434.23022
STEP 294 ================================
prereg loss 428.8322 regularization 695.1822 reg_novel 207.01193
loss 429.7344
STEP 295 ================================
prereg loss 425.34332 regularization 695.19214 reg_novel 206.92937
loss 426.24545
STEP 296 ================================
prereg loss 423.61237 regularization 695.2014 reg_novel 206.84805
loss 424.5144
STEP 297 ================================
prereg loss 422.89563 regularization 695.2097 reg_novel 206.7662
loss 423.7976
STEP 298 ================================
prereg loss 422.70685 regularization 695.21655 reg_novel 206.6869
loss 423.60876
STEP 299 ================================
prereg loss 422.34235 regularization 695.2248 reg_novel 206.60794
loss 423.24417
STEP 300 ================================
prereg loss 421.21674 regularization 695.2335 reg_novel 206.52945
loss 422.1185
STEP 301 ================================
prereg loss 419.45697 regularization 695.2434 reg_novel 206.45085
loss 420.35867
STEP 302 ================================
prereg loss 417.78278 regularization 695.25354 reg_novel 206.37009
loss 418.6844
STEP 303 ================================
prereg loss 416.65082 regularization 695.2663 reg_novel 206.28772
loss 417.55237
STEP 304 ================================
prereg loss 416.62704 regularization 695.2795 reg_novel 206.20573
loss 417.52853
STEP 305 ================================
prereg loss 417.13525 regularization 695.29395 reg_novel 206.12912
loss 418.03668
STEP 306 ================================
prereg loss 417.98602 regularization 695.3089 reg_novel 206.055
loss 418.8874
STEP 307 ================================
prereg loss 417.87775 regularization 695.3225 reg_novel 205.98221
loss 418.77905
STEP 308 ================================
prereg loss 416.29596 regularization 695.3398 reg_novel 205.91241
loss 417.1972
STEP 309 ================================
prereg loss 413.51532 regularization 695.35394 reg_novel 205.84412
loss 414.4165
STEP 310 ================================
prereg loss 410.8729 regularization 695.3697 reg_novel 205.77704
loss 411.77405
STEP 311 ================================
prereg loss 408.91476 regularization 695.3862 reg_novel 205.7128
loss 409.81586
STEP 312 ================================
prereg loss 406.78687 regularization 695.40485 reg_novel 205.6497
loss 407.68793
STEP 313 ================================
prereg loss 405.42722 regularization 695.42236 reg_novel 205.5859
loss 406.32822
STEP 314 ================================
prereg loss 404.35718 regularization 695.43933 reg_novel 205.52513
loss 405.25815
STEP 315 ================================
prereg loss 403.03613 regularization 695.45734 reg_novel 205.46599
loss 403.93704
STEP 316 ================================
prereg loss 400.6248 regularization 695.47766 reg_novel 205.40543
loss 401.52567
STEP 317 ================================
prereg loss 397.60236 regularization 695.50214 reg_novel 205.34386
loss 398.5032
STEP 318 ================================
prereg loss 394.704 regularization 695.52966 reg_novel 205.28445
loss 395.60483
STEP 319 ================================
prereg loss 392.5441 regularization 695.56165 reg_novel 205.22513
loss 393.4449
STEP 320 ================================
prereg loss 391.77356 regularization 695.59564 reg_novel 205.16585
loss 392.67432
STEP 321 ================================
prereg loss 391.75647 regularization 695.63605 reg_novel 205.10674
loss 392.65723
STEP 322 ================================
prereg loss 392.2198 regularization 695.67786 reg_novel 205.0508
loss 393.1205
STEP 323 ================================
prereg loss 391.7948 regularization 695.72046 reg_novel 204.9977
loss 392.69553
STEP 324 ================================
prereg loss 389.4417 regularization 695.7617 reg_novel 204.94633
loss 390.3424
STEP 325 ================================
prereg loss 385.97095 regularization 695.8079 reg_novel 204.89761
loss 386.87164
STEP 326 ================================
prereg loss 379.80344 regularization 695.853 reg_novel 204.85373
loss 380.70413
STEP 327 ================================
prereg loss 375.67062 regularization 695.8973 reg_novel 204.81374
loss 376.57135
STEP 328 ================================
prereg loss 372.91742 regularization 695.93585 reg_novel 204.7766
loss 373.81815
STEP 329 ================================
prereg loss 370.66278 regularization 695.9917 reg_novel 204.74542
loss 371.5635
STEP 330 ================================
prereg loss 368.2189 regularization 696.04333 reg_novel 204.71756
loss 369.11966
STEP 331 ================================
prereg loss 366.30975 regularization 696.0887 reg_novel 204.69102
loss 367.21054
STEP 332 ================================
prereg loss 364.20856 regularization 696.1347 reg_novel 204.6658
loss 365.10934
STEP 333 ================================
prereg loss 361.3907 regularization 696.177 reg_novel 204.64082
loss 362.2915
STEP 334 ================================
prereg loss 358.82288 regularization 696.2153 reg_novel 204.61772
loss 359.7237
STEP 335 ================================
prereg loss 357.0014 regularization 696.2494 reg_novel 204.59558
loss 357.90225
STEP 336 ================================
prereg loss 356.18878 regularization 696.2807 reg_novel 204.57295
loss 357.08963
STEP 337 ================================
prereg loss 356.07623 regularization 696.3109 reg_novel 204.5503
loss 356.97708
STEP 338 ================================
prereg loss 356.2048 regularization 696.3378 reg_novel 204.52832
loss 357.10568
STEP 339 ================================
prereg loss 356.12408 regularization 696.3611 reg_novel 204.5074
loss 357.02496
STEP 340 ================================
prereg loss 355.85074 regularization 696.3829 reg_novel 204.48943
loss 356.75162
STEP 341 ================================
prereg loss 355.0615 regularization 696.4033 reg_novel 204.4734
loss 355.96237
STEP 342 ================================
prereg loss 353.8911 regularization 696.42694 reg_novel 204.45506
loss 354.792
STEP 343 ================================
prereg loss 352.858 regularization 696.44684 reg_novel 204.43535
loss 353.75888
STEP 344 ================================
prereg loss 352.34262 regularization 696.46814 reg_novel 204.41533
loss 353.2435
STEP 345 ================================
prereg loss 352.21356 regularization 696.48785 reg_novel 204.39502
loss 353.11444
STEP 346 ================================
prereg loss 352.52277 regularization 696.5073 reg_novel 204.37515
loss 353.42365
STEP 347 ================================
prereg loss 352.4438 regularization 696.5259 reg_novel 204.3544
loss 353.34467
STEP 348 ================================
prereg loss 352.4546 regularization 696.5445 reg_novel 204.33252
loss 353.35547
STEP 349 ================================
prereg loss 352.4083 regularization 696.5647 reg_novel 204.30951
loss 353.30917
STEP 350 ================================
prereg loss 352.24158 regularization 696.58594 reg_novel 204.2843
loss 353.14246
STEP 351 ================================
prereg loss 352.0164 regularization 696.60657 reg_novel 204.25957
loss 352.91727
STEP 352 ================================
prereg loss 351.8079 regularization 696.62866 reg_novel 204.23543
loss 352.70877
STEP 353 ================================
prereg loss 351.95612 regularization 696.6499 reg_novel 204.21251
loss 352.85696
STEP 354 ================================
prereg loss 352.14026 regularization 696.6731 reg_novel 204.19315
loss 353.04114
STEP 355 ================================
prereg loss 352.3485 regularization 696.6955 reg_novel 204.17548
loss 353.2494
STEP 356 ================================
prereg loss 352.52127 regularization 696.71967 reg_novel 204.15953
loss 353.42215
STEP 357 ================================
prereg loss 352.668 regularization 696.74133 reg_novel 204.1461
loss 353.56888
STEP 358 ================================
prereg loss 353.32056 regularization 696.76306 reg_novel 204.13428
loss 354.22147
STEP 359 ================================
prereg loss 354.13324 regularization 696.7861 reg_novel 204.13495
loss 355.03415
STEP 360 ================================
prereg loss 354.97592 regularization 696.80804 reg_novel 204.13475
loss 355.87686
STEP 361 ================================
prereg loss 355.6416 regularization 696.83026 reg_novel 204.13538
loss 356.54257
STEP 362 ================================
prereg loss 355.95352 regularization 696.8522 reg_novel 204.13559
loss 356.85452
STEP 363 ================================
prereg loss 355.77707 regularization 696.87366 reg_novel 204.13441
loss 356.67807
STEP 364 ================================
prereg loss 355.28552 regularization 696.8952 reg_novel 204.1306
loss 356.18655
STEP 365 ================================
prereg loss 355.0616 regularization 696.91675 reg_novel 204.12407
loss 355.96265
STEP 366 ================================
prereg loss 354.51785 regularization 696.9394 reg_novel 204.11389
loss 355.4189
STEP 367 ================================
prereg loss 353.61584 regularization 696.9666 reg_novel 204.10019
loss 354.5169
STEP 368 ================================
prereg loss 352.463 regularization 696.9931 reg_novel 204.08472
loss 353.3641
STEP 369 ================================
prereg loss 351.1004 regularization 697.0192 reg_novel 204.07141
loss 352.0015
STEP 370 ================================
prereg loss 349.79636 regularization 697.0426 reg_novel 204.06032
loss 350.69745
STEP 371 ================================
prereg loss 347.521 regularization 697.0654 reg_novel 204.04895
loss 348.42212
STEP 372 ================================
prereg loss 346.0276 regularization 697.0883 reg_novel 204.03743
loss 346.9287
STEP 373 ================================
prereg loss 344.57996 regularization 697.1116 reg_novel 204.02502
loss 345.48108
STEP 374 ================================
prereg loss 343.22363 regularization 697.1407 reg_novel 204.02197
loss 344.1248
STEP 375 ================================
prereg loss 342.49716 regularization 697.1714 reg_novel 204.01338
loss 343.39835
STEP 376 ================================
prereg loss 342.80798 regularization 697.20026 reg_novel 204.0015
loss 343.7092
STEP 377 ================================
prereg loss 342.92752 regularization 697.2294 reg_novel 203.98897
loss 343.82874
STEP 378 ================================
prereg loss 341.90622 regularization 697.2553 reg_novel 203.97469
loss 342.80746
STEP 379 ================================
prereg loss 340.0808 regularization 697.2809 reg_novel 203.95483
loss 340.98206
STEP 380 ================================
prereg loss 337.783 regularization 697.305 reg_novel 203.93823
loss 338.68423
STEP 381 ================================
prereg loss 335.76364 regularization 697.3268 reg_novel 203.92256
loss 336.6649
STEP 382 ================================
prereg loss 334.44095 regularization 697.3477 reg_novel 203.90459
loss 335.3422
STEP 383 ================================
prereg loss 333.74548 regularization 697.36804 reg_novel 203.8838
loss 334.64673
STEP 384 ================================
prereg loss 333.26422 regularization 697.38806 reg_novel 203.8641
loss 334.16547
STEP 385 ================================
prereg loss 332.8485 regularization 697.4108 reg_novel 203.8449
loss 333.74976
STEP 386 ================================
prereg loss 332.48053 regularization 697.4299 reg_novel 203.82213
loss 333.38177
STEP 387 ================================
prereg loss 332.26288 regularization 697.4494 reg_novel 203.80423
loss 333.16412
STEP 388 ================================
prereg loss 332.46 regularization 697.4676 reg_novel 203.7913
loss 333.36124
STEP 389 ================================
prereg loss 332.89517 regularization 697.4853 reg_novel 203.77388
loss 333.79642
STEP 390 ================================
prereg loss 333.44046 regularization 697.50214 reg_novel 203.75403
loss 334.3417
STEP 391 ================================
prereg loss 334.11182 regularization 697.51794 reg_novel 203.73607
loss 335.01306
STEP 392 ================================
prereg loss 334.74585 regularization 697.53424 reg_novel 203.71732
loss 335.6471
STEP 393 ================================
prereg loss 335.29584 regularization 697.54834 reg_novel 203.69788
loss 336.19708
STEP 394 ================================
prereg loss 335.78616 regularization 697.5622 reg_novel 203.67567
loss 336.6874
STEP 395 ================================
prereg loss 336.28693 regularization 697.5749 reg_novel 203.6547
loss 337.18814
STEP 396 ================================
prereg loss 336.80576 regularization 697.58575 reg_novel 203.63736
loss 337.70697
STEP 397 ================================
prereg loss 337.23703 regularization 697.5957 reg_novel 203.61783
loss 338.13824
STEP 398 ================================
prereg loss 337.69882 regularization 697.60626 reg_novel 203.59508
loss 338.60004
STEP 399 ================================
prereg loss 338.1058 regularization 697.6155 reg_novel 203.57071
loss 339.007
STEP 400 ================================
prereg loss 339.61884 regularization 697.62665 reg_novel 203.54524
loss 340.52002
STEP 401 ================================
prereg loss 341.46906 regularization 697.63666 reg_novel 203.5194
loss 342.3702
STEP 402 ================================
prereg loss 343.6242 regularization 697.6483 reg_novel 203.49658
loss 344.52536
STEP 403 ================================
prereg loss 345.32507 regularization 697.66034 reg_novel 203.4804
loss 346.22623
STEP 404 ================================
prereg loss 346.42773 regularization 697.6755 reg_novel 203.46988
loss 347.3289
STEP 405 ================================
prereg loss 346.78085 regularization 697.6888 reg_novel 203.45935
loss 347.682
STEP 406 ================================
prereg loss 346.54263 regularization 697.7023 reg_novel 203.44565
loss 347.4438
STEP 407 ================================
prereg loss 346.04446 regularization 697.7151 reg_novel 203.43593
loss 346.94562
STEP 408 ================================
prereg loss 345.55145 regularization 697.7289 reg_novel 203.4312
loss 346.4526
STEP 409 ================================
prereg loss 345.1635 regularization 697.7428 reg_novel 203.42334
loss 346.06467
STEP 410 ================================
prereg loss 344.87634 regularization 697.7559 reg_novel 203.41437
loss 345.77753
STEP 411 ================================
prereg loss 344.56723 regularization 697.76953 reg_novel 203.40598
loss 345.4684
STEP 412 ================================
prereg loss 344.24988 regularization 697.7853 reg_novel 203.39793
loss 345.15106
STEP 413 ================================
prereg loss 343.93365 regularization 697.80084 reg_novel 203.38875
loss 344.83484
STEP 414 ================================
prereg loss 343.60403 regularization 697.81793 reg_novel 203.3771
loss 344.50522
STEP 415 ================================
prereg loss 343.37683 regularization 697.834 reg_novel 203.36598
loss 344.27805
STEP 416 ================================
prereg loss 343.23077 regularization 697.85236 reg_novel 203.35701
loss 344.132
STEP 417 ================================
prereg loss 343.11768 regularization 697.87067 reg_novel 203.3434
loss 344.0189
STEP 418 ================================
prereg loss 343.0488 regularization 697.89105 reg_novel 203.34055
loss 343.95004
STEP 419 ================================
prereg loss 342.60272 regularization 697.9132 reg_novel 203.353
loss 343.504
STEP 420 ================================
prereg loss 346.42224 regularization 697.9383 reg_novel 203.38147
loss 347.32355
STEP 421 ================================
prereg loss 350.73828 regularization 697.9637 reg_novel 203.41887
loss 351.63968
STEP 422 ================================
prereg loss 353.82336 regularization 697.99084 reg_novel 203.4625
loss 354.72482
STEP 423 ================================
prereg loss 354.90997 regularization 698.01697 reg_novel 203.51578
loss 355.8115
STEP 424 ================================
prereg loss 354.40265 regularization 698.0458 reg_novel 203.57869
loss 355.30426
STEP 425 ================================
prereg loss 352.58566 regularization 698.0765 reg_novel 203.64407
loss 353.4874
STEP 426 ================================
prereg loss 350.30396 regularization 698.10614 reg_novel 203.71043
loss 351.20578
STEP 427 ================================
prereg loss 347.36258 regularization 698.14026 reg_novel 203.7697
loss 348.2645
STEP 428 ================================
prereg loss 345.34564 regularization 698.1733 reg_novel 203.81938
loss 346.24762
STEP 429 ================================
prereg loss 343.6692 regularization 698.20825 reg_novel 203.85962
loss 344.57126
STEP 430 ================================
prereg loss 342.33276 regularization 698.24097 reg_novel 203.89108
loss 343.2349
STEP 431 ================================
prereg loss 341.0793 regularization 698.27246 reg_novel 203.91582
loss 341.9815
STEP 432 ================================
prereg loss 339.95602 regularization 698.3036 reg_novel 203.93443
loss 340.85828
STEP 433 ================================
prereg loss 338.9142 regularization 698.33374 reg_novel 203.94768
loss 339.8165
STEP 434 ================================
prereg loss 338.09454 regularization 698.363 reg_novel 203.9549
loss 338.99686
STEP 435 ================================
prereg loss 337.41522 regularization 698.3923 reg_novel 203.95625
loss 338.31757
STEP 436 ================================
prereg loss 336.6419 regularization 698.4251 reg_novel 203.95576
loss 337.54428
STEP 437 ================================
prereg loss 335.75464 regularization 698.45386 reg_novel 203.95494
loss 336.65704
STEP 438 ================================
prereg loss 334.31046 regularization 698.4819 reg_novel 203.95389
loss 335.2129
STEP 439 ================================
prereg loss 332.66205 regularization 698.5094 reg_novel 203.95325
loss 333.5645
STEP 440 ================================
prereg loss 331.01434 regularization 698.537 reg_novel 203.95114
loss 331.91684
STEP 441 ================================
prereg loss 329.4289 regularization 698.5646 reg_novel 203.9523
loss 330.33142
STEP 442 ================================
prereg loss 328.1252 regularization 698.5905 reg_novel 203.95842
loss 329.02777
STEP 443 ================================
prereg loss 327.17505 regularization 698.6161 reg_novel 203.9666
loss 328.07764
STEP 444 ================================
prereg loss 326.52597 regularization 698.64014 reg_novel 203.96956
loss 327.4286
STEP 445 ================================
prereg loss 326.0567 regularization 698.66583 reg_novel 203.9669
loss 326.95932
STEP 446 ================================
prereg loss 325.57443 regularization 698.68823 reg_novel 203.96162
loss 326.47708
STEP 447 ================================
prereg loss 325.0277 regularization 698.71075 reg_novel 203.95181
loss 325.93036
STEP 448 ================================
prereg loss 324.36426 regularization 698.7315 reg_novel 203.93643
loss 325.26694
STEP 449 ================================
prereg loss 323.76 regularization 698.7532 reg_novel 203.91673
loss 324.6627
STEP 450 ================================
prereg loss 323.25946 regularization 698.775 reg_novel 203.89645
loss 324.16214
STEP 451 ================================
prereg loss 322.87305 regularization 698.79614 reg_novel 203.87473
loss 323.77573
STEP 452 ================================
prereg loss 322.6024 regularization 698.81665 reg_novel 203.85469
loss 323.50507
STEP 453 ================================
prereg loss 322.4314 regularization 698.83734 reg_novel 203.83421
loss 323.33408
STEP 454 ================================
prereg loss 322.313 regularization 698.8571 reg_novel 203.81143
loss 323.21567
STEP 455 ================================
prereg loss 322.25928 regularization 698.8782 reg_novel 203.78781
loss 323.16196
STEP 456 ================================
prereg loss 322.2283 regularization 698.8997 reg_novel 203.76108
loss 323.13095
STEP 457 ================================
prereg loss 322.2671 regularization 698.9214 reg_novel 203.73053
loss 323.16974
STEP 458 ================================
prereg loss 322.63242 regularization 698.9422 reg_novel 203.69795
loss 323.53506
STEP 459 ================================
prereg loss 322.56488 regularization 698.9638 reg_novel 203.66298
loss 323.4675
STEP 460 ================================
prereg loss 322.9089 regularization 698.98474 reg_novel 203.62704
loss 323.81152
STEP 461 ================================
prereg loss 323.25183 regularization 699.0056 reg_novel 203.58975
loss 324.15442
STEP 462 ================================
prereg loss 323.5583 regularization 699.0255 reg_novel 203.54636
loss 324.46085
STEP 463 ================================
prereg loss 323.57416 regularization 699.0433 reg_novel 203.49706
loss 324.47668
STEP 464 ================================
prereg loss 323.61493 regularization 699.0612 reg_novel 203.44449
loss 324.51743
STEP 465 ================================
prereg loss 323.71704 regularization 699.0777 reg_novel 203.39001
loss 324.6195
STEP 466 ================================
prereg loss 323.8661 regularization 699.0953 reg_novel 203.33905
loss 324.76852
STEP 467 ================================
prereg loss 324.0344 regularization 699.1141 reg_novel 203.29309
loss 324.9368
STEP 468 ================================
prereg loss 324.37714 regularization 699.1375 reg_novel 203.25294
loss 325.27954
STEP 469 ================================
prereg loss 324.834 regularization 699.16345 reg_novel 203.21933
loss 325.7364
STEP 470 ================================
prereg loss 325.18994 regularization 699.18884 reg_novel 203.18509
loss 326.09232
STEP 471 ================================
prereg loss 325.4359 regularization 699.21497 reg_novel 203.14728
loss 326.3383
STEP 472 ================================
prereg loss 325.44748 regularization 699.2419 reg_novel 203.10797
loss 326.34982
STEP 473 ================================
prereg loss 325.602 regularization 699.2688 reg_novel 203.06909
loss 326.50433
STEP 474 ================================
prereg loss 325.48718 regularization 699.2942 reg_novel 203.03078
loss 326.3895
STEP 475 ================================
prereg loss 325.24673 regularization 699.3178 reg_novel 202.99286
loss 326.14905
STEP 476 ================================
prereg loss 324.9685 regularization 699.34326 reg_novel 202.9545
loss 325.8708
STEP 477 ================================
prereg loss 323.96445 regularization 699.3698 reg_novel 202.9157
loss 324.86673
STEP 478 ================================
prereg loss 323.42392 regularization 699.3955 reg_novel 202.87495
loss 324.3262
STEP 479 ================================
prereg loss 322.99466 regularization 699.4223 reg_novel 202.83311
loss 323.8969
STEP 480 ================================
prereg loss 322.60095 regularization 699.45056 reg_novel 202.79004
loss 323.5032
STEP 481 ================================
prereg loss 322.2194 regularization 699.4783 reg_novel 202.74529
loss 323.1216
STEP 482 ================================
prereg loss 321.85098 regularization 699.5074 reg_novel 202.70004
loss 322.7532
STEP 483 ================================
prereg loss 321.5114 regularization 699.53345 reg_novel 202.65417
loss 322.4136
STEP 484 ================================
prereg loss 321.28586 regularization 699.5609 reg_novel 202.6078
loss 322.18802
STEP 485 ================================
prereg loss 321.01193 regularization 699.5896 reg_novel 202.55347
loss 321.91406
STEP 486 ================================
prereg loss 320.8167 regularization 699.6172 reg_novel 202.50603
loss 321.71884
STEP 487 ================================
prereg loss 320.6881 regularization 699.6428 reg_novel 202.4625
loss 321.5902
STEP 488 ================================
prereg loss 320.5899 regularization 699.6676 reg_novel 202.4178
loss 321.492
STEP 489 ================================
prereg loss 320.49304 regularization 699.69196 reg_novel 202.37112
loss 321.3951
STEP 490 ================================
prereg loss 320.46045 regularization 699.7157 reg_novel 202.32867
loss 321.3625
STEP 491 ================================
prereg loss 320.5403 regularization 699.7388 reg_novel 202.29094
loss 321.44235
STEP 492 ================================
prereg loss 320.71344 regularization 699.76294 reg_novel 202.25426
loss 321.61545
STEP 493 ================================
prereg loss 321.1046 regularization 699.78845 reg_novel 202.21614
loss 322.00662
STEP 494 ================================
prereg loss 321.69324 regularization 699.8156 reg_novel 202.17825
loss 322.59525
STEP 495 ================================
prereg loss 322.32288 regularization 699.84125 reg_novel 202.14346
loss 323.22485
STEP 496 ================================
prereg loss 322.96655 regularization 699.8699 reg_novel 202.11046
loss 323.86853
STEP 497 ================================
prereg loss 323.55505 regularization 699.8969 reg_novel 202.07596
loss 324.45703
STEP 498 ================================
prereg loss 324.09073 regularization 699.9244 reg_novel 202.03795
loss 324.9927
STEP 499 ================================
prereg loss 324.63293 regularization 699.9501 reg_novel 201.999
loss 325.53488
STEP 500 ================================
prereg loss 325.1966 regularization 699.9761 reg_novel 201.96251
loss 326.09854
STEP 501 ================================
prereg loss 325.76764 regularization 700.0017 reg_novel 201.92612
loss 326.66956
STEP 502 ================================
prereg loss 326.38354 regularization 700.0275 reg_novel 201.88594
loss 327.28546
STEP 503 ================================
prereg loss 327.04437 regularization 700.0543 reg_novel 201.84247
loss 327.94626
STEP 504 ================================
prereg loss 327.7647 regularization 700.08124 reg_novel 201.79724
loss 328.6666
STEP 505 ================================
prereg loss 328.4949 regularization 700.1078 reg_novel 201.75296
loss 329.39676
STEP 506 ================================
prereg loss 329.28326 regularization 700.13104 reg_novel 201.70738
loss 330.1851
STEP 507 ================================
prereg loss 329.75095 regularization 700.16144 reg_novel 201.62392
loss 330.65274
STEP 508 ================================
prereg loss 330.45276 regularization 700.1897 reg_novel 201.5647
loss 331.35452
STEP 509 ================================
prereg loss 330.59033 regularization 700.2179 reg_novel 201.51682
loss 331.49207
STEP 510 ================================
prereg loss 330.1413 regularization 700.24335 reg_novel 201.46306
loss 331.043
STEP 511 ================================
prereg loss 329.323 regularization 700.26636 reg_novel 201.39882
loss 330.22467
STEP 512 ================================
prereg loss 328.71475 regularization 700.2905 reg_novel 201.34048
loss 329.6164
STEP 513 ================================
prereg loss 328.3658 regularization 700.3157 reg_novel 201.3019
loss 329.26743
STEP 514 ================================
prereg loss 328.1947 regularization 700.34174 reg_novel 201.27481
loss 329.0963
STEP 515 ================================
prereg loss 327.91574 regularization 700.3715 reg_novel 201.2407
loss 328.81735
STEP 516 ================================
prereg loss 327.80414 regularization 700.4001 reg_novel 201.19695
loss 328.70575
STEP 517 ================================
prereg loss 328.07648 regularization 700.4287 reg_novel 201.1527
loss 328.97806
STEP 518 ================================
prereg loss 328.53772 regularization 700.4607 reg_novel 201.11937
loss 329.4393
STEP 519 ================================
prereg loss 329.25534 regularization 700.49274 reg_novel 201.09428
loss 330.15692
STEP 520 ================================
prereg loss 330.07825 regularization 700.52704 reg_novel 201.06776
loss 330.97983
STEP 521 ================================
prereg loss 330.93555 regularization 700.5592 reg_novel 201.03421
loss 331.83713
STEP 522 ================================
prereg loss 331.57428 regularization 700.5904 reg_novel 200.99832
loss 332.47586
STEP 523 ================================
prereg loss 332.04706 regularization 700.62366 reg_novel 200.96997
loss 332.94864
STEP 524 ================================
prereg loss 332.4016 regularization 700.6553 reg_novel 200.95178
loss 333.30322
STEP 525 ================================
prereg loss 332.66486 regularization 700.68756 reg_novel 200.93541
loss 333.56647
STEP 526 ================================
prereg loss 332.84232 regularization 700.7176 reg_novel 200.91374
loss 333.74396
STEP 527 ================================
prereg loss 333.0401 regularization 700.7504 reg_novel 200.88872
loss 333.94174
STEP 528 ================================
prereg loss 333.39655 regularization 700.78546 reg_novel 200.86787
loss 334.2982
STEP 529 ================================
prereg loss 333.91272 regularization 700.8178 reg_novel 200.85228
loss 334.8144
STEP 530 ================================
prereg loss 334.6318 regularization 700.8539 reg_novel 200.83943
loss 335.5335
STEP 531 ================================
prereg loss 335.01154 regularization 700.8875 reg_novel 200.82098
loss 335.91324
STEP 532 ================================
prereg loss 335.02127 regularization 700.91956 reg_novel 200.79756
loss 335.92297
STEP 533 ================================
prereg loss 335.15018 regularization 700.95135 reg_novel 200.76271
loss 336.05188
STEP 534 ================================
prereg loss 335.3286 regularization 700.98364 reg_novel 200.7219
loss 336.23032
STEP 535 ================================
prereg loss 335.4631 regularization 701.01636 reg_novel 200.67786
loss 336.3648
STEP 536 ================================
prereg loss 335.68604 regularization 701.0513 reg_novel 200.64008
loss 336.58774
STEP 537 ================================
prereg loss 335.93878 regularization 701.086 reg_novel 200.6049
loss 336.84048
STEP 538 ================================
prereg loss 336.20123 regularization 701.12195 reg_novel 200.57259
loss 337.10294
STEP 539 ================================
prereg loss 336.50287 regularization 701.1602 reg_novel 200.54364
loss 337.40457
STEP 540 ================================
prereg loss 336.85767 regularization 701.1971 reg_novel 200.52148
loss 337.7594
STEP 541 ================================
prereg loss 335.38678 regularization 701.2362 reg_novel 200.50215
loss 336.2885
STEP 542 ================================
prereg loss 334.32565 regularization 701.2749 reg_novel 200.48055
loss 335.22742
STEP 543 ================================
prereg loss 333.5693 regularization 701.31104 reg_novel 200.45387
loss 334.47107
STEP 544 ================================
prereg loss 333.19434 regularization 701.34845 reg_novel 200.42712
loss 334.0961
STEP 545 ================================
prereg loss 332.87848 regularization 701.38525 reg_novel 200.40015
loss 333.78027
STEP 546 ================================
prereg loss 332.8483 regularization 701.42206 reg_novel 200.37416
loss 333.7501
STEP 547 ================================
prereg loss 333.29938 regularization 701.46027 reg_novel 200.34795
loss 334.20117
STEP 548 ================================
prereg loss 334.1164 regularization 701.49854 reg_novel 200.32144
loss 335.01822
STEP 549 ================================
prereg loss 334.94815 regularization 701.53314 reg_novel 200.29605
loss 335.84998
STEP 550 ================================
prereg loss 335.45044 regularization 701.5685 reg_novel 200.27292
loss 336.3523
STEP 551 ================================
prereg loss 335.2155 regularization 701.6032 reg_novel 200.25246
loss 336.11737
STEP 552 ================================
prereg loss 334.5486 regularization 701.6355 reg_novel 200.23465
loss 335.45047
STEP 553 ================================
prereg loss 333.65985 regularization 701.6692 reg_novel 200.21378
loss 334.56174
STEP 554 ================================
prereg loss 333.1687 regularization 701.7061 reg_novel 200.18854
loss 334.0706
STEP 555 ================================
prereg loss 331.92413 regularization 701.74316 reg_novel 200.13379
loss 332.82602
STEP 556 ================================
prereg loss 330.98822 regularization 701.7805 reg_novel 200.09143
loss 331.8901
STEP 557 ================================
prereg loss 330.25912 regularization 701.8189 reg_novel 200.06221
loss 331.161
STEP 558 ================================
prereg loss 329.70233 regularization 701.85724 reg_novel 200.03906
loss 330.60422
STEP 559 ================================
prereg loss 329.2185 regularization 701.89453 reg_novel 200.01772
loss 330.12042
STEP 560 ================================
prereg loss 328.7845 regularization 701.932 reg_novel 199.99536
loss 329.6864
STEP 561 ================================
prereg loss 328.50824 regularization 701.9685 reg_novel 199.97704
loss 329.4102
STEP 562 ================================
prereg loss 328.39667 regularization 702.00543 reg_novel 199.96695
loss 329.29865
STEP 563 ================================
prereg loss 328.3756 regularization 702.0435 reg_novel 199.96585
loss 329.27762
STEP 564 ================================
prereg loss 328.44214 regularization 702.0809 reg_novel 199.96893
loss 329.34418
STEP 565 ================================
prereg loss 328.65546 regularization 702.11884 reg_novel 199.97182
loss 329.55756
STEP 566 ================================
prereg loss 328.96317 regularization 702.1567 reg_novel 199.98114
loss 329.8653
STEP 567 ================================
prereg loss 329.36664 regularization 702.19507 reg_novel 199.99388
loss 330.26883
STEP 568 ================================
prereg loss 329.51746 regularization 702.2348 reg_novel 200.01025
loss 330.4197
STEP 569 ================================
prereg loss 329.802 regularization 702.2737 reg_novel 200.0309
loss 330.7043
STEP 570 ================================
prereg loss 330.18262 regularization 702.31177 reg_novel 200.04959
loss 331.085
STEP 571 ================================
prereg loss 330.2544 regularization 702.34985 reg_novel 200.06516
loss 331.1568
STEP 572 ================================
prereg loss 330.43628 regularization 702.38916 reg_novel 200.07687
loss 331.33875
STEP 573 ================================
prereg loss 330.7365 regularization 702.4273 reg_novel 200.0835
loss 331.639
STEP 574 ================================
prereg loss 331.67566 regularization 702.4636 reg_novel 200.08292
loss 332.57822
STEP 575 ================================
prereg loss 332.6278 regularization 702.49854 reg_novel 200.07532
loss 333.5304
STEP 576 ================================
prereg loss 333.419 regularization 702.5355 reg_novel 200.06332
loss 334.3216
STEP 577 ================================
prereg loss 334.13226 regularization 702.56885 reg_novel 200.04616
loss 335.03488
STEP 578 ================================
prereg loss 334.6374 regularization 702.60236 reg_novel 200.02573
loss 335.54
STEP 579 ================================
prereg loss 335.2978 regularization 702.6325 reg_novel 199.82619
loss 336.20026
STEP 580 ================================
prereg loss 336.04077 regularization 702.6608 reg_novel 199.64197
loss 336.94308
STEP 581 ================================
prereg loss 334.75925 regularization 702.69257 reg_novel 199.47386
loss 335.6614
STEP 582 ================================
prereg loss 331.3051 regularization 702.72363 reg_novel 199.32169
loss 332.20715
STEP 583 ================================
prereg loss 326.8159 regularization 702.75507 reg_novel 199.18634
loss 327.71783
STEP 584 ================================
prereg loss 322.96677 regularization 702.7902 reg_novel 199.06535
loss 323.86862
STEP 585 ================================
prereg loss 320.78256 regularization 702.8275 reg_novel 198.95682
loss 321.68436
STEP 586 ================================
prereg loss 320.13852 regularization 702.8622 reg_novel 198.86084
loss 321.04025
STEP 587 ================================
prereg loss 320.13446 regularization 702.8969 reg_novel 198.77515
loss 321.03613
STEP 588 ================================
prereg loss 319.90442 regularization 702.93304 reg_novel 198.69843
loss 320.80606
STEP 589 ================================
prereg loss 319.34674 regularization 702.971 reg_novel 198.62778
loss 320.24835
STEP 590 ================================
prereg loss 318.88745 regularization 703.00977 reg_novel 198.56277
loss 319.78903
STEP 591 ================================
prereg loss 319.57245 regularization 703.0486 reg_novel 198.50429
loss 320.474
STEP 592 ================================
prereg loss 321.09076 regularization 703.0867 reg_novel 198.45178
loss 321.9923
STEP 593 ================================
prereg loss 322.98776 regularization 703.1275 reg_novel 198.40602
loss 323.88928
STEP 594 ================================
prereg loss 324.65643 regularization 703.1679 reg_novel 198.3664
loss 325.55795
STEP 595 ================================
prereg loss 325.52768 regularization 703.20734 reg_novel 198.33192
loss 326.42923
STEP 596 ================================
prereg loss 325.56445 regularization 703.2465 reg_novel 198.3016
loss 326.466
STEP 597 ================================
prereg loss 325.1109 regularization 703.287 reg_novel 198.27423
loss 326.01245
STEP 598 ================================
prereg loss 324.64508 regularization 703.3239 reg_novel 198.24991
loss 325.54666
STEP 599 ================================
prereg loss 324.52182 regularization 703.3627 reg_novel 198.22679
loss 325.4234
STEP 600 ================================
prereg loss 324.684 regularization 703.4007 reg_novel 198.20447
loss 325.5856
STEP 601 ================================
prereg loss 325.08374 regularization 703.43744 reg_novel 198.1818
loss 325.98535
STEP 602 ================================
prereg loss 325.69696 regularization 703.47327 reg_novel 198.1591
loss 326.5986
STEP 603 ================================
prereg loss 326.5963 regularization 703.5053 reg_novel 198.13531
loss 327.49796
STEP 604 ================================
prereg loss 328.06207 regularization 703.5393 reg_novel 198.1124
loss 328.9637
STEP 605 ================================
prereg loss 329.8701 regularization 703.5709 reg_novel 198.09076
loss 330.77176
STEP 606 ================================
prereg loss 331.48755 regularization 703.60455 reg_novel 198.06963
loss 332.38922
STEP 607 ================================
prereg loss 332.50513 regularization 703.6346 reg_novel 198.04897
loss 333.4068
STEP 608 ================================
prereg loss 332.7499 regularization 703.6653 reg_novel 198.02686
loss 333.6516
STEP 609 ================================
prereg loss 332.4386 regularization 703.6959 reg_novel 198.00287
loss 333.3403
STEP 610 ================================
prereg loss 331.81973 regularization 703.7272 reg_novel 197.97763
loss 332.72144
STEP 611 ================================
prereg loss 331.13696 regularization 703.75555 reg_novel 197.95026
loss 332.03867
STEP 612 ================================
prereg loss 330.61005 regularization 703.7868 reg_novel 197.92046
loss 331.51175
STEP 613 ================================
prereg loss 330.50653 regularization 703.81445 reg_novel 197.89494
loss 331.40823
STEP 614 ================================
prereg loss 330.52222 regularization 703.844 reg_novel 197.87161
loss 331.42392
STEP 615 ================================
prereg loss 330.71118 regularization 703.8741 reg_novel 197.84738
loss 331.6129
STEP 616 ================================
prereg loss 331.0888 regularization 703.9014 reg_novel 197.82181
loss 331.99054
STEP 617 ================================
prereg loss 331.80215 regularization 703.9283 reg_novel 197.79536
loss 332.7039
STEP 618 ================================
prereg loss 333.00247 regularization 703.9552 reg_novel 197.76807
loss 333.9042
STEP 619 ================================
prereg loss 334.25732 regularization 703.98145 reg_novel 197.74077
loss 335.15906
STEP 620 ================================
prereg loss 335.26974 regularization 704.00867 reg_novel 197.71457
loss 336.17148
STEP 621 ================================
prereg loss 335.59894 regularization 704.0356 reg_novel 197.6893
loss 336.50067
STEP 622 ================================
prereg loss 334.8281 regularization 704.06805 reg_novel 197.66557
loss 335.72983
STEP 623 ================================
prereg loss 333.19012 regularization 704.1002 reg_novel 197.64308
loss 334.09186
STEP 624 ================================
prereg loss 331.33307 regularization 704.1329 reg_novel 197.62149
loss 332.23483
STEP 625 ================================
prereg loss 329.4743 regularization 704.16547 reg_novel 197.59976
loss 330.37607
STEP 626 ================================
prereg loss 327.89218 regularization 704.19916 reg_novel 197.57793
loss 328.79395
STEP 627 ================================
prereg loss 326.69785 regularization 704.23145 reg_novel 197.55301
loss 327.59964
STEP 628 ================================
prereg loss 325.77512 regularization 704.26495 reg_novel 197.52563
loss 326.6769
STEP 629 ================================
prereg loss 325.0801 regularization 704.30054 reg_novel 197.49393
loss 325.9819
STEP 630 ================================
prereg loss 324.4302 regularization 704.3344 reg_novel 197.45987
loss 325.332
STEP 631 ================================
prereg loss 323.7347 regularization 704.3692 reg_novel 197.42332
loss 324.6365
STEP 632 ================================
prereg loss 323.1108 regularization 704.40466 reg_novel 197.38536
loss 324.0126
STEP 633 ================================
prereg loss 322.59872 regularization 704.4385 reg_novel 197.3465
loss 323.50052
STEP 634 ================================
prereg loss 321.63885 regularization 704.4711 reg_novel 197.30722
loss 322.54062
STEP 635 ================================
prereg loss 320.2094 regularization 704.5025 reg_novel 197.26761
loss 321.11118
STEP 636 ================================
prereg loss 318.33435 regularization 704.5336 reg_novel 197.22675
loss 319.2361
STEP 637 ================================
prereg loss 316.6933 regularization 704.5667 reg_novel 197.18471
loss 317.59506
STEP 638 ================================
prereg loss 315.4108 regularization 704.6021 reg_novel 197.1419
loss 316.31253
STEP 639 ================================
prereg loss 314.29816 regularization 704.6362 reg_novel 197.09859
loss 315.1999
STEP 640 ================================
prereg loss 311.5723 regularization 704.67267 reg_novel 197.05286
loss 312.47403
STEP 641 ================================
prereg loss 308.8701 regularization 704.70825 reg_novel 197.00711
loss 309.7718
STEP 642 ================================
prereg loss 305.7087 regularization 704.74426 reg_novel 196.96123
loss 306.6104
STEP 643 ================================
prereg loss 302.60587 regularization 704.77826 reg_novel 196.91612
loss 303.50757
STEP 644 ================================
prereg loss 299.22122 regularization 704.8125 reg_novel 196.87215
loss 300.1229
STEP 645 ================================
prereg loss 295.563 regularization 704.8463 reg_novel 196.82974
loss 296.46466
STEP 646 ================================
prereg loss 291.75534 regularization 704.8829 reg_novel 196.78883
loss 292.657
STEP 647 ================================
prereg loss 288.12756 regularization 704.9199 reg_novel 196.75049
loss 289.02924
STEP 648 ================================
prereg loss 284.84012 regularization 704.9526 reg_novel 196.71336
loss 285.7418
STEP 649 ================================
prereg loss 281.93814 regularization 704.9874 reg_novel 196.67967
loss 282.8398
STEP 650 ================================
prereg loss 279.5329 regularization 705.02094 reg_novel 196.64806
loss 280.43457
STEP 651 ================================
prereg loss 277.1176 regularization 705.0532 reg_novel 196.6148
loss 278.0193
STEP 652 ================================
prereg loss 274.3274 regularization 705.08386 reg_novel 196.57985
loss 275.22906
STEP 653 ================================
prereg loss 271.1256 regularization 705.1101 reg_novel 196.54399
loss 272.02725
STEP 654 ================================
prereg loss 267.9244 regularization 705.13324 reg_novel 196.50781
loss 268.82605
STEP 655 ================================
prereg loss 264.81427 regularization 705.1558 reg_novel 196.47119
loss 265.7159
STEP 656 ================================
prereg loss 262.0954 regularization 705.1767 reg_novel 196.43323
loss 262.997
STEP 657 ================================
prereg loss 260.1938 regularization 705.19586 reg_novel 196.39433
loss 261.09537
STEP 658 ================================
prereg loss 258.5155 regularization 705.2112 reg_novel 196.35541
loss 259.41708
STEP 659 ================================
prereg loss 256.46292 regularization 705.2211 reg_novel 196.31964
loss 257.36447
STEP 660 ================================
prereg loss 253.96344 regularization 705.2283 reg_novel 196.28577
loss 254.86496
STEP 661 ================================
prereg loss 252.11336 regularization 705.233 reg_novel 196.25777
loss 253.01485
STEP 662 ================================
prereg loss 249.45702 regularization 705.23676 reg_novel 196.23341
loss 250.35849
STEP 663 ================================
prereg loss 248.22641 regularization 705.2381 reg_novel 196.18803
loss 249.12784
STEP 664 ================================
prereg loss 248.16196 regularization 705.2367 reg_novel 196.14874
loss 249.06334
STEP 665 ================================
prereg loss 256.02094 regularization 705.2312 reg_novel 196.11534
loss 256.92227
STEP 666 ================================
prereg loss 272.3999 regularization 705.2219 reg_novel 196.08937
loss 273.3012
STEP 667 ================================
prereg loss 294.52142 regularization 705.2109 reg_novel 196.06113
loss 295.4227
STEP 668 ================================
prereg loss 304.58124 regularization 705.19836 reg_novel 196.01866
loss 305.48245
STEP 669 ================================
prereg loss 304.1231 regularization 705.1817 reg_novel 195.94759
loss 305.02423
STEP 670 ================================
prereg loss 337.84375 regularization 705.16656 reg_novel 195.84966
loss 338.74475
STEP 671 ================================
prereg loss 380.85425 regularization 705.148 reg_novel 195.73027
loss 381.75513
STEP 672 ================================
prereg loss 359.44812 regularization 705.1304 reg_novel 195.59818
loss 360.34885
STEP 673 ================================
prereg loss 284.8369 regularization 705.11255 reg_novel 195.45914
loss 285.7375
STEP 674 ================================
prereg loss 229.51598 regularization 705.0978 reg_novel 195.32036
loss 230.4164
STEP 675 ================================
prereg loss 233.72513 regularization 705.08215 reg_novel 195.18614
loss 234.6254
STEP 676 ================================
prereg loss 263.9688 regularization 705.0684 reg_novel 195.06242
loss 264.86893
STEP 677 ================================
prereg loss 266.64722 regularization 705.05963 reg_novel 194.9521
loss 267.54724
STEP 678 ================================
prereg loss 235.26141 regularization 705.053 reg_novel 194.85988
loss 236.16133
STEP 679 ================================
prereg loss 204.37599 regularization 705.05316 reg_novel 194.78688
loss 205.27583
STEP 680 ================================
prereg loss 191.17802 regularization 705.06104 reg_novel 194.72931
loss 192.07782
STEP 681 ================================
prereg loss 191.66019 regularization 705.0687 reg_novel 194.68365
loss 192.55994
STEP 682 ================================
prereg loss 193.70988 regularization 705.0812 reg_novel 194.64636
loss 194.60962
STEP 683 ================================
prereg loss 197.81284 regularization 705.09485 reg_novel 194.59042
loss 198.71252
STEP 684 ================================
prereg loss 213.06436 regularization 705.11395 reg_novel 194.54167
loss 213.96402
STEP 685 ================================
prereg loss 238.42197 regularization 705.1371 reg_novel 194.49521
loss 239.3216
STEP 686 ================================
prereg loss 241.48538 regularization 705.16534 reg_novel 194.44183
loss 242.385
STEP 687 ================================
prereg loss 220.35764 regularization 705.19617 reg_novel 194.38177
loss 221.25722
STEP 688 ================================
prereg loss 203.93982 regularization 705.2294 reg_novel 194.3191
loss 204.83937
STEP 689 ================================
prereg loss 195.79869 regularization 705.2675 reg_novel 194.25923
loss 196.69821
STEP 690 ================================
prereg loss 192.6771 regularization 705.31134 reg_novel 194.20447
loss 193.57661
STEP 691 ================================
prereg loss 195.5038 regularization 705.35895 reg_novel 194.15451
loss 196.40332
STEP 692 ================================
prereg loss 205.65999 regularization 705.4068 reg_novel 194.10663
loss 206.55951
STEP 693 ================================
prereg loss 215.67114 regularization 705.4574 reg_novel 194.06282
loss 216.57066
STEP 694 ================================
prereg loss 216.29962 regularization 705.50275 reg_novel 194.02385
loss 217.19914
STEP 695 ================================
prereg loss 207.45871 regularization 705.5494 reg_novel 193.98877
loss 208.35825
STEP 696 ================================
prereg loss 196.91187 regularization 705.5975 reg_novel 193.95721
loss 197.81142
STEP 697 ================================
prereg loss 190.19125 regularization 705.6491 reg_novel 193.92578
loss 191.09084
STEP 698 ================================
prereg loss 188.78816 regularization 705.698 reg_novel 193.89969
loss 189.68776
STEP 699 ================================
prereg loss 188.64886 regularization 705.7458 reg_novel 193.88242
loss 189.5485
STEP 700 ================================
prereg loss 187.34138 regularization 705.7983 reg_novel 193.87596
loss 188.24106
STEP 701 ================================
prereg loss 188.67517 regularization 705.85315 reg_novel 193.89818
loss 189.57492
STEP 702 ================================
prereg loss 188.37872 regularization 705.8989 reg_novel 193.79231
loss 189.27841
STEP 703 ================================
prereg loss 192.79861 regularization 705.9425 reg_novel 193.69965
loss 193.69826
STEP 704 ================================
prereg loss 186.96227 regularization 705.9827 reg_novel 193.618
loss 187.86186
STEP 705 ================================
prereg loss 174.68962 regularization 706.02234 reg_novel 193.55006
loss 175.58919
STEP 706 ================================
prereg loss 168.0994 regularization 706.0597 reg_novel 193.50087
loss 168.99896
STEP 707 ================================
prereg loss 169.69293 regularization 706.0952 reg_novel 193.4755
loss 170.5925
STEP 708 ================================
prereg loss 171.22997 regularization 706.131 reg_novel 193.4864
loss 172.12958
STEP 709 ================================
prereg loss 167.27338 regularization 706.169 reg_novel 193.53078
loss 168.17308
STEP 710 ================================
prereg loss 163.32619 regularization 706.2074 reg_novel 193.60217
loss 164.226
STEP 711 ================================
prereg loss 164.81671 regularization 706.2462 reg_novel 193.69281
loss 165.71664
STEP 712 ================================
prereg loss 168.89658 regularization 706.28436 reg_novel 193.78867
loss 169.79665
STEP 713 ================================
prereg loss 170.594 regularization 706.326 reg_novel 193.88496
loss 171.4942
STEP 714 ================================
prereg loss 167.63113 regularization 706.36804 reg_novel 193.97932
loss 168.53148
STEP 715 ================================
prereg loss 162.36063 regularization 706.4172 reg_novel 194.07555
loss 163.26112
STEP 716 ================================
prereg loss 157.94171 regularization 706.4656 reg_novel 194.17409
loss 158.84235
STEP 717 ================================
prereg loss 156.16716 regularization 706.5135 reg_novel 194.27852
loss 157.06795
STEP 718 ================================
prereg loss 155.00726 regularization 706.5637 reg_novel 194.39133
loss 155.90822
STEP 719 ================================
prereg loss 153.8382 regularization 706.6165 reg_novel 194.5167
loss 154.73933
STEP 720 ================================
prereg loss 156.18968 regularization 706.67255 reg_novel 194.65225
loss 157.091
STEP 721 ================================
prereg loss 161.50238 regularization 706.7287 reg_novel 194.78795
loss 162.4039
STEP 722 ================================
prereg loss 165.23315 regularization 706.7877 reg_novel 194.91988
loss 166.13486
STEP 723 ================================
prereg loss 164.1069 regularization 706.84985 reg_novel 195.0417
loss 165.00879
STEP 724 ================================
prereg loss 159.39641 regularization 706.9143 reg_novel 195.12398
loss 160.29845
STEP 725 ================================
prereg loss 155.40852 regularization 706.9783 reg_novel 195.20477
loss 156.3107
STEP 726 ================================
prereg loss 153.56291 regularization 707.04193 reg_novel 195.28488
loss 154.46524
STEP 727 ================================
prereg loss 152.97394 regularization 707.10516 reg_novel 195.37038
loss 153.87642
STEP 728 ================================
prereg loss 152.85132 regularization 707.16876 reg_novel 195.4631
loss 153.75395
STEP 729 ================================
prereg loss 153.37321 regularization 707.23193 reg_novel 195.56274
loss 154.27602
STEP 730 ================================
prereg loss 154.5187 regularization 707.2967 reg_novel 195.6661
loss 155.42168
STEP 731 ================================
prereg loss 155.2486 regularization 707.3604 reg_novel 195.76556
loss 156.15172
STEP 732 ================================
prereg loss 155.4357 regularization 707.42395 reg_novel 195.86781
loss 156.33899
STEP 733 ================================
prereg loss 154.96443 regularization 707.4879 reg_novel 195.97168
loss 155.86789
STEP 734 ================================
prereg loss 154.8967 regularization 707.5501 reg_novel 196.0785
loss 155.80032
STEP 735 ================================
prereg loss 155.44511 regularization 707.6109 reg_novel 196.1753
loss 156.3489
STEP 736 ================================
prereg loss 155.63834 regularization 707.66815 reg_novel 196.26103
loss 156.54227
STEP 737 ================================
prereg loss 154.90997 regularization 707.726 reg_novel 196.33687
loss 155.81404
STEP 738 ================================
prereg loss 152.79588 regularization 707.78033 reg_novel 196.40375
loss 153.70007
STEP 739 ================================
prereg loss 149.89629 regularization 707.8345 reg_novel 196.46327
loss 150.80058
STEP 740 ================================
prereg loss 146.36447 regularization 707.8872 reg_novel 196.51231
loss 147.26888
STEP 741 ================================
prereg loss 143.24382 regularization 707.9412 reg_novel 196.58995
loss 144.14835
STEP 742 ================================
prereg loss 141.319 regularization 708.0002 reg_novel 196.70802
loss 142.22371
STEP 743 ================================
prereg loss 137.29807 regularization 708.0539 reg_novel 196.79181
loss 138.20291
STEP 744 ================================
prereg loss 139.19507 regularization 708.1055 reg_novel 196.88068
loss 140.10005
STEP 745 ================================
prereg loss 146.54895 regularization 708.1542 reg_novel 196.96843
loss 147.45407
STEP 746 ================================
prereg loss 147.86458 regularization 708.20514 reg_novel 197.05003
loss 148.76984
STEP 747 ================================
prereg loss 141.42184 regularization 708.25525 reg_novel 197.12074
loss 142.32722
STEP 748 ================================
prereg loss 139.43924 regularization 708.3113 reg_novel 197.1761
loss 140.34473
STEP 749 ================================
prereg loss 146.54039 regularization 708.3646 reg_novel 197.21321
loss 147.44597
STEP 750 ================================
prereg loss 153.70624 regularization 708.41754 reg_novel 197.23193
loss 154.6119
STEP 751 ================================
prereg loss 150.43484 regularization 708.4696 reg_novel 197.23454
loss 151.34055
STEP 752 ================================
prereg loss 137.74106 regularization 708.52185 reg_novel 197.2242
loss 138.6468
STEP 753 ================================
prereg loss 128.39351 regularization 708.574 reg_novel 197.20605
loss 129.29929
STEP 754 ================================
prereg loss 127.41134 regularization 708.6241 reg_novel 197.1928
loss 128.31715
STEP 755 ================================
prereg loss 128.13475 regularization 708.6745 reg_novel 197.18724
loss 129.04062
STEP 756 ================================
prereg loss 126.651276 regularization 708.7234 reg_novel 197.20325
loss 127.557205
STEP 757 ================================
prereg loss 123.83823 regularization 708.77045 reg_novel 197.24141
loss 124.74424
STEP 758 ================================
prereg loss 122.13264 regularization 708.81714 reg_novel 197.29778
loss 123.03875
STEP 759 ================================
prereg loss 121.904 regularization 708.86115 reg_novel 197.36844
loss 122.81023
STEP 760 ================================
prereg loss 121.92764 regularization 708.9037 reg_novel 197.43404
loss 122.833984
STEP 761 ================================
prereg loss 121.42174 regularization 708.9447 reg_novel 197.49338
loss 122.32818
STEP 762 ================================
prereg loss 120.91011 regularization 708.98486 reg_novel 197.54576
loss 121.81664
STEP 763 ================================
prereg loss 120.85806 regularization 709.02563 reg_novel 197.58966
loss 121.76468
STEP 764 ================================
prereg loss 120.803085 regularization 709.0662 reg_novel 197.6253
loss 121.70978
STEP 765 ================================
prereg loss 120.35304 regularization 709.1079 reg_novel 197.6538
loss 121.2598
STEP 766 ================================
prereg loss 119.73311 regularization 709.1448 reg_novel 197.67586
loss 120.63993
STEP 767 ================================
prereg loss 118.913956 regularization 709.1802 reg_novel 197.6894
loss 119.82082
STEP 768 ================================
prereg loss 117.85636 regularization 709.2144 reg_novel 197.69295
loss 118.76327
STEP 769 ================================
prereg loss 116.48219 regularization 709.2483 reg_novel 197.68942
loss 117.38913
STEP 770 ================================
prereg loss 115.45193 regularization 709.28204 reg_novel 197.68135
loss 116.358894
STEP 771 ================================
prereg loss 115.16992 regularization 709.3159 reg_novel 197.67108
loss 116.07691
STEP 772 ================================
prereg loss 115.19074 regularization 709.3502 reg_novel 197.65837
loss 116.09775
STEP 773 ================================
prereg loss 115.441864 regularization 709.3832 reg_novel 197.64459
loss 116.34889
STEP 774 ================================
prereg loss 115.43256 regularization 709.416 reg_novel 197.62817
loss 116.33961
STEP 775 ================================
prereg loss 115.42599 regularization 709.4486 reg_novel 197.60954
loss 116.333046
STEP 776 ================================
prereg loss 115.57705 regularization 709.48047 reg_novel 197.59204
loss 116.48412
STEP 777 ================================
prereg loss 115.65675 regularization 709.51013 reg_novel 197.57494
loss 116.563835
STEP 778 ================================
prereg loss 115.27127 regularization 709.5392 reg_novel 197.55518
loss 116.17837
STEP 779 ================================
prereg loss 114.74456 regularization 709.56696 reg_novel 197.53354
loss 115.65166
STEP 780 ================================
prereg loss 114.40054 regularization 709.5939 reg_novel 197.5112
loss 115.30765
STEP 781 ================================
prereg loss 114.35842 regularization 709.62036 reg_novel 197.48834
loss 115.26553
STEP 782 ================================
prereg loss 114.1638 regularization 709.6491 reg_novel 197.46265
loss 115.070915
STEP 783 ================================
prereg loss 113.69862 regularization 709.6785 reg_novel 197.43292
loss 114.605736
STEP 784 ================================
prereg loss 113.00226 regularization 709.7072 reg_novel 197.4009
loss 113.90936
STEP 785 ================================
prereg loss 112.53077 regularization 709.7384 reg_novel 197.36897
loss 113.437874
STEP 786 ================================
prereg loss 111.87521 regularization 709.76984 reg_novel 197.33421
loss 112.78232
STEP 787 ================================
prereg loss 111.11843 regularization 709.8006 reg_novel 197.30014
loss 112.025536
STEP 788 ================================
prereg loss 110.61684 regularization 709.8308 reg_novel 197.26726
loss 111.52393
STEP 789 ================================
prereg loss 110.00017 regularization 709.8623 reg_novel 197.23438
loss 110.907265
STEP 790 ================================
prereg loss 109.31618 regularization 709.89185 reg_novel 197.20168
loss 110.223274
STEP 791 ================================
prereg loss 108.588455 regularization 709.92065 reg_novel 197.17032
loss 109.495544
STEP 792 ================================
prereg loss 108.053154 regularization 709.95374 reg_novel 197.14243
loss 108.96025
STEP 793 ================================
prereg loss 107.497284 regularization 709.9865 reg_novel 197.116
loss 108.40439
STEP 794 ================================
prereg loss 106.92577 regularization 710.02124 reg_novel 197.09047
loss 107.832886
STEP 795 ================================
prereg loss 106.65949 regularization 710.056 reg_novel 197.0662
loss 107.56661
STEP 796 ================================
prereg loss 106.373764 regularization 710.09314 reg_novel 197.04468
loss 107.2809
STEP 797 ================================
prereg loss 106.505684 regularization 710.13184 reg_novel 197.0275
loss 107.41284
STEP 798 ================================
prereg loss 106.564674 regularization 710.1691 reg_novel 197.01036
loss 107.471855
STEP 799 ================================
prereg loss 106.6578 regularization 710.21063 reg_novel 196.9943
loss 107.565
STEP 800 ================================
prereg loss 106.72894 regularization 710.25195 reg_novel 196.97824
loss 107.63617
STEP 801 ================================
prereg loss 106.59897 regularization 710.2952 reg_novel 196.96198
loss 107.506226
STEP 802 ================================
prereg loss 106.4801 regularization 710.3371 reg_novel 196.94699
loss 107.38739
STEP 803 ================================
prereg loss 106.34093 regularization 710.37915 reg_novel 196.93295
loss 107.24824
STEP 804 ================================
prereg loss 106.20499 regularization 710.41644 reg_novel 196.91959
loss 107.11232
STEP 805 ================================
prereg loss 105.983734 regularization 710.4538 reg_novel 196.9076
loss 106.8911
STEP 806 ================================
prereg loss 105.77286 regularization 710.48846 reg_novel 196.89597
loss 106.680244
STEP 807 ================================
prereg loss 105.535675 regularization 710.5226 reg_novel 196.8856
loss 106.443085
STEP 808 ================================
prereg loss 105.085175 regularization 710.55457 reg_novel 196.87466
loss 105.99261
STEP 809 ================================
prereg loss 104.08586 regularization 710.58997 reg_novel 196.86154
loss 104.99331
STEP 810 ================================
prereg loss 103.060005 regularization 710.62524 reg_novel 196.84949
loss 103.96748
STEP 811 ================================
prereg loss 102.23427 regularization 710.6592 reg_novel 196.83887
loss 103.14177
STEP 812 ================================
prereg loss 101.64561 regularization 710.6939 reg_novel 196.82944
loss 102.55313
STEP 813 ================================
prereg loss 101.251366 regularization 710.7321 reg_novel 196.82115
loss 102.15892
STEP 814 ================================
prereg loss 101.37348 regularization 710.7729 reg_novel 196.81586
loss 102.28107
STEP 815 ================================
prereg loss 101.4877 regularization 710.81195 reg_novel 196.8112
loss 102.395325
STEP 816 ================================
prereg loss 101.546326 regularization 710.8534 reg_novel 196.80551
loss 102.45399
STEP 817 ================================
prereg loss 101.54545 regularization 710.89484 reg_novel 196.79988
loss 102.45314
STEP 818 ================================
prereg loss 101.44118 regularization 710.9345 reg_novel 196.79468
loss 102.34891
STEP 819 ================================
prereg loss 101.19191 regularization 710.9743 reg_novel 196.7906
loss 102.09968
STEP 820 ================================
prereg loss 100.46146 regularization 711.01575 reg_novel 196.78716
loss 101.36926
STEP 821 ================================
prereg loss 99.76172 regularization 711.0582 reg_novel 196.78447
loss 100.66956
STEP 822 ================================
prereg loss 99.08831 regularization 711.1018 reg_novel 196.78246
loss 99.99619
STEP 823 ================================
prereg loss 98.522995 regularization 711.147 reg_novel 196.7841
loss 99.43092
STEP 824 ================================
prereg loss 98.195465 regularization 711.1903 reg_novel 196.78978
loss 99.10345
STEP 825 ================================
prereg loss 98.088585 regularization 711.2327 reg_novel 196.79964
loss 98.99662
STEP 826 ================================
prereg loss 97.86805 regularization 711.2756 reg_novel 196.81319
loss 98.77614
STEP 827 ================================
prereg loss 97.62407 regularization 711.31836 reg_novel 196.83087
loss 98.53222
STEP 828 ================================
prereg loss 97.441696 regularization 711.36285 reg_novel 196.85225
loss 98.349915
STEP 829 ================================
prereg loss 97.43866 regularization 711.40735 reg_novel 196.8746
loss 98.34694
STEP 830 ================================
prereg loss 97.6237 regularization 711.4543 reg_novel 196.89796
loss 98.53206
STEP 831 ================================
prereg loss 97.954636 regularization 711.5001 reg_novel 196.92107
loss 98.86306
STEP 832 ================================
prereg loss 98.294075 regularization 711.5467 reg_novel 196.94261
loss 99.20257
STEP 833 ================================
prereg loss 98.60468 regularization 711.593 reg_novel 196.96053
loss 99.51324
STEP 834 ================================
prereg loss 98.77539 regularization 711.64014 reg_novel 196.97522
loss 99.684006
STEP 835 ================================
prereg loss 98.83443 regularization 711.6893 reg_novel 196.98843
loss 99.7431
STEP 836 ================================
prereg loss 98.8163 regularization 711.73676 reg_novel 196.99994
loss 99.72504
STEP 837 ================================
prereg loss 98.74055 regularization 711.78265 reg_novel 197.00957
loss 99.64934
STEP 838 ================================
prereg loss 98.54704 regularization 711.82947 reg_novel 197.01587
loss 99.45589
STEP 839 ================================
prereg loss 98.24082 regularization 711.877 reg_novel 197.01924
loss 99.14972
STEP 840 ================================
prereg loss 97.90901 regularization 711.92487 reg_novel 197.01988
loss 98.817955
STEP 841 ================================
prereg loss 97.63004 regularization 711.97125 reg_novel 197.01733
loss 98.53903
STEP 842 ================================
prereg loss 97.39569 regularization 712.0199 reg_novel 197.01288
loss 98.304726
STEP 843 ================================
prereg loss 97.201485 regularization 712.07056 reg_novel 197.0082
loss 98.110565
STEP 844 ================================
prereg loss 97.05274 regularization 712.1193 reg_novel 197.00299
loss 97.96186
STEP 845 ================================
prereg loss 96.967445 regularization 712.16656 reg_novel 196.99805
loss 97.87661
STEP 846 ================================
prereg loss 96.97327 regularization 712.21295 reg_novel 196.99318
loss 97.88247
STEP 847 ================================
prereg loss 97.020744 regularization 712.2587 reg_novel 196.9874
loss 97.92999
STEP 848 ================================
prereg loss 97.05395 regularization 712.30237 reg_novel 196.98015
loss 97.96323
STEP 849 ================================
prereg loss 97.043564 regularization 712.34674 reg_novel 196.97183
loss 97.95288
STEP 850 ================================
prereg loss 96.96654 regularization 712.38983 reg_novel 196.96219
loss 97.87589
STEP 851 ================================
prereg loss 96.81679 regularization 712.4339 reg_novel 196.94861
loss 97.72617
STEP 852 ================================
prereg loss 96.609726 regularization 712.4796 reg_novel 196.93318
loss 97.51914
STEP 853 ================================
prereg loss 96.39067 regularization 712.5249 reg_novel 196.91484
loss 97.30011
STEP 854 ================================
prereg loss 96.08809 regularization 712.5717 reg_novel 196.89365
loss 96.99755
STEP 855 ================================
prereg loss 95.694176 regularization 712.6189 reg_novel 196.87039
loss 96.60367
STEP 856 ================================
prereg loss 95.20725 regularization 712.66785 reg_novel 196.8444
loss 96.11677
STEP 857 ================================
prereg loss 94.65731 regularization 712.7185 reg_novel 196.81706
loss 95.56685
STEP 858 ================================
prereg loss 94.100006 regularization 712.7671 reg_novel 196.78882
loss 95.00956
STEP 859 ================================
prereg loss 93.57533 regularization 712.8152 reg_novel 196.76053
loss 94.48491
STEP 860 ================================
prereg loss 93.11476 regularization 712.86145 reg_novel 196.73213
loss 94.02435
STEP 861 ================================
prereg loss 92.74062 regularization 712.9107 reg_novel 196.70367
loss 93.65024
STEP 862 ================================
prereg loss 92.4941 regularization 712.9601 reg_novel 196.67476
loss 93.40374
STEP 863 ================================
prereg loss 92.30191 regularization 713.0082 reg_novel 196.64449
loss 93.21156
STEP 864 ================================
prereg loss 92.15429 regularization 713.05865 reg_novel 196.6142
loss 93.063965
STEP 865 ================================
prereg loss 92.00691 regularization 713.10974 reg_novel 196.582
loss 92.9166
STEP 866 ================================
prereg loss 91.85625 regularization 713.1601 reg_novel 196.54863
loss 92.76595
STEP 867 ================================
prereg loss 91.80905 regularization 713.21094 reg_novel 196.51581
loss 92.71878
STEP 868 ================================
prereg loss 91.886566 regularization 713.2623 reg_novel 196.48222
loss 92.79631
STEP 869 ================================
prereg loss 92.01463 regularization 713.3151 reg_novel 196.4468
loss 92.92439
STEP 870 ================================
prereg loss 92.04218 regularization 713.3663 reg_novel 196.4046
loss 92.95195
STEP 871 ================================
prereg loss 92.04071 regularization 713.4184 reg_novel 196.35681
loss 92.950485
STEP 872 ================================
prereg loss 92.12212 regularization 713.4754 reg_novel 196.30357
loss 93.031906
STEP 873 ================================
prereg loss 92.14389 regularization 713.53156 reg_novel 196.246
loss 93.053665
STEP 874 ================================
prereg loss 91.98517 regularization 713.58966 reg_novel 196.18698
loss 92.89494
STEP 875 ================================
prereg loss 91.97882 regularization 713.6499 reg_novel 196.1247
loss 92.888596
STEP 876 ================================
prereg loss 91.85131 regularization 713.71094 reg_novel 196.05858
loss 92.76108
STEP 877 ================================
prereg loss 91.340225 regularization 713.7735 reg_novel 195.98868
loss 92.249985
STEP 878 ================================
prereg loss 90.636505 regularization 713.83673 reg_novel 195.91641
loss 91.54626
STEP 879 ================================
prereg loss 90.216385 regularization 713.9006 reg_novel 195.84346
loss 91.12613
STEP 880 ================================
prereg loss 90.17522 regularization 713.967 reg_novel 195.77318
loss 91.08495
STEP 881 ================================
prereg loss 90.241745 regularization 714.03326 reg_novel 195.70709
loss 91.15148
STEP 882 ================================
prereg loss 90.39923 regularization 714.10077 reg_novel 195.64413
loss 91.308975
STEP 883 ================================
prereg loss 90.87337 regularization 714.17163 reg_novel 195.58229
loss 91.78312
STEP 884 ================================
prereg loss 91.54982 regularization 714.241 reg_novel 195.51744
loss 92.45958
STEP 885 ================================
prereg loss 92.13181 regularization 714.3124 reg_novel 195.4481
loss 93.04157
STEP 886 ================================
prereg loss 92.217926 regularization 714.3833 reg_novel 195.37256
loss 93.127686
STEP 887 ================================
prereg loss 91.991234 regularization 714.456 reg_novel 195.28581
loss 92.90098
STEP 888 ================================
prereg loss 91.84694 regularization 714.52826 reg_novel 195.19171
loss 92.75666
STEP 889 ================================
prereg loss 91.76833 regularization 714.5999 reg_novel 195.0929
loss 92.678024
STEP 890 ================================
prereg loss 91.80602 regularization 714.66986 reg_novel 194.99136
loss 92.71568
STEP 891 ================================
prereg loss 91.89671 regularization 714.7411 reg_novel 194.88748
loss 92.80634
STEP 892 ================================
prereg loss 92.08227 regularization 714.81256 reg_novel 194.7831
loss 92.99187
STEP 893 ================================
prereg loss 92.37919 regularization 714.8816 reg_novel 194.67787
loss 93.28875
STEP 894 ================================
prereg loss 92.76887 regularization 714.95 reg_novel 194.57124
loss 93.67839
STEP 895 ================================
prereg loss 93.25834 regularization 715.0179 reg_novel 194.46315
loss 94.167816
STEP 896 ================================
prereg loss 93.804596 regularization 715.0833 reg_novel 194.35332
loss 94.714035
STEP 897 ================================
prereg loss 94.23067 regularization 715.1506 reg_novel 194.2425
loss 95.14006
STEP 898 ================================
prereg loss 94.569405 regularization 715.21936 reg_novel 194.13153
loss 95.47875
STEP 899 ================================
prereg loss 94.8787 regularization 715.2872 reg_novel 194.01814
loss 95.788
STEP 900 ================================
prereg loss 95.14689 regularization 715.3557 reg_novel 193.90175
loss 96.056145
STEP 901 ================================
prereg loss 95.35814 regularization 715.4253 reg_novel 193.78255
loss 96.26735
STEP 902 ================================
prereg loss 95.565926 regularization 715.49335 reg_novel 193.66164
loss 96.47508
STEP 903 ================================
prereg loss 95.65233 regularization 715.5617 reg_novel 193.53616
loss 96.561424
STEP 904 ================================
prereg loss 95.62793 regularization 715.6289 reg_novel 193.40903
loss 96.536964
STEP 905 ================================
prereg loss 95.550476 regularization 715.6954 reg_novel 193.28044
loss 96.45945
STEP 906 ================================
prereg loss 95.50143 regularization 715.76294 reg_novel 193.14975
loss 96.41034
STEP 907 ================================
prereg loss 95.54879 regularization 715.8294 reg_novel 193.01819
loss 96.457634
STEP 908 ================================
prereg loss 95.69966 regularization 715.8937 reg_novel 192.88815
loss 96.608444
STEP 909 ================================
prereg loss 95.857056 regularization 715.9573 reg_novel 192.75977
loss 96.76577
STEP 910 ================================
prereg loss 95.964836 regularization 716.0221 reg_novel 192.63313
loss 96.87349
STEP 911 ================================
prereg loss 95.94059 regularization 716.08734 reg_novel 192.50583
loss 96.84918
STEP 912 ================================
prereg loss 95.92628 regularization 716.15045 reg_novel 192.37735
loss 96.83481
STEP 913 ================================
prereg loss 95.843636 regularization 716.21454 reg_novel 192.2457
loss 96.7521
STEP 914 ================================
prereg loss 95.619576 regularization 716.279 reg_novel 192.11081
loss 96.52796
STEP 915 ================================
prereg loss 95.30619 regularization 716.3458 reg_novel 191.97375
loss 96.21451
STEP 916 ================================
prereg loss 95.33027 regularization 716.41016 reg_novel 191.83739
loss 96.23852
STEP 917 ================================
prereg loss 95.300896 regularization 716.4734 reg_novel 191.70374
loss 96.209076
STEP 918 ================================
prereg loss 95.19655 regularization 716.5375 reg_novel 191.57512
loss 96.10466
STEP 919 ================================
prereg loss 95.15589 regularization 716.60364 reg_novel 191.45053
loss 96.06394
STEP 920 ================================
prereg loss 95.2458 regularization 716.6687 reg_novel 191.33049
loss 96.15379
STEP 921 ================================
prereg loss 95.45105 regularization 716.7345 reg_novel 191.21645
loss 96.359
STEP 922 ================================
prereg loss 95.411545 regularization 716.8014 reg_novel 191.10753
loss 96.31945
STEP 923 ================================
prereg loss 95.42519 regularization 716.86865 reg_novel 191.00569
loss 96.33307
STEP 924 ================================
prereg loss 95.3806 regularization 716.9366 reg_novel 190.90883
loss 96.288445
STEP 925 ================================
prereg loss 95.24798 regularization 717.00714 reg_novel 190.81465
loss 96.1558
STEP 926 ================================
prereg loss 94.98786 regularization 717.0754 reg_novel 190.7204
loss 95.89566
STEP 927 ================================
prereg loss 94.699326 regularization 717.146 reg_novel 190.62488
loss 95.60709
STEP 928 ================================
prereg loss 94.31749 regularization 717.2152 reg_novel 190.52502
loss 95.22523
STEP 929 ================================
prereg loss 93.79931 regularization 717.2838 reg_novel 190.4208
loss 94.707016
STEP 930 ================================
prereg loss 93.15022 regularization 717.35236 reg_novel 190.31415
loss 94.05789
STEP 931 ================================
prereg loss 92.35321 regularization 717.41724 reg_novel 190.20496
loss 93.26083
STEP 932 ================================
prereg loss 91.43238 regularization 717.4839 reg_novel 190.09326
loss 92.33996
STEP 933 ================================
prereg loss 90.4114 regularization 717.5482 reg_novel 189.97894
loss 91.318924
STEP 934 ================================
prereg loss 89.33505 regularization 717.61206 reg_novel 189.86136
loss 90.24252
STEP 935 ================================
prereg loss 88.10507 regularization 717.67426 reg_novel 189.74193
loss 89.01249
STEP 936 ================================
prereg loss 86.80243 regularization 717.73816 reg_novel 189.6218
loss 87.709785
STEP 937 ================================
prereg loss 85.52352 regularization 717.80005 reg_novel 189.50208
loss 86.430824
STEP 938 ================================
prereg loss 84.27701 regularization 717.862 reg_novel 189.38489
loss 85.18426
STEP 939 ================================
prereg loss 83.09656 regularization 717.923 reg_novel 189.2699
loss 84.00375
STEP 940 ================================
prereg loss 82.01989 regularization 717.98505 reg_novel 189.15558
loss 82.92703
STEP 941 ================================
prereg loss 81.125275 regularization 718.04706 reg_novel 189.04039
loss 82.032364
STEP 942 ================================
prereg loss 80.269585 regularization 718.1087 reg_novel 188.9219
loss 81.17661
STEP 943 ================================
prereg loss 79.36523 regularization 718.16974 reg_novel 188.79689
loss 80.272194
STEP 944 ================================
prereg loss 78.37682 regularization 718.233 reg_novel 188.66734
loss 79.28372
STEP 945 ================================
prereg loss 77.30482 regularization 718.298 reg_novel 188.53214
loss 78.21165
STEP 946 ================================
prereg loss 76.14973 regularization 718.36127 reg_novel 188.38779
loss 77.05647
STEP 947 ================================
prereg loss 74.9382 regularization 718.42566 reg_novel 188.23679
loss 75.844864
STEP 948 ================================
prereg loss 73.666435 regularization 718.489 reg_novel 188.07999
loss 74.573006
STEP 949 ================================
prereg loss 72.4155 regularization 718.5516 reg_novel 187.91983
loss 73.32197
STEP 950 ================================
prereg loss 71.19747 regularization 718.61444 reg_novel 187.75922
loss 72.10384
STEP 951 ================================
prereg loss 69.997025 regularization 718.6788 reg_novel 187.59889
loss 70.903305
STEP 952 ================================
prereg loss 68.87998 regularization 718.7437 reg_novel 187.44037
loss 69.78616
STEP 953 ================================
prereg loss 67.85793 regularization 718.8061 reg_novel 187.28433
loss 68.76402
STEP 954 ================================
prereg loss 66.91241 regularization 718.867 reg_novel 187.13019
loss 67.818405
STEP 955 ================================
prereg loss 66.01792 regularization 718.92633 reg_novel 186.97823
loss 66.92383
STEP 956 ================================
prereg loss 65.13678 regularization 718.98517 reg_novel 186.8285
loss 66.042595
STEP 957 ================================
prereg loss 64.251495 regularization 719.0411 reg_novel 186.67996
loss 65.15722
STEP 958 ================================
prereg loss 63.34555 regularization 719.0958 reg_novel 186.53217
loss 64.251175
STEP 959 ================================
prereg loss 62.438114 regularization 719.1488 reg_novel 186.38481
loss 63.343647
STEP 960 ================================
prereg loss 61.54443 regularization 719.20044 reg_novel 186.2395
loss 62.44987
STEP 961 ================================
prereg loss 60.653748 regularization 719.2504 reg_novel 186.09662
loss 61.559093
STEP 962 ================================
prereg loss 59.769142 regularization 719.30096 reg_novel 185.95743
loss 60.6744
STEP 963 ================================
prereg loss 58.9146 regularization 719.35 reg_novel 185.81886
loss 59.81977
STEP 964 ================================
prereg loss 58.11633 regularization 719.3977 reg_novel 185.67981
loss 59.021408
STEP 965 ================================
prereg loss 57.30448 regularization 719.4449 reg_novel 185.5385
loss 58.209465
STEP 966 ================================
prereg loss 56.502186 regularization 719.4897 reg_novel 185.39621
loss 57.40707
STEP 967 ================================
prereg loss 55.740013 regularization 719.53705 reg_novel 185.25615
loss 56.644806
STEP 968 ================================
prereg loss 55.07398 regularization 719.5841 reg_novel 185.1214
loss 55.978683
STEP 969 ================================
prereg loss 54.523067 regularization 719.62744 reg_novel 184.98674
loss 55.42768
STEP 970 ================================
prereg loss 54.079926 regularization 719.66925 reg_novel 184.85107
loss 54.984447
STEP 971 ================================
prereg loss 53.69834 regularization 719.71124 reg_novel 184.7157
loss 54.602768
STEP 972 ================================
prereg loss 53.35338 regularization 719.7554 reg_novel 184.5825
loss 54.257717
STEP 973 ================================
prereg loss 53.051086 regularization 719.8013 reg_novel 184.4521
loss 53.95534
STEP 974 ================================
prereg loss 52.78 regularization 719.8476 reg_novel 184.32733
loss 53.684174
STEP 975 ================================
prereg loss 52.504044 regularization 719.89636 reg_novel 184.20914
loss 53.40815
STEP 976 ================================
prereg loss 52.193382 regularization 719.94806 reg_novel 184.09763
loss 53.097427
STEP 977 ================================
prereg loss 51.763367 regularization 720.0025 reg_novel 183.98935
loss 52.66736
STEP 978 ================================
prereg loss 51.21959 regularization 720.0601 reg_novel 183.88324
loss 52.12353
STEP 979 ================================
prereg loss 50.56815 regularization 720.1183 reg_novel 183.77994
loss 51.47205
STEP 980 ================================
prereg loss 49.820698 regularization 720.1777 reg_novel 183.67792
loss 50.724552
STEP 981 ================================
prereg loss 49.061684 regularization 720.2382 reg_novel 183.57794
loss 49.9655
STEP 982 ================================
prereg loss 47.896618 regularization 720.29 reg_novel 183.45728
loss 48.800365
STEP 983 ================================
prereg loss 46.83553 regularization 720.3428 reg_novel 183.33914
loss 47.739212
STEP 984 ================================
prereg loss 45.921535 regularization 720.3965 reg_novel 183.22423
loss 46.825157
STEP 985 ================================
prereg loss 45.443317 regularization 720.4496 reg_novel 183.12318
loss 46.34689
STEP 986 ================================
prereg loss 45.17478 regularization 720.5028 reg_novel 183.03969
loss 46.078323
STEP 987 ================================
prereg loss 44.835083 regularization 720.55664 reg_novel 182.97308
loss 45.738613
STEP 988 ================================
prereg loss 44.742786 regularization 720.6102 reg_novel 182.91759
loss 45.646313
STEP 989 ================================
prereg loss 45.13508 regularization 720.6632 reg_novel 182.86346
loss 46.038605
STEP 990 ================================
prereg loss 45.737053 regularization 720.71826 reg_novel 182.80252
loss 46.640575
STEP 991 ================================
prereg loss 45.85763 regularization 720.77435 reg_novel 182.7284
loss 46.761135
STEP 992 ================================
prereg loss 45.016308 regularization 720.829 reg_novel 182.6407
loss 45.919777
STEP 993 ================================
prereg loss 43.67025 regularization 720.882 reg_novel 182.54333
loss 44.573677
STEP 994 ================================
prereg loss 42.487103 regularization 720.94104 reg_novel 182.44318
loss 43.390488
STEP 995 ================================
prereg loss 41.607933 regularization 720.99976 reg_novel 182.34906
loss 42.51128
STEP 996 ================================
prereg loss 40.81072 regularization 721.06055 reg_novel 182.26326
loss 41.714043
STEP 997 ================================
prereg loss 40.121174 regularization 721.1216 reg_novel 182.1864
loss 41.024483
STEP 998 ================================
prereg loss 39.49436 regularization 721.1842 reg_novel 182.11583
loss 40.397663
STEP 999 ================================
prereg loss 38.782757 regularization 721.25 reg_novel 182.04933
loss 39.686058
STEP 1000 ================================
prereg loss 38.10218 regularization 721.31573 reg_novel 181.98831
loss 39.005486
2022-06-14T11:48:46.252

julia> steps!(100)
2022-06-14T12:01:18.657
STEP 1 ================================
prereg loss 37.636208 regularization 721.38434 reg_novel 181.93529
loss 38.539528
STEP 2 ================================
prereg loss 37.560684 regularization 721.4554 reg_novel 181.89182
loss 38.46403
STEP 3 ================================
prereg loss 36.970264 regularization 721.5213 reg_novel 181.84026
loss 37.873627
STEP 4 ================================
prereg loss 37.06711 regularization 721.5857 reg_novel 181.78442
loss 37.970478
STEP 5 ================================
prereg loss 36.70422 regularization 721.65137 reg_novel 181.72237
loss 37.607594
STEP 6 ================================
prereg loss 35.69374 regularization 721.7161 reg_novel 181.6535
loss 36.59711
STEP 7 ================================
prereg loss 34.86948 regularization 721.78156 reg_novel 181.58382
loss 35.772846
STEP 8 ================================
prereg loss 34.411987 regularization 721.84595 reg_novel 181.52
loss 35.315353
STEP 9 ================================
prereg loss 33.974342 regularization 721.9108 reg_novel 181.46777
loss 34.87772
STEP 10 ================================
prereg loss 33.55681 regularization 721.9764 reg_novel 181.42519
loss 34.46021
STEP 11 ================================
prereg loss 33.449818 regularization 722.0421 reg_novel 181.38629
loss 34.353245
STEP 12 ================================
prereg loss 33.398144 regularization 722.1085 reg_novel 181.34325
loss 34.301594
STEP 13 ================================
prereg loss 32.992245 regularization 722.177 reg_novel 181.29486
loss 33.895718
STEP 14 ================================
prereg loss 32.431313 regularization 722.24786 reg_novel 181.24207
loss 33.3348
STEP 15 ================================
prereg loss 32.053963 regularization 722.3202 reg_novel 181.19029
loss 32.957474
STEP 16 ================================
prereg loss 31.683086 regularization 722.39307 reg_novel 181.14345
loss 32.586624
STEP 17 ================================
prereg loss 31.337759 regularization 722.46466 reg_novel 181.10152
loss 32.241325
STEP 18 ================================
prereg loss 31.241554 regularization 722.53546 reg_novel 181.07181
loss 32.14516
STEP 19 ================================
prereg loss 31.235489 regularization 722.6037 reg_novel 181.04803
loss 32.13914
STEP 20 ================================
prereg loss 30.81394 regularization 722.671 reg_novel 181.0272
loss 31.717638
STEP 21 ================================
prereg loss 30.421223 regularization 722.7376 reg_novel 181.01103
loss 31.324972
STEP 22 ================================
prereg loss 30.475388 regularization 722.8031 reg_novel 181.00291
loss 31.379194
STEP 23 ================================
prereg loss 30.148613 regularization 722.8658 reg_novel 181.0023
loss 31.05248
STEP 24 ================================
prereg loss 29.944632 regularization 722.92773 reg_novel 181.00713
loss 30.848566
STEP 25 ================================
prereg loss 30.093756 regularization 722.9885 reg_novel 181.01279
loss 30.997757
STEP 26 ================================
prereg loss 29.809067 regularization 723.04944 reg_novel 181.01381
loss 30.71313
STEP 27 ================================
prereg loss 28.90785 regularization 723.1111 reg_novel 181.00073
loss 29.811962
STEP 28 ================================
prereg loss 28.156069 regularization 723.17395 reg_novel 180.9792
loss 29.060223
STEP 29 ================================
prereg loss 27.883404 regularization 723.236 reg_novel 180.95732
loss 28.787598
STEP 30 ================================
prereg loss 27.65033 regularization 723.2977 reg_novel 180.94107
loss 28.55457
STEP 31 ================================
prereg loss 27.276293 regularization 723.36115 reg_novel 180.9319
loss 28.180586
STEP 32 ================================
prereg loss 27.07118 regularization 723.425 reg_novel 180.92474
loss 27.97553
STEP 33 ================================
prereg loss 27.065817 regularization 723.4876 reg_novel 180.9135
loss 27.970219
STEP 34 ================================
prereg loss 26.932367 regularization 723.5495 reg_novel 180.89427
loss 27.836811
STEP 35 ================================
prereg loss 26.64119 regularization 723.6091 reg_novel 180.866
loss 27.545664
STEP 36 ================================
prereg loss 26.340147 regularization 723.66925 reg_novel 180.82593
loss 27.244642
STEP 37 ================================
prereg loss 26.084923 regularization 723.7254 reg_novel 180.77849
loss 26.989428
STEP 38 ================================
prereg loss 25.895609 regularization 723.78284 reg_novel 180.74023
loss 26.800133
STEP 39 ================================
prereg loss 25.733067 regularization 723.8386 reg_novel 180.71072
loss 26.637615
STEP 40 ================================
prereg loss 25.57995 regularization 723.89355 reg_novel 180.69106
loss 26.484535
STEP 41 ================================
prereg loss 25.429596 regularization 723.94604 reg_novel 180.6795
loss 26.33422
STEP 42 ================================
prereg loss 25.28673 regularization 723.9992 reg_novel 180.67395
loss 26.191402
STEP 43 ================================
prereg loss 25.151161 regularization 724.0516 reg_novel 180.67348
loss 26.055885
STEP 44 ================================
prereg loss 25.056713 regularization 724.1022 reg_novel 180.67598
loss 25.96149
STEP 45 ================================
prereg loss 24.887133 regularization 724.1505 reg_novel 180.66277
loss 25.791946
STEP 46 ================================
prereg loss 24.68795 regularization 724.199 reg_novel 180.6423
loss 25.59279
STEP 47 ================================
prereg loss 24.429224 regularization 724.2481 reg_novel 180.6158
loss 25.334087
STEP 48 ================================
prereg loss 24.24748 regularization 724.2977 reg_novel 180.58707
loss 25.152365
STEP 49 ================================
prereg loss 24.153437 regularization 724.34814 reg_novel 180.55759
loss 25.058342
STEP 50 ================================
prereg loss 24.042852 regularization 724.399 reg_novel 180.5382
loss 24.94779
STEP 51 ================================
prereg loss 23.920553 regularization 724.4488 reg_novel 180.52771
loss 24.82553
STEP 52 ================================
prereg loss 23.876526 regularization 724.4963 reg_novel 180.51022
loss 24.781532
STEP 53 ================================
prereg loss 23.888498 regularization 724.544 reg_novel 180.48734
loss 24.79353
STEP 54 ================================
prereg loss 23.796267 regularization 724.5905 reg_novel 180.45369
loss 24.701311
STEP 55 ================================
prereg loss 23.616468 regularization 724.6339 reg_novel 180.42287
loss 24.521524
STEP 56 ================================
prereg loss 23.521769 regularization 724.6785 reg_novel 180.39763
loss 24.426846
STEP 57 ================================
prereg loss 23.495024 regularization 724.72345 reg_novel 180.38022
loss 24.400127
STEP 58 ================================
prereg loss 23.41806 regularization 724.7687 reg_novel 180.37169
loss 24.3232
STEP 59 ================================
prereg loss 23.32657 regularization 724.8113 reg_novel 180.36453
loss 24.231747
STEP 60 ================================
prereg loss 23.394255 regularization 724.856 reg_novel 180.36017
loss 24.29947
STEP 61 ================================
prereg loss 23.324604 regularization 724.9003 reg_novel 180.34691
loss 24.22985
STEP 62 ================================
prereg loss 23.126167 regularization 724.9441 reg_novel 180.3257
loss 24.031437
STEP 63 ================================
prereg loss 23.043476 regularization 724.9888 reg_novel 180.29941
loss 23.948765
STEP 64 ================================
prereg loss 23.058249 regularization 725.032 reg_novel 180.28053
loss 23.963562
STEP 65 ================================
prereg loss 23.004871 regularization 725.0732 reg_novel 180.27036
loss 23.910215
STEP 66 ================================
prereg loss 22.973671 regularization 725.11615 reg_novel 180.26556
loss 23.879053
STEP 67 ================================
prereg loss 23.043491 regularization 725.1558 reg_novel 180.25034
loss 23.948898
STEP 68 ================================
prereg loss 23.12291 regularization 725.1946 reg_novel 180.22385
loss 24.028328
STEP 69 ================================
prereg loss 22.985842 regularization 725.23334 reg_novel 180.19495
loss 23.89127
STEP 70 ================================
prereg loss 22.936798 regularization 725.2714 reg_novel 180.16687
loss 23.842236
STEP 71 ================================
prereg loss 23.110863 regularization 725.31177 reg_novel 180.14572
loss 24.016321
STEP 72 ================================
prereg loss 23.17023 regularization 725.35175 reg_novel 180.13535
loss 24.075718
STEP 73 ================================
prereg loss 23.056086 regularization 725.3922 reg_novel 180.13513
loss 23.961613
STEP 74 ================================
prereg loss 23.092213 regularization 725.4357 reg_novel 180.1407
loss 23.99779
STEP 75 ================================
prereg loss 23.2845 regularization 725.4776 reg_novel 180.13881
loss 24.190117
STEP 76 ================================
prereg loss 23.329552 regularization 725.5192 reg_novel 180.125
loss 24.235195
STEP 77 ================================
prereg loss 23.240368 regularization 725.55853 reg_novel 180.09492
loss 24.14602
STEP 78 ================================
prereg loss 23.109142 regularization 725.5977 reg_novel 180.05687
loss 24.014797
STEP 79 ================================
prereg loss 23.01905 regularization 725.63934 reg_novel 180.02492
loss 23.924715
STEP 80 ================================
prereg loss 22.98409 regularization 725.6808 reg_novel 180.00064
loss 23.889772
STEP 81 ================================
prereg loss 22.981035 regularization 725.72156 reg_novel 179.9847
loss 23.886742
STEP 82 ================================
prereg loss 22.999313 regularization 725.7605 reg_novel 179.97523
loss 23.905048
STEP 83 ================================
prereg loss 23.034653 regularization 725.8 reg_novel 179.97116
loss 23.940424
STEP 84 ================================
prereg loss 23.076723 regularization 725.8389 reg_novel 179.97057
loss 23.982533
STEP 85 ================================
prereg loss 23.106297 regularization 725.87665 reg_novel 179.9684
loss 24.012142
STEP 86 ================================
prereg loss 23.10031 regularization 725.91254 reg_novel 179.9632
loss 24.006186
STEP 87 ================================
prereg loss 23.027025 regularization 725.94946 reg_novel 179.95808
loss 23.932934
STEP 88 ================================
prereg loss 22.987556 regularization 725.9867 reg_novel 179.95433
loss 23.893497
STEP 89 ================================
prereg loss 22.997366 regularization 726.02277 reg_novel 179.95496
loss 23.903343
STEP 90 ================================
prereg loss 22.982147 regularization 726.0615 reg_novel 179.95175
loss 23.88816
STEP 91 ================================
prereg loss 22.953077 regularization 726.09937 reg_novel 179.94505
loss 23.859121
STEP 92 ================================
prereg loss 22.960876 regularization 726.13837 reg_novel 179.93379
loss 23.86695
STEP 93 ================================
prereg loss 22.994476 regularization 726.1766 reg_novel 179.9169
loss 23.90057
STEP 94 ================================
prereg loss 23.03079 regularization 726.21375 reg_novel 179.89078
loss 23.936895
STEP 95 ================================
prereg loss 23.06513 regularization 726.25195 reg_novel 179.86673
loss 23.971249
STEP 96 ================================
prereg loss 23.140757 regularization 726.289 reg_novel 179.84515
loss 24.04689
STEP 97 ================================
prereg loss 23.308338 regularization 726.3238 reg_novel 179.8257
loss 24.214487
STEP 98 ================================
prereg loss 23.512669 regularization 726.35864 reg_novel 179.81076
loss 24.418839
STEP 99 ================================
prereg loss 23.535124 regularization 726.3927 reg_novel 179.7996
loss 24.441317
STEP 100 ================================
prereg loss 23.55284 regularization 726.4261 reg_novel 179.7911
loss 24.459057
2022-06-14T13:24:33.678

julia> steps!(368)
2022-06-14T15:29:05.476
STEP 1 ================================
prereg loss 23.653376 regularization 726.45795 reg_novel 179.78252
loss 24.559616
STEP 2 ================================
prereg loss 23.71174 regularization 726.49005 reg_novel 179.7724
loss 24.618004
STEP 3 ================================
prereg loss 23.766703 regularization 726.52216 reg_novel 179.76213
loss 24.672987
STEP 4 ================================
prereg loss 23.90742 regularization 726.5564 reg_novel 179.75417
loss 24.81373
STEP 5 ================================
prereg loss 23.979818 regularization 726.5908 reg_novel 179.75122
loss 24.88616
STEP 6 ================================
prereg loss 23.958668 regularization 726.62524 reg_novel 179.7495
loss 24.865042
STEP 7 ================================
prereg loss 24.0733 regularization 726.66174 reg_novel 179.73914
loss 24.9797
STEP 8 ================================
prereg loss 24.138786 regularization 726.6967 reg_novel 179.71867
loss 25.045202
STEP 9 ================================
prereg loss 24.20119 regularization 726.73267 reg_novel 179.68843
loss 25.107613
STEP 10 ================================
prereg loss 24.329144 regularization 726.76825 reg_novel 179.66064
loss 25.235573
STEP 11 ================================
prereg loss 24.422222 regularization 726.8018 reg_novel 179.6376
loss 25.328661
STEP 12 ================================
prereg loss 24.523224 regularization 726.83527 reg_novel 179.61583
loss 25.429674
STEP 13 ================================
prereg loss 24.612694 regularization 726.86835 reg_novel 179.59192
loss 25.519154
STEP 14 ================================
prereg loss 24.70292 regularization 726.90173 reg_novel 179.56496
loss 25.609388
STEP 15 ================================
prereg loss 24.783789 regularization 726.9332 reg_novel 179.53545
loss 25.690258
STEP 16 ================================
prereg loss 24.913124 regularization 726.9637 reg_novel 179.5066
loss 25.819595
STEP 17 ================================
prereg loss 25.109535 regularization 726.992 reg_novel 179.48232
loss 26.01601
STEP 18 ================================
prereg loss 25.23627 regularization 727.02136 reg_novel 179.46439
loss 26.142757
STEP 19 ================================
prereg loss 25.175356 regularization 727.05054 reg_novel 179.45486
loss 26.081861
STEP 20 ================================
prereg loss 25.184984 regularization 727.0801 reg_novel 179.44847
loss 26.091513
STEP 21 ================================
prereg loss 25.298967 regularization 727.10913 reg_novel 179.44112
loss 26.205517
STEP 22 ================================
prereg loss 25.404583 regularization 727.1348 reg_novel 179.42775
loss 26.311146
STEP 23 ================================
prereg loss 25.486187 regularization 727.1618 reg_novel 179.40862
loss 26.392757
STEP 24 ================================
prereg loss 25.583475 regularization 727.18823 reg_novel 179.38495
loss 26.490047
STEP 25 ================================
prereg loss 25.722792 regularization 727.21576 reg_novel 179.35912
loss 26.629366
STEP 26 ================================
prereg loss 25.80823 regularization 727.24335 reg_novel 179.33469
loss 26.714808
STEP 27 ================================
prereg loss 25.887238 regularization 727.2694 reg_novel 179.311
loss 26.793818
STEP 28 ================================
prereg loss 25.978407 regularization 727.2969 reg_novel 179.28722
loss 26.88499
STEP 29 ================================
prereg loss 26.074196 regularization 727.32104 reg_novel 179.26302
loss 26.98078
STEP 30 ================================
prereg loss 26.180645 regularization 727.34656 reg_novel 179.23825
loss 27.08723
STEP 31 ================================
prereg loss 26.272593 regularization 727.37244 reg_novel 179.21533
loss 27.17918
STEP 32 ================================
prereg loss 26.383009 regularization 727.3997 reg_novel 179.19344
loss 27.289602
STEP 33 ================================
prereg loss 26.455593 regularization 727.4266 reg_novel 179.17096
loss 27.36219
STEP 34 ================================
prereg loss 26.529194 regularization 727.4512 reg_novel 179.14737
loss 27.435793
STEP 35 ================================
prereg loss 26.629011 regularization 727.47406 reg_novel 179.12256
loss 27.535608
STEP 36 ================================
prereg loss 26.741144 regularization 727.4975 reg_novel 179.09691
loss 27.64774
STEP 37 ================================
prereg loss 26.871523 regularization 727.51874 reg_novel 179.0712
loss 27.778112
STEP 38 ================================
prereg loss 26.987797 regularization 727.54285 reg_novel 179.04565
loss 27.894386
STEP 39 ================================
prereg loss 27.078506 regularization 727.5657 reg_novel 179.02031
loss 27.985092
STEP 40 ================================
prereg loss 27.167282 regularization 727.5889 reg_novel 178.9944
loss 28.073866
STEP 41 ================================
prereg loss 27.244732 regularization 727.6114 reg_novel 178.96738
loss 28.15131
STEP 42 ================================
prereg loss 27.320936 regularization 727.63385 reg_novel 178.93846
loss 28.227509
STEP 43 ================================
prereg loss 27.438822 regularization 727.6558 reg_novel 178.90869
loss 28.345387
STEP 44 ================================
prereg loss 27.577902 regularization 727.6791 reg_novel 178.88129
loss 28.484463
STEP 45 ================================
prereg loss 27.655394 regularization 727.69977 reg_novel 178.85796
loss 28.56195
STEP 46 ================================
prereg loss 27.71387 regularization 727.71985 reg_novel 178.8378
loss 28.620426
STEP 47 ================================
prereg loss 27.800222 regularization 727.73987 reg_novel 178.82004
loss 28.706783
STEP 48 ================================
prereg loss 27.911623 regularization 727.7607 reg_novel 178.80222
loss 28.818186
STEP 49 ================================
prereg loss 28.03457 regularization 727.7804 reg_novel 178.78252
loss 28.941133
STEP 50 ================================
prereg loss 28.155546 regularization 727.80023 reg_novel 178.7607
loss 29.062107
STEP 51 ================================
prereg loss 28.26685 regularization 727.82104 reg_novel 178.73772
loss 29.173409
STEP 52 ================================
prereg loss 28.347458 regularization 727.8395 reg_novel 178.71391
loss 29.254011
STEP 53 ================================
prereg loss 28.412083 regularization 727.857 reg_novel 178.68974
loss 29.31863
STEP 54 ================================
prereg loss 28.453804 regularization 727.8752 reg_novel 178.66422
loss 29.360344
STEP 55 ================================
prereg loss 28.473055 regularization 727.8912 reg_novel 178.63634
loss 29.379581
STEP 56 ================================
prereg loss 28.484148 regularization 727.9088 reg_novel 178.60715
loss 29.390663
STEP 57 ================================
prereg loss 28.527693 regularization 727.92487 reg_novel 178.57637
loss 29.434195
STEP 58 ================================
prereg loss 28.626947 regularization 727.9441 reg_novel 178.54596
loss 29.533438
STEP 59 ================================
prereg loss 28.722319 regularization 727.9627 reg_novel 178.51852
loss 29.6288
STEP 60 ================================
prereg loss 28.732456 regularization 727.9829 reg_novel 178.49382
loss 29.638933
STEP 61 ================================
prereg loss 28.716444 regularization 728.00134 reg_novel 178.47142
loss 29.622917
STEP 62 ================================
prereg loss 28.760832 regularization 728.0202 reg_novel 178.44843
loss 29.667301
STEP 63 ================================
prereg loss 28.799343 regularization 728.03986 reg_novel 178.42027
loss 29.705803
STEP 64 ================================
prereg loss 28.772911 regularization 728.05774 reg_novel 178.38585
loss 29.679356
STEP 65 ================================
prereg loss 28.751865 regularization 728.07733 reg_novel 178.34625
loss 29.658289
STEP 66 ================================
prereg loss 28.779379 regularization 728.0971 reg_novel 178.30574
loss 29.685781
STEP 67 ================================
prereg loss 28.781952 regularization 728.1161 reg_novel 178.26738
loss 29.688335
STEP 68 ================================
prereg loss 28.734562 regularization 728.136 reg_novel 178.23096
loss 29.640928
STEP 69 ================================
prereg loss 28.697493 regularization 728.1545 reg_novel 178.19528
loss 29.603842
STEP 70 ================================
prereg loss 28.691647 regularization 728.1732 reg_novel 178.15865
loss 29.597979
STEP 71 ================================
prereg loss 28.687485 regularization 728.19135 reg_novel 178.11722
loss 29.593794
STEP 72 ================================
prereg loss 28.668781 regularization 728.2091 reg_novel 178.07126
loss 29.575062
STEP 73 ================================
prereg loss 28.630552 regularization 728.2264 reg_novel 178.0187
loss 29.536797
STEP 74 ================================
prereg loss 28.58153 regularization 728.24286 reg_novel 177.95856
loss 29.487732
STEP 75 ================================
prereg loss 28.56728 regularization 728.2624 reg_novel 177.89252
loss 29.473434
STEP 76 ================================
prereg loss 28.738104 regularization 728.2806 reg_novel 177.827
loss 29.64421
STEP 77 ================================
prereg loss 29.371113 regularization 728.2969 reg_novel 177.77052
loss 30.27718
STEP 78 ================================
prereg loss 30.0432 regularization 728.3126 reg_novel 177.73183
loss 30.949244
STEP 79 ================================
prereg loss 29.7584 regularization 728.3303 reg_novel 177.71198
loss 30.664442
STEP 80 ================================
prereg loss 28.924753 regularization 728.34875 reg_novel 177.7043
loss 29.830807
STEP 81 ================================
prereg loss 29.157438 regularization 728.3682 reg_novel 177.69466
loss 30.063501
STEP 82 ================================
prereg loss 30.085903 regularization 728.3884 reg_novel 177.66821
loss 30.99196
STEP 83 ================================
prereg loss 29.806206 regularization 728.407 reg_novel 177.62083
loss 30.712234
STEP 84 ================================
prereg loss 29.066324 regularization 728.4261 reg_novel 177.5598
loss 29.972311
STEP 85 ================================
prereg loss 29.391222 regularization 728.44556 reg_novel 177.49695
loss 30.297165
STEP 86 ================================
prereg loss 29.519045 regularization 728.46515 reg_novel 177.44135
loss 30.424952
STEP 87 ================================
prereg loss 29.01776 regularization 728.48395 reg_novel 177.39088
loss 29.923634
STEP 88 ================================
prereg loss 29.186512 regularization 728.50366 reg_novel 177.33923
loss 30.092356
STEP 89 ================================
prereg loss 29.216223 regularization 728.52466 reg_novel 177.28001
loss 30.122028
STEP 90 ================================
prereg loss 29.03984 regularization 728.544 reg_novel 177.21219
loss 29.945597
STEP 91 ================================
prereg loss 29.537785 regularization 728.5636 reg_novel 177.14267
loss 30.443491
STEP 92 ================================
prereg loss 30.087059 regularization 728.5831 reg_novel 177.08304
loss 30.992725
STEP 93 ================================
prereg loss 30.017614 regularization 728.6027 reg_novel 177.04079
loss 30.923258
STEP 94 ================================
prereg loss 29.47132 regularization 728.6205 reg_novel 177.01025
loss 30.37695
STEP 95 ================================
prereg loss 29.416733 regularization 728.63824 reg_novel 176.98067
loss 30.322351
STEP 96 ================================
prereg loss 29.623814 regularization 728.65424 reg_novel 176.9465
loss 30.529415
STEP 97 ================================
prereg loss 29.573137 regularization 728.6721 reg_novel 176.90376
loss 30.478714
STEP 98 ================================
prereg loss 29.62398 regularization 728.6885 reg_novel 176.8545
loss 30.529522
STEP 99 ================================
prereg loss 29.731306 regularization 728.7045 reg_novel 176.80663
loss 30.636818
STEP 100 ================================
prereg loss 29.749113 regularization 728.7218 reg_novel 176.76353
loss 30.654598
STEP 101 ================================
prereg loss 29.786612 regularization 728.7383 reg_novel 176.71935
loss 30.69207
STEP 102 ================================
prereg loss 29.859385 regularization 728.75415 reg_novel 176.67137
loss 30.76481
STEP 103 ================================
prereg loss 29.797846 regularization 728.77185 reg_novel 176.61806
loss 30.703236
STEP 104 ================================
prereg loss 30.11232 regularization 728.7899 reg_novel 176.56642
loss 31.017675
STEP 105 ================================
prereg loss 30.017736 regularization 728.80756 reg_novel 176.51375
loss 30.923058
STEP 106 ================================
prereg loss 30.031042 regularization 728.82574 reg_novel 176.47171
loss 30.93634
STEP 107 ================================
prereg loss 29.948475 regularization 728.8445 reg_novel 176.43283
loss 30.853752
STEP 108 ================================
prereg loss 30.117544 regularization 728.8608 reg_novel 176.38757
loss 31.022793
STEP 109 ================================
prereg loss 29.997587 regularization 728.8767 reg_novel 176.34332
loss 30.902807
STEP 110 ================================
prereg loss 29.982178 regularization 728.8913 reg_novel 176.2874
loss 30.887356
STEP 111 ================================
prereg loss 30.238718 regularization 728.90674 reg_novel 176.23894
loss 31.143864
STEP 112 ================================
prereg loss 30.148869 regularization 728.92 reg_novel 176.18686
loss 31.053976
STEP 113 ================================
prereg loss 30.53732 regularization 728.9348 reg_novel 176.14908
loss 31.442404
STEP 114 ================================
prereg loss 30.306067 regularization 728.94574 reg_novel 176.106
loss 31.211119
STEP 115 ================================
prereg loss 30.622644 regularization 728.95905 reg_novel 176.08429
loss 31.527687
STEP 116 ================================
prereg loss 31.21017 regularization 728.9686 reg_novel 176.03607
loss 32.115177
STEP 117 ================================
prereg loss 30.968443 regularization 728.9828 reg_novel 176.01414
loss 31.87344
STEP 118 ================================
prereg loss 32.207485 regularization 728.9964 reg_novel 175.98988
loss 33.112473
STEP 119 ================================
prereg loss 33.203808 regularization 729.00665 reg_novel 175.93814
loss 34.108753
STEP 120 ================================
prereg loss 30.925623 regularization 729.02344 reg_novel 175.90443
loss 31.830551
STEP 121 ================================
prereg loss 40.28463 regularization 729.0486 reg_novel 175.89714
loss 41.189575
STEP 122 ================================
prereg loss 32.527348 regularization 729.0606 reg_novel 175.84323
loss 33.43225
STEP 123 ================================
prereg loss 37.037052 regularization 729.07886 reg_novel 175.8015
loss 37.941933
STEP 124 ================================
prereg loss 31.456873 regularization 729.1026 reg_novel 175.75298
loss 32.36173
STEP 125 ================================
prereg loss 39.917946 regularization 729.1287 reg_novel 175.72456
loss 40.8228
STEP 126 ================================
prereg loss 43.65606 regularization 729.16077 reg_novel 175.73315
loss 44.560955
STEP 127 ================================
prereg loss 46.77682 regularization 729.1791 reg_novel 175.70142
loss 47.6817
STEP 128 ================================
prereg loss 81.24303 regularization 729.20074 reg_novel 175.62717
loss 82.14786
STEP 129 ================================
prereg loss 47.6574 regularization 729.2224 reg_novel 175.49042
loss 48.56211
STEP 130 ================================
prereg loss 37.57449 regularization 729.24194 reg_novel 175.33023
loss 38.47906
STEP 131 ================================
prereg loss 89.62562 regularization 729.25946 reg_novel 175.22072
loss 90.5301
STEP 132 ================================
prereg loss 88.22227 regularization 729.279 reg_novel 175.2064
loss 89.126755
STEP 133 ================================
prereg loss 38.139606 regularization 729.30286 reg_novel 175.26204
loss 39.04417
STEP 134 ================================
prereg loss 46.46445 regularization 729.325 reg_novel 175.33134
loss 47.369106
STEP 135 ================================
prereg loss 85.92695 regularization 729.34576 reg_novel 175.34898
loss 86.83164
STEP 136 ================================
prereg loss 70.233696 regularization 729.36066 reg_novel 175.28438
loss 71.13834
STEP 137 ================================
prereg loss 35.00328 regularization 729.3711 reg_novel 175.15858
loss 35.90781
STEP 138 ================================
prereg loss 37.511433 regularization 729.3777 reg_novel 175.01881
loss 38.41583
STEP 139 ================================
prereg loss 51.327423 regularization 729.38525 reg_novel 174.90866
loss 52.231716
STEP 140 ================================
prereg loss 44.361843 regularization 729.39404 reg_novel 174.84521
loss 45.266083
STEP 141 ================================
prereg loss 32.960907 regularization 729.4027 reg_novel 174.818
loss 33.865128
STEP 142 ================================
prereg loss 31.466438 regularization 729.4114 reg_novel 174.80603
loss 32.370655
STEP 143 ================================
prereg loss 31.53246 regularization 729.4174 reg_novel 174.79367
loss 32.436672
STEP 144 ================================
prereg loss 31.04848 regularization 729.42523 reg_novel 174.7781
loss 31.952682
STEP 145 ================================
prereg loss 32.46254 regularization 729.43506 reg_novel 174.76318
loss 33.366737
STEP 146 ================================
prereg loss 32.527985 regularization 729.44543 reg_novel 174.75232
loss 33.432182
STEP 147 ================================
prereg loss 33.553974 regularization 729.4565 reg_novel 174.74147
loss 34.45817
STEP 148 ================================
prereg loss 38.995327 regularization 729.4709 reg_novel 174.71928
loss 39.899517
STEP 149 ================================
prereg loss 40.949135 regularization 729.4864 reg_novel 174.67517
loss 41.8533
STEP 150 ================================
prereg loss 34.991203 regularization 729.504 reg_novel 174.60968
loss 35.895317
STEP 151 ================================
prereg loss 31.503616 regularization 729.52277 reg_novel 174.53468
loss 32.407673
STEP 152 ================================
prereg loss 35.123398 regularization 729.54285 reg_novel 174.46753
loss 36.02741
STEP 153 ================================
prereg loss 36.51995 regularization 729.56573 reg_novel 174.4224
loss 37.42394
STEP 154 ================================
prereg loss 32.978035 regularization 729.58856 reg_novel 174.39914
loss 33.882023
STEP 155 ================================
prereg loss 31.119392 regularization 729.6109 reg_novel 174.3884
loss 32.02339
STEP 156 ================================
prereg loss 32.040863 regularization 729.63104 reg_novel 174.37558
loss 32.94487
STEP 157 ================================
prereg loss 32.255333 regularization 729.6477 reg_novel 174.34799
loss 33.15933
STEP 158 ================================
prereg loss 31.347326 regularization 729.661 reg_novel 174.30261
loss 32.25129
STEP 159 ================================
prereg loss 31.279264 regularization 729.67285 reg_novel 174.24791
loss 32.183186
STEP 160 ================================
prereg loss 32.54573 regularization 729.682 reg_novel 174.19688
loss 33.449608
STEP 161 ================================
prereg loss 33.742687 regularization 729.6906 reg_novel 174.15782
loss 34.646534
STEP 162 ================================
prereg loss 33.35064 regularization 729.69904 reg_novel 174.13463
loss 34.254475
STEP 163 ================================
prereg loss 32.037067 regularization 729.70953 reg_novel 174.1243
loss 32.940903
STEP 164 ================================
prereg loss 31.840588 regularization 729.71967 reg_novel 174.11809
loss 32.744427
STEP 165 ================================
prereg loss 32.97572 regularization 729.73114 reg_novel 174.10437
loss 33.879555
STEP 166 ================================
prereg loss 33.57341 regularization 729.74335 reg_novel 174.0766
loss 34.47723
STEP 167 ================================
prereg loss 32.718567 regularization 729.75476 reg_novel 174.0348
loss 33.622356
STEP 168 ================================
prereg loss 32.26333 regularization 729.76843 reg_novel 173.98763
loss 33.167084
STEP 169 ================================
prereg loss 33.15114 regularization 729.78436 reg_novel 173.94624
loss 34.05487
STEP 170 ================================
prereg loss 33.706207 regularization 729.8027 reg_novel 173.91632
loss 34.60993
STEP 171 ================================
prereg loss 32.913662 regularization 729.82 reg_novel 173.89699
loss 33.81738
STEP 172 ================================
prereg loss 32.823257 regularization 729.8377 reg_novel 173.88106
loss 33.726974
STEP 173 ================================
prereg loss 33.96831 regularization 729.85614 reg_novel 173.85896
loss 34.872025
STEP 174 ================================
prereg loss 33.66327 regularization 729.8749 reg_novel 173.82497
loss 34.56697
STEP 175 ================================
prereg loss 33.13179 regularization 729.8905 reg_novel 173.78493
loss 34.035465
STEP 176 ================================
prereg loss 34.92839 regularization 729.90533 reg_novel 173.74808
loss 35.832043
STEP 177 ================================
prereg loss 35.21477 regularization 729.9207 reg_novel 173.72017
loss 36.118412
STEP 178 ================================
prereg loss 33.729412 regularization 729.93463 reg_novel 173.7004
loss 34.63305
STEP 179 ================================
prereg loss 33.931057 regularization 729.9491 reg_novel 173.68095
loss 34.834686
STEP 180 ================================
prereg loss 34.232685 regularization 729.9616 reg_novel 173.65675
loss 35.136303
STEP 181 ================================
prereg loss 33.92137 regularization 729.97437 reg_novel 173.62607
loss 34.82497
STEP 182 ================================
prereg loss 34.28352 regularization 729.98706 reg_novel 173.5915
loss 35.1871
STEP 183 ================================
prereg loss 34.864998 regularization 730.0006 reg_novel 173.5604
loss 35.76856
STEP 184 ================================
prereg loss 34.811367 regularization 730.0157 reg_novel 173.53548
loss 35.71492
STEP 185 ================================
prereg loss 34.29715 regularization 730.0294 reg_novel 173.51567
loss 35.200695
STEP 186 ================================
prereg loss 34.345013 regularization 730.04297 reg_novel 173.4951
loss 35.24855
STEP 187 ================================
prereg loss 34.637394 regularization 730.0584 reg_novel 173.46962
loss 35.54092
STEP 188 ================================
prereg loss 34.46175 regularization 730.0735 reg_novel 173.43958
loss 35.365265
STEP 189 ================================
prereg loss 34.395805 regularization 730.08966 reg_novel 173.40587
loss 35.2993
STEP 190 ================================
prereg loss 34.478058 regularization 730.10626 reg_novel 173.37123
loss 35.381535
STEP 191 ================================
prereg loss 34.59476 regularization 730.12146 reg_novel 173.34068
loss 35.498222
STEP 192 ================================
prereg loss 34.585518 regularization 730.1376 reg_novel 173.31377
loss 35.488968
STEP 193 ================================
prereg loss 34.532024 regularization 730.1536 reg_novel 173.28824
loss 35.435467
STEP 194 ================================
prereg loss 34.680054 regularization 730.17065 reg_novel 173.26219
loss 35.58349
STEP 195 ================================
prereg loss 34.677322 regularization 730.18646 reg_novel 173.23369
loss 35.58074
STEP 196 ================================
prereg loss 34.67472 regularization 730.2031 reg_novel 173.20108
loss 35.578125
STEP 197 ================================
prereg loss 34.743023 regularization 730.2205 reg_novel 173.16512
loss 35.646408
STEP 198 ================================
prereg loss 35.07012 regularization 730.23956 reg_novel 173.12866
loss 35.97349
STEP 199 ================================
prereg loss 35.02155 regularization 730.25885 reg_novel 173.0944
loss 35.924904
STEP 200 ================================
prereg loss 34.91699 regularization 730.2775 reg_novel 173.06342
loss 35.820328
STEP 201 ================================
prereg loss 34.961292 regularization 730.29474 reg_novel 173.03398
loss 35.86462
STEP 202 ================================
prereg loss 35.080345 regularization 730.3129 reg_novel 173.00179
loss 35.98366
STEP 203 ================================
prereg loss 35.048317 regularization 730.3314 reg_novel 172.96553
loss 35.951614
STEP 204 ================================
prereg loss 35.19706 regularization 730.3482 reg_novel 172.9276
loss 36.100334
STEP 205 ================================
prereg loss 35.304314 regularization 730.3662 reg_novel 172.89056
loss 36.20757
STEP 206 ================================
prereg loss 35.323875 regularization 730.3844 reg_novel 172.85664
loss 36.227116
STEP 207 ================================
prereg loss 35.275715 regularization 730.40106 reg_novel 172.82239
loss 36.17894
STEP 208 ================================
prereg loss 35.32977 regularization 730.41583 reg_novel 172.78697
loss 36.23297
STEP 209 ================================
prereg loss 35.401268 regularization 730.43335 reg_novel 172.75172
loss 36.304455
STEP 210 ================================
prereg loss 35.46866 regularization 730.4519 reg_novel 172.71419
loss 36.371826
STEP 211 ================================
prereg loss 35.530865 regularization 730.46985 reg_novel 172.67552
loss 36.43401
STEP 212 ================================
prereg loss 35.59603 regularization 730.48914 reg_novel 172.63672
loss 36.499157
STEP 213 ================================
prereg loss 35.66882 regularization 730.50836 reg_novel 172.59737
loss 36.571926
STEP 214 ================================
prereg loss 35.695206 regularization 730.52826 reg_novel 172.55707
loss 36.59829
STEP 215 ================================
prereg loss 35.731213 regularization 730.5499 reg_novel 172.5172
loss 36.63428
STEP 216 ================================
prereg loss 35.777485 regularization 730.57104 reg_novel 172.47751
loss 36.680534
STEP 217 ================================
prereg loss 35.823658 regularization 730.59076 reg_novel 172.43826
loss 36.72669
STEP 218 ================================
prereg loss 35.874447 regularization 730.61273 reg_novel 172.39867
loss 36.77746
STEP 219 ================================
prereg loss 35.989872 regularization 730.63214 reg_novel 172.3582
loss 36.892864
STEP 220 ================================
prereg loss 36.091625 regularization 730.65216 reg_novel 172.31973
loss 36.9946
STEP 221 ================================
prereg loss 36.102345 regularization 730.67285 reg_novel 172.28322
loss 37.005302
STEP 222 ================================
prereg loss 36.116516 regularization 730.69385 reg_novel 172.2478
loss 37.01946
STEP 223 ================================
prereg loss 36.165165 regularization 730.7125 reg_novel 172.2123
loss 37.06809
STEP 224 ================================
prereg loss 36.214855 regularization 730.7333 reg_novel 172.17575
loss 37.117764
STEP 225 ================================
prereg loss 36.269 regularization 730.7538 reg_novel 172.13722
loss 37.17189
STEP 226 ================================
prereg loss 36.3862 regularization 730.7752 reg_novel 172.09921
loss 37.289074
STEP 227 ================================
prereg loss 36.4046 regularization 730.7965 reg_novel 172.06294
loss 37.307457
STEP 228 ================================
prereg loss 36.369553 regularization 730.8188 reg_novel 172.02798
loss 37.2724
STEP 229 ================================
prereg loss 36.39746 regularization 730.8415 reg_novel 171.99329
loss 37.300297
STEP 230 ================================
prereg loss 36.451576 regularization 730.8633 reg_novel 171.95673
loss 37.354397
STEP 231 ================================
prereg loss 36.45069 regularization 730.8861 reg_novel 171.91557
loss 37.353493
STEP 232 ================================
prereg loss 36.52076 regularization 730.9083 reg_novel 171.8749
loss 37.423542
STEP 233 ================================
prereg loss 36.623898 regularization 730.92975 reg_novel 171.83678
loss 37.526665
STEP 234 ================================
prereg loss 36.684486 regularization 730.952 reg_novel 171.80034
loss 37.58724
STEP 235 ================================
prereg loss 36.733364 regularization 730.9741 reg_novel 171.76459
loss 37.636105
STEP 236 ================================
prereg loss 36.80515 regularization 730.9967 reg_novel 171.7303
loss 37.707874
STEP 237 ================================
prereg loss 36.89057 regularization 731.01807 reg_novel 171.69589
loss 37.793285
STEP 238 ================================
prereg loss 36.987003 regularization 731.04144 reg_novel 171.66182
loss 37.889706
STEP 239 ================================
prereg loss 37.094765 regularization 731.0646 reg_novel 171.62863
loss 37.99746
STEP 240 ================================
prereg loss 37.181072 regularization 731.0876 reg_novel 171.59737
loss 38.083755
STEP 241 ================================
prereg loss 37.237812 regularization 731.11 reg_novel 171.56894
loss 38.14049
STEP 242 ================================
prereg loss 37.299774 regularization 731.13416 reg_novel 171.5416
loss 38.20245
STEP 243 ================================
prereg loss 37.378006 regularization 731.1582 reg_novel 171.51427
loss 38.280678
STEP 244 ================================
prereg loss 37.44467 regularization 731.1825 reg_novel 171.48503
loss 38.34734
STEP 245 ================================
prereg loss 37.517418 regularization 731.2084 reg_novel 171.45602
loss 38.420082
STEP 246 ================================
prereg loss 37.582912 regularization 731.236 reg_novel 171.42831
loss 38.485577
STEP 247 ================================
prereg loss 37.62905 regularization 731.26495 reg_novel 171.40202
loss 38.53172
STEP 248 ================================
prereg loss 37.66782 regularization 731.29297 reg_novel 171.37659
loss 38.570488
STEP 249 ================================
prereg loss 37.706024 regularization 731.3201 reg_novel 171.35251
loss 38.608696
STEP 250 ================================
prereg loss 37.73941 regularization 731.34937 reg_novel 171.3287
loss 38.64209
STEP 251 ================================
prereg loss 37.804897 regularization 731.37897 reg_novel 171.30446
loss 38.70758
STEP 252 ================================
prereg loss 37.945152 regularization 731.4098 reg_novel 171.28168
loss 38.847843
STEP 253 ================================
prereg loss 37.90873 regularization 731.4454 reg_novel 171.26924
loss 38.811443
STEP 254 ================================
prereg loss 37.807236 regularization 731.4789 reg_novel 171.2581
loss 38.709972
STEP 255 ================================
prereg loss 37.59921 regularization 731.51294 reg_novel 171.24782
loss 38.50197
STEP 256 ================================
prereg loss 37.5125 regularization 731.5456 reg_novel 171.23895
loss 38.415287
STEP 257 ================================
prereg loss 37.41902 regularization 731.5774 reg_novel 171.23045
loss 38.32183
STEP 258 ================================
prereg loss 37.259987 regularization 731.61005 reg_novel 171.22163
loss 38.16282
STEP 259 ================================
prereg loss 37.135246 regularization 731.64233 reg_novel 171.21248
loss 38.0381
STEP 260 ================================
prereg loss 37.216545 regularization 731.6739 reg_novel 171.20229
loss 38.119423
STEP 261 ================================
prereg loss 37.48182 regularization 731.7201 reg_novel 171.21303
loss 38.384754
STEP 262 ================================
prereg loss 36.592308 regularization 731.7654 reg_novel 171.21725
loss 37.49529
STEP 263 ================================
prereg loss 37.216637 regularization 731.8084 reg_novel 171.2203
loss 38.119667
STEP 264 ================================
prereg loss 36.64194 regularization 731.84845 reg_novel 171.22366
loss 37.545013
STEP 265 ================================
prereg loss 36.413204 regularization 731.8871 reg_novel 171.22693
loss 37.31632
STEP 266 ================================
prereg loss 39.817863 regularization 731.925 reg_novel 171.23495
loss 40.721024
STEP 267 ================================
prereg loss 39.18748 regularization 731.9573 reg_novel 171.23744
loss 40.090675
STEP 268 ================================
prereg loss 39.411057 regularization 731.98846 reg_novel 171.23184
loss 40.314278
STEP 269 ================================
prereg loss 37.85423 regularization 732.0176 reg_novel 171.22124
loss 38.75747
STEP 270 ================================
prereg loss 47.76866 regularization 732.0434 reg_novel 171.2252
loss 48.67193
STEP 271 ================================
prereg loss 51.89633 regularization 732.0714 reg_novel 171.25624
loss 52.79966
STEP 272 ================================
prereg loss 39.910217 regularization 732.1039 reg_novel 171.31549
loss 40.813637
STEP 273 ================================
prereg loss 40.10152 regularization 732.13983 reg_novel 171.38506
loss 41.005047
STEP 274 ================================
prereg loss 57.253468 regularization 732.17883 reg_novel 171.4377
loss 58.157085
STEP 275 ================================
prereg loss 57.44182 regularization 732.2209 reg_novel 171.45612
loss 58.345497
STEP 276 ================================
prereg loss 40.961887 regularization 732.26294 reg_novel 171.44371
loss 41.865593
STEP 277 ================================
prereg loss 37.98155 regularization 732.30646 reg_novel 171.42108
loss 38.885277
STEP 278 ================================
prereg loss 45.850254 regularization 732.35114 reg_novel 171.41026
loss 46.754017
STEP 279 ================================
prereg loss 44.021015 regularization 732.39996 reg_novel 171.42215
loss 44.92484
STEP 280 ================================
prereg loss 37.81586 regularization 732.4472 reg_novel 171.45032
loss 38.719757
STEP 281 ================================
prereg loss 38.548946 regularization 732.494 reg_novel 171.48064
loss 39.452923
STEP 282 ================================
prereg loss 40.1548 regularization 732.5366 reg_novel 171.50035
loss 41.058838
STEP 283 ================================
prereg loss 38.714714 regularization 732.57794 reg_novel 171.50677
loss 39.618797
STEP 284 ================================
prereg loss 39.17951 regularization 732.61957 reg_novel 171.50552
loss 40.083633
STEP 285 ================================
prereg loss 40.55589 regularization 732.66376 reg_novel 171.5048
loss 41.460056
STEP 286 ================================
prereg loss 39.765354 regularization 732.70917 reg_novel 171.50801
loss 40.66957
STEP 287 ================================
prereg loss 39.668766 regularization 732.7535 reg_novel 171.51065
loss 40.57303
STEP 288 ================================
prereg loss 41.028732 regularization 732.79846 reg_novel 171.50542
loss 41.933037
STEP 289 ================================
prereg loss 40.061928 regularization 732.84265 reg_novel 171.48825
loss 40.96626
STEP 290 ================================
prereg loss 38.290108 regularization 732.8857 reg_novel 171.46216
loss 39.194454
STEP 291 ================================
prereg loss 39.847137 regularization 732.9267 reg_novel 171.43564
loss 40.7515
STEP 292 ================================
prereg loss 41.766926 regularization 732.9668 reg_novel 171.41289
loss 42.671307
STEP 293 ================================
prereg loss 40.091873 regularization 733.0033 reg_novel 171.39586
loss 40.996273
STEP 294 ================================
prereg loss 38.299347 regularization 733.03937 reg_novel 171.38068
loss 39.203766
STEP 295 ================================
prereg loss 39.615623 regularization 733.0758 reg_novel 171.36102
loss 40.52006
STEP 296 ================================
prereg loss 40.932835 regularization 733.1111 reg_novel 171.33095
loss 41.837276
STEP 297 ================================
prereg loss 39.974304 regularization 733.1448 reg_novel 171.29019
loss 40.87874
STEP 298 ================================
prereg loss 38.90316 regularization 733.17834 reg_novel 171.241
loss 39.80758
STEP 299 ================================
prereg loss 38.96688 regularization 733.2114 reg_novel 171.18886
loss 39.87128
STEP 300 ================================
prereg loss 38.8945 regularization 733.24445 reg_novel 171.13692
loss 39.79888
STEP 301 ================================
prereg loss 38.637753 regularization 733.2767 reg_novel 171.08516
loss 39.542114
STEP 302 ================================
prereg loss 38.793964 regularization 733.3097 reg_novel 171.03174
loss 39.698307
STEP 303 ================================
prereg loss 38.70304 regularization 733.3413 reg_novel 170.97482
loss 39.607357
STEP 304 ================================
prereg loss 38.649715 regularization 733.37085 reg_novel 170.91524
loss 39.554
STEP 305 ================================
prereg loss 39.577763 regularization 733.39746 reg_novel 170.85773
loss 40.482018
STEP 306 ================================
prereg loss 40.402077 regularization 733.4239 reg_novel 170.80667
loss 41.30631
STEP 307 ================================
prereg loss 39.77216 regularization 733.448 reg_novel 170.76355
loss 40.676373
STEP 308 ================================
prereg loss 38.936474 regularization 733.47327 reg_novel 170.7245
loss 39.84067
STEP 309 ================================
prereg loss 39.318718 regularization 733.4966 reg_novel 170.68546
loss 40.2229
STEP 310 ================================
prereg loss 40.02686 regularization 733.52045 reg_novel 170.64055
loss 40.93102
STEP 311 ================================
prereg loss 39.825375 regularization 733.5475 reg_novel 170.58978
loss 40.72951
STEP 312 ================================
prereg loss 39.346924 regularization 733.5739 reg_novel 170.53325
loss 40.25103
STEP 313 ================================
prereg loss 39.305748 regularization 733.60144 reg_novel 170.4749
loss 40.209824
STEP 314 ================================
prereg loss 39.33016 regularization 733.6295 reg_novel 170.41856
loss 40.234207
STEP 315 ================================
prereg loss 39.245712 regularization 733.6575 reg_novel 170.36462
loss 40.149734
STEP 316 ================================
prereg loss 39.2726 regularization 733.6843 reg_novel 170.31175
loss 40.176594
STEP 317 ================================
prereg loss 39.294956 regularization 733.7117 reg_novel 170.25835
loss 40.198925
STEP 318 ================================
prereg loss 39.3355 regularization 733.7397 reg_novel 170.20636
loss 40.239445
STEP 319 ================================
prereg loss 39.47141 regularization 733.7662 reg_novel 170.1571
loss 40.37533
STEP 320 ================================
prereg loss 39.453667 regularization 733.7938 reg_novel 170.11188
loss 40.35757
STEP 321 ================================
prereg loss 39.382786 regularization 733.8192 reg_novel 170.06905
loss 40.286674
STEP 322 ================================
prereg loss 39.41712 regularization 733.84546 reg_novel 170.02782
loss 40.320995
STEP 323 ================================
prereg loss 39.356224 regularization 733.87146 reg_novel 169.98672
loss 40.260082
STEP 324 ================================
prereg loss 39.20552 regularization 733.8969 reg_novel 169.94342
loss 40.10936
STEP 325 ================================
prereg loss 39.215218 regularization 733.9218 reg_novel 169.90038
loss 40.11904
STEP 326 ================================
prereg loss 39.260845 regularization 733.9478 reg_novel 169.8594
loss 40.164654
STEP 327 ================================
prereg loss 39.09538 regularization 733.97455 reg_novel 169.8204
loss 39.999172
STEP 328 ================================
prereg loss 38.875435 regularization 734.001 reg_novel 169.78313
loss 39.77922
STEP 329 ================================
prereg loss 38.80747 regularization 734.0291 reg_novel 169.74704
loss 39.711246
STEP 330 ================================
prereg loss 38.859734 regularization 734.05566 reg_novel 169.71075
loss 39.7635
STEP 331 ================================
prereg loss 38.852554 regularization 734.0812 reg_novel 169.67534
loss 39.75631
STEP 332 ================================
prereg loss 38.83076 regularization 734.10944 reg_novel 169.64043
loss 39.734512
STEP 333 ================================
prereg loss 38.888733 regularization 734.13715 reg_novel 169.60754
loss 39.792477
STEP 334 ================================
prereg loss 38.989998 regularization 734.16473 reg_novel 169.57663
loss 39.893738
STEP 335 ================================
prereg loss 39.100815 regularization 734.19104 reg_novel 169.54681
loss 40.00455
STEP 336 ================================
prereg loss 39.081223 regularization 734.21545 reg_novel 169.51768
loss 39.984955
STEP 337 ================================
prereg loss 39.059956 regularization 734.2394 reg_novel 169.488
loss 39.963684
STEP 338 ================================
prereg loss 39.123974 regularization 734.26196 reg_novel 169.45724
loss 40.027695
STEP 339 ================================
prereg loss 39.153065 regularization 734.2827 reg_novel 169.42372
loss 40.05677
STEP 340 ================================
prereg loss 39.155037 regularization 734.3017 reg_novel 169.38693
loss 40.058727
STEP 341 ================================
prereg loss 39.203316 regularization 734.3215 reg_novel 169.34703
loss 40.106983
STEP 342 ================================
prereg loss 39.312706 regularization 734.34235 reg_novel 169.30392
loss 40.21635
STEP 343 ================================
prereg loss 39.442482 regularization 734.3626 reg_novel 169.2577
loss 40.346104
STEP 344 ================================
prereg loss 39.429115 regularization 734.38416 reg_novel 169.20955
loss 40.33271
STEP 345 ================================
prereg loss 39.36579 regularization 734.4044 reg_novel 169.16013
loss 40.269356
STEP 346 ================================
prereg loss 39.377384 regularization 734.4248 reg_novel 169.11119
loss 40.280922
STEP 347 ================================
prereg loss 39.43072 regularization 734.4445 reg_novel 169.06413
loss 40.33423
STEP 348 ================================
prereg loss 39.44709 regularization 734.4655 reg_novel 169.01918
loss 40.350574
STEP 349 ================================
prereg loss 39.48222 regularization 734.4878 reg_novel 168.97523
loss 40.38568
STEP 350 ================================
prereg loss 39.520096 regularization 734.5078 reg_novel 168.93251
loss 40.423534
STEP 351 ================================
prereg loss 39.560802 regularization 734.5275 reg_novel 168.89067
loss 40.464222
STEP 352 ================================
prereg loss 39.664417 regularization 734.5459 reg_novel 168.84949
loss 40.567814
STEP 353 ================================
prereg loss 39.720753 regularization 734.5632 reg_novel 168.8081
loss 40.624123
STEP 354 ================================
prereg loss 39.72208 regularization 734.5803 reg_novel 168.76662
loss 40.625427
STEP 355 ================================
prereg loss 39.740704 regularization 734.598 reg_novel 168.7255
loss 40.644028
STEP 356 ================================
prereg loss 39.790325 regularization 734.61707 reg_novel 168.68341
loss 40.693626
STEP 357 ================================
prereg loss 39.940456 regularization 734.6369 reg_novel 168.64085
loss 40.843735
STEP 358 ================================
prereg loss 40.02704 regularization 734.6577 reg_novel 168.59691
loss 40.930294
STEP 359 ================================
prereg loss 40.003704 regularization 734.67676 reg_novel 168.55127
loss 40.906933
STEP 360 ================================
prereg loss 40.014736 regularization 734.69946 reg_novel 168.50328
loss 40.91794
STEP 361 ================================
prereg loss 40.076164 regularization 734.72064 reg_novel 168.45628
loss 40.97934
STEP 362 ================================
prereg loss 40.13179 regularization 734.74146 reg_novel 168.41171
loss 41.034943
STEP 363 ================================
prereg loss 40.243042 regularization 734.76215 reg_novel 168.36891
loss 41.14617
STEP 364 ================================
prereg loss 40.35804 regularization 734.78357 reg_novel 168.32504
loss 41.26115
STEP 365 ================================
prereg loss 40.27108 regularization 734.80383 reg_novel 168.27934
loss 41.174164
STEP 366 ================================
prereg loss 40.127434 regularization 734.82544 reg_novel 168.23163
loss 41.03049
STEP 367 ================================
prereg loss 40.076626 regularization 734.84564 reg_novel 168.18398
loss 40.979656
STEP 368 ================================
prereg loss 40.038467 regularization 734.86536 reg_novel 168.13818
loss 40.94147
2022-06-14T20:34:49.953

julia> count(trainable["network_matrix"])
7521

julia> count(trainable["fixed_matrix"])
20

julia> trainable["fixed_matrix"]
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 20 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "accum-6-1" => Dict("dict-1"=>Dict("accum-6-1"=>Dict("dict"=>1.0)))
  "accum-3-1" => Dict("dict-1"=>Dict("accum-3-1"=>Dict("dict"=>1.0)))
  "accum-1-1" => Dict("dict-1"=>Dict("accum-1-1"=>Dict("dict"=>1.0)))
  "accum-3-3" => Dict("dict-1"=>Dict("accum-3-3"=>Dict("dict"=>1.0)))
  "accum-4-3" => Dict("dict-1"=>Dict("accum-4-3"=>Dict("dict"=>1.0)))
  "accum-5-1" => Dict("dict-1"=>Dict("accum-5-1"=>Dict("dict"=>1.0)))
  "accum-6-3" => Dict("dict-1"=>Dict("accum-6-3"=>Dict("dict"=>1.0)))
  "accum-5-3" => Dict("dict-1"=>Dict("accum-5-3"=>Dict("dict"=>1.0)))
  "accum-3-2" => Dict("dict-1"=>Dict("accum-3-2"=>Dict("dict"=>1.0)))
  "accum-4-2" => Dict("dict-1"=>Dict("accum-4-2"=>Dict("dict"=>1.0)))
  "accum-2-3" => Dict("dict-1"=>Dict("accum-2-3"=>Dict("dict"=>1.0)))
  "accum-1-3" => Dict("dict-1"=>Dict("accum-1-3"=>Dict("dict"=>1.0)))
  "accum-4-1" => Dict("dict-1"=>Dict("accum-4-1"=>Dict("dict"=>1.0)))
  "accum-5-2" => Dict("dict-1"=>Dict("accum-5-2"=>Dict("dict"=>1.0)))
  "accum-6-2" => Dict("dict-1"=>Dict("accum-6-2"=>Dict("dict"=>1.0)))
  "accum-2-1" => Dict("dict-1"=>Dict("accum-2-1"=>Dict("dict"=>1.0)))
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "accum-2-2" => Dict("dict-1"=>Dict("accum-2-2"=>Dict("dict"=>1.0)))
  "accum-1-2" => Dict("dict-1"=>Dict("accum-1-2"=>Dict("dict"=>1.0)))

julia> close(io)
```
