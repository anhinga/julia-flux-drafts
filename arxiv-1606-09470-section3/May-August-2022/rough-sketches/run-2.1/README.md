# Variant of run 2 with the governor on "dot" output

Now the hiccup at the step 77 is still very pronounced, but the network does not blow-up.

The loss is evolving in the wrong direction for a while after that hiccup,
the finds the right direction. The result is too smooth, too far from what we really
want, not at all as promising as in run 1.

We should consider "curriculum BPTT" with gradually increasing time window
for how long we run the network (start with the first moment of non-trivial
difference from 0 output, then gradually extend).

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
prereg loss 577.4583 regularization 556.0789
loss 633.0662
DONE: adam_step!
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 7 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.0994167), "accum-1"=>Dict("dict"=>-0.0954369, "true"=>…
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.141752), "accum-1"=>Dict("dict"=>0.0276652, "true"=>-…
  "compare-1" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.451135), "accum-1"=>Dict("dict"=>-0.0990737, "true"=>0.…
  "dot-1"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.141141), "accum-1"=>Dict("dict"=>0.212951, "true"=>0.0…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0191045), "accum-1"=>Dict("dict"=>0.196098, "true"=>-0.…

julia> for i in 1:10
           printlog_v(io, "STEP ", i, " ================================")
           training_step!()
       end
STEP 1 ================================
prereg loss 576.17017 regularization 552.19763
loss 631.38995
STEP 2 ================================
prereg loss 574.8844 regularization 548.3373
loss 629.71814
STEP 3 ================================
prereg loss 573.60596 regularization 544.4967
loss 628.0556
STEP 4 ================================
prereg loss 572.3319 regularization 540.6821
loss 626.40015
STEP 5 ================================
prereg loss 571.06067 regularization 536.88666
loss 624.7493
STEP 6 ================================
prereg loss 569.79285 regularization 533.11444
loss 623.1043
STEP 7 ================================
prereg loss 568.52814 regularization 529.365
loss 621.46466
STEP 8 ================================
prereg loss 567.2647 regularization 525.63727
loss 619.8284
STEP 9 ================================
prereg loss 566.00336 regularization 521.93243
loss 618.1966
STEP 10 ================================
prereg loss 564.7443 regularization 518.2566
loss 616.57

julia> for i in 1:500
           printlog_v(io, "STEP ", i, " ================================")
           training_step!()
       end
STEP 1 ================================
prereg loss 563.48773 regularization 514.6086
loss 614.9486
STEP 2 ================================
prereg loss 562.2333 regularization 510.98544
loss 613.33185
STEP 3 ================================
prereg loss 560.9816 regularization 507.3827
loss 611.7199
STEP 4 ================================
prereg loss 559.7328 regularization 503.805
loss 610.1133
STEP 5 ================================
prereg loss 558.4871 regularization 500.25214
loss 608.5123
STEP 6 ================================
prereg loss 557.2442 regularization 496.7267
loss 606.9169
STEP 7 ================================
prereg loss 556.00354 regularization 493.23224
loss 605.3268
STEP 8 ================================
prereg loss 554.765 regularization 489.7612
loss 603.74115
STEP 9 ================================
prereg loss 553.5284 regularization 486.31516
loss 602.1599
STEP 10 ================================
prereg loss 552.29297 regularization 482.89206
loss 600.58215
STEP 11 ================================
prereg loss 551.0593 regularization 479.49426
loss 599.0087
STEP 12 ================================
prereg loss 549.82623 regularization 476.12094
loss 597.43835
STEP 13 ================================
prereg loss 548.59375 regularization 472.77563
loss 595.87134
STEP 14 ================================
prereg loss 547.3619 regularization 469.45926
loss 594.3078
STEP 15 ================================
prereg loss 546.1304 regularization 466.16367
loss 592.74677
STEP 16 ================================
prereg loss 544.8906 regularization 462.88928
loss 591.17957
STEP 17 ================================
prereg loss 543.6513 regularization 459.6425
loss 589.61554
STEP 18 ================================
prereg loss 542.33875 regularization 456.4179
loss 587.9805
STEP 19 ================================
prereg loss 540.99835 regularization 453.22278
loss 586.3206
STEP 20 ================================
prereg loss 539.64044 regularization 450.06107
loss 584.64655
STEP 21 ================================
prereg loss 538.2616 regularization 446.93152
loss 582.9548
STEP 22 ================================
prereg loss 536.85767 regularization 443.84055
loss 581.2417
STEP 23 ================================
prereg loss 535.4254 regularization 440.77917
loss 579.50336
STEP 24 ================================
prereg loss 533.96106 regularization 437.7431
loss 577.73535
STEP 25 ================================
prereg loss 532.462 regularization 434.7347
loss 575.9354
STEP 26 ================================
prereg loss 530.9259 regularization 431.75113
loss 574.101
STEP 27 ================================
prereg loss 529.3493 regularization 428.79382
loss 572.2287
STEP 28 ================================
prereg loss 527.7305 regularization 425.86374
loss 570.3169
STEP 29 ================================
prereg loss 526.0668 regularization 422.96255
loss 568.36304
STEP 30 ================================
prereg loss 524.35614 regularization 420.0863
loss 566.36475
STEP 31 ================================
prereg loss 522.5974 regularization 417.23224
loss 564.3206
STEP 32 ================================
prereg loss 520.78546 regularization 414.40546
loss 562.226
STEP 33 ================================
prereg loss 518.9178 regularization 411.6001
loss 560.0778
STEP 34 ================================
prereg loss 516.9904 regularization 408.8185
loss 557.87225
STEP 35 ================================
prereg loss 515.00006 regularization 406.05692
loss 555.6058
STEP 36 ================================
prereg loss 512.9431 regularization 403.3154
loss 553.27466
STEP 37 ================================
prereg loss 510.81607 regularization 400.60037
loss 550.8761
STEP 38 ================================
prereg loss 508.61453 regularization 397.9155
loss 548.40607
STEP 39 ================================
prereg loss 506.33444 regularization 395.25378
loss 545.8598
STEP 40 ================================
prereg loss 503.97177 regularization 392.61246
loss 543.23303
STEP 41 ================================
prereg loss 501.52057 regularization 389.99167
loss 540.5197
STEP 42 ================================
prereg loss 498.97586 regularization 387.3909
loss 537.71497
STEP 43 ================================
prereg loss 496.3317 regularization 384.80975
loss 534.8127
STEP 44 ================================
prereg loss 493.582 regularization 382.2521
loss 531.8072
STEP 45 ================================
prereg loss 490.72046 regularization 379.7192
loss 528.6924
STEP 46 ================================
prereg loss 487.7393 regularization 377.21573
loss 525.4609
STEP 47 ================================
prereg loss 484.631 regularization 374.73322
loss 522.1043
STEP 48 ================================
prereg loss 481.38715 regularization 372.27228
loss 518.6144
STEP 49 ================================
prereg loss 477.99786 regularization 369.83975
loss 514.9818
STEP 50 ================================
prereg loss 474.45358 regularization 367.43365
loss 511.19696
STEP 51 ================================
prereg loss 470.7422 regularization 365.05045
loss 507.24722
STEP 52 ================================
prereg loss 466.85114 regularization 362.68515
loss 503.11966
STEP 53 ================================
prereg loss 462.76633 regularization 360.34164
loss 498.80048
STEP 54 ================================
prereg loss 458.47113 regularization 358.0223
loss 494.27338
STEP 55 ================================
prereg loss 453.94727 regularization 355.71872
loss 489.51913
STEP 56 ================================
prereg loss 449.17395 regularization 353.43604
loss 484.51755
STEP 57 ================================
prereg loss 444.1558 regularization 351.17072
loss 479.27286
STEP 58 ================================
prereg loss 438.86502 regularization 348.92557
loss 473.75757
STEP 59 ================================
prereg loss 433.27045 regularization 346.7
loss 467.94046
STEP 60 ================================
prereg loss 427.3375 regularization 344.48987
loss 461.78647
STEP 61 ================================
prereg loss 421.02863 regularization 342.29858
loss 455.25848
STEP 62 ================================
prereg loss 414.2958 regularization 340.11884
loss 448.30768
STEP 63 ================================
prereg loss 407.08414 regularization 337.96228
loss 440.88037
STEP 64 ================================
prereg loss 399.3299 regularization 335.82635
loss 432.91254
STEP 65 ================================
prereg loss 390.95908 regularization 333.70932
loss 424.33002
STEP 66 ================================
prereg loss 381.8819 regularization 331.6173
loss 415.04364
STEP 67 ================================
prereg loss 371.9928 regularization 329.53995
loss 404.94678
STEP 68 ================================
prereg loss 361.1468 regularization 327.48447
loss 393.89523
STEP 69 ================================
prereg loss 349.18893 regularization 325.4516
loss 381.7341
STEP 70 ================================
prereg loss 335.90366 regularization 323.4393
loss 368.2476
STEP 71 ================================
prereg loss 321.0291 regularization 321.4468
loss 353.1738
STEP 72 ================================
prereg loss 304.18076 regularization 319.4709
loss 336.12784
STEP 73 ================================
prereg loss 284.78363 regularization 317.51526
loss 316.53516
STEP 74 ================================
prereg loss 262.27426 regularization 315.57483
loss 293.83176
STEP 75 ================================
prereg loss 236.76846 regularization 313.64795
loss 268.13327
STEP 76 ================================
prereg loss 219.27948 regularization 311.71204
loss 250.45068
STEP 77 ================================
prereg loss 31278.742 regularization 309.74545
loss 31309.717
STEP 78 ================================
prereg loss 217.25525 regularization 307.57364
loss 248.01262
STEP 79 ================================
prereg loss 213.77293 regularization 305.42682
loss 244.31561
STEP 80 ================================
prereg loss 223.56534 regularization 303.30743
loss 253.89609
STEP 81 ================================
prereg loss 232.64897 regularization 301.2133
loss 262.7703
STEP 82 ================================
prereg loss 240.49423 regularization 299.13968
loss 270.4082
STEP 83 ================================
prereg loss 247.4064 regularization 297.08847
loss 277.11523
STEP 84 ================================
prereg loss 253.50816 regularization 295.05795
loss 283.01395
STEP 85 ================================
prereg loss 258.91394 regularization 293.0472
loss 288.21866
STEP 86 ================================
prereg loss 263.7221 regularization 291.05524
loss 292.82764
STEP 87 ================================
prereg loss 268.01382 regularization 289.08466
loss 296.9223
STEP 88 ================================
prereg loss 271.85806 regularization 287.1317
loss 300.57123
STEP 89 ================================
prereg loss 275.312 regularization 285.19772
loss 303.8318
STEP 90 ================================
prereg loss 278.42294 regularization 283.27997
loss 306.75095
STEP 91 ================================
prereg loss 281.078 regularization 281.3803
loss 309.21603
STEP 92 ================================
prereg loss 283.33707 regularization 279.49988
loss 311.28705
STEP 93 ================================
prereg loss 285.25034 regularization 277.63495
loss 313.01382
STEP 94 ================================
prereg loss 286.90555 regularization 275.7876
loss 314.4843
STEP 95 ================================
prereg loss 288.3343 regularization 273.9532
loss 315.7296
STEP 96 ================================
prereg loss 289.5673 regularization 272.13467
loss 316.78076
STEP 97 ================================
prereg loss 290.62793 regularization 270.32974
loss 317.6609
STEP 98 ================================
prereg loss 291.67853 regularization 268.53845
loss 318.53238
STEP 99 ================================
prereg loss 292.71936 regularization 266.76178
loss 319.39554
STEP 100 ================================
prereg loss 293.75003 regularization 265.00092
loss 320.25012
STEP 101 ================================
prereg loss 294.76782 regularization 263.25537
loss 321.09335
STEP 102 ================================
prereg loss 295.77237 regularization 261.5211
loss 321.92447
STEP 103 ================================
prereg loss 296.76358 regularization 259.80203
loss 322.74377
STEP 104 ================================
prereg loss 297.6054 regularization 258.09338
loss 323.41473
STEP 105 ================================
prereg loss 298.31683 regularization 256.39755
loss 323.9566
STEP 106 ================================
prereg loss 298.9119 regularization 254.7144
loss 324.38333
STEP 107 ================================
prereg loss 299.4029 regularization 253.04228
loss 324.70712
STEP 108 ================================
prereg loss 299.79788 regularization 251.38585
loss 324.93646
STEP 109 ================================
prereg loss 300.241 regularization 249.7433
loss 325.21533
STEP 110 ================================
prereg loss 300.67978 regularization 248.11061
loss 325.49084
STEP 111 ================================
prereg loss 301.11597 regularization 246.49187
loss 325.76517
STEP 112 ================================
prereg loss 301.54755 regularization 244.8872
loss 326.03625
STEP 113 ================================
prereg loss 301.84137 regularization 243.2946
loss 326.17084
STEP 114 ================================
prereg loss 302.01154 regularization 241.7151
loss 326.18304
STEP 115 ================================
prereg loss 302.068 regularization 240.15076
loss 326.08307
STEP 116 ================================
prereg loss 302.15332 regularization 238.59827
loss 326.01315
STEP 117 ================================
prereg loss 302.26428 regularization 237.05623
loss 325.9699
STEP 118 ================================
prereg loss 302.266 regularization 235.52478
loss 325.81848
STEP 119 ================================
prereg loss 302.16995 regularization 234.00317
loss 325.57028
STEP 120 ================================
prereg loss 302.1672 regularization 232.49309
loss 325.4165
STEP 121 ================================
prereg loss 302.17017 regularization 230.99635
loss 325.2698
STEP 122 ================================
prereg loss 302.1784 regularization 229.5063
loss 325.12903
STEP 123 ================================
prereg loss 302.0601 regularization 228.02489
loss 324.86258
STEP 124 ================================
prereg loss 301.82803 regularization 226.55579
loss 324.4836
STEP 125 ================================
prereg loss 301.49026 regularization 225.09442
loss 323.9997
STEP 126 ================================
prereg loss 301.05627 regularization 223.64507
loss 323.42078
STEP 127 ================================
prereg loss 300.66602 regularization 222.20668
loss 322.8867
STEP 128 ================================
prereg loss 300.26672 regularization 220.77681
loss 322.3444
STEP 129 ================================
prereg loss 299.86237 regularization 219.35637
loss 321.798
STEP 130 ================================
prereg loss 299.45355 regularization 217.95068
loss 321.24863
STEP 131 ================================
prereg loss 298.90604 regularization 216.55792
loss 320.56183
STEP 132 ================================
prereg loss 298.23102 regularization 215.17516
loss 319.74854
STEP 133 ================================
prereg loss 297.43896 regularization 213.80086
loss 318.81906
STEP 134 ================================
prereg loss 296.6802 regularization 212.43425
loss 317.92365
STEP 135 ================================
prereg loss 295.99893 regularization 211.07784
loss 317.10672
STEP 136 ================================
prereg loss 295.39096 regularization 209.73218
loss 316.36417
STEP 137 ================================
prereg loss 294.79486 regularization 208.39761
loss 315.6346
STEP 138 ================================
prereg loss 294.20627 regularization 207.06917
loss 314.91318
STEP 139 ================================
prereg loss 293.6249 regularization 205.7475
loss 314.19965
STEP 140 ================================
prereg loss 293.05014 regularization 204.43611
loss 313.49374
STEP 141 ================================
prereg loss 292.57538 regularization 203.13873
loss 312.88925
STEP 142 ================================
prereg loss 292.19293 regularization 201.85307
loss 312.37823
STEP 143 ================================
prereg loss 291.89426 regularization 200.57372
loss 311.95163
STEP 144 ================================
prereg loss 291.66815 regularization 199.30046
loss 311.5982
STEP 145 ================================
prereg loss 291.36487 regularization 198.04015
loss 311.16888
STEP 146 ================================
prereg loss 291.04538 regularization 196.78821
loss 310.7242
STEP 147 ================================
prereg loss 290.6227 regularization 195.54375
loss 310.1771
STEP 148 ================================
prereg loss 290.24664 regularization 194.3108
loss 309.67773
STEP 149 ================================
prereg loss 289.91394 regularization 193.08469
loss 309.2224
STEP 150 ================================
prereg loss 289.42435 regularization 191.87006
loss 308.61136
STEP 151 ================================
prereg loss 288.94037 regularization 190.66302
loss 308.00668
STEP 152 ================================
prereg loss 288.31476 regularization 189.46312
loss 307.26108
STEP 153 ================================
prereg loss 287.61224 regularization 188.26927
loss 306.43918
STEP 154 ================================
prereg loss 287.0752 regularization 187.08716
loss 305.7839
STEP 155 ================================
prereg loss 286.6903 regularization 185.91089
loss 305.2814
STEP 156 ================================
prereg loss 286.39328 regularization 184.74309
loss 304.86758
STEP 157 ================================
prereg loss 286.02805 regularization 183.58405
loss 304.38644
STEP 158 ================================
prereg loss 285.59622 regularization 182.4368
loss 303.8399
STEP 159 ================================
prereg loss 285.01126 regularization 181.29703
loss 303.14096
STEP 160 ================================
prereg loss 284.34076 regularization 180.16605
loss 302.35736
STEP 161 ================================
prereg loss 283.74698 regularization 179.0451
loss 301.6515
STEP 162 ================================
prereg loss 283.22314 regularization 177.93134
loss 301.01627
STEP 163 ================================
prereg loss 282.85666 regularization 176.82245
loss 300.5389
STEP 164 ================================
prereg loss 282.6311 regularization 175.7221
loss 300.2033
STEP 165 ================================
prereg loss 282.32495 regularization 174.63196
loss 299.78815
STEP 166 ================================
prereg loss 281.9508 regularization 173.55359
loss 299.30615
STEP 167 ================================
prereg loss 281.4136 regularization 172.47995
loss 298.6616
STEP 168 ================================
prereg loss 280.88547 regularization 171.41234
loss 298.0267
STEP 169 ================================
prereg loss 280.3671 regularization 170.35323
loss 297.4024
STEP 170 ================================
prereg loss 279.7544 regularization 169.3006
loss 296.68445
STEP 171 ================================
prereg loss 279.30972 regularization 168.2543
loss 296.13516
STEP 172 ================================
prereg loss 278.86053 regularization 167.21559
loss 295.5821
STEP 173 ================================
prereg loss 278.4107 regularization 166.18344
loss 295.02905
STEP 174 ================================
prereg loss 278.06122 regularization 165.1597
loss 294.57718
STEP 175 ================================
prereg loss 277.7044 regularization 164.14507
loss 294.11893
STEP 176 ================================
prereg loss 277.33746 regularization 163.1381
loss 293.65128
STEP 177 ================================
prereg loss 276.85965 regularization 162.13728
loss 293.07336
STEP 178 ================================
prereg loss 276.28348 regularization 161.14343
loss 292.39783
STEP 179 ================================
prereg loss 275.66226 regularization 160.15442
loss 291.6777
STEP 180 ================================
prereg loss 275.05356 regularization 159.17484
loss 290.97104
STEP 181 ================================
prereg loss 274.56143 regularization 158.20769
loss 290.3822
STEP 182 ================================
prereg loss 274.0727 regularization 157.24551
loss 289.79724
STEP 183 ================================
prereg loss 273.6477 regularization 156.28595
loss 289.2763
STEP 184 ================================
prereg loss 273.38235 regularization 155.33638
loss 288.916
STEP 185 ================================
prereg loss 273.09958 regularization 154.39572
loss 288.53915
STEP 186 ================================
prereg loss 272.63626 regularization 153.46158
loss 287.98242
STEP 187 ================================
prereg loss 272.01007 regularization 152.53317
loss 287.2634
STEP 188 ================================
prereg loss 271.5085 regularization 151.6098
loss 286.6695
STEP 189 ================================
prereg loss 271.17285 regularization 150.69185
loss 286.24203
STEP 190 ================================
prereg loss 270.72272 regularization 149.78094
loss 285.7008
STEP 191 ================================
prereg loss 270.2739 regularization 148.87404
loss 285.1613
STEP 192 ================================
prereg loss 269.76593 regularization 147.9763
loss 284.56357
STEP 193 ================================
prereg loss 269.1534 regularization 147.08716
loss 283.86212
STEP 194 ================================
prereg loss 268.72025 regularization 146.2034
loss 283.34058
STEP 195 ================================
prereg loss 268.45505 regularization 145.32185
loss 282.98724
STEP 196 ================================
prereg loss 268.17496 regularization 144.45041
loss 282.62
STEP 197 ================================
prereg loss 267.714 regularization 143.58449
loss 282.07245
STEP 198 ================================
prereg loss 267.1929 regularization 142.72557
loss 281.46545
STEP 199 ================================
prereg loss 266.67606 regularization 141.87236
loss 280.86328
STEP 200 ================================
prereg loss 266.05722 regularization 141.02544
loss 280.15976
STEP 201 ================================
prereg loss 265.45743 regularization 140.1884
loss 279.47626
STEP 202 ================================
prereg loss 265.04233 regularization 139.35912
loss 278.97824
STEP 203 ================================
prereg loss 264.73526 regularization 138.52809
loss 278.58807
STEP 204 ================================
prereg loss 264.41257 regularization 137.70323
loss 278.1829
STEP 205 ================================
prereg loss 263.9677 regularization 136.88719
loss 277.65643
STEP 206 ================================
prereg loss 263.4117 regularization 136.07607
loss 277.01932
STEP 207 ================================
prereg loss 262.86987 regularization 135.26884
loss 276.39676
STEP 208 ================================
prereg loss 262.4497 regularization 134.468
loss 275.8965
STEP 209 ================================
prereg loss 262.14035 regularization 133.67126
loss 275.50748
STEP 210 ================================
prereg loss 261.7596 regularization 132.8818
loss 275.0478
STEP 211 ================================
prereg loss 261.431 regularization 132.1013
loss 274.64114
STEP 212 ================================
prereg loss 260.97852 regularization 131.32474
loss 274.111
STEP 213 ================================
prereg loss 260.3544 regularization 130.55249
loss 273.40964
STEP 214 ================================
prereg loss 259.74658 regularization 129.78838
loss 272.72543
STEP 215 ================================
prereg loss 259.3296 regularization 129.02582
loss 272.23218
STEP 216 ================================
prereg loss 259.08728 regularization 128.26947
loss 271.9142
STEP 217 ================================
prereg loss 258.82806 regularization 127.521484
loss 271.5802
STEP 218 ================================
prereg loss 258.49374 regularization 126.77905
loss 271.17166
STEP 219 ================================
prereg loss 257.9726 regularization 126.036644
loss 270.57626
STEP 220 ================================
prereg loss 257.27914 regularization 125.30152
loss 269.8093
STEP 221 ================================
prereg loss 256.66885 regularization 124.57249
loss 269.1261
STEP 222 ================================
prereg loss 256.13843 regularization 123.847404
loss 268.52316
STEP 223 ================================
prereg loss 255.73602 regularization 123.1268
loss 268.0487
STEP 224 ================================
prereg loss 255.51617 regularization 122.40985
loss 267.75717
STEP 225 ================================
prereg loss 255.21733 regularization 121.7041
loss 267.38776
STEP 226 ================================
prereg loss 254.84164 regularization 121.006615
loss 266.9423
STEP 227 ================================
prereg loss 254.33797 regularization 120.311874
loss 266.36914
STEP 228 ================================
prereg loss 253.90048 regularization 119.62232
loss 265.8627
STEP 229 ================================
prereg loss 253.52885 regularization 118.93938
loss 265.4228
STEP 230 ================================
prereg loss 252.9703 regularization 118.25934
loss 264.79623
STEP 231 ================================
prereg loss 252.54958 regularization 117.58343
loss 264.30792
STEP 232 ================================
prereg loss 252.06352 regularization 116.91479
loss 263.755
STEP 233 ================================
prereg loss 251.58168 regularization 116.24845
loss 263.2065
STEP 234 ================================
prereg loss 251.16968 regularization 115.58668
loss 262.72833
STEP 235 ================================
prereg loss 250.75438 regularization 114.92891
loss 262.24728
STEP 236 ================================
prereg loss 250.27902 regularization 114.274826
loss 261.7065
STEP 237 ================================
prereg loss 250.00081 regularization 113.62627
loss 261.36343
STEP 238 ================================
prereg loss 249.58163 regularization 112.9834
loss 260.87997
STEP 239 ================================
prereg loss 249.0342 regularization 112.34129
loss 260.2683
STEP 240 ================================
prereg loss 248.62186 regularization 111.70364
loss 259.79224
STEP 241 ================================
prereg loss 248.33826 regularization 111.076164
loss 259.44586
STEP 242 ================================
prereg loss 248.04356 regularization 110.45256
loss 259.0888
STEP 243 ================================
prereg loss 247.6139 regularization 109.8296
loss 258.59686
STEP 244 ================================
prereg loss 247.1839 regularization 109.21317
loss 258.10522
STEP 245 ================================
prereg loss 246.68883 regularization 108.60037
loss 257.54886
STEP 246 ================================
prereg loss 246.07487 regularization 107.995026
loss 256.8744
STEP 247 ================================
prereg loss 245.47731 regularization 107.39436
loss 256.21674
STEP 248 ================================
prereg loss 245.03035 regularization 106.79596
loss 255.70995
STEP 249 ================================
prereg loss 244.65321 regularization 106.20198
loss 255.2734
STEP 250 ================================
prereg loss 244.46732 regularization 105.61405
loss 255.02872
STEP 251 ================================
prereg loss 244.25826 regularization 105.0261
loss 254.76086
STEP 252 ================================
prereg loss 243.8324 regularization 104.445496
loss 254.27695
STEP 253 ================================
prereg loss 243.21313 regularization 103.87264
loss 253.6004
STEP 254 ================================
prereg loss 242.74872 regularization 103.30118
loss 253.07884
STEP 255 ================================
prereg loss 242.4895 regularization 102.73031
loss 252.76253
STEP 256 ================================
prereg loss 242.08327 regularization 102.16781
loss 252.30005
STEP 257 ================================
prereg loss 241.47551 regularization 101.60834
loss 251.63634
STEP 258 ================================
prereg loss 241.08829 regularization 101.04895
loss 251.19318
STEP 259 ================================
prereg loss 240.83144 regularization 100.496346
loss 250.88107
STEP 260 ================================
prereg loss 240.56223 regularization 99.948494
loss 250.55707
STEP 261 ================================
prereg loss 240.14992 regularization 99.40746
loss 250.09067
STEP 262 ================================
prereg loss 239.7378 regularization 98.87382
loss 249.62517
STEP 263 ================================
prereg loss 239.189 regularization 98.33737
loss 249.02274
STEP 264 ================================
prereg loss 238.58595 regularization 97.80358
loss 248.36632
STEP 265 ================================
prereg loss 238.14165 regularization 97.27635
loss 247.86928
STEP 266 ================================
prereg loss 237.70389 regularization 96.75652
loss 247.37955
STEP 267 ================================
prereg loss 237.34012 regularization 96.237915
loss 246.96391
STEP 268 ================================
prereg loss 237.1751 regularization 95.72226
loss 246.74731
STEP 269 ================================
prereg loss 236.91893 regularization 95.21014
loss 246.43994
STEP 270 ================================
prereg loss 236.51385 regularization 94.70196
loss 245.98405
STEP 271 ================================
prereg loss 235.90279 regularization 94.19942
loss 245.32272
STEP 272 ================================
prereg loss 235.45459 regularization 93.69744
loss 244.82434
STEP 273 ================================
prereg loss 235.22263 regularization 93.20054
loss 244.54268
STEP 274 ================================
prereg loss 234.83372 regularization 92.7095
loss 244.10468
STEP 275 ================================
prereg loss 234.30243 regularization 92.21705
loss 243.52414
STEP 276 ================================
prereg loss 233.92159 regularization 91.731346
loss 243.09473
STEP 277 ================================
prereg loss 233.68387 regularization 91.252846
loss 242.80916
STEP 278 ================================
prereg loss 233.43275 regularization 90.775475
loss 242.5103
STEP 279 ================================
prereg loss 233.03166 regularization 90.29751
loss 242.06142
STEP 280 ================================
prereg loss 232.48802 regularization 89.82641
loss 241.47066
STEP 281 ================================
prereg loss 231.88953 regularization 89.35881
loss 240.82541
STEP 282 ================================
prereg loss 231.31561 regularization 88.89578
loss 240.20518
STEP 283 ================================
prereg loss 230.90208 regularization 88.43252
loss 239.74533
STEP 284 ================================
prereg loss 230.63959 regularization 87.971855
loss 239.43677
STEP 285 ================================
prereg loss 230.43706 regularization 87.51541
loss 239.1886
STEP 286 ================================
prereg loss 230.2817 regularization 87.06739
loss 238.98843
STEP 287 ================================
prereg loss 229.95778 regularization 86.61892
loss 238.61967
STEP 288 ================================
prereg loss 229.41107 regularization 86.17414
loss 238.02849
STEP 289 ================================
prereg loss 228.81644 regularization 85.73478
loss 237.38991
STEP 290 ================================
prereg loss 228.39217 regularization 85.29807
loss 236.92197
STEP 291 ================================
prereg loss 228.19283 regularization 84.86359
loss 236.67918
STEP 292 ================================
prereg loss 227.97664 regularization 84.43585
loss 236.42023
STEP 293 ================================
prereg loss 227.52551 regularization 84.00787
loss 235.9263
STEP 294 ================================
prereg loss 227.15164 regularization 83.584114
loss 235.51006
STEP 295 ================================
prereg loss 226.77388 regularization 83.16543
loss 235.09042
STEP 296 ================================
prereg loss 226.39796 regularization 82.750046
loss 234.67297
STEP 297 ================================
prereg loss 226.02933 regularization 82.34072
loss 234.2634
STEP 298 ================================
prereg loss 225.66069 regularization 81.9337
loss 233.85406
STEP 299 ================================
prereg loss 225.29202 regularization 81.5235
loss 233.44437
STEP 300 ================================
prereg loss 224.85197 regularization 81.11539
loss 232.96352
STEP 301 ================================
prereg loss 224.42009 regularization 80.717606
loss 232.49185
STEP 302 ================================
prereg loss 223.99576 regularization 80.32317
loss 232.02808
STEP 303 ================================
prereg loss 223.6504 regularization 79.92696
loss 231.6431
STEP 304 ================================
prereg loss 223.36943 regularization 79.5355
loss 231.32298
STEP 305 ================================
prereg loss 223.0087 regularization 79.145676
loss 230.92326
STEP 306 ================================
prereg loss 222.65309 regularization 78.76015
loss 230.52911
STEP 307 ================================
prereg loss 222.22694 regularization 78.375824
loss 230.06453
STEP 308 ================================
prereg loss 221.9636 regularization 77.99152
loss 229.76274
STEP 309 ================================
prereg loss 221.76408 regularization 77.61186
loss 229.52527
STEP 310 ================================
prereg loss 221.39398 regularization 77.24037
loss 229.11801
STEP 311 ================================
prereg loss 220.86902 regularization 76.86949
loss 228.55597
STEP 312 ================================
prereg loss 220.35356 regularization 76.50002
loss 228.00357
STEP 313 ================================
prereg loss 220.03648 regularization 76.134445
loss 227.64993
STEP 314 ================================
prereg loss 219.89903 regularization 75.76889
loss 227.47592
STEP 315 ================================
prereg loss 219.77242 regularization 75.4045
loss 227.31287
STEP 316 ================================
prereg loss 219.64856 regularization 75.04732
loss 227.15329
STEP 317 ================================
prereg loss 219.29692 regularization 74.69325
loss 226.76625
STEP 318 ================================
prereg loss 218.81755 regularization 74.33985
loss 226.25154
STEP 319 ================================
prereg loss 218.37088 regularization 73.984985
loss 225.76938
STEP 320 ================================
prereg loss 217.95995 regularization 73.63337
loss 225.32329
STEP 321 ================================
prereg loss 217.81384 regularization 73.28673
loss 225.14252
STEP 322 ================================
prereg loss 217.89865 regularization 72.94383
loss 225.19304
STEP 323 ================================
prereg loss 217.96547 regularization 72.59919
loss 225.22539
STEP 324 ================================
prereg loss 217.94324 regularization 72.25766
loss 225.169
STEP 325 ================================
prereg loss 217.69008 regularization 71.92436
loss 224.8825
STEP 326 ================================
prereg loss 217.45255 regularization 71.59251
loss 224.6118
STEP 327 ================================
prereg loss 217.30205 regularization 71.26029
loss 224.42807
STEP 328 ================================
prereg loss 217.00012 regularization 70.93179
loss 224.0933
STEP 329 ================================
prereg loss 216.62224 regularization 70.605835
loss 223.68282
STEP 330 ================================
prereg loss 216.47757 regularization 70.28075
loss 223.50565
STEP 331 ================================
prereg loss 216.46625 regularization 69.95817
loss 223.46207
STEP 332 ================================
prereg loss 216.57906 regularization 69.638626
loss 223.54292
STEP 333 ================================
prereg loss 216.4958 regularization 69.32037
loss 223.42784
STEP 334 ================================
prereg loss 216.22992 regularization 69.004074
loss 223.13033
STEP 335 ================================
prereg loss 215.79904 regularization 68.684654
loss 222.66751
STEP 336 ================================
prereg loss 215.1451 regularization 68.37187
loss 221.98228
STEP 337 ================================
prereg loss 214.67995 regularization 68.06602
loss 221.48654
STEP 338 ================================
prereg loss 214.3872 regularization 67.76207
loss 221.16342
STEP 339 ================================
prereg loss 214.3488 regularization 67.45595
loss 221.09439
STEP 340 ================================
prereg loss 214.53087 regularization 67.15504
loss 221.24637
STEP 341 ================================
prereg loss 214.45416 regularization 66.85628
loss 221.13979
STEP 342 ================================
prereg loss 214.22365 regularization 66.55795
loss 220.87944
STEP 343 ================================
prereg loss 213.77815 regularization 66.26234
loss 220.40439
STEP 344 ================================
prereg loss 213.36765 regularization 65.96741
loss 219.96439
STEP 345 ================================
prereg loss 213.22282 regularization 65.67605
loss 219.79044
STEP 346 ================================
prereg loss 213.06117 regularization 65.3901
loss 219.60019
STEP 347 ================================
prereg loss 212.8842 regularization 65.10087
loss 219.39429
STEP 348 ================================
prereg loss 212.84354 regularization 64.81102
loss 219.32465
STEP 349 ================================
prereg loss 212.7715 regularization 64.52669
loss 219.22417
STEP 350 ================================
prereg loss 212.6714 regularization 64.24537
loss 219.09593
STEP 351 ================================
prereg loss 212.39668 regularization 63.966415
loss 218.79332
STEP 352 ================================
prereg loss 211.98315 regularization 63.69165
loss 218.35233
STEP 353 ================================
prereg loss 211.53152 regularization 63.416935
loss 217.87321
STEP 354 ================================
prereg loss 211.12433 regularization 63.140903
loss 217.43842
STEP 355 ================================
prereg loss 210.90578 regularization 62.868454
loss 217.19263
STEP 356 ================================
prereg loss 210.86237 regularization 62.599117
loss 217.12228
STEP 357 ================================
prereg loss 210.88786 regularization 62.332905
loss 217.12115
STEP 358 ================================
prereg loss 210.94208 regularization 62.069466
loss 217.14902
STEP 359 ================================
prereg loss 210.79893 regularization 61.80208
loss 216.97914
STEP 360 ================================
prereg loss 210.40492 regularization 61.538532
loss 216.55878
STEP 361 ================================
prereg loss 209.95268 regularization 61.28297
loss 216.08098
STEP 362 ================================
prereg loss 209.67226 regularization 61.026867
loss 215.77495
STEP 363 ================================
prereg loss 209.62074 regularization 60.769028
loss 215.69765
STEP 364 ================================
prereg loss 209.57022 regularization 60.514587
loss 215.62167
STEP 365 ================================
prereg loss 209.28543 regularization 60.262566
loss 215.31169
STEP 366 ================================
prereg loss 209.09189 regularization 60.013596
loss 215.09325
STEP 367 ================================
prereg loss 208.90353 regularization 59.7655
loss 214.88008
STEP 368 ================================
prereg loss 208.72523 regularization 59.518658
loss 214.6771
STEP 369 ================================
prereg loss 208.56818 regularization 59.273315
loss 214.49551
STEP 370 ================================
prereg loss 208.39717 regularization 59.031315
loss 214.30031
STEP 371 ================================
prereg loss 208.21371 regularization 58.789433
loss 214.09265
STEP 372 ================================
prereg loss 207.94531 regularization 58.54997
loss 213.80031
STEP 373 ================================
prereg loss 207.66866 regularization 58.313766
loss 213.50003
STEP 374 ================================
prereg loss 207.38495 regularization 58.077328
loss 213.19269
STEP 375 ================================
prereg loss 207.16862 regularization 57.84092
loss 212.95271
STEP 376 ================================
prereg loss 207.03233 regularization 57.609737
loss 212.7933
STEP 377 ================================
prereg loss 206.83325 regularization 57.379074
loss 212.57117
STEP 378 ================================
prereg loss 206.65657 regularization 57.148663
loss 212.37143
STEP 379 ================================
prereg loss 206.42334 regularization 56.920033
loss 212.11534
STEP 380 ================================
prereg loss 206.36798 regularization 56.69437
loss 212.03741
STEP 381 ================================
prereg loss 206.38286 regularization 56.473
loss 212.03015
STEP 382 ================================
prereg loss 206.20518 regularization 56.255512
loss 211.83073
STEP 383 ================================
prereg loss 205.85399 regularization 56.03488
loss 211.45747
STEP 384 ================================
prereg loss 205.49551 regularization 55.812572
loss 211.07677
STEP 385 ================================
prereg loss 205.30078 regularization 55.598343
loss 210.86061
STEP 386 ================================
prereg loss 205.25359 regularization 55.38637
loss 210.79222
STEP 387 ================================
prereg loss 205.18964 regularization 55.175583
loss 210.7072
STEP 388 ================================
prereg loss 205.13023 regularization 54.967094
loss 210.62694
STEP 389 ================================
prereg loss 204.83864 regularization 54.756176
loss 210.31425
STEP 390 ================================
prereg loss 204.41653 regularization 54.544918
loss 209.87103
STEP 391 ================================
prereg loss 204.02289 regularization 54.336174
loss 209.45651
STEP 392 ================================
prereg loss 203.66077 regularization 54.130814
loss 209.07385
STEP 393 ================================
prereg loss 203.56534 regularization 53.92666
loss 208.95801
STEP 394 ================================
prereg loss 203.67513 regularization 53.72451
loss 209.04758
STEP 395 ================================
prereg loss 203.74582 regularization 53.520756
loss 209.0979
STEP 396 ================================
prereg loss 203.70749 regularization 53.31967
loss 209.03946
STEP 397 ================================
prereg loss 203.41559 regularization 53.125404
loss 208.72813
STEP 398 ================================
prereg loss 203.1175 regularization 52.927715
loss 208.41026
STEP 399 ================================
prereg loss 202.88754 regularization 52.72887
loss 208.16043
STEP 400 ================================
prereg loss 202.51689 regularization 52.534805
loss 207.77037
STEP 401 ================================
prereg loss 202.10922 regularization 52.342148
loss 207.34344
STEP 402 ================================
prereg loss 201.97063 regularization 52.15116
loss 207.18575
STEP 403 ================================
prereg loss 201.9973 regularization 51.959003
loss 207.1932
STEP 404 ================================
prereg loss 202.17621 regularization 51.767483
loss 207.35295
STEP 405 ================================
prereg loss 202.17989 regularization 51.57929
loss 207.33781
STEP 406 ================================
prereg loss 201.99309 regularization 51.39726
loss 207.13281
STEP 407 ================================
prereg loss 201.63455 regularization 51.21098
loss 206.75565
STEP 408 ================================
prereg loss 201.0469 regularization 51.02401
loss 206.1493
STEP 409 ================================
prereg loss 200.64641 regularization 50.843033
loss 205.73071
STEP 410 ================================
prereg loss 200.41618 regularization 50.662556
loss 205.48244
STEP 411 ================================
prereg loss 200.4131 regularization 50.48163
loss 205.46126
STEP 412 ================================
prereg loss 200.63339 regularization 50.302647
loss 205.66365
STEP 413 ================================
prereg loss 200.5941 regularization 50.12298
loss 205.6064
STEP 414 ================================
prereg loss 200.40057 regularization 49.94439
loss 205.39502
STEP 415 ================================
prereg loss 199.99174 regularization 49.770134
loss 204.96877
STEP 416 ================================
prereg loss 199.61598 regularization 49.5958
loss 204.57556
STEP 417 ================================
prereg loss 199.50839 regularization 49.423187
loss 204.45071
STEP 418 ================================
prereg loss 199.3861 regularization 49.253403
loss 204.31143
STEP 419 ================================
prereg loss 199.25053 regularization 49.079212
loss 204.15846
STEP 420 ================================
prereg loss 199.25076 regularization 48.905735
loss 204.14134
STEP 421 ================================
prereg loss 199.21533 regularization 48.7412
loss 204.08945
STEP 422 ================================
prereg loss 199.14789 regularization 48.57896
loss 204.00578
STEP 423 ================================
prereg loss 198.90456 regularization 48.414734
loss 203.74603
STEP 424 ================================
prereg loss 198.52325 regularization 48.253807
loss 203.34863
STEP 425 ================================
prereg loss 198.10812 regularization 48.090744
loss 202.9172
STEP 426 ================================
prereg loss 197.74152 regularization 47.92884
loss 202.5344
STEP 427 ================================
prereg loss 197.56502 regularization 47.77101
loss 202.34212
STEP 428 ================================
prereg loss 197.56439 regularization 47.61563
loss 202.32596
STEP 429 ================================
prereg loss 197.62997 regularization 47.458664
loss 202.37584
STEP 430 ================================
prereg loss 197.71996 regularization 47.303783
loss 202.45033
STEP 431 ================================
prereg loss 197.61143 regularization 47.14672
loss 202.32611
STEP 432 ================================
prereg loss 197.25113 regularization 46.99008
loss 201.95013
STEP 433 ================================
prereg loss 196.83516 regularization 46.839417
loss 201.5191
STEP 434 ================================
prereg loss 196.5907 regularization 46.68938
loss 201.25964
STEP 435 ================================
prereg loss 196.57413 regularization 46.536636
loss 201.22778
STEP 436 ================================
prereg loss 196.56258 regularization 46.389206
loss 201.20149
STEP 437 ================================
prereg loss 196.31764 regularization 46.2451
loss 200.94215
STEP 438 ================================
prereg loss 196.16098 regularization 46.09741
loss 200.77072
STEP 439 ================================
prereg loss 196.00795 regularization 45.95139
loss 200.60309
STEP 440 ================================
prereg loss 195.86295 regularization 45.804497
loss 200.44339
STEP 441 ================================
prereg loss 195.74298 regularization 45.66234
loss 200.30922
STEP 442 ================================
prereg loss 195.61111 regularization 45.524612
loss 200.16357
STEP 443 ================================
prereg loss 195.46837 regularization 45.382088
loss 200.00658
STEP 444 ================================
prereg loss 195.24278 regularization 45.238422
loss 199.76663
STEP 445 ================================
prereg loss 195.00476 regularization 45.09876
loss 199.51463
STEP 446 ================================
prereg loss 194.75578 regularization 44.96233
loss 199.25201
STEP 447 ================================
prereg loss 194.57028 regularization 44.822525
loss 199.05254
STEP 448 ================================
prereg loss 194.46306 regularization 44.685577
loss 198.93161
STEP 449 ================================
prereg loss 194.29796 regularization 44.550682
loss 198.75302
STEP 450 ================================
prereg loss 194.15869 regularization 44.416847
loss 198.60037
STEP 451 ================================
prereg loss 193.96707 regularization 44.28454
loss 198.39552
STEP 452 ================================
prereg loss 193.9528 regularization 44.15276
loss 198.36809
STEP 453 ================================
prereg loss 194.00569 regularization 44.021423
loss 198.40784
STEP 454 ================================
prereg loss 193.86555 regularization 43.891045
loss 198.25465
STEP 455 ================================
prereg loss 193.55171 regularization 43.75915
loss 197.92763
STEP 456 ================================
prereg loss 193.2272 regularization 43.62786
loss 197.59
STEP 457 ================================
prereg loss 193.06503 regularization 43.503403
loss 197.41537
STEP 458 ================================
prereg loss 193.04932 regularization 43.379787
loss 197.3873
STEP 459 ================================
prereg loss 193.01888 regularization 43.252266
loss 197.3441
STEP 460 ================================
prereg loss 192.99661 regularization 43.12809
loss 197.30942
STEP 461 ================================
prereg loss 192.74438 regularization 43.003212
loss 197.04471
STEP 462 ================================
prereg loss 192.36285 regularization 42.87942
loss 196.6508
STEP 463 ================================
prereg loss 192.00764 regularization 42.7574
loss 196.28339
STEP 464 ================================
prereg loss 191.682 regularization 42.63656
loss 195.94566
STEP 465 ================================
prereg loss 191.62154 regularization 42.51742
loss 195.87328
STEP 466 ================================
prereg loss 191.76363 regularization 42.40094
loss 196.00372
STEP 467 ================================
prereg loss 191.86847 regularization 42.28449
loss 196.09692
STEP 468 ================================
prereg loss 191.8674 regularization 42.1671
loss 196.0841
STEP 469 ================================
prereg loss 191.61386 regularization 42.054108
loss 195.81927
STEP 470 ================================
prereg loss 191.3507 regularization 41.94161
loss 195.54486
STEP 471 ================================
prereg loss 191.15147 regularization 41.827835
loss 195.33426
STEP 472 ================================
prereg loss 190.81398 regularization 41.717884
loss 194.98576
STEP 473 ================================
prereg loss 190.44397 regularization 41.606388
loss 194.60461
STEP 474 ================================
prereg loss 190.34096 regularization 41.493507
loss 194.49031
STEP 475 ================================
prereg loss 190.40244 regularization 41.380714
loss 194.54051
STEP 476 ================================
prereg loss 190.6154 regularization 41.269722
loss 194.74237
STEP 477 ================================
prereg loss 190.65501 regularization 41.16351
loss 194.77136
STEP 478 ================================
prereg loss 190.50479 regularization 41.060688
loss 194.61086
STEP 479 ================================
prereg loss 190.18367 regularization 40.950924
loss 194.27876
STEP 480 ================================
prereg loss 189.63564 regularization 40.839523
loss 193.71959
STEP 481 ================================
prereg loss 189.27206 regularization 40.736042
loss 193.34567
STEP 482 ================================
prereg loss 189.0761 regularization 40.63269
loss 193.13936
STEP 483 ================================
prereg loss 189.10341 regularization 40.527573
loss 193.15616
STEP 484 ================================
prereg loss 189.35196 regularization 40.424896
loss 193.39445
STEP 485 ================================
prereg loss 189.34544 regularization 40.322327
loss 193.37767
STEP 486 ================================
prereg loss 189.18715 regularization 40.222557
loss 193.20941
STEP 487 ================================
prereg loss 188.81754 regularization 40.125042
loss 192.83003
STEP 488 ================================
prereg loss 188.4789 regularization 40.02492
loss 192.48138
STEP 489 ================================
prereg loss 188.4052 regularization 39.92485
loss 192.39767
STEP 490 ================================
prereg loss 188.31845 regularization 39.828026
loss 192.30125
STEP 491 ================================
prereg loss 188.21983 regularization 39.72872
loss 192.1927
STEP 492 ================================
prereg loss 188.25415 regularization 39.629463
loss 192.2171
STEP 493 ================================
prereg loss 188.24991 regularization 39.536198
loss 192.20352
STEP 494 ================================
prereg loss 188.21106 regularization 39.440804
loss 192.15514
STEP 495 ================================
prereg loss 187.99823 regularization 39.342724
loss 191.9325
STEP 496 ================================
prereg loss 187.65054 regularization 39.2513
loss 191.57567
STEP 497 ================================
prereg loss 187.2736 regularization 39.15972
loss 191.18958
STEP 498 ================================
prereg loss 186.94753 regularization 39.065495
loss 190.85408
STEP 499 ================================
prereg loss 186.80911 regularization 38.971775
loss 190.70628
STEP 500 ================================
prereg loss 186.84438 regularization 38.87951
loss 190.73233

julia> a = deepcopy(trainable["network_matrix"])
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 7 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "accum-1"   => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.000145953), "accum-1"=>Dict("dict"=>-0.000118451, "tru…
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0266746), "accum-1"=>Dict("dict"=>0.00724047, "true"…
  "compare-1" => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0763112), "accum-1"=>Dict("dict"=>7.66667f-5, "true"=>6…
  "dot-1"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.000280062), "accum-1"=>Dict("dict"=>-8.50308f-5, "true…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0193579), "accum-1"=>Dict("dict"=>0.155729, "true"=>0.0…

julia> open("510-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(a))
           println(f)
           end

julia> serialize("510-steps-matrix.ser", a)

julia> function count(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
           d = 0
           for i in keys(x)
               for j in keys(x[i])
                   for m in keys(x[i][j])
                       for n in keys(x[i][j][m])
                           d += 1
           end end end end
           d
       end
count (generic function with 1 method)

julia> count(a)
932

julia> function count_neg_interval(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, min_lim::Float32, max_lim::Float32, vocal = false)
           d = 0
           for i in keys(x)
               for j in keys(x[i])
                   for m in keys(x[i][j])
                       for n in keys(x[i][j][m])
                           v = x[i][j][m][n]
                           if v <= min_lim || v >= max_lim
                               d += 1
                               if vocal
                                   println(i, " ", j, " ",  m, " ", n, " ", v)
                               end
                           end
           end end end end
           d
       end
count_neg_interval (generic function with 2 methods)

julia> count_neg_interval(a, -0.8f0, 0.8f0)
2

julia> count_neg_interval(a, -0.8f0, 0.8f0, true)
timer timer timer timer 1.0
input timer timer timer 1.0
2

julia> count_neg_interval(a, -0.7f0, 0.7f0)
2

julia> count_neg_interval(a, -0.6f0, 0.6f0)
2

julia> count_neg_interval(a, -0.5f0, 0.5f0)
2

julia> count_neg_interval(a, -0.4f0, 0.4f0)
4

julia> count_neg_interval(a, -0.4f0, 0.4f0, true)
timer timer timer timer 1.0
accum-1 dict-2 compare-1 false -0.48165774
dot-1 dict-2 eos char 0.42419204
input timer timer timer 1.0
4

julia> count_neg_interval(a, -0.3f0, 0.3f0)
11

julia> count_neg_interval(a, -0.3f0, 0.3f0, true)
timer timer timer timer 1.0
accum-1 dict-2 const_1 const_1 -0.3669775
accum-1 dict-2 accum-1 dict 0.3422943
accum-1 dict-2 compare-1 false -0.48165774
compare-1 dict-2 accum-1 dict -0.30546954
compare-1 dict-2 compare-1 false 0.3068506
dot-1 dict-2 eos char 0.42419204
dot-1 dict-1 eos char -0.37849674
input timer timer timer 1.0
norm-1 dict dot-1 dict-1 0.35585737
norm-1 dict norm-1 norm 0.30212817
11

julia> count_neg_interval(a, -0.2f0, 0.2f0)
22

julia> count_neg_interval(a, -0.2f0, 0.2f0, true)
timer timer timer timer 1.0
accum-1 dict-2 const_1 const_1 -0.3669775
accum-1 dict-2 accum-1 dict 0.3422943
accum-1 dict-2 compare-1 false -0.48165774
accum-1 dict-2 dot-1 dot 0.26285893
accum-1 dict-2 norm-1 norm -0.2076287
accum-1 dict-1 dot-1 norm 0.21474217
output dict-2 compare-1 false 0.20753086
output dict-2 dot-1 dict -0.20032229
compare-1 dict-2 accum-1 dict -0.30546954
compare-1 dict-2 compare-1 false 0.3068506
compare-1 dict-1 accum-1 dict 0.2556906
dot-1 dict dot-1 dict-1 -0.21448228
dot-1 dict-2 accum-1 dict 0.26500067
dot-1 dict-2 eos char 0.42419204
dot-1 dict-1 compare-1 false 0.2233898
dot-1 dict-1 input char -0.22596502
dot-1 dict-1 norm-1 norm -0.2358027
dot-1 dict-1 eos char -0.37849674
input timer timer timer 1.0
norm-1 dict dot-1 dict-1 0.35585737
norm-1 dict norm-1 norm 0.30212817
22

julia> count_neg_interval(a, -0.1f0, 0.1f0)
47

julia> count_neg_interval(a, -0.01f0, 0.01f0)
122

julia> count_neg_interval(a, -0.001f0, 0.001f0)
145

julia> count_neg_interval(a, -0.0001f0, 0.0001f0)
436

julia> close(io)
```
