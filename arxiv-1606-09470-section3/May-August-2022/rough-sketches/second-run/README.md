# Second training run, May 30, 2022

Comparing to the first run:
  * reduced the number of neurons 5 fold, number of links 25 fold
  * increased backprop-through-time 4 fold (from 35 time steps to 140; works fast, no delays at this size)
  * instrumented better

The training was much more smooth, less dramatic (no interesting intermediate structures, unlike the first run)
until it had blown up (we need to put some governors at least on functions like `dot` in the recurrent situation,
people invented things like `relu6` for a reason, because recurrent situations are prone to blowing up).

**One thing we do notice below is that convergence speeds up before things blow up**

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
prereg loss NaN regularization 309.74545
loss NaN
STEP 78 ================================
prereg loss NaN regularization NaN
loss NaN
[...]
```
