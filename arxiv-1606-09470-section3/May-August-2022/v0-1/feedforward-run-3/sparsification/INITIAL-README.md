# Sparsification experiment (post-feed-forward-run-3)

---

## Work with the new baseline is continuing in the `post-sparse16-2500.md` file.

---

### Sparsification current track record

```
> python
>>> 983/1486.0
0.66150740242261108
>>> 863/1486.0
0.58075370121130554
>>> 747/1486.0
0.5026917900403769
>>> 645/1486.0
0.43405114401076716
>>> 564/1486.0
0.37954239569313591
>>> 863/983.0
0.87792472024415058
>>> 747/863.0
0.86558516801854002
>>> 645/747.0
0.86345381526104414
>>> 564/645.0
0.87441860465116283
```

This is a very smooth rate, if we continue in this fashion, that's how many rounds one would need:

```
>>> 564*0.87
490.68000000000001
>>> 564*(0.87**2)
426.89159999999998
>>> 564*(0.87**3)
371.395692
>>> 564*(0.87**4)
323.11425204
>>> 564*(0.87**5)
281.10939927480001
>>> 564*(0.87**6)
244.565177369076
>>> 564*(0.87**7)
212.7717043110961
>>> 564*(0.87**8)
185.11138275065363
>>> 564*(0.87**9)
161.04690299306867
>>> 564*(0.87**10)
140.11080560396974
>>> 564*(0.87**20)
34.806804693250697
>>> 564*(0.87**30)
8.6468252589918251
```

(This is just a prediction of the road ahead at this point, we don't really know
how this would work.)

---

Increase L1 regularization 10-fold (perhaps, we should increase it more), then start with
cutting about a third of the weights and doing 100 training steps.

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0373403), "input"=>Dict("char"=>-0.659739), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0107552), "eos"=>Dict("char"=>-0.0898339), "norm-2-1"=>D…
  "norm-2-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0365615), "const_1"=>Dict("const_1"=>-0.0120121), "accum…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.785447), "const_1"=>Dict("const_1"=>-0.0231249), "accum…
  "norm-3-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-1.08927f-5), "eos"=>Dict("char"=>-0.0298369), "norm-2-1"=>…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.162649), "eos"=>Dict("char"=>-0.0574637), "norm-2-1"=>…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.149583), "eos"=>Dict("char"=>0.0248046), "norm-2-1"=>D…
  "norm-4-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-5.20143f-5), "eos"=>Dict("char"=>-0.0697002), "norm-2-1"=>…
  "norm-5-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.00011641), "eos"=>Dict("char"=>0.0106769), "norm-2-1"=>Di…
  "accum-1-1"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.00374793), "input"=>Dict("char"=>0.401236), "eos"=>…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0953306), "input"=>Dict("char"=>0.000458776), "eos…
  "norm-1-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-2.26905f-5), "input"=>Dict("char"=>0.154205), "eos"=>D…
  "norm-4-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>8.39625f-5), "eos"=>Dict("char"=>-0.0622646), "norm-2-1"=>D…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18436), "eos"=>Dict("char"=>0.0902524), "norm-2-1"=>Dic…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.312614), "eos"=>Dict("char"=>6.53708f-5), "norm-2-1"=>D…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.109429), "eos"=>Dict("char"=>3.00823f-5), "norm-2-1"=>…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.092883), "input"=>Dict("char"=>0.931947), "eos"=>Di…
  "output"      => Dict("dict-2"=>Dict("eos"=>Dict("char"=>0.000258934), "norm-2-1"=>Dict("norm"=>0.0488535), "dot-2-2"…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.574602), "const_1"=>Dict("const_1"=>0.0539226), "accum-1-…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.135357), "const_1"=>Dict("const_1"=>-0.000848358), "acc…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.585263), "const_1"=>Dict("const_1"=>-0.135658), "accum-…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0338988), "eos"=>Dict("char"=>-0.000221385), "norm-2-1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0775538), "eos"=>Dict("char"=>-3.46814f-5), "norm-2-1"=…
  "accum-5-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-8.16594f-5), "eos"=>Dict("char"=>-9.62556f-5), "norm-2-1…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.268588), "eos"=>Dict("char"=>0.000128406), "norm-2-1"=…
  ⋮             => ⋮

julia> sparse1 = sparsecopy(trainable["network_matrix"], 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0373403), "input"=>Dict("char"=>-0.659739), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0107552), "compare-3-2"=>Dict("true"=>0.585676, "false"=…
  "norm-2-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0365615), "const_1"=>Dict("const_1"=>-0.0120121), "accum…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.785447), "dot-1-2"=>Dict("dot"=>0.0827358), "const_1"=>…
  "norm-3-1"    => Dict("dict"=>Dict("compare-2-1"=>Dict("false"=>-0.0222607), "compare-2-2"=>Dict("true"=>-0.0280498),…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.162649), "norm-2-1"=>Dict("norm"=>0.13021), "dot-2-2"=…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.149583), "norm-2-1"=>Dict("norm"=>0.115444), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("compare-3-2"=>Dict("false"=>0.0740114), "norm-2-1"=>Dict("norm"=>0.11059), "dot-2…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>-0.0143552), "accum-3-1"=>Dict("dict"=>-0.0381838), "acc…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.401236), "eos"=>Dict("char"=>-0.0155596)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0953306)), "dict-1"=>Dict("const_1"=>Dict("const_1…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.154205), "eos"=>Dict("char"=>-0.154139)))
  "norm-4-1"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0622646), "accum-3-1"=>Dict("dict"=>0.0989448), "dot-3-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18436), "compare-3-2"=>Dict("true"=>0.245485, "false"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.312614), "norm-2-1"=>Dict("norm"=>0.251639), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.109429), "norm-2-1"=>Dict("norm"=>0.251355), "dot-2-2"…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.092883), "input"=>Dict("char"=>0.931947), "eos"=>Di…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.147797), "norm-2-1"=>Dict("norm"=>0.0488535), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.574602), "dot-1-2"=>Dict("dot"=>-0.186978), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.135357), "dot-1-2"=>Dict("dot"=>-0.145245), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.585263), "dot-1-2"=>Dict("dot"=>-0.221863), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0338988), "norm-2-1"=>Dict("norm"=>0.222862), "dot-2-2…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0775538), "norm-2-1"=>Dict("norm"=>-0.323644), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.14905), "norm-3-1"=>Dict("norm"=>0.0745653), "norm-4-…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.268588), "norm-2-1"=>Dict("norm"=>-0.101266), "dot-2-2…
  ⋮             => ⋮

julia> count(sparse1)
983

julia> trainable["network_matrix"] = sparse1
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0373403), "input"=>Dict("char"=>-0.659739), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0107552), "compare-3-2"=>Dict("true"=>0.585676, "false"=…
  "norm-2-1"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.0365615), "const_1"=>Dict("const_1"=>-0.0120121), "accum…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.785447), "dot-1-2"=>Dict("dot"=>0.0827358), "const_1"=>…
  "norm-3-1"    => Dict("dict"=>Dict("compare-2-1"=>Dict("false"=>-0.0222607), "compare-2-2"=>Dict("true"=>-0.0280498),…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.162649), "norm-2-1"=>Dict("norm"=>0.13021), "dot-2-2"=…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.149583), "norm-2-1"=>Dict("norm"=>0.115444), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("compare-3-2"=>Dict("false"=>0.0740114), "norm-2-1"=>Dict("norm"=>0.11059), "dot-2…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>-0.0143552), "accum-3-1"=>Dict("dict"=>-0.0381838), "acc…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.401236), "eos"=>Dict("char"=>-0.0155596)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0953306)), "dict-1"=>Dict("const_1"=>Dict("const_1…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.154205), "eos"=>Dict("char"=>-0.154139)))
  "norm-4-1"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0622646), "accum-3-1"=>Dict("dict"=>0.0989448), "dot-3-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18436), "compare-3-2"=>Dict("true"=>0.245485, "false"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.312614), "norm-2-1"=>Dict("norm"=>0.251639), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.109429), "norm-2-1"=>Dict("norm"=>0.251355), "dot-2-2"…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.092883), "input"=>Dict("char"=>0.931947), "eos"=>Di…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.147797), "norm-2-1"=>Dict("norm"=>0.0488535), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.574602), "dot-1-2"=>Dict("dot"=>-0.186978), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.135357), "dot-1-2"=>Dict("dot"=>-0.145245), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.585263), "dot-1-2"=>Dict("dot"=>-0.221863), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0338988), "norm-2-1"=>Dict("norm"=>0.222862), "dot-2-2…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0775538), "norm-2-1"=>Dict("norm"=>-0.323644), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.14905), "norm-3-1"=>Dict("norm"=>0.0745653), "norm-4-…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.268588), "norm-2-1"=>Dict("norm"=>-0.101266), "dot-2-2…
  ⋮             => ⋮

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T12:13:45.745
STEP 1 ================================
prereg loss 44.1151 reg_l1 119.21201 reg_l2 35.95371
loss 45.30722
STEP 2 ================================
prereg loss 10.105582 reg_l1 119.17498 reg_l2 35.92133
loss 11.297332
STEP 3 ================================
prereg loss 8.398633 reg_l1 119.12842 reg_l2 35.90804
loss 9.589917
STEP 4 ================================
prereg loss 15.799606 reg_l1 119.04268 reg_l2 35.897526
loss 16.990032
STEP 5 ================================
prereg loss 13.6968565 reg_l1 118.958565 reg_l2 35.895714
loss 14.886442
STEP 6 ================================
prereg loss 7.7505703 reg_l1 118.875046 reg_l2 35.89918
loss 8.939321
STEP 7 ================================
prereg loss 3.5636246 reg_l1 118.7884 reg_l2 35.904266
loss 4.7515087
STEP 8 ================================
prereg loss 3.4919305 reg_l1 118.69803 reg_l2 35.90715
loss 4.6789107
STEP 9 ================================
prereg loss 5.319251 reg_l1 118.60515 reg_l2 35.905045
loss 6.5053024
STEP 10 ================================
prereg loss 6.7320576 reg_l1 118.509834 reg_l2 35.89736
loss 7.9171557
STEP 11 ================================
prereg loss 6.3630834 reg_l1 118.40978 reg_l2 35.884216
loss 7.547181
STEP 12 ================================
prereg loss 4.3522315 reg_l1 118.30699 reg_l2 35.866863
loss 5.535301
STEP 13 ================================
prereg loss 2.1836412 reg_l1 118.201744 reg_l2 35.846603
loss 3.3656588
STEP 14 ================================
prereg loss 1.127064 reg_l1 118.09628 reg_l2 35.824806
loss 2.3080268
STEP 15 ================================
prereg loss 1.2998699 reg_l1 117.99275 reg_l2 35.802807
loss 2.4797974
STEP 16 ================================
prereg loss 2.1516168 reg_l1 117.88791 reg_l2 35.781586
loss 3.3304958
STEP 17 ================================
prereg loss 2.9217927 reg_l1 117.783035 reg_l2 35.761784
loss 4.099623
STEP 18 ================================
prereg loss 3.0831184 reg_l1 117.67758 reg_l2 35.743767
loss 4.2598944
STEP 19 ================================
prereg loss 2.7024527 reg_l1 117.57255 reg_l2 35.7275
loss 3.8781781
STEP 20 ================================
prereg loss 2.2359385 reg_l1 117.46911 reg_l2 35.712563
loss 3.4106297
STEP 21 ================================
prereg loss 2.1603522 reg_l1 117.36537 reg_l2 35.698288
loss 3.3340058
STEP 22 ================================
prereg loss 2.5394275 reg_l1 117.262764 reg_l2 35.683926
loss 3.7120552
STEP 23 ================================
prereg loss 2.8724537 reg_l1 117.16145 reg_l2 35.6689
loss 4.0440683
STEP 24 ================================
prereg loss 2.6656055 reg_l1 117.05984 reg_l2 35.652813
loss 3.836204
STEP 25 ================================
prereg loss 1.9489028 reg_l1 116.9581 reg_l2 35.63565
loss 3.1184838
STEP 26 ================================
prereg loss 1.0817076 reg_l1 116.85705 reg_l2 35.617573
loss 2.250278
STEP 27 ================================
prereg loss 0.4356414 reg_l1 116.75515 reg_l2 35.59893
loss 1.6031929
STEP 28 ================================
prereg loss 0.2371994 reg_l1 116.65448 reg_l2 35.580402
loss 1.4037442
STEP 29 ================================
prereg loss 0.5040857 reg_l1 116.55682 reg_l2 35.56279
loss 1.6696539
STEP 30 ================================
prereg loss 1.014067 reg_l1 116.462395 reg_l2 35.546814
loss 2.178691
STEP 31 ================================
prereg loss 1.5063128 reg_l1 116.370415 reg_l2 35.53263
loss 2.670017
STEP 32 ================================
prereg loss 1.6444161 reg_l1 116.281586 reg_l2 35.52019
loss 2.807232
STEP 33 ================================
prereg loss 1.3783411 reg_l1 116.19387 reg_l2 35.509083
loss 2.5402799
STEP 34 ================================
prereg loss 1.0534749 reg_l1 116.109695 reg_l2 35.49913
loss 2.214572
STEP 35 ================================
prereg loss 0.8499824 reg_l1 116.02727 reg_l2 35.489895
loss 2.0102549
STEP 36 ================================
prereg loss 0.75964093 reg_l1 115.94788 reg_l2 35.480865
loss 1.9191197
STEP 37 ================================
prereg loss 0.69074076 reg_l1 115.870995 reg_l2 35.471413
loss 1.8494506
STEP 38 ================================
prereg loss 0.55373013 reg_l1 115.79435 reg_l2 35.461082
loss 1.7116736
STEP 39 ================================
prereg loss 0.3961035 reg_l1 115.719055 reg_l2 35.450005
loss 1.5532941
STEP 40 ================================
prereg loss 0.24978688 reg_l1 115.641266 reg_l2 35.43851
loss 1.4061995
STEP 41 ================================
prereg loss 0.1855621 reg_l1 115.56147 reg_l2 35.426815
loss 1.3411769
STEP 42 ================================
prereg loss 0.24129133 reg_l1 115.483864 reg_l2 35.415085
loss 1.3961298
STEP 43 ================================
prereg loss 0.31520417 reg_l1 115.40786 reg_l2 35.403194
loss 1.4692827
STEP 44 ================================
prereg loss 0.3535293 reg_l1 115.330666 reg_l2 35.39137
loss 1.5068359
STEP 45 ================================
prereg loss 0.33080786 reg_l1 115.25147 reg_l2 35.379677
loss 1.4833226
STEP 46 ================================
prereg loss 0.27079725 reg_l1 115.173515 reg_l2 35.367973
loss 1.4225324
STEP 47 ================================
prereg loss 0.24401943 reg_l1 115.09484 reg_l2 35.356026
loss 1.3949678
STEP 48 ================================
prereg loss 0.25308204 reg_l1 115.01519 reg_l2 35.343536
loss 1.4032339
STEP 49 ================================
prereg loss 0.27667785 reg_l1 114.93667 reg_l2 35.33044
loss 1.4260445
STEP 50 ================================
prereg loss 0.27547836 reg_l1 114.85951 reg_l2 35.31668
loss 1.4240735
STEP 51 ================================
prereg loss 0.25738755 reg_l1 114.78643 reg_l2 35.302383
loss 1.4052517
STEP 52 ================================
prereg loss 0.24439114 reg_l1 114.711754 reg_l2 35.287518
loss 1.3915086
STEP 53 ================================
prereg loss 0.21616563 reg_l1 114.637115 reg_l2 35.272236
loss 1.3625368
STEP 54 ================================
prereg loss 0.18113995 reg_l1 114.56148 reg_l2 35.2568
loss 1.3267547
STEP 55 ================================
prereg loss 0.15823758 reg_l1 114.48671 reg_l2 35.241375
loss 1.3031046
STEP 56 ================================
prereg loss 0.17330335 reg_l1 114.4122 reg_l2 35.226162
loss 1.3174254
STEP 57 ================================
prereg loss 0.19575858 reg_l1 114.339775 reg_l2 35.21121
loss 1.3391563
STEP 58 ================================
prereg loss 0.18545403 reg_l1 114.26997 reg_l2 35.19654
loss 1.3281537
STEP 59 ================================
prereg loss 0.16591066 reg_l1 114.202156 reg_l2 35.182186
loss 1.3079321
STEP 60 ================================
prereg loss 0.15079793 reg_l1 114.13206 reg_l2 35.16809
loss 1.2921185
STEP 61 ================================
prereg loss 0.14253446 reg_l1 114.06583 reg_l2 35.15409
loss 1.2831928
STEP 62 ================================
prereg loss 0.1364136 reg_l1 114.00537 reg_l2 35.140125
loss 1.2764672
STEP 63 ================================
prereg loss 0.12987687 reg_l1 113.94373 reg_l2 35.1261
loss 1.2693142
STEP 64 ================================
prereg loss 0.13043895 reg_l1 113.88434 reg_l2 35.111958
loss 1.2692823
STEP 65 ================================
prereg loss 0.1291256 reg_l1 113.82397 reg_l2 35.097866
loss 1.2673652
STEP 66 ================================
prereg loss 0.13073933 reg_l1 113.76298 reg_l2 35.083916
loss 1.2683691
STEP 67 ================================
prereg loss 0.13453412 reg_l1 113.703224 reg_l2 35.07
loss 1.2715664
STEP 68 ================================
prereg loss 0.13918816 reg_l1 113.64451 reg_l2 35.056232
loss 1.2756332
STEP 69 ================================
prereg loss 0.13951217 reg_l1 113.58836 reg_l2 35.04274
loss 1.2753958
STEP 70 ================================
prereg loss 0.13226083 reg_l1 113.530106 reg_l2 35.029522
loss 1.2675618
STEP 71 ================================
prereg loss 0.12659952 reg_l1 113.4698 reg_l2 35.01655
loss 1.2612976
STEP 72 ================================
prereg loss 0.12350721 reg_l1 113.40898 reg_l2 35.00378
loss 1.2575971
STEP 73 ================================
prereg loss 0.122077905 reg_l1 113.3498 reg_l2 34.991154
loss 1.2555759
STEP 74 ================================
prereg loss 0.116936065 reg_l1 113.29219 reg_l2 34.978764
loss 1.249858
STEP 75 ================================
prereg loss 0.11493589 reg_l1 113.23306 reg_l2 34.96656
loss 1.2472665
STEP 76 ================================
prereg loss 0.115323916 reg_l1 113.17557 reg_l2 34.954514
loss 1.2470796
STEP 77 ================================
prereg loss 0.11639055 reg_l1 113.12252 reg_l2 34.942657
loss 1.2476158
STEP 78 ================================
prereg loss 0.1174559 reg_l1 113.07398 reg_l2 34.93105
loss 1.2481956
STEP 79 ================================
prereg loss 0.11767113 reg_l1 113.0257 reg_l2 34.91966
loss 1.2479281
STEP 80 ================================
prereg loss 0.11328385 reg_l1 112.97541 reg_l2 34.908344
loss 1.2430379
STEP 81 ================================
prereg loss 0.10516325 reg_l1 112.92432 reg_l2 34.89712
loss 1.2344064
STEP 82 ================================
prereg loss 0.099821925 reg_l1 112.87375 reg_l2 34.885918
loss 1.2285594
STEP 83 ================================
prereg loss 0.09706068 reg_l1 112.82244 reg_l2 34.87463
loss 1.225285
STEP 84 ================================
prereg loss 0.096357115 reg_l1 112.77288 reg_l2 34.863083
loss 1.2240859
STEP 85 ================================
prereg loss 0.09671295 reg_l1 112.72456 reg_l2 34.851307
loss 1.2239586
STEP 86 ================================
prereg loss 0.097830296 reg_l1 112.67662 reg_l2 34.839264
loss 1.2245965
STEP 87 ================================
prereg loss 0.096735686 reg_l1 112.62748 reg_l2 34.827003
loss 1.2230105
STEP 88 ================================
prereg loss 0.095528655 reg_l1 112.57681 reg_l2 34.814465
loss 1.2212967
STEP 89 ================================
prereg loss 0.09424759 reg_l1 112.526276 reg_l2 34.801697
loss 1.2195103
STEP 90 ================================
prereg loss 0.092684135 reg_l1 112.47402 reg_l2 34.78873
loss 1.2174244
STEP 91 ================================
prereg loss 0.08923737 reg_l1 112.41995 reg_l2 34.775597
loss 1.2134368
STEP 92 ================================
prereg loss 0.08682969 reg_l1 112.369835 reg_l2 34.762276
loss 1.210528
STEP 93 ================================
prereg loss 0.08582337 reg_l1 112.321434 reg_l2 34.748768
loss 1.2090378
STEP 94 ================================
prereg loss 0.08611901 reg_l1 112.27215 reg_l2 34.735023
loss 1.2088405
STEP 95 ================================
prereg loss 0.08652455 reg_l1 112.220436 reg_l2 34.72111
loss 1.2087289
STEP 96 ================================
prereg loss 0.08777406 reg_l1 112.168686 reg_l2 34.706993
loss 1.2094609
STEP 97 ================================
prereg loss 0.08822353 reg_l1 112.118225 reg_l2 34.692642
loss 1.2094058
STEP 98 ================================
prereg loss 0.088087715 reg_l1 112.069275 reg_l2 34.678112
loss 1.2087804
STEP 99 ================================
prereg loss 0.08750709 reg_l1 112.02228 reg_l2 34.663464
loss 1.2077299
STEP 100 ================================
prereg loss 0.08637122 reg_l1 111.980156 reg_l2 34.64861
loss 1.2061727
2022-06-26T13:09:52.400

julia> serialize("sparse1-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse1-after-100-steps-opt.ser", opt)

julia> count_interval(sparse1, -0.001f0, 0.001f0)
79

julia> count_interval(sparse1, -0.01f0, 0.01f0)
120

julia> count_interval(sparse1, -0.02f0, 0.02f0)
199

julia> sparse2 = sparsecopy(trainable["network_matrix"], 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0381937), "input"=>Dict("char"=>-0.659814), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0880069), "norm-2-1"=>Dict("norm"=>0.5149), "dot-2-2"=>Dict…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0555578), "accum-1-2"=>Dict("dict"=>0.270597), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.762205), "dot-1-2"=>Dict("dot"=>0.0621671), "const_1"=>…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.198967), "accum-1-1"=>Dict("dict"=>0.125556), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0798136), "norm-2-1"=>Dict("norm"=>0.119047), "dot-2-2…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.142183), "norm-2-1"=>Dict("norm"=>0.117732), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.100037), "dot-2-2"=>Dict("dot"=>-0.026358), "norm-3-1"=…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>-0.0172446), "accum-3-1"=>Dict("dict"=>-0.0353115), "acc…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.402526), "eos"=>Dict("char"=>-0.015288)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0960263)), "dict-1"=>Dict("const_1"=>Dict("const_1…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.153783), "eos"=>Dict("char"=>-0.155504)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113524), "accum-3-1"=>Dict("dict"=>0.103822), "accum-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.192712), "eos"=>Dict("char"=>0.090299), "dot-2-2"=>Dict…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.311745), "norm-2-1"=>Dict("norm"=>0.254284), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0911772), "norm-2-1"=>Dict("norm"=>0.246793), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.09257), "input"=>Dict("char"=>0.9272), "eos"=>Dict(…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.148033), "norm-2-1"=>Dict("norm"=>0.0484099), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.572239), "dot-1-2"=>Dict("dot"=>-0.188896), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.137054), "dot-1-2"=>Dict("dot"=>-0.142973), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.577706), "dot-1-2"=>Dict("dot"=>-0.213213), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0338169), "norm-2-1"=>Dict("norm"=>0.226362), "dot-2-2…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0789405), "norm-2-1"=>Dict("norm"=>-0.322813), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.145884), "norm-3-1"=>Dict("norm"=>0.0720711), "norm-4…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.269177), "norm-2-1"=>Dict("norm"=>-0.101793), "dot-2-2…
  ⋮             => ⋮

julia> count(sparse2)
863

julia> trainable["network_matrix"] = sparse2
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0381937), "input"=>Dict("char"=>-0.659814), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0880069), "norm-2-1"=>Dict("norm"=>0.5149), "dot-2-2"=>Dict…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0555578), "accum-1-2"=>Dict("dict"=>0.270597), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.762205), "dot-1-2"=>Dict("dot"=>0.0621671), "const_1"=>…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.198967), "accum-1-1"=>Dict("dict"=>0.125556), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0798136), "norm-2-1"=>Dict("norm"=>0.119047), "dot-2-2…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.142183), "norm-2-1"=>Dict("norm"=>0.117732), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.100037), "dot-2-2"=>Dict("dot"=>-0.026358), "norm-3-1"=…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>-0.0172446), "accum-3-1"=>Dict("dict"=>-0.0353115), "acc…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.402526), "eos"=>Dict("char"=>-0.015288)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0960263)), "dict-1"=>Dict("const_1"=>Dict("const_1…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.153783), "eos"=>Dict("char"=>-0.155504)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113524), "accum-3-1"=>Dict("dict"=>0.103822), "accum-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.192712), "eos"=>Dict("char"=>0.090299), "dot-2-2"=>Dict…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.311745), "norm-2-1"=>Dict("norm"=>0.254284), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0911772), "norm-2-1"=>Dict("norm"=>0.246793), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.09257), "input"=>Dict("char"=>0.9272), "eos"=>Dict(…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.148033), "norm-2-1"=>Dict("norm"=>0.0484099), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.572239), "dot-1-2"=>Dict("dot"=>-0.188896), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.137054), "dot-1-2"=>Dict("dot"=>-0.142973), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.577706), "dot-1-2"=>Dict("dot"=>-0.213213), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0338169), "norm-2-1"=>Dict("norm"=>0.226362), "dot-2-2…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0789405), "norm-2-1"=>Dict("norm"=>-0.322813), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.145884), "norm-3-1"=>Dict("norm"=>0.0720711), "norm-4…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.269177), "norm-2-1"=>Dict("norm"=>-0.101793), "dot-2-2…
  ⋮             => ⋮

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())
```

Let's do a further 100 steps with the same configuration after we've cut another 12%
of the current count.

```
julia> steps!(100)
2022-06-26T13:21:22.697
STEP 1 ================================
prereg loss 11.720483 reg_l1 111.69034 reg_l2 34.631832
loss 12.837386
STEP 2 ================================
prereg loss 2.0157142 reg_l1 111.699295 reg_l2 34.652287
loss 3.132707
STEP 3 ================================
prereg loss 5.122853 reg_l1 111.67449 reg_l2 34.65052
loss 6.239598
STEP 4 ================================
prereg loss 3.8104625 reg_l1 111.63554 reg_l2 34.63566
loss 4.926818
STEP 5 ================================
prereg loss 1.8371193 reg_l1 111.602135 reg_l2 34.61358
loss 2.9531407
STEP 6 ================================
prereg loss 0.8089798 reg_l1 111.56013 reg_l2 34.59344
loss 1.924581
STEP 7 ================================
prereg loss 2.6664863 reg_l1 111.50921 reg_l2 34.573338
loss 3.7815783
STEP 8 ================================
prereg loss 2.247309 reg_l1 111.450905 reg_l2 34.551468
loss 3.3618178
STEP 9 ================================
prereg loss 1.8030269 reg_l1 111.38895 reg_l2 34.53201
loss 2.9169164
STEP 10 ================================
prereg loss 0.64140266 reg_l1 111.3253 reg_l2 34.516582
loss 1.7546556
STEP 11 ================================
prereg loss 0.6424634 reg_l1 111.26131 reg_l2 34.50216
loss 1.7550764
STEP 12 ================================
prereg loss 1.1899208 reg_l1 111.20018 reg_l2 34.48573
loss 2.3019226
STEP 13 ================================
prereg loss 1.5225685 reg_l1 111.1433 reg_l2 34.466927
loss 2.6340015
STEP 14 ================================
prereg loss 1.3474215 reg_l1 111.09098 reg_l2 34.447826
loss 2.4583313
STEP 15 ================================
prereg loss 0.55984455 reg_l1 111.0433 reg_l2 34.429897
loss 1.6702776
STEP 16 ================================
prereg loss 0.70009375 reg_l1 111.0003 reg_l2 34.412292
loss 1.8100967
STEP 17 ================================
prereg loss 1.0180898 reg_l1 110.9593 reg_l2 34.39432
loss 2.1276827
STEP 18 ================================
prereg loss 1.1286898 reg_l1 110.9183 reg_l2 34.376682
loss 2.2378726
STEP 19 ================================
prereg loss 1.1029413 reg_l1 110.87707 reg_l2 34.361458
loss 2.211712
STEP 20 ================================
prereg loss 0.6683364 reg_l1 110.83527 reg_l2 34.348667
loss 1.776689
STEP 21 ================================
prereg loss 0.40476945 reg_l1 110.7942 reg_l2 34.336845
loss 1.5127114
STEP 22 ================================
prereg loss 0.27242792 reg_l1 110.75203 reg_l2 34.32396
loss 1.3799481
STEP 23 ================================
prereg loss 0.4375498 reg_l1 110.709175 reg_l2 34.309425
loss 1.5446416
STEP 24 ================================
prereg loss 0.768408 reg_l1 110.66606 reg_l2 34.29399
loss 1.8750687
STEP 25 ================================
prereg loss 0.61303216 reg_l1 110.62335 reg_l2 34.277985
loss 1.7192657
STEP 26 ================================
prereg loss 0.38647464 reg_l1 110.58125 reg_l2 34.260765
loss 1.4922872
STEP 27 ================================
prereg loss 0.23700622 reg_l1 110.53786 reg_l2 34.241817
loss 1.3423847
STEP 28 ================================
prereg loss 0.24873528 reg_l1 110.49296 reg_l2 34.221878
loss 1.3536649
STEP 29 ================================
prereg loss 0.37337634 reg_l1 110.44573 reg_l2 34.202328
loss 1.4778336
STEP 30 ================================
prereg loss 0.34727478 reg_l1 110.3991 reg_l2 34.184242
loss 1.4512658
STEP 31 ================================
prereg loss 0.29711318 reg_l1 110.35043 reg_l2 34.16695
loss 1.4006175
STEP 32 ================================
prereg loss 0.18557131 reg_l1 110.29998 reg_l2 34.149372
loss 1.2885711
STEP 33 ================================
prereg loss 0.18984273 reg_l1 110.24964 reg_l2 34.131645
loss 1.2923391
STEP 34 ================================
prereg loss 0.288148 reg_l1 110.20074 reg_l2 34.11421
loss 1.3901553
STEP 35 ================================
prereg loss 0.28461802 reg_l1 110.153305 reg_l2 34.097015
loss 1.3861511
STEP 36 ================================
prereg loss 0.249304 reg_l1 110.106995 reg_l2 34.079166
loss 1.350374
STEP 37 ================================
prereg loss 0.16577983 reg_l1 110.06134 reg_l2 34.060318
loss 1.2663932
STEP 38 ================================
prereg loss 0.15856978 reg_l1 110.01681 reg_l2 34.041176
loss 1.2587379
STEP 39 ================================
prereg loss 0.19429559 reg_l1 109.97286 reg_l2 34.02265
loss 1.2940242
STEP 40 ================================
prereg loss 0.22406635 reg_l1 109.927704 reg_l2 34.00467
loss 1.3233434
STEP 41 ================================
prereg loss 0.20209758 reg_l1 109.88108 reg_l2 33.986534
loss 1.3009083
STEP 42 ================================
prereg loss 0.13957095 reg_l1 109.83261 reg_l2 33.967846
loss 1.237897
STEP 43 ================================
prereg loss 0.14815454 reg_l1 109.78338 reg_l2 33.948982
loss 1.2459882
STEP 44 ================================
prereg loss 0.17181672 reg_l1 109.73602 reg_l2 33.929916
loss 1.269177
STEP 45 ================================
prereg loss 0.19484566 reg_l1 109.688354 reg_l2 33.91018
loss 1.2917292
STEP 46 ================================
prereg loss 0.17189138 reg_l1 109.64065 reg_l2 33.88928
loss 1.2682978
STEP 47 ================================
prereg loss 0.14068188 reg_l1 109.591606 reg_l2 33.867634
loss 1.2365979
STEP 48 ================================
prereg loss 0.12444168 reg_l1 109.54117 reg_l2 33.84596
loss 1.2198534
STEP 49 ================================
prereg loss 0.14241165 reg_l1 109.48982 reg_l2 33.824684
loss 1.2373099
STEP 50 ================================
prereg loss 0.15974565 reg_l1 109.437904 reg_l2 33.80358
loss 1.2541248
STEP 51 ================================
prereg loss 0.14205523 reg_l1 109.38534 reg_l2 33.782722
loss 1.2359086
STEP 52 ================================
prereg loss 0.12798636 reg_l1 109.33333 reg_l2 33.762516
loss 1.2213196
STEP 53 ================================
prereg loss 0.12162186 reg_l1 109.28262 reg_l2 33.743015
loss 1.2144481
STEP 54 ================================
prereg loss 0.13249172 reg_l1 109.23296 reg_l2 33.72359
loss 1.2248213
STEP 55 ================================
prereg loss 0.1318525 reg_l1 109.18435 reg_l2 33.703793
loss 1.223696
STEP 56 ================================
prereg loss 0.124520324 reg_l1 109.135796 reg_l2 33.68387
loss 1.2158782
STEP 57 ================================
prereg loss 0.110659294 reg_l1 109.08701 reg_l2 33.664116
loss 1.2015294
STEP 58 ================================
prereg loss 0.11652425 reg_l1 109.03845 reg_l2 33.64439
loss 1.2069087
STEP 59 ================================
prereg loss 0.12041054 reg_l1 108.98978 reg_l2 33.62449
loss 1.2103083
STEP 60 ================================
prereg loss 0.11874522 reg_l1 108.94155 reg_l2 33.604527
loss 1.2081606
STEP 61 ================================
prereg loss 0.11155866 reg_l1 108.89246 reg_l2 33.584843
loss 1.2004833
STEP 62 ================================
prereg loss 0.11059259 reg_l1 108.84334 reg_l2 33.565075
loss 1.199026
STEP 63 ================================
prereg loss 0.11385752 reg_l1 108.79401 reg_l2 33.54484
loss 1.2017976
STEP 64 ================================
prereg loss 0.1192519 reg_l1 108.74305 reg_l2 33.524166
loss 1.2066823
STEP 65 ================================
prereg loss 0.11387843 reg_l1 108.6916 reg_l2 33.50331
loss 1.2007945
STEP 66 ================================
prereg loss 0.106181055 reg_l1 108.639694 reg_l2 33.48236
loss 1.192578
STEP 67 ================================
prereg loss 0.104648486 reg_l1 108.586815 reg_l2 33.461235
loss 1.1905166
STEP 68 ================================
prereg loss 0.108136505 reg_l1 108.53286 reg_l2 33.44009
loss 1.1934651
STEP 69 ================================
prereg loss 0.108486034 reg_l1 108.479126 reg_l2 33.419434
loss 1.1932772
STEP 70 ================================
prereg loss 0.106333986 reg_l1 108.42555 reg_l2 33.399128
loss 1.1905894
STEP 71 ================================
prereg loss 0.103027664 reg_l1 108.37265 reg_l2 33.37885
loss 1.1867542
STEP 72 ================================
prereg loss 0.103149205 reg_l1 108.31976 reg_l2 33.358517
loss 1.1863468
STEP 73 ================================
prereg loss 0.10221145 reg_l1 108.266525 reg_l2 33.338226
loss 1.1848767
STEP 74 ================================
prereg loss 0.10073012 reg_l1 108.21337 reg_l2 33.317787
loss 1.1828637
STEP 75 ================================
prereg loss 0.09809077 reg_l1 108.16073 reg_l2 33.297043
loss 1.179698
STEP 76 ================================
prereg loss 0.09805128 reg_l1 108.107956 reg_l2 33.27613
loss 1.1791308
STEP 77 ================================
prereg loss 0.098596975 reg_l1 108.05435 reg_l2 33.255234
loss 1.1791404
STEP 78 ================================
prereg loss 0.098839775 reg_l1 108.00102 reg_l2 33.234234
loss 1.1788499
STEP 79 ================================
prereg loss 0.09718959 reg_l1 107.94765 reg_l2 33.213062
loss 1.176666
STEP 80 ================================
prereg loss 0.097494654 reg_l1 107.89372 reg_l2 33.191814
loss 1.1764318
STEP 81 ================================
prereg loss 0.0974176 reg_l1 107.838585 reg_l2 33.17056
loss 1.1758034
STEP 82 ================================
prereg loss 0.09722404 reg_l1 107.783745 reg_l2 33.149128
loss 1.1750615
STEP 83 ================================
prereg loss 0.0961803 reg_l1 107.72837 reg_l2 33.127552
loss 1.1734641
STEP 84 ================================
prereg loss 0.09557316 reg_l1 107.672295 reg_l2 33.105984
loss 1.1722962
STEP 85 ================================
prereg loss 0.09598181 reg_l1 107.616 reg_l2 33.084583
loss 1.1721418
STEP 86 ================================
prereg loss 0.096384935 reg_l1 107.55955 reg_l2 33.063232
loss 1.1719804
STEP 87 ================================
prereg loss 0.095389955 reg_l1 107.50353 reg_l2 33.041943
loss 1.1704253
STEP 88 ================================
prereg loss 0.093921006 reg_l1 107.44737 reg_l2 33.020798
loss 1.1683948
STEP 89 ================================
prereg loss 0.093257815 reg_l1 107.39062 reg_l2 32.99971
loss 1.167164
STEP 90 ================================
prereg loss 0.092747025 reg_l1 107.33521 reg_l2 32.978474
loss 1.1660991
STEP 91 ================================
prereg loss 0.09253001 reg_l1 107.27913 reg_l2 32.957096
loss 1.1653212
STEP 92 ================================
prereg loss 0.091785416 reg_l1 107.22277 reg_l2 32.935696
loss 1.1640131
STEP 93 ================================
prereg loss 0.09168729 reg_l1 107.16786 reg_l2 32.914146
loss 1.163366
STEP 94 ================================
prereg loss 0.0915431 reg_l1 107.11497 reg_l2 32.89246
loss 1.1626928
STEP 95 ================================
prereg loss 0.091303505 reg_l1 107.06195 reg_l2 32.870857
loss 1.1619229
STEP 96 ================================
prereg loss 0.09091446 reg_l1 107.00803 reg_l2 32.849228
loss 1.1609948
STEP 97 ================================
prereg loss 0.09070338 reg_l1 106.95302 reg_l2 32.827553
loss 1.1602335
STEP 98 ================================
prereg loss 0.09104068 reg_l1 106.897224 reg_l2 32.805897
loss 1.160013
STEP 99 ================================
prereg loss 0.090810746 reg_l1 106.84086 reg_l2 32.78434
loss 1.1592194
STEP 100 ================================
prereg loss 0.090661354 reg_l1 106.78411 reg_l2 32.762806
loss 1.1585025
2022-06-26T14:09:25.469

julia> count_interval(sparse2, -0.001f0, 0.001f0)
15

julia> count_interval(sparse2, -0.01f0, 0.01f0)
37

julia> count_interval(sparse2, -0.02f0, 0.02f0)
116

julia> serialize("sparse2-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse2-after-100-steps-opt.ser", opt)

julia> close(io)
```

Let's increase L1-regularization further 3-fold, and try 0.02 cutoff.

```
$ diff loss.jl loss-original.jl
67c67
<     l += 0.03f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2

$ diff test.jl test-original.jl
36c36
<     l += 0.03f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse2-after-100-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

and the run is

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0362336), "input"=>Dict("char"=>-0.662331), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("compare-3-2"=>Dict("true"=>0.553771, "false"=>0.221764), "norm-2-1"=>Dict("norm"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0539431), "accum-1-2"=>Dict("dict"=>0.271262), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.750608), "dot-1-2"=>Dict("dot"=>0.0451928), "const_1"=>…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.199055), "accum-1-1"=>Dict("dict"=>0.125396), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0028469), "norm-2-1"=>Dict("norm"=>0.102111), "dot-2-2"…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.131324), "eos"=>Dict("char"=>0.0257352), "norm-2-1"=>D…
  "norm-4-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>6.82895f-5), "norm-2-1"=>Dict("norm"=>3.72792f-5), "dot-2-2"=>…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>-0.0190309), "accum-3-1"=>Dict("dict"=>-0.0357708), "acc…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.404076), "eos"=>Dict("char"=>-0.0163524)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.096304)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.156448), "eos"=>Dict("char"=>-0.155623)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.10934), "accum-3-1"=>Dict("dict"=>0.101579), "accum-2-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.194236), "compare-3-2"=>Dict("true"=>0.216067, "false"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.31006), "norm-2-1"=>Dict("norm"=>0.253497), "dot-2-2"=>…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0784427), "norm-2-1"=>Dict("norm"=>0.247689), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0905524), "input"=>Dict("char"=>0.926554), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.147603), "norm-2-1"=>Dict("norm"=>0.0492598), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.571818), "dot-1-2"=>Dict("dot"=>-0.192454), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.137253), "dot-1-2"=>Dict("dot"=>-0.140961), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.564208), "dot-1-2"=>Dict("dot"=>-0.200835), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0107677), "norm-3-2"=>Dict("norm"=>0.191682), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0790171), "norm-2-1"=>Dict("norm"=>-0.323285), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("dot-4-2"=>Dict("dot"=>-0.0823427), "accum-4-2"=>Dict("dict"=>0.0352205), "norm-…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.269527), "norm-2-1"=>Dict("norm"=>-0.0982553), "dot-2-…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
863

julia> sparse3 = sparsecopy(trainable["network_matrix"], 0.02f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 40 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0362336), "input"=>Dict("char"=>-0.662331), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0797549), "norm-2-1"=>Dict("norm"=>0.512316), "dot-2-2"=>Di…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0539431), "accum-1-2"=>Dict("dict"=>0.271262), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.750608), "dot-1-2"=>Dict("dot"=>0.0451928), "accum-1-1"…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.199055), "accum-1-1"=>Dict("dict"=>0.125396), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.102111), "dot-2-2"=>Dict("dot"=>0.0701463), "dot-2-1"…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.131324), "norm-2-1"=>Dict("norm"=>0.11759), "dot-2-2"=…
  "norm-4-2"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>0.0293011), "dot-3-1"=>Dict("dot"=>-0.0455326), "dot-3-2…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0357708), "accum-4-1"=>Dict("dict"=>-0.0278194), "inp…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.404076)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.096304)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.156448), "eos"=>Dict("char"=>-0.155623)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.10934), "accum-3-1"=>Dict("dict"=>0.101579), "accum-2-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.194236), "eos"=>Dict("char"=>0.09212), "dot-2-2"=>Dict(…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.31006), "norm-2-1"=>Dict("norm"=>0.253497), "dot-2-2"=>…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0784427), "norm-2-1"=>Dict("norm"=>0.247689), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0905524), "input"=>Dict("char"=>0.926554), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.147603), "norm-2-1"=>Dict("norm"=>0.0492598), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.571818), "dot-1-2"=>Dict("dot"=>-0.192454), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.137253), "dot-1-2"=>Dict("dot"=>-0.140961), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.564208), "dot-1-2"=>Dict("dot"=>-0.200835), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.2264), "dot-2-2"=>Dict("dot"=>-0.0912349), "const_1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0790171), "norm-2-1"=>Dict("norm"=>-0.323285), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.13778), "norm-3-1"=>Dict("norm"=>0.0685698), "norm-4-…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.269527), "norm-2-1"=>Dict("norm"=>-0.0982553), "dot-2-…
  ⋮             => ⋮

julia> count(sparse3)
747

julia> trainable["network_matrix"] = sparse3
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 40 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0362336), "input"=>Dict("char"=>-0.662331), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.0797549), "norm-2-1"=>Dict("norm"=>0.512316), "dot-2-2"=>Di…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0539431), "accum-1-2"=>Dict("dict"=>0.271262), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.750608), "dot-1-2"=>Dict("dot"=>0.0451928), "accum-1-1"…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.199055), "accum-1-1"=>Dict("dict"=>0.125396), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.102111), "dot-2-2"=>Dict("dot"=>0.0701463), "dot-2-1"…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.131324), "norm-2-1"=>Dict("norm"=>0.11759), "dot-2-2"=…
  "norm-4-2"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>0.0293011), "dot-3-1"=>Dict("dot"=>-0.0455326), "dot-3-2…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0357708), "accum-4-1"=>Dict("dict"=>-0.0278194), "inp…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.404076)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.096304)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.156448), "eos"=>Dict("char"=>-0.155623)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.10934), "accum-3-1"=>Dict("dict"=>0.101579), "accum-2-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.194236), "eos"=>Dict("char"=>0.09212), "dot-2-2"=>Dict(…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.31006), "norm-2-1"=>Dict("norm"=>0.253497), "dot-2-2"=>…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0784427), "norm-2-1"=>Dict("norm"=>0.247689), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0905524), "input"=>Dict("char"=>0.926554), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.147603), "norm-2-1"=>Dict("norm"=>0.0492598), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.571818), "dot-1-2"=>Dict("dot"=>-0.192454), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.137253), "dot-1-2"=>Dict("dot"=>-0.140961), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.564208), "dot-1-2"=>Dict("dot"=>-0.200835), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.2264), "dot-2-2"=>Dict("dot"=>-0.0912349), "const_1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0790171), "norm-2-1"=>Dict("norm"=>-0.323285), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.13778), "norm-3-1"=>Dict("norm"=>0.0685698), "norm-4-…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.269527), "norm-2-1"=>Dict("norm"=>-0.0982553), "dot-2-…
  ⋮             => ⋮

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T14:28:37.041
STEP 1 ================================
prereg loss 71.85832 reg_l1 105.36236 reg_l2 32.720856
loss 75.019196
STEP 2 ================================
prereg loss 56.865353 reg_l1 105.40537 reg_l2 32.760548
loss 60.027515
STEP 3 ================================
prereg loss 44.467144 reg_l1 105.43712 reg_l2 32.792103
loss 47.630257
STEP 4 ================================
prereg loss 34.061016 reg_l1 105.463524 reg_l2 32.819965
loss 37.224922
STEP 5 ================================
prereg loss 25.867052 reg_l1 105.48569 reg_l2 32.844124
loss 29.031622
STEP 6 ================================
prereg loss 19.922382 reg_l1 105.50062 reg_l2 32.86322
loss 23.0874
STEP 7 ================================
prereg loss 15.89685 reg_l1 105.50888 reg_l2 32.878006
loss 19.062117
STEP 8 ================================
prereg loss 13.466919 reg_l1 105.512726 reg_l2 32.889664
loss 16.632301
STEP 9 ================================
prereg loss 12.1806555 reg_l1 105.51339 reg_l2 32.898758
loss 15.346057
STEP 10 ================================
prereg loss 11.34623 reg_l1 105.51171 reg_l2 32.905445
loss 14.51158
STEP 11 ================================
prereg loss 10.464413 reg_l1 105.50819 reg_l2 32.910034
loss 13.629658
STEP 12 ================================
prereg loss 9.3142605 reg_l1 105.503334 reg_l2 32.912933
loss 12.479361
STEP 13 ================================
prereg loss 8.005778 reg_l1 105.49561 reg_l2 32.91343
loss 11.170647
STEP 14 ================================
prereg loss 6.7096467 reg_l1 105.48404 reg_l2 32.911037
loss 9.874168
STEP 15 ================================
prereg loss 5.528567 reg_l1 105.468124 reg_l2 32.905613
loss 8.692611
STEP 16 ================================
prereg loss 4.5635653 reg_l1 105.446175 reg_l2 32.896996
loss 7.7269506
STEP 17 ================================
prereg loss 3.9109507 reg_l1 105.417336 reg_l2 32.885567
loss 7.0734706
STEP 18 ================================
prereg loss 3.5437598 reg_l1 105.38099 reg_l2 32.87161
loss 6.7051897
STEP 19 ================================
prereg loss 3.4475167 reg_l1 105.33725 reg_l2 32.855465
loss 6.607634
STEP 20 ================================
prereg loss 3.5408602 reg_l1 105.2859 reg_l2 32.837433
loss 6.699437
STEP 21 ================================
prereg loss 3.6759262 reg_l1 105.22677 reg_l2 32.817657
loss 6.8327293
STEP 22 ================================
prereg loss 3.7170503 reg_l1 105.16121 reg_l2 32.79612
loss 6.8718863
STEP 23 ================================
prereg loss 3.620847 reg_l1 105.08974 reg_l2 32.77309
loss 6.773539
STEP 24 ================================
prereg loss 3.4396296 reg_l1 105.010796 reg_l2 32.74862
loss 6.5899534
STEP 25 ================================
prereg loss 3.2382493 reg_l1 104.92453 reg_l2 32.722603
loss 6.3859854
STEP 26 ================================
prereg loss 3.0790696 reg_l1 104.831345 reg_l2 32.69476
loss 6.22401
STEP 27 ================================
prereg loss 2.9901648 reg_l1 104.73101 reg_l2 32.664772
loss 6.1320953
STEP 28 ================================
prereg loss 2.9458287 reg_l1 104.62341 reg_l2 32.63224
loss 6.084531
STEP 29 ================================
prereg loss 2.9190176 reg_l1 104.508095 reg_l2 32.597008
loss 6.0542603
STEP 30 ================================
prereg loss 2.87411 reg_l1 104.38592 reg_l2 32.558987
loss 6.0056877
STEP 31 ================================
prereg loss 2.781928 reg_l1 104.25739 reg_l2 32.518562
loss 5.90965
STEP 32 ================================
prereg loss 2.6418886 reg_l1 104.12532 reg_l2 32.47608
loss 5.765648
STEP 33 ================================
prereg loss 2.4587922 reg_l1 103.98992 reg_l2 32.43214
loss 5.57849
STEP 34 ================================
prereg loss 2.2471104 reg_l1 103.85211 reg_l2 32.387344
loss 5.3626738
STEP 35 ================================
prereg loss 2.0389798 reg_l1 103.71447 reg_l2 32.34222
loss 5.1504135
STEP 36 ================================
prereg loss 1.866467 reg_l1 103.57871 reg_l2 32.297195
loss 4.9738283
STEP 37 ================================
prereg loss 1.7481526 reg_l1 103.44662 reg_l2 32.25272
loss 4.851551
STEP 38 ================================
prereg loss 1.6877606 reg_l1 103.31714 reg_l2 32.209316
loss 4.7872744
STEP 39 ================================
prereg loss 1.6761733 reg_l1 103.19089 reg_l2 32.16726
loss 4.7718997
STEP 40 ================================
prereg loss 1.6960067 reg_l1 103.06947 reg_l2 32.126923
loss 4.7880907
STEP 41 ================================
prereg loss 1.7271998 reg_l1 102.95377 reg_l2 32.088562
loss 4.815813
STEP 42 ================================
prereg loss 1.7527298 reg_l1 102.84306 reg_l2 32.0524
loss 4.8380218
STEP 43 ================================
prereg loss 1.761347 reg_l1 102.73765 reg_l2 32.018444
loss 4.8434763
STEP 44 ================================
prereg loss 1.7501138 reg_l1 102.63852 reg_l2 31.986685
loss 4.8292694
STEP 45 ================================
prereg loss 1.7204927 reg_l1 102.54499 reg_l2 31.956974
loss 4.7968426
STEP 46 ================================
prereg loss 1.6788324 reg_l1 102.45725 reg_l2 31.929195
loss 4.75255
STEP 47 ================================
prereg loss 1.6334524 reg_l1 102.374504 reg_l2 31.903076
loss 4.704687
STEP 48 ================================
prereg loss 1.5905007 reg_l1 102.29636 reg_l2 31.878407
loss 4.6593914
STEP 49 ================================
prereg loss 1.5514406 reg_l1 102.222725 reg_l2 31.854954
loss 4.618122
STEP 50 ================================
prereg loss 1.51553 reg_l1 102.15333 reg_l2 31.832575
loss 4.5801296
STEP 51 ================================
prereg loss 1.4798152 reg_l1 102.08748 reg_l2 31.811087
loss 4.5424395
STEP 52 ================================
prereg loss 1.4430404 reg_l1 102.02342 reg_l2 31.79018
loss 4.503743
STEP 53 ================================
prereg loss 1.4051301 reg_l1 101.96107 reg_l2 31.769684
loss 4.463962
STEP 54 ================================
prereg loss 1.3665853 reg_l1 101.89937 reg_l2 31.749386
loss 4.4235663
STEP 55 ================================
prereg loss 1.3293809 reg_l1 101.83969 reg_l2 31.72919
loss 4.3845716
STEP 56 ================================
prereg loss 1.2958308 reg_l1 101.781204 reg_l2 31.709084
loss 4.349267
STEP 57 ================================
prereg loss 1.2677492 reg_l1 101.72347 reg_l2 31.688946
loss 4.3194532
STEP 58 ================================
prereg loss 1.2457832 reg_l1 101.66611 reg_l2 31.668858
loss 4.2957664
STEP 59 ================================
prereg loss 1.2295187 reg_l1 101.60638 reg_l2 31.648695
loss 4.27771
STEP 60 ================================
prereg loss 1.2208968 reg_l1 101.54545 reg_l2 31.628351
loss 4.26726
STEP 61 ================================
prereg loss 1.2177417 reg_l1 101.484856 reg_l2 31.607727
loss 4.262287
STEP 62 ================================
prereg loss 1.2117605 reg_l1 101.42316 reg_l2 31.5867
loss 4.254455
STEP 63 ================================
prereg loss 1.2005143 reg_l1 101.35787 reg_l2 31.565292
loss 4.2412505
STEP 64 ================================
prereg loss 1.1866616 reg_l1 101.2892 reg_l2 31.54369
loss 4.2253375
STEP 65 ================================
prereg loss 1.1709952 reg_l1 101.21896 reg_l2 31.52204
loss 4.2075644
STEP 66 ================================
prereg loss 1.1552168 reg_l1 101.14767 reg_l2 31.500235
loss 4.1896467
STEP 67 ================================
prereg loss 1.1447086 reg_l1 101.07661 reg_l2 31.478062
loss 4.1770067
STEP 68 ================================
prereg loss 1.1386911 reg_l1 101.00448 reg_l2 31.455275
loss 4.168825
STEP 69 ================================
prereg loss 1.1323783 reg_l1 100.93111 reg_l2 31.431768
loss 4.1603117
STEP 70 ================================
prereg loss 1.1246803 reg_l1 100.856834 reg_l2 31.407652
loss 4.150385
STEP 71 ================================
prereg loss 1.1144371 reg_l1 100.78176 reg_l2 31.383238
loss 4.13789
STEP 72 ================================
prereg loss 1.1007303 reg_l1 100.70597 reg_l2 31.3587
loss 4.121909
STEP 73 ================================
prereg loss 1.086109 reg_l1 100.62868 reg_l2 31.334198
loss 4.1049695
STEP 74 ================================
prereg loss 1.0753732 reg_l1 100.549706 reg_l2 31.309689
loss 4.0918646
STEP 75 ================================
prereg loss 1.0681511 reg_l1 100.47096 reg_l2 31.285194
loss 4.08228
STEP 76 ================================
prereg loss 1.063128 reg_l1 100.392075 reg_l2 31.260773
loss 4.07489
STEP 77 ================================
prereg loss 1.059077 reg_l1 100.31258 reg_l2 31.236647
loss 4.0684543
STEP 78 ================================
prereg loss 1.0536078 reg_l1 100.23444 reg_l2 31.212957
loss 4.0606413
STEP 79 ================================
prereg loss 1.0454757 reg_l1 100.156494 reg_l2 31.189867
loss 4.0501704
STEP 80 ================================
prereg loss 1.0356284 reg_l1 100.080315 reg_l2 31.1673
loss 4.038038
STEP 81 ================================
prereg loss 1.0257329 reg_l1 100.00549 reg_l2 31.145058
loss 4.0258975
STEP 82 ================================
prereg loss 1.0168743 reg_l1 99.93302 reg_l2 31.123064
loss 4.014865
STEP 83 ================================
prereg loss 1.0084367 reg_l1 99.86044 reg_l2 31.101315
loss 4.0042496
STEP 84 ================================
prereg loss 1.0007933 reg_l1 99.78771 reg_l2 31.079708
loss 3.9944248
STEP 85 ================================
prereg loss 0.9934968 reg_l1 99.71592 reg_l2 31.058355
loss 3.9849744
STEP 86 ================================
prereg loss 0.98586273 reg_l1 99.6453 reg_l2 31.037264
loss 3.9752216
STEP 87 ================================
prereg loss 0.97822607 reg_l1 99.57657 reg_l2 31.016314
loss 3.9655232
STEP 88 ================================
prereg loss 0.97137326 reg_l1 99.50636 reg_l2 30.995426
loss 3.9565642
STEP 89 ================================
prereg loss 0.96552044 reg_l1 99.43577 reg_l2 30.974514
loss 3.9485934
STEP 90 ================================
prereg loss 0.96088624 reg_l1 99.36592 reg_l2 30.953571
loss 3.9418638
STEP 91 ================================
prereg loss 0.9564688 reg_l1 99.29831 reg_l2 30.932775
loss 3.9354181
STEP 92 ================================
prereg loss 0.9511829 reg_l1 99.23256 reg_l2 30.912193
loss 3.9281595
STEP 93 ================================
prereg loss 0.94511503 reg_l1 99.17246 reg_l2 30.891815
loss 3.9202888
STEP 94 ================================
prereg loss 0.93935364 reg_l1 99.109276 reg_l2 30.871454
loss 3.912632
STEP 95 ================================
prereg loss 0.9340098 reg_l1 99.044525 reg_l2 30.851034
loss 3.9053454
STEP 96 ================================
prereg loss 0.92898554 reg_l1 98.97912 reg_l2 30.830433
loss 3.898359
STEP 97 ================================
prereg loss 0.9244989 reg_l1 98.91215 reg_l2 30.80976
loss 3.8918633
STEP 98 ================================
prereg loss 0.9199395 reg_l1 98.84304 reg_l2 30.789171
loss 3.8852308
STEP 99 ================================
prereg loss 0.9148915 reg_l1 98.77321 reg_l2 30.7686
loss 3.8780878
STEP 100 ================================
prereg loss 0.909931 reg_l1 98.703995 reg_l2 30.747993
loss 3.8710508
2022-06-26T15:10:59.887

julia> serialize("sparse3-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse3-after-100-steps-opt.ser", opt)

julia> count_interval(sparse3, -0.001f0, 0.001f0)
25

julia> count_interval(sparse3, -0.01f0, 0.01f0)
56

julia> count_interval(sparse3, -0.02f0, 0.02f0)
86

julia> count_interval(sparse3, -0.03f0, 0.03f0)
124

julia> count_interval(sparse3, -0.025f0, 0.025f0)
102

julia> close(io)
```

Let's double L1 regularization again, and use 0.025 for the next cutoff.

```
$ diff loss.jl loss-original.jl
67c67
<     l += 0.06f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2

$ diff test.jl test-original.jl
36c36
<     l += 0.06f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse3-after-100-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

Here is what we've done:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 40 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0358022), "input"=>Dict("char"=>-0.630359), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("compare-3-2"=>Dict("true"=>0.458113, "false"=>0.217364), "norm-2-1"=>Dict("norm"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500253), "accum-1-2"=>Dict("dict"=>0.274687), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.691826), "dot-1-2"=>Dict("dot"=>-0.00846714), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208141), "accum-1-1"=>Dict("dict"=>0.13427), "accum-1-…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0968717), "dot-2-2"=>Dict("dot"=>0.0450305), "dot-2-1…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0709888), "eos"=>Dict("char"=>0.0681364), "norm-2-1"=>…
  "norm-4-2"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>0.00891355), "dot-3-1"=>Dict("dot"=>-0.0623201), "dot-3-…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0517302), "accum-4-1"=>Dict("dict"=>-0.0361064), "inp…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.461722)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.104751)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.161252), "eos"=>Dict("char"=>-0.161107)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0956983), "accum-3-1"=>Dict("dict"=>0.0769411), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.260676), "compare-3-2"=>Dict("true"=>0.116217, "false"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.339063), "norm-2-1"=>Dict("norm"=>0.238238), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0844511), "norm-3-2"=>Dict("norm"=>0.17532), "norm-2-1…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0814641), "input"=>Dict("char"=>0.926292), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152864), "norm-2-1"=>Dict("norm"=>0.0605347), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.529796), "dot-1-2"=>Dict("dot"=>-0.190539), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.132671), "dot-1-2"=>Dict("dot"=>-0.129151), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.512245), "dot-1-2"=>Dict("dot"=>-0.149787), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-2-1"=>Dict("false"=>0.062739), "compare-3-2"=>Dict("true"=>0.00114825, …
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0746426), "norm-2-1"=>Dict("norm"=>-0.307124), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("dot-4-2"=>Dict("dot"=>-0.000765605), "accum-4-2"=>Dict("dict"=>0.000102585), "n…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.305095), "norm-2-1"=>Dict("norm"=>-0.0931677), "dot-2-…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
747

julia> sparse4 = sparsecopy(trainable["network_matrix"], 0.025f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 40 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0358022), "input"=>Dict("char"=>-0.630359), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.510768), "dot-2-2"=>Dict("dot"=>-0.15108), "norm-3-1"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500253), "accum-1-2"=>Dict("dict"=>0.274687), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.691826), "accum-1-1"=>Dict("dict"=>0.189648), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208141), "accum-1-1"=>Dict("dict"=>0.13427), "accum-1-…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0968717), "dot-2-2"=>Dict("dot"=>0.0450305), "dot-2-1…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0709888), "norm-2-1"=>Dict("norm"=>0.116373), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("dot-3-1"=>Dict("dot"=>-0.0623201), "dot-3-2"=>Dict("dot"=>0.0547277), "accum-2-2"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0517302), "accum-4-1"=>Dict("dict"=>-0.0361064)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.461722)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.104751)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.161252), "eos"=>Dict("char"=>-0.161107)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0956983), "accum-3-1"=>Dict("dict"=>0.0769411), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.260676), "eos"=>Dict("char"=>0.125085), "dot-2-2"=>Dict…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.339063), "norm-2-1"=>Dict("norm"=>0.238238), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0844511), "norm-2-1"=>Dict("norm"=>0.265766), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0814641), "input"=>Dict("char"=>0.926292), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152864), "norm-2-1"=>Dict("norm"=>0.0605347), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.529796), "dot-1-2"=>Dict("dot"=>-0.190539), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.132671), "dot-1-2"=>Dict("dot"=>-0.129151), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.512245), "dot-1-2"=>Dict("dot"=>-0.149787), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.225657), "dot-2-2"=>Dict("dot"=>-0.0833997), "const_1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0746426), "norm-2-1"=>Dict("norm"=>-0.307124), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0377796)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.305095), "norm-2-1"=>Dict("norm"=>-0.0931677), "dot-2-…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse4
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 40 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0358022), "input"=>Dict("char"=>-0.630359), "eos"=…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.510768), "dot-2-2"=>Dict("dot"=>-0.15108), "norm-3-1"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500253), "accum-1-2"=>Dict("dict"=>0.274687), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.691826), "accum-1-1"=>Dict("dict"=>0.189648), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208141), "accum-1-1"=>Dict("dict"=>0.13427), "accum-1-…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0968717), "dot-2-2"=>Dict("dot"=>0.0450305), "dot-2-1…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0709888), "norm-2-1"=>Dict("norm"=>0.116373), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("dot-3-1"=>Dict("dot"=>-0.0623201), "dot-3-2"=>Dict("dot"=>0.0547277), "accum-2-2"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0517302), "accum-4-1"=>Dict("dict"=>-0.0361064)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.461722)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.104751)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.161252), "eos"=>Dict("char"=>-0.161107)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0956983), "accum-3-1"=>Dict("dict"=>0.0769411), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.260676), "eos"=>Dict("char"=>0.125085), "dot-2-2"=>Dict…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.339063), "norm-2-1"=>Dict("norm"=>0.238238), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0844511), "norm-2-1"=>Dict("norm"=>0.265766), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0814641), "input"=>Dict("char"=>0.926292), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152864), "norm-2-1"=>Dict("norm"=>0.0605347), "dot-2-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.529796), "dot-1-2"=>Dict("dot"=>-0.190539), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.132671), "dot-1-2"=>Dict("dot"=>-0.129151), "compare-1-…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.512245), "dot-1-2"=>Dict("dot"=>-0.149787), "const_1"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.225657), "dot-2-2"=>Dict("dot"=>-0.0833997), "const_1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0746426), "norm-2-1"=>Dict("norm"=>-0.307124), "dot-2-2…
  "accum-5-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0377796)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.305095), "norm-2-1"=>Dict("norm"=>-0.0931677), "dot-2-…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
645

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T15:24:46.370
STEP 1 ================================
prereg loss 14.363609 reg_l1 97.675156 reg_l2 30.71138
loss 20.22412
STEP 2 ================================
prereg loss 3.9033158 reg_l1 97.718155 reg_l2 30.73108
loss 9.766405
STEP 3 ================================
prereg loss 5.7964144 reg_l1 97.723434 reg_l2 30.739716
loss 11.659821
STEP 4 ================================
prereg loss 6.1146073 reg_l1 97.692154 reg_l2 30.731329
loss 11.976136
STEP 5 ================================
prereg loss 3.4237883 reg_l1 97.63954 reg_l2 30.712143
loss 9.282161
STEP 6 ================================
prereg loss 1.5416923 reg_l1 97.57595 reg_l2 30.687153
loss 7.3962493
STEP 7 ================================
prereg loss 1.8586534 reg_l1 97.50713 reg_l2 30.658686
loss 7.7090816
STEP 8 ================================
prereg loss 3.1461837 reg_l1 97.43967 reg_l2 30.629662
loss 8.992563
STEP 9 ================================
prereg loss 3.5597804 reg_l1 97.378334 reg_l2 30.602905
loss 9.40248
STEP 10 ================================
prereg loss 2.75379 reg_l1 97.32376 reg_l2 30.579353
loss 8.593216
STEP 11 ================================
prereg loss 1.6459875 reg_l1 97.27261 reg_l2 30.558004
loss 7.482344
STEP 12 ================================
prereg loss 1.2065924 reg_l1 97.22053 reg_l2 30.537075
loss 7.039824
STEP 13 ================================
prereg loss 1.66118 reg_l1 97.162125 reg_l2 30.514362
loss 7.490907
STEP 14 ================================
prereg loss 2.360472 reg_l1 97.09391 reg_l2 30.488148
loss 8.186107
STEP 15 ================================
prereg loss 2.5213299 reg_l1 97.01503 reg_l2 30.457695
loss 8.342232
STEP 16 ================================
prereg loss 2.0132542 reg_l1 96.926994 reg_l2 30.423204
loss 7.8288736
STEP 17 ================================
prereg loss 1.3587666 reg_l1 96.83133 reg_l2 30.385664
loss 7.1686463
STEP 18 ================================
prereg loss 1.0878733 reg_l1 96.73095 reg_l2 30.346382
loss 6.8917303
STEP 19 ================================
prereg loss 1.3030659 reg_l1 96.62942 reg_l2 30.306917
loss 7.1008306
STEP 20 ================================
prereg loss 1.6570292 reg_l1 96.530106 reg_l2 30.268757
loss 7.4488354
STEP 21 ================================
prereg loss 1.7236975 reg_l1 96.4351 reg_l2 30.232742
loss 7.5098033
STEP 22 ================================
prereg loss 1.4424567 reg_l1 96.34476 reg_l2 30.198942
loss 7.223142
STEP 23 ================================
prereg loss 1.0785693 reg_l1 96.25746 reg_l2 30.166666
loss 6.854017
STEP 24 ================================
prereg loss 0.9031873 reg_l1 96.171005 reg_l2 30.134808
loss 6.6734476
STEP 25 ================================
prereg loss 0.9753238 reg_l1 96.08263 reg_l2 30.10222
loss 6.740281
STEP 26 ================================
prereg loss 1.1331383 reg_l1 95.99034 reg_l2 30.067915
loss 6.8925586
STEP 27 ================================
prereg loss 1.1725978 reg_l1 95.89314 reg_l2 30.031487
loss 6.926186
STEP 28 ================================
prereg loss 1.0467814 reg_l1 95.79152 reg_l2 29.993143
loss 6.7942724
STEP 29 ================================
prereg loss 0.9047823 reg_l1 95.687515 reg_l2 29.95366
loss 6.6460333
STEP 30 ================================
prereg loss 0.87524134 reg_l1 95.58356 reg_l2 29.913998
loss 6.610255
STEP 31 ================================
prereg loss 0.9525179 reg_l1 95.48113 reg_l2 29.87503
loss 6.681386
STEP 32 ================================
prereg loss 1.0337435 reg_l1 95.38159 reg_l2 29.837404
loss 6.7566385
STEP 33 ================================
prereg loss 1.0232033 reg_l1 95.28493 reg_l2 29.801506
loss 6.7402987
STEP 34 ================================
prereg loss 0.9158681 reg_l1 95.190796 reg_l2 29.767145
loss 6.627316
STEP 35 ================================
prereg loss 0.7897362 reg_l1 95.09909 reg_l2 29.733833
loss 6.495682
STEP 36 ================================
prereg loss 0.71968955 reg_l1 95.010155 reg_l2 29.70091
loss 6.4202986
STEP 37 ================================
prereg loss 0.71901286 reg_l1 94.92012 reg_l2 29.66764
loss 6.41422
STEP 38 ================================
prereg loss 0.7371706 reg_l1 94.82702 reg_l2 29.633387
loss 6.4267917
STEP 39 ================================
prereg loss 0.71884626 reg_l1 94.7307 reg_l2 29.597864
loss 6.402688
STEP 40 ================================
prereg loss 0.6650321 reg_l1 94.632164 reg_l2 29.561167
loss 6.342962
STEP 41 ================================
prereg loss 0.6230233 reg_l1 94.53092 reg_l2 29.523684
loss 6.294878
STEP 42 ================================
prereg loss 0.62978166 reg_l1 94.42827 reg_l2 29.486126
loss 6.295478
STEP 43 ================================
prereg loss 0.6664431 reg_l1 94.326035 reg_l2 29.449093
loss 6.326005
STEP 44 ================================
prereg loss 0.6812411 reg_l1 94.22609 reg_l2 29.41302
loss 6.3348064
STEP 45 ================================
prereg loss 0.65395635 reg_l1 94.13042 reg_l2 29.378016
loss 6.301781
STEP 46 ================================
prereg loss 0.6137002 reg_l1 94.036514 reg_l2 29.343714
loss 6.2558913
STEP 47 ================================
prereg loss 0.5998285 reg_l1 93.9422 reg_l2 29.309511
loss 6.2363605
STEP 48 ================================
prereg loss 0.61234915 reg_l1 93.84531 reg_l2 29.274734
loss 6.2430673
STEP 49 ================================
prereg loss 0.62097204 reg_l1 93.74573 reg_l2 29.238914
loss 6.2457156
STEP 50 ================================
prereg loss 0.606529 reg_l1 93.64271 reg_l2 29.201967
loss 6.225091
STEP 51 ================================
prereg loss 0.5837974 reg_l1 93.53776 reg_l2 29.163982
loss 6.1960626
STEP 52 ================================
prereg loss 0.578452 reg_l1 93.43089 reg_l2 29.125542
loss 6.1843057
STEP 53 ================================
prereg loss 0.59182197 reg_l1 93.32393 reg_l2 29.087156
loss 6.191258
STEP 54 ================================
prereg loss 0.6016485 reg_l1 93.2194 reg_l2 29.049284
loss 6.1948123
STEP 55 ================================
prereg loss 0.5928844 reg_l1 93.11658 reg_l2 29.012197
loss 6.179879
STEP 56 ================================
prereg loss 0.57335806 reg_l1 93.0192 reg_l2 28.975737
loss 6.15451
STEP 57 ================================
prereg loss 0.5616545 reg_l1 92.923454 reg_l2 28.939625
loss 6.1370616
STEP 58 ================================
prereg loss 0.5653041 reg_l1 92.82642 reg_l2 28.903465
loss 6.1348896
STEP 59 ================================
prereg loss 0.5706781 reg_l1 92.728035 reg_l2 28.866854
loss 6.1343603
STEP 60 ================================
prereg loss 0.56601083 reg_l1 92.62698 reg_l2 28.829718
loss 6.12363
STEP 61 ================================
prereg loss 0.5570542 reg_l1 92.52513 reg_l2 28.792183
loss 6.108562
STEP 62 ================================
prereg loss 0.5558217 reg_l1 92.42331 reg_l2 28.754612
loss 6.10122
STEP 63 ================================
prereg loss 0.5616567 reg_l1 92.32154 reg_l2 28.717396
loss 6.1009493
STEP 64 ================================
prereg loss 0.5626154 reg_l1 92.22249 reg_l2 28.680779
loss 6.0959644
STEP 65 ================================
prereg loss 0.554468 reg_l1 92.12485 reg_l2 28.644806
loss 6.081959
STEP 66 ================================
prereg loss 0.5461892 reg_l1 92.03018 reg_l2 28.609179
loss 6.0680003
STEP 67 ================================
prereg loss 0.5434245 reg_l1 91.937546 reg_l2 28.573599
loss 6.059677
STEP 68 ================================
prereg loss 0.5442778 reg_l1 91.843445 reg_l2 28.537693
loss 6.0548844
STEP 69 ================================
prereg loss 0.5434447 reg_l1 91.74803 reg_l2 28.501307
loss 6.0483265
STEP 70 ================================
prereg loss 0.54128355 reg_l1 91.65357 reg_l2 28.464514
loss 6.040498
STEP 71 ================================
prereg loss 0.54144025 reg_l1 91.55983 reg_l2 28.427505
loss 6.0350304
STEP 72 ================================
prereg loss 0.54407984 reg_l1 91.46501 reg_l2 28.39059
loss 6.0319805
STEP 73 ================================
prereg loss 0.54536456 reg_l1 91.368996 reg_l2 28.353928
loss 6.027504
STEP 74 ================================
prereg loss 0.5430188 reg_l1 91.2733 reg_l2 28.317644
loss 6.019417
STEP 75 ================================
prereg loss 0.5392213 reg_l1 91.17656 reg_l2 28.28158
loss 6.0098147
STEP 76 ================================
prereg loss 0.53701675 reg_l1 91.07872 reg_l2 28.24562
loss 6.00174
STEP 77 ================================
prereg loss 0.53662467 reg_l1 90.97937 reg_l2 28.209505
loss 5.995387
STEP 78 ================================
prereg loss 0.53575647 reg_l1 90.87969 reg_l2 28.173084
loss 5.988538
STEP 79 ================================
prereg loss 0.53438133 reg_l1 90.77961 reg_l2 28.136448
loss 5.981158
STEP 80 ================================
prereg loss 0.534276 reg_l1 90.67944 reg_l2 28.099733
loss 5.9750423
STEP 81 ================================
prereg loss 0.53519803 reg_l1 90.5795 reg_l2 28.063185
loss 5.969968
STEP 82 ================================
prereg loss 0.53486514 reg_l1 90.48188 reg_l2 28.026997
loss 5.9637775
STEP 83 ================================
prereg loss 0.5326113 reg_l1 90.383446 reg_l2 27.991077
loss 5.955618
STEP 84 ================================
prereg loss 0.5302531 reg_l1 90.28648 reg_l2 27.95547
loss 5.947442
STEP 85 ================================
prereg loss 0.5289053 reg_l1 90.191574 reg_l2 27.919922
loss 5.9403996
STEP 86 ================================
prereg loss 0.5279708 reg_l1 90.09731 reg_l2 27.884344
loss 5.9338093
STEP 87 ================================
prereg loss 0.52683777 reg_l1 90.00319 reg_l2 27.848679
loss 5.927029
STEP 88 ================================
prereg loss 0.5258577 reg_l1 89.90805 reg_l2 27.81298
loss 5.9203405
STEP 89 ================================
prereg loss 0.52558684 reg_l1 89.81221 reg_l2 27.777367
loss 5.914319
STEP 90 ================================
prereg loss 0.5255845 reg_l1 89.71707 reg_l2 27.741987
loss 5.908609
STEP 91 ================================
prereg loss 0.5248675 reg_l1 89.62181 reg_l2 27.70684
loss 5.902176
STEP 92 ================================
prereg loss 0.5234504 reg_l1 89.52751 reg_l2 27.671936
loss 5.895101
STEP 93 ================================
prereg loss 0.52218395 reg_l1 89.43283 reg_l2 27.637192
loss 5.8881536
STEP 94 ================================
prereg loss 0.521512 reg_l1 89.33863 reg_l2 27.602472
loss 5.8818297
STEP 95 ================================
prereg loss 0.52113724 reg_l1 89.24442 reg_l2 27.56773
loss 5.8758025
STEP 96 ================================
prereg loss 0.5208774 reg_l1 89.15166 reg_l2 27.532898
loss 5.8699765
STEP 97 ================================
prereg loss 0.52108943 reg_l1 89.0598 reg_l2 27.498075
loss 5.8646774
STEP 98 ================================
prereg loss 0.5214938 reg_l1 88.96772 reg_l2 27.463366
loss 5.859557
STEP 99 ================================
prereg loss 0.5216034 reg_l1 88.87543 reg_l2 27.428883
loss 5.8541293
STEP 100 ================================
prereg loss 0.5210778 reg_l1 88.78343 reg_l2 27.394543
loss 5.8480835
2022-06-26T15:57:35.184

julia> serialize("sparse4-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse4-after-100-steps-opt.ser", opt)

julia> count_interval(sparse4, -0.001f0, 0.001f0)
25

julia> count_interval(sparse4, -0.01f0, 0.01f0)
37

julia> count_interval(sparse4, -0.02f0, 0.02f0)
49

julia> count_interval(sparse4, -0.03f0, 0.03f0)
81

julia> count_interval(sparse4, -0.04f0, 0.04f0)
121

julia> count_interval(sparse4, -0.05f0, 0.05f0)
168

julia> count_neg_interval(sparse4, -1.00f0, 1.00f0)
2

julia> count_neg_interval(sparse4, -0.8f0, 0.8f0)
9

julia> count_neg_interval(sparse4, -0.5f0, 0.5f0)
21

julia> count_neg_interval(sparse4, -0.3f0, 0.3f0)
61

julia> # let's try 0.03 cutoff and same regularization

julia> sparse5 = sparsecopy(trainable["network_matrix"], 0.03f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 39 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.621733), "eos"=>Dict("char"=>-0.0585393)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.511286), "dot-2-2"=>Dict("dot"=>-0.135401), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0466449), "accum-1-2"=>Dict("dict"=>0.277625), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.646154), "accum-1-1"=>Dict("dict"=>0.185104), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208114), "accum-1-1"=>Dict("dict"=>0.133779), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0547426), "dot-2-1"=>Dict("dot"=>-0.130495)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0597682), "norm-2-1"=>Dict("norm"=>0.119194), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0743818), "accum-1-2"=>Dict("dict"=>0.217296), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0547695), "accum-4-1"=>Dict("dict"=>-0.0311545)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.459703)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.10328)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16022), "eos"=>Dict("char"=>-0.160266)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0863766), "accum-3-1"=>Dict("dict"=>0.0709378), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28218), "dot-2-2"=>Dict("dot"=>-0.148223), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.323512), "norm-2-1"=>Dict("norm"=>0.236528), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0506336), "norm-2-1"=>Dict("norm"=>0.261369), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0421611), "input"=>Dict("char"=>0.878656), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.162344), "norm-2-1"=>Dict("norm"=>0.0640643), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.505434), "dot-1-2"=>Dict("dot"=>-0.180713), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0740083), "dot-1-2"=>Dict("dot"=>-0.0738982)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.465071), "dot-1-2"=>Dict("dot"=>-0.0691788), "const_1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-2-1"=>Dict("false"=>0.0549196), "compare-3-2"=>Dict("false"=>0.0715782)…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0682483), "norm-2-1"=>Dict("norm"=>-0.309208), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.308841), "norm-2-1"=>Dict("norm"=>-0.0810245), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0995431), "const_1"=>Dict("const_1"=>0.0415581), "comp…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse5
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 39 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.621733), "eos"=>Dict("char"=>-0.0585393)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.511286), "dot-2-2"=>Dict("dot"=>-0.135401), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0466449), "accum-1-2"=>Dict("dict"=>0.277625), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.646154), "accum-1-1"=>Dict("dict"=>0.185104), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208114), "accum-1-1"=>Dict("dict"=>0.133779), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0547426), "dot-2-1"=>Dict("dot"=>-0.130495)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0597682), "norm-2-1"=>Dict("norm"=>0.119194), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0743818), "accum-1-2"=>Dict("dict"=>0.217296), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0547695), "accum-4-1"=>Dict("dict"=>-0.0311545)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.459703)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.10328)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16022), "eos"=>Dict("char"=>-0.160266)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0863766), "accum-3-1"=>Dict("dict"=>0.0709378), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28218), "dot-2-2"=>Dict("dot"=>-0.148223), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.323512), "norm-2-1"=>Dict("norm"=>0.236528), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0506336), "norm-2-1"=>Dict("norm"=>0.261369), "dot-2-2…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0421611), "input"=>Dict("char"=>0.878656), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.162344), "norm-2-1"=>Dict("norm"=>0.0640643), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.505434), "dot-1-2"=>Dict("dot"=>-0.180713), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0740083), "dot-1-2"=>Dict("dot"=>-0.0738982)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.465071), "dot-1-2"=>Dict("dot"=>-0.0691788), "const_1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-2-1"=>Dict("false"=>0.0549196), "compare-3-2"=>Dict("false"=>0.0715782)…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0682483), "norm-2-1"=>Dict("norm"=>-0.309208), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.308841), "norm-2-1"=>Dict("norm"=>-0.0810245), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0995431), "const_1"=>Dict("const_1"=>0.0415581), "comp…
  ⋮             => ⋮

julia> count(sparse5)
564

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T16:06:16.626
STEP 1 ================================
prereg loss 24.588352 reg_l1 87.64461 reg_l2 27.336674
loss 29.847029
STEP 2 ================================
prereg loss 14.073548 reg_l1 87.50461 reg_l2 27.262869
loss 19.323826
STEP 3 ================================
prereg loss 9.29701 reg_l1 87.39108 reg_l2 27.203833
loss 14.540476
STEP 4 ================================
prereg loss 7.7301984 reg_l1 87.301834 reg_l2 27.154554
loss 12.968308
STEP 5 ================================
prereg loss 7.397195 reg_l1 87.2405 reg_l2 27.116499
loss 12.631624
STEP 6 ================================
prereg loss 7.2101674 reg_l1 87.20563 reg_l2 27.0896
loss 12.442505
STEP 7 ================================
prereg loss 7.056356 reg_l1 87.18968 reg_l2 27.071384
loss 12.287737
STEP 8 ================================
prereg loss 6.9343033 reg_l1 87.18618 reg_l2 27.058912
loss 12.165474
STEP 9 ================================
prereg loss 6.6233582 reg_l1 87.189095 reg_l2 27.05033
loss 11.854704
STEP 10 ================================
prereg loss 5.8259144 reg_l1 87.195694 reg_l2 27.04514
loss 11.057655
STEP 11 ================================
prereg loss 4.7750793 reg_l1 87.20395 reg_l2 27.042484
loss 10.007317
STEP 12 ================================
prereg loss 3.7596045 reg_l1 87.210976 reg_l2 27.040922
loss 8.992263
STEP 13 ================================
prereg loss 2.8604133 reg_l1 87.21397 reg_l2 27.03875
loss 8.093251
STEP 14 ================================
prereg loss 2.156463 reg_l1 87.210594 reg_l2 27.034449
loss 7.389098
STEP 15 ================================
prereg loss 1.8204759 reg_l1 87.199814 reg_l2 27.027363
loss 7.052465
STEP 16 ================================
prereg loss 1.9319943 reg_l1 87.1818 reg_l2 27.017225
loss 7.1629024
STEP 17 ================================
prereg loss 2.3567457 reg_l1 87.15734 reg_l2 27.004135
loss 7.586186
STEP 18 ================================
prereg loss 2.7865348 reg_l1 87.127335 reg_l2 26.988611
loss 8.014175
STEP 19 ================================
prereg loss 2.935973 reg_l1 87.09288 reg_l2 26.971231
loss 8.161546
STEP 20 ================================
prereg loss 2.7135265 reg_l1 87.055145 reg_l2 26.952442
loss 7.9368353
STEP 21 ================================
prereg loss 2.2369254 reg_l1 87.0143 reg_l2 26.932526
loss 7.4577827
STEP 22 ================================
prereg loss 1.7186092 reg_l1 86.970146 reg_l2 26.911547
loss 6.936818
STEP 23 ================================
prereg loss 1.3435087 reg_l1 86.9223 reg_l2 26.889444
loss 6.558847
STEP 24 ================================
prereg loss 1.1842457 reg_l1 86.87031 reg_l2 26.866213
loss 6.396464
STEP 25 ================================
prereg loss 1.1986996 reg_l1 86.81392 reg_l2 26.84187
loss 6.4075346
STEP 26 ================================
prereg loss 1.2875923 reg_l1 86.7537 reg_l2 26.81662
loss 6.4928145
STEP 27 ================================
prereg loss 1.3678484 reg_l1 86.6905 reg_l2 26.790874
loss 6.5692782
STEP 28 ================================
prereg loss 1.4086971 reg_l1 86.6257 reg_l2 26.765087
loss 6.6062393
STEP 29 ================================
prereg loss 1.4181061 reg_l1 86.560616 reg_l2 26.73984
loss 6.611743
STEP 30 ================================
prereg loss 1.4021614 reg_l1 86.49653 reg_l2 26.715563
loss 6.5919533
STEP 31 ================================
prereg loss 1.3549744 reg_l1 86.43426 reg_l2 26.69265
loss 6.5410295
STEP 32 ================================
prereg loss 1.2629919 reg_l1 86.37437 reg_l2 26.67124
loss 6.4454536
STEP 33 ================================
prereg loss 1.1293486 reg_l1 86.31663 reg_l2 26.651201
loss 6.3083463
STEP 34 ================================
prereg loss 0.98360777 reg_l1 86.260605 reg_l2 26.632254
loss 6.159244
STEP 35 ================================
prereg loss 0.86851734 reg_l1 86.205444 reg_l2 26.613924
loss 6.040844
STEP 36 ================================
prereg loss 0.81290793 reg_l1 86.151276 reg_l2 26.59565
loss 5.981984
STEP 37 ================================
prereg loss 0.8179888 reg_l1 86.095924 reg_l2 26.57681
loss 5.983744
STEP 38 ================================
prereg loss 0.8496341 reg_l1 86.037704 reg_l2 26.556938
loss 6.011896
STEP 39 ================================
prereg loss 0.86770564 reg_l1 85.97626 reg_l2 26.535694
loss 6.0262814
STEP 40 ================================
prereg loss 0.85886896 reg_l1 85.91297 reg_l2 26.513037
loss 6.013647
STEP 41 ================================
prereg loss 0.83644646 reg_l1 85.84743 reg_l2 26.489178
loss 5.987292
STEP 42 ================================
prereg loss 0.81189066 reg_l1 85.78 reg_l2 26.464495
loss 5.9586906
STEP 43 ================================
prereg loss 0.7840229 reg_l1 85.71149 reg_l2 26.439402
loss 5.926712
STEP 44 ================================
prereg loss 0.7453641 reg_l1 85.64284 reg_l2 26.414297
loss 5.8839345
STEP 45 ================================
prereg loss 0.6989836 reg_l1 85.574524 reg_l2 26.3894
loss 5.833455
STEP 46 ================================
prereg loss 0.66228163 reg_l1 85.50677 reg_l2 26.364908
loss 5.7926874
STEP 47 ================================
prereg loss 0.65133995 reg_l1 85.43936 reg_l2 26.34075
loss 5.7777014
STEP 48 ================================
prereg loss 0.6659556 reg_l1 85.37249 reg_l2 26.316875
loss 5.788305
STEP 49 ================================
prereg loss 0.6886929 reg_l1 85.305466 reg_l2 26.293188
loss 5.8070207
STEP 50 ================================
prereg loss 0.6995216 reg_l1 85.2384 reg_l2 26.269592
loss 5.8138256
STEP 51 ================================
prereg loss 0.691679 reg_l1 85.17119 reg_l2 26.246124
loss 5.80195
STEP 52 ================================
prereg loss 0.67361194 reg_l1 85.10527 reg_l2 26.222836
loss 5.779928
STEP 53 ================================
prereg loss 0.65799713 reg_l1 85.03955 reg_l2 26.199785
loss 5.7603703
STEP 54 ================================
prereg loss 0.6505398 reg_l1 84.97399 reg_l2 26.177052
loss 5.748979
STEP 55 ================================
prereg loss 0.6484583 reg_l1 84.90859 reg_l2 26.154636
loss 5.742974
STEP 56 ================================
prereg loss 0.64659786 reg_l1 84.84338 reg_l2 26.132416
loss 5.7372007
STEP 57 ================================
prereg loss 0.64425164 reg_l1 84.77971 reg_l2 26.110203
loss 5.7310343
STEP 58 ================================
prereg loss 0.6432978 reg_l1 84.715225 reg_l2 26.087835
loss 5.726211
STEP 59 ================================
prereg loss 0.6446125 reg_l1 84.64955 reg_l2 26.065008
loss 5.723585
STEP 60 ================================
prereg loss 0.6450378 reg_l1 84.5821 reg_l2 26.04155
loss 5.7199636
STEP 61 ================================
prereg loss 0.6407872 reg_l1 84.51351 reg_l2 26.017435
loss 5.711598
STEP 62 ================================
prereg loss 0.63207483 reg_l1 84.44558 reg_l2 25.992712
loss 5.6988096
STEP 63 ================================
prereg loss 0.62296253 reg_l1 84.37666 reg_l2 25.967524
loss 5.685562
STEP 64 ================================
prereg loss 0.61708385 reg_l1 84.30722 reg_l2 25.942122
loss 5.675517
STEP 65 ================================
prereg loss 0.6147984 reg_l1 84.237236 reg_l2 25.916752
loss 5.6690326
STEP 66 ================================
prereg loss 0.61397773 reg_l1 84.167274 reg_l2 25.891584
loss 5.6640143
STEP 67 ================================
prereg loss 0.6121598 reg_l1 84.097466 reg_l2 25.866796
loss 5.6580076
STEP 68 ================================
prereg loss 0.60951614 reg_l1 84.02813 reg_l2 25.842379
loss 5.6512036
STEP 69 ================================
prereg loss 0.6071747 reg_l1 83.95892 reg_l2 25.81825
loss 5.64471
STEP 70 ================================
prereg loss 0.6048784 reg_l1 83.88966 reg_l2 25.794355
loss 5.638258
STEP 71 ================================
prereg loss 0.6014556 reg_l1 83.82058 reg_l2 25.770521
loss 5.6306906
STEP 72 ================================
prereg loss 0.5966095 reg_l1 83.75217 reg_l2 25.74671
loss 5.6217394
STEP 73 ================================
prereg loss 0.5917198 reg_l1 83.68329 reg_l2 25.722857
loss 5.6127167
STEP 74 ================================
prereg loss 0.58864784 reg_l1 83.61383 reg_l2 25.698957
loss 5.6054773
STEP 75 ================================
prereg loss 0.5878417 reg_l1 83.543915 reg_l2 25.674988
loss 5.6004763
STEP 76 ================================
prereg loss 0.58793926 reg_l1 83.474525 reg_l2 25.651007
loss 5.5964108
STEP 77 ================================
prereg loss 0.5871236 reg_l1 83.4063 reg_l2 25.626999
loss 5.5915017
STEP 78 ================================
prereg loss 0.5848461 reg_l1 83.33951 reg_l2 25.602945
loss 5.5852165
STEP 79 ================================
prereg loss 0.5819659 reg_l1 83.27282 reg_l2 25.578768
loss 5.578335
STEP 80 ================================
prereg loss 0.5794423 reg_l1 83.205925 reg_l2 25.55448
loss 5.571798
STEP 81 ================================
prereg loss 0.5774659 reg_l1 83.13962 reg_l2 25.530033
loss 5.565843
STEP 82 ================================
prereg loss 0.57579166 reg_l1 83.072235 reg_l2 25.505423
loss 5.560126
STEP 83 ================================
prereg loss 0.57440215 reg_l1 83.00415 reg_l2 25.48073
loss 5.5546513
STEP 84 ================================
prereg loss 0.5736375 reg_l1 82.93494 reg_l2 25.456078
loss 5.5497336
STEP 85 ================================
prereg loss 0.5731534 reg_l1 82.86596 reg_l2 25.431532
loss 5.5451107
STEP 86 ================================
prereg loss 0.5723826 reg_l1 82.7978 reg_l2 25.407175
loss 5.5402503
STEP 87 ================================
prereg loss 0.57098436 reg_l1 82.73293 reg_l2 25.38305
loss 5.5349603
STEP 88 ================================
prereg loss 0.56909126 reg_l1 82.66927 reg_l2 25.359129
loss 5.5292478
STEP 89 ================================
prereg loss 0.56711227 reg_l1 82.604706 reg_l2 25.335299
loss 5.5233946
STEP 90 ================================
prereg loss 0.5653753 reg_l1 82.54082 reg_l2 25.31159
loss 5.517824
STEP 91 ================================
prereg loss 0.5639036 reg_l1 82.47551 reg_l2 25.287859
loss 5.5124345
STEP 92 ================================
prereg loss 0.56269705 reg_l1 82.41017 reg_l2 25.264053
loss 5.507307
STEP 93 ================================
prereg loss 0.5617159 reg_l1 82.34511 reg_l2 25.24018
loss 5.5024223
STEP 94 ================================
prereg loss 0.560908 reg_l1 82.28023 reg_l2 25.216246
loss 5.497721
STEP 95 ================================
prereg loss 0.5601193 reg_l1 82.21489 reg_l2 25.192268
loss 5.4930124
STEP 96 ================================
prereg loss 0.5591495 reg_l1 82.14939 reg_l2 25.168283
loss 5.4881124
STEP 97 ================================
prereg loss 0.5579904 reg_l1 82.08374 reg_l2 25.144297
loss 5.483015
STEP 98 ================================
prereg loss 0.556602 reg_l1 82.01696 reg_l2 25.120377
loss 5.4776196
STEP 99 ================================
prereg loss 0.5552918 reg_l1 81.94955 reg_l2 25.096487
loss 5.4722643
STEP 100 ================================
prereg loss 0.5542126 reg_l1 81.88186 reg_l2 25.072622
loss 5.467124
2022-06-26T16:35:00.966

julia> serialize("sparse5-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse5-after-100-steps-opt.ser", opt)

julia> count_interval(sparse5, -0.001f0, 0.001f0)
11

julia> count_interval(sparse5, -0.01f0, 0.01f0)
22

julia> count_interval(sparse5, -0.02f0, 0.02f0)
32

julia> count_interval(sparse5, -0.03f0, 0.03f0)
57

julia> count_interval(sparse5, -0.04f0, 0.04f0)
90

julia> count_interval(sparse5, -0.035f0, 0.035f0)
77

julia> sparse6 = sparsecopy(trainable["network_matrix"], 0.035f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 38 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.624569), "eos"=>Dict("char"=>-0.0538681)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.498795), "dot-2-2"=>Dict("dot"=>-0.140624), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500905), "accum-1-2"=>Dict("dict"=>0.273798), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.620794), "accum-1-1"=>Dict("dict"=>0.187603), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208462), "accum-1-1"=>Dict("dict"=>0.133629), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0382812), "dot-2-1"=>Dict("dot"=>-0.123287)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0530322), "norm-2-1"=>Dict("norm"=>0.114906), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0748954), "accum-1-2"=>Dict("dict"=>0.217998), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0424763), "accum-4-1"=>Dict("dict"=>-0.0387523)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.437468)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.105087)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16401), "eos"=>Dict("char"=>-0.165561)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.092103), "accum-3-1"=>Dict("dict"=>0.0649227), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.284893), "dot-2-2"=>Dict("dot"=>-0.14086), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.310706), "norm-2-1"=>Dict("norm"=>0.239986), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264682), "dot-2-2"=>Dict("dot"=>-0.0861608), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.866871), "eos"=>Dict("char"=>-0.168826)), "dict-1"=>Dict…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152496), "norm-2-1"=>Dict("norm"=>0.0524601), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.49528), "dot-1-2"=>Dict("dot"=>-0.171768), "const_1"=>Dic…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0377467), "dot-1-2"=>Dict("dot"=>-0.0650893)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.437934), "const_1"=>Dict("const_1"=>-0.0943157), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-3-2"=>Dict("false"=>0.0612659), "norm-3-2"=>Dict("norm"=>0.191481), "no…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.063485), "norm-2-1"=>Dict("norm"=>-0.306142), "dot-2-2"…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.300142), "norm-2-1"=>Dict("norm"=>-0.0732445), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0446614), "const_1"=>Dict("const_1"=>0.0366508)), "dic…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse6
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 38 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.624569), "eos"=>Dict("char"=>-0.0538681)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.498795), "dot-2-2"=>Dict("dot"=>-0.140624), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500905), "accum-1-2"=>Dict("dict"=>0.273798), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.620794), "accum-1-1"=>Dict("dict"=>0.187603), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208462), "accum-1-1"=>Dict("dict"=>0.133629), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0382812), "dot-2-1"=>Dict("dot"=>-0.123287)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0530322), "norm-2-1"=>Dict("norm"=>0.114906), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0748954), "accum-1-2"=>Dict("dict"=>0.217998), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0424763), "accum-4-1"=>Dict("dict"=>-0.0387523)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.437468)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.105087)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16401), "eos"=>Dict("char"=>-0.165561)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.092103), "accum-3-1"=>Dict("dict"=>0.0649227), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.284893), "dot-2-2"=>Dict("dot"=>-0.14086), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.310706), "norm-2-1"=>Dict("norm"=>0.239986), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264682), "dot-2-2"=>Dict("dot"=>-0.0861608), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.866871), "eos"=>Dict("char"=>-0.168826)), "dict-1"=>Dict…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152496), "norm-2-1"=>Dict("norm"=>0.0524601), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.49528), "dot-1-2"=>Dict("dot"=>-0.171768), "const_1"=>Dic…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0377467), "dot-1-2"=>Dict("dot"=>-0.0650893)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.437934), "const_1"=>Dict("const_1"=>-0.0943157), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-3-2"=>Dict("false"=>0.0612659), "norm-3-2"=>Dict("norm"=>0.191481), "no…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.063485), "norm-2-1"=>Dict("norm"=>-0.306142), "dot-2-2"…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.300142), "norm-2-1"=>Dict("norm"=>-0.0732445), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0446614), "const_1"=>Dict("const_1"=>0.0366508)), "dic…
  ⋮             => ⋮

julia> count(sparse6)
487

julia> # let's do 100 more steps with this regularization

julia> steps!(100)
2022-06-26T16:58:43.420
STEP 1 ================================
prereg loss 4.7622166 reg_l1 80.338776 reg_l2 25.008953
loss 9.582542
STEP 2 ================================
prereg loss 3.9024541 reg_l1 80.30586 reg_l2 24.986488
loss 8.720806
STEP 3 ================================
prereg loss 2.9226763 reg_l1 80.26669 reg_l2 24.964193
loss 7.738678
STEP 4 ================================
prereg loss 2.3966131 reg_l1 80.22251 reg_l2 24.94062
loss 7.209964
STEP 5 ================================
prereg loss 2.2107708 reg_l1 80.17244 reg_l2 24.91442
loss 7.021117
STEP 6 ================================
prereg loss 1.9566786 reg_l1 80.11583 reg_l2 24.885128
loss 6.763628
STEP 7 ================================
prereg loss 1.5265589 reg_l1 80.05417 reg_l2 24.853407
loss 6.3298087
STEP 8 ================================
prereg loss 1.1645913 reg_l1 79.99113 reg_l2 24.820642
loss 5.964059
STEP 9 ================================
prereg loss 1.0824105 reg_l1 79.92786 reg_l2 24.788574
loss 5.878082
STEP 10 ================================
prereg loss 1.2126933 reg_l1 79.863174 reg_l2 24.758642
loss 6.0044837
STEP 11 ================================
prereg loss 1.3175822 reg_l1 79.798065 reg_l2 24.731558
loss 6.105466
STEP 12 ================================
prereg loss 1.2810757 reg_l1 79.73113 reg_l2 24.707222
loss 6.0649433
STEP 13 ================================
prereg loss 1.2125815 reg_l1 79.66191 reg_l2 24.68479
loss 5.992296
STEP 14 ================================
prereg loss 1.2396718 reg_l1 79.59058 reg_l2 24.663033
loss 6.0151067
STEP 15 ================================
prereg loss 1.3271794 reg_l1 79.516716 reg_l2 24.640728
loss 6.098182
STEP 16 ================================
prereg loss 1.331619 reg_l1 79.443016 reg_l2 24.617117
loss 6.0982
STEP 17 ================================
prereg loss 1.1958227 reg_l1 79.37284 reg_l2 24.592121
loss 5.958193
STEP 18 ================================
prereg loss 1.017152 reg_l1 79.307594 reg_l2 24.566345
loss 5.7756076
STEP 19 ================================
prereg loss 0.91778475 reg_l1 79.24572 reg_l2 24.54081
loss 5.672528
STEP 20 ================================
prereg loss 0.9155897 reg_l1 79.191826 reg_l2 24.516592
loss 5.6670995
STEP 21 ================================
prereg loss 0.93602675 reg_l1 79.14275 reg_l2 24.494587
loss 5.684592
STEP 22 ================================
prereg loss 0.9190896 reg_l1 79.097435 reg_l2 24.475042
loss 5.6649356
STEP 23 ================================
prereg loss 0.8857732 reg_l1 79.0576 reg_l2 24.457756
loss 5.629229
STEP 24 ================================
prereg loss 0.88468647 reg_l1 79.021805 reg_l2 24.441925
loss 5.6259947
STEP 25 ================================
prereg loss 0.9184918 reg_l1 78.98813 reg_l2 24.426586
loss 5.657779
STEP 26 ================================
prereg loss 0.94306535 reg_l1 78.953766 reg_l2 24.410957
loss 5.680291
STEP 27 ================================
prereg loss 0.9258411 reg_l1 78.91936 reg_l2 24.394554
loss 5.661002
STEP 28 ================================
prereg loss 0.881995 reg_l1 78.88538 reg_l2 24.377338
loss 5.615118
STEP 29 ================================
prereg loss 0.85026443 reg_l1 78.853294 reg_l2 24.359594
loss 5.581462
STEP 30 ================================
prereg loss 0.8411581 reg_l1 78.8211 reg_l2 24.341818
loss 5.5704236
STEP 31 ================================
prereg loss 0.82982785 reg_l1 78.78835 reg_l2 24.32436
loss 5.557129
STEP 32 ================================
prereg loss 0.7933496 reg_l1 78.75585 reg_l2 24.307386
loss 5.5187006
STEP 33 ================================
prereg loss 0.7430088 reg_l1 78.72149 reg_l2 24.290766
loss 5.4662976
STEP 34 ================================
prereg loss 0.7101018 reg_l1 78.685555 reg_l2 24.274025
loss 5.431235
STEP 35 ================================
prereg loss 0.7069413 reg_l1 78.64952 reg_l2 24.256786
loss 5.4259124
STEP 36 ================================
prereg loss 0.71708494 reg_l1 78.61336 reg_l2 24.238573
loss 5.433886
STEP 37 ================================
prereg loss 0.71761507 reg_l1 78.57415 reg_l2 24.219288
loss 5.432064
STEP 38 ================================
prereg loss 0.70625514 reg_l1 78.53123 reg_l2 24.199087
loss 5.4181285
STEP 39 ================================
prereg loss 0.6956917 reg_l1 78.48668 reg_l2 24.178373
loss 5.4048924
STEP 40 ================================
prereg loss 0.6910126 reg_l1 78.44161 reg_l2 24.157616
loss 5.3975096
STEP 41 ================================
prereg loss 0.6838978 reg_l1 78.395134 reg_l2 24.137207
loss 5.3876057
STEP 42 ================================
prereg loss 0.6672874 reg_l1 78.34783 reg_l2 24.117281
loss 5.368157
STEP 43 ================================
prereg loss 0.6459256 reg_l1 78.30029 reg_l2 24.097727
loss 5.343943
STEP 44 ================================
prereg loss 0.630252 reg_l1 78.25215 reg_l2 24.078318
loss 5.325381
STEP 45 ================================
prereg loss 0.62354493 reg_l1 78.20406 reg_l2 24.058722
loss 5.3157883
STEP 46 ================================
prereg loss 0.61951536 reg_l1 78.154655 reg_l2 24.038698
loss 5.3087945
STEP 47 ================================
prereg loss 0.6127401 reg_l1 78.10475 reg_l2 24.018152
loss 5.299025
STEP 48 ================================
prereg loss 0.6055964 reg_l1 78.05581 reg_l2 23.997206
loss 5.288945
STEP 49 ================================
prereg loss 0.60284936 reg_l1 78.00743 reg_l2 23.976158
loss 5.283295
STEP 50 ================================
prereg loss 0.60419667 reg_l1 77.95937 reg_l2 23.955284
loss 5.281759
STEP 51 ================================
prereg loss 0.604314 reg_l1 77.91139 reg_l2 23.93484
loss 5.2789974
STEP 52 ================================
prereg loss 0.5995332 reg_l1 77.8645 reg_l2 23.91492
loss 5.2714033
STEP 53 ================================
prereg loss 0.5918238 reg_l1 77.82187 reg_l2 23.89544
loss 5.261136
STEP 54 ================================
prereg loss 0.5851249 reg_l1 77.78076 reg_l2 23.876232
loss 5.251971
STEP 55 ================================
prereg loss 0.5804316 reg_l1 77.73894 reg_l2 23.85703
loss 5.2447677
STEP 56 ================================
prereg loss 0.57557416 reg_l1 77.69789 reg_l2 23.837696
loss 5.2374477
STEP 57 ================================
prereg loss 0.5691347 reg_l1 77.65635 reg_l2 23.818077
loss 5.2285156
STEP 58 ================================
prereg loss 0.5627184 reg_l1 77.61319 reg_l2 23.798185
loss 5.2195096
STEP 59 ================================
prereg loss 0.5586213 reg_l1 77.57 reg_l2 23.778131
loss 5.2128215
STEP 60 ================================
prereg loss 0.5567468 reg_l1 77.5266 reg_l2 23.758009
loss 5.208343
STEP 61 ================================
prereg loss 0.55491906 reg_l1 77.48222 reg_l2 23.737902
loss 5.2038527
STEP 62 ================================
prereg loss 0.55181503 reg_l1 77.43732 reg_l2 23.717842
loss 5.198054
STEP 63 ================================
prereg loss 0.54836726 reg_l1 77.39182 reg_l2 23.697712
loss 5.1918764
STEP 64 ================================
prereg loss 0.5459518 reg_l1 77.34553 reg_l2 23.677399
loss 5.186683
STEP 65 ================================
prereg loss 0.5442847 reg_l1 77.29823 reg_l2 23.656754
loss 5.1821785
STEP 66 ================================
prereg loss 0.5419705 reg_l1 77.24891 reg_l2 23.63572
loss 5.1769047
STEP 67 ================================
prereg loss 0.538638 reg_l1 77.19781 reg_l2 23.614346
loss 5.1705065
STEP 68 ================================
prereg loss 0.5350894 reg_l1 77.14587 reg_l2 23.592752
loss 5.1638412
STEP 69 ================================
prereg loss 0.53220415 reg_l1 77.09525 reg_l2 23.571095
loss 5.1579194
STEP 70 ================================
prereg loss 0.5298144 reg_l1 77.04403 reg_l2 23.549492
loss 5.152456
STEP 71 ================================
prereg loss 0.52708906 reg_l1 76.99218 reg_l2 23.52805
loss 5.14662
STEP 72 ================================
prereg loss 0.5240958 reg_l1 76.940704 reg_l2 23.50677
loss 5.140538
STEP 73 ================================
prereg loss 0.5214953 reg_l1 76.89004 reg_l2 23.485521
loss 5.1348977
STEP 74 ================================
prereg loss 0.51947284 reg_l1 76.84184 reg_l2 23.464283
loss 5.129983
STEP 75 ================================
prereg loss 0.5176499 reg_l1 76.794205 reg_l2 23.442913
loss 5.1253023
STEP 76 ================================
prereg loss 0.51566 reg_l1 76.74617 reg_l2 23.421398
loss 5.12043
STEP 77 ================================
prereg loss 0.5136941 reg_l1 76.69682 reg_l2 23.399744
loss 5.115504
STEP 78 ================================
prereg loss 0.5120022 reg_l1 76.647484 reg_l2 23.378084
loss 5.1108513
STEP 79 ================================
prereg loss 0.5104087 reg_l1 76.598145 reg_l2 23.356472
loss 5.1062975
STEP 80 ================================
prereg loss 0.5086708 reg_l1 76.54958 reg_l2 23.33498
loss 5.101646
STEP 81 ================================
prereg loss 0.50675607 reg_l1 76.50133 reg_l2 23.31365
loss 5.096836
STEP 82 ================================
prereg loss 0.50496495 reg_l1 76.45283 reg_l2 23.292425
loss 5.0921345
STEP 83 ================================
prereg loss 0.5034037 reg_l1 76.4041 reg_l2 23.271208
loss 5.0876493
STEP 84 ================================
prereg loss 0.50190204 reg_l1 76.356544 reg_l2 23.249994
loss 5.083295
STEP 85 ================================
prereg loss 0.5003335 reg_l1 76.30919 reg_l2 23.228683
loss 5.0788846
STEP 86 ================================
prereg loss 0.4988198 reg_l1 76.2608 reg_l2 23.207333
loss 5.074468
STEP 87 ================================
prereg loss 0.49753007 reg_l1 76.212425 reg_l2 23.18591
loss 5.0702753
STEP 88 ================================
prereg loss 0.4963705 reg_l1 76.16387 reg_l2 23.164503
loss 5.0662026
STEP 89 ================================
prereg loss 0.495156 reg_l1 76.11492 reg_l2 23.143116
loss 5.062051
STEP 90 ================================
prereg loss 0.493866 reg_l1 76.06612 reg_l2 23.12175
loss 5.0578327
STEP 91 ================================
prereg loss 0.49261 reg_l1 76.01571 reg_l2 23.100378
loss 5.0535526
STEP 92 ================================
prereg loss 0.4914649 reg_l1 75.96441 reg_l2 23.078989
loss 5.0493298
STEP 93 ================================
prereg loss 0.49043432 reg_l1 75.91308 reg_l2 23.05758
loss 5.045219
STEP 94 ================================
prereg loss 0.48939973 reg_l1 75.86131 reg_l2 23.036097
loss 5.0410786
STEP 95 ================================
prereg loss 0.48835918 reg_l1 75.81095 reg_l2 23.014565
loss 5.037016
STEP 96 ================================
prereg loss 0.48736164 reg_l1 75.76183 reg_l2 22.992998
loss 5.0330715
STEP 97 ================================
prereg loss 0.48644695 reg_l1 75.711655 reg_l2 22.971403
loss 5.029146
STEP 98 ================================
prereg loss 0.48561692 reg_l1 75.66146 reg_l2 22.949842
loss 5.025305
STEP 99 ================================
prereg loss 0.48483476 reg_l1 75.611534 reg_l2 22.928295
loss 5.021527
STEP 100 ================================
prereg loss 0.4840798 reg_l1 75.56156 reg_l2 22.90677
loss 5.0177736
2022-06-26T17:27:24.112

julia> serialize("sparse6-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse6-after-100-steps-opt.ser", opt)

julia> count_interval(sparse6, -0.001f0, 0.001f0)
74

julia> count_interval(sparse6, -0.01f0, 0.01f0)
88

julia> count_interval(sparse6, -0.02f0, 0.02f0)
96

julia> count_interval(sparse6, -0.03f0, 0.03f0)
105

julia> # wow, impressive sparsification this time;

julia> # let's cut 88 this time

julia> sparse7 = sparsecopy(trainable["network_matrix"], 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 38 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.628355), "eos"=>Dict("char"=>-0.0518717)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.506288), "dot-2-2"=>Dict("dot"=>-0.124343), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0459664), "accum-1-2"=>Dict("dict"=>0.277576), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.569567), "accum-1-1"=>Dict("dict"=>0.186765), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.213107), "accum-1-1"=>Dict("dict"=>0.137431), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0117869), "dot-2-1"=>Dict("dot"=>-0.114015)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0326672), "norm-2-1"=>Dict("norm"=>0.113723), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0795275), "accum-1-2"=>Dict("dict"=>0.22328), "input"=…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0409616), "accum-4-1"=>Dict("dict"=>-0.0337115)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.40513)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.106706)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.158892), "eos"=>Dict("char"=>-0.167264)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0928913), "accum-3-1"=>Dict("dict"=>0.0634402), "accum…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.289566), "dot-2-2"=>Dict("dot"=>-0.146518), "norm-3-1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.303162), "norm-2-1"=>Dict("norm"=>0.239409), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.263478), "dot-2-2"=>Dict("dot"=>-0.0810841), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.833983), "eos"=>Dict("char"=>-0.16936)), "dict-1"=>Dict(…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.15612), "norm-2-1"=>Dict("norm"=>0.0544599), "dot-3-1…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.481697), "dot-1-2"=>Dict("dot"=>-0.163256), "const_1"=>Di…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0331634)), "dict-1"=>Dict("dot-1-2"=>Dict("dot"=>0.033…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.387295), "const_1"=>Dict("const_1"=>-0.0817838), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>0.194711), "norm-2-1"=>Dict("norm"=>0.22513), "dot-2-2"…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0549963), "norm-2-1"=>Dict("norm"=>-0.307358), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.277496), "norm-2-1"=>Dict("norm"=>-0.0774656), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.038239)))
  ⋮             => ⋮

julia> count(sparse7)
476

julia> count(sparse6)
564

julia> # ah, we have not reinitialized

julia> # the optimizer, and that was the error

julia> # let's commit the log and ponder what to do

julia> close(io)
```

The sparse6 part of the run is annulled, has to be redone.

```
$ diff test.jl test-original.jl
36c36
<     l += 0.06f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse5-after-100-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

let's do the run:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 39 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.624569), "eos"=>Dict("char"=>-0.0538681)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.498795), "dot-2-2"=>Dict("dot"=>-0.140624), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500905), "accum-1-2"=>Dict("dict"=>0.273798), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.620794), "accum-1-1"=>Dict("dict"=>0.187603), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208462), "accum-1-1"=>Dict("dict"=>0.133629), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0382812), "dot-2-1"=>Dict("dot"=>-0.123287)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0530322), "eos"=>Dict("char"=>0.0725988), "norm-2-1"=>…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0748954), "accum-1-2"=>Dict("dict"=>0.217998), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0424763), "accum-4-1"=>Dict("dict"=>-0.0387523)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.437468)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.105087)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16401), "eos"=>Dict("char"=>-0.165561)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.092103), "accum-3-1"=>Dict("dict"=>0.0649227), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.284893), "dot-2-2"=>Dict("dot"=>-0.14086), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.310706), "dot-4-2"=>Dict("dot"=>-0.153675), "norm-2-1"=…
  "compare-4-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.00557858), "compare-2-1"=>Dict("false"=>0.0270125), "n…
  "dot-1-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0305354), "input"=>Dict("char"=>0.866871), "eos"=>D…
  "output"      => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.0420614), "norm-5-2"=>Dict("norm"=>0.152496), "norm-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.49528), "dot-1-2"=>Dict("dot"=>-0.171768), "const_1"=>Dic…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0377467), "dot-1-2"=>Dict("dot"=>-0.0650893)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.437934), "dot-1-2"=>Dict("dot"=>0.000684974), "const_1"…
  "compare-4-1" => Dict("dict-2"=>Dict("compare-3-2"=>Dict("false"=>0.0612659), "norm-3-2"=>Dict("norm"=>0.191481), "no…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.063485), "norm-2-1"=>Dict("norm"=>-0.306142), "dot-2-2"…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.300142), "dot-4-2"=>Dict("dot"=>0.103599), "norm-2-1"=…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0446614), "const_1"=>Dict("const_1"=>0.0366508), "comp…
  ⋮             => ⋮

julia> sparse6 = sparsecopy(trainable["network_matrix"], 0.035f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 38 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.624569), "eos"=>Dict("char"=>-0.0538681)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.498795), "dot-2-2"=>Dict("dot"=>-0.140624), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500905), "accum-1-2"=>Dict("dict"=>0.273798), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.620794), "accum-1-1"=>Dict("dict"=>0.187603), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208462), "accum-1-1"=>Dict("dict"=>0.133629), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0382812), "dot-2-1"=>Dict("dot"=>-0.123287)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0530322), "norm-2-1"=>Dict("norm"=>0.114906), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0748954), "accum-1-2"=>Dict("dict"=>0.217998), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0424763), "accum-4-1"=>Dict("dict"=>-0.0387523)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.437468)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.105087)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16401), "eos"=>Dict("char"=>-0.165561)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.092103), "accum-3-1"=>Dict("dict"=>0.0649227), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.284893), "dot-2-2"=>Dict("dot"=>-0.14086), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.310706), "norm-2-1"=>Dict("norm"=>0.239986), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264682), "dot-2-2"=>Dict("dot"=>-0.0861608), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.866871), "eos"=>Dict("char"=>-0.168826)), "dict-1"=>Dict…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152496), "norm-2-1"=>Dict("norm"=>0.0524601), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.49528), "dot-1-2"=>Dict("dot"=>-0.171768), "const_1"=>Dic…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0377467), "dot-1-2"=>Dict("dot"=>-0.0650893)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.437934), "const_1"=>Dict("const_1"=>-0.0943157), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>0.191481), "compare-2-1"=>Dict("false"=>0.0358769), "no…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.063485), "norm-2-1"=>Dict("norm"=>-0.306142), "dot-2-2"…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.300142), "norm-2-1"=>Dict("norm"=>-0.0732445), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0446614), "const_1"=>Dict("const_1"=>0.0366508)), "dic…
  ⋮             => ⋮

julia> count(sparse6)
487

julia> trainable["network_matrix"] = sparse6
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 38 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.624569), "eos"=>Dict("char"=>-0.0538681)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.498795), "dot-2-2"=>Dict("dot"=>-0.140624), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0500905), "accum-1-2"=>Dict("dict"=>0.273798), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.620794), "accum-1-1"=>Dict("dict"=>0.187603), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.208462), "accum-1-1"=>Dict("dict"=>0.133629), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0382812), "dot-2-1"=>Dict("dot"=>-0.123287)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0530322), "norm-2-1"=>Dict("norm"=>0.114906), "dot-2-2…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0748954), "accum-1-2"=>Dict("dict"=>0.217998), "input"…
  "norm-5-1"    => Dict("dict"=>Dict("accum-3-1"=>Dict("dict"=>-0.0424763), "accum-4-1"=>Dict("dict"=>-0.0387523)))
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.437468)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.105087)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.16401), "eos"=>Dict("char"=>-0.165561)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.092103), "accum-3-1"=>Dict("dict"=>0.0649227), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.284893), "dot-2-2"=>Dict("dot"=>-0.14086), "norm-3-1"=>…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.310706), "norm-2-1"=>Dict("norm"=>0.239986), "dot-2-2"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264682), "dot-2-2"=>Dict("dot"=>-0.0861608), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.866871), "eos"=>Dict("char"=>-0.168826)), "dict-1"=>Dict…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.152496), "norm-2-1"=>Dict("norm"=>0.0524601), "dot-3-…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.49528), "dot-1-2"=>Dict("dot"=>-0.171768), "const_1"=>Dic…
  "compare-2-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0377467), "dot-1-2"=>Dict("dot"=>-0.0650893)), "dict-1"…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.437934), "const_1"=>Dict("const_1"=>-0.0943157), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>0.191481), "compare-2-1"=>Dict("false"=>0.0358769), "no…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.063485), "norm-2-1"=>Dict("norm"=>-0.306142), "dot-2-2"…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.300142), "norm-2-1"=>Dict("norm"=>-0.0732445), "dot-2-…
  "compare-2-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0446614), "const_1"=>Dict("const_1"=>0.0366508)), "dic…
  ⋮             => ⋮

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T17:48:43.682
STEP 1 ================================
prereg loss 4.7622166 reg_l1 80.338776 reg_l2 25.008953
loss 9.582542
STEP 2 ================================
prereg loss 3.8370173 reg_l1 80.3618 reg_l2 25.0242
loss 8.658725
STEP 3 ================================
prereg loss 2.6029794 reg_l1 80.27427 reg_l2 24.981285
loss 7.4194355
STEP 4 ================================
prereg loss 1.7194203 reg_l1 80.17773 reg_l2 24.932804
loss 6.530084
STEP 5 ================================
prereg loss 1.681225 reg_l1 80.081696 reg_l2 24.887457
loss 6.4861264
STEP 6 ================================
prereg loss 1.6135368 reg_l1 79.99505 reg_l2 24.850983
loss 6.4132395
STEP 7 ================================
prereg loss 1.2781521 reg_l1 79.918526 reg_l2 24.822737
loss 6.0732636
STEP 8 ================================
prereg loss 1.0767894 reg_l1 79.8463 reg_l2 24.798786
loss 5.867567
STEP 9 ================================
prereg loss 1.1836954 reg_l1 79.77199 reg_l2 24.775211
loss 5.9700146
STEP 10 ================================
prereg loss 1.3143976 reg_l1 79.692154 reg_l2 24.749403
loss 6.0959263
STEP 11 ================================
prereg loss 1.2446363 reg_l1 79.60738 reg_l2 24.721067
loss 6.021079
STEP 12 ================================
prereg loss 1.1166258 reg_l1 79.52036 reg_l2 24.691393
loss 5.8878474
STEP 13 ================================
prereg loss 1.1059759 reg_l1 79.434425 reg_l2 24.662088
loss 5.8720417
STEP 14 ================================
prereg loss 1.1629355 reg_l1 79.35347 reg_l2 24.634937
loss 5.924144
STEP 15 ================================
prereg loss 1.1421299 reg_l1 79.28083 reg_l2 24.61137
loss 5.8989797
STEP 16 ================================
prereg loss 1.0425 reg_l1 79.21777 reg_l2 24.59169
loss 5.795566
STEP 17 ================================
prereg loss 0.99770963 reg_l1 79.16318 reg_l2 24.575104
loss 5.7475004
STEP 18 ================================
prereg loss 1.0479106 reg_l1 79.115036 reg_l2 24.560139
loss 5.7948127
STEP 19 ================================
prereg loss 1.0922091 reg_l1 79.0712 reg_l2 24.545326
loss 5.836481
STEP 20 ================================
prereg loss 1.0476539 reg_l1 79.03011 reg_l2 24.529749
loss 5.78946
STEP 21 ================================
prereg loss 0.9630571 reg_l1 78.9912 reg_l2 24.51336
loss 5.702529
STEP 22 ================================
prereg loss 0.93493366 reg_l1 78.9551 reg_l2 24.496756
loss 5.67224
STEP 23 ================================
prereg loss 0.9724606 reg_l1 78.92268 reg_l2 24.480816
loss 5.707822
STEP 24 ================================
prereg loss 0.9874329 reg_l1 78.894165 reg_l2 24.46615
loss 5.7210827
STEP 25 ================================
prereg loss 0.93079585 reg_l1 78.86899 reg_l2 24.452578
loss 5.662935
STEP 26 ================================
prereg loss 0.85080105 reg_l1 78.84502 reg_l2 24.439342
loss 5.581502
STEP 27 ================================
prereg loss 0.8097417 reg_l1 78.82018 reg_l2 24.425373
loss 5.5389524
STEP 28 ================================
prereg loss 0.8018904 reg_l1 78.79186 reg_l2 24.4096
loss 5.529402
STEP 29 ================================
prereg loss 0.7844696 reg_l1 78.75879 reg_l2 24.391497
loss 5.509997
STEP 30 ================================
prereg loss 0.75316954 reg_l1 78.720726 reg_l2 24.371298
loss 5.476413
STEP 31 ================================
prereg loss 0.7368694 reg_l1 78.67851 reg_l2 24.349813
loss 5.45758
STEP 32 ================================
prereg loss 0.7407308 reg_l1 78.6332 reg_l2 24.327904
loss 5.4587226
STEP 33 ================================
prereg loss 0.7383707 reg_l1 78.58564 reg_l2 24.306147
loss 5.4535093
STEP 34 ================================
prereg loss 0.71885693 reg_l1 78.5354 reg_l2 24.28449
loss 5.4309807
STEP 35 ================================
prereg loss 0.69848424 reg_l1 78.48176 reg_l2 24.262424
loss 5.4073896
STEP 36 ================================
prereg loss 0.6875601 reg_l1 78.4235 reg_l2 24.239164
loss 5.39297
STEP 37 ================================
prereg loss 0.67746073 reg_l1 78.36004 reg_l2 24.214273
loss 5.3790627
STEP 38 ================================
prereg loss 0.65856504 reg_l1 78.29185 reg_l2 24.187693
loss 5.356076
STEP 39 ================================
prereg loss 0.63885295 reg_l1 78.21993 reg_l2 24.159836
loss 5.332049
STEP 40 ================================
prereg loss 0.6295184 reg_l1 78.146675 reg_l2 24.131521
loss 5.318319
STEP 41 ================================
prereg loss 0.62745553 reg_l1 78.07472 reg_l2 24.103554
loss 5.311939
STEP 42 ================================
prereg loss 0.6229274 reg_l1 78.00587 reg_l2 24.076483
loss 5.303279
STEP 43 ================================
prereg loss 0.6168431 reg_l1 77.942215 reg_l2 24.050474
loss 5.293376
STEP 44 ================================
prereg loss 0.6154927 reg_l1 77.88083 reg_l2 24.025156
loss 5.2883425
STEP 45 ================================
prereg loss 0.6158122 reg_l1 77.820816 reg_l2 24.000124
loss 5.2850614
STEP 46 ================================
prereg loss 0.60965717 reg_l1 77.76053 reg_l2 23.97495
loss 5.275289
STEP 47 ================================
prereg loss 0.59841037 reg_l1 77.700035 reg_l2 23.949564
loss 5.260412
STEP 48 ================================
prereg loss 0.59007657 reg_l1 77.63956 reg_l2 23.924187
loss 5.24845
STEP 49 ================================
prereg loss 0.5850571 reg_l1 77.579636 reg_l2 23.899115
loss 5.2398353
STEP 50 ================================
prereg loss 0.57730144 reg_l1 77.52038 reg_l2 23.874533
loss 5.228524
STEP 51 ================================
prereg loss 0.5669522 reg_l1 77.461426 reg_l2 23.850351
loss 5.2146378
STEP 52 ================================
prereg loss 0.56052893 reg_l1 77.40198 reg_l2 23.82623
loss 5.2046475
STEP 53 ================================
prereg loss 0.5589222 reg_l1 77.34091 reg_l2 23.801727
loss 5.199377
STEP 54 ================================
prereg loss 0.557023 reg_l1 77.2787 reg_l2 23.776573
loss 5.193745
STEP 55 ================================
prereg loss 0.5533141 reg_l1 77.216125 reg_l2 23.750732
loss 5.1862817
STEP 56 ================================
prereg loss 0.5502608 reg_l1 77.151924 reg_l2 23.724419
loss 5.1793766
STEP 57 ================================
prereg loss 0.5481135 reg_l1 77.08674 reg_l2 23.697956
loss 5.1733174
STEP 58 ================================
prereg loss 0.5441059 reg_l1 77.020676 reg_l2 23.671463
loss 5.1653466
STEP 59 ================================
prereg loss 0.5383198 reg_l1 76.9546 reg_l2 23.644836
loss 5.155596
STEP 60 ================================
prereg loss 0.53303146 reg_l1 76.88871 reg_l2 23.617851
loss 5.146354
STEP 61 ================================
prereg loss 0.5285362 reg_l1 76.82232 reg_l2 23.590271
loss 5.1378756
STEP 62 ================================
prereg loss 0.5237547 reg_l1 76.75457 reg_l2 23.56204
loss 5.129029
STEP 63 ================================
prereg loss 0.5191428 reg_l1 76.68503 reg_l2 23.533285
loss 5.120244
STEP 64 ================================
prereg loss 0.515784 reg_l1 76.613884 reg_l2 23.504368
loss 5.1126165
STEP 65 ================================
prereg loss 0.5129551 reg_l1 76.54348 reg_l2 23.475582
loss 5.105564
STEP 66 ================================
prereg loss 0.50956917 reg_l1 76.47434 reg_l2 23.447136
loss 5.0980296
STEP 67 ================================
prereg loss 0.5061516 reg_l1 76.405396 reg_l2 23.418976
loss 5.090475
STEP 68 ================================
prereg loss 0.50331616 reg_l1 76.335724 reg_l2 23.390896
loss 5.08346
STEP 69 ================================
prereg loss 0.5004027 reg_l1 76.26574 reg_l2 23.362682
loss 5.0763474
STEP 70 ================================
prereg loss 0.49714962 reg_l1 76.194916 reg_l2 23.334227
loss 5.0688443
STEP 71 ================================
prereg loss 0.49434116 reg_l1 76.12319 reg_l2 23.305658
loss 5.061733
STEP 72 ================================
prereg loss 0.49209234 reg_l1 76.050934 reg_l2 23.277105
loss 5.055148
STEP 73 ================================
prereg loss 0.4897718 reg_l1 75.978035 reg_l2 23.248716
loss 5.048454
STEP 74 ================================
prereg loss 0.48755944 reg_l1 75.90537 reg_l2 23.220419
loss 5.0418816
STEP 75 ================================
prereg loss 0.4860375 reg_l1 75.83436 reg_l2 23.192118
loss 5.0360985
STEP 76 ================================
prereg loss 0.48480386 reg_l1 75.7627 reg_l2 23.16361
loss 5.0305657
STEP 77 ================================
prereg loss 0.4831239 reg_l1 75.69016 reg_l2 23.134787
loss 5.0245333
STEP 78 ================================
prereg loss 0.4813465 reg_l1 75.61678 reg_l2 23.105682
loss 5.0183535
STEP 79 ================================
prereg loss 0.47998792 reg_l1 75.54177 reg_l2 23.076448
loss 5.012494
STEP 80 ================================
prereg loss 0.47866356 reg_l1 75.46657 reg_l2 23.047192
loss 5.0066576
STEP 81 ================================
prereg loss 0.4771222 reg_l1 75.39321 reg_l2 23.01799
loss 5.000715
STEP 82 ================================
prereg loss 0.47577897 reg_l1 75.319115 reg_l2 22.988724
loss 4.994926
STEP 83 ================================
prereg loss 0.4747176 reg_l1 75.24617 reg_l2 22.959293
loss 4.9894876
STEP 84 ================================
prereg loss 0.47363272 reg_l1 75.17316 reg_l2 22.92967
loss 4.984022
STEP 85 ================================
prereg loss 0.4725932 reg_l1 75.09843 reg_l2 22.900013
loss 4.978499
STEP 86 ================================
prereg loss 0.4715719 reg_l1 75.02221 reg_l2 22.87039
loss 4.972904
STEP 87 ================================
prereg loss 0.470484 reg_l1 74.94619 reg_l2 22.840925
loss 4.967255
STEP 88 ================================
prereg loss 0.46933526 reg_l1 74.87022 reg_l2 22.811617
loss 4.961548
STEP 89 ================================
prereg loss 0.46831068 reg_l1 74.794304 reg_l2 22.782349
loss 4.955969
STEP 90 ================================
prereg loss 0.46726912 reg_l1 74.718994 reg_l2 22.753115
loss 4.9504085
STEP 91 ================================
prereg loss 0.46629098 reg_l1 74.643 reg_l2 22.723948
loss 4.944871
STEP 92 ================================
prereg loss 0.4653825 reg_l1 74.566864 reg_l2 22.694908
loss 4.9393945
STEP 93 ================================
prereg loss 0.4644387 reg_l1 74.49127 reg_l2 22.666119
loss 4.933915
STEP 94 ================================
prereg loss 0.4635266 reg_l1 74.41578 reg_l2 22.637438
loss 4.9284735
STEP 95 ================================
prereg loss 0.46259096 reg_l1 74.33962 reg_l2 22.60885
loss 4.9229684
STEP 96 ================================
prereg loss 0.46161664 reg_l1 74.26434 reg_l2 22.580343
loss 4.917477
STEP 97 ================================
prereg loss 0.46063876 reg_l1 74.19086 reg_l2 22.551912
loss 4.9120903
STEP 98 ================================
prereg loss 0.45961607 reg_l1 74.11749 reg_l2 22.523575
loss 4.906666
STEP 99 ================================
prereg loss 0.4585724 reg_l1 74.04611 reg_l2 22.495312
loss 4.901339
STEP 100 ================================
prereg loss 0.45754102 reg_l1 73.97407 reg_l2 22.467043
loss 4.895985
2022-06-26T18:13:31.517

julia> serialize("sparse6-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse6-after-100-steps-opt.ser", opt)

julia> count_interval(sparse6, -0.001f0, 0.001f0)
16

julia> count_interval(sparse6, -0.01f0, 0.01f0)
24

julia> count_interval(sparse6, -0.02f0, 0.02f0)
35

julia> count_interval(sparse6, -0.03f0, 0.03f0)
41

julia> count_interval(sparse6, -0.04f0, 0.04f0)
65

julia> # ok, that's normal, let's continue, the next cutoff is 0.04

julia> sparse7 = sparsecopy(trainable["network_matrix"], 0.04f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.630873), "eos"=>Dict("char"=>-0.0484975)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.506873), "dot-2-2"=>Dict("dot"=>-0.125633), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0436771), "accum-1-2"=>Dict("dict"=>0.279308), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.557893), "accum-1-1"=>Dict("dict"=>0.186819), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.214383), "accum-1-1"=>Dict("dict"=>0.138538), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.107364)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>0.0478848), "norm-2-1"=>Dict("norm"=>0.112275), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0804973), "accum-1-2"=>Dict("dict"=>0.224105), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.2943), "norm-2-1"=>Dict("norm"=>0.241076), "dot-2-2"=>D…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.385409)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.10773)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.161007), "eos"=>Dict("char"=>-0.168951)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0942965), "accum-3-1"=>Dict("dict"=>0.061985), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.280929), "dot-2-2"=>Dict("dot"=>-0.144584), "norm-3-1"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.156333), "norm-2-1"=>Dict("norm"=>0.0528005), "dot-3-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264977), "dot-2-2"=>Dict("dot"=>-0.0796453), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.830051), "eos"=>Dict("char"=>-0.168948)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.477869), "dot-1-2"=>Dict("dot"=>-0.164416), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.371938), "const_1"=>Dict("const_1"=>-0.074429), "accum-…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.22576), "dot-2-2"=>Dict("dot"=>-0.0655617), "norm-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0481084), "norm-2-1"=>Dict("norm"=>-0.307835), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.260039), "norm-2-1"=>Dict("norm"=>-0.0771509), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.224), "accum-1-1"=>Dict("dict"=>0.129106), "accum-1-2"…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.161322), "accum-3-1"=>Dict("dict"=>-0.0555534), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.053492), "dot-2-1"=>Dict("dot"=>0.111347)))
  ⋮             => ⋮

julia> count(sparse7)
422

julia> trainable["network_matrix"] = sparse7
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.630873), "eos"=>Dict("char"=>-0.0484975)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.506873), "dot-2-2"=>Dict("dot"=>-0.125633), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0436771), "accum-1-2"=>Dict("dict"=>0.279308), "input…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.557893), "accum-1-1"=>Dict("dict"=>0.186819), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.214383), "accum-1-1"=>Dict("dict"=>0.138538), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.107364)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>0.0478848), "norm-2-1"=>Dict("norm"=>0.112275), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0804973), "accum-1-2"=>Dict("dict"=>0.224105), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.2943), "norm-2-1"=>Dict("norm"=>0.241076), "dot-2-2"=>D…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.385409)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.10773)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.161007), "eos"=>Dict("char"=>-0.168951)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0942965), "accum-3-1"=>Dict("dict"=>0.061985), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.280929), "dot-2-2"=>Dict("dot"=>-0.144584), "norm-3-1"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.156333), "norm-2-1"=>Dict("norm"=>0.0528005), "dot-3-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.264977), "dot-2-2"=>Dict("dot"=>-0.0796453), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.830051), "eos"=>Dict("char"=>-0.168948)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.477869), "dot-1-2"=>Dict("dot"=>-0.164416), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.371938), "const_1"=>Dict("const_1"=>-0.074429), "accum-…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.22576), "dot-2-2"=>Dict("dot"=>-0.0655617), "norm-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0481084), "norm-2-1"=>Dict("norm"=>-0.307835), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.260039), "norm-2-1"=>Dict("norm"=>-0.0771509), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.224), "accum-1-1"=>Dict("dict"=>0.129106), "accum-1-2"…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.161322), "accum-3-1"=>Dict("dict"=>-0.0555534), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.053492), "dot-2-1"=>Dict("dot"=>0.111347)))
  ⋮             => ⋮

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T18:21:14.990
STEP 1 ================================
prereg loss 67.65978 reg_l1 72.69135 reg_l2 22.400953
loss 72.02126
STEP 2 ================================
prereg loss 54.2789 reg_l1 72.61332 reg_l2 22.359667
loss 58.6357
STEP 3 ================================
prereg loss 44.540142 reg_l1 72.53133 reg_l2 22.320301
loss 48.89202
STEP 4 ================================
prereg loss 38.083607 reg_l1 72.458725 reg_l2 22.288685
loss 42.43113
STEP 5 ================================
prereg loss 33.83488 reg_l1 72.40522 reg_l2 22.268932
loss 38.179195
STEP 6 ================================
prereg loss 31.084995 reg_l1 72.37241 reg_l2 22.26089
loss 35.42734
STEP 7 ================================
prereg loss 28.652285 reg_l1 72.35734 reg_l2 22.262548
loss 32.993725
STEP 8 ================================
prereg loss 26.133337 reg_l1 72.35411 reg_l2 22.271214
loss 30.474583
STEP 9 ================================
prereg loss 23.449688 reg_l1 72.35953 reg_l2 22.28537
loss 27.79126
STEP 10 ================================
prereg loss 20.60263 reg_l1 72.37102 reg_l2 22.303415
loss 24.944891
STEP 11 ================================
prereg loss 17.856985 reg_l1 72.3866 reg_l2 22.324242
loss 22.20018
STEP 12 ================================
prereg loss 15.553051 reg_l1 72.40344 reg_l2 22.346197
loss 19.897257
STEP 13 ================================
prereg loss 13.691306 reg_l1 72.41967 reg_l2 22.368074
loss 18.036486
STEP 14 ================================
prereg loss 12.205752 reg_l1 72.433876 reg_l2 22.38895
loss 16.551785
STEP 15 ================================
prereg loss 11.051726 reg_l1 72.44507 reg_l2 22.408106
loss 15.39843
STEP 16 ================================
prereg loss 10.1870985 reg_l1 72.45207 reg_l2 22.424822
loss 14.534223
STEP 17 ================================
prereg loss 9.466361 reg_l1 72.4544 reg_l2 22.438583
loss 13.813625
STEP 18 ================================
prereg loss 8.755314 reg_l1 72.45179 reg_l2 22.449167
loss 13.102421
STEP 19 ================================
prereg loss 7.9924483 reg_l1 72.44434 reg_l2 22.456614
loss 12.339109
STEP 20 ================================
prereg loss 7.2059646 reg_l1 72.43265 reg_l2 22.46122
loss 11.551924
STEP 21 ================================
prereg loss 6.466994 reg_l1 72.41736 reg_l2 22.46344
loss 10.812035
STEP 22 ================================
prereg loss 5.843473 reg_l1 72.39918 reg_l2 22.463766
loss 10.187424
STEP 23 ================================
prereg loss 5.3578253 reg_l1 72.37889 reg_l2 22.462757
loss 9.700559
STEP 24 ================================
prereg loss 4.9720545 reg_l1 72.357086 reg_l2 22.460875
loss 9.313479
STEP 25 ================================
prereg loss 4.6087317 reg_l1 72.334366 reg_l2 22.458494
loss 8.948793
STEP 26 ================================
prereg loss 4.1892853 reg_l1 72.31116 reg_l2 22.455952
loss 8.527954
STEP 27 ================================
prereg loss 3.6959555 reg_l1 72.28798 reg_l2 22.453352
loss 8.033235
STEP 28 ================================
prereg loss 3.1556594 reg_l1 72.26496 reg_l2 22.450714
loss 7.491557
STEP 29 ================================
prereg loss 2.6236522 reg_l1 72.24206 reg_l2 22.447939
loss 6.9581757
STEP 30 ================================
prereg loss 2.1702032 reg_l1 72.21903 reg_l2 22.444765
loss 6.503345
STEP 31 ================================
prereg loss 1.8404506 reg_l1 72.19553 reg_l2 22.440903
loss 6.1721826
STEP 32 ================================
prereg loss 1.6260879 reg_l1 72.17126 reg_l2 22.436049
loss 5.9563637
STEP 33 ================================
prereg loss 1.4909247 reg_l1 72.145874 reg_l2 22.430016
loss 5.8196774
STEP 34 ================================
prereg loss 1.381646 reg_l1 72.11922 reg_l2 22.422714
loss 5.708799
STEP 35 ================================
prereg loss 1.2782644 reg_l1 72.091354 reg_l2 22.414265
loss 5.6037455
STEP 36 ================================
prereg loss 1.1939335 reg_l1 72.06249 reg_l2 22.404902
loss 5.517683
STEP 37 ================================
prereg loss 1.157011 reg_l1 72.03312 reg_l2 22.394987
loss 5.478998
STEP 38 ================================
prereg loss 1.1746395 reg_l1 72.003525 reg_l2 22.384901
loss 5.494851
STEP 39 ================================
prereg loss 1.2307549 reg_l1 71.97426 reg_l2 22.375025
loss 5.54921
STEP 40 ================================
prereg loss 1.2927191 reg_l1 71.94559 reg_l2 22.365637
loss 5.609454
STEP 41 ================================
prereg loss 1.3281753 reg_l1 71.91772 reg_l2 22.35688
loss 5.643238
STEP 42 ================================
prereg loss 1.3176429 reg_l1 71.89069 reg_l2 22.34881
loss 5.6310844
STEP 43 ================================
prereg loss 1.2636424 reg_l1 71.8644 reg_l2 22.341297
loss 5.575506
STEP 44 ================================
prereg loss 1.1861728 reg_l1 71.83856 reg_l2 22.334146
loss 5.4964867
STEP 45 ================================
prereg loss 1.1096911 reg_l1 71.81283 reg_l2 22.327017
loss 5.418461
STEP 46 ================================
prereg loss 1.0501759 reg_l1 71.786804 reg_l2 22.319597
loss 5.3573837
STEP 47 ================================
prereg loss 1.0117648 reg_l1 71.76006 reg_l2 22.311558
loss 5.3173685
STEP 48 ================================
prereg loss 0.98723376 reg_l1 71.73235 reg_l2 22.302612
loss 5.291175
STEP 49 ================================
prereg loss 0.9665453 reg_l1 71.70346 reg_l2 22.292616
loss 5.2687526
STEP 50 ================================
prereg loss 0.9446349 reg_l1 71.673386 reg_l2 22.281519
loss 5.245038
STEP 51 ================================
prereg loss 0.9238432 reg_l1 71.6423 reg_l2 22.269358
loss 5.2223816
STEP 52 ================================
prereg loss 0.91010195 reg_l1 71.61029 reg_l2 22.2563
loss 5.2067194
STEP 53 ================================
prereg loss 0.9058408 reg_l1 71.57776 reg_l2 22.242525
loss 5.200506
STEP 54 ================================
prereg loss 0.90669507 reg_l1 71.544975 reg_l2 22.228277
loss 5.1993933
STEP 55 ================================
prereg loss 0.9045441 reg_l1 71.51217 reg_l2 22.213749
loss 5.1952744
STEP 56 ================================
prereg loss 0.8915681 reg_l1 71.47953 reg_l2 22.199091
loss 5.18034
STEP 57 ================================
prereg loss 0.865616 reg_l1 71.44764 reg_l2 22.184366
loss 5.152474
STEP 58 ================================
prereg loss 0.8311404 reg_l1 71.41718 reg_l2 22.169632
loss 5.1161714
STEP 59 ================================
prereg loss 0.79615444 reg_l1 71.38665 reg_l2 22.15484
loss 5.0793533
STEP 60 ================================
prereg loss 0.7673719 reg_l1 71.355995 reg_l2 22.139914
loss 5.048732
STEP 61 ================================
prereg loss 0.74718684 reg_l1 71.32499 reg_l2 22.124823
loss 5.0266857
STEP 62 ================================
prereg loss 0.73381484 reg_l1 71.29369 reg_l2 22.109509
loss 5.0114365
STEP 63 ================================
prereg loss 0.7240163 reg_l1 71.26282 reg_l2 22.093988
loss 4.999785
STEP 64 ================================
prereg loss 0.7155472 reg_l1 71.23265 reg_l2 22.078331
loss 4.9895062
STEP 65 ================================
prereg loss 0.70814526 reg_l1 71.202156 reg_l2 22.062634
loss 4.980274
STEP 66 ================================
prereg loss 0.7028839 reg_l1 71.17156 reg_l2 22.04705
loss 4.9731774
STEP 67 ================================
prereg loss 0.7001827 reg_l1 71.14104 reg_l2 22.031734
loss 4.968645
STEP 68 ================================
prereg loss 0.69889754 reg_l1 71.110725 reg_l2 22.016808
loss 4.965541
STEP 69 ================================
prereg loss 0.69689465 reg_l1 71.080826 reg_l2 22.002386
loss 4.9617443
STEP 70 ================================
prereg loss 0.69234675 reg_l1 71.05143 reg_l2 21.988504
loss 4.9554324
STEP 71 ================================
prereg loss 0.6849074 reg_l1 71.02251 reg_l2 21.975147
loss 4.9462576
STEP 72 ================================
prereg loss 0.67567027 reg_l1 70.99488 reg_l2 21.962275
loss 4.935363
STEP 73 ================================
prereg loss 0.666488 reg_l1 70.96771 reg_l2 21.949764
loss 4.924551
STEP 74 ================================
prereg loss 0.6588234 reg_l1 70.94176 reg_l2 21.937513
loss 4.915329
STEP 75 ================================
prereg loss 0.6530365 reg_l1 70.916435 reg_l2 21.925375
loss 4.9080224
STEP 76 ================================
prereg loss 0.648507 reg_l1 70.89105 reg_l2 21.913244
loss 4.9019704
STEP 77 ================================
prereg loss 0.6443506 reg_l1 70.86671 reg_l2 21.901043
loss 4.896353
STEP 78 ================================
prereg loss 0.64010906 reg_l1 70.84236 reg_l2 21.888737
loss 4.8906507
STEP 79 ================================
prereg loss 0.6359666 reg_l1 70.81783 reg_l2 21.876308
loss 4.8850365
STEP 80 ================================
prereg loss 0.63225514 reg_l1 70.79258 reg_l2 21.863775
loss 4.87981
STEP 81 ================================
prereg loss 0.62906563 reg_l1 70.766815 reg_l2 21.851206
loss 4.8750744
STEP 82 ================================
prereg loss 0.62598515 reg_l1 70.74091 reg_l2 21.838612
loss 4.87044
STEP 83 ================================
prereg loss 0.6225122 reg_l1 70.714836 reg_l2 21.82604
loss 4.865402
STEP 84 ================================
prereg loss 0.6183551 reg_l1 70.68814 reg_l2 21.813465
loss 4.8596435
STEP 85 ================================
prereg loss 0.6137046 reg_l1 70.660934 reg_l2 21.800922
loss 4.8533607
STEP 86 ================================
prereg loss 0.6091042 reg_l1 70.63318 reg_l2 21.788359
loss 4.847095
STEP 87 ================================
prereg loss 0.60501957 reg_l1 70.60484 reg_l2 21.775734
loss 4.84131
STEP 88 ================================
prereg loss 0.6015614 reg_l1 70.576584 reg_l2 21.763042
loss 4.8361564
STEP 89 ================================
prereg loss 0.5985586 reg_l1 70.54815 reg_l2 21.750221
loss 4.831447
STEP 90 ================================
prereg loss 0.59567845 reg_l1 70.52027 reg_l2 21.737278
loss 4.8268943
STEP 91 ================================
prereg loss 0.5927366 reg_l1 70.49265 reg_l2 21.724243
loss 4.8222957
STEP 92 ================================
prereg loss 0.5897414 reg_l1 70.464294 reg_l2 21.711111
loss 4.817599
STEP 93 ================================
prereg loss 0.5867969 reg_l1 70.43521 reg_l2 21.697927
loss 4.812909
STEP 94 ================================
prereg loss 0.5839457 reg_l1 70.40555 reg_l2 21.684748
loss 4.8082786
STEP 95 ================================
prereg loss 0.5811079 reg_l1 70.37564 reg_l2 21.671598
loss 4.8036466
STEP 96 ================================
prereg loss 0.5781527 reg_l1 70.345764 reg_l2 21.658525
loss 4.798898
STEP 97 ================================
prereg loss 0.57504296 reg_l1 70.315605 reg_l2 21.645506
loss 4.7939796
STEP 98 ================================
prereg loss 0.5718617 reg_l1 70.28551 reg_l2 21.632526
loss 4.788992
STEP 99 ================================
prereg loss 0.5687524 reg_l1 70.25551 reg_l2 21.619596
loss 4.784083
STEP 100 ================================
prereg loss 0.5658537 reg_l1 70.22502 reg_l2 21.606638
loss 4.7793546
2022-06-26T18:42:26.624

julia> serialize("sparse7-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse7-after-100-steps-opt.ser", opt)

julia> count_interval(sparse7, -0.001f0, 0.001f0)
4

julia> count_interval(sparse7, -0.01f0, 0.01f0)
7

julia> count_interval(sparse7, -0.02f0, 0.02f0)
9

julia> count_interval(sparse7, -0.03f0, 0.03f0)
14

julia> count_interval(sparse7, -0.04f0, 0.04f0)
29

julia> count_interval(sparse7, -0.05f0, 0.05f0)
55

julia> # 0.05 will be the next cutoff, but let's adjust the regularization up

julia> close(io)

julia> count(sparse7)
422
```

adjusting regularization:

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
< trainable["network_matrix"] = deserialize("sparse7-after-100-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

5 more sparsification runs at this regularization (details are quite interesting):

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.628245), "eos"=>Dict("char"=>-0.0480654)), "dict-1"=>Di…
  "norm-5-2"    => Dict("dict"=>Dict("dot-4-2"=>Dict("dot"=>-0.0843514), "accum-3-2"=>Dict("dict"=>0.0716006), "norm-2-…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-1"=>Dict("dict"=>-0.0363), "accum-1-2"=>Dict("dict"=>0.286462), "input"=>…
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.547688), "accum-1-1"=>Dict("dict"=>0.188143), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222774), "accum-1-1"=>Dict("dict"=>0.146722), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0904367)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>0.0461049), "norm-2-1"=>Dict("norm"=>0.106316), "dot-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0876405), "accum-1-2"=>Dict("dict"=>0.231306), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.315995), "dot-4-2"=>Dict("dot"=>-0.170605), "norm-2-1"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.359319)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.112731)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.148578), "eos"=>Dict("char"=>-0.173666)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113096), "accum-3-1"=>Dict("dict"=>0.0433832), "accum-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.247219), "eos"=>Dict("char"=>0.0998818), "dot-2-2"=>Dic…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.145839), "norm-2-1"=>Dict("norm"=>0.0538744), "dot-5-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.275334), "dot-2-2"=>Dict("dot"=>-0.0633489), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.834701), "eos"=>Dict("char"=>-0.170935)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.47315), "dot-1-2"=>Dict("dot"=>-0.164298), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.356833), "const_1"=>Dict("const_1"=>-0.0715583), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.237081), "dot-2-2"=>Dict("dot"=>-0.0489186), "norm-3-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0405154), "norm-2-1"=>Dict("norm"=>-0.313679), "dot-2-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.229743), "dot-4-2"=>Dict("dot"=>0.100364), "norm-2-1"=…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.233006), "accum-1-1"=>Dict("dict"=>0.137935), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.102045), "accum-3-1"=>Dict("dict"=>-0.0500565), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.0502465), "dot-2-1"=>Dict("dot"=>0.115528)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
422

julia> sparse7 = sparsecopy(trainable["network_matrix"], 0.05f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.628245)), "dict-1"=>Dict("input"=>Dict("char"=>0.380886…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.50126), "dot-2-2"=>Dict("dot"=>-0.119233), "norm-3-1"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.286462), "input"=>Dict("char"=>-0.338188)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.547688), "accum-1-1"=>Dict("dict"=>0.188143), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222774), "accum-1-1"=>Dict("dict"=>0.146722), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0904367)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106316), "dot-2-2"=>Dict("dot"=>0.245332), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0876405), "accum-1-2"=>Dict("dict"=>0.231306), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.315995), "norm-2-1"=>Dict("norm"=>0.207903), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.359319)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.112731)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.148578), "eos"=>Dict("char"=>-0.173666)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113096), "accum-1-1"=>Dict("dict"=>0.104904)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.247219), "dot-2-2"=>Dict("dot"=>-0.129768), "norm-3-1"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.145839), "norm-2-1"=>Dict("norm"=>0.0538744), "dot-5-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.275334), "dot-2-2"=>Dict("dot"=>-0.0633489), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.834701), "eos"=>Dict("char"=>-0.170935)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.47315), "dot-1-2"=>Dict("dot"=>-0.164298), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.356833), "const_1"=>Dict("const_1"=>-0.0715583), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.237081), "norm-3-1"=>Dict("norm"=>0.188398), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0947535), "norm-2-1"=>Dict("norm"=>-0.313679), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.229743), "norm-2-1"=>Dict("norm"=>-0.0732798), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.233006), "accum-1-1"=>Dict("dict"=>0.137935), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.102045), "accum-3-1"=>Dict("dict"=>-0.0500565), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.0502465), "dot-2-1"=>Dict("dot"=>0.115528)))
  ⋮             => ⋮

julia> sparse8 = sparsecopy(trainable["network_matrix"], 0.05f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.628245)), "dict-1"=>Dict("input"=>Dict("char"=>0.380886…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.50126), "dot-2-2"=>Dict("dot"=>-0.119233), "norm-3-1"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.286462), "input"=>Dict("char"=>-0.338188)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.547688), "accum-1-1"=>Dict("dict"=>0.188143), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222774), "accum-1-1"=>Dict("dict"=>0.146722), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0904367)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106316), "dot-2-2"=>Dict("dot"=>0.245332), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0876405), "accum-1-2"=>Dict("dict"=>0.231306), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.315995), "norm-2-1"=>Dict("norm"=>0.207903), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.359319)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.112731)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.148578), "eos"=>Dict("char"=>-0.173666)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113096), "accum-1-1"=>Dict("dict"=>0.104904)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.247219), "dot-2-2"=>Dict("dot"=>-0.129768), "norm-3-1"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.145839), "norm-2-1"=>Dict("norm"=>0.0538744), "dot-5-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.275334), "dot-2-2"=>Dict("dot"=>-0.0633489), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.834701), "eos"=>Dict("char"=>-0.170935)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.47315), "dot-1-2"=>Dict("dot"=>-0.164298), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.356833), "const_1"=>Dict("const_1"=>-0.0715583), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.237081), "norm-3-1"=>Dict("norm"=>0.188398), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0947535), "norm-2-1"=>Dict("norm"=>-0.313679), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.229743), "norm-2-1"=>Dict("norm"=>-0.0732798), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.233006), "accum-1-1"=>Dict("dict"=>0.137935), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.102045), "accum-3-1"=>Dict("dict"=>-0.0500565), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.0502465), "dot-2-1"=>Dict("dot"=>0.115528)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse8
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 35 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.628245)), "dict-1"=>Dict("input"=>Dict("char"=>0.380886…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.50126), "dot-2-2"=>Dict("dot"=>-0.119233), "norm-3-1"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.286462), "input"=>Dict("char"=>-0.338188)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.547688), "accum-1-1"=>Dict("dict"=>0.188143), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222774), "accum-1-1"=>Dict("dict"=>0.146722), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0904367)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106316), "dot-2-2"=>Dict("dot"=>0.245332), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0876405), "accum-1-2"=>Dict("dict"=>0.231306), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.315995), "norm-2-1"=>Dict("norm"=>0.207903), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.359319)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.112731)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.148578), "eos"=>Dict("char"=>-0.173666)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.113096), "accum-1-1"=>Dict("dict"=>0.104904)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.247219), "dot-2-2"=>Dict("dot"=>-0.129768), "norm-3-1"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.145839), "norm-2-1"=>Dict("norm"=>0.0538744), "dot-5-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.275334), "dot-2-2"=>Dict("dot"=>-0.0633489), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.834701), "eos"=>Dict("char"=>-0.170935)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.47315), "dot-1-2"=>Dict("dot"=>-0.164298), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.356833), "const_1"=>Dict("const_1"=>-0.0715583), "accum…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.237081), "norm-3-1"=>Dict("norm"=>0.188398), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0947535), "norm-2-1"=>Dict("norm"=>-0.313679), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.229743), "norm-2-1"=>Dict("norm"=>-0.0732798), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.233006), "accum-1-1"=>Dict("dict"=>0.137935), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.102045), "accum-3-1"=>Dict("dict"=>-0.0500565), "dot-3…
  "accum-3-2"   => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.0502465), "dot-2-1"=>Dict("dot"=>0.115528)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
367

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T19:02:48.642
STEP 1 ================================
prereg loss 25.747795 reg_l1 68.31843 reg_l2 21.518398
loss 32.57964
STEP 2 ================================
prereg loss 18.425486 reg_l1 68.315414 reg_l2 21.524221
loss 25.257027
STEP 3 ================================
prereg loss 12.471906 reg_l1 68.32956 reg_l2 21.535067
loss 19.304861
STEP 4 ================================
prereg loss 8.476264 reg_l1 68.3382 reg_l2 21.540073
loss 15.310084
STEP 5 ================================
prereg loss 6.701505 reg_l1 68.33425 reg_l2 21.535711
loss 13.53493
STEP 6 ================================
prereg loss 6.3738117 reg_l1 68.315994 reg_l2 21.521296
loss 13.205412
STEP 7 ================================
prereg loss 6.264342 reg_l1 68.28466 reg_l2 21.498734
loss 13.092808
STEP 8 ================================
prereg loss 5.888938 reg_l1 68.24355 reg_l2 21.47072
loss 12.713293
STEP 9 ================================
prereg loss 5.2121 reg_l1 68.196754 reg_l2 21.440046
loss 12.031775
STEP 10 ================================
prereg loss 4.517016 reg_l1 68.14739 reg_l2 21.408758
loss 11.331755
STEP 11 ================================
prereg loss 4.080887 reg_l1 68.0993 reg_l2 21.37929
loss 10.890817
STEP 12 ================================
prereg loss 3.990513 reg_l1 68.05625 reg_l2 21.353458
loss 10.796139
STEP 13 ================================
prereg loss 4.060871 reg_l1 68.0216 reg_l2 21.3328
loss 10.863031
STEP 14 ================================
prereg loss 3.9604561 reg_l1 67.99639 reg_l2 21.317831
loss 10.760096
STEP 15 ================================
prereg loss 3.5510209 reg_l1 67.97947 reg_l2 21.307959
loss 10.348968
STEP 16 ================================
prereg loss 2.9036827 reg_l1 67.96886 reg_l2 21.302185
loss 9.700569
STEP 17 ================================
prereg loss 2.1961827 reg_l1 67.96251 reg_l2 21.299318
loss 8.992434
STEP 18 ================================
prereg loss 1.6311018 reg_l1 67.95819 reg_l2 21.29811
loss 8.426921
STEP 19 ================================
prereg loss 1.3467723 reg_l1 67.95341 reg_l2 21.297087
loss 8.142114
STEP 20 ================================
prereg loss 1.3234135 reg_l1 67.94568 reg_l2 21.294771
loss 8.117982
STEP 21 ================================
prereg loss 1.4477755 reg_l1 67.932785 reg_l2 21.289839
loss 8.241055
STEP 22 ================================
prereg loss 1.5828452 reg_l1 67.91316 reg_l2 21.281322
loss 8.374162
STEP 23 ================================
prereg loss 1.6329015 reg_l1 67.8865 reg_l2 21.268944
loss 8.421552
STEP 24 ================================
prereg loss 1.5889755 reg_l1 67.85311 reg_l2 21.25289
loss 8.374287
STEP 25 ================================
prereg loss 1.4990182 reg_l1 67.814125 reg_l2 21.233799
loss 8.280431
STEP 26 ================================
prereg loss 1.4161586 reg_l1 67.770836 reg_l2 21.212519
loss 8.193242
STEP 27 ================================
prereg loss 1.3825967 reg_l1 67.72469 reg_l2 21.18994
loss 8.1550665
STEP 28 ================================
prereg loss 1.3827897 reg_l1 67.67713 reg_l2 21.16698
loss 8.150503
STEP 29 ================================
prereg loss 1.3831754 reg_l1 67.62931 reg_l2 21.1444
loss 8.146107
STEP 30 ================================
prereg loss 1.3457131 reg_l1 67.5823 reg_l2 21.122746
loss 8.103943
STEP 31 ================================
prereg loss 1.2633345 reg_l1 67.536446 reg_l2 21.10227
loss 8.016979
STEP 32 ================================
prereg loss 1.1634738 reg_l1 67.4918 reg_l2 21.082947
loss 7.912654
STEP 33 ================================
prereg loss 1.0844619 reg_l1 67.448265 reg_l2 21.064608
loss 7.8292885
STEP 34 ================================
prereg loss 1.0590563 reg_l1 67.40523 reg_l2 21.046835
loss 7.799579
STEP 35 ================================
prereg loss 1.0885042 reg_l1 67.36203 reg_l2 21.029192
loss 7.8247075
STEP 36 ================================
prereg loss 1.144912 reg_l1 67.318115 reg_l2 21.011269
loss 7.8767233
STEP 37 ================================
prereg loss 1.1948382 reg_l1 67.27322 reg_l2 20.992855
loss 7.9221597
STEP 38 ================================
prereg loss 1.2163817 reg_l1 67.22745 reg_l2 20.973953
loss 7.9391265
STEP 39 ================================
prereg loss 1.2055904 reg_l1 67.18093 reg_l2 20.954693
loss 7.9236836
STEP 40 ================================
prereg loss 1.1746097 reg_l1 67.1342 reg_l2 20.935287
loss 7.88803
STEP 41 ================================
prereg loss 1.141614 reg_l1 67.08783 reg_l2 20.916142
loss 7.850397
STEP 42 ================================
prereg loss 1.1173486 reg_l1 67.04248 reg_l2 20.897621
loss 7.8215966
STEP 43 ================================
prereg loss 1.0997275 reg_l1 66.99874 reg_l2 20.880114
loss 7.799602
STEP 44 ================================
prereg loss 1.0814607 reg_l1 66.956764 reg_l2 20.863768
loss 7.777137
STEP 45 ================================
prereg loss 1.0565611 reg_l1 66.916756 reg_l2 20.848658
loss 7.7482367
STEP 46 ================================
prereg loss 1.0265242 reg_l1 66.87834 reg_l2 20.834576
loss 7.7143583
STEP 47 ================================
prereg loss 0.99699545 reg_l1 66.84114 reg_l2 20.821262
loss 7.6811094
STEP 48 ================================
prereg loss 0.97713304 reg_l1 66.80454 reg_l2 20.808344
loss 7.657587
STEP 49 ================================
prereg loss 0.9713818 reg_l1 66.76792 reg_l2 20.795368
loss 7.648174
STEP 50 ================================
prereg loss 0.9742114 reg_l1 66.730644 reg_l2 20.781986
loss 7.647276
STEP 51 ================================
prereg loss 0.98003316 reg_l1 66.6923 reg_l2 20.76788
loss 7.6492634
STEP 52 ================================
prereg loss 0.9809642 reg_l1 66.65265 reg_l2 20.75291
loss 7.6462293
STEP 53 ================================
prereg loss 0.97386134 reg_l1 66.61169 reg_l2 20.737019
loss 7.63503
STEP 54 ================================
prereg loss 0.9613399 reg_l1 66.56961 reg_l2 20.72039
loss 7.618301
STEP 55 ================================
prereg loss 0.94927526 reg_l1 66.52684 reg_l2 20.703276
loss 7.601959
STEP 56 ================================
prereg loss 0.9419655 reg_l1 66.483795 reg_l2 20.685987
loss 7.5903454
STEP 57 ================================
prereg loss 0.93976635 reg_l1 66.44101 reg_l2 20.668844
loss 7.5838675
STEP 58 ================================
prereg loss 0.939928 reg_l1 66.39883 reg_l2 20.652107
loss 7.5798106
STEP 59 ================================
prereg loss 0.9394243 reg_l1 66.35757 reg_l2 20.635931
loss 7.575181
STEP 60 ================================
prereg loss 0.9372037 reg_l1 66.31829 reg_l2 20.62036
loss 7.569033
STEP 61 ================================
prereg loss 0.93437165 reg_l1 66.280464 reg_l2 20.60533
loss 7.562418
STEP 62 ================================
prereg loss 0.9326316 reg_l1 66.24308 reg_l2 20.590687
loss 7.5569396
STEP 63 ================================
prereg loss 0.9324423 reg_l1 66.20769 reg_l2 20.576258
loss 7.553211
STEP 64 ================================
prereg loss 0.932622 reg_l1 66.17194 reg_l2 20.561863
loss 7.5498166
STEP 65 ================================
prereg loss 0.931043 reg_l1 66.135704 reg_l2 20.547321
loss 7.544614
STEP 66 ================================
prereg loss 0.92641157 reg_l1 66.0988 reg_l2 20.532568
loss 7.5362916
STEP 67 ================================
prereg loss 0.91953266 reg_l1 66.06123 reg_l2 20.51761
loss 7.525656
STEP 68 ================================
prereg loss 0.91245925 reg_l1 66.02314 reg_l2 20.502495
loss 7.5147734
STEP 69 ================================
prereg loss 0.90685207 reg_l1 65.984535 reg_l2 20.487286
loss 7.505306
STEP 70 ================================
prereg loss 0.90299433 reg_l1 65.94552 reg_l2 20.472086
loss 7.497546
STEP 71 ================================
prereg loss 0.90061283 reg_l1 65.90632 reg_l2 20.456944
loss 7.491245
STEP 72 ================================
prereg loss 0.898649 reg_l1 65.866875 reg_l2 20.441963
loss 7.4853363
STEP 73 ================================
prereg loss 0.896064 reg_l1 65.82792 reg_l2 20.427105
loss 7.4788556
STEP 74 ================================
prereg loss 0.8928676 reg_l1 65.78914 reg_l2 20.412338
loss 7.4717817
STEP 75 ================================
prereg loss 0.88964987 reg_l1 65.75011 reg_l2 20.397522
loss 7.4646606
STEP 76 ================================
prereg loss 0.8872597 reg_l1 65.712166 reg_l2 20.382553
loss 7.458476
STEP 77 ================================
prereg loss 0.88569045 reg_l1 65.673965 reg_l2 20.367304
loss 7.453087
STEP 78 ================================
prereg loss 0.8844795 reg_l1 65.63456 reg_l2 20.351744
loss 7.4479356
STEP 79 ================================
prereg loss 0.88326114 reg_l1 65.59406 reg_l2 20.335817
loss 7.4426675
STEP 80 ================================
prereg loss 0.88213694 reg_l1 65.55259 reg_l2 20.319605
loss 7.437396
STEP 81 ================================
prereg loss 0.881475 reg_l1 65.51026 reg_l2 20.303207
loss 7.4325013
STEP 82 ================================
prereg loss 0.8814839 reg_l1 65.46742 reg_l2 20.286753
loss 7.4282265
STEP 83 ================================
prereg loss 0.8818882 reg_l1 65.42442 reg_l2 20.270357
loss 7.4243307
STEP 84 ================================
prereg loss 0.882083 reg_l1 65.38182 reg_l2 20.254124
loss 7.420265
STEP 85 ================================
prereg loss 0.8815525 reg_l1 65.339836 reg_l2 20.238096
loss 7.4155364
STEP 86 ================================
prereg loss 0.8801777 reg_l1 65.2978 reg_l2 20.222288
loss 7.4099574
STEP 87 ================================
prereg loss 0.87822163 reg_l1 65.25631 reg_l2 20.206656
loss 7.4038525
STEP 88 ================================
prereg loss 0.875662 reg_l1 65.21555 reg_l2 20.191118
loss 7.3972173
STEP 89 ================================
prereg loss 0.8730007 reg_l1 65.17482 reg_l2 20.175585
loss 7.390483
STEP 90 ================================
prereg loss 0.87031907 reg_l1 65.13435 reg_l2 20.159998
loss 7.3837543
STEP 91 ================================
prereg loss 0.86761886 reg_l1 65.093124 reg_l2 20.144314
loss 7.3769317
STEP 92 ================================
prereg loss 0.86501163 reg_l1 65.051254 reg_l2 20.128536
loss 7.370137
STEP 93 ================================
prereg loss 0.86265874 reg_l1 65.009224 reg_l2 20.112709
loss 7.3635817
STEP 94 ================================
prereg loss 0.8606296 reg_l1 64.96695 reg_l2 20.096834
loss 7.3573246
STEP 95 ================================
prereg loss 0.8588184 reg_l1 64.924225 reg_l2 20.081003
loss 7.351241
STEP 96 ================================
prereg loss 0.8570642 reg_l1 64.88106 reg_l2 20.065243
loss 7.34517
STEP 97 ================================
prereg loss 0.85526735 reg_l1 64.8376 reg_l2 20.049538
loss 7.339028
STEP 98 ================================
prereg loss 0.8534481 reg_l1 64.7955 reg_l2 20.033909
loss 7.3329983
STEP 99 ================================
prereg loss 0.85170835 reg_l1 64.75458 reg_l2 20.01835
loss 7.327166
STEP 100 ================================
prereg loss 0.8501507 reg_l1 64.71361 reg_l2 20.002804
loss 7.3215113
2022-06-26T19:22:26.865

julia> serialize("sparse8-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse8-after-100-steps-opt.ser", opt)

julia> count_interval(sparse8, -0.001f0, 0.001f0)
4

julia> count_interval(sparse8, -0.01f0, 0.01f0)
7

julia> count_interval(sparse8, -0.02f0, 0.02f0)
9

julia> count_interval(sparse8, -0.03f0, 0.03f0)
13

julia> count_interval(sparse8, -0.04f0, 0.04f0)
17

julia> count_interval(sparse8, -0.05f0, 0.05f0)
27

julia> count_interval(sparse8, -0.06f0, 0.06f0)
42

julia> count_interval(sparse8, -0.07f0, 0.07f0)
66

julia> count_interval(sparse8, -0.065f0, 0.065f0)
53

julia> # let's have this one as a cutoff (I did not want to go below 48)

julia> sparse9 = sparsecopy(trainable["network_matrix"], 0.065f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 34 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.656124)), "dict-1"=>Dict("input"=>Dict("char"=>0.396953…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.500765), "dot-2-2"=>Dict("dot"=>-0.101307), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.283512), "input"=>Dict("char"=>-0.366482)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.526419), "accum-1-1"=>Dict("dict"=>0.186198), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222218), "accum-1-1"=>Dict("dict"=>0.145418), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0857186)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106884), "dot-2-2"=>Dict("dot"=>0.253008), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0866293), "accum-1-2"=>Dict("dict"=>0.230751), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.278277), "norm-2-1"=>Dict("norm"=>0.218924), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.320382)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.111978)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.165467), "eos"=>Dict("char"=>-0.171668)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.104228)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234833), "dot-2-2"=>Dict("dot"=>-0.14215), "norm-3-1"=>…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.134936), "dot-3-1"=>Dict("dot"=>0.101938), "compare-5…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271586), "dot-2-2"=>Dict("dot"=>-0.0716346), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.808927), "eos"=>Dict("char"=>-0.168379)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.503781), "dot-1-2"=>Dict("dot"=>-0.157311), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28316), "accum-1-2"=>Dict("dict"=>-0.263936), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.234012), "norm-3-1"=>Dict("norm"=>0.185978), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0873626), "norm-2-1"=>Dict("norm"=>-0.309181), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.179793), "norm-2-1"=>Dict("norm"=>-0.0729632), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.232275), "accum-1-1"=>Dict("dict"=>0.136528), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0982746), "compare-3-2"=>Dict("true"=>0.416766)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.103119)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse9
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 34 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.656124)), "dict-1"=>Dict("input"=>Dict("char"=>0.396953…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.500765), "dot-2-2"=>Dict("dot"=>-0.101307), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.283512), "input"=>Dict("char"=>-0.366482)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.526419), "accum-1-1"=>Dict("dict"=>0.186198), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222218), "accum-1-1"=>Dict("dict"=>0.145418), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0857186)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106884), "dot-2-2"=>Dict("dot"=>0.253008), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0866293), "accum-1-2"=>Dict("dict"=>0.230751), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.278277), "norm-2-1"=>Dict("norm"=>0.218924), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.320382)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.111978)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.165467), "eos"=>Dict("char"=>-0.171668)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.104228)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234833), "dot-2-2"=>Dict("dot"=>-0.14215), "norm-3-1"=>…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.134936), "dot-3-1"=>Dict("dot"=>0.101938), "compare-5…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271586), "dot-2-2"=>Dict("dot"=>-0.0716346), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.808927), "eos"=>Dict("char"=>-0.168379)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.503781), "dot-1-2"=>Dict("dot"=>-0.157311), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28316), "accum-1-2"=>Dict("dict"=>-0.263936), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.234012), "norm-3-1"=>Dict("norm"=>0.185978), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0873626), "norm-2-1"=>Dict("norm"=>-0.309181), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.179793), "norm-2-1"=>Dict("norm"=>-0.0729632), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.232275), "accum-1-1"=>Dict("dict"=>0.136528), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0982746), "compare-3-2"=>Dict("true"=>0.416766)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.103119)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
314

julia> steps!(100)
2022-06-26T19:26:00.227
STEP 1 ================================
prereg loss 1.435752 reg_l1 62.387386 reg_l2 19.866703
loss 7.6744905
STEP 2 ================================
prereg loss 1.3607731 reg_l1 62.38006 reg_l2 19.853968
loss 7.598779
STEP 3 ================================
prereg loss 1.2257751 reg_l1 62.372456 reg_l2 19.842672
loss 7.463021
STEP 4 ================================
prereg loss 1.0767872 reg_l1 62.364758 reg_l2 19.832617
loss 7.313263
STEP 5 ================================
prereg loss 0.95768124 reg_l1 62.35646 reg_l2 19.823479
loss 7.1933274
STEP 6 ================================
prereg loss 0.8938995 reg_l1 62.346893 reg_l2 19.814837
loss 7.1285887
STEP 7 ================================
prereg loss 0.88692456 reg_l1 62.334793 reg_l2 19.806177
loss 7.1204042
STEP 8 ================================
prereg loss 0.9134334 reg_l1 62.319183 reg_l2 19.797056
loss 7.145352
ERROR: InterruptException:
Stacktrace:
  [...] I INTERRUPTED THIS, BECAUSE I FORGOT TO RE-INITIALIZE THE OPTIMIZER


julia> sparse9 = sparsecopy(sparse8, 0.065f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 34 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.656124)), "dict-1"=>Dict("input"=>Dict("char"=>0.396953…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.500765), "dot-2-2"=>Dict("dot"=>-0.101307), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.283512), "input"=>Dict("char"=>-0.366482)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.526419), "accum-1-1"=>Dict("dict"=>0.186198), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222218), "accum-1-1"=>Dict("dict"=>0.145418), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0857186)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106884), "dot-2-2"=>Dict("dot"=>0.253008), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0866293), "accum-1-2"=>Dict("dict"=>0.230751), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.278277), "norm-2-1"=>Dict("norm"=>0.218924), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.320382)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.111978)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.165467), "eos"=>Dict("char"=>-0.171668)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.104228)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234833), "dot-2-2"=>Dict("dot"=>-0.14215), "norm-3-1"=>…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.134936), "dot-3-1"=>Dict("dot"=>0.101938), "compare-5…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271586), "dot-2-2"=>Dict("dot"=>-0.0716346), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.808927), "eos"=>Dict("char"=>-0.168379)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.503781), "dot-1-2"=>Dict("dot"=>-0.157311), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28316), "accum-1-2"=>Dict("dict"=>-0.263936), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.234012), "norm-3-1"=>Dict("norm"=>0.185978), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0873626), "norm-2-1"=>Dict("norm"=>-0.309181), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.179793), "norm-2-1"=>Dict("norm"=>-0.0729632), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.232275), "accum-1-1"=>Dict("dict"=>0.136528), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0982746), "compare-3-2"=>Dict("true"=>0.416766)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.103119)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse9
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 34 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.656124)), "dict-1"=>Dict("input"=>Dict("char"=>0.396953…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.500765), "dot-2-2"=>Dict("dot"=>-0.101307), "norm-3-1"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.283512), "input"=>Dict("char"=>-0.366482)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.526419), "accum-1-1"=>Dict("dict"=>0.186198), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.222218), "accum-1-1"=>Dict("dict"=>0.145418), "accum-1…
  "accum-3-1"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.0857186)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.106884), "dot-2-2"=>Dict("dot"=>0.253008), "norm-2-2"…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0866293), "accum-1-2"=>Dict("dict"=>0.230751), "input"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.278277), "norm-2-1"=>Dict("norm"=>0.218924), "dot-2-2"=…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.320382)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.111978)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.165467), "eos"=>Dict("char"=>-0.171668)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.104228)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234833), "dot-2-2"=>Dict("dot"=>-0.14215), "norm-3-1"=>…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.134936), "dot-3-1"=>Dict("dot"=>0.101938), "compare-5…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271586), "dot-2-2"=>Dict("dot"=>-0.0716346), "norm-3-…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.808927), "eos"=>Dict("char"=>-0.168379)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.503781), "dot-1-2"=>Dict("dot"=>-0.157311), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.28316), "accum-1-2"=>Dict("dict"=>-0.263936), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.234012), "norm-3-1"=>Dict("norm"=>0.185978), "norm-2-…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0873626), "norm-2-1"=>Dict("norm"=>-0.309181), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.179793), "norm-2-1"=>Dict("norm"=>-0.0729632), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.232275), "accum-1-1"=>Dict("dict"=>0.136528), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0982746), "compare-3-2"=>Dict("true"=>0.416766)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.103119)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
314

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T19:28:43.208
STEP 1 ================================
prereg loss 1.435752 reg_l1 62.387386 reg_l2 19.866703
loss 7.6744905
STEP 2 ================================
prereg loss 0.91664296 reg_l1 62.3814 reg_l2 19.868692
loss 7.1547832
STEP 3 ================================
prereg loss 1.2212191 reg_l1 62.340096 reg_l2 19.845331
loss 7.455229
STEP 4 ================================
prereg loss 1.0552373 reg_l1 62.31473 reg_l2 19.836298
loss 7.2867107
STEP 5 ================================
prereg loss 1.1208115 reg_l1 62.274273 reg_l2 19.819796
loss 7.348239
STEP 6 ================================
prereg loss 1.0376964 reg_l1 62.214344 reg_l2 19.791693
loss 7.259131
STEP 7 ================================
prereg loss 0.89181036 reg_l1 62.14505 reg_l2 19.75791
loss 7.1063156
STEP 8 ================================
prereg loss 0.9306998 reg_l1 62.075397 reg_l2 19.724365
loss 7.13824
STEP 9 ================================
prereg loss 1.0145833 reg_l1 62.01523 reg_l2 19.697023
loss 7.2161064
STEP 10 ================================
prereg loss 0.99883455 reg_l1 61.96734 reg_l2 19.677114
loss 7.1955686
STEP 11 ================================
prereg loss 0.9793008 reg_l1 61.928238 reg_l2 19.662043
loss 7.172125
STEP 12 ================================
prereg loss 0.9920566 reg_l1 61.892715 reg_l2 19.648245
loss 7.181328
STEP 13 ================================
prereg loss 0.95936555 reg_l1 61.856457 reg_l2 19.633032
loss 7.145011
STEP 14 ================================
prereg loss 0.88600343 reg_l1 61.817753 reg_l2 19.615648
loss 7.067779
STEP 15 ================================
prereg loss 0.85112864 reg_l1 61.777576 reg_l2 19.597195
loss 7.0288863
STEP 16 ================================
prereg loss 0.8690047 reg_l1 61.73767 reg_l2 19.579235
loss 7.042772
STEP 17 ================================
prereg loss 0.87820095 reg_l1 61.698727 reg_l2 19.562555
loss 7.048074
STEP 18 ================================
prereg loss 0.8613235 reg_l1 61.65959 reg_l2 19.546558
loss 7.0272827
STEP 19 ================================
prereg loss 0.8515076 reg_l1 61.617725 reg_l2 19.529577
loss 7.0132804
STEP 20 ================================
prereg loss 0.8515365 reg_l1 61.572033 reg_l2 19.5102
loss 7.0087395
STEP 21 ================================
prereg loss 0.84112483 reg_l1 61.520832 reg_l2 19.487305
loss 6.9932084
STEP 22 ================================
prereg loss 0.8206234 reg_l1 61.464733 reg_l2 19.461252
loss 6.967097
STEP 23 ================================
prereg loss 0.8204474 reg_l1 61.40599 reg_l2 19.433544
loss 6.9610467
STEP 24 ================================
prereg loss 0.8432201 reg_l1 61.348194 reg_l2 19.406372
loss 6.9780397
STEP 25 ================================
prereg loss 0.8460578 reg_l1 61.293762 reg_l2 19.38135
loss 6.9754343
STEP 26 ================================
prereg loss 0.82235205 reg_l1 61.242714 reg_l2 19.35849
loss 6.9466233
STEP 27 ================================
prereg loss 0.80476266 reg_l1 61.193016 reg_l2 19.336536
loss 6.9240646
STEP 28 ================================
prereg loss 0.8040727 reg_l1 61.14179 reg_l2 19.313774
loss 6.918252
STEP 29 ================================
prereg loss 0.8026173 reg_l1 61.087376 reg_l2 19.289127
loss 6.911355
STEP 30 ================================
prereg loss 0.79770213 reg_l1 61.02953 reg_l2 19.262503
loss 6.9006553
STEP 31 ================================
prereg loss 0.79710436 reg_l1 60.96957 reg_l2 19.234705
loss 6.8940616
STEP 32 ================================
prereg loss 0.7973551 reg_l1 60.90904 reg_l2 19.20681
loss 6.888259
STEP 33 ================================
prereg loss 0.7911396 reg_l1 60.8494 reg_l2 19.179708
loss 6.8760796
STEP 34 ================================
prereg loss 0.78504264 reg_l1 60.79098 reg_l2 19.153568
loss 6.864141
STEP 35 ================================
prereg loss 0.78824663 reg_l1 60.732998 reg_l2 19.127905
loss 6.8615465
STEP 36 ================================
prereg loss 0.7936563 reg_l1 60.674496 reg_l2 19.102106
loss 6.861106
STEP 37 ================================
prereg loss 0.79083127 reg_l1 60.615166 reg_l2 19.075964
loss 6.852348
STEP 38 ================================
prereg loss 0.78365266 reg_l1 60.55561 reg_l2 19.049812
loss 6.839214
STEP 39 ================================
prereg loss 0.7797076 reg_l1 60.49673 reg_l2 19.024244
loss 6.8293805
STEP 40 ================================
prereg loss 0.7773317 reg_l1 60.439274 reg_l2 18.999712
loss 6.8212595
STEP 41 ================================
prereg loss 0.7741165 reg_l1 60.382988 reg_l2 18.97607
loss 6.8124156
STEP 42 ================================
prereg loss 0.77220684 reg_l1 60.32691 reg_l2 18.952675
loss 6.804898
STEP 43 ================================
prereg loss 0.7712968 reg_l1 60.269833 reg_l2 18.928743
loss 6.7982802
STEP 44 ================================
prereg loss 0.76868665 reg_l1 60.211117 reg_l2 18.903936
loss 6.7897987
STEP 45 ================================
prereg loss 0.7666138 reg_l1 60.15115 reg_l2 18.878448
loss 6.781729
STEP 46 ================================
prereg loss 0.76819706 reg_l1 60.09115 reg_l2 18.853037
loss 6.777312
STEP 47 ================================
prereg loss 0.76982605 reg_l1 60.032295 reg_l2 18.828457
loss 6.7730556
STEP 48 ================================
prereg loss 0.7673166 reg_l1 59.974957 reg_l2 18.804932
loss 6.7648125
STEP 49 ================================
prereg loss 0.7630947 reg_l1 59.918648 reg_l2 18.78216
loss 6.7549596
STEP 50 ================================
prereg loss 0.760098 reg_l1 59.862274 reg_l2 18.759443
loss 6.7463255
STEP 51 ================================
prereg loss 0.7577317 reg_l1 59.804996 reg_l2 18.736265
loss 6.7382317
STEP 52 ================================
prereg loss 0.75606346 reg_l1 59.746483 reg_l2 18.712538
loss 6.730712
STEP 53 ================================
prereg loss 0.7555633 reg_l1 59.687492 reg_l2 18.688595
loss 6.724313
STEP 54 ================================
prereg loss 0.75469595 reg_l1 59.62847 reg_l2 18.664843
loss 6.717543
STEP 55 ================================
prereg loss 0.75304437 reg_l1 59.569695 reg_l2 18.641468
loss 6.7100143
STEP 56 ================================
prereg loss 0.75242776 reg_l1 59.510994 reg_l2 18.618296
loss 6.703527
STEP 57 ================================
prereg loss 0.75277406 reg_l1 59.451916 reg_l2 18.595068
loss 6.697966
STEP 58 ================================
prereg loss 0.75235134 reg_l1 59.39225 reg_l2 18.57166
loss 6.6915765
STEP 59 ================================
prereg loss 0.7511576 reg_l1 59.332294 reg_l2 18.54825
loss 6.684387
STEP 60 ================================
prereg loss 0.7498925 reg_l1 59.272465 reg_l2 18.52512
loss 6.6771393
STEP 61 ================================
prereg loss 0.7484621 reg_l1 59.213078 reg_l2 18.50246
loss 6.6697702
STEP 62 ================================
prereg loss 0.74689543 reg_l1 59.154083 reg_l2 18.480223
loss 6.662304
STEP 63 ================================
prereg loss 0.74588954 reg_l1 59.094883 reg_l2 18.458038
loss 6.655378
STEP 64 ================================
prereg loss 0.74511564 reg_l1 59.034912 reg_l2 18.435566
loss 6.6486073
STEP 65 ================================
prereg loss 0.7443646 reg_l1 58.974056 reg_l2 18.412745
loss 6.6417704
STEP 66 ================================
prereg loss 0.74413604 reg_l1 58.912777 reg_l2 18.389809
loss 6.6354136
STEP 67 ================================
prereg loss 0.74417776 reg_l1 58.85161 reg_l2 18.367136
loss 6.6293387
STEP 68 ================================
prereg loss 0.7435394 reg_l1 58.790943 reg_l2 18.344938
loss 6.622634
STEP 69 ================================
prereg loss 0.74222255 reg_l1 58.730675 reg_l2 18.323154
loss 6.6152897
STEP 70 ================================
prereg loss 0.7408569 reg_l1 58.67036 reg_l2 18.301527
loss 6.607893
STEP 71 ================================
prereg loss 0.7396265 reg_l1 58.609562 reg_l2 18.279766
loss 6.6005826
STEP 72 ================================
prereg loss 0.73864526 reg_l1 58.548195 reg_l2 18.257809
loss 6.593465
STEP 73 ================================
prereg loss 0.7378328 reg_l1 58.48827 reg_l2 18.235764
loss 6.5866594
STEP 74 ================================
prereg loss 0.7370962 reg_l1 58.428226 reg_l2 18.21383
loss 6.579919
STEP 75 ================================
prereg loss 0.73624825 reg_l1 58.36816 reg_l2 18.192112
loss 6.573065
STEP 76 ================================
prereg loss 0.7354905 reg_l1 58.307915 reg_l2 18.170595
loss 6.566282
STEP 77 ================================
prereg loss 0.73487854 reg_l1 58.247456 reg_l2 18.149166
loss 6.559624
STEP 78 ================================
prereg loss 0.73410404 reg_l1 58.186573 reg_l2 18.127722
loss 6.5527616
STEP 79 ================================
prereg loss 0.7331378 reg_l1 58.125614 reg_l2 18.106352
loss 6.545699
STEP 80 ================================
prereg loss 0.7321092 reg_l1 58.06456 reg_l2 18.085152
loss 6.538565
STEP 81 ================================
prereg loss 0.7308705 reg_l1 58.004337 reg_l2 18.064186
loss 6.5313044
STEP 82 ================================
prereg loss 0.7296433 reg_l1 57.945023 reg_l2 18.04342
loss 6.5241456
STEP 83 ================================
prereg loss 0.7285374 reg_l1 57.885284 reg_l2 18.022728
loss 6.517066
STEP 84 ================================
prereg loss 0.7275208 reg_l1 57.825882 reg_l2 18.002064
loss 6.5101094
STEP 85 ================================
prereg loss 0.72660357 reg_l1 57.768703 reg_l2 17.981432
loss 6.5034738
STEP 86 ================================
prereg loss 0.72557527 reg_l1 57.71268 reg_l2 17.96098
loss 6.496844
STEP 87 ================================
prereg loss 0.72436947 reg_l1 57.656326 reg_l2 17.940851
loss 6.490002
STEP 88 ================================
prereg loss 0.72264063 reg_l1 57.600037 reg_l2 17.9211
loss 6.482644
STEP 89 ================================
prereg loss 0.72072744 reg_l1 57.54565 reg_l2 17.901691
loss 6.4752927
STEP 90 ================================
prereg loss 0.7186605 reg_l1 57.492977 reg_l2 17.882456
loss 6.467958
STEP 91 ================================
prereg loss 0.7166377 reg_l1 57.441055 reg_l2 17.863329
loss 6.4607434
STEP 92 ================================
prereg loss 0.7145388 reg_l1 57.38939 reg_l2 17.844366
loss 6.453478
STEP 93 ================================
prereg loss 0.7124183 reg_l1 57.33776 reg_l2 17.825624
loss 6.4461946
STEP 94 ================================
prereg loss 0.7102235 reg_l1 57.28562 reg_l2 17.80718
loss 6.438786
STEP 95 ================================
prereg loss 0.7079543 reg_l1 57.232964 reg_l2 17.789
loss 6.431251
STEP 96 ================================
prereg loss 0.7056412 reg_l1 57.179756 reg_l2 17.771017
loss 6.423617
STEP 97 ================================
prereg loss 0.7032727 reg_l1 57.127167 reg_l2 17.753145
loss 6.4159894
STEP 98 ================================
prereg loss 0.7008839 reg_l1 57.075466 reg_l2 17.735367
loss 6.4084306
STEP 99 ================================
prereg loss 0.69852006 reg_l1 57.02369 reg_l2 17.717745
loss 6.4008894
STEP 100 ================================
prereg loss 0.6961886 reg_l1 56.972553 reg_l2 17.70027
loss 6.393444
2022-06-26T19:45:33.143

julia> # a very nice starting point, and nice regularization

julia> # training is slow, but very smooth, with this level of regularization

julia> serialize("sparse9-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse9-after-100-steps-opt.ser", opt)

julia> count_interval(sparse9, -0.001f0, 0.001f0)
6

julia> count_interval(sparse9, -0.01f0, 0.01f0)
13

julia> count_interval(sparse9, -0.02f0, 0.02f0)
16

julia> count_interval(sparse9, -0.03f0, 0.03f0)
22

julia> count_interval(sparse9, -0.04f0, 0.04f0)
28

julia> count_interval(sparse9, -0.05f0, 0.05f0)
33

julia> count_interval(sparse9, -0.06f0, 0.06f0)
41

julia> count_interval(sparse9, -0.065f0, 0.065f0)
48

julia> count_interval(sparse9, -0.07f0, 0.07f0)
54

julia> count_interval(sparse9, -0.065f0, 0.065f0)
48

julia> # let's do this one

julia> sparse10 = sparsecopy(sparse9, 0.065f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.6858)), "dict-1"=>Dict("input"=>Dict("char"=>0.377512),…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.496172), "norm-3-1"=>Dict("norm"=>0.418255), "norm-4-2"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.295197), "input"=>Dict("char"=>-0.34533)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.444606), "accum-1-1"=>Dict("dict"=>0.185167), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.237286), "accum-1-1"=>Dict("dict"=>0.155357), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0960874), "accum-1-2"=>Dict("dict"=>0.242646)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0962412), "dot-2-2"=>Dict("dot"=>0.256674), "norm-2-2…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.223177), "norm-2-1"=>Dict("norm"=>0.210341), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.0998435), "dot-3-1"=>Dict("dot"=>0.110265), "compare-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.221956)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.11841)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.179044), "eos"=>Dict("char"=>-0.178914)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.109233)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.23779), "dot-2-2"=>Dict("dot"=>-0.149247), "norm-3-1"=>…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.274394), "norm-3-1"=>Dict("norm"=>0.177106), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.724327), "eos"=>Dict("char"=>-0.168895)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.45908), "dot-1-2"=>Dict("dot"=>-0.139746), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.185447), "accum-1-2"=>Dict("dict"=>-0.27172), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.243197), "norm-3-1"=>Dict("norm"=>0.198816), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0666553), "norm-2-1"=>Dict("norm"=>-0.310677), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.101223), "norm-2-1"=>Dict("norm"=>-0.0825053), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.246502), "accum-1-1"=>Dict("dict"=>0.146235), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0719541), "compare-3-2"=>Dict("true"=>0.316767)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.0966963)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.129759), "compare-4-2"=>Dict("true"=>0.823516)), "di…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse10
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.6858)), "dict-1"=>Dict("input"=>Dict("char"=>0.377512),…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.496172), "norm-3-1"=>Dict("norm"=>0.418255), "norm-4-2"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.295197), "input"=>Dict("char"=>-0.34533)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.444606), "accum-1-1"=>Dict("dict"=>0.185167), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.237286), "accum-1-1"=>Dict("dict"=>0.155357), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0960874), "accum-1-2"=>Dict("dict"=>0.242646)))
  "dot-3-1"     => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.0962412), "dot-2-2"=>Dict("dot"=>0.256674), "norm-2-2…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.223177), "norm-2-1"=>Dict("norm"=>0.210341), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.0998435), "dot-3-1"=>Dict("dot"=>0.110265), "compare-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.221956)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.11841)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.179044), "eos"=>Dict("char"=>-0.178914)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.109233)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.23779), "dot-2-2"=>Dict("dot"=>-0.149247), "norm-3-1"=>…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.274394), "norm-3-1"=>Dict("norm"=>0.177106), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.724327), "eos"=>Dict("char"=>-0.168895)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.45908), "dot-1-2"=>Dict("dot"=>-0.139746), "const_1"=>Dic…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.185447), "accum-1-2"=>Dict("dict"=>-0.27172), "input"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.243197), "norm-3-1"=>Dict("norm"=>0.198816), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("dot-1-2"=>Dict("dot"=>-0.0666553), "norm-2-1"=>Dict("norm"=>-0.310677), "norm-2…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.101223), "norm-2-1"=>Dict("norm"=>-0.0825053), "dot-2-…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.246502), "accum-1-1"=>Dict("dict"=>0.146235), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0719541), "compare-3-2"=>Dict("true"=>0.316767)), "d…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.0966963)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.129759), "compare-4-2"=>Dict("true"=>0.823516)), "di…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
266

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T19:50:28.833
STEP 1 ================================
prereg loss 2.9905622 reg_l1 55.38505 reg_l2 17.609358
loss 8.529067
STEP 2 ================================
prereg loss 2.546235 reg_l1 55.33505 reg_l2 17.58427
loss 8.079741
STEP 3 ================================
prereg loss 2.4807656 reg_l1 55.331573 reg_l2 17.575571
loss 8.013923
STEP 4 ================================
prereg loss 2.2888122 reg_l1 55.34723 reg_l2 17.576256
loss 7.823535
STEP 5 ================================
prereg loss 2.0176795 reg_l1 55.373596 reg_l2 17.581741
loss 7.5550394
STEP 6 ================================
prereg loss 1.7677753 reg_l1 55.398075 reg_l2 17.585945
loss 7.307583
STEP 7 ================================
prereg loss 1.5874438 reg_l1 55.416298 reg_l2 17.58676
loss 7.1290736
STEP 8 ================================
prereg loss 1.4850663 reg_l1 55.429504 reg_l2 17.585262
loss 7.028017
STEP 9 ================================
prereg loss 1.3991032 reg_l1 55.43941 reg_l2 17.582876
loss 6.943044
STEP 10 ================================
prereg loss 1.2728708 reg_l1 55.447197 reg_l2 17.580545
loss 6.8175907
STEP 11 ================================
prereg loss 1.1303967 reg_l1 55.453053 reg_l2 17.578247
loss 6.675702
STEP 12 ================================
prereg loss 1.0243787 reg_l1 55.455956 reg_l2 17.575045
loss 6.5699744
STEP 13 ================================
prereg loss 0.9588809 reg_l1 55.45474 reg_l2 17.570051
loss 6.504355
STEP 14 ================================
prereg loss 0.9165711 reg_l1 55.450367 reg_l2 17.563509
loss 6.461608
STEP 15 ================================
prereg loss 0.8810549 reg_l1 55.4443 reg_l2 17.556168
loss 6.425485
STEP 16 ================================
prereg loss 0.84526545 reg_l1 55.438725 reg_l2 17.54923
loss 6.3891377
STEP 17 ================================
prereg loss 0.8028261 reg_l1 55.434963 reg_l2 17.543547
loss 6.3463225
STEP 18 ================================
prereg loss 0.75570464 reg_l1 55.432686 reg_l2 17.539104
loss 6.298973
STEP 19 ================================
prereg loss 0.7176442 reg_l1 55.430275 reg_l2 17.53517
loss 6.2606716
STEP 20 ================================
prereg loss 0.7010161 reg_l1 55.42461 reg_l2 17.530136
loss 6.243477
STEP 21 ================================
prereg loss 0.70055175 reg_l1 55.41341 reg_l2 17.522762
loss 6.241893
STEP 22 ================================
prereg loss 0.698591 reg_l1 55.39474 reg_l2 17.51212
loss 6.238065
STEP 23 ================================
prereg loss 0.69238824 reg_l1 55.369305 reg_l2 17.498522
loss 6.2293186
STEP 24 ================================
prereg loss 0.68606585 reg_l1 55.338306 reg_l2 17.482613
loss 6.2198963
STEP 25 ================================
prereg loss 0.6836598 reg_l1 55.303326 reg_l2 17.465267
loss 6.213992
STEP 26 ================================
prereg loss 0.68550485 reg_l1 55.265827 reg_l2 17.447186
loss 6.2120876
STEP 27 ================================
prereg loss 0.6890967 reg_l1 55.227924 reg_l2 17.42929
loss 6.2118893
STEP 28 ================================
prereg loss 0.6909754 reg_l1 55.189587 reg_l2 17.411358
loss 6.209934
STEP 29 ================================
prereg loss 0.68623275 reg_l1 55.15074 reg_l2 17.393206
loss 6.201307
STEP 30 ================================
prereg loss 0.67183936 reg_l1 55.111828 reg_l2 17.375051
loss 6.183022
STEP 31 ================================
prereg loss 0.6546336 reg_l1 55.0723 reg_l2 17.356583
loss 6.161864
STEP 32 ================================
prereg loss 0.64081556 reg_l1 55.03224 reg_l2 17.337887
loss 6.14404
STEP 33 ================================
prereg loss 0.6345785 reg_l1 54.99118 reg_l2 17.318884
loss 6.133697
STEP 34 ================================
prereg loss 0.6322512 reg_l1 54.948902 reg_l2 17.299435
loss 6.1271415
STEP 35 ================================
prereg loss 0.62900215 reg_l1 54.90513 reg_l2 17.279478
loss 6.119515
STEP 36 ================================
prereg loss 0.62325364 reg_l1 54.859974 reg_l2 17.259005
loss 6.1092515
STEP 37 ================================
prereg loss 0.61672723 reg_l1 54.813766 reg_l2 17.238134
loss 6.098104
STEP 38 ================================
prereg loss 0.61177176 reg_l1 54.7673 reg_l2 17.217196
loss 6.0885015
STEP 39 ================================
prereg loss 0.60882837 reg_l1 54.721493 reg_l2 17.196634
loss 6.080978
STEP 40 ================================
prereg loss 0.60565615 reg_l1 54.677296 reg_l2 17.176888
loss 6.0733857
STEP 41 ================================
prereg loss 0.5997038 reg_l1 54.63529 reg_l2 17.1583
loss 6.063233
STEP 42 ================================
prereg loss 0.5900424 reg_l1 54.595554 reg_l2 17.140968
loss 6.049598
STEP 43 ================================
prereg loss 0.5788425 reg_l1 54.557667 reg_l2 17.124632
loss 6.0346093
STEP 44 ================================
prereg loss 0.56918484 reg_l1 54.520718 reg_l2 17.108871
loss 6.0212564
STEP 45 ================================
prereg loss 0.5621414 reg_l1 54.48371 reg_l2 17.09318
loss 6.010513
STEP 46 ================================
prereg loss 0.5565017 reg_l1 54.445923 reg_l2 17.077188
loss 6.0010943
STEP 47 ================================
prereg loss 0.5507276 reg_l1 54.407005 reg_l2 17.060743
loss 5.9914284
STEP 48 ================================
prereg loss 0.5443535 reg_l1 54.367043 reg_l2 17.04391
loss 5.9810576
STEP 49 ================================
prereg loss 0.53786886 reg_l1 54.32642 reg_l2 17.026915
loss 5.970511
STEP 50 ================================
prereg loss 0.5319835 reg_l1 54.28566 reg_l2 17.009993
loss 5.9605494
STEP 51 ================================
prereg loss 0.52656364 reg_l1 54.245216 reg_l2 16.9933
loss 5.9510856
STEP 52 ================================
prereg loss 0.5204349 reg_l1 54.2055 reg_l2 16.977
loss 5.940985
STEP 53 ================================
prereg loss 0.51281244 reg_l1 54.166546 reg_l2 16.961082
loss 5.929467
STEP 54 ================================
prereg loss 0.5041858 reg_l1 54.12821 reg_l2 16.945509
loss 5.9170065
STEP 55 ================================
prereg loss 0.4958586 reg_l1 54.090374 reg_l2 16.930254
loss 5.9048963
STEP 56 ================================
prereg loss 0.48858628 reg_l1 54.05271 reg_l2 16.915205
loss 5.8938575
STEP 57 ================================
prereg loss 0.48214498 reg_l1 54.014877 reg_l2 16.900211
loss 5.8836327
STEP 58 ================================
prereg loss 0.47600034 reg_l1 53.976494 reg_l2 16.885073
loss 5.8736496
STEP 59 ================================
prereg loss 0.46966892 reg_l1 53.93735 reg_l2 16.869678
loss 5.8634043
STEP 60 ================================
prereg loss 0.4633629 reg_l1 53.897575 reg_l2 16.854074
loss 5.8531203
STEP 61 ================================
prereg loss 0.45754156 reg_l1 53.85762 reg_l2 16.83849
loss 5.8433037
STEP 62 ================================
prereg loss 0.4521089 reg_l1 53.81795 reg_l2 16.823175
loss 5.8339043
STEP 63 ================================
prereg loss 0.4465057 reg_l1 53.77889 reg_l2 16.808342
loss 5.8243947
STEP 64 ================================
prereg loss 0.44047824 reg_l1 53.740524 reg_l2 16.793983
loss 5.814531
STEP 65 ================================
prereg loss 0.43391415 reg_l1 53.702408 reg_l2 16.779903
loss 5.804155
STEP 66 ================================
prereg loss 0.4273501 reg_l1 53.664642 reg_l2 16.7661
loss 5.793814
STEP 67 ================================
prereg loss 0.42120916 reg_l1 53.6267 reg_l2 16.752285
loss 5.7838798
STEP 68 ================================
prereg loss 0.41539788 reg_l1 53.58822 reg_l2 16.738308
loss 5.7742195
STEP 69 ================================
prereg loss 0.40968817 reg_l1 53.54911 reg_l2 16.724113
loss 5.7645993
STEP 70 ================================
prereg loss 0.40401825 reg_l1 53.509586 reg_l2 16.709845
loss 5.754977
STEP 71 ================================
prereg loss 0.39868554 reg_l1 53.46987 reg_l2 16.695639
loss 5.7456727
STEP 72 ================================
prereg loss 0.3937611 reg_l1 53.430103 reg_l2 16.681599
loss 5.7367716
STEP 73 ================================
prereg loss 0.38897434 reg_l1 53.39105 reg_l2 16.667747
loss 5.7280793
STEP 74 ================================
prereg loss 0.3840418 reg_l1 53.35323 reg_l2 16.65408
loss 5.7193646
STEP 75 ================================
prereg loss 0.379067 reg_l1 53.315388 reg_l2 16.640644
loss 5.7106056
STEP 76 ================================
prereg loss 0.37427574 reg_l1 53.277397 reg_l2 16.627413
loss 5.7020154
STEP 77 ================================
prereg loss 0.36977047 reg_l1 53.239094 reg_l2 16.614311
loss 5.69368
STEP 78 ================================
prereg loss 0.36557615 reg_l1 53.20009 reg_l2 16.601141
loss 5.685585
STEP 79 ================================
prereg loss 0.36163208 reg_l1 53.16013 reg_l2 16.587723
loss 5.6776447
STEP 80 ================================
prereg loss 0.35787815 reg_l1 53.119102 reg_l2 16.574007
loss 5.6697884
STEP 81 ================================
prereg loss 0.35446194 reg_l1 53.07715 reg_l2 16.560047
loss 5.662177
STEP 82 ================================
prereg loss 0.35136285 reg_l1 53.03453 reg_l2 16.545965
loss 5.6548157
STEP 83 ================================
prereg loss 0.34841254 reg_l1 52.99143 reg_l2 16.53188
loss 5.6475554
STEP 84 ================================
prereg loss 0.34541234 reg_l1 52.94788 reg_l2 16.517807
loss 5.6402
STEP 85 ================================
prereg loss 0.34243014 reg_l1 52.903717 reg_l2 16.503632
loss 5.632802
STEP 86 ================================
prereg loss 0.33958635 reg_l1 52.858685 reg_l2 16.489218
loss 5.625455
STEP 87 ================================
prereg loss 0.33696678 reg_l1 52.812664 reg_l2 16.474483
loss 5.6182337
STEP 88 ================================
prereg loss 0.33457527 reg_l1 52.766407 reg_l2 16.459467
loss 5.611216
STEP 89 ================================
prereg loss 0.33233225 reg_l1 52.719536 reg_l2 16.444206
loss 5.6042857
STEP 90 ================================
prereg loss 0.3302363 reg_l1 52.672195 reg_l2 16.428736
loss 5.597456
STEP 91 ================================
prereg loss 0.3283453 reg_l1 52.625263 reg_l2 16.4131
loss 5.590872
STEP 92 ================================
prereg loss 0.32662317 reg_l1 52.57746 reg_l2 16.397308
loss 5.584369
STEP 93 ================================
prereg loss 0.32498387 reg_l1 52.528973 reg_l2 16.38145
loss 5.5778813
STEP 94 ================================
prereg loss 0.32337108 reg_l1 52.480003 reg_l2 16.365622
loss 5.5713716
STEP 95 ================================
prereg loss 0.3217715 reg_l1 52.430664 reg_l2 16.349867
loss 5.564838
STEP 96 ================================
prereg loss 0.3202422 reg_l1 52.380867 reg_l2 16.334154
loss 5.558329
STEP 97 ================================
prereg loss 0.31884712 reg_l1 52.330498 reg_l2 16.318396
loss 5.551897
STEP 98 ================================
prereg loss 0.3175867 reg_l1 52.28009 reg_l2 16.302504
loss 5.545596
STEP 99 ================================
prereg loss 0.31644663 reg_l1 52.22915 reg_l2 16.286482
loss 5.539362
STEP 100 ================================
prereg loss 0.31541145 reg_l1 52.17761 reg_l2 16.27041
loss 5.5331726
2022-06-26T20:05:27.427

julia> serialize("sparse10-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse10-after-100-steps-opt.ser", opt)

julia> count_interval(sparse10, -0.001f0, 0.001f0)
3

julia> count_interval(sparse10, -0.01f0, 0.01f0)
7

julia> count_interval(sparse10, -0.02f0, 0.02f0)
8

julia> count_interval(sparse10, -0.03f0, 0.03f0)
9

julia> count_interval(sparse10, -0.04f0, 0.04f0)
13

julia> count_interval(sparse10, -0.05f0, 0.05f0)
17

julia> count_interval(sparse10, -0.06f0, 0.06f0)
23

julia> count_interval(sparse10, -0.07f0, 0.07f0)
36

julia> count_interval(sparse10, -0.075f0, 0.075f0)
42

julia> count_interval(sparse10, -0.08f0, 0.08f0)
46

julia> count_interval(sparse10, -0.075f0, 0.075f0)
42

julia> # let's do this; let's keep current regularization

julia> sparse11 = sparsecopy(sparse10, 0.075f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.699062)), "dict-1"=>Dict("input"=>Dict("char"=>0.353492…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.479972), "norm-3-1"=>Dict("norm"=>0.404823), "norm-4-2"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.304764), "input"=>Dict("char"=>-0.329261)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.355807), "accum-1-1"=>Dict("dict"=>0.179042), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.249914), "accum-1-1"=>Dict("dict"=>0.157277), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0950023), "accum-1-2"=>Dict("dict"=>0.24516)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.275138), "norm-2-2"=>Dict("norm"=>0.109979), "dot-2-1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.143597), "norm-2-1"=>Dict("norm"=>0.214221), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.0777549), "dot-3-1"=>Dict("dot"=>0.103789), "compare-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.123018)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.122536)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.151656), "eos"=>Dict("char"=>-0.18612)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.103067)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.267137), "dot-2-2"=>Dict("dot"=>-0.178836), "norm-3-1"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.273971), "norm-3-1"=>Dict("norm"=>0.174526), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.626577), "eos"=>Dict("char"=>-0.162402)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.426298), "dot-1-2"=>Dict("dot"=>-0.118223), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0834957), "accum-1-2"=>Dict("dict"=>-0.285544), "eos"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.253424), "norm-3-1"=>Dict("norm"=>0.212682), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.314724), "norm-2-2"=>Dict("norm"=>0.235472), "const_…
  "compare-5-2" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.146894), "norm-2-1"=>Dict("norm"=>-0.106962), "dot-2…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.259126), "accum-1-1"=>Dict("dict"=>0.149051), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.216768)), "dict-1"=>Dict("compare-3-2"=>Dict("true…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.102022)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.12287), "compare-4-2"=>Dict("true"=>0.723542)), "dic…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse11
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.699062)), "dict-1"=>Dict("input"=>Dict("char"=>0.353492…
  "norm-5-2"    => Dict("dict"=>Dict("norm-2-1"=>Dict("norm"=>0.479972), "norm-3-1"=>Dict("norm"=>0.404823), "norm-4-2"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.304764), "input"=>Dict("char"=>-0.329261)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.355807), "accum-1-1"=>Dict("dict"=>0.179042), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.249914), "accum-1-1"=>Dict("dict"=>0.157277), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0950023), "accum-1-2"=>Dict("dict"=>0.24516)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.275138), "norm-2-2"=>Dict("norm"=>0.109979), "dot-2-1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.143597), "norm-2-1"=>Dict("norm"=>0.214221), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("norm-5-2"=>Dict("norm"=>0.0777549), "dot-3-1"=>Dict("dot"=>0.103789), "compare-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.123018)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.122536)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.151656), "eos"=>Dict("char"=>-0.18612)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.103067)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.267137), "dot-2-2"=>Dict("dot"=>-0.178836), "norm-3-1"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.273971), "norm-3-1"=>Dict("norm"=>0.174526), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.626577), "eos"=>Dict("char"=>-0.162402)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.426298), "dot-1-2"=>Dict("dot"=>-0.118223), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0834957), "accum-1-2"=>Dict("dict"=>-0.285544), "eos"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.253424), "norm-3-1"=>Dict("norm"=>0.212682), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.314724), "norm-2-2"=>Dict("norm"=>0.235472), "const_…
  "compare-5-2" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.146894), "norm-2-1"=>Dict("norm"=>-0.106962), "dot-2…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.259126), "accum-1-1"=>Dict("dict"=>0.149051), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.216768)), "dict-1"=>Dict("compare-3-2"=>Dict("true…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.102022)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.12287), "compare-4-2"=>Dict("true"=>0.723542)), "dic…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
224

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T20:09:21.342
STEP 1 ================================
prereg loss 0.43126366 reg_l1 50.142773 reg_l2 16.134995
loss 5.445541
STEP 2 ================================
prereg loss 0.66301626 reg_l1 50.13277 reg_l2 16.136454
loss 5.6762934
STEP 3 ================================
prereg loss 0.4005691 reg_l1 50.111965 reg_l2 16.1214
loss 5.4117656
STEP 4 ================================
prereg loss 0.5953962 reg_l1 50.102215 reg_l2 16.114662
loss 5.6056175
STEP 5 ================================
prereg loss 0.5096802 reg_l1 50.101635 reg_l2 16.11555
loss 5.519844
STEP 6 ================================
ERROR: InterruptException:
Stacktrace:
[...] I INTERRUPTED THIS

julia> # no, look, let's do a stronger cut-off, this start is too soft

julia> sparse11 = sparsecopy(sparse10, 0.08f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.699062)), "dict-1"=>Dict("input"=>Dict("char"=>0.353492…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.405757), "norm-2-1"=>Dict("norm"=>0.479972), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.304764), "input"=>Dict("char"=>-0.329261)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.355807), "accum-1-1"=>Dict("dict"=>0.179042), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.249914), "accum-1-1"=>Dict("dict"=>0.157277), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0950023), "accum-1-2"=>Dict("dict"=>0.24516)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.275138), "norm-2-2"=>Dict("norm"=>0.109979), "dot-2-1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.143597), "norm-2-1"=>Dict("norm"=>0.214221), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.103789), "compare-5-1"=>Dict("true"=>0.327313), "dot-4-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.123018)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.122536)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.151656), "eos"=>Dict("char"=>-0.18612)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.103067)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.267137), "dot-2-2"=>Dict("dot"=>-0.178836), "norm-3-1"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.273971), "norm-3-1"=>Dict("norm"=>0.174526), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.626577), "eos"=>Dict("char"=>-0.162402)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.426298), "dot-1-2"=>Dict("dot"=>-0.118223), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0834957), "accum-1-2"=>Dict("dict"=>-0.285544), "eos"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.253424), "norm-3-1"=>Dict("norm"=>0.212682), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.314724), "norm-2-2"=>Dict("norm"=>0.235472), "const_…
  "compare-5-2" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.146894), "norm-2-1"=>Dict("norm"=>-0.106962), "dot-2…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.259126), "accum-1-1"=>Dict("dict"=>0.149051), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.216768)), "dict-1"=>Dict("compare-3-2"=>Dict("true…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.102022)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.12287), "compare-4-2"=>Dict("true"=>0.723542)), "dic…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse11
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 32 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.699062)), "dict-1"=>Dict("input"=>Dict("char"=>0.353492…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.405757), "norm-2-1"=>Dict("norm"=>0.479972), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.304764), "input"=>Dict("char"=>-0.329261)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.355807), "accum-1-1"=>Dict("dict"=>0.179042), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.249914), "accum-1-1"=>Dict("dict"=>0.157277), "accum-1…
  "norm-4-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.0950023), "accum-1-2"=>Dict("dict"=>0.24516)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.275138), "norm-2-2"=>Dict("norm"=>0.109979), "dot-2-1"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.143597), "norm-2-1"=>Dict("norm"=>0.214221), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.103789), "compare-5-1"=>Dict("true"=>0.327313), "dot-4-…
  "accum-1-1"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.123018)))
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.122536)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("input"=>Dict("char"=>0.151656), "eos"=>Dict("char"=>-0.18612)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.103067)))
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.267137), "dot-2-2"=>Dict("dot"=>-0.178836), "norm-3-1"=…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.273971), "norm-3-1"=>Dict("norm"=>0.174526), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.626577), "eos"=>Dict("char"=>-0.162402)), "dict-1"=>Dict…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.426298), "dot-1-2"=>Dict("dot"=>-0.118223), "const_1"=>Di…
  "dot-2-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0834957), "accum-1-2"=>Dict("dict"=>-0.285544), "eos"=>…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.253424), "norm-3-1"=>Dict("norm"=>0.212682), "dot-3-1…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.314724), "norm-2-2"=>Dict("norm"=>0.235472), "const_…
  "compare-5-2" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.146894), "norm-2-1"=>Dict("norm"=>-0.106962), "dot-2…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.259126), "accum-1-1"=>Dict("dict"=>0.149051), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.216768)), "dict-1"=>Dict("compare-3-2"=>Dict("true…
  "accum-3-2"   => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>0.102022)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.12287), "compare-4-2"=>Dict("true"=>0.723542)), "dic…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
220

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T20:11:31.856
STEP 1 ================================
prereg loss 2.1955383 reg_l1 49.833225 reg_l2 16.111032
loss 7.1788607
STEP 2 ================================
prereg loss 1.514137 reg_l1 49.843216 reg_l2 16.123806
loss 6.498459
STEP 3 ================================
prereg loss 1.7375537 reg_l1 49.850487 reg_l2 16.130428
loss 6.7226024
STEP 4 ================================
prereg loss 1.5732217 reg_l1 49.835308 reg_l2 16.122025
loss 6.5567527
STEP 5 ================================
prereg loss 1.2930468 reg_l1 49.810806 reg_l2 16.10763
loss 6.2741275
STEP 6 ================================
prereg loss 1.1969031 reg_l1 49.782757 reg_l2 16.09194
loss 6.175179
STEP 7 ================================
prereg loss 1.227123 reg_l1 49.753872 reg_l2 16.076973
loss 6.20251
STEP 8 ================================
prereg loss 1.2048987 reg_l1 49.726353 reg_l2 16.063879
loss 6.177534
STEP 9 ================================
prereg loss 1.0945783 reg_l1 49.701965 reg_l2 16.053291
loss 6.064775
STEP 10 ================================
prereg loss 0.9745801 reg_l1 49.681072 reg_l2 16.045095
loss 5.9426875
STEP 11 ================================
prereg loss 0.9121555 reg_l1 49.66343 reg_l2 16.03875
loss 5.8784986
STEP 12 ================================
prereg loss 0.90203786 reg_l1 49.648083 reg_l2 16.033335
loss 5.866846
STEP 13 ================================
prereg loss 0.8889103 reg_l1 49.633522 reg_l2 16.027744
loss 5.8522625
STEP 14 ================================
prereg loss 0.837312 reg_l1 49.618397 reg_l2 16.021198
loss 5.7991514
STEP 15 ================================
prereg loss 0.76090425 reg_l1 49.602226 reg_l2 16.013584
loss 5.721127
STEP 16 ================================
prereg loss 0.69917244 reg_l1 49.585285 reg_l2 16.00532
loss 5.657701
STEP 17 ================================
prereg loss 0.6757727 reg_l1 49.568092 reg_l2 15.996993
loss 5.632582
STEP 18 ================================
prereg loss 0.67805207 reg_l1 49.550976 reg_l2 15.989098
loss 5.6331496
STEP 19 ================================
prereg loss 0.6745261 reg_l1 49.534203 reg_l2 15.981897
loss 5.6279464
STEP 20 ================================
prereg loss 0.64788896 reg_l1 49.517918 reg_l2 15.975472
loss 5.599681
STEP 21 ================================
prereg loss 0.6096813 reg_l1 49.502136 reg_l2 15.969735
loss 5.559895
STEP 22 ================================
prereg loss 0.58255976 reg_l1 49.486805 reg_l2 15.964441
loss 5.53124
STEP 23 ================================
prereg loss 0.57720685 reg_l1 49.471706 reg_l2 15.959281
loss 5.524378
STEP 24 ================================
prereg loss 0.58350706 reg_l1 49.45666 reg_l2 15.953857
loss 5.5291734
STEP 25 ================================
prereg loss 0.5833217 reg_l1 49.44144 reg_l2 15.94794
loss 5.527466
STEP 26 ================================
prereg loss 0.57021743 reg_l1 49.426117 reg_l2 15.941507
loss 5.5128293
STEP 27 ================================
prereg loss 0.5526303 reg_l1 49.410896 reg_l2 15.93477
loss 5.49372
STEP 28 ================================
prereg loss 0.54145765 reg_l1 49.39614 reg_l2 15.928057
loss 5.481072
STEP 29 ================================
prereg loss 0.54053015 reg_l1 49.38196 reg_l2 15.921666
loss 5.4787264
STEP 30 ================================
prereg loss 0.5426669 reg_l1 49.36845 reg_l2 15.915771
loss 5.479512
STEP 31 ================================
prereg loss 0.53965914 reg_l1 49.355373 reg_l2 15.910318
loss 5.4751964
STEP 32 ================================
prereg loss 0.53049076 reg_l1 49.34232 reg_l2 15.905124
loss 5.464723
STEP 33 ================================
prereg loss 0.5217263 reg_l1 49.32886 reg_l2 15.899837
loss 5.4546123
STEP 34 ================================
prereg loss 0.51835775 reg_l1 49.314503 reg_l2 15.894088
loss 5.449808
STEP 35 ================================
prereg loss 0.5194345 reg_l1 49.298996 reg_l2 15.88757
loss 5.449334
STEP 36 ================================
prereg loss 0.5201987 reg_l1 49.282204 reg_l2 15.880139
loss 5.448419
STEP 37 ================================
prereg loss 0.5169027 reg_l1 49.26436 reg_l2 15.871838
loss 5.4433384
STEP 38 ================================
prereg loss 0.5120831 reg_l1 49.245895 reg_l2 15.862941
loss 5.4366727
STEP 39 ================================
prereg loss 0.5092509 reg_l1 49.227367 reg_l2 15.853839
loss 5.431988
STEP 40 ================================
prereg loss 0.5088865 reg_l1 49.209183 reg_l2 15.844873
loss 5.429805
STEP 41 ================================
prereg loss 0.50808406 reg_l1 49.19161 reg_l2 15.836283
loss 5.427245
STEP 42 ================================
prereg loss 0.5041086 reg_l1 49.17459 reg_l2 15.828103
loss 5.4215674
STEP 43 ================================
prereg loss 0.49816248 reg_l1 49.157856 reg_l2 15.820187
loss 5.413948
STEP 44 ================================
prereg loss 0.49307138 reg_l1 49.141006 reg_l2 15.812245
loss 5.407172
STEP 45 ================================
prereg loss 0.4905796 reg_l1 49.1235 reg_l2 15.803938
loss 5.40293
STEP 46 ================================
prereg loss 0.4892987 reg_l1 49.10499 reg_l2 15.795015
loss 5.399798
STEP 47 ================================
prereg loss 0.48691678 reg_l1 49.08537 reg_l2 15.785379
loss 5.3954535
STEP 48 ================================
prereg loss 0.4834322 reg_l1 49.064793 reg_l2 15.775134
loss 5.3899117
STEP 49 ================================
prereg loss 0.48051465 reg_l1 49.04367 reg_l2 15.764505
loss 5.3848815
STEP 50 ================================
prereg loss 0.4791228 reg_l1 49.022415 reg_l2 15.753811
loss 5.3813643
STEP 51 ================================
prereg loss 0.47816226 reg_l1 49.001423 reg_l2 15.743323
loss 5.3783045
STEP 52 ================================
prereg loss 0.47612396 reg_l1 48.980946 reg_l2 15.733203
loss 5.3742185
STEP 53 ================================
prereg loss 0.47265902 reg_l1 48.960976 reg_l2 15.723475
loss 5.368757
STEP 54 ================================
prereg loss 0.46898732 reg_l1 48.94138 reg_l2 15.714009
loss 5.3631253
STEP 55 ================================
prereg loss 0.4661507 reg_l1 48.92177 reg_l2 15.704587
loss 5.358328
STEP 56 ================================
prereg loss 0.46398294 reg_l1 48.90187 reg_l2 15.694984
loss 5.3541703
STEP 57 ================================
prereg loss 0.46165386 reg_l1 48.881477 reg_l2 15.685079
loss 5.3498015
STEP 58 ================================
prereg loss 0.45909113 reg_l1 48.860497 reg_l2 15.674838
loss 5.345141
STEP 59 ================================
prereg loss 0.4568192 reg_l1 48.83899 reg_l2 15.664344
loss 5.3407183
STEP 60 ================================
prereg loss 0.4551437 reg_l1 48.817204 reg_l2 15.653763
loss 5.3368645
STEP 61 ================================
prereg loss 0.45371467 reg_l1 48.79537 reg_l2 15.643253
loss 5.333252
STEP 62 ================================
prereg loss 0.45197567 reg_l1 48.773712 reg_l2 15.632953
loss 5.329347
STEP 63 ================================
prereg loss 0.44989616 reg_l1 48.752224 reg_l2 15.622876
loss 5.325119
STEP 64 ================================
prereg loss 0.447807 reg_l1 48.73096 reg_l2 15.61298
loss 5.320903
STEP 65 ================================
prereg loss 0.44579235 reg_l1 48.709835 reg_l2 15.603184
loss 5.316776
STEP 66 ================================
prereg loss 0.44372457 reg_l1 48.688683 reg_l2 15.593395
loss 5.312593
STEP 67 ================================
prereg loss 0.44150814 reg_l1 48.66732 reg_l2 15.583527
loss 5.3082404
STEP 68 ================================
prereg loss 0.43927857 reg_l1 48.645786 reg_l2 15.57358
loss 5.3038573
STEP 69 ================================
prereg loss 0.4372356 reg_l1 48.62396 reg_l2 15.563552
loss 5.299631
STEP 70 ================================
prereg loss 0.4353816 reg_l1 48.601925 reg_l2 15.553484
loss 5.295574
STEP 71 ================================
prereg loss 0.43353984 reg_l1 48.57975 reg_l2 15.543433
loss 5.291515
STEP 72 ================================
prereg loss 0.4316079 reg_l1 48.557423 reg_l2 15.533413
loss 5.28735
STEP 73 ================================
prereg loss 0.4296625 reg_l1 48.53499 reg_l2 15.523438
loss 5.2831616
STEP 74 ================================
prereg loss 0.4277932 reg_l1 48.51249 reg_l2 15.513485
loss 5.2790422
STEP 75 ================================
prereg loss 0.4259555 reg_l1 48.489834 reg_l2 15.50352
loss 5.2749386
STEP 76 ================================
prereg loss 0.42403784 reg_l1 48.467075 reg_l2 15.493516
loss 5.2707458
STEP 77 ================================
prereg loss 0.42202467 reg_l1 48.44421 reg_l2 15.48349
loss 5.2664456
STEP 78 ================================
prereg loss 0.42001387 reg_l1 48.421158 reg_l2 15.473435
loss 5.26213
STEP 79 ================================
prereg loss 0.41807 reg_l1 48.398026 reg_l2 15.463396
loss 5.2578726
STEP 80 ================================
prereg loss 0.41616338 reg_l1 48.37472 reg_l2 15.453368
loss 5.253636
STEP 81 ================================
prereg loss 0.41424638 reg_l1 48.35132 reg_l2 15.443352
loss 5.2493787
STEP 82 ================================
prereg loss 0.41234764 reg_l1 48.327713 reg_l2 15.4333315
loss 5.245119
STEP 83 ================================
prereg loss 0.41052946 reg_l1 48.303898 reg_l2 15.423285
loss 5.2409196
STEP 84 ================================
prereg loss 0.40882537 reg_l1 48.27984 reg_l2 15.41317
loss 5.2368093
STEP 85 ================================
prereg loss 0.40713823 reg_l1 48.2566 reg_l2 15.403021
loss 5.2327986
STEP 86 ================================
prereg loss 0.40543884 reg_l1 48.233982 reg_l2 15.392824
loss 5.228837
STEP 87 ================================
prereg loss 0.4037463 reg_l1 48.21106 reg_l2 15.382625
loss 5.224852
STEP 88 ================================
prereg loss 0.40206775 reg_l1 48.18789 reg_l2 15.372448
loss 5.2208567
STEP 89 ================================
prereg loss 0.40037894 reg_l1 48.164494 reg_l2 15.362332
loss 5.2168283
STEP 90 ================================
prereg loss 0.39872023 reg_l1 48.1409 reg_l2 15.352256
loss 5.2128105
STEP 91 ================================
prereg loss 0.39708564 reg_l1 48.117058 reg_l2 15.342197
loss 5.2087917
STEP 92 ================================
prereg loss 0.39548495 reg_l1 48.093033 reg_l2 15.332176
loss 5.204788
STEP 93 ================================
prereg loss 0.39391422 reg_l1 48.06878 reg_l2 15.322172
loss 5.2007923
STEP 94 ================================
prereg loss 0.39235252 reg_l1 48.044365 reg_l2 15.312172
loss 5.1967893
STEP 95 ================================
prereg loss 0.3907802 reg_l1 48.01979 reg_l2 15.302216
loss 5.1927595
STEP 96 ================================
prereg loss 0.38919473 reg_l1 47.995102 reg_l2 15.292302
loss 5.1887054
STEP 97 ================================
prereg loss 0.38759392 reg_l1 47.970318 reg_l2 15.282451
loss 5.1846256
STEP 98 ================================
prereg loss 0.38597667 reg_l1 47.94723 reg_l2 15.272659
loss 5.1807
STEP 99 ================================
prereg loss 0.38434997 reg_l1 47.924053 reg_l2 15.262922
loss 5.1767554
STEP 100 ================================
prereg loss 0.38273236 reg_l1 47.90096 reg_l2 15.253223
loss 5.172828
2022-06-26T20:24:39.255

julia> serialize("sparse11-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse11-after-100-steps-opt.ser", opt)

julia> count_interval(sparse11, -0.001f0, 0.001f0)
1

julia> count_interval(sparse11, -0.01f0, 0.01f0)
2

julia> count_interval(sparse11, -0.02f0, 0.02f0)
3

julia> count_interval(sparse11, -0.03f0, 0.03f0)
5

julia> count_interval(sparse11, -0.04f0, 0.04f0)
6

julia> count_interval(sparse11, -0.05f0, 0.05f0)
6

julia> count_interval(sparse11, -0.06f0, 0.06f0)
7

julia> count_interval(sparse11, -0.07f0, 0.07f0)
8

julia> count_interval(sparse11, -0.08f0, 0.08f0)
10

julia> count_interval(sparse11, -0.09f0, 0.09f0)
16

julia> count_interval(sparse11, -0.1f0, 0.1f0)
24

julia> count_interval(sparse11, -0.11f0, 0.11f0)
35

julia> # let's do this; the question is whether to increase regularization

julia> # let's try to do this with the current regularization

julia> sparse12 = sparsecopy(sparse11, 0.11f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 30 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.707525)), "dict-1"=>Dict("input"=>Dict("char"=>0.317691…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("true"=>0.370432, "false"=>0.175551), "norm-2-1"=>Dict("norm"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.314984), "input"=>Dict("char"=>-0.297963)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.273227), "accum-1-1"=>Dict("dict"=>0.168445), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.257779), "accum-1-1"=>Dict("dict"=>0.15265), "accum-1-…
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.251713)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.282654), "dot-2-1"=>Dict("dot"=>-0.296432), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.138245), "norm-2-1"=>Dict("norm"=>0.215249), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.12521), "compare-5-1"=>Dict("true"=>0.341836), "dot-4-1…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.256795), "dot-1-2"=>Dict("dot"=>-0.128332), "dot-2-2"=>…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.124802)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.186876)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.118687)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.272564), "norm-3-1"=>Dict("norm"=>0.173158), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.527487)), "dict-1"=>Dict("input"=>Dict("char"=>-0.110379…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.370026), "norm-1-2"=>Dict("norm"=>0.178194)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.294056), "eos"=>Dict("char"=>-0.337553)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.25668), "norm-3-1"=>Dict("norm"=>0.217501), "dot-3-2"…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.321369), "norm-2-2"=>Dict("norm"=>0.224203), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("compare-4-2"=>Dict("false"=>-0.158797), "dot-2-2"=>Dict("dot"=>0.2482), "norm-3…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.266966), "accum-1-1"=>Dict("dict"=>0.145247), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.116768)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.122621), "compare-4-2"=>Dict("true"=>0.623145)), "di…
  "dot-4-1"     => Dict("dict-2"=>Dict("accum-3-2"=>Dict("dict"=>-0.119014), "dot-2-2"=>Dict("dot"=>0.252334), "dot-2-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.155803)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse12
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 30 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.707525)), "dict-1"=>Dict("input"=>Dict("char"=>0.317691…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("true"=>0.370432, "false"=>0.175551), "norm-2-1"=>Dict("norm"=…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.314984), "input"=>Dict("char"=>-0.297963)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.273227), "accum-1-1"=>Dict("dict"=>0.168445), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.257779), "accum-1-1"=>Dict("dict"=>0.15265), "accum-1-…
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.251713)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.282654), "dot-2-1"=>Dict("dot"=>-0.296432), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.138245), "norm-2-1"=>Dict("norm"=>0.215249), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.12521), "compare-5-1"=>Dict("true"=>0.341836), "dot-4-1…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.256795), "dot-1-2"=>Dict("dot"=>-0.128332), "dot-2-2"=>…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.124802)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.186876)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.118687)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.272564), "norm-3-1"=>Dict("norm"=>0.173158), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.527487)), "dict-1"=>Dict("input"=>Dict("char"=>-0.110379…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.370026), "norm-1-2"=>Dict("norm"=>0.178194)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.294056), "eos"=>Dict("char"=>-0.337553)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.25668), "norm-3-1"=>Dict("norm"=>0.217501), "dot-3-2"…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.321369), "norm-2-2"=>Dict("norm"=>0.224203), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("compare-4-2"=>Dict("false"=>-0.158797), "dot-2-2"=>Dict("dot"=>0.2482), "norm-3…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.266966), "accum-1-1"=>Dict("dict"=>0.145247), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.116768)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.122621), "compare-4-2"=>Dict("true"=>0.623145)), "di…
  "dot-4-1"     => Dict("dict-2"=>Dict("accum-3-2"=>Dict("dict"=>-0.119014), "dot-2-2"=>Dict("dot"=>0.252334), "dot-2-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.155803)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
185

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T22:34:04.543
STEP 1 ================================
prereg loss 34.411175 reg_l1 45.07741 reg_l2 14.985773
loss 38.918915
STEP 2 ================================
prereg loss 30.655678 reg_l1 45.154415 reg_l2 15.029656
loss 35.17112
STEP 3 ================================
prereg loss 26.983046 reg_l1 45.23146 reg_l2 15.073883
loss 31.506191
STEP 4 ================================
prereg loss 23.471773 reg_l1 45.308117 reg_l2 15.118171
loss 28.002584
STEP 5 ================================
prereg loss 20.233297 reg_l1 45.38485 reg_l2 15.162706
loss 24.771782
STEP 6 ================================
prereg loss 17.18194 reg_l1 45.460316 reg_l2 15.206842
loss 21.727972
STEP 7 ================================
prereg loss 14.498701 reg_l1 45.534718 reg_l2 15.250717
loss 19.052174
STEP 8 ================================
prereg loss 12.221661 reg_l1 45.6083 reg_l2 15.294397
loss 16.78249
STEP 9 ================================
prereg loss 10.377911 reg_l1 45.681023 reg_l2 15.337846
loss 14.9460125
STEP 10 ================================
prereg loss 8.9740715 reg_l1 45.752495 reg_l2 15.38079
loss 13.549321
STEP 11 ================================
prereg loss 8.012018 reg_l1 45.82199 reg_l2 15.422831
loss 12.594217
STEP 12 ================================
prereg loss 7.45495 reg_l1 45.88856 reg_l2 15.46335
loss 12.043806
STEP 13 ================================
prereg loss 7.2292504 reg_l1 45.950916 reg_l2 15.501563
loss 11.824343
STEP 14 ================================
prereg loss 7.2331543 reg_l1 46.00774 reg_l2 15.536698
loss 11.833929
STEP 15 ================================
prereg loss 7.351283 reg_l1 46.05793 reg_l2 15.568097
loss 11.957076
STEP 16 ================================
prereg loss 7.4779534 reg_l1 46.10078 reg_l2 15.595316
loss 12.088032
STEP 17 ================================
prereg loss 7.5350833 reg_l1 46.135918 reg_l2 15.618187
loss 12.148675
STEP 18 ================================
prereg loss 7.4858394 reg_l1 46.163387 reg_l2 15.636733
loss 12.102179
STEP 19 ================================
prereg loss 7.32965 reg_l1 46.183445 reg_l2 15.651118
loss 11.947994
STEP 20 ================================
prereg loss 7.0910206 reg_l1 46.19635 reg_l2 15.661552
loss 11.710655
STEP 21 ================================
prereg loss 6.7989917 reg_l1 46.20236 reg_l2 15.668206
loss 11.419228
STEP 22 ================================
prereg loss 6.4642353 reg_l1 46.20228 reg_l2 15.671507
loss 11.084463
STEP 23 ================================
prereg loss 6.142337 reg_l1 46.197052 reg_l2 15.672014
loss 10.762042
STEP 24 ================================
prereg loss 5.86741 reg_l1 46.187542 reg_l2 15.670234
loss 10.486164
STEP 25 ================================
prereg loss 5.660044 reg_l1 46.17449 reg_l2 15.6666
loss 10.2774935
STEP 26 ================================
prereg loss 5.525928 reg_l1 46.15845 reg_l2 15.661471
loss 10.141773
STEP 27 ================================
prereg loss 5.4574175 reg_l1 46.139977 reg_l2 15.655144
loss 10.071415
STEP 28 ================================
prereg loss 5.437616 reg_l1 46.119476 reg_l2 15.647845
loss 10.049563
STEP 29 ================================
prereg loss 5.445807 reg_l1 46.09729 reg_l2 15.639765
loss 10.055536
STEP 30 ================================
prereg loss 5.462375 reg_l1 46.07378 reg_l2 15.63111
loss 10.069754
STEP 31 ================================
prereg loss 5.471815 reg_l1 46.0493 reg_l2 15.622049
loss 10.076746
STEP 32 ================================
prereg loss 5.463672 reg_l1 46.024082 reg_l2 15.612712
loss 10.06608
STEP 33 ================================
prereg loss 5.43215 reg_l1 45.99831 reg_l2 15.603221
loss 10.0319805
STEP 34 ================================
prereg loss 5.375327 reg_l1 45.97216 reg_l2 15.593622
loss 9.972544
STEP 35 ================================
prereg loss 5.2943544 reg_l1 45.94565 reg_l2 15.583959
loss 9.88892
STEP 36 ================================
prereg loss 5.19294 reg_l1 45.918842 reg_l2 15.574232
loss 9.784824
STEP 37 ================================
prereg loss 5.076857 reg_l1 45.89173 reg_l2 15.564431
loss 9.666031
STEP 38 ================================
prereg loss 4.953351 reg_l1 45.864292 reg_l2 15.554513
loss 9.53978
STEP 39 ================================
prereg loss 4.8302355 reg_l1 45.83642 reg_l2 15.544436
loss 9.4138775
STEP 40 ================================
prereg loss 4.714687 reg_l1 45.808094 reg_l2 15.534134
loss 9.295496
STEP 41 ================================
prereg loss 4.613173 reg_l1 45.77912 reg_l2 15.523521
loss 9.191086
STEP 42 ================================
prereg loss 4.5283036 reg_l1 45.749424 reg_l2 15.512531
loss 9.103247
STEP 43 ================================
prereg loss 4.4585104 reg_l1 45.71884 reg_l2 15.50108
loss 9.030395
STEP 44 ================================
prereg loss 4.401279 reg_l1 45.687313 reg_l2 15.48912
loss 8.970011
STEP 45 ================================
prereg loss 4.355215 reg_l1 45.654827 reg_l2 15.476608
loss 8.920698
STEP 46 ================================
prereg loss 4.3186707 reg_l1 45.62134 reg_l2 15.463533
loss 8.880805
STEP 47 ================================
prereg loss 4.291856 reg_l1 45.587025 reg_l2 15.449961
loss 8.850558
STEP 48 ================================
prereg loss 4.2600245 reg_l1 45.552166 reg_l2 15.43598
loss 8.815241
STEP 49 ================================
prereg loss 4.2202287 reg_l1 45.516773 reg_l2 15.421635
loss 8.771906
STEP 50 ================================
prereg loss 4.175185 reg_l1 45.480957 reg_l2 15.406978
loss 8.723281
STEP 51 ================================
prereg loss 4.1279345 reg_l1 45.444828 reg_l2 15.39208
loss 8.672417
STEP 52 ================================
prereg loss 4.080949 reg_l1 45.40855 reg_l2 15.3770685
loss 8.621803
STEP 53 ================================
prereg loss 4.0364666 reg_l1 45.372326 reg_l2 15.362032
loss 8.5737
STEP 54 ================================
prereg loss 3.9961298 reg_l1 45.33632 reg_l2 15.347112
loss 8.529761
STEP 55 ================================
prereg loss 3.9613128 reg_l1 45.300743 reg_l2 15.332439
loss 8.491387
STEP 56 ================================
prereg loss 3.9330702 reg_l1 45.265736 reg_l2 15.318135
loss 8.459644
STEP 57 ================================
prereg loss 3.9115317 reg_l1 45.23139 reg_l2 15.304309
loss 8.434671
STEP 58 ================================
prereg loss 3.8955934 reg_l1 45.197926 reg_l2 15.291055
loss 8.415386
STEP 59 ================================
prereg loss 3.883097 reg_l1 45.165386 reg_l2 15.278451
loss 8.399635
STEP 60 ================================
prereg loss 3.8713853 reg_l1 45.13384 reg_l2 15.266569
loss 8.384769
STEP 61 ================================
prereg loss 3.8581197 reg_l1 45.103325 reg_l2 15.255422
loss 8.368452
STEP 62 ================================
prereg loss 3.8418827 reg_l1 45.07383 reg_l2 15.245007
loss 8.349266
STEP 63 ================================
prereg loss 3.8224347 reg_l1 45.04535 reg_l2 15.23532
loss 8.32697
STEP 64 ================================
prereg loss 3.800414 reg_l1 45.017796 reg_l2 15.226311
loss 8.302194
STEP 65 ================================
prereg loss 3.7768917 reg_l1 44.991085 reg_l2 15.217921
loss 8.276001
STEP 66 ================================
prereg loss 3.7532673 reg_l1 44.965218 reg_l2 15.210086
loss 8.249789
STEP 67 ================================
prereg loss 3.730793 reg_l1 44.940002 reg_l2 15.202729
loss 8.224793
STEP 68 ================================
prereg loss 3.7097316 reg_l1 44.915367 reg_l2 15.195781
loss 8.201268
STEP 69 ================================
prereg loss 3.69066 reg_l1 44.89129 reg_l2 15.189173
loss 8.179789
STEP 70 ================================
prereg loss 3.6741726 reg_l1 44.867634 reg_l2 15.182837
loss 8.160936
STEP 71 ================================
prereg loss 3.662613 reg_l1 44.84447 reg_l2 15.176775
loss 8.14706
STEP 72 ================================
prereg loss 3.6523573 reg_l1 44.821735 reg_l2 15.170963
loss 8.134531
STEP 73 ================================
prereg loss 3.6422153 reg_l1 44.799408 reg_l2 15.165334
loss 8.122156
STEP 74 ================================
prereg loss 3.6317298 reg_l1 44.777428 reg_l2 15.15988
loss 8.109472
STEP 75 ================================
prereg loss 3.6206596 reg_l1 44.755722 reg_l2 15.154572
loss 8.096231
STEP 76 ================================
prereg loss 3.6089957 reg_l1 44.734295 reg_l2 15.149404
loss 8.082425
STEP 77 ================================
prereg loss 3.5969236 reg_l1 44.713135 reg_l2 15.144373
loss 8.068237
STEP 78 ================================
prereg loss 3.5847619 reg_l1 44.692234 reg_l2 15.139476
loss 8.053986
STEP 79 ================================
prereg loss 3.5727315 reg_l1 44.671593 reg_l2 15.134741
loss 8.03989
STEP 80 ================================
prereg loss 3.5610309 reg_l1 44.651237 reg_l2 15.130164
loss 8.0261545
STEP 81 ================================
prereg loss 3.5498154 reg_l1 44.63118 reg_l2 15.1257715
loss 8.012934
STEP 82 ================================
prereg loss 3.5391119 reg_l1 44.611412 reg_l2 15.121561
loss 8.000253
STEP 83 ================================
prereg loss 3.5288296 reg_l1 44.591908 reg_l2 15.117551
loss 7.9880204
STEP 84 ================================
prereg loss 3.5187535 reg_l1 44.572693 reg_l2 15.11373
loss 7.9760227
STEP 85 ================================
prereg loss 3.5086727 reg_l1 44.55373 reg_l2 15.110114
loss 7.964046
STEP 86 ================================
prereg loss 3.498397 reg_l1 44.535034 reg_l2 15.106703
loss 7.9519005
STEP 87 ================================
prereg loss 3.4877949 reg_l1 44.516582 reg_l2 15.103493
loss 7.939453
STEP 88 ================================
prereg loss 3.4769504 reg_l1 44.498318 reg_l2 15.100455
loss 7.9267826
STEP 89 ================================
prereg loss 3.4658349 reg_l1 44.480267 reg_l2 15.09759
loss 7.9138613
STEP 90 ================================
prereg loss 3.4545007 reg_l1 44.462406 reg_l2 15.09487
loss 7.9007416
STEP 91 ================================
prereg loss 3.4430485 reg_l1 44.44469 reg_l2 15.092269
loss 7.8875175
STEP 92 ================================
prereg loss 3.4316165 reg_l1 44.42706 reg_l2 15.089754
loss 7.874323
STEP 93 ================================
prereg loss 3.4203064 reg_l1 44.409466 reg_l2 15.087278
loss 7.861253
STEP 94 ================================
prereg loss 3.4094205 reg_l1 44.391876 reg_l2 15.084836
loss 7.848608
STEP 95 ================================
prereg loss 3.3987093 reg_l1 44.374275 reg_l2 15.08239
loss 7.836137
STEP 96 ================================
prereg loss 3.3881526 reg_l1 44.35661 reg_l2 15.079911
loss 7.8238134
STEP 97 ================================
prereg loss 3.378056 reg_l1 44.33884 reg_l2 15.077391
loss 7.81194
STEP 98 ================================
prereg loss 3.368897 reg_l1 44.321125 reg_l2 15.074861
loss 7.8010097
STEP 99 ================================
prereg loss 3.3597136 reg_l1 44.30339 reg_l2 15.072328
loss 7.790053
STEP 100 ================================
prereg loss 3.350562 reg_l1 44.285656 reg_l2 15.069786
loss 7.7791276
2022-06-26T22:46:17.210

julia> # this could have used a more agressive regularization

julia> # and perhaps we pruned too much at once

julia> serialize("sparse12-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse12-after-100-steps-opt.ser", opt)

julia> count_interval(sparse12, -0.001f0, 0.001f0)
0

julia> count_interval(sparse12, -0.01f0, 0.01f0)
0

julia> count_interval(sparse12, -0.02f0, 0.02f0)
4

julia> count_interval(sparse12, -0.03f0, 0.03f0)
7

julia> count_interval(sparse12, -0.04f0, 0.04f0)
7

julia> count_interval(sparse12, -0.05f0, 0.05f0)
10

julia> count_interval(sparse12, -0.06f0, 0.06f0)
12

julia> count_interval(sparse12, -0.07f0, 0.07f0)
13

julia> count_interval(sparse12, -0.08f0, 0.08f0)
14

julia> count_interval(sparse12, -0.09f0, 0.09f0)
16

julia> count_interval(sparse12, -0.1f0, 0.1f0)
19

julia> count_interval(sparse12, -0.11f0, 0.11f0)
23

julia> count_interval(sparse12, -0.12f0, 0.12f0)
27

julia> count_interval(sparse12, -0.115f0, 0.115f0)
25

julia> close(io)
```

We are going to double the regularization:

```
$ diff loss.jl loss-original.jl
67c67
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2

$ diff test.jl test-original.jl
36c36
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse12-after-100-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")

```

and the old sparsification schedule looks like this, but might
actually be too aggressive (a smaller-size network might need a
different strategy; a training iteration is cheaper, but fewer
degrees of freedom are available):

```
> python
>>> 160*0.87
139.19999999999999
>>> 160*(0.87**2)
121.104
>>> 160*(0.87**3)
105.36048
>>> 160*(0.87**4)
91.663617599999995
>>> 160*(0.87**5)
79.747347312000002
>>> 160*(0.87**6)
69.380192161440007
>>> 160*(0.87**7)
60.360767180452797
>>> 160*(0.87**8)
52.513867446993942
>>> 160*(0.87**9)
45.687064678884724
>>> 160*(0.87**10)
39.747746270629712
>>> 160*(0.87**11)
34.580539255447846
```

Here is the new console (at stage 16, 109 weights, I stopped and decided to train it well,
see if it still has good properties and such):

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 30 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.714486)), "dict-1"=>Dict("input"=>Dict("char"=>0.29105)…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.440476), "norm-2-1"=>Dict("norm"=>0.509713), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.332002), "input"=>Dict("char"=>-0.277309)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.24494), "accum-1-1"=>Dict("dict"=>0.0684445), "accum-1-…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.260291), "accum-1-1"=>Dict("dict"=>0.0526496), "accum-…
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.265958)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.288049), "dot-2-1"=>Dict("dot"=>-0.302829), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18648), "norm-2-1"=>Dict("norm"=>0.179415), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.123568), "compare-5-1"=>Dict("true"=>0.357151), "dot-4-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.331208), "dot-1-2"=>Dict("dot"=>-0.0246763), "dot-2-2"=…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.148093)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.219738)))
  "norm-4-1"    => Dict("dict"=>Dict("accum-3-2"=>Dict("dict"=>0.0186872)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.291283), "norm-3-1"=>Dict("norm"=>0.192904), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.426058)), "dict-1"=>Dict("input"=>Dict("char"=>-0.012021…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.323833), "norm-1-2"=>Dict("norm"=>0.219184)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.306382), "eos"=>Dict("char"=>-0.350704)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.257256), "norm-3-1"=>Dict("norm"=>0.219852), "dot-3-2…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.304261), "norm-2-2"=>Dict("norm"=>0.219463), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("norm-3-2"=>Dict("norm"=>-0.10768), "dot-2-2"=>Dict("dot"=>0.285198), "norm-3-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.27499), "accum-1-1"=>Dict("dict"=>0.0452468), "accum-1…
  "dot-4-2"     => Dict("dict-2"=>Dict("compare-3-2"=>Dict("true"=>0.0167676)))
  "dot-5-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>0.0226205), "compare-4-2"=>Dict("true"=>0.522924)), "d…
  "dot-4-1"     => Dict("dict-2"=>Dict("accum-3-2"=>Dict("dict"=>-0.0190136), "dot-2-2"=>Dict("dot"=>0.262999), "dot-2-…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.179676)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
185

julia> sparse13 = sparsecopy(trainable["network_matrix"], 0.115f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 28 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.714486)), "dict-1"=>Dict("input"=>Dict("char"=>0.29105)…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("true"=>0.28059, "false"=>0.200048), "norm-2-1"=>Dict("norm"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.332002), "input"=>Dict("char"=>-0.277309)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.24494), "accum-1-2"=>Dict("dict"=>-0.19687), "input"=>D…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.260291), "accum-1-2"=>Dict("dict"=>0.298964)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.265958)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.288049), "dot-2-1"=>Dict("dot"=>-0.302829), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18648), "norm-2-1"=>Dict("norm"=>0.179415), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.123568), "compare-5-1"=>Dict("true"=>0.357151), "dot-4-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.331208), "norm-3-2"=>Dict("norm"=>0.157658), "dot-2-2"=…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.148093)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.219738)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.291283), "norm-3-1"=>Dict("norm"=>0.192904), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.426058)), "dict-1"=>Dict("eos"=>Dict("char"=>0.195947)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.323833), "norm-1-2"=>Dict("norm"=>0.219184)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.306382), "eos"=>Dict("char"=>-0.350704)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.257256), "norm-3-1"=>Dict("norm"=>0.219852), "dot-3-2…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.304261), "norm-2-2"=>Dict("norm"=>0.219463), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.285198), "norm-3-1"=>Dict("norm"=>-0.115674), "dot-2-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.27499), "accum-1-2"=>Dict("dict"=>0.300418), "input"=>…
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.522924)), "dict-1"=>Dict("compare-3-1"=>Dict("fals…
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.262999), "dot-2-1"=>Dict("dot"=>-0.120582), "dot-3-1"=>…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.179676)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.168253), "dot-2-2"=>Dict("dot"=>0.315473), "norm-2-2"=…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.519119)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse13
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 28 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.714486)), "dict-1"=>Dict("input"=>Dict("char"=>0.29105)…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("true"=>0.28059, "false"=>0.200048), "norm-2-1"=>Dict("norm"=>…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.332002), "input"=>Dict("char"=>-0.277309)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.24494), "accum-1-2"=>Dict("dict"=>-0.19687), "input"=>D…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.260291), "accum-1-2"=>Dict("dict"=>0.298964)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.265958)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.288049), "dot-2-1"=>Dict("dot"=>-0.302829), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18648), "norm-2-1"=>Dict("norm"=>0.179415), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-1"=>Dict("dot"=>0.123568), "compare-5-1"=>Dict("true"=>0.357151), "dot-4-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.331208), "norm-3-2"=>Dict("norm"=>0.157658), "dot-2-2"=…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.148093)), "dict-1"=>Dict("const_1"=>Dict("const_1"…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.219738)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.291283), "norm-3-1"=>Dict("norm"=>0.192904), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.426058)), "dict-1"=>Dict("eos"=>Dict("char"=>0.195947)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.323833), "norm-1-2"=>Dict("norm"=>0.219184)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.306382), "eos"=>Dict("char"=>-0.350704)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.257256), "norm-3-1"=>Dict("norm"=>0.219852), "dot-3-2…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.304261), "norm-2-2"=>Dict("norm"=>0.219463), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.285198), "norm-3-1"=>Dict("norm"=>-0.115674), "dot-2-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.27499), "accum-1-2"=>Dict("dict"=>0.300418), "input"=>…
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.522924)), "dict-1"=>Dict("compare-3-1"=>Dict("fals…
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.262999), "dot-2-1"=>Dict("dot"=>-0.120582), "dot-3-1"=>…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.179676)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.168253), "dot-2-2"=>Dict("dot"=>0.315473), "norm-2-2"=…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.519119)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
160

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T23:24:23.996
STEP 1 ================================
prereg loss 9.586601 reg_l1 42.633533 reg_l2 14.929739
loss 18.113308
STEP 2 ================================
prereg loss 7.5305533 reg_l1 42.585526 reg_l2 14.89806
loss 16.047659
STEP 3 ================================
prereg loss 6.442701 reg_l1 42.53859 reg_l2 14.869965
loss 14.950418
STEP 4 ================================
prereg loss 6.0382457 reg_l1 42.49195 reg_l2 14.842906
loss 14.536636
STEP 5 ================================
prereg loss 5.939406 reg_l1 42.450798 reg_l2 14.819152
loss 14.429565
STEP 6 ================================
prereg loss 5.9708085 reg_l1 42.418034 reg_l2 14.800331
loss 14.454416
STEP 7 ================================
prereg loss 6.0477943 reg_l1 42.39465 reg_l2 14.787078
loss 14.526725
STEP 8 ================================
prereg loss 6.090002 reg_l1 42.38034 reg_l2 14.779247
loss 14.566071
STEP 9 ================================
prereg loss 6.078769 reg_l1 42.37403 reg_l2 14.776161
loss 14.5535755
STEP 10 ================================
prereg loss 6.0137043 reg_l1 42.37447 reg_l2 14.7769985
loss 14.488599
STEP 11 ================================
prereg loss 5.8915987 reg_l1 42.38051 reg_l2 14.78094
loss 14.367701
STEP 12 ================================
prereg loss 5.712959 reg_l1 42.39106 reg_l2 14.787335
loss 14.191172
STEP 13 ================================
prereg loss 5.4919915 reg_l1 42.405155 reg_l2 14.795597
loss 13.973022
STEP 14 ================================
prereg loss 5.271332 reg_l1 42.421753 reg_l2 14.805202
loss 13.755683
STEP 15 ================================
prereg loss 5.078622 reg_l1 42.43979 reg_l2 14.8155155
loss 13.56658
STEP 16 ================================
prereg loss 4.937765 reg_l1 42.458073 reg_l2 14.825941
loss 13.429379
STEP 17 ================================
prereg loss 4.8567886 reg_l1 42.47532 reg_l2 14.835801
loss 13.351852
STEP 18 ================================
prereg loss 4.823994 reg_l1 42.490467 reg_l2 14.844499
loss 13.322088
STEP 19 ================================
prereg loss 4.8127613 reg_l1 42.502563 reg_l2 14.851492
loss 13.313274
STEP 20 ================================
prereg loss 4.792903 reg_l1 42.511013 reg_l2 14.856454
loss 13.295106
STEP 21 ================================
prereg loss 4.743209 reg_l1 42.5156 reg_l2 14.859225
loss 13.246328
STEP 22 ================================
prereg loss 4.6579247 reg_l1 42.516388 reg_l2 14.8598585
loss 13.161202
STEP 23 ================================
prereg loss 4.545314 reg_l1 42.51378 reg_l2 14.858559
loss 13.04807
STEP 24 ================================
prereg loss 4.4215517 reg_l1 42.508415 reg_l2 14.8556795
loss 12.923235
STEP 25 ================================
prereg loss 4.3034673 reg_l1 42.500965 reg_l2 14.851619
loss 12.803661
STEP 26 ================================
prereg loss 4.2031507 reg_l1 42.492268 reg_l2 14.846832
loss 12.701605
STEP 27 ================================
prereg loss 4.1253886 reg_l1 42.483154 reg_l2 14.84179
loss 12.622019
STEP 28 ================================
prereg loss 4.0663323 reg_l1 42.47444 reg_l2 14.836934
loss 12.56122
STEP 29 ================================
prereg loss 4.018935 reg_l1 42.466774 reg_l2 14.83263
loss 12.51229
STEP 30 ================================
prereg loss 3.9756205 reg_l1 42.460644 reg_l2 14.829178
loss 12.46775
STEP 31 ================================
prereg loss 3.9334345 reg_l1 42.456406 reg_l2 14.8267765
loss 12.424716
STEP 32 ================================
prereg loss 3.8851647 reg_l1 42.454117 reg_l2 14.8255005
loss 12.375988
STEP 33 ================================
prereg loss 3.827227 reg_l1 42.453587 reg_l2 14.8252735
loss 12.3179455
STEP 34 ================================
prereg loss 3.7613165 reg_l1 42.454506 reg_l2 14.825919
loss 12.252217
STEP 35 ================================
prereg loss 3.692369 reg_l1 42.456387 reg_l2 14.827141
loss 12.183647
STEP 36 ================================
prereg loss 3.62661 reg_l1 42.45866 reg_l2 14.828644
loss 12.118342
STEP 37 ================================
prereg loss 3.5712986 reg_l1 42.460762 reg_l2 14.830068
loss 12.063451
STEP 38 ================================
prereg loss 3.525933 reg_l1 42.462105 reg_l2 14.83107
loss 12.018354
STEP 39 ================================
prereg loss 3.4895463 reg_l1 42.462204 reg_l2 14.83138
loss 11.981987
STEP 40 ================================
prereg loss 3.4588115 reg_l1 42.460712 reg_l2 14.830768
loss 11.950954
STEP 41 ================================
prereg loss 3.4292405 reg_l1 42.457302 reg_l2 14.829084
loss 11.920701
STEP 42 ================================
prereg loss 3.3971858 reg_l1 42.45189 reg_l2 14.826282
loss 11.887564
STEP 43 ================================
prereg loss 3.361773 reg_l1 42.444614 reg_l2 14.822397
loss 11.850697
STEP 44 ================================
prereg loss 3.3240595 reg_l1 42.43567 reg_l2 14.81758
loss 11.811193
STEP 45 ================================
prereg loss 3.2871685 reg_l1 42.425404 reg_l2 14.812021
loss 11.772249
STEP 46 ================================
prereg loss 3.2542975 reg_l1 42.414207 reg_l2 14.805969
loss 11.737139
STEP 47 ================================
prereg loss 3.2276244 reg_l1 42.40254 reg_l2 14.799661
loss 11.708132
STEP 48 ================================
prereg loss 3.206718 reg_l1 42.39075 reg_l2 14.7933445
loss 11.684868
STEP 49 ================================
prereg loss 3.1894877 reg_l1 42.379208 reg_l2 14.787213
loss 11.665329
STEP 50 ================================
prereg loss 3.1736474 reg_l1 42.368225 reg_l2 14.78142
loss 11.647293
STEP 51 ================================
prereg loss 3.1550927 reg_l1 42.357857 reg_l2 14.776059
loss 11.626665
STEP 52 ================================
prereg loss 3.1337519 reg_l1 42.348145 reg_l2 14.771136
loss 11.603381
STEP 53 ================================
prereg loss 3.1118727 reg_l1 42.338955 reg_l2 14.76654
loss 11.579664
STEP 54 ================================
prereg loss 3.0911665 reg_l1 42.330044 reg_l2 14.762159
loss 11.557176
STEP 55 ================================
prereg loss 3.0723808 reg_l1 42.321175 reg_l2 14.757836
loss 11.536616
STEP 56 ================================
prereg loss 3.055678 reg_l1 42.312042 reg_l2 14.753405
loss 11.518087
STEP 57 ================================
prereg loss 3.0407887 reg_l1 42.302296 reg_l2 14.748693
loss 11.501248
STEP 58 ================================
prereg loss 3.0271745 reg_l1 42.291733 reg_l2 14.743567
loss 11.485521
STEP 59 ================================
prereg loss 3.0142615 reg_l1 42.28017 reg_l2 14.737916
loss 11.470296
STEP 60 ================================
prereg loss 3.0016236 reg_l1 42.26751 reg_l2 14.731689
loss 11.455126
STEP 61 ================================
prereg loss 2.9891088 reg_l1 42.253757 reg_l2 14.724918
loss 11.43986
STEP 62 ================================
prereg loss 2.9768462 reg_l1 42.23907 reg_l2 14.717675
loss 11.42466
STEP 63 ================================
prereg loss 2.9651828 reg_l1 42.223644 reg_l2 14.710063
loss 11.409912
STEP 64 ================================
prereg loss 2.9544692 reg_l1 42.207745 reg_l2 14.702256
loss 11.396019
STEP 65 ================================
prereg loss 2.9448159 reg_l1 42.191597 reg_l2 14.694377
loss 11.383135
STEP 66 ================================
prereg loss 2.9359493 reg_l1 42.17554 reg_l2 14.686603
loss 11.3710575
STEP 67 ================================
prereg loss 2.927327 reg_l1 42.15971 reg_l2 14.679028
loss 11.359269
STEP 68 ================================
prereg loss 2.9184296 reg_l1 42.144238 reg_l2 14.671718
loss 11.347277
STEP 69 ================================
prereg loss 2.908817 reg_l1 42.12915 reg_l2 14.6647005
loss 11.334647
STEP 70 ================================
prereg loss 2.898605 reg_l1 42.114372 reg_l2 14.657942
loss 11.32148
STEP 71 ================================
prereg loss 2.8882 reg_l1 42.099834 reg_l2 14.651353
loss 11.3081665
STEP 72 ================================
prereg loss 2.8780806 reg_l1 42.08531 reg_l2 14.644851
loss 11.295143
STEP 73 ================================
prereg loss 2.8685486 reg_l1 42.07061 reg_l2 14.638314
loss 11.282671
STEP 74 ================================
prereg loss 2.8596175 reg_l1 42.05555 reg_l2 14.631634
loss 11.270727
STEP 75 ================================
prereg loss 2.8511348 reg_l1 42.04001 reg_l2 14.624745
loss 11.259136
STEP 76 ================================
prereg loss 2.8429365 reg_l1 42.023903 reg_l2 14.617596
loss 11.247717
STEP 77 ================================
prereg loss 2.8349402 reg_l1 42.007042 reg_l2 14.610082
loss 11.236348
STEP 78 ================================
prereg loss 2.8271601 reg_l1 41.98952 reg_l2 14.602252
loss 11.225064
STEP 79 ================================
prereg loss 2.8196645 reg_l1 41.971455 reg_l2 14.594166
loss 11.213955
STEP 80 ================================
prereg loss 2.8125277 reg_l1 41.953014 reg_l2 14.585927
loss 11.203131
STEP 81 ================================
prereg loss 2.8056757 reg_l1 41.93434 reg_l2 14.577614
loss 11.192544
STEP 82 ================================
prereg loss 2.7989767 reg_l1 41.915592 reg_l2 14.569315
loss 11.182096
STEP 83 ================================
prereg loss 2.7922792 reg_l1 41.896896 reg_l2 14.561094
loss 11.1716585
STEP 84 ================================
prereg loss 2.7854788 reg_l1 41.878277 reg_l2 14.552998
loss 11.161134
STEP 85 ================================
prereg loss 2.7785294 reg_l1 41.85996 reg_l2 14.545099
loss 11.150521
STEP 86 ================================
prereg loss 2.7714102 reg_l1 41.84185 reg_l2 14.537395
loss 11.13978
STEP 87 ================================
prereg loss 2.7642512 reg_l1 41.823917 reg_l2 14.529849
loss 11.129034
STEP 88 ================================
prereg loss 2.757046 reg_l1 41.80613 reg_l2 14.522404
loss 11.118272
STEP 89 ================================
prereg loss 2.7499602 reg_l1 41.788292 reg_l2 14.515
loss 11.107618
STEP 90 ================================
prereg loss 2.7431145 reg_l1 41.770382 reg_l2 14.507602
loss 11.097191
STEP 91 ================================
prereg loss 2.7365003 reg_l1 41.752346 reg_l2 14.50015
loss 11.086969
STEP 92 ================================
prereg loss 2.7300603 reg_l1 41.734066 reg_l2 14.492636
loss 11.076874
STEP 93 ================================
prereg loss 2.723747 reg_l1 41.715603 reg_l2 14.485049
loss 11.066868
STEP 94 ================================
prereg loss 2.717588 reg_l1 41.696762 reg_l2 14.477286
loss 11.05694
STEP 95 ================================
prereg loss 2.711584 reg_l1 41.677586 reg_l2 14.469377
loss 11.047101
STEP 96 ================================
prereg loss 2.7057467 reg_l1 41.65817 reg_l2 14.461375
loss 11.03738
STEP 97 ================================
prereg loss 2.700036 reg_l1 41.638565 reg_l2 14.453314
loss 11.027749
STEP 98 ================================
prereg loss 2.6944056 reg_l1 41.618885 reg_l2 14.445245
loss 11.018183
STEP 99 ================================
prereg loss 2.6887527 reg_l1 41.59939 reg_l2 14.437306
loss 11.008631
STEP 100 ================================
prereg loss 2.6830602 reg_l1 41.580025 reg_l2 14.42951
loss 10.999065
2022-06-26T23:35:35.496

julia> serialize("sparse13-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse13-after-100-steps-opt.ser", opt)

julia> count_interval(sparse13, -0.01f0, 0.01f0)
0

julia> count_interval(sparse13, -0.02f0, 0.02f0)
0

julia> count_interval(sparse13, -0.03f0, 0.03f0)
0

julia> count_interval(sparse13, -0.04f0, 0.04f0)
0

julia> count_interval(sparse13, -0.05f0, 0.05f0)
0

julia> count_interval(sparse13, -0.06f0, 0.06f0)
0

julia> count_interval(sparse13, -0.07f0, 0.07f0)
0

julia> count_interval(sparse13, -0.08f0, 0.08f0)
0

julia> count_interval(sparse13, -0.09f0, 0.09f0)
1

julia> count_interval(sparse13, -0.10f0, 0.10f0)
3

julia> count_interval(sparse13, -0.11f0, 0.11f0)
4

julia> count_interval(sparse13, -0.12f0, 0.12f0)
4

julia> count_interval(sparse13, -0.13f0, 0.13f0)
9

julia> count_interval(sparse13, -0.14f0, 0.14f0)
14

julia> count_interval(sparse13, -0.15f0, 0.15f0)
20

julia> # it's quite large, but we can try nonetheless

julia> # we can always backtrack

julia> sparse14 = sparsecopy(trainable["network_matrix"], 0.15f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 28 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.687032)), "dict-1"=>Dict("input"=>Dict("char"=>0.227481…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.420494), "norm-2-1"=>Dict("norm"=>0.485099), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.34062), "input"=>Dict("char"=>-0.28699)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.151663), "accum-1-2"=>Dict("dict"=>-0.186832), "eos"=>D…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.276603), "accum-1-2"=>Dict("dict"=>0.317016)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.238456)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.29241), "dot-2-1"=>Dict("dot"=>-0.306359), "accum-2-2"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18537), "norm-2-1"=>Dict("norm"=>0.180457), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.208602), "compare-5-1"=>Dict("true"=>0.40066), "dot-5-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.245451), "compare-4-2"=>Dict("false"=>0.2389), "const_1…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.150195)))
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.229896)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.293975), "norm-3-1"=>Dict("norm"=>0.200337), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.32606)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.250354), "norm-1-2"=>Dict("norm"=>0.216701)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.307617), "eos"=>Dict("char"=>-0.365921)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.280737), "norm-3-1"=>Dict("norm"=>0.245952), "compare…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.308869), "norm-2-2"=>Dict("norm"=>0.174517), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.27412), "norm-3-1"=>Dict("norm"=>-0.155631), "compare-4…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.286968), "accum-1-2"=>Dict("dict"=>0.314088)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.422926)), "dict-1"=>Dict("compare-4-2"=>Dict("true…
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.272652), "dot-3-1"=>Dict("dot"=>0.199894), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.183275)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.317679), "norm-2-2"=>Dict("norm"=>0.165936), "dot-2-1"=…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.542306)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse14
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 28 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.687032)), "dict-1"=>Dict("input"=>Dict("char"=>0.227481…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.420494), "norm-2-1"=>Dict("norm"=>0.485099), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.34062), "input"=>Dict("char"=>-0.28699)))
  "dot-2-2"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.151663), "accum-1-2"=>Dict("dict"=>-0.186832), "eos"=>D…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.276603), "accum-1-2"=>Dict("dict"=>0.317016)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.238456)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.29241), "dot-2-1"=>Dict("dot"=>-0.306359), "accum-2-2"=…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.18537), "norm-2-1"=>Dict("norm"=>0.180457), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.208602), "compare-5-1"=>Dict("true"=>0.40066), "dot-5-…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.245451), "compare-4-2"=>Dict("false"=>0.2389), "const_1…
  "compare-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.150195)))
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.229896)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.293975), "norm-3-1"=>Dict("norm"=>0.200337), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.32606)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.250354), "norm-1-2"=>Dict("norm"=>0.216701)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.307617), "eos"=>Dict("char"=>-0.365921)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.280737), "norm-3-1"=>Dict("norm"=>0.245952), "compare…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.308869), "norm-2-2"=>Dict("norm"=>0.174517), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.27412), "norm-3-1"=>Dict("norm"=>-0.155631), "compare-4…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.286968), "accum-1-2"=>Dict("dict"=>0.314088)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.422926)), "dict-1"=>Dict("compare-4-2"=>Dict("true…
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.272652), "dot-3-1"=>Dict("dot"=>0.199894), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.183275)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.317679), "norm-2-2"=>Dict("norm"=>0.165936), "dot-2-1"=…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.542306)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
140

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T23:41:23.209
STEP 1 ================================
prereg loss 30.422218 reg_l1 39.023117 reg_l2 14.09325
loss 38.22684
STEP 2 ================================
prereg loss 26.45379 reg_l1 39.039116 reg_l2 14.109279
loss 34.261612
STEP 3 ================================
prereg loss 22.817385 reg_l1 39.055172 reg_l2 14.125594
loss 30.62842
STEP 4 ================================
prereg loss 19.458937 reg_l1 39.071285 reg_l2 14.142185
loss 27.273193
STEP 5 ================================
prereg loss 16.427393 reg_l1 39.08746 reg_l2 14.159021
loss 24.244884
STEP 6 ================================
prereg loss 13.822078 reg_l1 39.10377 reg_l2 14.176059
loss 21.642832
STEP 7 ================================
prereg loss 11.654863 reg_l1 39.120068 reg_l2 14.193151
loss 19.478878
STEP 8 ================================
prereg loss 9.946658 reg_l1 39.13577 reg_l2 14.209849
loss 17.773811
STEP 9 ================================
prereg loss 8.692036 reg_l1 39.151127 reg_l2 14.226149
loss 16.52226
STEP 10 ================================
prereg loss 7.8614206 reg_l1 39.16607 reg_l2 14.24186
loss 15.694634
STEP 11 ================================
prereg loss 7.4000025 reg_l1 39.180412 reg_l2 14.256746
loss 15.236085
STEP 12 ================================
prereg loss 7.232927 reg_l1 39.194008 reg_l2 14.270571
loss 15.071729
STEP 13 ================================
prereg loss 7.3046103 reg_l1 39.206566 reg_l2 14.283092
loss 15.145924
STEP 14 ================================
prereg loss 7.4748707 reg_l1 39.217957 reg_l2 14.294153
loss 15.318462
STEP 15 ================================
prereg loss 7.651361 reg_l1 39.228024 reg_l2 14.303696
loss 15.496965
STEP 16 ================================
prereg loss 7.7684355 reg_l1 39.236893 reg_l2 14.311749
loss 15.615814
STEP 17 ================================
prereg loss 7.789491 reg_l1 39.244686 reg_l2 14.318444
loss 15.638429
STEP 18 ================================
prereg loss 7.701357 reg_l1 39.2517 reg_l2 14.323998
loss 15.551697
STEP 19 ================================
prereg loss 7.514048 reg_l1 39.25825 reg_l2 14.328659
loss 15.365698
STEP 20 ================================
prereg loss 7.251681 reg_l1 39.264687 reg_l2 14.332713
loss 15.104618
STEP 21 ================================
prereg loss 6.9445252 reg_l1 39.271378 reg_l2 14.336473
loss 14.798801
STEP 22 ================================
prereg loss 6.624019 reg_l1 39.278625 reg_l2 14.340216
loss 14.479744
STEP 23 ================================
prereg loss 6.313972 reg_l1 39.28679 reg_l2 14.344243
loss 14.1713295
STEP 24 ================================
prereg loss 6.0322423 reg_l1 39.296158 reg_l2 14.348815
loss 13.891474
STEP 25 ================================
prereg loss 5.7887764 reg_l1 39.306942 reg_l2 14.354168
loss 13.650166
STEP 26 ================================
prereg loss 5.5861216 reg_l1 39.319443 reg_l2 14.360524
loss 13.45001
STEP 27 ================================
prereg loss 5.420801 reg_l1 39.333755 reg_l2 14.368036
loss 13.287552
STEP 28 ================================
prereg loss 5.2858677 reg_l1 39.349995 reg_l2 14.376837
loss 13.155867
STEP 29 ================================
prereg loss 5.1713214 reg_l1 39.3682 reg_l2 14.387009
loss 13.044961
STEP 30 ================================
prereg loss 5.0666456 reg_l1 39.38828 reg_l2 14.39854
loss 12.944302
STEP 31 ================================
prereg loss 4.9645214 reg_l1 39.410156 reg_l2 14.411399
loss 12.846553
STEP 32 ================================
prereg loss 4.8572073 reg_l1 39.43364 reg_l2 14.425516
loss 12.743935
STEP 33 ================================
prereg loss 4.738741 reg_l1 39.45843 reg_l2 14.440754
loss 12.630427
STEP 34 ================================
prereg loss 4.611798 reg_l1 39.484295 reg_l2 14.456958
loss 12.508657
STEP 35 ================================
prereg loss 4.482841 reg_l1 39.510853 reg_l2 14.473903
loss 12.385012
STEP 36 ================================
prereg loss 4.355044 reg_l1 39.537712 reg_l2 14.49135
loss 12.262587
STEP 37 ================================
prereg loss 4.2334557 reg_l1 39.564503 reg_l2 14.509039
loss 12.146357
STEP 38 ================================
prereg loss 4.123391 reg_l1 39.590824 reg_l2 14.526734
loss 12.041555
STEP 39 ================================
prereg loss 4.0260763 reg_l1 39.616524 reg_l2 14.544287
loss 11.949381
STEP 40 ================================
prereg loss 3.9451263 reg_l1 39.641216 reg_l2 14.5614605
loss 11.873369
STEP 41 ================================
prereg loss 3.8800273 reg_l1 39.66456 reg_l2 14.578022
loss 11.812939
STEP 42 ================================
prereg loss 3.8269312 reg_l1 39.686337 reg_l2 14.593787
loss 11.764198
STEP 43 ================================
prereg loss 3.785122 reg_l1 39.706352 reg_l2 14.608615
loss 11.726393
STEP 44 ================================
prereg loss 3.7522826 reg_l1 39.724533 reg_l2 14.622389
loss 11.697189
STEP 45 ================================
prereg loss 3.7261765 reg_l1 39.74077 reg_l2 14.635035
loss 11.674331
STEP 46 ================================
prereg loss 3.7024603 reg_l1 39.755066 reg_l2 14.64649
loss 11.653473
STEP 47 ================================
prereg loss 3.6804473 reg_l1 39.767426 reg_l2 14.6567745
loss 11.633932
STEP 48 ================================
prereg loss 3.6589649 reg_l1 39.777992 reg_l2 14.665928
loss 11.614563
STEP 49 ================================
prereg loss 3.635469 reg_l1 39.786816 reg_l2 14.674003
loss 11.592833
STEP 50 ================================
prereg loss 3.60738 reg_l1 39.794113 reg_l2 14.681078
loss 11.566202
STEP 51 ================================
prereg loss 3.5761182 reg_l1 39.80004 reg_l2 14.6873
loss 11.536126
STEP 52 ================================
prereg loss 3.5437934 reg_l1 39.804893 reg_l2 14.692806
loss 11.504772
STEP 53 ================================
prereg loss 3.5125163 reg_l1 39.80886 reg_l2 14.697765
loss 11.474289
STEP 54 ================================
prereg loss 3.4840295 reg_l1 39.812183 reg_l2 14.702338
loss 11.446466
STEP 55 ================================
prereg loss 3.4594057 reg_l1 39.815147 reg_l2 14.7066765
loss 11.422435
STEP 56 ================================
prereg loss 3.4392495 reg_l1 39.817856 reg_l2 14.710933
loss 11.402821
STEP 57 ================================
prereg loss 3.4233782 reg_l1 39.820595 reg_l2 14.715262
loss 11.387497
STEP 58 ================================
prereg loss 3.409089 reg_l1 39.823444 reg_l2 14.719749
loss 11.373777
STEP 59 ================================
prereg loss 3.3949485 reg_l1 39.826523 reg_l2 14.724483
loss 11.360253
STEP 60 ================================
prereg loss 3.379727 reg_l1 39.829838 reg_l2 14.729457
loss 11.345695
STEP 61 ================================
prereg loss 3.3626072 reg_l1 39.833416 reg_l2 14.734709
loss 11.32929
STEP 62 ================================
prereg loss 3.3437595 reg_l1 39.837254 reg_l2 14.740204
loss 11.311211
STEP 63 ================================
prereg loss 3.3247893 reg_l1 39.8412 reg_l2 14.745875
loss 11.29303
STEP 64 ================================
prereg loss 3.3051188 reg_l1 39.845226 reg_l2 14.751663
loss 11.274164
STEP 65 ================================
prereg loss 3.2852886 reg_l1 39.849194 reg_l2 14.7575
loss 11.255127
STEP 66 ================================
prereg loss 3.266197 reg_l1 39.853065 reg_l2 14.763321
loss 11.23681
STEP 67 ================================
prereg loss 3.2479622 reg_l1 39.8567 reg_l2 14.769058
loss 11.219302
STEP 68 ================================
prereg loss 3.2309263 reg_l1 39.86003 reg_l2 14.774636
loss 11.202932
STEP 69 ================================
prereg loss 3.2152522 reg_l1 39.86298 reg_l2 14.779987
loss 11.187848
STEP 70 ================================
prereg loss 3.200588 reg_l1 39.865482 reg_l2 14.785078
loss 11.173684
STEP 71 ================================
prereg loss 3.1866562 reg_l1 39.86749 reg_l2 14.789852
loss 11.160154
STEP 72 ================================
prereg loss 3.1731722 reg_l1 39.86898 reg_l2 14.794303
loss 11.146969
STEP 73 ================================
prereg loss 3.1598847 reg_l1 39.869938 reg_l2 14.798414
loss 11.133872
STEP 74 ================================
prereg loss 3.146625 reg_l1 39.87042 reg_l2 14.802206
loss 11.120708
STEP 75 ================================
prereg loss 3.1333373 reg_l1 39.87045 reg_l2 14.805699
loss 11.107428
STEP 76 ================================
prereg loss 3.1200573 reg_l1 39.87006 reg_l2 14.808919
loss 11.0940695
STEP 77 ================================
prereg loss 3.1068935 reg_l1 39.86934 reg_l2 14.811921
loss 11.080761
STEP 78 ================================
prereg loss 3.0939739 reg_l1 39.868347 reg_l2 14.814746
loss 11.067643
STEP 79 ================================
prereg loss 3.0813973 reg_l1 39.86715 reg_l2 14.817446
loss 11.054828
STEP 80 ================================
prereg loss 3.0692255 reg_l1 39.865814 reg_l2 14.820077
loss 11.042389
STEP 81 ================================
prereg loss 3.057433 reg_l1 39.864433 reg_l2 14.822685
loss 11.030319
STEP 82 ================================
prereg loss 3.0459538 reg_l1 39.863033 reg_l2 14.825307
loss 11.01856
STEP 83 ================================
prereg loss 3.034686 reg_l1 39.86167 reg_l2 14.827978
loss 11.007021
STEP 84 ================================
prereg loss 3.0235083 reg_l1 39.860355 reg_l2 14.830717
loss 10.99558
STEP 85 ================================
prereg loss 3.0123112 reg_l1 39.859104 reg_l2 14.83355
loss 10.984132
STEP 86 ================================
prereg loss 3.0010352 reg_l1 39.857956 reg_l2 14.836457
loss 10.972627
STEP 87 ================================
prereg loss 2.9896622 reg_l1 39.85687 reg_l2 14.8394575
loss 10.961037
STEP 88 ================================
prereg loss 2.9780555 reg_l1 39.855835 reg_l2 14.842524
loss 10.949223
STEP 89 ================================
prereg loss 2.9664354 reg_l1 39.85482 reg_l2 14.8456335
loss 10.937399
STEP 90 ================================
prereg loss 2.954893 reg_l1 39.853794 reg_l2 14.848765
loss 10.925652
STEP 91 ================================
prereg loss 2.9434917 reg_l1 39.852715 reg_l2 14.851895
loss 10.914035
STEP 92 ================================
prereg loss 2.9322872 reg_l1 39.851574 reg_l2 14.855
loss 10.902602
STEP 93 ================================
prereg loss 2.921312 reg_l1 39.850327 reg_l2 14.858058
loss 10.891377
STEP 94 ================================
prereg loss 2.9105783 reg_l1 39.848923 reg_l2 14.86104
loss 10.880363
STEP 95 ================================
prereg loss 2.9000664 reg_l1 39.8474 reg_l2 14.863939
loss 10.869547
STEP 96 ================================
prereg loss 2.8897564 reg_l1 39.845695 reg_l2 14.866737
loss 10.858895
STEP 97 ================================
prereg loss 2.8796265 reg_l1 39.84383 reg_l2 14.869441
loss 10.8483925
STEP 98 ================================
prereg loss 2.8696547 reg_l1 39.841774 reg_l2 14.872037
loss 10.838009
STEP 99 ================================
prereg loss 2.8598228 reg_l1 39.839584 reg_l2 14.874543
loss 10.82774
STEP 100 ================================
prereg loss 2.8501327 reg_l1 39.837257 reg_l2 14.876956
loss 10.817584
2022-06-26T23:51:09.804

julia> serialize("sparse14-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse14-after-100-steps-opt.ser", opt)

julia> count_interval(sparse14, -0.01f0, 0.01f0)
0

julia> count_interval(sparse14, -0.02f0, 0.02f0)
0

julia> count_interval(sparse14, -0.03f0, 0.03f0)
0

julia> count_interval(sparse14, -0.04f0, 0.04f0)
0

julia> count_interval(sparse14, -0.05f0, 0.05f0)
0

julia> count_interval(sparse14, -0.06f0, 0.06f0)
1

julia> count_interval(sparse14, -0.07f0, 0.07f0)
1

julia> count_interval(sparse14, -0.08f0, 0.08f0)
2

julia> count_interval(sparse14, -0.09f0, 0.09f0)
2

julia> count_interval(sparse14, -0.10f0, 0.10f0)
2

julia> count_interval(sparse14, -0.11f0, 0.11f0)
3

julia> count_interval(sparse14, -0.12f0, 0.12f0)
3

julia> count_interval(sparse14, -0.13f0, 0.13f0)
4

julia> count_interval(sparse14, -0.14f0, 0.14f0)
5

julia> count_interval(sparse14, -0.15f0, 0.15f0)
5

julia> count_interval(sparse14, -0.16f0, 0.16f0)
9

julia> count_interval(sparse14, -0.17f0, 0.17f0)
10

julia> count_interval(sparse14, -0.18f0, 0.18f0)
13

julia> count_interval(sparse14, -0.19f0, 0.19f0)
19

julia> sparse15 = sparsecopy(trainable["network_matrix"], 0.19f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 27 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.74201)), "dict-1"=>Dict("input"=>Dict("char"=>0.212431)…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>0.190204), "norm-2-1"=>Dict("norm"=>0.484002), "norm-…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.333653), "input"=>Dict("char"=>-0.2314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.195564), "eos"=>Dict("char"=>-0.811038)), "dict-1"=…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.276827), "accum-1-2"=>Dict("dict"=>0.317998)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.236139)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.306581), "dot-2-1"=>Dict("dot"=>-0.318998), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234716), "norm-2-1"=>Dict("norm"=>0.198341), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.224056), "compare-5-1"=>Dict("true"=>0.425682), "dot-5…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.236381), "compare-4-1"=>Dict("false"=>0.285142), "compa…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.267031)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.287034), "norm-3-1"=>Dict("norm"=>0.194652), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.226061)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.261109), "norm-1-2"=>Dict("norm"=>0.241817)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.319136), "eos"=>Dict("char"=>-0.382387)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271038), "norm-3-1"=>Dict("norm"=>0.237233), "compare…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.281266), "norm-2-2"=>Dict("norm"=>0.206139), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.308967), "dot-3-1"=>Dict("dot"=>0.190322), "dot-3-2"=>D…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.288374), "accum-1-2"=>Dict("dict"=>0.31623)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.322927)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.288312), "dot-3-1"=>Dict("dot"=>0.201107), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.220835)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.336022), "dot-2-1"=>Dict("dot"=>-0.317189), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.496763)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.821865)))
  ⋮             => ⋮

julia> trainable["network_matrix"] = sparse15
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 27 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.74201)), "dict-1"=>Dict("input"=>Dict("char"=>0.212431)…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>0.190204), "norm-2-1"=>Dict("norm"=>0.484002), "norm-…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.333653), "input"=>Dict("char"=>-0.2314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.195564), "eos"=>Dict("char"=>-0.811038)), "dict-1"=…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.276827), "accum-1-2"=>Dict("dict"=>0.317998)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.236139)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.306581), "dot-2-1"=>Dict("dot"=>-0.318998), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.234716), "norm-2-1"=>Dict("norm"=>0.198341), "dot-2-2"=…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.224056), "compare-5-1"=>Dict("true"=>0.425682), "dot-5…
  "dot-5-1"     => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.236381), "compare-4-1"=>Dict("false"=>0.285142), "compa…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.267031)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.287034), "norm-3-1"=>Dict("norm"=>0.194652), "compare…
  "dot-1-2"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.226061)))
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.261109), "norm-1-2"=>Dict("norm"=>0.241817)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.319136), "eos"=>Dict("char"=>-0.382387)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.271038), "norm-3-1"=>Dict("norm"=>0.237233), "compare…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.281266), "norm-2-2"=>Dict("norm"=>0.206139), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.308967), "dot-3-1"=>Dict("dot"=>0.190322), "dot-3-2"=>D…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.288374), "accum-1-2"=>Dict("dict"=>0.31623)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.322927)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.288312), "dot-3-1"=>Dict("dot"=>0.201107), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.220835)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.336022), "dot-2-1"=>Dict("dot"=>-0.317189), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.496763)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.821865)))
  ⋮             => ⋮

julia> count(trainable["network_matrix"])
121

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-26T23:57:32.720
STEP 1 ================================
prereg loss 34.992397 reg_l1 36.91237 reg_l2 14.402786
loss 42.37487
STEP 2 ================================
prereg loss 31.421286 reg_l1 36.875366 reg_l2 14.373743
loss 38.79636
STEP 3 ================================
prereg loss 28.351103 reg_l1 36.839077 reg_l2 14.345518
loss 35.718918
STEP 4 ================================
prereg loss 25.751667 reg_l1 36.804054 reg_l2 14.318668
loss 33.112476
STEP 5 ================================
prereg loss 23.57992 reg_l1 36.771427 reg_l2 14.294125
loss 30.934206
STEP 6 ================================
prereg loss 21.785002 reg_l1 36.742928 reg_l2 14.273191
loss 29.133587
STEP 7 ================================
prereg loss 20.284822 reg_l1 36.72017 reg_l2 14.257011
loss 27.628857
STEP 8 ================================
prereg loss 19.0428 reg_l1 36.704323 reg_l2 14.2463045
loss 26.383665
STEP 9 ================================
prereg loss 17.968323 reg_l1 36.696266 reg_l2 14.241388
loss 25.307575
STEP 10 ================================
prereg loss 17.014576 reg_l1 36.696205 reg_l2 14.242188
loss 24.353817
STEP 11 ================================
prereg loss 16.145159 reg_l1 36.704113 reg_l2 14.248464
loss 23.485981
STEP 12 ================================
prereg loss 15.336681 reg_l1 36.719563 reg_l2 14.259789
loss 22.680593
STEP 13 ================================
prereg loss 14.575301 reg_l1 36.74198 reg_l2 14.275724
loss 21.923698
STEP 14 ================================
prereg loss 13.834465 reg_l1 36.77052 reg_l2 14.295656
loss 21.188568
STEP 15 ================================
prereg loss 13.11124 reg_l1 36.804253 reg_l2 14.319013
loss 20.472092
STEP 16 ================================
prereg loss 12.407315 reg_l1 36.84232 reg_l2 14.345207
loss 19.77578
STEP 17 ================================
prereg loss 11.7275 reg_l1 36.88391 reg_l2 14.373747
loss 19.104282
STEP 18 ================================
prereg loss 11.078316 reg_l1 36.928276 reg_l2 14.404137
loss 18.46397
STEP 19 ================================
prereg loss 10.4669075 reg_l1 36.974712 reg_l2 14.435952
loss 17.86185
STEP 20 ================================
prereg loss 9.900132 reg_l1 37.022552 reg_l2 14.468775
loss 17.304642
STEP 21 ================================
prereg loss 9.383673 reg_l1 37.07125 reg_l2 14.502236
loss 16.797924
STEP 22 ================================
prereg loss 8.921558 reg_l1 37.120216 reg_l2 14.535976
loss 16.345602
STEP 23 ================================
prereg loss 8.515585 reg_l1 37.168903 reg_l2 14.569648
loss 15.949366
STEP 24 ================================
prereg loss 8.16496 reg_l1 37.216866 reg_l2 14.602931
loss 15.608334
STEP 25 ================================
prereg loss 7.866291 reg_l1 37.26358 reg_l2 14.635513
loss 15.319008
STEP 26 ================================
prereg loss 7.6137595 reg_l1 37.308655 reg_l2 14.667116
loss 15.075491
STEP 27 ================================
prereg loss 7.3996987 reg_l1 37.35168 reg_l2 14.697489
loss 14.870035
STEP 28 ================================
prereg loss 7.215317 reg_l1 37.392353 reg_l2 14.726408
loss 14.693788
STEP 29 ================================
prereg loss 7.051823 reg_l1 37.430393 reg_l2 14.753712
loss 14.537902
STEP 30 ================================
prereg loss 6.9014444 reg_l1 37.46565 reg_l2 14.779276
loss 14.394574
STEP 31 ================================
prereg loss 6.758293 reg_l1 37.498005 reg_l2 14.803035
loss 14.2578945
STEP 32 ================================
prereg loss 6.618806 reg_l1 37.52748 reg_l2 14.82498
loss 14.124302
STEP 33 ================================
prereg loss 6.48236 reg_l1 37.554134 reg_l2 14.845156
loss 13.993187
STEP 34 ================================
prereg loss 6.350307 reg_l1 37.5781 reg_l2 14.863664
loss 13.865927
STEP 35 ================================
prereg loss 6.22526 reg_l1 37.599586 reg_l2 14.880611
loss 13.745177
STEP 36 ================================
prereg loss 6.1090493 reg_l1 37.618885 reg_l2 14.896185
loss 13.632826
STEP 37 ================================
prereg loss 6.003835 reg_l1 37.636223 reg_l2 14.910521
loss 13.53108
STEP 38 ================================
prereg loss 5.911067 reg_l1 37.65183 reg_l2 14.923745
loss 13.441433
STEP 39 ================================
prereg loss 5.8314652 reg_l1 37.665947 reg_l2 14.936005
loss 13.364655
STEP 40 ================================
prereg loss 5.7714663 reg_l1 37.678776 reg_l2 14.947445
loss 13.307221
STEP 41 ================================
prereg loss 5.7241464 reg_l1 37.690655 reg_l2 14.958259
loss 13.262278
STEP 42 ================================
prereg loss 5.682772 reg_l1 37.70172 reg_l2 14.968479
loss 13.223116
STEP 43 ================================
prereg loss 5.6387525 reg_l1 37.711983 reg_l2 14.978155
loss 13.181149
STEP 44 ================================
prereg loss 5.5840454 reg_l1 37.72188 reg_l2 14.987465
loss 13.128422
STEP 45 ================================
prereg loss 5.5332365 reg_l1 37.73135 reg_l2 14.996378
loss 13.079506
STEP 46 ================================
prereg loss 5.484785 reg_l1 37.74062 reg_l2 15.005071
loss 13.032909
STEP 47 ================================
prereg loss 5.4392962 reg_l1 37.74953 reg_l2 15.0134535
loss 12.9892025
STEP 48 ================================
prereg loss 5.3964868 reg_l1 37.757923 reg_l2 15.021446
loss 12.9480715
STEP 49 ================================
prereg loss 5.352444 reg_l1 37.765663 reg_l2 15.028959
loss 12.905577
STEP 50 ================================
prereg loss 5.3065295 reg_l1 37.77258 reg_l2 15.0358925
loss 12.861046
STEP 51 ================================
prereg loss 5.2590775 reg_l1 37.77858 reg_l2 15.042185
loss 12.814794
STEP 52 ================================
prereg loss 5.2111692 reg_l1 37.783554 reg_l2 15.047769
loss 12.7678795
STEP 53 ================================
prereg loss 5.164095 reg_l1 37.787434 reg_l2 15.052615
loss 12.721582
STEP 54 ================================
prereg loss 5.118909 reg_l1 37.790195 reg_l2 15.056701
loss 12.676949
STEP 55 ================================
prereg loss 5.076103 reg_l1 37.791794 reg_l2 15.060031
loss 12.634462
STEP 56 ================================
prereg loss 5.035457 reg_l1 37.792267 reg_l2 15.062611
loss 12.59391
STEP 57 ================================
prereg loss 4.9962077 reg_l1 37.79167 reg_l2 15.064494
loss 12.554543
STEP 58 ================================
prereg loss 4.9575014 reg_l1 37.790115 reg_l2 15.065749
loss 12.515524
STEP 59 ================================
prereg loss 4.9186687 reg_l1 37.78772 reg_l2 15.0664625
loss 12.476213
STEP 60 ================================
prereg loss 4.8791537 reg_l1 37.784626 reg_l2 15.066733
loss 12.436079
STEP 61 ================================
prereg loss 4.8384542 reg_l1 37.781017 reg_l2 15.06669
loss 12.394657
STEP 62 ================================
prereg loss 4.796639 reg_l1 37.77707 reg_l2 15.06644
loss 12.352053
STEP 63 ================================
prereg loss 4.7542844 reg_l1 37.772892 reg_l2 15.066066
loss 12.308863
STEP 64 ================================
prereg loss 4.712166 reg_l1 37.768612 reg_l2 15.065659
loss 12.265888
STEP 65 ================================
prereg loss 4.6709666 reg_l1 37.76435 reg_l2 15.0652895
loss 12.223837
STEP 66 ================================
prereg loss 4.6311564 reg_l1 37.760174 reg_l2 15.065016
loss 12.183191
STEP 67 ================================
prereg loss 4.5928826 reg_l1 37.756123 reg_l2 15.06489
loss 12.144108
STEP 68 ================================
prereg loss 4.5560007 reg_l1 37.752274 reg_l2 15.064924
loss 12.106455
STEP 69 ================================
prereg loss 4.520152 reg_l1 37.748592 reg_l2 15.065144
loss 12.06987
STEP 70 ================================
prereg loss 4.4849267 reg_l1 37.745125 reg_l2 15.065539
loss 12.033952
STEP 71 ================================
prereg loss 4.4500003 reg_l1 37.741806 reg_l2 15.066093
loss 11.998362
STEP 72 ================================
prereg loss 4.4151883 reg_l1 37.738605 reg_l2 15.066778
loss 11.96291
STEP 73 ================================
prereg loss 4.38051 reg_l1 37.735474 reg_l2 15.067573
loss 11.927605
STEP 74 ================================
prereg loss 4.346116 reg_l1 37.732365 reg_l2 15.068432
loss 11.89259
STEP 75 ================================
prereg loss 4.3122315 reg_l1 37.72922 reg_l2 15.069318
loss 11.858076
STEP 76 ================================
prereg loss 4.279045 reg_l1 37.72596 reg_l2 15.070193
loss 11.824238
STEP 77 ================================
prereg loss 4.2466636 reg_l1 37.722588 reg_l2 15.071038
loss 11.791182
STEP 78 ================================
prereg loss 4.2150745 reg_l1 37.719048 reg_l2 15.071823
loss 11.758884
STEP 79 ================================
prereg loss 4.184163 reg_l1 37.715294 reg_l2 15.072538
loss 11.727222
STEP 80 ================================
prereg loss 4.153772 reg_l1 37.71137 reg_l2 15.073167
loss 11.696046
STEP 81 ================================
prereg loss 4.1238656 reg_l1 37.707233 reg_l2 15.073725
loss 11.665312
STEP 82 ================================
prereg loss 4.094367 reg_l1 37.70297 reg_l2 15.074243
loss 11.63496
STEP 83 ================================
prereg loss 4.0651345 reg_l1 37.698605 reg_l2 15.074747
loss 11.604856
STEP 84 ================================
prereg loss 4.0362015 reg_l1 37.694157 reg_l2 15.07524
loss 11.575033
STEP 85 ================================
prereg loss 4.0076375 reg_l1 37.689655 reg_l2 15.075753
loss 11.545568
STEP 86 ================================
prereg loss 3.979531 reg_l1 37.68514 reg_l2 15.076312
loss 11.516559
STEP 87 ================================
prereg loss 3.9519649 reg_l1 37.680634 reg_l2 15.076922
loss 11.488091
STEP 88 ================================
prereg loss 3.925438 reg_l1 37.676178 reg_l2 15.077615
loss 11.460674
STEP 89 ================================
prereg loss 3.8993852 reg_l1 37.67199 reg_l2 15.078529
loss 11.433784
STEP 90 ================================
prereg loss 3.8730726 reg_l1 37.668053 reg_l2 15.07964
loss 11.406683
STEP 91 ================================
prereg loss 3.8470507 reg_l1 37.664257 reg_l2 15.080903
loss 11.379902
STEP 92 ================================
prereg loss 3.8220606 reg_l1 37.66033 reg_l2 15.082115
loss 11.354127
STEP 93 ================================
prereg loss 3.7974324 reg_l1 37.656204 reg_l2 15.083251
loss 11.328673
STEP 94 ================================
prereg loss 3.7731025 reg_l1 37.651867 reg_l2 15.084292
loss 11.303476
STEP 95 ================================
prereg loss 3.749044 reg_l1 37.64733 reg_l2 15.0852375
loss 11.27851
STEP 96 ================================
prereg loss 3.7252893 reg_l1 37.642605 reg_l2 15.086106
loss 11.253811
STEP 97 ================================
prereg loss 3.7018971 reg_l1 37.637737 reg_l2 15.086911
loss 11.2294445
STEP 98 ================================
prereg loss 3.6789186 reg_l1 37.632755 reg_l2 15.087708
loss 11.20547
STEP 99 ================================
prereg loss 3.6563942 reg_l1 37.627766 reg_l2 15.088521
loss 11.181948
STEP 100 ================================
prereg loss 3.6343336 reg_l1 37.622807 reg_l2 15.0894
loss 11.1588955
2022-06-27T00:06:36.119

julia> # it's possible that I am moving too fast and would need to backtrack eventually

julia> serialize("sparse15-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse15-after-100-steps-opt.ser", opt)

julia> count_interval(sparse15, -0.05f0, 0.05f0)
0

julia> count_interval(sparse15, -0.06f0, 0.06f0)
0

julia> count_interval(sparse15, -0.07f0, 0.07f0)
0

julia> count_interval(sparse15, -0.08f0, 0.08f0)
0

julia> count_interval(sparse15, -0.09f0, 0.09f0)
0

julia> count_interval(sparse15, -0.10f0, 0.10f0)
2

julia> count_interval(sparse15, -0.11f0, 0.11f0)
2

julia> count_interval(sparse15, -0.12f0, 0.12f0)
3

julia> count_interval(sparse15, -0.13f0, 0.13f0)
5

julia> count_interval(sparse15, -0.14f0, 0.14f0)
5

julia> count_interval(sparse15, -0.15f0, 0.15f0)
5

julia> count_interval(sparse15, -0.16f0, 0.16f0)
5

julia> count_interval(sparse15, -0.17f0, 0.17f0)
9

julia> count_interval(sparse15, -0.18f0, 0.18f0)
11

julia> count_interval(sparse15, -0.19f0, 0.19f0)
12

julia> count_interval(sparse15, -0.20f0, 0.20f0)
12

julia> count_interval(sparse15, -0.21f0, 0.21f0)
13

julia> count_interval(sparse15, -0.19f0, 0.19f0)
12

julia> # let's do this

julia> sparse16 = sparsecopy(trainable["network_matrix"], 0.19f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.735798)), "dict-1"=>Dict("eos"=>Dict("char"=>0.730239)))
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.444016), "norm-2-1"=>Dict("norm"=>0.506384), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.354819), "input"=>Dict("char"=>-0.272899)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.87008)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.2236…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.304485), "accum-1-2"=>Dict("dict"=>0.345916)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.257788)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.319533), "dot-2-1"=>Dict("dot"=>-0.322384), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.22255), "norm-2-1"=>Dict("norm"=>0.237651), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.20601), "compare-5-1"=>Dict("true"=>0.471731), "dot-5-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.326264), "compare-4-2"=>Dict("false"=>0.266998)),…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.287015)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.311677), "norm-3-1"=>Dict("norm"=>0.219816), "compare…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.245932), "norm-1-2"=>Dict("norm"=>0.247488)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.314081), "eos"=>Dict("char"=>-0.393255)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.296154), "norm-3-1"=>Dict("norm"=>0.26287), "compare-…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.254983), "norm-2-2"=>Dict("norm"=>0.214895), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.258929), "dot-3-2"=>Dict("dot"=>-0.223609), "dot-4-1"=>…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.316264), "accum-1-2"=>Dict("dict"=>0.344219)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.222928)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.307018), "dot-3-1"=>Dict("dot"=>0.223663), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.242458)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.342491), "dot-2-1"=>Dict("dot"=>-0.303811), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.496239)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.847182)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.21666)))

julia> trainable["network_matrix"] = sparse16
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>-0.735798)), "dict-1"=>Dict("eos"=>Dict("char"=>0.730239)))
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.444016), "norm-2-1"=>Dict("norm"=>0.506384), "norm-3-1"…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.354819), "input"=>Dict("char"=>-0.272899)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.87008)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.2236…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.304485), "accum-1-2"=>Dict("dict"=>0.345916)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.257788)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.319533), "dot-2-1"=>Dict("dot"=>-0.322384), "accum-2-2"…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.22255), "norm-2-1"=>Dict("norm"=>0.237651), "dot-2-2"=>…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.20601), "compare-5-1"=>Dict("true"=>0.471731), "dot-5-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.326264), "compare-4-2"=>Dict("false"=>0.266998)),…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>-0.287015)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.311677), "norm-3-1"=>Dict("norm"=>0.219816), "compare…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.245932), "norm-1-2"=>Dict("norm"=>0.247488)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.314081), "eos"=>Dict("char"=>-0.393255)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.296154), "norm-3-1"=>Dict("norm"=>0.26287), "compare-…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.254983), "norm-2-2"=>Dict("norm"=>0.214895), "compar…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.258929), "dot-3-2"=>Dict("dot"=>-0.223609), "dot-4-1"=>…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>0.316264), "accum-1-2"=>Dict("dict"=>0.344219)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>0.222928)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.307018), "dot-3-1"=>Dict("dot"=>0.223663), "dot-3-2"=>D…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.242458)), "dict-1"=>Dict("const_1"=>Dict("const_1"=…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.342491), "dot-2-1"=>Dict("dot"=>-0.303811), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.496239)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.847182)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.21666)))

julia> count(trainable["network_matrix"])
109

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-27T00:15:31.032
STEP 1 ================================
prereg loss 59.988968 reg_l1 35.883892 reg_l2 14.827332
loss 67.16575
STEP 2 ================================
prereg loss 57.540398 reg_l1 35.92088 reg_l2 14.852069
loss 64.72457
STEP 3 ================================
prereg loss 54.821564 reg_l1 35.9563 reg_l2 14.87636
loss 62.012825
STEP 4 ================================
prereg loss 51.93579 reg_l1 35.986725 reg_l2 14.8984585
loss 59.133137
STEP 5 ================================
prereg loss 49.054867 reg_l1 36.013344 reg_l2 14.918872
loss 56.257534
STEP 6 ================================
prereg loss 46.187138 reg_l1 36.037987 reg_l2 14.938501
loss 53.394733
STEP 7 ================================
prereg loss 43.345787 reg_l1 36.06169 reg_l2 14.9578495
loss 50.558125
STEP 8 ================================
prereg loss 40.541847 reg_l1 36.085075 reg_l2 14.977241
loss 47.75886
STEP 9 ================================
prereg loss 37.78321 reg_l1 36.108707 reg_l2 14.996973
loss 45.00495
STEP 10 ================================
prereg loss 35.065327 reg_l1 36.132988 reg_l2 15.01727
loss 42.291924
STEP 11 ================================
prereg loss 32.402058 reg_l1 36.158173 reg_l2 15.038273
loss 39.633694
STEP 12 ================================
prereg loss 29.808342 reg_l1 36.184116 reg_l2 15.059885
loss 37.045166
STEP 13 ================================
prereg loss 27.29681 reg_l1 36.210503 reg_l2 15.081946
loss 34.53891
STEP 14 ================================
prereg loss 24.880144 reg_l1 36.236984 reg_l2 15.104234
loss 32.12754
STEP 15 ================================
prereg loss 22.570044 reg_l1 36.263203 reg_l2 15.12654
loss 29.822685
STEP 16 ================================
prereg loss 20.37775 reg_l1 36.28892 reg_l2 15.148738
loss 27.635534
STEP 17 ================================
prereg loss 18.315142 reg_l1 36.314056 reg_l2 15.170753
loss 25.577953
STEP 18 ================================
prereg loss 16.39332 reg_l1 36.33863 reg_l2 15.192579
loss 23.661047
STEP 19 ================================
prereg loss 14.608174 reg_l1 36.362762 reg_l2 15.21426
loss 21.880726
STEP 20 ================================
prereg loss 12.9918585 reg_l1 36.386536 reg_l2 15.235837
loss 20.269165
STEP 21 ================================
prereg loss 11.555541 reg_l1 36.410023 reg_l2 15.257319
loss 18.837545
STEP 22 ================================
prereg loss 10.308226 reg_l1 36.43323 reg_l2 15.27867
loss 17.594872
STEP 23 ================================
prereg loss 9.247968 reg_l1 36.455746 reg_l2 15.299598
loss 16.539116
STEP 24 ================================
prereg loss 8.375508 reg_l1 36.477436 reg_l2 15.3200035
loss 15.670996
STEP 25 ================================
prereg loss 7.6549573 reg_l1 36.498062 reg_l2 15.339701
loss 14.95457
STEP 26 ================================
prereg loss 7.057084 reg_l1 36.51735 reg_l2 15.358503
loss 14.360554
STEP 27 ================================
prereg loss 6.6538725 reg_l1 36.53494 reg_l2 15.376153
loss 13.96086
STEP 28 ================================
prereg loss 6.430196 reg_l1 36.55052 reg_l2 15.392402
loss 13.7403
STEP 29 ================================
prereg loss 6.3632836 reg_l1 36.563824 reg_l2 15.407027
loss 13.676048
STEP 30 ================================
prereg loss 6.423632 reg_l1 36.574665 reg_l2 15.419864
loss 13.738565
STEP 31 ================================
prereg loss 6.5763626 reg_l1 36.582924 reg_l2 15.430771
loss 13.892948
STEP 32 ================================
prereg loss 6.783479 reg_l1 36.588642 reg_l2 15.43972
loss 14.101208
STEP 33 ================================
prereg loss 7.0067954 reg_l1 36.591812 reg_l2 15.446632
loss 14.325158
STEP 34 ================================
prereg loss 7.2120676 reg_l1 36.592403 reg_l2 15.451473
loss 14.530548
STEP 35 ================================
prereg loss 7.3722777 reg_l1 36.59039 reg_l2 15.454223
loss 14.690355
STEP 36 ================================
prereg loss 7.4696684 reg_l1 36.585842 reg_l2 15.4548855
loss 14.786837
STEP 37 ================================
prereg loss 7.4960356 reg_l1 36.578766 reg_l2 15.453538
loss 14.811789
STEP 38 ================================
prereg loss 7.4519615 reg_l1 36.569313 reg_l2 15.450286
loss 14.765824
STEP 39 ================================
prereg loss 7.3452535 reg_l1 36.55768 reg_l2 15.445312
loss 14.65679
STEP 40 ================================
prereg loss 7.188937 reg_l1 36.5441 reg_l2 15.43882
loss 14.497757
STEP 41 ================================
prereg loss 6.9988728 reg_l1 36.52896 reg_l2 15.431075
loss 14.304665
STEP 42 ================================
prereg loss 6.7914586 reg_l1 36.51258 reg_l2 15.4223385
loss 14.093975
STEP 43 ================================
prereg loss 6.5817027 reg_l1 36.49538 reg_l2 15.412897
loss 13.880779
STEP 44 ================================
prereg loss 6.3820252 reg_l1 36.47768 reg_l2 15.403009
loss 13.677561
STEP 45 ================================
prereg loss 6.2018137 reg_l1 36.45975 reg_l2 15.392899
loss 13.493764
STEP 46 ================================
prereg loss 6.0472 reg_l1 36.44183 reg_l2 15.38276
loss 13.335566
STEP 47 ================================
prereg loss 5.9212236 reg_l1 36.424095 reg_l2 15.372746
loss 13.206043
STEP 48 ================================
prereg loss 5.8240256 reg_l1 36.40665 reg_l2 15.362999
loss 13.105356
STEP 49 ================================
prereg loss 5.7535286 reg_l1 36.38965 reg_l2 15.353651
loss 13.031458
STEP 50 ================================
prereg loss 5.706069 reg_l1 36.37322 reg_l2 15.344809
loss 12.980713
STEP 51 ================================
prereg loss 5.6771164 reg_l1 36.3575 reg_l2 15.336577
loss 12.948616
STEP 52 ================================
prereg loss 5.6617494 reg_l1 36.34262 reg_l2 15.329059
loss 12.930273
STEP 53 ================================
prereg loss 5.6550364 reg_l1 36.328674 reg_l2 15.322323
loss 12.920772
STEP 54 ================================
prereg loss 5.6524115 reg_l1 36.31578 reg_l2 15.3164215
loss 12.915567
STEP 55 ================================
prereg loss 5.650017 reg_l1 36.303913 reg_l2 15.311362
loss 12.910799
STEP 56 ================================
prereg loss 5.644883 reg_l1 36.292824 reg_l2 15.306994
loss 12.903448
STEP 57 ================================
prereg loss 5.634817 reg_l1 36.282486 reg_l2 15.30331
loss 12.8913145
STEP 58 ================================
prereg loss 5.6185064 reg_l1 36.272835 reg_l2 15.300267
loss 12.873074
STEP 59 ================================
prereg loss 5.595368 reg_l1 36.263798 reg_l2 15.297825
loss 12.848127
STEP 60 ================================
prereg loss 5.565564 reg_l1 36.255325 reg_l2 15.295916
loss 12.816629
STEP 61 ================================
prereg loss 5.52969 reg_l1 36.247547 reg_l2 15.294615
loss 12.7792
STEP 62 ================================
prereg loss 5.4887924 reg_l1 36.240402 reg_l2 15.293859
loss 12.736874
STEP 63 ================================
prereg loss 5.444175 reg_l1 36.233837 reg_l2 15.293581
loss 12.690943
STEP 64 ================================
prereg loss 5.3972178 reg_l1 36.227753 reg_l2 15.293707
loss 12.642769
STEP 65 ================================
prereg loss 5.349345 reg_l1 36.22204 reg_l2 15.294156
loss 12.593753
STEP 66 ================================
prereg loss 5.3018646 reg_l1 36.21659 reg_l2 15.294832
loss 12.545183
STEP 67 ================================
prereg loss 5.2558956 reg_l1 36.211266 reg_l2 15.295656
loss 12.498149
STEP 68 ================================
prereg loss 5.21229 reg_l1 36.20596 reg_l2 15.296532
loss 12.453482
STEP 69 ================================
prereg loss 5.1716466 reg_l1 36.20035 reg_l2 15.2972765
loss 12.411716
STEP 70 ================================
prereg loss 5.134202 reg_l1 36.194416 reg_l2 15.297838
loss 12.373085
STEP 71 ================================
prereg loss 5.0999684 reg_l1 36.1881 reg_l2 15.2981825
loss 12.337588
STEP 72 ================================
prereg loss 5.0687284 reg_l1 36.181374 reg_l2 15.298262
loss 12.305003
STEP 73 ================================
prereg loss 5.040101 reg_l1 36.17424 reg_l2 15.298065
loss 12.274949
STEP 74 ================================
prereg loss 5.0135083 reg_l1 36.166855 reg_l2 15.29766
loss 12.24688
STEP 75 ================================
prereg loss 4.988383 reg_l1 36.15917 reg_l2 15.297019
loss 12.220217
STEP 76 ================================
prereg loss 4.964178 reg_l1 36.1511 reg_l2 15.296094
loss 12.194399
STEP 77 ================================
prereg loss 4.9403796 reg_l1 36.142666 reg_l2 15.294896
loss 12.168913
STEP 78 ================================
prereg loss 4.916562 reg_l1 36.133823 reg_l2 15.293381
loss 12.143327
STEP 79 ================================
prereg loss 4.8924828 reg_l1 36.124386 reg_l2 15.291471
loss 12.11736
STEP 80 ================================
prereg loss 4.8679123 reg_l1 36.114407 reg_l2 15.289186
loss 12.090794
STEP 81 ================================
prereg loss 4.8427587 reg_l1 36.10393 reg_l2 15.286559
loss 12.063545
STEP 82 ================================
prereg loss 4.817076 reg_l1 36.09302 reg_l2 15.28363
loss 12.035681
STEP 83 ================================
prereg loss 4.7908955 reg_l1 36.081917 reg_l2 15.280529
loss 12.007278
STEP 84 ================================
prereg loss 4.7643657 reg_l1 36.07065 reg_l2 15.277296
loss 11.978496
STEP 85 ================================
prereg loss 4.7376924 reg_l1 36.059223 reg_l2 15.273942
loss 11.949537
STEP 86 ================================
prereg loss 4.7110806 reg_l1 36.04768 reg_l2 15.270503
loss 11.920616
STEP 87 ================================
prereg loss 4.6847425 reg_l1 36.03585 reg_l2 15.266893
loss 11.891912
STEP 88 ================================
prereg loss 4.6587605 reg_l1 36.023785 reg_l2 15.263169
loss 11.863518
STEP 89 ================================
prereg loss 4.633269 reg_l1 36.011555 reg_l2 15.259354
loss 11.83558
STEP 90 ================================
prereg loss 4.6083016 reg_l1 35.99922 reg_l2 15.2555065
loss 11.8081455
STEP 91 ================================
prereg loss 4.58387 reg_l1 35.986813 reg_l2 15.251655
loss 11.781233
STEP 92 ================================
prereg loss 4.559924 reg_l1 35.97459 reg_l2 15.247931
loss 11.754843
STEP 93 ================================
prereg loss 4.5363793 reg_l1 35.96256 reg_l2 15.244352
loss 11.728891
STEP 94 ================================
prereg loss 4.5131917 reg_l1 35.950687 reg_l2 15.240919
loss 11.703329
STEP 95 ================================
prereg loss 4.490318 reg_l1 35.93881 reg_l2 15.237535
loss 11.67808
STEP 96 ================================
prereg loss 4.4676595 reg_l1 35.92694 reg_l2 15.234218
loss 11.653048
STEP 97 ================================
prereg loss 4.4451385 reg_l1 35.915092 reg_l2 15.230975
loss 11.628157
STEP 98 ================================
prereg loss 4.4226975 reg_l1 35.903297 reg_l2 15.22782
loss 11.603357
STEP 99 ================================
prereg loss 4.400259 reg_l1 35.891758 reg_l2 15.224857
loss 11.578611
STEP 100 ================================
prereg loss 4.3778315 reg_l1 35.88024 reg_l2 15.221978
loss 11.55388
2022-06-27T00:24:07.031

julia> # let's continue training, make sure we are not at a dead end

julia> steps!(200)
2022-06-27T00:28:25.802
STEP 1 ================================
prereg loss 4.355354 reg_l1 35.868973 reg_l2 15.219273
loss 11.529148
STEP 2 ================================
prereg loss 4.3328834 reg_l1 35.857708 reg_l2 15.216635
loss 11.504425
STEP 3 ================================
prereg loss 4.3104362 reg_l1 35.846455 reg_l2 15.214042
loss 11.479727
STEP 4 ================================
prereg loss 4.288045 reg_l1 35.835175 reg_l2 15.211486
loss 11.45508
STEP 5 ================================
prereg loss 4.2656865 reg_l1 35.824066 reg_l2 15.209065
loss 11.4305
STEP 6 ================================
prereg loss 4.2434125 reg_l1 35.813114 reg_l2 15.206774
loss 11.406035
STEP 7 ================================
prereg loss 4.221279 reg_l1 35.80209 reg_l2 15.204457
loss 11.381697
STEP 8 ================================
prereg loss 4.199311 reg_l1 35.79099 reg_l2 15.202144
loss 11.357509
STEP 9 ================================
prereg loss 4.177523 reg_l1 35.779797 reg_l2 15.199801
loss 11.333483
STEP 10 ================================
prereg loss 4.1559443 reg_l1 35.768524 reg_l2 15.197421
loss 11.309649
STEP 11 ================================
prereg loss 4.1345663 reg_l1 35.757133 reg_l2 15.194995
loss 11.285994
STEP 12 ================================
prereg loss 4.1133957 reg_l1 35.74564 reg_l2 15.192523
loss 11.262524
STEP 13 ================================
prereg loss 4.0924244 reg_l1 35.734005 reg_l2 15.189991
loss 11.239225
STEP 14 ================================
prereg loss 4.071683 reg_l1 35.722244 reg_l2 15.187399
loss 11.216131
STEP 15 ================================
prereg loss 4.051056 reg_l1 35.710567 reg_l2 15.184836
loss 11.19317
STEP 16 ================================
prereg loss 4.030552 reg_l1 35.69894 reg_l2 15.182312
loss 11.17034
STEP 17 ================================
prereg loss 4.0102415 reg_l1 35.687157 reg_l2 15.179691
loss 11.147673
STEP 18 ================================
prereg loss 3.9900875 reg_l1 35.6752 reg_l2 15.176995
loss 11.125128
STEP 19 ================================
prereg loss 3.970088 reg_l1 35.66311 reg_l2 15.174216
loss 11.10271
STEP 20 ================================
prereg loss 3.950224 reg_l1 35.650856 reg_l2 15.1713505
loss 11.080395
STEP 21 ================================
prereg loss 3.9304967 reg_l1 35.638493 reg_l2 15.168419
loss 11.058195
STEP 22 ================================
prereg loss 3.9109185 reg_l1 35.625984 reg_l2 15.165418
loss 11.036116
STEP 23 ================================
prereg loss 3.8914769 reg_l1 35.61338 reg_l2 15.162365
loss 11.0141535
STEP 24 ================================
prereg loss 3.8721557 reg_l1 35.600647 reg_l2 15.159247
loss 10.992285
STEP 25 ================================
prereg loss 3.8529828 reg_l1 35.587822 reg_l2 15.156094
loss 10.970548
STEP 26 ================================
prereg loss 3.8339422 reg_l1 35.574917 reg_l2 15.152897
loss 10.948926
STEP 27 ================================
prereg loss 3.8150434 reg_l1 35.561935 reg_l2 15.149662
loss 10.927431
STEP 28 ================================
prereg loss 3.7962418 reg_l1 35.549065 reg_l2 15.146509
loss 10.906055
STEP 29 ================================
prereg loss 3.7775822 reg_l1 35.536114 reg_l2 15.143336
loss 10.884806
STEP 30 ================================
prereg loss 3.7590532 reg_l1 35.523106 reg_l2 15.140127
loss 10.863674
STEP 31 ================================
prereg loss 3.7406545 reg_l1 35.51001 reg_l2 15.13691
loss 10.842657
STEP 32 ================================
prereg loss 3.722386 reg_l1 35.496857 reg_l2 15.133677
loss 10.821757
STEP 33 ================================
prereg loss 3.7042356 reg_l1 35.483677 reg_l2 15.130445
loss 10.800971
STEP 34 ================================
prereg loss 3.686204 reg_l1 35.470444 reg_l2 15.127206
loss 10.7802925
STEP 35 ================================
prereg loss 3.6682901 reg_l1 35.457184 reg_l2 15.123969
loss 10.7597275
STEP 36 ================================
prereg loss 3.6504917 reg_l1 35.443874 reg_l2 15.120736
loss 10.739267
STEP 37 ================================
prereg loss 3.6327958 reg_l1 35.43054 reg_l2 15.117507
loss 10.718904
STEP 38 ================================
prereg loss 3.6152086 reg_l1 35.41718 reg_l2 15.114274
loss 10.698645
STEP 39 ================================
prereg loss 3.5977309 reg_l1 35.403774 reg_l2 15.111054
loss 10.678486
STEP 40 ================================
prereg loss 3.5803564 reg_l1 35.39034 reg_l2 15.107833
loss 10.658424
STEP 41 ================================
prereg loss 3.5630894 reg_l1 35.376877 reg_l2 15.104619
loss 10.638465
STEP 42 ================================
prereg loss 3.5459235 reg_l1 35.3634 reg_l2 15.101404
loss 10.618604
STEP 43 ================================
prereg loss 3.5288572 reg_l1 35.34988 reg_l2 15.098189
loss 10.598833
STEP 44 ================================
prereg loss 3.5118957 reg_l1 35.336323 reg_l2 15.094981
loss 10.579161
STEP 45 ================================
prereg loss 3.495235 reg_l1 35.322727 reg_l2 15.091772
loss 10.55978
STEP 46 ================================
prereg loss 3.4787412 reg_l1 35.309708 reg_l2 15.088957
loss 10.540683
STEP 47 ================================
prereg loss 3.462322 reg_l1 35.297123 reg_l2 15.086481
loss 10.521747
STEP 48 ================================
prereg loss 3.4460075 reg_l1 35.2848 reg_l2 15.08428
loss 10.502968
STEP 49 ================================
prereg loss 3.4296548 reg_l1 35.272663 reg_l2 15.082295
loss 10.484187
STEP 50 ================================
prereg loss 3.413331 reg_l1 35.260647 reg_l2 15.080496
loss 10.46546
STEP 51 ================================
prereg loss 3.3971362 reg_l1 35.248825 reg_l2 15.078864
loss 10.446901
STEP 52 ================================
prereg loss 3.381039 reg_l1 35.2372 reg_l2 15.0774
loss 10.428479
STEP 53 ================================
prereg loss 3.3648767 reg_l1 35.226006 reg_l2 15.076184
loss 10.410078
STEP 54 ================================
prereg loss 3.3488107 reg_l1 35.214912 reg_l2 15.075043
loss 10.391793
STEP 55 ================================
prereg loss 3.3329153 reg_l1 35.20377 reg_l2 15.073913
loss 10.373669
STEP 56 ================================
prereg loss 3.3171525 reg_l1 35.19248 reg_l2 15.072744
loss 10.355648
STEP 57 ================================
prereg loss 3.3015049 reg_l1 35.180977 reg_l2 15.071522
loss 10.3377
STEP 58 ================================
prereg loss 3.286012 reg_l1 35.169315 reg_l2 15.070215
loss 10.319875
STEP 59 ================================
prereg loss 3.2706814 reg_l1 35.157486 reg_l2 15.068837
loss 10.302179
STEP 60 ================================
prereg loss 3.2554638 reg_l1 35.145535 reg_l2 15.067387
loss 10.284571
STEP 61 ================================
prereg loss 3.2403407 reg_l1 35.13345 reg_l2 15.065843
loss 10.267031
STEP 62 ================================
prereg loss 3.2253706 reg_l1 35.12117 reg_l2 15.064201
loss 10.249605
STEP 63 ================================
prereg loss 3.2105546 reg_l1 35.10865 reg_l2 15.06243
loss 10.232285
STEP 64 ================================
prereg loss 3.1958673 reg_l1 35.095875 reg_l2 15.06053
loss 10.215042
STEP 65 ================================
prereg loss 3.181305 reg_l1 35.082836 reg_l2 15.058501
loss 10.197872
STEP 66 ================================
prereg loss 3.1668723 reg_l1 35.06961 reg_l2 15.056374
loss 10.180795
STEP 67 ================================
prereg loss 3.1524694 reg_l1 35.056007 reg_l2 15.05406
loss 10.163671
STEP 68 ================================
prereg loss 3.1382105 reg_l1 35.042156 reg_l2 15.051606
loss 10.146642
STEP 69 ================================
prereg loss 3.1240368 reg_l1 35.028088 reg_l2 15.049035
loss 10.129654
STEP 70 ================================
prereg loss 3.109956 reg_l1 35.013874 reg_l2 15.04638
loss 10.112731
STEP 71 ================================
prereg loss 3.096284 reg_l1 34.999508 reg_l2 15.043658
loss 10.096186
STEP 72 ================================
prereg loss 3.0824294 reg_l1 34.985954 reg_l2 15.041298
loss 10.07962
STEP 73 ================================
prereg loss 3.0694098 reg_l1 34.97319 reg_l2 15.039341
loss 10.064048
STEP 74 ================================
prereg loss 3.0553148 reg_l1 34.959713 reg_l2 15.037134
loss 10.047257
STEP 75 ================================
prereg loss 3.0424984 reg_l1 34.945667 reg_l2 15.034732
loss 10.031631
STEP 76 ================================
prereg loss 3.0293434 reg_l1 34.932766 reg_l2 15.032902
loss 10.015897
STEP 77 ================================
prereg loss 3.0154836 reg_l1 34.920933 reg_l2 15.031596
loss 9.99967
STEP 78 ================================
prereg loss 3.0010543 reg_l1 34.909954 reg_l2 15.030698
loss 9.983046
STEP 79 ================================
prereg loss 2.986658 reg_l1 34.899464 reg_l2 15.030008
loss 9.966551
STEP 80 ================================
prereg loss 2.9734967 reg_l1 34.888073 reg_l2 15.028896
loss 9.951112
STEP 81 ================================
prereg loss 2.9600086 reg_l1 34.87578 reg_l2 15.027364
loss 9.935164
STEP 82 ================================
prereg loss 2.9469388 reg_l1 34.86271 reg_l2 15.025451
loss 9.91948
STEP 83 ================================
prereg loss 2.9340737 reg_l1 34.849697 reg_l2 15.023487
loss 9.904014
STEP 84 ================================
prereg loss 2.9209895 reg_l1 34.83669 reg_l2 15.02145
loss 9.888328
STEP 85 ================================
prereg loss 2.9082725 reg_l1 34.82307 reg_l2 15.019085
loss 9.872887
STEP 86 ================================
prereg loss 2.895667 reg_l1 34.80881 reg_l2 15.016398
loss 9.8574295
STEP 87 ================================
prereg loss 2.8831446 reg_l1 34.794 reg_l2 15.013438
loss 9.841945
STEP 88 ================================
prereg loss 2.870742 reg_l1 34.77948 reg_l2 15.010582
loss 9.826638
STEP 89 ================================
prereg loss 2.8586638 reg_l1 34.7653 reg_l2 15.007873
loss 9.811724
STEP 90 ================================
prereg loss 2.846301 reg_l1 34.75048 reg_l2 15.004901
loss 9.796397
STEP 91 ================================
prereg loss 2.8346384 reg_l1 34.735172 reg_l2 15.001731
loss 9.7816725
STEP 92 ================================
prereg loss 2.822746 reg_l1 34.720577 reg_l2 14.998904
loss 9.766862
STEP 93 ================================
prereg loss 2.8104079 reg_l1 34.706745 reg_l2 14.99644
loss 9.751757
STEP 94 ================================
prereg loss 2.7979152 reg_l1 34.693493 reg_l2 14.994254
loss 9.736614
STEP 95 ================================
prereg loss 2.785977 reg_l1 34.679623 reg_l2 14.991827
loss 9.721901
STEP 96 ================================
prereg loss 2.7741065 reg_l1 34.66518 reg_l2 14.989162
loss 9.707143
STEP 97 ================================
prereg loss 2.7622137 reg_l1 34.651268 reg_l2 14.986734
loss 9.692467
STEP 98 ================================
prereg loss 2.7503738 reg_l1 34.637783 reg_l2 14.98449
loss 9.677931
STEP 99 ================================
prereg loss 2.738671 reg_l1 34.623722 reg_l2 14.982017
loss 9.663416
STEP 100 ================================
prereg loss 2.726994 reg_l1 34.61046 reg_l2 14.979303
loss 9.649086
STEP 101 ================================
prereg loss 2.7159169 reg_l1 34.597965 reg_l2 14.976427
loss 9.63551
STEP 102 ================================
prereg loss 2.704306 reg_l1 34.58573 reg_l2 14.973758
loss 9.621452
STEP 103 ================================
prereg loss 2.6927683 reg_l1 34.573734 reg_l2 14.971276
loss 9.607515
STEP 104 ================================
prereg loss 2.6816 reg_l1 34.56102 reg_l2 14.968586
loss 9.593804
STEP 105 ================================
prereg loss 2.6701634 reg_l1 34.54762 reg_l2 14.965693
loss 9.579687
STEP 106 ================================
prereg loss 2.6594114 reg_l1 34.53368 reg_l2 14.96265
loss 9.566147
STEP 107 ================================
prereg loss 2.64836 reg_l1 34.520283 reg_l2 14.959921
loss 9.552417
STEP 108 ================================
prereg loss 2.6367931 reg_l1 34.507416 reg_l2 14.9574795
loss 9.538277
STEP 109 ================================
prereg loss 2.626132 reg_l1 34.494915 reg_l2 14.955266
loss 9.525115
STEP 110 ================================
prereg loss 2.6151135 reg_l1 34.48166 reg_l2 14.952806
loss 9.511445
STEP 111 ================================
prereg loss 2.6036499 reg_l1 34.46774 reg_l2 14.950129
loss 9.497198
STEP 112 ================================
prereg loss 2.5934606 reg_l1 34.453327 reg_l2 14.947275
loss 9.484126
STEP 113 ================================
prereg loss 2.5828547 reg_l1 34.439453 reg_l2 14.944661
loss 9.470745
STEP 114 ================================
prereg loss 2.5715008 reg_l1 34.426006 reg_l2 14.94226
loss 9.456702
STEP 115 ================================
prereg loss 2.5606248 reg_l1 34.41333 reg_l2 14.940018
loss 9.443291
STEP 116 ================================
prereg loss 2.550334 reg_l1 34.40097 reg_l2 14.937536
loss 9.430529
STEP 117 ================================
prereg loss 2.5393863 reg_l1 34.387814 reg_l2 14.934819
loss 9.416949
STEP 118 ================================
prereg loss 2.5288298 reg_l1 34.374012 reg_l2 14.931912
loss 9.403632
STEP 119 ================================
prereg loss 2.5186179 reg_l1 34.360523 reg_l2 14.929191
loss 9.390722
STEP 120 ================================
prereg loss 2.507842 reg_l1 34.347363 reg_l2 14.926652
loss 9.377315
STEP 121 ================================
prereg loss 2.4975889 reg_l1 34.33443 reg_l2 14.924253
loss 9.364475
STEP 122 ================================
prereg loss 2.4874425 reg_l1 34.320827 reg_l2 14.92163
loss 9.351608
STEP 123 ================================
prereg loss 2.4768598 reg_l1 34.306576 reg_l2 14.918807
loss 9.338175
STEP 124 ================================
prereg loss 2.4669707 reg_l1 34.292164 reg_l2 14.915804
loss 9.325403
STEP 125 ================================
prereg loss 2.4569948 reg_l1 34.28056 reg_l2 14.913016
loss 9.313107
STEP 126 ================================
prereg loss 2.4464362 reg_l1 34.26912 reg_l2 14.910445
loss 9.300261
STEP 127 ================================
prereg loss 2.4366634 reg_l1 34.25776 reg_l2 14.908026
loss 9.288216
STEP 128 ================================
prereg loss 2.4268155 reg_l1 34.245552 reg_l2 14.905398
loss 9.275927
STEP 129 ================================
prereg loss 2.416404 reg_l1 34.232555 reg_l2 14.902569
loss 9.262916
STEP 130 ================================
prereg loss 2.406903 reg_l1 34.2192 reg_l2 14.899577
loss 9.250743
STEP 131 ================================
prereg loss 2.3972871 reg_l1 34.20652 reg_l2 14.896796
loss 9.238591
STEP 132 ================================
prereg loss 2.3869908 reg_l1 34.194088 reg_l2 14.894222
loss 9.225808
STEP 133 ================================
prereg loss 2.3772595 reg_l1 34.181824 reg_l2 14.891798
loss 9.213624
STEP 134 ================================
prereg loss 2.3677905 reg_l1 34.168827 reg_l2 14.889166
loss 9.201556
STEP 135 ================================
prereg loss 2.357819 reg_l1 34.155857 reg_l2 14.886387
loss 9.188991
STEP 136 ================================
prereg loss 2.3481493 reg_l1 34.14227 reg_l2 14.883439
loss 9.176603
STEP 137 ================================
prereg loss 2.3387866 reg_l1 34.12888 reg_l2 14.88066
loss 9.164562
STEP 138 ================================
prereg loss 2.3289115 reg_l1 34.115807 reg_l2 14.878024
loss 9.152073
STEP 139 ================================
prereg loss 2.3195193 reg_l1 34.103065 reg_l2 14.875209
loss 9.140133
STEP 140 ================================
prereg loss 2.310148 reg_l1 34.090008 reg_l2 14.872218
loss 9.12815
STEP 141 ================================
prereg loss 2.3009245 reg_l1 34.07664 reg_l2 14.869092
loss 9.116253
STEP 142 ================================
prereg loss 2.2915165 reg_l1 34.063587 reg_l2 14.866216
loss 9.104234
STEP 143 ================================
prereg loss 2.2824392 reg_l1 34.04994 reg_l2 14.863199
loss 9.092426
STEP 144 ================================
prereg loss 2.2731795 reg_l1 34.036716 reg_l2 14.860462
loss 9.080523
STEP 145 ================================
prereg loss 2.2639105 reg_l1 34.023453 reg_l2 14.857593
loss 9.068602
STEP 146 ================================
prereg loss 2.2552345 reg_l1 34.00979 reg_l2 14.854575
loss 9.057192
STEP 147 ================================
prereg loss 2.2460358 reg_l1 33.997112 reg_l2 14.85189
loss 9.045459
STEP 148 ================================
prereg loss 2.237237 reg_l1 33.9848 reg_l2 14.849482
loss 9.034197
STEP 149 ================================
prereg loss 2.2303417 reg_l1 33.97193 reg_l2 14.846927
loss 9.024728
STEP 150 ================================
prereg loss 2.2236025 reg_l1 33.958897 reg_l2 14.844222
loss 9.015382
STEP 151 ================================
prereg loss 2.217383 reg_l1 33.9457 reg_l2 14.841384
loss 9.006523
STEP 152 ================================
prereg loss 2.2109604 reg_l1 33.93302 reg_l2 14.838693
loss 8.997564
STEP 153 ================================
prereg loss 2.2048607 reg_l1 33.92133 reg_l2 14.836088
loss 8.989126
STEP 154 ================================
prereg loss 2.1989455 reg_l1 33.909992 reg_l2 14.833329
loss 8.980944
STEP 155 ================================
prereg loss 2.1929595 reg_l1 33.898643 reg_l2 14.830427
loss 8.972689
STEP 156 ================================
prereg loss 2.1872098 reg_l1 33.886986 reg_l2 14.827404
loss 8.964607
STEP 157 ================================
prereg loss 2.1817555 reg_l1 33.876465 reg_l2 14.82426
loss 8.957048
STEP 158 ================================
prereg loss 2.1762817 reg_l1 33.867455 reg_l2 14.821011
loss 8.949773
STEP 159 ================================
prereg loss 2.170833 reg_l1 33.85763 reg_l2 14.81768
loss 8.94236
STEP 160 ================================
prereg loss 2.1656108 reg_l1 33.84743 reg_l2 14.814291
loss 8.935097
STEP 161 ================================
prereg loss 2.1604702 reg_l1 33.836803 reg_l2 14.810871
loss 8.927831
STEP 162 ================================
prereg loss 2.1552773 reg_l1 33.825607 reg_l2 14.80744
loss 8.920399
STEP 163 ================================
prereg loss 2.1501715 reg_l1 33.81449 reg_l2 14.804329
loss 8.91307
STEP 164 ================================
prereg loss 2.1450827 reg_l1 33.802773 reg_l2 14.8012
loss 8.905638
STEP 165 ================================
prereg loss 2.1398995 reg_l1 33.792316 reg_l2 14.7984085
loss 8.898363
STEP 166 ================================
prereg loss 2.1346972 reg_l1 33.782253 reg_l2 14.79557
loss 8.891148
STEP 167 ================================
prereg loss 2.1298492 reg_l1 33.77149 reg_l2 14.792679
loss 8.884147
STEP 168 ================================
prereg loss 2.1245928 reg_l1 33.761875 reg_l2 14.790076
loss 8.876968
STEP 169 ================================
prereg loss 2.1194687 reg_l1 33.752525 reg_l2 14.787678
loss 8.869974
STEP 170 ================================
prereg loss 2.1144705 reg_l1 33.74291 reg_l2 14.785163
loss 8.863052
STEP 171 ================================
prereg loss 2.1093473 reg_l1 33.732887 reg_l2 14.782534
loss 8.855925
STEP 172 ================================
prereg loss 2.1044488 reg_l1 33.72313 reg_l2 14.779773
loss 8.849075
STEP 173 ================================
prereg loss 2.0998003 reg_l1 33.7127 reg_l2 14.776887
loss 8.84234
STEP 174 ================================
prereg loss 2.0947793 reg_l1 33.70208 reg_l2 14.774068
loss 8.835196
STEP 175 ================================
prereg loss 2.0900695 reg_l1 33.691933 reg_l2 14.771123
loss 8.828456
STEP 176 ================================
prereg loss 2.0854537 reg_l1 33.68146 reg_l2 14.768076
loss 8.821746
STEP 177 ================================
prereg loss 2.0807462 reg_l1 33.670277 reg_l2 14.764948
loss 8.814802
STEP 178 ================================
prereg loss 2.0762556 reg_l1 33.65848 reg_l2 14.761742
loss 8.807952
STEP 179 ================================
prereg loss 2.0715566 reg_l1 33.64762 reg_l2 14.758759
loss 8.801081
STEP 180 ================================
prereg loss 2.0670407 reg_l1 33.63789 reg_l2 14.75599
loss 8.794619
STEP 181 ================================
prereg loss 2.0624664 reg_l1 33.627663 reg_l2 14.75312
loss 8.787999
STEP 182 ================================
prereg loss 2.057679 reg_l1 33.617214 reg_l2 14.750164
loss 8.781122
STEP 183 ================================
prereg loss 2.053405 reg_l1 33.606155 reg_l2 14.747117
loss 8.774636
STEP 184 ================================
prereg loss 2.0487883 reg_l1 33.595955 reg_l2 14.744254
loss 8.76798
STEP 185 ================================
prereg loss 2.0440633 reg_l1 33.586475 reg_l2 14.741545
loss 8.761358
STEP 186 ================================
prereg loss 2.0397367 reg_l1 33.576317 reg_l2 14.738705
loss 8.755
STEP 187 ================================
prereg loss 2.0362995 reg_l1 33.56587 reg_l2 14.735759
loss 8.749474
STEP 188 ================================
prereg loss 2.0328956 reg_l1 33.555267 reg_l2 14.732708
loss 8.743949
STEP 189 ================================
prereg loss 2.0298865 reg_l1 33.544186 reg_l2 14.72956
loss 8.738724
STEP 190 ================================
prereg loss 2.026538 reg_l1 33.53374 reg_l2 14.726519
loss 8.733286
STEP 191 ================================
prereg loss 2.0234127 reg_l1 33.523266 reg_l2 14.723378
loss 8.7280655
STEP 192 ================================
prereg loss 2.0204356 reg_l1 33.512524 reg_l2 14.720165
loss 8.72294
STEP 193 ================================
prereg loss 2.0174365 reg_l1 33.50216 reg_l2 14.716907
loss 8.717869
STEP 194 ================================
prereg loss 2.0143535 reg_l1 33.494335 reg_l2 14.713624
loss 8.713221
STEP 195 ================================
prereg loss 2.011542 reg_l1 33.486168 reg_l2 14.7102995
loss 8.7087755
STEP 196 ================================
prereg loss 2.0082526 reg_l1 33.477985 reg_l2 14.707233
loss 8.70385
STEP 197 ================================
prereg loss 2.0051293 reg_l1 33.469315 reg_l2 14.704369
loss 8.698992
STEP 198 ================================
prereg loss 2.0019672 reg_l1 33.460434 reg_l2 14.701403
loss 8.694054
STEP 199 ================================
prereg loss 1.9985707 reg_l1 33.4528 reg_l2 14.698307
loss 8.689131
STEP 200 ================================
prereg loss 1.9958738 reg_l1 33.44486 reg_l2 14.695062
loss 8.684846
2022-06-27T00:46:43.632

julia> steps!(500)
2022-06-27T00:54:56.038
STEP 1 ================================
prereg loss 1.9929249 reg_l1 33.436954 reg_l2 14.691924
loss 8.680316
STEP 2 ================================
prereg loss 1.9896885 reg_l1 33.429214 reg_l2 14.688869
loss 8.675531
STEP 3 ================================
prereg loss 1.9864166 reg_l1 33.420887 reg_l2 14.685854
loss 8.670594
STEP 4 ================================
prereg loss 1.9835962 reg_l1 33.41175 reg_l2 14.682605
loss 8.665946
STEP 5 ================================
prereg loss 1.9805559 reg_l1 33.402508 reg_l2 14.67916
loss 8.661057
STEP 6 ================================
prereg loss 1.9778757 reg_l1 33.392838 reg_l2 14.675517
loss 8.656443
STEP 7 ================================
prereg loss 1.9751229 reg_l1 33.384304 reg_l2 14.671975
loss 8.651983
STEP 8 ================================
prereg loss 1.9720727 reg_l1 33.376038 reg_l2 14.66854
loss 8.647281
STEP 9 ================================
prereg loss 1.9689636 reg_l1 33.367294 reg_l2 14.665194
loss 8.642423
STEP 10 ================================
prereg loss 1.9660596 reg_l1 33.35757 reg_l2 14.661641
loss 8.637573
STEP 11 ================================
prereg loss 1.9630563 reg_l1 33.349106 reg_l2 14.657891
loss 8.632877
STEP 12 ================================
prereg loss 1.9601781 reg_l1 33.341114 reg_l2 14.654205
loss 8.628401
STEP 13 ================================
prereg loss 1.9572436 reg_l1 33.332546 reg_l2 14.650559
loss 8.623753
STEP 14 ================================
prereg loss 1.9544318 reg_l1 33.322773 reg_l2 14.6466875
loss 8.618986
STEP 15 ================================
prereg loss 1.9517273 reg_l1 33.31366 reg_l2 14.642846
loss 8.614459
STEP 16 ================================
prereg loss 1.9489586 reg_l1 33.30418 reg_l2 14.63881
loss 8.609795
STEP 17 ================================
prereg loss 1.9463758 reg_l1 33.29433 reg_l2 14.634599
loss 8.605242
STEP 18 ================================
prereg loss 1.94366 reg_l1 33.28412 reg_l2 14.63047
loss 8.600484
STEP 19 ================================
prereg loss 1.9410057 reg_l1 33.273792 reg_l2 14.626444
loss 8.595764
STEP 20 ================================
prereg loss 1.9384139 reg_l1 33.2641 reg_l2 14.6222725
loss 8.591234
STEP 21 ================================
prereg loss 1.935713 reg_l1 33.25431 reg_l2 14.618001
loss 8.5865755
STEP 22 ================================
prereg loss 1.9330815 reg_l1 33.244198 reg_l2 14.613885
loss 8.581921
STEP 23 ================================
prereg loss 1.9304448 reg_l1 33.233864 reg_l2 14.609659
loss 8.577218
STEP 24 ================================
prereg loss 1.927781 reg_l1 33.224426 reg_l2 14.605614
loss 8.572666
STEP 25 ================================
prereg loss 1.9251206 reg_l1 33.214993 reg_l2 14.601472
loss 8.568119
STEP 26 ================================
prereg loss 1.9225651 reg_l1 33.204895 reg_l2 14.597242
loss 8.563544
STEP 27 ================================
prereg loss 1.9198713 reg_l1 33.194912 reg_l2 14.5932045
loss 8.558853
STEP 28 ================================
prereg loss 1.9174162 reg_l1 33.185005 reg_l2 14.589345
loss 8.554418
STEP 29 ================================
prereg loss 1.9147981 reg_l1 33.175293 reg_l2 14.585403
loss 8.549857
STEP 30 ================================
prereg loss 1.9120564 reg_l1 33.165752 reg_l2 14.581387
loss 8.545207
STEP 31 ================================
prereg loss 1.9097172 reg_l1 33.15566 reg_l2 14.5773
loss 8.540849
STEP 32 ================================
prereg loss 1.9069815 reg_l1 33.14558 reg_l2 14.573374
loss 8.536098
STEP 33 ================================
prereg loss 1.9041706 reg_l1 33.136425 reg_l2 14.569595
loss 8.531456
STEP 34 ================================
prereg loss 1.9016817 reg_l1 33.12664 reg_l2 14.565726
loss 8.52701
STEP 35 ================================
prereg loss 1.8989482 reg_l1 33.11702 reg_l2 14.561774
loss 8.522352
STEP 36 ================================
prereg loss 1.896489 reg_l1 33.106976 reg_l2 14.557735
loss 8.517884
STEP 37 ================================
prereg loss 1.8939389 reg_l1 33.09739 reg_l2 14.553899
loss 8.513416
STEP 38 ================================
prereg loss 1.8911487 reg_l1 33.087612 reg_l2 14.550233
loss 8.508672
STEP 39 ================================
prereg loss 1.8886011 reg_l1 33.077164 reg_l2 14.546462
loss 8.504034
STEP 40 ================================
prereg loss 1.8859239 reg_l1 33.067127 reg_l2 14.542595
loss 8.49935
STEP 41 ================================
prereg loss 1.883597 reg_l1 33.05743 reg_l2 14.538643
loss 8.495083
STEP 42 ================================
prereg loss 1.8809327 reg_l1 33.0481 reg_l2 14.53491
loss 8.490553
STEP 43 ================================
prereg loss 1.8781573 reg_l1 33.039616 reg_l2 14.531367
loss 8.48608
STEP 44 ================================
prereg loss 1.875549 reg_l1 33.031788 reg_l2 14.527701
loss 8.481907
STEP 45 ================================
prereg loss 1.872796 reg_l1 33.023396 reg_l2 14.523913
loss 8.477475
STEP 46 ================================
prereg loss 1.8700961 reg_l1 33.014557 reg_l2 14.520267
loss 8.473007
STEP 47 ================================
prereg loss 1.8674742 reg_l1 33.004887 reg_l2 14.516454
loss 8.4684515
STEP 48 ================================
prereg loss 1.8647456 reg_l1 32.996124 reg_l2 14.512756
loss 8.46397
STEP 49 ================================
prereg loss 1.8620747 reg_l1 32.988186 reg_l2 14.508871
loss 8.459712
STEP 50 ================================
prereg loss 1.8595802 reg_l1 32.98104 reg_l2 14.504808
loss 8.455789
STEP 51 ================================
prereg loss 1.8568712 reg_l1 32.97353 reg_l2 14.500854
loss 8.451577
STEP 52 ================================
prereg loss 1.8542619 reg_l1 32.96608 reg_l2 14.496973
loss 8.447477
STEP 53 ================================
prereg loss 1.851569 reg_l1 32.958786 reg_l2 14.492895
loss 8.443326
STEP 54 ================================
prereg loss 1.8490579 reg_l1 32.950394 reg_l2 14.488617
loss 8.4391365
STEP 55 ================================
prereg loss 1.8465768 reg_l1 32.94185 reg_l2 14.48441
loss 8.434947
STEP 56 ================================
prereg loss 1.8439043 reg_l1 32.933514 reg_l2 14.480278
loss 8.430607
STEP 57 ================================
prereg loss 1.8415267 reg_l1 32.924774 reg_l2 14.476177
loss 8.426481
STEP 58 ================================
prereg loss 1.839056 reg_l1 32.91539 reg_l2 14.471883
loss 8.422134
STEP 59 ================================
prereg loss 1.8364987 reg_l1 32.905273 reg_l2 14.4674
loss 8.417553
STEP 60 ================================
prereg loss 1.834124 reg_l1 32.896263 reg_l2 14.462979
loss 8.413377
STEP 61 ================================
prereg loss 1.83166 reg_l1 32.88703 reg_l2 14.458616
loss 8.409066
STEP 62 ================================
prereg loss 1.82927 reg_l1 32.87737 reg_l2 14.454081
loss 8.404744
STEP 63 ================================
prereg loss 1.826797 reg_l1 32.86799 reg_l2 14.449409
loss 8.400394
STEP 64 ================================
prereg loss 1.8243814 reg_l1 32.859356 reg_l2 14.444849
loss 8.396253
STEP 65 ================================
prereg loss 1.8219728 reg_l1 32.850883 reg_l2 14.440162
loss 8.392149
STEP 66 ================================
prereg loss 1.8196093 reg_l1 32.842262 reg_l2 14.435617
loss 8.388062
STEP 67 ================================
prereg loss 1.8172266 reg_l1 32.832447 reg_l2 14.430954
loss 8.383717
STEP 68 ================================
prereg loss 1.814887 reg_l1 32.823837 reg_l2 14.426471
loss 8.379654
STEP 69 ================================
prereg loss 1.8125463 reg_l1 32.814365 reg_l2 14.421861
loss 8.37542
STEP 70 ================================
prereg loss 1.8101474 reg_l1 32.804707 reg_l2 14.417446
loss 8.371089
STEP 71 ================================
prereg loss 1.807786 reg_l1 32.795063 reg_l2 14.412902
loss 8.366798
STEP 72 ================================
prereg loss 1.8054507 reg_l1 32.78551 reg_l2 14.408534
loss 8.362553
STEP 73 ================================
prereg loss 1.8030493 reg_l1 32.775913 reg_l2 14.4040365
loss 8.3582325
STEP 74 ================================
prereg loss 1.8008513 reg_l1 32.767334 reg_l2 14.399422
loss 8.354319
STEP 75 ================================
prereg loss 1.7984056 reg_l1 32.75875 reg_l2 14.394974
loss 8.350156
STEP 76 ================================
prereg loss 1.7961619 reg_l1 32.749756 reg_l2 14.390668
loss 8.346113
STEP 77 ================================
prereg loss 1.7937909 reg_l1 32.74025 reg_l2 14.386244
loss 8.341841
STEP 78 ================================
prereg loss 1.7912848 reg_l1 32.73083 reg_l2 14.381718
loss 8.337451
STEP 79 ================================
prereg loss 1.7894119 reg_l1 32.720432 reg_l2 14.377057
loss 8.333498
STEP 80 ================================
prereg loss 1.7870793 reg_l1 32.71035 reg_l2 14.372568
loss 8.329149
STEP 81 ================================
prereg loss 1.7844179 reg_l1 32.701283 reg_l2 14.368253
loss 8.324675
STEP 82 ================================
prereg loss 1.7823151 reg_l1 32.691757 reg_l2 14.363807
loss 8.320666
STEP 83 ================================
prereg loss 1.7799852 reg_l1 32.681686 reg_l2 14.35927
loss 8.316322
STEP 84 ================================
prereg loss 1.778059 reg_l1 32.67191 reg_l2 14.35465
loss 8.312441
STEP 85 ================================
prereg loss 1.7756946 reg_l1 32.663216 reg_l2 14.350319
loss 8.308338
STEP 86 ================================
prereg loss 1.7729883 reg_l1 32.65464 reg_l2 14.346237
loss 8.303917
STEP 87 ================================
prereg loss 1.7709424 reg_l1 32.646294 reg_l2 14.342351
loss 8.300201
STEP 88 ================================
prereg loss 1.7683514 reg_l1 32.637844 reg_l2 14.338302
loss 8.29592
STEP 89 ================================
prereg loss 1.7657357 reg_l1 32.628616 reg_l2 14.33409
loss 8.291459
STEP 90 ================================
prereg loss 1.7635865 reg_l1 32.618927 reg_l2 14.329887
loss 8.287373
STEP 91 ================================
prereg loss 1.761191 reg_l1 32.609646 reg_l2 14.325651
loss 8.28312
STEP 92 ================================
prereg loss 1.7590082 reg_l1 32.60032 reg_l2 14.321204
loss 8.279072
STEP 93 ================================
prereg loss 1.7568163 reg_l1 32.590164 reg_l2 14.316597
loss 8.274849
STEP 94 ================================
prereg loss 1.7544934 reg_l1 32.57964 reg_l2 14.311864
loss 8.270421
STEP 95 ================================
prereg loss 1.7525318 reg_l1 32.569523 reg_l2 14.307006
loss 8.266437
STEP 96 ================================
prereg loss 1.7503476 reg_l1 32.560135 reg_l2 14.302322
loss 8.262375
STEP 97 ================================
prereg loss 1.7480905 reg_l1 32.550583 reg_l2 14.29786
loss 8.258207
STEP 98 ================================
prereg loss 1.7459177 reg_l1 32.54059 reg_l2 14.293298
loss 8.254035
STEP 99 ================================
prereg loss 1.7439302 reg_l1 32.530136 reg_l2 14.2886505
loss 8.249957
STEP 100 ================================
prereg loss 1.7417061 reg_l1 32.520226 reg_l2 14.284294
loss 8.245751
STEP 101 ================================
prereg loss 1.7391341 reg_l1 32.51107 reg_l2 14.280189
loss 8.241348
STEP 102 ================================
prereg loss 1.7372708 reg_l1 32.502247 reg_l2 14.276256
loss 8.2377205
STEP 103 ================================
prereg loss 1.7348093 reg_l1 32.49303 reg_l2 14.272176
loss 8.233416
STEP 104 ================================
prereg loss 1.7323264 reg_l1 32.483166 reg_l2 14.267939
loss 8.228959
STEP 105 ================================
prereg loss 1.7305335 reg_l1 32.47362 reg_l2 14.263501
loss 8.225258
STEP 106 ================================
prereg loss 1.7281986 reg_l1 32.463974 reg_l2 14.259024
loss 8.220994
STEP 107 ================================
prereg loss 1.7258178 reg_l1 32.4539 reg_l2 14.254524
loss 8.216598
STEP 108 ================================
prereg loss 1.7240188 reg_l1 32.443672 reg_l2 14.249864
loss 8.212753
STEP 109 ================================
prereg loss 1.7217889 reg_l1 32.433132 reg_l2 14.245116
loss 8.208416
STEP 110 ================================
prereg loss 1.719692 reg_l1 32.42263 reg_l2 14.240278
loss 8.204218
STEP 111 ================================
prereg loss 1.7177693 reg_l1 32.412178 reg_l2 14.235634
loss 8.200205
STEP 112 ================================
prereg loss 1.7154696 reg_l1 32.402145 reg_l2 14.2311945
loss 8.195899
STEP 113 ================================
prereg loss 1.7136223 reg_l1 32.39239 reg_l2 14.226946
loss 8.192101
STEP 114 ================================
prereg loss 1.7114038 reg_l1 32.38347 reg_l2 14.2226
loss 8.188097
STEP 115 ================================
prereg loss 1.7089429 reg_l1 32.37718 reg_l2 14.218155
loss 8.184379
STEP 116 ================================
prereg loss 1.7068868 reg_l1 32.371185 reg_l2 14.213833
loss 8.181124
STEP 117 ================================
prereg loss 1.7046725 reg_l1 32.364235 reg_l2 14.209579
loss 8.17752
STEP 118 ================================
prereg loss 1.7026259 reg_l1 32.35606 reg_l2 14.20515
loss 8.173838
STEP 119 ================================
prereg loss 1.7005167 reg_l1 32.348503 reg_l2 14.200578
loss 8.1702175
STEP 120 ================================
prereg loss 1.6985302 reg_l1 32.34025 reg_l2 14.195858
loss 8.16658
STEP 121 ================================
prereg loss 1.6963738 reg_l1 32.332233 reg_l2 14.191273
loss 8.162821
STEP 122 ================================
prereg loss 1.6944165 reg_l1 32.323406 reg_l2 14.186543
loss 8.159098
STEP 123 ================================
prereg loss 1.6925598 reg_l1 32.314 reg_l2 14.182029
loss 8.15536
STEP 124 ================================
prereg loss 1.6904169 reg_l1 32.305023 reg_l2 14.177422
loss 8.151422
STEP 125 ================================
prereg loss 1.6888977 reg_l1 32.295803 reg_l2 14.172731
loss 8.148059
STEP 126 ================================
prereg loss 1.6870465 reg_l1 32.28653 reg_l2 14.168318
loss 8.144352
STEP 127 ================================
prereg loss 1.6847268 reg_l1 32.27747 reg_l2 14.164142
loss 8.140221
STEP 128 ================================
prereg loss 1.6830554 reg_l1 32.268913 reg_l2 14.160146
loss 8.136838
STEP 129 ================================
prereg loss 1.6808758 reg_l1 32.261837 reg_l2 14.15603
loss 8.133244
STEP 130 ================================
prereg loss 1.6784533 reg_l1 32.255272 reg_l2 14.151788
loss 8.129508
STEP 131 ================================
prereg loss 1.6767954 reg_l1 32.24777 reg_l2 14.147347
loss 8.126349
STEP 132 ================================
prereg loss 1.674788 reg_l1 32.239964 reg_l2 14.142844
loss 8.122781
STEP 133 ================================
prereg loss 1.6725821 reg_l1 32.23164 reg_l2 14.1383
loss 8.11891
STEP 134 ================================
prereg loss 1.670958 reg_l1 32.22271 reg_l2 14.133601
loss 8.1154995
STEP 135 ================================
prereg loss 1.6690248 reg_l1 32.213337 reg_l2 14.1288395
loss 8.111692
STEP 136 ================================
prereg loss 1.6670225 reg_l1 32.203777 reg_l2 14.124026
loss 8.107778
STEP 137 ================================
prereg loss 1.6652517 reg_l1 32.1952 reg_l2 14.119425
loss 8.104292
STEP 138 ================================
prereg loss 1.663108 reg_l1 32.18684 reg_l2 14.115045
loss 8.100476
STEP 139 ================================
prereg loss 1.6611654 reg_l1 32.17773 reg_l2 14.110583
loss 8.096712
STEP 140 ================================
prereg loss 1.6593599 reg_l1 32.16966 reg_l2 14.106052
loss 8.093292
STEP 141 ================================
prereg loss 1.6572341 reg_l1 32.161995 reg_l2 14.101804
loss 8.089633
STEP 142 ================================
prereg loss 1.6553227 reg_l1 32.154766 reg_l2 14.097803
loss 8.086276
STEP 143 ================================
prereg loss 1.6531441 reg_l1 32.147106 reg_l2 14.093709
loss 8.082565
STEP 144 ================================
prereg loss 1.6511005 reg_l1 32.13855 reg_l2 14.089508
loss 8.078811
STEP 145 ================================
prereg loss 1.6490737 reg_l1 32.130688 reg_l2 14.08543
loss 8.075212
STEP 146 ================================
prereg loss 1.6469746 reg_l1 32.123066 reg_l2 14.081418
loss 8.0715885
STEP 147 ================================
prereg loss 1.645058 reg_l1 32.115 reg_l2 14.077222
loss 8.068058
STEP 148 ================================
prereg loss 1.6430075 reg_l1 32.10662 reg_l2 14.072881
loss 8.064332
STEP 149 ================================
prereg loss 1.6412312 reg_l1 32.09793 reg_l2 14.068368
loss 8.060818
STEP 150 ================================
prereg loss 1.6392553 reg_l1 32.08936 reg_l2 14.063942
loss 8.057127
STEP 151 ================================
prereg loss 1.637534 reg_l1 32.081005 reg_l2 14.059607
loss 8.053735
STEP 152 ================================
prereg loss 1.6357292 reg_l1 32.07225 reg_l2 14.055155
loss 8.0501795
STEP 153 ================================
prereg loss 1.6336898 reg_l1 32.063457 reg_l2 14.050611
loss 8.046381
STEP 154 ================================
prereg loss 1.6318007 reg_l1 32.054607 reg_l2 14.046221
loss 8.042723
STEP 155 ================================
prereg loss 1.6300304 reg_l1 32.047234 reg_l2 14.041963
loss 8.039477
STEP 156 ================================
prereg loss 1.6281114 reg_l1 32.039463 reg_l2 14.037575
loss 8.036004
STEP 157 ================================
prereg loss 1.6262112 reg_l1 32.031178 reg_l2 14.03306
loss 8.032447
STEP 158 ================================
prereg loss 1.6242875 reg_l1 32.02248 reg_l2 14.028618
loss 8.028784
STEP 159 ================================
prereg loss 1.6225821 reg_l1 32.013794 reg_l2 14.024225
loss 8.025341
STEP 160 ================================
prereg loss 1.6208376 reg_l1 32.005653 reg_l2 14.01968
loss 8.021968
STEP 161 ================================
prereg loss 1.6189013 reg_l1 31.99732 reg_l2 14.01502
loss 8.018366
STEP 162 ================================
prereg loss 1.6173686 reg_l1 31.988073 reg_l2 14.0102
loss 8.014983
STEP 163 ================================
prereg loss 1.6155837 reg_l1 31.978481 reg_l2 14.005467
loss 8.01128
STEP 164 ================================
prereg loss 1.6138521 reg_l1 31.969494 reg_l2 14.000844
loss 8.007751
STEP 165 ================================
prereg loss 1.6122618 reg_l1 31.960793 reg_l2 13.996123
loss 8.00442
STEP 166 ================================
prereg loss 1.6103644 reg_l1 31.951683 reg_l2 13.991344
loss 8.000701
STEP 167 ================================
prereg loss 1.6086415 reg_l1 31.942558 reg_l2 13.986788
loss 7.9971533
STEP 168 ================================
prereg loss 1.6070434 reg_l1 31.934254 reg_l2 13.982431
loss 7.993894
STEP 169 ================================
prereg loss 1.6051754 reg_l1 31.925602 reg_l2 13.977987
loss 7.990296
STEP 170 ================================
prereg loss 1.6034348 reg_l1 31.916788 reg_l2 13.973445
loss 7.9867926
STEP 171 ================================
prereg loss 1.6016182 reg_l1 31.90817 reg_l2 13.969057
loss 7.9832525
STEP 172 ================================
prereg loss 1.5998425 reg_l1 31.899708 reg_l2 13.964759
loss 7.979784
STEP 173 ================================
prereg loss 1.5981348 reg_l1 31.891525 reg_l2 13.96033
loss 7.9764395
STEP 174 ================================
prereg loss 1.5962746 reg_l1 31.882385 reg_l2 13.955785
loss 7.9727516
STEP 175 ================================
prereg loss 1.5948627 reg_l1 31.873339 reg_l2 13.951086
loss 7.96953
STEP 176 ================================
prereg loss 1.5931189 reg_l1 31.864248 reg_l2 13.946514
loss 7.965969
STEP 177 ================================
prereg loss 1.591373 reg_l1 31.855175 reg_l2 13.942078
loss 7.962408
STEP 178 ================================
prereg loss 1.5897324 reg_l1 31.846636 reg_l2 13.937543
loss 7.9590597
STEP 179 ================================
prereg loss 1.5878211 reg_l1 31.83748 reg_l2 13.932934
loss 7.955317
STEP 180 ================================
prereg loss 1.5861055 reg_l1 31.828558 reg_l2 13.928489
loss 7.951817
STEP 181 ================================
prereg loss 1.5845561 reg_l1 31.819643 reg_l2 13.923903
loss 7.948485
STEP 182 ================================
prereg loss 1.5827817 reg_l1 31.810741 reg_l2 13.919492
loss 7.94493
STEP 183 ================================
prereg loss 1.5811449 reg_l1 31.801172 reg_l2 13.914951
loss 7.9413795
STEP 184 ================================
prereg loss 1.5796341 reg_l1 31.791954 reg_l2 13.910598
loss 7.938025
STEP 185 ================================
prereg loss 1.5778087 reg_l1 31.783165 reg_l2 13.906159
loss 7.934442
STEP 186 ================================
prereg loss 1.5763133 reg_l1 31.77474 reg_l2 13.901599
loss 7.931261
STEP 187 ================================
prereg loss 1.5745854 reg_l1 31.766037 reg_l2 13.897227
loss 7.927793
STEP 188 ================================
prereg loss 1.5726742 reg_l1 31.757511 reg_l2 13.893012
loss 7.9241767
STEP 189 ================================
prereg loss 1.5710789 reg_l1 31.748533 reg_l2 13.888641
loss 7.9207854
STEP 190 ================================
prereg loss 1.569372 reg_l1 31.739258 reg_l2 13.884154
loss 7.917224
STEP 191 ================================
prereg loss 1.5677215 reg_l1 31.730396 reg_l2 13.87982
loss 7.9138007
STEP 192 ================================
prereg loss 1.5661888 reg_l1 31.720695 reg_l2 13.875338
loss 7.910328
STEP 193 ================================
prereg loss 1.5645537 reg_l1 31.711206 reg_l2 13.871017
loss 7.906795
STEP 194 ================================
prereg loss 1.5629122 reg_l1 31.701803 reg_l2 13.866582
loss 7.9032726
STEP 195 ================================
prereg loss 1.5615144 reg_l1 31.69289 reg_l2 13.86204
loss 7.9000926
STEP 196 ================================
prereg loss 1.5597888 reg_l1 31.684277 reg_l2 13.857697
loss 7.896644
STEP 197 ================================
prereg loss 1.5582502 reg_l1 31.675888 reg_l2 13.853538
loss 7.893428
STEP 198 ================================
prereg loss 1.5565395 reg_l1 31.666447 reg_l2 13.849284
loss 7.889829
STEP 199 ================================
prereg loss 1.5547625 reg_l1 31.657204 reg_l2 13.844923
loss 7.8862033
STEP 200 ================================
prereg loss 1.5531211 reg_l1 31.64793 reg_l2 13.840638
loss 7.882707
STEP 201 ================================
prereg loss 1.5516194 reg_l1 31.63905 reg_l2 13.836401
loss 7.8794293
STEP 202 ================================
prereg loss 1.5500579 reg_l1 31.629953 reg_l2 13.832027
loss 7.8760486
STEP 203 ================================
prereg loss 1.5483365 reg_l1 31.619982 reg_l2 13.827541
loss 7.872333
STEP 204 ================================
prereg loss 1.5470874 reg_l1 31.609734 reg_l2 13.822896
loss 7.869034
STEP 205 ================================
prereg loss 1.5454967 reg_l1 31.600454 reg_l2 13.818377
loss 7.865587
STEP 206 ================================
prereg loss 1.543958 reg_l1 31.5912 reg_l2 13.814
loss 7.862198
STEP 207 ================================
prereg loss 1.542528 reg_l1 31.581457 reg_l2 13.809554
loss 7.8588195
STEP 208 ================================
prereg loss 1.5408732 reg_l1 31.571321 reg_l2 13.80507
loss 7.8551373
STEP 209 ================================
prereg loss 1.5393355 reg_l1 31.562223 reg_l2 13.80083
loss 7.85178
STEP 210 ================================
prereg loss 1.5377235 reg_l1 31.55335 reg_l2 13.796811
loss 7.848394
STEP 211 ================================
prereg loss 1.5360683 reg_l1 31.544018 reg_l2 13.792682
loss 7.844872
STEP 212 ================================
prereg loss 1.5346247 reg_l1 31.534826 reg_l2 13.788432
loss 7.84159
STEP 213 ================================
prereg loss 1.5329568 reg_l1 31.525305 reg_l2 13.784323
loss 7.8380175
STEP 214 ================================
prereg loss 1.531456 reg_l1 31.516546 reg_l2 13.780309
loss 7.8347654
STEP 215 ================================
prereg loss 1.5299124 reg_l1 31.50809 reg_l2 13.776167
loss 7.8315306
STEP 216 ================================
prereg loss 1.5281708 reg_l1 31.498817 reg_l2 13.771909
loss 7.8279343
STEP 217 ================================
prereg loss 1.5269077 reg_l1 31.489065 reg_l2 13.767476
loss 7.8247204
STEP 218 ================================
prereg loss 1.5253564 reg_l1 31.479256 reg_l2 13.763093
loss 7.8212075
STEP 219 ================================
prereg loss 1.5238488 reg_l1 31.469025 reg_l2 13.758791
loss 7.8176537
STEP 220 ================================
prereg loss 1.5225135 reg_l1 31.459604 reg_l2 13.754392
loss 7.8144345
STEP 221 ================================
prereg loss 1.5208218 reg_l1 31.450039 reg_l2 13.749942
loss 7.81083
STEP 222 ================================
prereg loss 1.5193702 reg_l1 31.440403 reg_l2 13.745701
loss 7.807451
STEP 223 ================================
prereg loss 1.5179436 reg_l1 31.430832 reg_l2 13.741649
loss 7.8041105
STEP 224 ================================
prereg loss 1.5163047 reg_l1 31.42123 reg_l2 13.737507
loss 7.8005505
STEP 225 ================================
prereg loss 1.5149432 reg_l1 31.411768 reg_l2 13.733251
loss 7.797297
STEP 226 ================================
prereg loss 1.513389 reg_l1 31.402212 reg_l2 13.729135
loss 7.793832
STEP 227 ================================
prereg loss 1.5118774 reg_l1 31.393425 reg_l2 13.725135
loss 7.7905626
STEP 228 ================================
prereg loss 1.5104572 reg_l1 31.38454 reg_l2 13.721024
loss 7.7873654
STEP 229 ================================
prereg loss 1.508821 reg_l1 31.376722 reg_l2 13.716819
loss 7.7841654
STEP 230 ================================
prereg loss 1.5077752 reg_l1 31.370203 reg_l2 13.712477
loss 7.781816
STEP 231 ================================
prereg loss 1.5062685 reg_l1 31.363838 reg_l2 13.708301
loss 7.779036
STEP 232 ================================
prereg loss 1.5045365 reg_l1 31.357643 reg_l2 13.704301
loss 7.7760653
STEP 233 ================================
prereg loss 1.5031307 reg_l1 31.350481 reg_l2 13.700189
loss 7.7732267
STEP 234 ================================
prereg loss 1.5015365 reg_l1 31.343018 reg_l2 13.695994
loss 7.77014
STEP 235 ================================
prereg loss 1.4999974 reg_l1 31.33577 reg_l2 13.691969
loss 7.767152
STEP 236 ================================
prereg loss 1.4986286 reg_l1 31.32835 reg_l2 13.688079
loss 7.764299
STEP 237 ================================
prereg loss 1.4970738 reg_l1 31.320395 reg_l2 13.684048
loss 7.7611527
STEP 238 ================================
prereg loss 1.4957035 reg_l1 31.31203 reg_l2 13.679864
loss 7.75811
STEP 239 ================================
prereg loss 1.494258 reg_l1 31.302845 reg_l2 13.675735
loss 7.754827
STEP 240 ================================
prereg loss 1.4929428 reg_l1 31.29416 reg_l2 13.671666
loss 7.7517753
STEP 241 ================================
prereg loss 1.491614 reg_l1 31.285618 reg_l2 13.667471
loss 7.7487373
STEP 242 ================================
prereg loss 1.4900424 reg_l1 31.276398 reg_l2 13.663192
loss 7.745322
STEP 243 ================================
prereg loss 1.4890944 reg_l1 31.267658 reg_l2 13.658783
loss 7.742626
STEP 244 ================================
prereg loss 1.4877043 reg_l1 31.260317 reg_l2 13.654547
loss 7.7397676
STEP 245 ================================
prereg loss 1.4860282 reg_l1 31.254154 reg_l2 13.650515
loss 7.7368593
STEP 246 ================================
prereg loss 1.4847537 reg_l1 31.246815 reg_l2 13.646384
loss 7.7341166
STEP 247 ================================
prereg loss 1.4833752 reg_l1 31.239136 reg_l2 13.642206
loss 7.731202
STEP 248 ================================
prereg loss 1.4818969 reg_l1 31.231266 reg_l2 13.638307
loss 7.7281504
STEP 249 ================================
prereg loss 1.4803985 reg_l1 31.223408 reg_l2 13.634625
loss 7.7250805
STEP 250 ================================
prereg loss 1.478863 reg_l1 31.215963 reg_l2 13.630822
loss 7.7220554
STEP 251 ================================
prereg loss 1.4776392 reg_l1 31.207632 reg_l2 13.626872
loss 7.719166
STEP 252 ================================
prereg loss 1.4761684 reg_l1 31.19882 reg_l2 13.623016
loss 7.715933
STEP 253 ================================
prereg loss 1.4748116 reg_l1 31.19071 reg_l2 13.619225
loss 7.7129536
STEP 254 ================================
prereg loss 1.4734781 reg_l1 31.183268 reg_l2 13.615304
loss 7.7101316
STEP 255 ================================
prereg loss 1.4718654 reg_l1 31.175875 reg_l2 13.611266
loss 7.707041
STEP 256 ================================
prereg loss 1.4707785 reg_l1 31.167885 reg_l2 13.607069
loss 7.7043557
STEP 257 ================================
prereg loss 1.4694254 reg_l1 31.160164 reg_l2 13.602938
loss 7.701458
STEP 258 ================================
prereg loss 1.468009 reg_l1 31.151766 reg_l2 13.598912
loss 7.6983624
STEP 259 ================================
prereg loss 1.4668498 reg_l1 31.142769 reg_l2 13.594781
loss 7.6954036
STEP 260 ================================
prereg loss 1.4655099 reg_l1 31.133802 reg_l2 13.590606
loss 7.6922703
STEP 261 ================================
prereg loss 1.4641651 reg_l1 31.126457 reg_l2 13.586668
loss 7.689457
STEP 262 ================================
prereg loss 1.462796 reg_l1 31.119186 reg_l2 13.582939
loss 7.686633
STEP 263 ================================
prereg loss 1.4612898 reg_l1 31.11135 reg_l2 13.5791235
loss 7.68356
STEP 264 ================================
prereg loss 1.4600353 reg_l1 31.103092 reg_l2 13.575169
loss 7.680654
STEP 265 ================================
prereg loss 1.4586337 reg_l1 31.09552 reg_l2 13.571324
loss 7.677738
STEP 266 ================================
prereg loss 1.4572768 reg_l1 31.08793 reg_l2 13.567563
loss 7.674863
STEP 267 ================================
prereg loss 1.4560294 reg_l1 31.079706 reg_l2 13.5636635
loss 7.671971
STEP 268 ================================
prereg loss 1.4545274 reg_l1 31.071934 reg_l2 13.559649
loss 7.6689143
STEP 269 ================================
prereg loss 1.4536608 reg_l1 31.063492 reg_l2 13.555482
loss 7.6663594
STEP 270 ================================
prereg loss 1.4523556 reg_l1 31.055426 reg_l2 13.551464
loss 7.6634407
STEP 271 ================================
prereg loss 1.4508281 reg_l1 31.047424 reg_l2 13.547623
loss 7.660313
STEP 272 ================================
prereg loss 1.449653 reg_l1 31.039068 reg_l2 13.543679
loss 7.657467
STEP 273 ================================
prereg loss 1.4484525 reg_l1 31.03031 reg_l2 13.539674
loss 7.6545143
STEP 274 ================================
prereg loss 1.4471269 reg_l1 31.022316 reg_l2 13.535933
loss 7.6515903
STEP 275 ================================
prereg loss 1.445702 reg_l1 31.015442 reg_l2 13.532413
loss 7.6487904
STEP 276 ================================
prereg loss 1.4442422 reg_l1 31.007956 reg_l2 13.5287695
loss 7.645833
STEP 277 ================================
prereg loss 1.4429989 reg_l1 31.00042 reg_l2 13.524978
loss 7.643083
STEP 278 ================================
prereg loss 1.4415867 reg_l1 30.992458 reg_l2 13.52125
loss 7.6400785
STEP 279 ================================
prereg loss 1.4403783 reg_l1 30.984695 reg_l2 13.517565
loss 7.637317
STEP 280 ================================
prereg loss 1.4391056 reg_l1 30.976763 reg_l2 13.513743
loss 7.634458
STEP 281 ================================
prereg loss 1.4376378 reg_l1 30.968939 reg_l2 13.509802
loss 7.631426
STEP 282 ================================
prereg loss 1.4368925 reg_l1 30.96018 reg_l2 13.505663
loss 7.6289287
STEP 283 ================================
prereg loss 1.4357189 reg_l1 30.95092 reg_l2 13.501634
loss 7.625903
STEP 284 ================================
prereg loss 1.4342036 reg_l1 30.942398 reg_l2 13.497762
loss 7.6226835
STEP 285 ================================
prereg loss 1.4331851 reg_l1 30.93413 reg_l2 13.493791
loss 7.6200113
STEP 286 ================================
prereg loss 1.4320487 reg_l1 30.92589 reg_l2 13.489786
loss 7.617227
STEP 287 ================================
prereg loss 1.430689 reg_l1 30.918678 reg_l2 13.4861
loss 7.6144247
STEP 288 ================================
prereg loss 1.4292375 reg_l1 30.911514 reg_l2 13.482662
loss 7.6115403
STEP 289 ================================
prereg loss 1.4277998 reg_l1 30.903555 reg_l2 13.479105
loss 7.608511
STEP 290 ================================
prereg loss 1.4267708 reg_l1 30.895145 reg_l2 13.475386
loss 7.6057997
STEP 291 ================================
prereg loss 1.4254574 reg_l1 30.886786 reg_l2 13.471749
loss 7.6028147
STEP 292 ================================
prereg loss 1.4241484 reg_l1 30.879051 reg_l2 13.468166
loss 7.599959
STEP 293 ================================
prereg loss 1.4229807 reg_l1 30.871346 reg_l2 13.464445
loss 7.59725
STEP 294 ================================
prereg loss 1.4215434 reg_l1 30.862934 reg_l2 13.4606285
loss 7.5941305
STEP 295 ================================
prereg loss 1.4207274 reg_l1 30.854664 reg_l2 13.456642
loss 7.59166
STEP 296 ================================
prereg loss 1.4195311 reg_l1 30.84621 reg_l2 13.452772
loss 7.5887737
STEP 297 ================================
prereg loss 1.4180065 reg_l1 30.838297 reg_l2 13.449057
loss 7.5856657
STEP 298 ================================
prereg loss 1.4169629 reg_l1 30.829536 reg_l2 13.445219
loss 7.5828705
STEP 299 ================================
prereg loss 1.4159683 reg_l1 30.82069 reg_l2 13.441326
loss 7.5801067
STEP 300 ================================
prereg loss 1.4147217 reg_l1 30.812925 reg_l2 13.437738
loss 7.5773067
STEP 301 ================================
prereg loss 1.4132631 reg_l1 30.806074 reg_l2 13.43441
loss 7.574478
STEP 302 ================================
prereg loss 1.4118341 reg_l1 30.798304 reg_l2 13.430963
loss 7.571495
STEP 303 ================================
prereg loss 1.4108142 reg_l1 30.789688 reg_l2 13.427358
loss 7.568752
STEP 304 ================================
prereg loss 1.4095169 reg_l1 30.7815 reg_l2 13.4238405
loss 7.565817
STEP 305 ================================
prereg loss 1.4082321 reg_l1 30.773535 reg_l2 13.420377
loss 7.562939
STEP 306 ================================
prereg loss 1.4071676 reg_l1 30.765486 reg_l2 13.416763
loss 7.5602646
STEP 307 ================================
prereg loss 1.4058499 reg_l1 30.757015 reg_l2 13.413042
loss 7.557253
STEP 308 ================================
prereg loss 1.4050864 reg_l1 30.747852 reg_l2 13.409163
loss 7.554657
STEP 309 ================================
prereg loss 1.4038771 reg_l1 30.739588 reg_l2 13.405432
loss 7.551795
STEP 310 ================================
prereg loss 1.402437 reg_l1 30.732044 reg_l2 13.401868
loss 7.5488462
STEP 311 ================================
prereg loss 1.4013305 reg_l1 30.72388 reg_l2 13.398209
loss 7.546107
STEP 312 ================================
prereg loss 1.4003026 reg_l1 30.714956 reg_l2 13.394484
loss 7.543294
STEP 313 ================================
prereg loss 1.3991122 reg_l1 30.706865 reg_l2 13.391018
loss 7.5404854
STEP 314 ================================
prereg loss 1.3976518 reg_l1 30.698927 reg_l2 13.387766
loss 7.537437
STEP 315 ================================
prereg loss 1.3964593 reg_l1 30.690577 reg_l2 13.384364
loss 7.5345745
STEP 316 ================================
prereg loss 1.3955376 reg_l1 30.682365 reg_l2 13.380815
loss 7.532011
STEP 317 ================================
prereg loss 1.3942629 reg_l1 30.674467 reg_l2 13.377404
loss 7.529156
STEP 318 ================================
prereg loss 1.3932002 reg_l1 30.666174 reg_l2 13.37411
loss 7.526435
STEP 319 ================================
prereg loss 1.3919709 reg_l1 30.657959 reg_l2 13.370723
loss 7.5235624
STEP 320 ================================
prereg loss 1.3905953 reg_l1 30.649872 reg_l2 13.367228
loss 7.52057
STEP 321 ================================
prereg loss 1.3894079 reg_l1 30.6417 reg_l2 13.363749
loss 7.517748
STEP 322 ================================
prereg loss 1.3884519 reg_l1 30.633371 reg_l2 13.3602705
loss 7.515126
STEP 323 ================================
prereg loss 1.3873414 reg_l1 30.624851 reg_l2 13.35668
loss 7.512312
STEP 324 ================================
prereg loss 1.3860219 reg_l1 30.615904 reg_l2 13.352994
loss 7.509203
STEP 325 ================================
prereg loss 1.3848702 reg_l1 30.608059 reg_l2 13.349373
loss 7.506482
STEP 326 ================================
prereg loss 1.3839313 reg_l1 30.599945 reg_l2 13.345581
loss 7.50392
STEP 327 ================================
prereg loss 1.3828285 reg_l1 30.591158 reg_l2 13.341941
loss 7.5010605
STEP 328 ================================
prereg loss 1.3818287 reg_l1 30.58186 reg_l2 13.338226
loss 7.498201
STEP 329 ================================
prereg loss 1.3808186 reg_l1 30.573328 reg_l2 13.334766
loss 7.4954844
STEP 330 ================================
prereg loss 1.3797386 reg_l1 30.564629 reg_l2 13.331266
loss 7.4926643
STEP 331 ================================
prereg loss 1.3785006 reg_l1 30.556181 reg_l2 13.328007
loss 7.4897366
STEP 332 ================================
prereg loss 1.3776416 reg_l1 30.548319 reg_l2 13.32491
loss 7.4873056
STEP 333 ================================
prereg loss 1.37627 reg_l1 30.54068 reg_l2 13.321754
loss 7.4844065
STEP 334 ================================
prereg loss 1.3749113 reg_l1 30.532667 reg_l2 13.318489
loss 7.481445
STEP 335 ================================
prereg loss 1.3741629 reg_l1 30.524595 reg_l2 13.314954
loss 7.479082
STEP 336 ================================
prereg loss 1.3729084 reg_l1 30.51554 reg_l2 13.311385
loss 7.476016
STEP 337 ================================
prereg loss 1.3720974 reg_l1 30.506495 reg_l2 13.3078575
loss 7.4733963
STEP 338 ================================
prereg loss 1.3712718 reg_l1 30.49733 reg_l2 13.30431
loss 7.4707375
STEP 339 ================================
prereg loss 1.3699101 reg_l1 30.48783 reg_l2 13.300797
loss 7.4674764
STEP 340 ================================
prereg loss 1.3689548 reg_l1 30.479797 reg_l2 13.297513
loss 7.4649143
STEP 341 ================================
prereg loss 1.3676381 reg_l1 30.472248 reg_l2 13.294421
loss 7.4620876
STEP 342 ================================
prereg loss 1.3664591 reg_l1 30.463642 reg_l2 13.2912
loss 7.4591875
STEP 343 ================================
prereg loss 1.3655742 reg_l1 30.454988 reg_l2 13.287857
loss 7.456572
STEP 344 ================================
prereg loss 1.3643788 reg_l1 30.44689 reg_l2 13.284678
loss 7.453757
STEP 345 ================================
prereg loss 1.3634403 reg_l1 30.438856 reg_l2 13.28165
loss 7.451212
STEP 346 ================================
prereg loss 1.3622457 reg_l1 30.430141 reg_l2 13.278552
loss 7.448274
STEP 347 ================================
prereg loss 1.3611752 reg_l1 30.421438 reg_l2 13.275365
loss 7.4454627
STEP 348 ================================
prereg loss 1.3601644 reg_l1 30.413063 reg_l2 13.272246
loss 7.4427767
STEP 349 ================================
prereg loss 1.3589745 reg_l1 30.404755 reg_l2 13.269186
loss 7.4399257
STEP 350 ================================
prereg loss 1.3579371 reg_l1 30.396786 reg_l2 13.265977
loss 7.437294
STEP 351 ================================
prereg loss 1.3567317 reg_l1 30.38831 reg_l2 13.262677
loss 7.434394
STEP 352 ================================
prereg loss 1.3556508 reg_l1 30.379658 reg_l2 13.259475
loss 7.4315825
STEP 353 ================================
prereg loss 1.3547397 reg_l1 30.370846 reg_l2 13.256119
loss 7.428909
STEP 354 ================================
prereg loss 1.3536566 reg_l1 30.362146 reg_l2 13.252896
loss 7.426086
STEP 355 ================================
prereg loss 1.3527472 reg_l1 30.352848 reg_l2 13.249553
loss 7.423317
STEP 356 ================================
prereg loss 1.3517992 reg_l1 30.343832 reg_l2 13.246398
loss 7.4205656
STEP 357 ================================
prereg loss 1.3507882 reg_l1 30.33569 reg_l2 13.243163
loss 7.4179263
STEP 358 ================================
prereg loss 1.3497416 reg_l1 30.327436 reg_l2 13.240122
loss 7.415229
STEP 359 ================================
prereg loss 1.3485892 reg_l1 30.318306 reg_l2 13.236976
loss 7.4122505
STEP 360 ================================
prereg loss 1.347523 reg_l1 30.310085 reg_l2 13.233961
loss 7.40954
STEP 361 ================================
prereg loss 1.3464029 reg_l1 30.301825 reg_l2 13.230815
loss 7.406768
STEP 362 ================================
prereg loss 1.3457117 reg_l1 30.293028 reg_l2 13.227503
loss 7.4043174
STEP 363 ================================
prereg loss 1.344661 reg_l1 30.283684 reg_l2 13.224318
loss 7.4013977
STEP 364 ================================
prereg loss 1.3437858 reg_l1 30.274292 reg_l2 13.221285
loss 7.3986444
STEP 365 ================================
prereg loss 1.3427474 reg_l1 30.266071 reg_l2 13.218206
loss 7.3959618
STEP 366 ================================
prereg loss 1.341718 reg_l1 30.25788 reg_l2 13.21508
loss 7.3932943
STEP 367 ================================
prereg loss 1.3407257 reg_l1 30.24905 reg_l2 13.212102
loss 7.3905354
STEP 368 ================================
prereg loss 1.3395072 reg_l1 30.240282 reg_l2 13.209228
loss 7.3875637
STEP 369 ================================
prereg loss 1.3385067 reg_l1 30.232075 reg_l2 13.206217
loss 7.3849216
STEP 370 ================================
prereg loss 1.3375013 reg_l1 30.223654 reg_l2 13.203088
loss 7.3822317
STEP 371 ================================
prereg loss 1.3364406 reg_l1 30.214874 reg_l2 13.200057
loss 7.3794155
STEP 372 ================================
prereg loss 1.335757 reg_l1 30.206001 reg_l2 13.19711
loss 7.376957
STEP 373 ================================
prereg loss 1.3346717 reg_l1 30.197607 reg_l2 13.194108
loss 7.374193
STEP 374 ================================
prereg loss 1.3336337 reg_l1 30.188477 reg_l2 13.191018
loss 7.3713293
STEP 375 ================================
prereg loss 1.3326483 reg_l1 30.179768 reg_l2 13.187975
loss 7.368602
STEP 376 ================================
prereg loss 1.3316063 reg_l1 30.171139 reg_l2 13.184976
loss 7.365834
STEP 377 ================================
prereg loss 1.330686 reg_l1 30.162607 reg_l2 13.181845
loss 7.363208
STEP 378 ================================
prereg loss 1.3297018 reg_l1 30.15368 reg_l2 13.178638
loss 7.3604383
STEP 379 ================================
prereg loss 1.3287462 reg_l1 30.144093 reg_l2 13.175575
loss 7.357565
STEP 380 ================================
prereg loss 1.3279716 reg_l1 30.135147 reg_l2 13.172654
loss 7.355001
STEP 381 ================================
prereg loss 1.3268944 reg_l1 30.126408 reg_l2 13.169693
loss 7.3521757
STEP 382 ================================
prereg loss 1.3261541 reg_l1 30.117561 reg_l2 13.166645
loss 7.3496666
STEP 383 ================================
prereg loss 1.325205 reg_l1 30.108006 reg_l2 13.163723
loss 7.346806
STEP 384 ================================
prereg loss 1.324042 reg_l1 30.098978 reg_l2 13.160914
loss 7.3438377
STEP 385 ================================
prereg loss 1.3231517 reg_l1 30.090664 reg_l2 13.158004
loss 7.3412843
STEP 386 ================================
prereg loss 1.3221239 reg_l1 30.08207 reg_l2 13.155018
loss 7.338538
STEP 387 ================================
prereg loss 1.3210998 reg_l1 30.07334 reg_l2 13.152185
loss 7.3357677
STEP 388 ================================
prereg loss 1.320398 reg_l1 30.064407 reg_l2 13.149484
loss 7.3332796
STEP 389 ================================
prereg loss 1.3192693 reg_l1 30.055645 reg_l2 13.146733
loss 7.330398
STEP 390 ================================
prereg loss 1.3183776 reg_l1 30.047235 reg_l2 13.143879
loss 7.3278246
STEP 391 ================================
prereg loss 1.3174248 reg_l1 30.038338 reg_l2 13.141048
loss 7.3250923
STEP 392 ================================
prereg loss 1.3163719 reg_l1 30.029116 reg_l2 13.1382265
loss 7.322195
STEP 393 ================================
prereg loss 1.3155721 reg_l1 30.020119 reg_l2 13.135274
loss 7.3195963
STEP 394 ================================
prereg loss 1.3144883 reg_l1 30.011143 reg_l2 13.132255
loss 7.316717
STEP 395 ================================
prereg loss 1.3135947 reg_l1 30.001999 reg_l2 13.129357
loss 7.313995
STEP 396 ================================
prereg loss 1.3128668 reg_l1 29.992516 reg_l2 13.126338
loss 7.31137
STEP 397 ================================
prereg loss 1.3118747 reg_l1 29.983707 reg_l2 13.123496
loss 7.3086166
STEP 398 ================================
prereg loss 1.3113179 reg_l1 29.974804 reg_l2 13.120836
loss 7.3062787
STEP 399 ================================
prereg loss 1.3101271 reg_l1 29.965536 reg_l2 13.118189
loss 7.3032346
STEP 400 ================================
prereg loss 1.3092377 reg_l1 29.956787 reg_l2 13.115455
loss 7.3005953
STEP 401 ================================
prereg loss 1.3083218 reg_l1 29.948265 reg_l2 13.112772
loss 7.297975
STEP 402 ================================
prereg loss 1.3071222 reg_l1 29.939423 reg_l2 13.110098
loss 7.2950068
STEP 403 ================================
prereg loss 1.3063843 reg_l1 29.929625 reg_l2 13.107237
loss 7.2923098
STEP 404 ================================
prereg loss 1.3053939 reg_l1 29.920269 reg_l2 13.104303
loss 7.289448
STEP 405 ================================
prereg loss 1.3049768 reg_l1 29.91121 reg_l2 13.101266
loss 7.287219
STEP 406 ================================
prereg loss 1.3041792 reg_l1 29.902054 reg_l2 13.098463
loss 7.2845902
STEP 407 ================================
prereg loss 1.3030263 reg_l1 29.89348 reg_l2 13.095926
loss 7.2817225
STEP 408 ================================
prereg loss 1.3026625 reg_l1 29.885662 reg_l2 13.093637
loss 7.2797947
STEP 409 ================================
prereg loss 1.3012235 reg_l1 29.878513 reg_l2 13.091413
loss 7.276926
STEP 410 ================================
prereg loss 1.2998546 reg_l1 29.871433 reg_l2 13.08912
loss 7.2741413
STEP 411 ================================
prereg loss 1.2990048 reg_l1 29.86348 reg_l2 13.086702
loss 7.271701
STEP 412 ================================
prereg loss 1.2978714 reg_l1 29.854683 reg_l2 13.084082
loss 7.2688084
STEP 413 ================================
prereg loss 1.2972001 reg_l1 29.845928 reg_l2 13.081172
loss 7.266386
STEP 414 ================================
prereg loss 1.296338 reg_l1 29.83651 reg_l2 13.078153
loss 7.26364
STEP 415 ================================
prereg loss 1.2953984 reg_l1 29.827013 reg_l2 13.07504
loss 7.260801
STEP 416 ================================
prereg loss 1.2946339 reg_l1 29.817486 reg_l2 13.07201
loss 7.258131
STEP 417 ================================
prereg loss 1.2939004 reg_l1 29.808895 reg_l2 13.069119
loss 7.2556796
STEP 418 ================================
prereg loss 1.2928568 reg_l1 29.800663 reg_l2 13.066189
loss 7.2529893
STEP 419 ================================
prereg loss 1.2923098 reg_l1 29.793177 reg_l2 13.063179
loss 7.250945
STEP 420 ================================
prereg loss 1.2914957 reg_l1 29.785782 reg_l2 13.060369
loss 7.2486525
STEP 421 ================================
prereg loss 1.2903321 reg_l1 29.779192 reg_l2 13.057737
loss 7.246171
STEP 422 ================================
prereg loss 1.2897054 reg_l1 29.771694 reg_l2 13.055257
loss 7.2440443
STEP 423 ================================
prereg loss 1.2885944 reg_l1 29.763474 reg_l2 13.05275
loss 7.241289
STEP 424 ================================
prereg loss 1.2872927 reg_l1 29.75541 reg_l2 13.050161
loss 7.2383747
STEP 425 ================================
prereg loss 1.286499 reg_l1 29.747976 reg_l2 13.047458
loss 7.2360945
STEP 426 ================================
prereg loss 1.285627 reg_l1 29.739525 reg_l2 13.044596
loss 7.233532
STEP 427 ================================
prereg loss 1.2849207 reg_l1 29.729694 reg_l2 13.041531
loss 7.2308598
STEP 428 ================================
prereg loss 1.2840134 reg_l1 29.720032 reg_l2 13.03835
loss 7.2280197
STEP 429 ================================
prereg loss 1.2834529 reg_l1 29.711185 reg_l2 13.0350485
loss 7.2256904
STEP 430 ================================
prereg loss 1.2827181 reg_l1 29.703346 reg_l2 13.031871
loss 7.2233877
STEP 431 ================================
prereg loss 1.2818277 reg_l1 29.694962 reg_l2 13.028896
loss 7.2208204
STEP 432 ================================
prereg loss 1.2809999 reg_l1 29.686356 reg_l2 13.025889
loss 7.2182713
STEP 433 ================================
prereg loss 1.280475 reg_l1 29.67842 reg_l2 13.022845
loss 7.216159
STEP 434 ================================
prereg loss 1.2795924 reg_l1 29.670074 reg_l2 13.020073
loss 7.2136073
STEP 435 ================================
prereg loss 1.2783786 reg_l1 29.662577 reg_l2 13.017512
loss 7.210894
STEP 436 ================================
prereg loss 1.2778966 reg_l1 29.655336 reg_l2 13.015094
loss 7.2089643
STEP 437 ================================
prereg loss 1.276674 reg_l1 29.647959 reg_l2 13.012662
loss 7.2062654
STEP 438 ================================
prereg loss 1.2755268 reg_l1 29.64025 reg_l2 13.010107
loss 7.203577
STEP 439 ================================
prereg loss 1.2747998 reg_l1 29.631369 reg_l2 13.007364
loss 7.2010736
STEP 440 ================================
prereg loss 1.2739468 reg_l1 29.622648 reg_l2 13.004417
loss 7.1984763
STEP 441 ================================
prereg loss 1.2733955 reg_l1 29.613249 reg_l2 13.001266
loss 7.1960454
STEP 442 ================================
prereg loss 1.2726266 reg_l1 29.60371 reg_l2 12.998051
loss 7.193369
STEP 443 ================================
prereg loss 1.2723038 reg_l1 29.594486 reg_l2 12.994774
loss 7.191201
STEP 444 ================================
prereg loss 1.2716873 reg_l1 29.585815 reg_l2 12.991777
loss 7.1888504
STEP 445 ================================
prereg loss 1.2706904 reg_l1 29.57788 reg_l2 12.989114
loss 7.186267
STEP 446 ================================
prereg loss 1.2703764 reg_l1 29.569706 reg_l2 12.986784
loss 7.1843176
STEP 447 ================================
prereg loss 1.2689389 reg_l1 29.562256 reg_l2 12.984578
loss 7.1813903
STEP 448 ================================
prereg loss 1.2678347 reg_l1 29.554441 reg_l2 12.982337
loss 7.178723
STEP 449 ================================
prereg loss 1.2671353 reg_l1 29.547161 reg_l2 12.979991
loss 7.1765676
STEP 450 ================================
prereg loss 1.265986 reg_l1 29.54172 reg_l2 12.977465
loss 7.1743298
STEP 451 ================================
prereg loss 1.2654053 reg_l1 29.53531 reg_l2 12.974644
loss 7.172467
STEP 452 ================================
prereg loss 1.2646984 reg_l1 29.529419 reg_l2 12.971745
loss 7.1705823
STEP 453 ================================
prereg loss 1.263848 reg_l1 29.523867 reg_l2 12.968811
loss 7.168621
STEP 454 ================================
prereg loss 1.2631718 reg_l1 29.517656 reg_l2 12.966038
loss 7.166703
STEP 455 ================================
prereg loss 1.2623078 reg_l1 29.511402 reg_l2 12.963451
loss 7.164588
STEP 456 ================================
prereg loss 1.2615871 reg_l1 29.504562 reg_l2 12.960775
loss 7.1625
STEP 457 ================================
prereg loss 1.260732 reg_l1 29.497654 reg_l2 12.958313
loss 7.160263
STEP 458 ================================
prereg loss 1.2600241 reg_l1 29.490374 reg_l2 12.955752
loss 7.1580987
STEP 459 ================================
prereg loss 1.2591522 reg_l1 29.482964 reg_l2 12.953322
loss 7.1557446
STEP 460 ================================
prereg loss 1.2586256 reg_l1 29.476198 reg_l2 12.950979
loss 7.1538653
STEP 461 ================================
prereg loss 1.2577354 reg_l1 29.469868 reg_l2 12.9485655
loss 7.151709
STEP 462 ================================
prereg loss 1.2568089 reg_l1 29.462452 reg_l2 12.946035
loss 7.149299
STEP 463 ================================
prereg loss 1.2560997 reg_l1 29.453875 reg_l2 12.943435
loss 7.146875
STEP 464 ================================
prereg loss 1.2554517 reg_l1 29.44562 reg_l2 12.940605
loss 7.1445756
STEP 465 ================================
prereg loss 1.2549655 reg_l1 29.437838 reg_l2 12.937614
loss 7.1425333
STEP 466 ================================
prereg loss 1.2543844 reg_l1 29.430613 reg_l2 12.93479
loss 7.140507
STEP 467 ================================
prereg loss 1.2537742 reg_l1 29.423475 reg_l2 12.931981
loss 7.138469
STEP 468 ================================
prereg loss 1.2529479 reg_l1 29.417086 reg_l2 12.92951
loss 7.136365
STEP 469 ================================
prereg loss 1.2524176 reg_l1 29.41157 reg_l2 12.927344
loss 7.134732
STEP 470 ================================
prereg loss 1.2511283 reg_l1 29.405842 reg_l2 12.925211
loss 7.1322966
STEP 471 ================================
prereg loss 1.2506388 reg_l1 29.3995 reg_l2 12.922921
loss 7.130539
STEP 472 ================================
prereg loss 1.2498775 reg_l1 29.392412 reg_l2 12.920597
loss 7.12836
STEP 473 ================================
prereg loss 1.2487874 reg_l1 29.38517 reg_l2 12.918198
loss 7.1258216
STEP 474 ================================
prereg loss 1.2483823 reg_l1 29.37795 reg_l2 12.915596
loss 7.123973
STEP 475 ================================
prereg loss 1.2475897 reg_l1 29.370182 reg_l2 12.912964
loss 7.121626
STEP 476 ================================
prereg loss 1.247126 reg_l1 29.362413 reg_l2 12.910271
loss 7.119609
STEP 477 ================================
prereg loss 1.2465234 reg_l1 29.355955 reg_l2 12.907759
loss 7.1177144
STEP 478 ================================
prereg loss 1.2455714 reg_l1 29.34969 reg_l2 12.90546
loss 7.115509
STEP 479 ================================
prereg loss 1.2452037 reg_l1 29.342773 reg_l2 12.903351
loss 7.113758
STEP 480 ================================
prereg loss 1.2440466 reg_l1 29.336552 reg_l2 12.901279
loss 7.111357
STEP 481 ================================
prereg loss 1.2430271 reg_l1 29.330965 reg_l2 12.899126
loss 7.1092205
STEP 482 ================================
prereg loss 1.2423682 reg_l1 29.324558 reg_l2 12.89689
loss 7.10728
STEP 483 ================================
prereg loss 1.2413783 reg_l1 29.317463 reg_l2 12.894503
loss 7.104871
STEP 484 ================================
prereg loss 1.2411913 reg_l1 29.310556 reg_l2 12.892017
loss 7.103303
STEP 485 ================================
prereg loss 1.24053 reg_l1 29.30382 reg_l2 12.889516
loss 7.101294
STEP 486 ================================
prereg loss 1.2392962 reg_l1 29.297152 reg_l2 12.887013
loss 7.0987263
STEP 487 ================================
prereg loss 1.239214 reg_l1 29.289904 reg_l2 12.884275
loss 7.0971947
STEP 488 ================================
prereg loss 1.2386924 reg_l1 29.28192 reg_l2 12.881534
loss 7.095076
STEP 489 ================================
prereg loss 1.2377771 reg_l1 29.274271 reg_l2 12.87887
loss 7.0926313
STEP 490 ================================
prereg loss 1.237575 reg_l1 29.267195 reg_l2 12.876413
loss 7.091014
STEP 491 ================================
prereg loss 1.2367756 reg_l1 29.260023 reg_l2 12.874077
loss 7.0887804
STEP 492 ================================
prereg loss 1.2357717 reg_l1 29.253061 reg_l2 12.871826
loss 7.086384
STEP 493 ================================
prereg loss 1.2351907 reg_l1 29.247162 reg_l2 12.869715
loss 7.0846233
STEP 494 ================================
prereg loss 1.2342206 reg_l1 29.24025 reg_l2 12.8676405
loss 7.0822706
STEP 495 ================================
prereg loss 1.2333336 reg_l1 29.233633 reg_l2 12.865542
loss 7.0800605
STEP 496 ================================
prereg loss 1.232869 reg_l1 29.226746 reg_l2 12.8632965
loss 7.0782185
STEP 497 ================================
prereg loss 1.2319148 reg_l1 29.219963 reg_l2 12.861005
loss 7.0759077
STEP 498 ================================
prereg loss 1.2314559 reg_l1 29.212742 reg_l2 12.858558
loss 7.074004
STEP 499 ================================
prereg loss 1.2308791 reg_l1 29.204752 reg_l2 12.856117
loss 7.07183
STEP 500 ================================
prereg loss 1.2299635 reg_l1 29.197168 reg_l2 12.853743
loss 7.069397
2022-06-27T01:39:16.307

julia> serialize("sparse16-after-800-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse16-after-800-steps-opt.ser", opt)
```

Let's continue, 700 more steps:

```
julia> steps!(700)
2022-06-27T01:50:29.808
STEP 1 ================================
prereg loss 1.2300963 reg_l1 29.190987 reg_l2 12.851471
loss 7.0682936
STEP 2 ================================
prereg loss 1.2290107 reg_l1 29.18459 reg_l2 12.849308
loss 7.0659285
STEP 3 ================================
prereg loss 1.2278539 reg_l1 29.177807 reg_l2 12.847148
loss 7.063415
STEP 4 ================================
prereg loss 1.2273307 reg_l1 29.170673 reg_l2 12.844891
loss 7.0614653
STEP 5 ================================
prereg loss 1.2264501 reg_l1 29.16412 reg_l2 12.842508
loss 7.059274
STEP 6 ================================
prereg loss 1.2260476 reg_l1 29.156971 reg_l2 12.840037
loss 7.0574417
STEP 7 ================================
prereg loss 1.2256335 reg_l1 29.149544 reg_l2 12.837541
loss 7.0555425
STEP 8 ================================
prereg loss 1.2245371 reg_l1 29.141752 reg_l2 12.835086
loss 7.052888
STEP 9 ================================
prereg loss 1.2245045 reg_l1 29.13427 reg_l2 12.832489
loss 7.0513587
STEP 10 ================================
prereg loss 1.224031 reg_l1 29.127094 reg_l2 12.8300085
loss 7.04945
STEP 11 ================================
prereg loss 1.2230786 reg_l1 29.120644 reg_l2 12.827694
loss 7.0472074
STEP 12 ================================
prereg loss 1.2223932 reg_l1 29.11376 reg_l2 12.8256
loss 7.045145
STEP 13 ================================
prereg loss 1.2216356 reg_l1 29.106876 reg_l2 12.823547
loss 7.0430107
STEP 14 ================================
prereg loss 1.2207061 reg_l1 29.100498 reg_l2 12.821502
loss 7.040806
STEP 15 ================================
prereg loss 1.2200528 reg_l1 29.094421 reg_l2 12.819562
loss 7.038937
STEP 16 ================================
prereg loss 1.2191303 reg_l1 29.087805 reg_l2 12.817618
loss 7.0366917
STEP 17 ================================
prereg loss 1.2185977 reg_l1 29.080482 reg_l2 12.8156185
loss 7.0346947
STEP 18 ================================
prereg loss 1.2180386 reg_l1 29.072742 reg_l2 12.813508
loss 7.032587
STEP 19 ================================
prereg loss 1.2170895 reg_l1 29.065096 reg_l2 12.811323
loss 7.030109
STEP 20 ================================
prereg loss 1.2167972 reg_l1 29.058046 reg_l2 12.808901
loss 7.0284066
STEP 21 ================================
prereg loss 1.2162428 reg_l1 29.050907 reg_l2 12.806394
loss 7.0264244
STEP 22 ================================
prereg loss 1.2154586 reg_l1 29.042393 reg_l2 12.803898
loss 7.023937
STEP 23 ================================
prereg loss 1.2150544 reg_l1 29.034502 reg_l2 12.801334
loss 7.021955
STEP 24 ================================
prereg loss 1.2145853 reg_l1 29.027222 reg_l2 12.798815
loss 7.0200295
STEP 25 ================================
prereg loss 1.2139504 reg_l1 29.02077 reg_l2 12.796649
loss 7.0181046
STEP 26 ================================
prereg loss 1.2129928 reg_l1 29.013578 reg_l2 12.794772
loss 7.0157084
STEP 27 ================================
prereg loss 1.2124398 reg_l1 29.006964 reg_l2 12.793083
loss 7.013833
STEP 28 ================================
prereg loss 1.2114612 reg_l1 29.000982 reg_l2 12.791364
loss 7.0116577
STEP 29 ================================
prereg loss 1.2105509 reg_l1 28.994608 reg_l2 12.789481
loss 7.0094724
STEP 30 ================================
prereg loss 1.2099142 reg_l1 28.987194 reg_l2 12.78736
loss 7.0073533
STEP 31 ================================
prereg loss 1.2093713 reg_l1 28.97913 reg_l2 12.78487
loss 7.0051975
STEP 32 ================================
prereg loss 1.2089295 reg_l1 28.97053 reg_l2 12.782139
loss 7.0030355
STEP 33 ================================
prereg loss 1.2086077 reg_l1 28.962458 reg_l2 12.779322
loss 7.001099
STEP 34 ================================
prereg loss 1.2081373 reg_l1 28.954662 reg_l2 12.776804
loss 6.99907
STEP 35 ================================
prereg loss 1.2078255 reg_l1 28.947699 reg_l2 12.774684
loss 6.9973655
STEP 36 ================================
prereg loss 1.206692 reg_l1 28.940546 reg_l2 12.772706
loss 6.9948015
STEP 37 ================================
prereg loss 1.2058764 reg_l1 28.934607 reg_l2 12.770983
loss 6.992798
STEP 38 ================================
prereg loss 1.205024 reg_l1 28.92811 reg_l2 12.7693405
loss 6.9906464
STEP 39 ================================
prereg loss 1.2043192 reg_l1 28.92077 reg_l2 12.767476
loss 6.988474
STEP 40 ================================
prereg loss 1.203664 reg_l1 28.913752 reg_l2 12.765381
loss 6.9864144
STEP 41 ================================
prereg loss 1.2031845 reg_l1 28.906282 reg_l2 12.763011
loss 6.9844413
STEP 42 ================================
prereg loss 1.2027084 reg_l1 28.898064 reg_l2 12.760606
loss 6.9823213
STEP 43 ================================
prereg loss 1.2021911 reg_l1 28.88923 reg_l2 12.75813
loss 6.9800377
STEP 44 ================================
prereg loss 1.2018648 reg_l1 28.88119 reg_l2 12.755664
loss 6.9781027
STEP 45 ================================
prereg loss 1.2013266 reg_l1 28.87444 reg_l2 12.753521
loss 6.9762144
STEP 46 ================================
prereg loss 1.2004497 reg_l1 28.86773 reg_l2 12.751723
loss 6.973995
STEP 47 ================================
prereg loss 1.2000486 reg_l1 28.861027 reg_l2 12.750203
loss 6.972254
STEP 48 ================================
prereg loss 1.1988184 reg_l1 28.854265 reg_l2 12.748733
loss 6.9696712
STEP 49 ================================
prereg loss 1.1979676 reg_l1 28.848482 reg_l2 12.747114
loss 6.9676642
STEP 50 ================================
prereg loss 1.1973599 reg_l1 28.841795 reg_l2 12.745247
loss 6.965719
STEP 51 ================================
prereg loss 1.1967201 reg_l1 28.833652 reg_l2 12.743097
loss 6.963451
STEP 52 ================================
prereg loss 1.1963931 reg_l1 28.825567 reg_l2 12.740742
loss 6.9615064
STEP 53 ================================
prereg loss 1.19567 reg_l1 28.817894 reg_l2 12.7383795
loss 6.959249
STEP 54 ================================
prereg loss 1.1952646 reg_l1 28.809847 reg_l2 12.735953
loss 6.9572344
STEP 55 ================================
prereg loss 1.1948379 reg_l1 28.802288 reg_l2 12.733694
loss 6.9552956
STEP 56 ================================
prereg loss 1.1941086 reg_l1 28.79464 reg_l2 12.731648
loss 6.9530363
STEP 57 ================================
prereg loss 1.1936529 reg_l1 28.787146 reg_l2 12.729855
loss 6.951082
STEP 58 ================================
prereg loss 1.1927667 reg_l1 28.780418 reg_l2 12.728125
loss 6.9488506
STEP 59 ================================
prereg loss 1.1920956 reg_l1 28.773521 reg_l2 12.726349
loss 6.9468
STEP 60 ================================
prereg loss 1.1914188 reg_l1 28.766418 reg_l2 12.72462
loss 6.9447026
STEP 61 ================================
prereg loss 1.1904422 reg_l1 28.759396 reg_l2 12.722849
loss 6.9423213
STEP 62 ================================
prereg loss 1.1902877 reg_l1 28.752123 reg_l2 12.721037
loss 6.9407125
STEP 63 ================================
prereg loss 1.1895301 reg_l1 28.745249 reg_l2 12.71924
loss 6.9385796
STEP 64 ================================
prereg loss 1.1883234 reg_l1 28.738255 reg_l2 12.717425
loss 6.9359746
STEP 65 ================================
prereg loss 1.1881791 reg_l1 28.731028 reg_l2 12.715295
loss 6.934385
STEP 66 ================================
prereg loss 1.1877117 reg_l1 28.723045 reg_l2 12.712996
loss 6.932321
STEP 67 ================================
prereg loss 1.1869658 reg_l1 28.71427 reg_l2 12.710648
loss 6.92982
STEP 68 ================================
prereg loss 1.1872015 reg_l1 28.705439 reg_l2 12.708472
loss 6.9282894
STEP 69 ================================
prereg loss 1.1864507 reg_l1 28.697857 reg_l2 12.706501
loss 6.9260225
STEP 70 ================================
prereg loss 1.1854687 reg_l1 28.690994 reg_l2 12.704688
loss 6.9236674
STEP 71 ================================
prereg loss 1.1849732 reg_l1 28.683525 reg_l2 12.703051
loss 6.9216785
STEP 72 ================================
prereg loss 1.1840961 reg_l1 28.676392 reg_l2 12.701442
loss 6.9193745
STEP 73 ================================
prereg loss 1.1830968 reg_l1 28.669767 reg_l2 12.699811
loss 6.9170504
STEP 74 ================================
prereg loss 1.1827215 reg_l1 28.662468 reg_l2 12.698014
loss 6.9152155
STEP 75 ================================
prereg loss 1.1818436 reg_l1 28.654985 reg_l2 12.6961775
loss 6.912841
STEP 76 ================================
prereg loss 1.1814075 reg_l1 28.647295 reg_l2 12.69421
loss 6.9108667
STEP 77 ================================
prereg loss 1.1809036 reg_l1 28.639227 reg_l2 12.6922655
loss 6.908749
STEP 78 ================================
prereg loss 1.1801419 reg_l1 28.631008 reg_l2 12.690383
loss 6.9063435
STEP 79 ================================
prereg loss 1.1799687 reg_l1 28.623116 reg_l2 12.68862
loss 6.904592
STEP 80 ================================
prereg loss 1.1790811 reg_l1 28.616621 reg_l2 12.686932
loss 6.9024053
STEP 81 ================================
prereg loss 1.1781992 reg_l1 28.609747 reg_l2 12.685221
loss 6.900149
STEP 82 ================================
prereg loss 1.177662 reg_l1 28.601694 reg_l2 12.683461
loss 6.8980007
STEP 83 ================================
prereg loss 1.1767993 reg_l1 28.593895 reg_l2 12.681601
loss 6.8955784
STEP 84 ================================
prereg loss 1.1765643 reg_l1 28.58627 reg_l2 12.6796875
loss 6.8938184
STEP 85 ================================
prereg loss 1.176009 reg_l1 28.578707 reg_l2 12.677776
loss 6.891751
STEP 86 ================================
prereg loss 1.1749175 reg_l1 28.570831 reg_l2 12.67588
loss 6.889084
STEP 87 ================================
prereg loss 1.1748949 reg_l1 28.56276 reg_l2 12.673755
loss 6.887447
STEP 88 ================================
prereg loss 1.1744931 reg_l1 28.554943 reg_l2 12.671621
loss 6.885482
STEP 89 ================================
prereg loss 1.1736883 reg_l1 28.546402 reg_l2 12.66958
loss 6.882969
STEP 90 ================================
prereg loss 1.1733813 reg_l1 28.5385 reg_l2 12.667774
loss 6.8810816
STEP 91 ================================
prereg loss 1.1725737 reg_l1 28.530935 reg_l2 12.666101
loss 6.878761
STEP 92 ================================
prereg loss 1.1716493 reg_l1 28.523857 reg_l2 12.664507
loss 6.876421
STEP 93 ================================
prereg loss 1.1711216 reg_l1 28.517708 reg_l2 12.663008
loss 6.8746634
STEP 94 ================================
prereg loss 1.1702536 reg_l1 28.50994 reg_l2 12.661483
loss 6.8722415
STEP 95 ================================
prereg loss 1.1694796 reg_l1 28.502254 reg_l2 12.659886
loss 6.8699303
STEP 96 ================================
prereg loss 1.1690967 reg_l1 28.49454 reg_l2 12.658135
loss 6.868005
STEP 97 ================================
prereg loss 1.1682107 reg_l1 28.48666 reg_l2 12.656348
loss 6.8655424
STEP 98 ================================
prereg loss 1.1679109 reg_l1 28.47817 reg_l2 12.654399
loss 6.8635454
STEP 99 ================================
prereg loss 1.1674876 reg_l1 28.469109 reg_l2 12.652445
loss 6.8613095
STEP 100 ================================
prereg loss 1.1667539 reg_l1 28.461472 reg_l2 12.650555
loss 6.8590484
STEP 101 ================================
prereg loss 1.1666505 reg_l1 28.45395 reg_l2 12.648828
loss 6.857441
STEP 102 ================================
prereg loss 1.1657455 reg_l1 28.445658 reg_l2 12.647224
loss 6.8548775
STEP 103 ================================
prereg loss 1.1647557 reg_l1 28.438066 reg_l2 12.645656
loss 6.8523693
STEP 104 ================================
prereg loss 1.1642582 reg_l1 28.43118 reg_l2 12.644061
loss 6.8504944
STEP 105 ================================
prereg loss 1.1634059 reg_l1 28.423656 reg_l2 12.642366
loss 6.8481374
STEP 106 ================================
prereg loss 1.1631032 reg_l1 28.415741 reg_l2 12.640596
loss 6.8462515
STEP 107 ================================
prereg loss 1.1625838 reg_l1 28.407396 reg_l2 12.638836
loss 6.8440633
STEP 108 ================================
prereg loss 1.1615212 reg_l1 28.399052 reg_l2 12.637101
loss 6.8413315
STEP 109 ================================
prereg loss 1.1614151 reg_l1 28.391754 reg_l2 12.635157
loss 6.839766
STEP 110 ================================
prereg loss 1.1609461 reg_l1 28.383749 reg_l2 12.633198
loss 6.837696
STEP 111 ================================
prereg loss 1.160142 reg_l1 28.37503 reg_l2 12.6313
loss 6.8351483
STEP 112 ================================
prereg loss 1.1599424 reg_l1 28.366524 reg_l2 12.629612
loss 6.833247
STEP 113 ================================
prereg loss 1.1591378 reg_l1 28.358932 reg_l2 12.628078
loss 6.8309245
STEP 114 ================================
prereg loss 1.1582396 reg_l1 28.351885 reg_l2 12.62662
loss 6.828617
STEP 115 ================================
prereg loss 1.1577361 reg_l1 28.344776 reg_l2 12.625255
loss 6.8266916
STEP 116 ================================
prereg loss 1.1568737 reg_l1 28.33659 reg_l2 12.623844
loss 6.8241916
STEP 117 ================================
prereg loss 1.1561649 reg_l1 28.32845 reg_l2 12.622366
loss 6.8218546
STEP 118 ================================
prereg loss 1.1557778 reg_l1 28.320766 reg_l2 12.620786
loss 6.8199315
STEP 119 ================================
prereg loss 1.1548471 reg_l1 28.312483 reg_l2 12.619192
loss 6.8173437
STEP 120 ================================
prereg loss 1.154525 reg_l1 28.303633 reg_l2 12.617435
loss 6.8152514
STEP 121 ================================
prereg loss 1.1540291 reg_l1 28.296473 reg_l2 12.615666
loss 6.813324
STEP 122 ================================
prereg loss 1.1532419 reg_l1 28.2884 reg_l2 12.613949
loss 6.8109217
STEP 123 ================================
prereg loss 1.1530616 reg_l1 28.27974 reg_l2 12.61238
loss 6.8090096
STEP 124 ================================
prereg loss 1.1521251 reg_l1 28.271946 reg_l2 12.610937
loss 6.8065147
STEP 125 ================================
prereg loss 1.151262 reg_l1 28.264984 reg_l2 12.609496
loss 6.8042593
STEP 126 ================================
prereg loss 1.1507467 reg_l1 28.25751 reg_l2 12.608031
loss 6.802249
STEP 127 ================================
prereg loss 1.1499242 reg_l1 28.249092 reg_l2 12.60645
loss 6.7997427
STEP 128 ================================
prereg loss 1.1495316 reg_l1 28.240211 reg_l2 12.604795
loss 6.797574
STEP 129 ================================
prereg loss 1.1490912 reg_l1 28.231993 reg_l2 12.603107
loss 6.79549
STEP 130 ================================
prereg loss 1.148078 reg_l1 28.224274 reg_l2 12.601463
loss 6.792933
STEP 131 ================================
prereg loss 1.1479589 reg_l1 28.215422 reg_l2 12.599628
loss 6.7910433
STEP 132 ================================
prereg loss 1.1474622 reg_l1 28.206856 reg_l2 12.597803
loss 6.7888336
STEP 133 ================================
prereg loss 1.1466378 reg_l1 28.198694 reg_l2 12.596065
loss 6.786377
STEP 134 ================================
prereg loss 1.146303 reg_l1 28.190672 reg_l2 12.594531
loss 6.7844377
STEP 135 ================================
prereg loss 1.1455036 reg_l1 28.183073 reg_l2 12.593113
loss 6.7821183
STEP 136 ================================
prereg loss 1.1446671 reg_l1 28.1774 reg_l2 12.591749
loss 6.7801476
STEP 137 ================================
prereg loss 1.1441272 reg_l1 28.170694 reg_l2 12.590437
loss 6.7782664
STEP 138 ================================
prereg loss 1.1432601 reg_l1 28.163467 reg_l2 12.589074
loss 6.775954
STEP 139 ================================
prereg loss 1.1426905 reg_l1 28.155888 reg_l2 12.587636
loss 6.773868
STEP 140 ================================
prereg loss 1.1422577 reg_l1 28.148586 reg_l2 12.58612
loss 6.771975
STEP 141 ================================
prereg loss 1.1413248 reg_l1 28.141382 reg_l2 12.584602
loss 6.769601
STEP 142 ================================
prereg loss 1.1410725 reg_l1 28.133638 reg_l2 12.582893
loss 6.7678003
STEP 143 ================================
prereg loss 1.1405869 reg_l1 28.125183 reg_l2 12.581156
loss 6.7656236
STEP 144 ================================
prereg loss 1.1398116 reg_l1 28.116276 reg_l2 12.579477
loss 6.763067
STEP 145 ================================
prereg loss 1.139776 reg_l1 28.108715 reg_l2 12.577984
loss 6.7615194
STEP 146 ================================
prereg loss 1.138802 reg_l1 28.101973 reg_l2 12.576675
loss 6.7591968
STEP 147 ================================
prereg loss 1.1378976 reg_l1 28.09439 reg_l2 12.575418
loss 6.756776
STEP 148 ================================
prereg loss 1.1374161 reg_l1 28.08626 reg_l2 12.574154
loss 6.754668
STEP 149 ================================
prereg loss 1.1365495 reg_l1 28.078655 reg_l2 12.572769
loss 6.7522807
STEP 150 ================================
prereg loss 1.1360286 reg_l1 28.071175 reg_l2 12.571289
loss 6.7502637
STEP 151 ================================
prereg loss 1.1356686 reg_l1 28.063614 reg_l2 12.569763
loss 6.7483916
STEP 152 ================================
prereg loss 1.1346486 reg_l1 28.056784 reg_l2 12.568303
loss 6.746005
STEP 153 ================================
prereg loss 1.1343329 reg_l1 28.049421 reg_l2 12.566695
loss 6.744217
STEP 154 ================================
prereg loss 1.1337899 reg_l1 28.041592 reg_l2 12.565067
loss 6.7421083
STEP 155 ================================
prereg loss 1.1329751 reg_l1 28.033886 reg_l2 12.563476
loss 6.7397523
STEP 156 ================================
prereg loss 1.1326368 reg_l1 28.025885 reg_l2 12.562046
loss 6.737814
STEP 157 ================================
prereg loss 1.1318505 reg_l1 28.018522 reg_l2 12.56071
loss 6.7355547
STEP 158 ================================
prereg loss 1.1310503 reg_l1 28.01121 reg_l2 12.559408
loss 6.7332926
STEP 159 ================================
prereg loss 1.1305033 reg_l1 28.00275 reg_l2 12.558164
loss 6.7310534
STEP 160 ================================
prereg loss 1.1296444 reg_l1 27.99571 reg_l2 12.556869
loss 6.7287865
STEP 161 ================================
prereg loss 1.1290656 reg_l1 27.98852 reg_l2 12.555518
loss 6.7267694
STEP 162 ================================
prereg loss 1.1285969 reg_l1 27.980515 reg_l2 12.554096
loss 6.7247
STEP 163 ================================
prereg loss 1.1276686 reg_l1 27.972149 reg_l2 12.552666
loss 6.7220984
STEP 164 ================================
prereg loss 1.1274735 reg_l1 27.964628 reg_l2 12.551024
loss 6.720399
STEP 165 ================================
prereg loss 1.1269737 reg_l1 27.95701 reg_l2 12.5493765
loss 6.7183757
STEP 166 ================================
prereg loss 1.1261547 reg_l1 27.94831 reg_l2 12.547802
loss 6.7158165
STEP 167 ================================
prereg loss 1.1260828 reg_l1 27.939873 reg_l2 12.546421
loss 6.7140574
STEP 168 ================================
prereg loss 1.1250811 reg_l1 27.932594 reg_l2 12.545223
loss 6.7116
STEP 169 ================================
prereg loss 1.1241826 reg_l1 27.925932 reg_l2 12.5440645
loss 6.709369
STEP 170 ================================
prereg loss 1.1236967 reg_l1 27.92092 reg_l2 12.542879
loss 6.707881
STEP 171 ================================
prereg loss 1.1225494 reg_l1 27.91568 reg_l2 12.54155
loss 6.7056856
STEP 172 ================================
prereg loss 1.1213161 reg_l1 27.908976 reg_l2 12.540001
loss 6.703111
STEP 173 ================================
prereg loss 1.1201874 reg_l1 27.901567 reg_l2 12.53823
loss 6.700501
STEP 174 ================================
prereg loss 1.1186624 reg_l1 27.89336 reg_l2 12.536317
loss 6.6973343
STEP 175 ================================
prereg loss 1.1171386 reg_l1 27.88534 reg_l2 12.534125
loss 6.694206
STEP 176 ================================
prereg loss 1.1155497 reg_l1 27.876802 reg_l2 12.5314455
loss 6.6909103
STEP 177 ================================
prereg loss 1.1138285 reg_l1 27.866755 reg_l2 12.528275
loss 6.6871796
STEP 178 ================================
prereg loss 1.1122093 reg_l1 27.856396 reg_l2 12.524775
loss 6.6834884
STEP 179 ================================
prereg loss 1.1107986 reg_l1 27.846172 reg_l2 12.521152
loss 6.6800327
STEP 180 ================================
prereg loss 1.1093485 reg_l1 27.836927 reg_l2 12.517503
loss 6.676734
STEP 181 ================================
prereg loss 1.1080513 reg_l1 27.827625 reg_l2 12.5138
loss 6.6735764
STEP 182 ================================
prereg loss 1.1071552 reg_l1 27.817038 reg_l2 12.509943
loss 6.6705627
STEP 183 ================================
prereg loss 1.1062478 reg_l1 27.805927 reg_l2 12.506207
loss 6.6674333
STEP 184 ================================
prereg loss 1.1055665 reg_l1 27.79613 reg_l2 12.502883
loss 6.6647925
STEP 185 ================================
prereg loss 1.1046305 reg_l1 27.787302 reg_l2 12.499907
loss 6.662091
STEP 186 ================================
prereg loss 1.1034586 reg_l1 27.779604 reg_l2 12.497598
loss 6.65938
STEP 187 ================================
prereg loss 1.1022611 reg_l1 27.771889 reg_l2 12.495796
loss 6.656639
STEP 188 ================================
prereg loss 1.100888 reg_l1 27.764597 reg_l2 12.49409
loss 6.6538076
STEP 189 ================================
prereg loss 1.1001045 reg_l1 27.757418 reg_l2 12.492216
loss 6.651588
STEP 190 ================================
prereg loss 1.0990748 reg_l1 27.74921 reg_l2 12.49019
loss 6.648917
STEP 191 ================================
prereg loss 1.0986425 reg_l1 27.7401 reg_l2 12.4880495
loss 6.6466627
STEP 192 ================================
prereg loss 1.0976686 reg_l1 27.73271 reg_l2 12.486004
loss 6.644211
STEP 193 ================================
prereg loss 1.0960803 reg_l1 27.725372 reg_l2 12.484047
loss 6.641155
STEP 194 ================================
prereg loss 1.0951042 reg_l1 27.716858 reg_l2 12.481861
loss 6.638476
STEP 195 ================================
prereg loss 1.0936685 reg_l1 27.70769 reg_l2 12.479395
loss 6.635206
STEP 196 ================================
prereg loss 1.0923012 reg_l1 27.698013 reg_l2 12.476702
loss 6.631904
STEP 197 ================================
prereg loss 1.0916537 reg_l1 27.688261 reg_l2 12.473976
loss 6.6293063
STEP 198 ================================
prereg loss 1.0902995 reg_l1 27.679722 reg_l2 12.471466
loss 6.626244
STEP 199 ================================
prereg loss 1.089608 reg_l1 27.670639 reg_l2 12.469028
loss 6.6237354
STEP 200 ================================
prereg loss 1.0887401 reg_l1 27.662464 reg_l2 12.466827
loss 6.621233
STEP 201 ================================
prereg loss 1.0875186 reg_l1 27.654827 reg_l2 12.464829
loss 6.618484
STEP 202 ================================
prereg loss 1.086215 reg_l1 27.646334 reg_l2 12.463031
loss 6.615482
STEP 203 ================================
prereg loss 1.0851029 reg_l1 27.638216 reg_l2 12.4612055
loss 6.6127462
STEP 204 ================================
prereg loss 1.0836482 reg_l1 27.630093 reg_l2 12.459335
loss 6.609667
STEP 205 ================================
prereg loss 1.0823765 reg_l1 27.62302 reg_l2 12.457375
loss 6.606981
STEP 206 ================================
prereg loss 1.0814303 reg_l1 27.61441 reg_l2 12.455203
loss 6.6043124
STEP 207 ================================
prereg loss 1.0805817 reg_l1 27.605291 reg_l2 12.452879
loss 6.60164
STEP 208 ================================
prereg loss 1.0793003 reg_l1 27.596573 reg_l2 12.45048
loss 6.598615
STEP 209 ================================
prereg loss 1.0782973 reg_l1 27.587713 reg_l2 12.447859
loss 6.59584
STEP 210 ================================
prereg loss 1.0772048 reg_l1 27.578259 reg_l2 12.44503
loss 6.5928564
STEP 211 ================================
prereg loss 1.0764561 reg_l1 27.567642 reg_l2 12.442161
loss 6.5899844
STEP 212 ================================
prereg loss 1.0755591 reg_l1 27.557705 reg_l2 12.439408
loss 6.5871
STEP 213 ================================
prereg loss 1.0743994 reg_l1 27.549725 reg_l2 12.436807
loss 6.5843444
STEP 214 ================================
prereg loss 1.0735716 reg_l1 27.541145 reg_l2 12.43442
loss 6.581801
STEP 215 ================================
prereg loss 1.072535 reg_l1 27.532887 reg_l2 12.432195
loss 6.5791125
STEP 216 ================================
prereg loss 1.0717636 reg_l1 27.524603 reg_l2 12.430126
loss 6.576684
STEP 217 ================================
prereg loss 1.0702826 reg_l1 27.516333 reg_l2 12.428137
loss 6.5735493
STEP 218 ================================
prereg loss 1.0688694 reg_l1 27.508533 reg_l2 12.426039
loss 6.5705757
STEP 219 ================================
prereg loss 1.0678393 reg_l1 27.499104 reg_l2 12.423525
loss 6.56766
STEP 220 ================================
prereg loss 1.0667537 reg_l1 27.488913 reg_l2 12.420526
loss 6.5645366
STEP 221 ================================
prereg loss 1.0660943 reg_l1 27.479033 reg_l2 12.417299
loss 6.561901
STEP 222 ================================
prereg loss 1.0653714 reg_l1 27.46816 reg_l2 12.414171
loss 6.559004
STEP 223 ================================
prereg loss 1.0650052 reg_l1 27.458103 reg_l2 12.4112215
loss 6.556626
STEP 224 ================================
prereg loss 1.0643237 reg_l1 27.449604 reg_l2 12.408831
loss 6.554245
STEP 225 ================================
prereg loss 1.0629147 reg_l1 27.44202 reg_l2 12.406989
loss 6.551319
STEP 226 ================================
prereg loss 1.0610429 reg_l1 27.435144 reg_l2 12.4055195
loss 6.548072
STEP 227 ================================
prereg loss 1.0602306 reg_l1 27.427872 reg_l2 12.40412
loss 6.545805
STEP 228 ================================
prereg loss 1.0588647 reg_l1 27.41984 reg_l2 12.402734
loss 6.542833
STEP 229 ================================
prereg loss 1.057596 reg_l1 27.412521 reg_l2 12.401203
loss 6.5401
STEP 230 ================================
prereg loss 1.0566486 reg_l1 27.404787 reg_l2 12.399233
loss 6.5376062
STEP 231 ================================
prereg loss 1.0555636 reg_l1 27.394764 reg_l2 12.396655
loss 6.5345163
STEP 232 ================================
prereg loss 1.054856 reg_l1 27.383768 reg_l2 12.39367
loss 6.5316095
STEP 233 ================================
prereg loss 1.0540724 reg_l1 27.37345 reg_l2 12.3906765
loss 6.5287623
STEP 234 ================================
prereg loss 1.0529563 reg_l1 27.364014 reg_l2 12.387787
loss 6.5257587
STEP 235 ================================
prereg loss 1.0523363 reg_l1 27.354317 reg_l2 12.384865
loss 6.5231996
STEP 236 ================================
prereg loss 1.0513592 reg_l1 27.344915 reg_l2 12.382081
loss 6.5203424
STEP 237 ================================
prereg loss 1.0501053 reg_l1 27.336178 reg_l2 12.379479
loss 6.5173407
STEP 238 ================================
prereg loss 1.0496237 reg_l1 27.327238 reg_l2 12.377164
loss 6.515072
STEP 239 ================================
prereg loss 1.0482582 reg_l1 27.319025 reg_l2 12.375115
loss 6.5120635
STEP 240 ================================
prereg loss 1.0468435 reg_l1 27.312094 reg_l2 12.373151
loss 6.5092626
STEP 241 ================================
prereg loss 1.0458795 reg_l1 27.304417 reg_l2 12.370972
loss 6.506763
STEP 242 ================================
prereg loss 1.044491 reg_l1 27.294561 reg_l2 12.36839
loss 6.5034037
STEP 243 ================================
prereg loss 1.0438617 reg_l1 27.28409 reg_l2 12.365498
loss 6.50068
STEP 244 ================================
prereg loss 1.0432045 reg_l1 27.273973 reg_l2 12.362678
loss 6.497999
STEP 245 ================================
prereg loss 1.0421023 reg_l1 27.264738 reg_l2 12.360049
loss 6.49505
STEP 246 ================================
prereg loss 1.0414935 reg_l1 27.256056 reg_l2 12.357647
loss 6.492705
STEP 247 ================================
prereg loss 1.0403262 reg_l1 27.246925 reg_l2 12.355492
loss 6.4897113
STEP 248 ================================
prereg loss 1.0392189 reg_l1 27.238426 reg_l2 12.353519
loss 6.486904
STEP 249 ================================
prereg loss 1.0379592 reg_l1 27.230598 reg_l2 12.35158
loss 6.484079
STEP 250 ================================
prereg loss 1.0367111 reg_l1 27.22273 reg_l2 12.349585
loss 6.4812574
STEP 251 ================================
prereg loss 1.0358775 reg_l1 27.214119 reg_l2 12.347294
loss 6.4787016
STEP 252 ================================
prereg loss 1.0347444 reg_l1 27.204412 reg_l2 12.344802
loss 6.475627
STEP 253 ================================
prereg loss 1.0338926 reg_l1 27.194857 reg_l2 12.34208
loss 6.472864
STEP 254 ================================
prereg loss 1.0330364 reg_l1 27.1854 reg_l2 12.33932
loss 6.470116
STEP 255 ================================
prereg loss 1.0323412 reg_l1 27.176283 reg_l2 12.336556
loss 6.467598
STEP 256 ================================
prereg loss 1.0313855 reg_l1 27.167076 reg_l2 12.334028
loss 6.464801
STEP 257 ================================
prereg loss 1.0303068 reg_l1 27.157991 reg_l2 12.331743
loss 6.461905
STEP 258 ================================
prereg loss 1.0292189 reg_l1 27.149296 reg_l2 12.329499
loss 6.459078
STEP 259 ================================
prereg loss 1.0283073 reg_l1 27.139944 reg_l2 12.327239
loss 6.4562964
STEP 260 ================================
prereg loss 1.0271977 reg_l1 27.131317 reg_l2 12.325109
loss 6.453461
STEP 261 ================================
prereg loss 1.0263649 reg_l1 27.123552 reg_l2 12.323084
loss 6.4510756
STEP 262 ================================
prereg loss 1.0250876 reg_l1 27.115328 reg_l2 12.321094
loss 6.4481535
STEP 263 ================================
prereg loss 1.0237786 reg_l1 27.107233 reg_l2 12.319034
loss 6.4452252
STEP 264 ================================
prereg loss 1.0226666 reg_l1 27.098862 reg_l2 12.316761
loss 6.442439
STEP 265 ================================
prereg loss 1.0218619 reg_l1 27.089445 reg_l2 12.314256
loss 6.439751
STEP 266 ================================
prereg loss 1.0211966 reg_l1 27.079391 reg_l2 12.311704
loss 6.4370747
STEP 267 ================================
prereg loss 1.0201021 reg_l1 27.06929 reg_l2 12.309254
loss 6.43396
STEP 268 ================================
prereg loss 1.0196928 reg_l1 27.059927 reg_l2 12.306738
loss 6.4316783
STEP 269 ================================
prereg loss 1.018903 reg_l1 27.050764 reg_l2 12.304371
loss 6.429056
STEP 270 ================================
prereg loss 1.0175562 reg_l1 27.041014 reg_l2 12.302162
loss 6.425759
STEP 271 ================================
prereg loss 1.0166574 reg_l1 27.031416 reg_l2 12.30006
loss 6.4229407
STEP 272 ================================
prereg loss 1.0156997 reg_l1 27.023878 reg_l2 12.298071
loss 6.4204755
STEP 273 ================================
prereg loss 1.0143335 reg_l1 27.016573 reg_l2 12.29618
loss 6.4176483
STEP 274 ================================
prereg loss 1.0132979 reg_l1 27.008184 reg_l2 12.2941065
loss 6.414935
STEP 275 ================================
prereg loss 1.0122063 reg_l1 26.998804 reg_l2 12.291652
loss 6.4119673
STEP 276 ================================
prereg loss 1.0114169 reg_l1 26.988874 reg_l2 12.288866
loss 6.409192
STEP 277 ================================
prereg loss 1.0105366 reg_l1 26.979206 reg_l2 12.286064
loss 6.406378
STEP 278 ================================
prereg loss 1.0100079 reg_l1 26.968983 reg_l2 12.28334
loss 6.4038043
STEP 279 ================================
prereg loss 1.0092101 reg_l1 26.959023 reg_l2 12.281042
loss 6.401015
STEP 280 ================================
prereg loss 1.007851 reg_l1 26.950914 reg_l2 12.279174
loss 6.398034
STEP 281 ================================
prereg loss 1.0067418 reg_l1 26.942785 reg_l2 12.277559
loss 6.395299
STEP 282 ================================
prereg loss 1.0056676 reg_l1 26.934242 reg_l2 12.275944
loss 6.392516
STEP 283 ================================
prereg loss 1.0046248 reg_l1 26.926088 reg_l2 12.274234
loss 6.3898425
STEP 284 ================================
prereg loss 1.0036322 reg_l1 26.917154 reg_l2 12.272232
loss 6.387063
STEP 285 ================================
prereg loss 1.0023905 reg_l1 26.908573 reg_l2 12.269928
loss 6.384105
STEP 286 ================================
prereg loss 1.0014856 reg_l1 26.89984 reg_l2 12.267402
loss 6.3814535
STEP 287 ================================
prereg loss 1.000517 reg_l1 26.88984 reg_l2 12.264859
loss 6.3784847
STEP 288 ================================
prereg loss 0.99979866 reg_l1 26.879322 reg_l2 12.262312
loss 6.3756633
STEP 289 ================================
prereg loss 0.99902785 reg_l1 26.870317 reg_l2 12.259972
loss 6.373091
STEP 290 ================================
prereg loss 0.99803853 reg_l1 26.862282 reg_l2 12.257871
loss 6.370495
STEP 291 ================================
prereg loss 0.9973015 reg_l1 26.853485 reg_l2 12.256031
loss 6.3679986
STEP 292 ================================
prereg loss 0.99588186 reg_l1 26.844479 reg_l2 12.254307
loss 6.364778
STEP 293 ================================
prereg loss 0.99471414 reg_l1 26.836407 reg_l2 12.252464
loss 6.3619957
STEP 294 ================================
prereg loss 0.9938084 reg_l1 26.826807 reg_l2 12.250165
loss 6.35917
STEP 295 ================================
prereg loss 0.99276483 reg_l1 26.816235 reg_l2 12.247415
loss 6.356012
STEP 296 ================================
prereg loss 0.99210113 reg_l1 26.80583 reg_l2 12.244534
loss 6.353267
STEP 297 ================================
prereg loss 0.99116516 reg_l1 26.796028 reg_l2 12.241804
loss 6.350371
STEP 298 ================================
prereg loss 0.99082667 reg_l1 26.786402 reg_l2 12.239213
loss 6.348107
STEP 299 ================================
prereg loss 0.99015516 reg_l1 26.776865 reg_l2 12.237092
loss 6.345528
STEP 300 ================================
prereg loss 0.98906296 reg_l1 26.768833 reg_l2 12.235427
loss 6.3428297
STEP 301 ================================
prereg loss 0.9875125 reg_l1 26.761396 reg_l2 12.234158
loss 6.339792
STEP 302 ================================
prereg loss 0.9865725 reg_l1 26.75304 reg_l2 12.233078
loss 6.337181
STEP 303 ================================
prereg loss 0.9849568 reg_l1 26.745333 reg_l2 12.232018
loss 6.3340235
STEP 304 ================================
prereg loss 0.9837864 reg_l1 26.738155 reg_l2 12.23071
loss 6.3314176
STEP 305 ================================
prereg loss 0.98288023 reg_l1 26.729414 reg_l2 12.228847
loss 6.328763
STEP 306 ================================
prereg loss 0.98194456 reg_l1 26.718773 reg_l2 12.226426
loss 6.3256993
STEP 307 ================================
prereg loss 0.9813931 reg_l1 26.707996 reg_l2 12.223776
loss 6.3229923
STEP 308 ================================
prereg loss 0.98053265 reg_l1 26.697947 reg_l2 12.221209
loss 6.3201222
STEP 309 ================================
prereg loss 0.9796318 reg_l1 26.68836 reg_l2 12.218714
loss 6.317304
STEP 310 ================================
prereg loss 0.97891647 reg_l1 26.678888 reg_l2 12.216314
loss 6.3146944
STEP 311 ================================
prereg loss 0.97773075 reg_l1 26.669214 reg_l2 12.214041
loss 6.3115735
STEP 312 ================================
prereg loss 0.9767165 reg_l1 26.660183 reg_l2 12.21188
loss 6.308753
STEP 313 ================================
prereg loss 0.97568274 reg_l1 26.651588 reg_l2 12.209781
loss 6.3060007
STEP 314 ================================
prereg loss 0.97450644 reg_l1 26.642937 reg_l2 12.207727
loss 6.303094
STEP 315 ================================
prereg loss 0.9737648 reg_l1 26.633196 reg_l2 12.2055
loss 6.300404
STEP 316 ================================
prereg loss 0.97288865 reg_l1 26.62233 reg_l2 12.203155
loss 6.2973547
STEP 317 ================================
prereg loss 0.9726876 reg_l1 26.611603 reg_l2 12.200905
loss 6.295008
STEP 318 ================================
prereg loss 0.9717284 reg_l1 26.602024 reg_l2 12.198962
loss 6.2921333
STEP 319 ================================
prereg loss 0.9707527 reg_l1 26.593864 reg_l2 12.19724
loss 6.2895255
STEP 320 ================================
prereg loss 0.9698966 reg_l1 26.58604 reg_l2 12.195681
loss 6.287105
STEP 321 ================================
prereg loss 0.96850026 reg_l1 26.577236 reg_l2 12.194127
loss 6.2839475
STEP 322 ================================
prereg loss 0.96751094 reg_l1 26.567585 reg_l2 12.192539
loss 6.281028
STEP 323 ================================
prereg loss 0.96648794 reg_l1 26.55851 reg_l2 12.191014
loss 6.27819
STEP 324 ================================
prereg loss 0.96503854 reg_l1 26.55 reg_l2 12.189569
loss 6.2750387
STEP 325 ================================
prereg loss 0.96410733 reg_l1 26.542072 reg_l2 12.187906
loss 6.272522
STEP 326 ================================
prereg loss 0.96335405 reg_l1 26.531916 reg_l2 12.185772
loss 6.2697372
STEP 327 ================================
prereg loss 0.96257293 reg_l1 26.521044 reg_l2 12.183494
loss 6.266782
STEP 328 ================================
prereg loss 0.9619262 reg_l1 26.510887 reg_l2 12.181257
loss 6.264104
STEP 329 ================================
prereg loss 0.96088517 reg_l1 26.501543 reg_l2 12.1795
loss 6.2611938
STEP 330 ================================
prereg loss 0.9596769 reg_l1 26.49298 reg_l2 12.177893
loss 6.2582726
STEP 331 ================================
prereg loss 0.95860946 reg_l1 26.484013 reg_l2 12.176416
loss 6.255412
STEP 332 ================================
prereg loss 0.9575633 reg_l1 26.474966 reg_l2 12.174894
loss 6.252557
STEP 333 ================================
prereg loss 0.95675707 reg_l1 26.466488 reg_l2 12.173205
loss 6.250055
STEP 334 ================================
prereg loss 0.955634 reg_l1 26.457472 reg_l2 12.171461
loss 6.2471285
STEP 335 ================================
prereg loss 0.9545595 reg_l1 26.44819 reg_l2 12.169542
loss 6.2441974
STEP 336 ================================
prereg loss 0.9538006 reg_l1 26.439396 reg_l2 12.167269
loss 6.24168
STEP 337 ================================
prereg loss 0.9528671 reg_l1 26.431139 reg_l2 12.164972
loss 6.2390947
STEP 338 ================================
prereg loss 0.95224583 reg_l1 26.424513 reg_l2 12.162683
loss 6.2371483
STEP 339 ================================
prereg loss 0.9515239 reg_l1 26.417513 reg_l2 12.160814
loss 6.2350264
STEP 340 ================================
prereg loss 0.9503228 reg_l1 26.411537 reg_l2 12.159129
loss 6.2326303
STEP 341 ================================
prereg loss 0.9491044 reg_l1 26.406124 reg_l2 12.157651
loss 6.230329
STEP 342 ================================
prereg loss 0.94789916 reg_l1 26.399689 reg_l2 12.156073
loss 6.227837
STEP 343 ================================
prereg loss 0.94709307 reg_l1 26.391739 reg_l2 12.154222
loss 6.225441
STEP 344 ================================
prereg loss 0.9462244 reg_l1 26.384096 reg_l2 12.1522875
loss 6.2230434
STEP 345 ================================
prereg loss 0.9453373 reg_l1 26.376556 reg_l2 12.150278
loss 6.220649
STEP 346 ================================
prereg loss 0.9447204 reg_l1 26.367933 reg_l2 12.148113
loss 6.218307
STEP 347 ================================
prereg loss 0.943846 reg_l1 26.35853 reg_l2 12.146009
loss 6.2155523
STEP 348 ================================
prereg loss 0.943088 reg_l1 26.349833 reg_l2 12.144115
loss 6.2130547
STEP 349 ================================
prereg loss 0.9416935 reg_l1 26.342478 reg_l2 12.142402
loss 6.210189
STEP 350 ================================
prereg loss 0.9406593 reg_l1 26.334507 reg_l2 12.140677
loss 6.2075605
STEP 351 ================================
prereg loss 0.939613 reg_l1 26.326685 reg_l2 12.138931
loss 6.20495
STEP 352 ================================
prereg loss 0.9385968 reg_l1 26.31979 reg_l2 12.137121
loss 6.2025547
STEP 353 ================================
prereg loss 0.9379084 reg_l1 26.312553 reg_l2 12.135262
loss 6.2004194
STEP 354 ================================
prereg loss 0.93683815 reg_l1 26.304708 reg_l2 12.133485
loss 6.19778
STEP 355 ================================
prereg loss 0.93639624 reg_l1 26.297216 reg_l2 12.1316
loss 6.1958394
STEP 356 ================================
prereg loss 0.9356935 reg_l1 26.289522 reg_l2 12.129832
loss 6.193598
STEP 357 ================================
prereg loss 0.93452156 reg_l1 26.281935 reg_l2 12.128273
loss 6.190909
STEP 358 ================================
prereg loss 0.9339885 reg_l1 26.273968 reg_l2 12.126944
loss 6.188782
STEP 359 ================================
prereg loss 0.93231416 reg_l1 26.266838 reg_l2 12.125848
loss 6.1856823
STEP 360 ================================
prereg loss 0.931044 reg_l1 26.260927 reg_l2 12.124687
loss 6.1832294
STEP 361 ================================
prereg loss 0.9301359 reg_l1 26.253971 reg_l2 12.123016
loss 6.18093
STEP 362 ================================
prereg loss 0.9290461 reg_l1 26.245886 reg_l2 12.12081
loss 6.1782236
STEP 363 ================================
prereg loss 0.92855304 reg_l1 26.237223 reg_l2 12.118404
loss 6.1759977
STEP 364 ================================
prereg loss 0.9276605 reg_l1 26.228823 reg_l2 12.116127
loss 6.173425
STEP 365 ================================
prereg loss 0.9271057 reg_l1 26.220854 reg_l2 12.113922
loss 6.1712766
STEP 366 ================================
prereg loss 0.9262828 reg_l1 26.212921 reg_l2 12.111942
loss 6.168867
STEP 367 ================================
prereg loss 0.9251294 reg_l1 26.205418 reg_l2 12.110111
loss 6.166213
STEP 368 ================================
prereg loss 0.92412287 reg_l1 26.197088 reg_l2 12.108475
loss 6.1635404
STEP 369 ================================
prereg loss 0.9233514 reg_l1 26.189028 reg_l2 12.106928
loss 6.161157
STEP 370 ================================
prereg loss 0.9221882 reg_l1 26.183084 reg_l2 12.105507
loss 6.1588054
STEP 371 ================================
prereg loss 0.92110443 reg_l1 26.176489 reg_l2 12.104138
loss 6.156402
STEP 372 ================================
prereg loss 0.91997993 reg_l1 26.169155 reg_l2 12.1026745
loss 6.153811
STEP 373 ================================
prereg loss 0.9189684 reg_l1 26.162271 reg_l2 12.101052
loss 6.1514225
STEP 374 ================================
prereg loss 0.9179263 reg_l1 26.155035 reg_l2 12.099319
loss 6.1489334
STEP 375 ================================
prereg loss 0.91704714 reg_l1 26.147093 reg_l2 12.097395
loss 6.146466
STEP 376 ================================
prereg loss 0.91629106 reg_l1 26.138546 reg_l2 12.095376
loss 6.1440005
STEP 377 ================================
prereg loss 0.9155719 reg_l1 26.130434 reg_l2 12.093265
loss 6.141659
STEP 378 ================================
prereg loss 0.9145505 reg_l1 26.122143 reg_l2 12.091387
loss 6.138979
STEP 379 ================================
prereg loss 0.9135872 reg_l1 26.114357 reg_l2 12.089509
loss 6.1364584
STEP 380 ================================
prereg loss 0.912701 reg_l1 26.106531 reg_l2 12.087716
loss 6.1340075
STEP 381 ================================
prereg loss 0.9118357 reg_l1 26.09935 reg_l2 12.085862
loss 6.1317058
STEP 382 ================================
prereg loss 0.9109679 reg_l1 26.091082 reg_l2 12.083924
loss 6.1291842
STEP 383 ================================
prereg loss 0.91029084 reg_l1 26.082241 reg_l2 12.081858
loss 6.126739
STEP 384 ================================
prereg loss 0.9093539 reg_l1 26.073956 reg_l2 12.07992
loss 6.124145
STEP 385 ================================
prereg loss 0.9086247 reg_l1 26.066767 reg_l2 12.078254
loss 6.1219783
STEP 386 ================================
prereg loss 0.9073252 reg_l1 26.05974 reg_l2 12.076791
loss 6.119273
STEP 387 ================================
prereg loss 0.9063883 reg_l1 26.052933 reg_l2 12.075352
loss 6.116975
STEP 388 ================================
prereg loss 0.90552074 reg_l1 26.045563 reg_l2 12.073869
loss 6.1146336
STEP 389 ================================
prereg loss 0.9043788 reg_l1 26.037645 reg_l2 12.072325
loss 6.111908
STEP 390 ================================
prereg loss 0.90349966 reg_l1 26.029236 reg_l2 12.070662
loss 6.109347
STEP 391 ================================
prereg loss 0.902484 reg_l1 26.02156 reg_l2 12.068989
loss 6.1067963
STEP 392 ================================
prereg loss 0.90175235 reg_l1 26.013605 reg_l2 12.067235
loss 6.1044736
STEP 393 ================================
prereg loss 0.90079033 reg_l1 26.005367 reg_l2 12.065518
loss 6.101864
STEP 394 ================================
prereg loss 0.8999515 reg_l1 25.997147 reg_l2 12.063863
loss 6.099381
STEP 395 ================================
prereg loss 0.8991596 reg_l1 25.990643 reg_l2 12.062245
loss 6.097288
STEP 396 ================================
prereg loss 0.8980071 reg_l1 25.983591 reg_l2 12.060694
loss 6.094725
STEP 397 ================================
prereg loss 0.89694214 reg_l1 25.976063 reg_l2 12.05914
loss 6.092155
STEP 398 ================================
prereg loss 0.8959651 reg_l1 25.96835 reg_l2 12.057493
loss 6.089635
STEP 399 ================================
prereg loss 0.8950031 reg_l1 25.96039 reg_l2 12.055774
loss 6.087081
STEP 400 ================================
prereg loss 0.8939236 reg_l1 25.952942 reg_l2 12.054022
loss 6.084512
STEP 401 ================================
prereg loss 0.8931301 reg_l1 25.945024 reg_l2 12.0520935
loss 6.082135
STEP 402 ================================
prereg loss 0.89225906 reg_l1 25.93669 reg_l2 12.050152
loss 6.079597
STEP 403 ================================
prereg loss 0.8916656 reg_l1 25.928751 reg_l2 12.048392
loss 6.077416
STEP 404 ================================
prereg loss 0.8905692 reg_l1 25.921448 reg_l2 12.046808
loss 6.0748587
STEP 405 ================================
prereg loss 0.890006 reg_l1 25.914236 reg_l2 12.045242
loss 6.0728536
STEP 406 ================================
prereg loss 0.88917834 reg_l1 25.906645 reg_l2 12.043801
loss 6.0705075
STEP 407 ================================
prereg loss 0.8878648 reg_l1 25.899147 reg_l2 12.042379
loss 6.0676947
STEP 408 ================================
prereg loss 0.8869065 reg_l1 25.89135 reg_l2 12.040933
loss 6.0651765
STEP 409 ================================
prereg loss 0.88588995 reg_l1 25.884438 reg_l2 12.039528
loss 6.0627775
STEP 410 ================================
prereg loss 0.884685 reg_l1 25.876944 reg_l2 12.038116
loss 6.060074
STEP 411 ================================
prereg loss 0.88387805 reg_l1 25.869276 reg_l2 12.036407
loss 6.0577335
STEP 412 ================================
prereg loss 0.88307077 reg_l1 25.86016 reg_l2 12.034224
loss 6.0551033
STEP 413 ================================
prereg loss 0.88249063 reg_l1 25.850555 reg_l2 12.031768
loss 6.052602
STEP 414 ================================
prereg loss 0.8818891 reg_l1 25.840439 reg_l2 12.02936
loss 6.0499773
STEP 415 ================================
prereg loss 0.8810197 reg_l1 25.83148 reg_l2 12.027406
loss 6.0473156
STEP 416 ================================
prereg loss 0.8799299 reg_l1 25.824965 reg_l2 12.025954
loss 6.044923
STEP 417 ================================
prereg loss 0.87850815 reg_l1 25.818516 reg_l2 12.024664
loss 6.0422115
STEP 418 ================================
prereg loss 0.87779474 reg_l1 25.810553 reg_l2 12.0232
loss 6.0399055
STEP 419 ================================
prereg loss 0.87675864 reg_l1 25.80176 reg_l2 12.021581
loss 6.037111
STEP 420 ================================
prereg loss 0.876119 reg_l1 25.793507 reg_l2 12.019897
loss 6.0348206
STEP 421 ================================
prereg loss 0.87532604 reg_l1 25.785538 reg_l2 12.018388
loss 6.032434
STEP 422 ================================
prereg loss 0.8740839 reg_l1 25.77737 reg_l2 12.017057
loss 6.029558
STEP 423 ================================
prereg loss 0.87329876 reg_l1 25.769896 reg_l2 12.015724
loss 6.027278
STEP 424 ================================
prereg loss 0.8723925 reg_l1 25.762157 reg_l2 12.014344
loss 6.024824
STEP 425 ================================
prereg loss 0.87145865 reg_l1 25.754496 reg_l2 12.012867
loss 6.022358
STEP 426 ================================
prereg loss 0.8703121 reg_l1 25.746956 reg_l2 12.0113535
loss 6.0197034
STEP 427 ================================
prereg loss 0.86934894 reg_l1 25.739294 reg_l2 12.009629
loss 6.017208
STEP 428 ================================
prereg loss 0.86846197 reg_l1 25.731165 reg_l2 12.0078
loss 6.014695
STEP 429 ================================
prereg loss 0.8676811 reg_l1 25.723001 reg_l2 12.005935
loss 6.0122814
STEP 430 ================================
prereg loss 0.8670745 reg_l1 25.71422 reg_l2 12.004121
loss 6.0099187
STEP 431 ================================
prereg loss 0.86626637 reg_l1 25.706255 reg_l2 12.002625
loss 6.0075173
STEP 432 ================================
prereg loss 0.86502624 reg_l1 25.699242 reg_l2 12.001436
loss 6.004874
STEP 433 ================================
prereg loss 0.8641502 reg_l1 25.692415 reg_l2 12.000446
loss 6.002633
STEP 434 ================================
prereg loss 0.8628206 reg_l1 25.685595 reg_l2 11.999518
loss 5.9999394
STEP 435 ================================
prereg loss 0.8618167 reg_l1 25.6787 reg_l2 11.998398
loss 5.9975567
STEP 436 ================================
prereg loss 0.8609697 reg_l1 25.670536 reg_l2 11.996763
loss 5.9950767
STEP 437 ================================
prereg loss 0.8601922 reg_l1 25.661474 reg_l2 11.994649
loss 5.9924874
STEP 438 ================================
prereg loss 0.85956484 reg_l1 25.652193 reg_l2 11.992409
loss 5.9900036
STEP 439 ================================
prereg loss 0.85870516 reg_l1 25.642637 reg_l2 11.990261
loss 5.9872327
STEP 440 ================================
prereg loss 0.8576771 reg_l1 25.634394 reg_l2 11.988384
loss 5.9845557
STEP 441 ================================
prereg loss 0.85660183 reg_l1 25.627037 reg_l2 11.986707
loss 5.9820094
STEP 442 ================================
prereg loss 0.85564077 reg_l1 25.619118 reg_l2 11.985033
loss 5.9794645
STEP 443 ================================
prereg loss 0.854857 reg_l1 25.610325 reg_l2 11.983325
loss 5.976922
STEP 444 ================================
prereg loss 0.8539627 reg_l1 25.601418 reg_l2 11.98172
loss 5.9742465
STEP 445 ================================
prereg loss 0.8534417 reg_l1 25.593027 reg_l2 11.980262
loss 5.9720473
STEP 446 ================================
prereg loss 0.8524063 reg_l1 25.58517 reg_l2 11.978996
loss 5.969441
STEP 447 ================================
prereg loss 0.85145473 reg_l1 25.577374 reg_l2 11.97779
loss 5.9669294
STEP 448 ================================
prereg loss 0.8505255 reg_l1 25.56998 reg_l2 11.976496
loss 5.9645214
STEP 449 ================================
prereg loss 0.84950787 reg_l1 25.562445 reg_l2 11.9750805
loss 5.961997
STEP 450 ================================
prereg loss 0.848493 reg_l1 25.553976 reg_l2 11.973602
loss 5.9592886
STEP 451 ================================
prereg loss 0.84738684 reg_l1 25.54691 reg_l2 11.972107
loss 5.956769
STEP 452 ================================
prereg loss 0.84644425 reg_l1 25.538849 reg_l2 11.970442
loss 5.954214
STEP 453 ================================
prereg loss 0.8457799 reg_l1 25.529795 reg_l2 11.96847
loss 5.951739
STEP 454 ================================
prereg loss 0.84502864 reg_l1 25.520235 reg_l2 11.9665575
loss 5.9490757
STEP 455 ================================
prereg loss 0.84489393 reg_l1 25.511335 reg_l2 11.965002
loss 5.947161
STEP 456 ================================
prereg loss 0.8432193 reg_l1 25.504223 reg_l2 11.963846
loss 5.944064
STEP 457 ================================
prereg loss 0.84259677 reg_l1 25.497206 reg_l2 11.962664
loss 5.9420376
STEP 458 ================================
prereg loss 0.841715 reg_l1 25.489143 reg_l2 11.9613285
loss 5.9395437
STEP 459 ================================
prereg loss 0.8403982 reg_l1 25.4801 reg_l2 11.959808
loss 5.9364185
STEP 460 ================================
prereg loss 0.8402717 reg_l1 25.471104 reg_l2 11.958373
loss 5.934492
STEP 461 ================================
prereg loss 0.8388725 reg_l1 25.463968 reg_l2 11.957314
loss 5.9316664
STEP 462 ================================
prereg loss 0.83746153 reg_l1 25.4573 reg_l2 11.956402
loss 5.9289217
STEP 463 ================================
prereg loss 0.8366951 reg_l1 25.44973 reg_l2 11.95512
loss 5.9266415
STEP 464 ================================
prereg loss 0.83555275 reg_l1 25.441095 reg_l2 11.953404
loss 5.923772
STEP 465 ================================
prereg loss 0.8348916 reg_l1 25.431995 reg_l2 11.951476
loss 5.921291
STEP 466 ================================
prereg loss 0.83392346 reg_l1 25.423176 reg_l2 11.949654
loss 5.9185586
STEP 467 ================================
prereg loss 0.8331899 reg_l1 25.414501 reg_l2 11.947872
loss 5.9160905
STEP 468 ================================
prereg loss 0.8324579 reg_l1 25.405642 reg_l2 11.946234
loss 5.9135866
STEP 469 ================================
prereg loss 0.83142155 reg_l1 25.397236 reg_l2 11.944762
loss 5.9108686
STEP 470 ================================
prereg loss 0.830984 reg_l1 25.38875 reg_l2 11.94355
loss 5.9087343
STEP 471 ================================
prereg loss 0.8295816 reg_l1 25.38149 reg_l2 11.942603
loss 5.90588
STEP 472 ================================
prereg loss 0.8285315 reg_l1 25.375015 reg_l2 11.941639
loss 5.903535
STEP 473 ================================
prereg loss 0.827516 reg_l1 25.367102 reg_l2 11.94031
loss 5.9009366
STEP 474 ================================
prereg loss 0.82663727 reg_l1 25.357664 reg_l2 11.938621
loss 5.89817
STEP 475 ================================
prereg loss 0.8261119 reg_l1 25.34896 reg_l2 11.936913
loss 5.895904
STEP 476 ================================
prereg loss 0.8248491 reg_l1 25.340748 reg_l2 11.935403
loss 5.8929987
STEP 477 ================================
prereg loss 0.8242226 reg_l1 25.33226 reg_l2 11.933803
loss 5.8906746
STEP 478 ================================
prereg loss 0.8233566 reg_l1 25.323202 reg_l2 11.932091
loss 5.887997
STEP 479 ================================
prereg loss 0.82228 reg_l1 25.313814 reg_l2 11.930373
loss 5.8850427
STEP 480 ================================
prereg loss 0.82241553 reg_l1 25.305073 reg_l2 11.928917
loss 5.88343
STEP 481 ================================
prereg loss 0.8207345 reg_l1 25.297958 reg_l2 11.927962
loss 5.8803263
STEP 482 ================================
prereg loss 0.8197461 reg_l1 25.290943 reg_l2 11.927143
loss 5.877935
STEP 483 ================================
prereg loss 0.81908935 reg_l1 25.283028 reg_l2 11.926077
loss 5.875695
STEP 484 ================================
prereg loss 0.8176427 reg_l1 25.273842 reg_l2 11.92462
loss 5.8724113
STEP 485 ================================
prereg loss 0.81711257 reg_l1 25.265768 reg_l2 11.923011
loss 5.870266
STEP 486 ================================
prereg loss 0.8160901 reg_l1 25.257475 reg_l2 11.921693
loss 5.867585
STEP 487 ================================
prereg loss 0.8145185 reg_l1 25.25022 reg_l2 11.920603
loss 5.8645625
STEP 488 ================================
prereg loss 0.81408876 reg_l1 25.24269 reg_l2 11.919143
loss 5.8626266
STEP 489 ================================
prereg loss 0.813144 reg_l1 25.23285 reg_l2 11.917351
loss 5.859714
STEP 490 ================================
prereg loss 0.8123467 reg_l1 25.221691 reg_l2 11.915525
loss 5.8566847
STEP 491 ================================
prereg loss 0.8118652 reg_l1 25.212484 reg_l2 11.91386
loss 5.8543625
STEP 492 ================================
prereg loss 0.81117517 reg_l1 25.205437 reg_l2 11.912504
loss 5.852263
STEP 493 ================================
prereg loss 0.8100901 reg_l1 25.198421 reg_l2 11.911616
loss 5.8497744
STEP 494 ================================
prereg loss 0.8087307 reg_l1 25.191217 reg_l2 11.910857
loss 5.8469744
STEP 495 ================================
prereg loss 0.8073534 reg_l1 25.182913 reg_l2 11.910018
loss 5.843936
STEP 496 ================================
prereg loss 0.8065421 reg_l1 25.17503 reg_l2 11.908977
loss 5.841548
STEP 497 ================================
prereg loss 0.8054158 reg_l1 25.167753 reg_l2 11.907916
loss 5.8389664
STEP 498 ================================
prereg loss 0.80445313 reg_l1 25.159071 reg_l2 11.906663
loss 5.8362675
STEP 499 ================================
prereg loss 0.8038212 reg_l1 25.149881 reg_l2 11.904983
loss 5.8337975
STEP 500 ================================
prereg loss 0.8029207 reg_l1 25.14105 reg_l2 11.903171
loss 5.831131
STEP 501 ================================
prereg loss 0.80228627 reg_l1 25.133253 reg_l2 11.901449
loss 5.828937
STEP 502 ================================
prereg loss 0.80116135 reg_l1 25.125689 reg_l2 11.899882
loss 5.826299
STEP 503 ================================
prereg loss 0.80037373 reg_l1 25.117477 reg_l2 11.898281
loss 5.823869
STEP 504 ================================
prereg loss 0.7995716 reg_l1 25.108334 reg_l2 11.896528
loss 5.8212385
STEP 505 ================================
prereg loss 0.7989232 reg_l1 25.099522 reg_l2 11.894674
loss 5.8188276
STEP 506 ================================
prereg loss 0.7981576 reg_l1 25.090788 reg_l2 11.892937
loss 5.816315
STEP 507 ================================
prereg loss 0.7969762 reg_l1 25.082567 reg_l2 11.891371
loss 5.8134894
STEP 508 ================================
prereg loss 0.7963082 reg_l1 25.073757 reg_l2 11.889637
loss 5.8110595
STEP 509 ================================
prereg loss 0.79529923 reg_l1 25.064579 reg_l2 11.887829
loss 5.808215
STEP 510 ================================
prereg loss 0.7946756 reg_l1 25.056324 reg_l2 11.886124
loss 5.8059406
STEP 511 ================================
prereg loss 0.79372674 reg_l1 25.048777 reg_l2 11.884678
loss 5.8034825
STEP 512 ================================
prereg loss 0.7926126 reg_l1 25.0416 reg_l2 11.883422
loss 5.8009324
STEP 513 ================================
prereg loss 0.7917269 reg_l1 25.034065 reg_l2 11.882109
loss 5.79854
STEP 514 ================================
prereg loss 0.79060227 reg_l1 25.025162 reg_l2 11.880582
loss 5.7956347
STEP 515 ================================
prereg loss 0.7900784 reg_l1 25.016155 reg_l2 11.87892
loss 5.793309
STEP 516 ================================
prereg loss 0.7891333 reg_l1 25.007715 reg_l2 11.877452
loss 5.790676
STEP 517 ================================
prereg loss 0.7883482 reg_l1 24.999603 reg_l2 11.876099
loss 5.788269
STEP 518 ================================
prereg loss 0.78734684 reg_l1 24.991968 reg_l2 11.874917
loss 5.7857404
STEP 519 ================================
prereg loss 0.7860436 reg_l1 24.983944 reg_l2 11.873839
loss 5.7828326
STEP 520 ================================
prereg loss 0.7851785 reg_l1 24.976408 reg_l2 11.872864
loss 5.7804604
STEP 521 ================================
prereg loss 0.78403056 reg_l1 24.969769 reg_l2 11.871989
loss 5.777984
STEP 522 ================================
prereg loss 0.7829149 reg_l1 24.962622 reg_l2 11.871061
loss 5.7754393
STEP 523 ================================
prereg loss 0.78226286 reg_l1 24.954165 reg_l2 11.869682
loss 5.7730956
STEP 524 ================================
prereg loss 0.7812841 reg_l1 24.944836 reg_l2 11.867913
loss 5.7702513
STEP 525 ================================
prereg loss 0.78057337 reg_l1 24.935884 reg_l2 11.865915
loss 5.7677503
STEP 526 ================================
prereg loss 0.779762 reg_l1 24.926634 reg_l2 11.863968
loss 5.765089
STEP 527 ================================
prereg loss 0.77935255 reg_l1 24.917044 reg_l2 11.862067
loss 5.7627616
STEP 528 ================================
prereg loss 0.7785079 reg_l1 24.908493 reg_l2 11.860513
loss 5.760206
STEP 529 ================================
prereg loss 0.77735263 reg_l1 24.900991 reg_l2 11.859243
loss 5.757551
STEP 530 ================================
prereg loss 0.7766216 reg_l1 24.893959 reg_l2 11.858206
loss 5.755413
STEP 531 ================================
prereg loss 0.77532166 reg_l1 24.88825 reg_l2 11.857359
loss 5.7529716
STEP 532 ================================
prereg loss 0.77424186 reg_l1 24.882252 reg_l2 11.856444
loss 5.7506924
STEP 533 ================================
prereg loss 0.773378 reg_l1 24.874275 reg_l2 11.855014
loss 5.748233
STEP 534 ================================
prereg loss 0.7723674 reg_l1 24.864954 reg_l2 11.853001
loss 5.7453585
STEP 535 ================================
prereg loss 0.77180743 reg_l1 24.856262 reg_l2 11.85086
loss 5.74306
STEP 536 ================================
prereg loss 0.7708124 reg_l1 24.847015 reg_l2 11.84889
loss 5.740216
STEP 537 ================================
prereg loss 0.770204 reg_l1 24.838068 reg_l2 11.846921
loss 5.737818
STEP 538 ================================
prereg loss 0.76939744 reg_l1 24.830112 reg_l2 11.844943
loss 5.7354198
STEP 539 ================================
prereg loss 0.76827717 reg_l1 24.821081 reg_l2 11.842971
loss 5.7324934
STEP 540 ================================
prereg loss 0.76770943 reg_l1 24.812113 reg_l2 11.840996
loss 5.730132
STEP 541 ================================
prereg loss 0.76676244 reg_l1 24.804379 reg_l2 11.839272
loss 5.7276382
STEP 542 ================================
prereg loss 0.7657501 reg_l1 24.797153 reg_l2 11.837919
loss 5.7251806
STEP 543 ================================
prereg loss 0.76479304 reg_l1 24.790047 reg_l2 11.836814
loss 5.722802
STEP 544 ================================
prereg loss 0.7636306 reg_l1 24.78355 reg_l2 11.835735
loss 5.7203407
STEP 545 ================================
prereg loss 0.7626363 reg_l1 24.775734 reg_l2 11.834455
loss 5.717783
STEP 546 ================================
prereg loss 0.7616977 reg_l1 24.767582 reg_l2 11.832786
loss 5.7152143
STEP 547 ================================
prereg loss 0.7609173 reg_l1 24.758972 reg_l2 11.830849
loss 5.712712
STEP 548 ================================
prereg loss 0.7604785 reg_l1 24.750391 reg_l2 11.828901
loss 5.710557
STEP 549 ================================
prereg loss 0.7595407 reg_l1 24.741884 reg_l2 11.827419
loss 5.7079177
STEP 550 ================================
prereg loss 0.7583602 reg_l1 24.733498 reg_l2 11.82636
loss 5.70506
STEP 551 ================================
prereg loss 0.75727135 reg_l1 24.726397 reg_l2 11.825314
loss 5.702551
STEP 552 ================================
prereg loss 0.7564225 reg_l1 24.719536 reg_l2 11.824005
loss 5.70033
STEP 553 ================================
prereg loss 0.75553083 reg_l1 24.711939 reg_l2 11.8223095
loss 5.697919
STEP 554 ================================
prereg loss 0.7547892 reg_l1 24.702862 reg_l2 11.8204155
loss 5.6953616
STEP 555 ================================
prereg loss 0.75393355 reg_l1 24.693869 reg_l2 11.818566
loss 5.692707
STEP 556 ================================
prereg loss 0.75325394 reg_l1 24.685667 reg_l2 11.816753
loss 5.6903872
STEP 557 ================================
prereg loss 0.75233567 reg_l1 24.677916 reg_l2 11.815144
loss 5.6879187
STEP 558 ================================
prereg loss 0.7510833 reg_l1 24.670145 reg_l2 11.813702
loss 5.6851125
STEP 559 ================================
prereg loss 0.7504117 reg_l1 24.662226 reg_l2 11.812397
loss 5.6828566
STEP 560 ================================
prereg loss 0.74916136 reg_l1 24.65483 reg_l2 11.811274
loss 5.6801276
STEP 561 ================================
prereg loss 0.7482184 reg_l1 24.647787 reg_l2 11.810065
loss 5.677776
STEP 562 ================================
prereg loss 0.74743515 reg_l1 24.6387 reg_l2 11.808418
loss 5.675175
STEP 563 ================================
prereg loss 0.74668765 reg_l1 24.628122 reg_l2 11.80636
loss 5.672312
STEP 564 ================================
prereg loss 0.7462376 reg_l1 24.61851 reg_l2 11.804301
loss 5.6699395
STEP 565 ================================
prereg loss 0.7455582 reg_l1 24.610275 reg_l2 11.8025055
loss 5.6676135
STEP 566 ================================
prereg loss 0.74454623 reg_l1 24.60233 reg_l2 11.8011875
loss 5.6650124
STEP 567 ================================
prereg loss 0.74317265 reg_l1 24.59468 reg_l2 11.80014
loss 5.662109
STEP 568 ================================
prereg loss 0.7421514 reg_l1 24.58774 reg_l2 11.799074
loss 5.6596994
STEP 569 ================================
prereg loss 0.7411847 reg_l1 24.580832 reg_l2 11.798005
loss 5.657351
STEP 570 ================================
prereg loss 0.74005294 reg_l1 24.57344 reg_l2 11.796936
loss 5.6547413
STEP 571 ================================
prereg loss 0.73916596 reg_l1 24.565708 reg_l2 11.795604
loss 5.6523075
STEP 572 ================================
prereg loss 0.7383499 reg_l1 24.557339 reg_l2 11.793868
loss 5.649818
STEP 573 ================================
prereg loss 0.73760825 reg_l1 24.547947 reg_l2 11.791881
loss 5.6471977
STEP 574 ================================
prereg loss 0.73679686 reg_l1 24.538208 reg_l2 11.789863
loss 5.6444387
STEP 575 ================================
prereg loss 0.736302 reg_l1 24.529371 reg_l2 11.787859
loss 5.642176
STEP 576 ================================
prereg loss 0.7354202 reg_l1 24.521046 reg_l2 11.786175
loss 5.6396294
STEP 577 ================================
prereg loss 0.73414636 reg_l1 24.513006 reg_l2 11.784851
loss 5.6367474
STEP 578 ================================
prereg loss 0.73356897 reg_l1 24.505589 reg_l2 11.783794
loss 5.634687
STEP 579 ================================
prereg loss 0.73212874 reg_l1 24.498638 reg_l2 11.783006
loss 5.6318564
STEP 580 ================================
prereg loss 0.7311613 reg_l1 24.491907 reg_l2 11.782133
loss 5.629543
STEP 581 ================================
prereg loss 0.7302852 reg_l1 24.483974 reg_l2 11.780689
loss 5.62708
STEP 582 ================================
prereg loss 0.72939885 reg_l1 24.47505 reg_l2 11.778748
loss 5.6244087
STEP 583 ================================
prereg loss 0.72875553 reg_l1 24.464968 reg_l2 11.776724
loss 5.621749
STEP 584 ================================
prereg loss 0.7276882 reg_l1 24.456034 reg_l2 11.774874
loss 5.618895
STEP 585 ================================
prereg loss 0.7271905 reg_l1 24.447304 reg_l2 11.773001
loss 5.6166515
STEP 586 ================================
prereg loss 0.72645956 reg_l1 24.438334 reg_l2 11.771323
loss 5.614126
STEP 587 ================================
prereg loss 0.725396 reg_l1 24.429438 reg_l2 11.769976
loss 5.611284
STEP 588 ================================
prereg loss 0.7246754 reg_l1 24.421509 reg_l2 11.769052
loss 5.6089773
STEP 589 ================================
prereg loss 0.72320044 reg_l1 24.4148 reg_l2 11.768414
loss 5.6061606
STEP 590 ================================
prereg loss 0.72228646 reg_l1 24.408808 reg_l2 11.767663
loss 5.604048
STEP 591 ================================
prereg loss 0.7213608 reg_l1 24.400995 reg_l2 11.766337
loss 5.6015596
STEP 592 ================================
prereg loss 0.72043276 reg_l1 24.391376 reg_l2 11.764487
loss 5.598708
STEP 593 ================================
prereg loss 0.7197505 reg_l1 24.38157 reg_l2 11.762595
loss 5.5960646
STEP 594 ================================
prereg loss 0.7185419 reg_l1 24.372786 reg_l2 11.7609
loss 5.593099
STEP 595 ================================
prereg loss 0.7180387 reg_l1 24.36441 reg_l2 11.759154
loss 5.590921
STEP 596 ================================
prereg loss 0.7172393 reg_l1 24.355442 reg_l2 11.757554
loss 5.588328
STEP 597 ================================
prereg loss 0.716145 reg_l1 24.347178 reg_l2 11.756222
loss 5.585581
STEP 598 ================================
prereg loss 0.71523255 reg_l1 24.339525 reg_l2 11.755274
loss 5.5831375
STEP 599 ================================
prereg loss 0.71390873 reg_l1 24.331964 reg_l2 11.754527
loss 5.580302
STEP 600 ================================
prereg loss 0.71300197 reg_l1 24.325232 reg_l2 11.753672
loss 5.5780487
STEP 601 ================================
prereg loss 0.71214795 reg_l1 24.316916 reg_l2 11.752285
loss 5.575531
STEP 602 ================================
prereg loss 0.7113677 reg_l1 24.307247 reg_l2 11.75046
loss 5.5728173
STEP 603 ================================
prereg loss 0.71068484 reg_l1 24.297981 reg_l2 11.748625
loss 5.570281
STEP 604 ================================
prereg loss 0.709889 reg_l1 24.289248 reg_l2 11.746932
loss 5.5677385
STEP 605 ================================
prereg loss 0.70892537 reg_l1 24.280666 reg_l2 11.745501
loss 5.5650587
STEP 606 ================================
prereg loss 0.707641 reg_l1 24.272795 reg_l2 11.744263
loss 5.5622
STEP 607 ================================
prereg loss 0.70667213 reg_l1 24.26474 reg_l2 11.74294
loss 5.5596204
STEP 608 ================================
prereg loss 0.7057475 reg_l1 24.256252 reg_l2 11.741547
loss 5.5569983
STEP 609 ================================
prereg loss 0.7048648 reg_l1 24.247879 reg_l2 11.740147
loss 5.554441
STEP 610 ================================
prereg loss 0.7040393 reg_l1 24.238964 reg_l2 11.738673
loss 5.551832
STEP 611 ================================
prereg loss 0.70324904 reg_l1 24.229979 reg_l2 11.737185
loss 5.549245
STEP 612 ================================
prereg loss 0.70240253 reg_l1 24.221106 reg_l2 11.73579
loss 5.5466237
STEP 613 ================================
prereg loss 0.70156354 reg_l1 24.212997 reg_l2 11.734393
loss 5.5441628
STEP 614 ================================
prereg loss 0.70077085 reg_l1 24.204847 reg_l2 11.733194
loss 5.5417404
STEP 615 ================================
prereg loss 0.6996241 reg_l1 24.197065 reg_l2 11.732113
loss 5.539037
STEP 616 ================================
prereg loss 0.6987616 reg_l1 24.189232 reg_l2 11.730906
loss 5.5366077
STEP 617 ================================
prereg loss 0.6975945 reg_l1 24.180132 reg_l2 11.729445
loss 5.5336213
STEP 618 ================================
prereg loss 0.6968137 reg_l1 24.17073 reg_l2 11.72774
loss 5.5309596
STEP 619 ================================
prereg loss 0.6959188 reg_l1 24.162687 reg_l2 11.726137
loss 5.5284567
STEP 620 ================================
prereg loss 0.6951949 reg_l1 24.153625 reg_l2 11.724628
loss 5.52592
STEP 621 ================================
prereg loss 0.6943709 reg_l1 24.14514 reg_l2 11.723327
loss 5.523399
STEP 622 ================================
prereg loss 0.6932626 reg_l1 24.136303 reg_l2 11.722253
loss 5.520523
STEP 623 ================================
prereg loss 0.6923155 reg_l1 24.127537 reg_l2 11.721201
loss 5.517823
STEP 624 ================================
prereg loss 0.6914017 reg_l1 24.119247 reg_l2 11.720132
loss 5.515251
STEP 625 ================================
prereg loss 0.6904145 reg_l1 24.111547 reg_l2 11.719074
loss 5.512724
STEP 626 ================================
prereg loss 0.6894421 reg_l1 24.102919 reg_l2 11.717917
loss 5.510026
STEP 627 ================================
prereg loss 0.68849593 reg_l1 24.094555 reg_l2 11.716639
loss 5.507407
STEP 628 ================================
prereg loss 0.68760633 reg_l1 24.086027 reg_l2 11.715335
loss 5.504812
STEP 629 ================================
prereg loss 0.6866777 reg_l1 24.077486 reg_l2 11.714028
loss 5.5021753
STEP 630 ================================
prereg loss 0.6858326 reg_l1 24.068815 reg_l2 11.7127075
loss 5.4995956
STEP 631 ================================
prereg loss 0.684851 reg_l1 24.060856 reg_l2 11.711397
loss 5.4970226
STEP 632 ================================
prereg loss 0.68414634 reg_l1 24.053017 reg_l2 11.710116
loss 5.49475
STEP 633 ================================
prereg loss 0.68305075 reg_l1 24.046394 reg_l2 11.708982
loss 5.4923296
STEP 634 ================================
prereg loss 0.68202335 reg_l1 24.038355 reg_l2 11.707846
loss 5.4896946
STEP 635 ================================
prereg loss 0.68123317 reg_l1 24.029552 reg_l2 11.706367
loss 5.4871435
STEP 636 ================================
prereg loss 0.6803663 reg_l1 24.020418 reg_l2 11.704764
loss 5.4844503
STEP 637 ================================
prereg loss 0.67963064 reg_l1 24.012299 reg_l2 11.703232
loss 5.4820905
STEP 638 ================================
prereg loss 0.67889833 reg_l1 24.003883 reg_l2 11.701872
loss 5.4796753
STEP 639 ================================
prereg loss 0.67783165 reg_l1 23.996277 reg_l2 11.7008
loss 5.477087
STEP 640 ================================
prereg loss 0.6765102 reg_l1 23.98978 reg_l2 11.699805
loss 5.4744663
STEP 641 ================================
prereg loss 0.67598236 reg_l1 23.98249 reg_l2 11.698826
loss 5.472481
STEP 642 ================================
prereg loss 0.6747739 reg_l1 23.975647 reg_l2 11.69807
loss 5.469903
STEP 643 ================================
prereg loss 0.6736884 reg_l1 23.968527 reg_l2 11.6973295
loss 5.467394
STEP 644 ================================
prereg loss 0.6728909 reg_l1 23.959988 reg_l2 11.696089
loss 5.4648886
STEP 645 ================================
prereg loss 0.67198753 reg_l1 23.95149 reg_l2 11.694312
loss 5.4622855
STEP 646 ================================
prereg loss 0.67140996 reg_l1 23.94252 reg_l2 11.692488
loss 5.459914
STEP 647 ================================
prereg loss 0.6703551 reg_l1 23.934196 reg_l2 11.690921
loss 5.4571943
STEP 648 ================================
prereg loss 0.6698229 reg_l1 23.926222 reg_l2 11.689365
loss 5.455067
STEP 649 ================================
prereg loss 0.66885316 reg_l1 23.9188 reg_l2 11.687975
loss 5.4526134
STEP 650 ================================
prereg loss 0.66759837 reg_l1 23.910887 reg_l2 11.686811
loss 5.4497757
STEP 651 ================================
prereg loss 0.667073 reg_l1 23.9037 reg_l2 11.685976
loss 5.447813
STEP 652 ================================
prereg loss 0.66545314 reg_l1 23.897106 reg_l2 11.685496
loss 5.4448743
STEP 653 ================================
prereg loss 0.66457546 reg_l1 23.889708 reg_l2 11.684922
loss 5.4425173
STEP 654 ================================
prereg loss 0.6636659 reg_l1 23.880667 reg_l2 11.683712
loss 5.4397993
STEP 655 ================================
prereg loss 0.6629511 reg_l1 23.871492 reg_l2 11.682078
loss 5.4372497
STEP 656 ================================
prereg loss 0.66220725 reg_l1 23.86286 reg_l2 11.680618
loss 5.434779
STEP 657 ================================
prereg loss 0.6609908 reg_l1 23.8557 reg_l2 11.679423
loss 5.432131
STEP 658 ================================
prereg loss 0.66009593 reg_l1 23.848455 reg_l2 11.678215
loss 5.4297867
STEP 659 ================================
prereg loss 0.6590951 reg_l1 23.840122 reg_l2 11.676942
loss 5.4271197
STEP 660 ================================
prereg loss 0.65828764 reg_l1 23.831627 reg_l2 11.675704
loss 5.424613
STEP 661 ================================
prereg loss 0.6572094 reg_l1 23.824354 reg_l2 11.674611
loss 5.4220805
STEP 662 ================================
prereg loss 0.6562534 reg_l1 23.815811 reg_l2 11.67358
loss 5.4194155
STEP 663 ================================
prereg loss 0.6553648 reg_l1 23.80752 reg_l2 11.672435
loss 5.416869
STEP 664 ================================
prereg loss 0.6545707 reg_l1 23.800497 reg_l2 11.67141
loss 5.41467
STEP 665 ================================
prereg loss 0.6534088 reg_l1 23.793354 reg_l2 11.670486
loss 5.41208
STEP 666 ================================
prereg loss 0.6525569 reg_l1 23.785175 reg_l2 11.669402
loss 5.409592
STEP 667 ================================
prereg loss 0.65154535 reg_l1 23.776447 reg_l2 11.668082
loss 5.406835
STEP 668 ================================
prereg loss 0.6508705 reg_l1 23.76816 reg_l2 11.666609
loss 5.4045024
STEP 669 ================================
prereg loss 0.64993674 reg_l1 23.759798 reg_l2 11.665311
loss 5.4018965
STEP 670 ================================
prereg loss 0.64895755 reg_l1 23.751598 reg_l2 11.664304
loss 5.3992777
STEP 671 ================================
prereg loss 0.6480884 reg_l1 23.744152 reg_l2 11.663303
loss 5.396919
STEP 672 ================================
prereg loss 0.6469936 reg_l1 23.737616 reg_l2 11.662387
loss 5.394517
STEP 673 ================================
prereg loss 0.64598095 reg_l1 23.730043 reg_l2 11.66132
loss 5.3919897
STEP 674 ================================
prereg loss 0.6449726 reg_l1 23.721693 reg_l2 11.660118
loss 5.3893113
STEP 675 ================================
prereg loss 0.64409107 reg_l1 23.713541 reg_l2 11.658744
loss 5.3867993
STEP 676 ================================
prereg loss 0.6433668 reg_l1 23.704937 reg_l2 11.65745
loss 5.384354
STEP 677 ================================
prereg loss 0.6423678 reg_l1 23.69622 reg_l2 11.656301
loss 5.381612
STEP 678 ================================
prereg loss 0.6414936 reg_l1 23.688265 reg_l2 11.6553545
loss 5.379147
STEP 679 ================================
prereg loss 0.64057666 reg_l1 23.680267 reg_l2 11.654397
loss 5.3766303
STEP 680 ================================
prereg loss 0.6397046 reg_l1 23.671858 reg_l2 11.653308
loss 5.3740764
STEP 681 ================================
prereg loss 0.63867474 reg_l1 23.664501 reg_l2 11.652208
loss 5.371575
STEP 682 ================================
prereg loss 0.6376711 reg_l1 23.657347 reg_l2 11.651133
loss 5.3691406
STEP 683 ================================
prereg loss 0.63668275 reg_l1 23.649714 reg_l2 11.650079
loss 5.366626
STEP 684 ================================
prereg loss 0.63564557 reg_l1 23.641296 reg_l2 11.649005
loss 5.363905
STEP 685 ================================
prereg loss 0.634791 reg_l1 23.633093 reg_l2 11.647788
loss 5.3614097
STEP 686 ================================
prereg loss 0.6340672 reg_l1 23.624279 reg_l2 11.646563
loss 5.358923
STEP 687 ================================
prereg loss 0.63302296 reg_l1 23.616608 reg_l2 11.645586
loss 5.356344
STEP 688 ================================
prereg loss 0.6321884 reg_l1 23.608898 reg_l2 11.644836
loss 5.353968
STEP 689 ================================
prereg loss 0.63100046 reg_l1 23.601377 reg_l2 11.644225
loss 5.351276
STEP 690 ================================
prereg loss 0.63012135 reg_l1 23.5942 reg_l2 11.643446
loss 5.3489614
STEP 691 ================================
prereg loss 0.62919056 reg_l1 23.585762 reg_l2 11.642202
loss 5.346343
STEP 692 ================================
prereg loss 0.6283877 reg_l1 23.576017 reg_l2 11.640643
loss 5.3435917
STEP 693 ================================
prereg loss 0.6276289 reg_l1 23.567507 reg_l2 11.639157
loss 5.3411303
STEP 694 ================================
prereg loss 0.6267787 reg_l1 23.558928 reg_l2 11.637884
loss 5.3385644
STEP 695 ================================
prereg loss 0.6258752 reg_l1 23.550484 reg_l2 11.636947
loss 5.335972
STEP 696 ================================
prereg loss 0.624641 reg_l1 23.54363 reg_l2 11.636337
loss 5.3333673
STEP 697 ================================
prereg loss 0.6235535 reg_l1 23.53717 reg_l2 11.635716
loss 5.330988
STEP 698 ================================
prereg loss 0.6226139 reg_l1 23.52909 reg_l2 11.634928
loss 5.328432
STEP 699 ================================
prereg loss 0.6215901 reg_l1 23.51988 reg_l2 11.63401
loss 5.3255663
STEP 700 ================================
prereg loss 0.62072784 reg_l1 23.511515 reg_l2 11.632988
loss 5.323031
2022-06-27T02:52:31.731

julia> serialize("sparse16-after-1500-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse16-after-1500-steps-opt.ser", opt)
```

1000 more steps (note that the training is much faster now because the models are smaller):

```
julia> steps!(1000)
2022-06-27T02:56:45.057
STEP 1 ================================
prereg loss 0.61975026 reg_l1 23.50365 reg_l2 11.63198
loss 5.3204803
STEP 2 ================================
prereg loss 0.6190258 reg_l1 23.495598 reg_l2 11.630826
loss 5.3181453
STEP 3 ================================
prereg loss 0.61799365 reg_l1 23.486809 reg_l2 11.629659
loss 5.315356
STEP 4 ================================
prereg loss 0.6175608 reg_l1 23.478539 reg_l2 11.628684
loss 5.3132687
STEP 5 ================================
prereg loss 0.61604345 reg_l1 23.472036 reg_l2 11.628087
loss 5.310451
STEP 6 ================================
prereg loss 0.6151759 reg_l1 23.465261 reg_l2 11.627449
loss 5.308228
STEP 7 ================================
prereg loss 0.6140918 reg_l1 23.456429 reg_l2 11.626396
loss 5.3053775
STEP 8 ================================
prereg loss 0.6132592 reg_l1 23.446882 reg_l2 11.625059
loss 5.3026357
STEP 9 ================================
prereg loss 0.61247194 reg_l1 23.438766 reg_l2 11.623904
loss 5.3002253
STEP 10 ================================
prereg loss 0.61129874 reg_l1 23.431154 reg_l2 11.623001
loss 5.2975297
STEP 11 ================================
prereg loss 0.61043006 reg_l1 23.423866 reg_l2 11.622037
loss 5.2952037
STEP 12 ================================
prereg loss 0.6095676 reg_l1 23.416307 reg_l2 11.620961
loss 5.292829
STEP 13 ================================
prereg loss 0.60870343 reg_l1 23.408016 reg_l2 11.619957
loss 5.290307
STEP 14 ================================
prereg loss 0.60756433 reg_l1 23.399776 reg_l2 11.619088
loss 5.28752
STEP 15 ================================
prereg loss 0.60680956 reg_l1 23.392313 reg_l2 11.618046
loss 5.285272
STEP 16 ================================
prereg loss 0.60575503 reg_l1 23.384365 reg_l2 11.61697
loss 5.282628
STEP 17 ================================
prereg loss 0.6050366 reg_l1 23.376959 reg_l2 11.616077
loss 5.2804284
STEP 18 ================================
prereg loss 0.6037103 reg_l1 23.369923 reg_l2 11.615475
loss 5.2776947
STEP 19 ================================
prereg loss 0.6027652 reg_l1 23.362337 reg_l2 11.614843
loss 5.275233
STEP 20 ================================
prereg loss 0.6017561 reg_l1 23.353857 reg_l2 11.613831
loss 5.2725277
STEP 21 ================================
prereg loss 0.60111964 reg_l1 23.345436 reg_l2 11.612526
loss 5.270207
STEP 22 ================================
prereg loss 0.6002265 reg_l1 23.336521 reg_l2 11.611389
loss 5.267531
STEP 23 ================================
prereg loss 0.5992841 reg_l1 23.328276 reg_l2 11.610606
loss 5.2649393
STEP 24 ================================
prereg loss 0.59830046 reg_l1 23.320395 reg_l2 11.609899
loss 5.2623796
STEP 25 ================================
prereg loss 0.59714437 reg_l1 23.312859 reg_l2 11.609192
loss 5.259716
STEP 26 ================================
prereg loss 0.5962115 reg_l1 23.304874 reg_l2 11.608266
loss 5.2571864
STEP 27 ================================
prereg loss 0.5952967 reg_l1 23.297132 reg_l2 11.607304
loss 5.2547235
STEP 28 ================================
prereg loss 0.5943162 reg_l1 23.28912 reg_l2 11.606358
loss 5.25214
STEP 29 ================================
prereg loss 0.5933664 reg_l1 23.280947 reg_l2 11.605541
loss 5.2495556
STEP 30 ================================
prereg loss 0.5924358 reg_l1 23.27325 reg_l2 11.60472
loss 5.247086
STEP 31 ================================
prereg loss 0.59148926 reg_l1 23.265938 reg_l2 11.603994
loss 5.244677
STEP 32 ================================
prereg loss 0.59058964 reg_l1 23.258951 reg_l2 11.603233
loss 5.2423797
STEP 33 ================================
prereg loss 0.5896705 reg_l1 23.25147 reg_l2 11.602421
loss 5.239965
STEP 34 ================================
prereg loss 0.5887631 reg_l1 23.24358 reg_l2 11.601578
loss 5.237479
STEP 35 ================================
prereg loss 0.58774126 reg_l1 23.235662 reg_l2 11.60075
loss 5.234874
STEP 36 ================================
prereg loss 0.5867505 reg_l1 23.228765 reg_l2 11.5998955
loss 5.232504
STEP 37 ================================
prereg loss 0.5857284 reg_l1 23.220522 reg_l2 11.599014
loss 5.2298326
STEP 38 ================================
prereg loss 0.5848242 reg_l1 23.212223 reg_l2 11.598066
loss 5.2272687
STEP 39 ================================
prereg loss 0.58394396 reg_l1 23.20481 reg_l2 11.597113
loss 5.224906
STEP 40 ================================
prereg loss 0.58294845 reg_l1 23.197865 reg_l2 11.5963745
loss 5.222522
STEP 41 ================================
prereg loss 0.5820295 reg_l1 23.190405 reg_l2 11.595668
loss 5.2201104
STEP 42 ================================
prereg loss 0.5809938 reg_l1 23.18213 reg_l2 11.59512
loss 5.21742
STEP 43 ================================
prereg loss 0.5799993 reg_l1 23.175524 reg_l2 11.594507
loss 5.215104
STEP 44 ================================
prereg loss 0.57906777 reg_l1 23.168371 reg_l2 11.593722
loss 5.212742
STEP 45 ================================
prereg loss 0.57825255 reg_l1 23.16159 reg_l2 11.592814
loss 5.2105703
STEP 46 ================================
prereg loss 0.57731926 reg_l1 23.15481 reg_l2 11.591957
loss 5.208281
STEP 47 ================================
prereg loss 0.5762458 reg_l1 23.148367 reg_l2 11.591143
loss 5.2059193
STEP 48 ================================
prereg loss 0.5754545 reg_l1 23.141335 reg_l2 11.59014
loss 5.203721
STEP 49 ================================
prereg loss 0.57441795 reg_l1 23.134623 reg_l2 11.589173
loss 5.2013426
STEP 50 ================================
prereg loss 0.5738044 reg_l1 23.127523 reg_l2 11.588483
loss 5.1993093
STEP 51 ================================
prereg loss 0.57231206 reg_l1 23.121265 reg_l2 11.588135
loss 5.196565
STEP 52 ================================
prereg loss 0.5715984 reg_l1 23.115028 reg_l2 11.587616
loss 5.1946044
STEP 53 ================================
prereg loss 0.570478 reg_l1 23.107548 reg_l2 11.58662
loss 5.1919875
STEP 54 ================================
prereg loss 0.56988084 reg_l1 23.098425 reg_l2 11.585339
loss 5.189566
STEP 55 ================================
prereg loss 0.56893 reg_l1 23.090319 reg_l2 11.584336
loss 5.186994
STEP 56 ================================
prereg loss 0.56835014 reg_l1 23.08276 reg_l2 11.583524
loss 5.184902
STEP 57 ================================
prereg loss 0.56742764 reg_l1 23.075996 reg_l2 11.58297
loss 5.182627
STEP 58 ================================
prereg loss 0.5659682 reg_l1 23.069965 reg_l2 11.58271
loss 5.179961
STEP 59 ================================
prereg loss 0.5652339 reg_l1 23.06342 reg_l2 11.582728
loss 5.1779175
STEP 60 ================================
prereg loss 0.5635975 reg_l1 23.057817 reg_l2 11.583021
loss 5.1751614
STEP 61 ================================
prereg loss 0.5625853 reg_l1 23.052042 reg_l2 11.583155
loss 5.1729937
STEP 62 ================================
prereg loss 0.5615497 reg_l1 23.045816 reg_l2 11.582514
loss 5.170713
STEP 63 ================================
prereg loss 0.56077415 reg_l1 23.037321 reg_l2 11.581307
loss 5.1682386
STEP 64 ================================
prereg loss 0.55993134 reg_l1 23.029034 reg_l2 11.580164
loss 5.165738
STEP 65 ================================
prereg loss 0.5588918 reg_l1 23.022633 reg_l2 11.579144
loss 5.1634183
STEP 66 ================================
prereg loss 0.55815244 reg_l1 23.014738 reg_l2 11.57807
loss 5.1611004
STEP 67 ================================
prereg loss 0.55726963 reg_l1 23.00661 reg_l2 11.577029
loss 5.1585917
STEP 68 ================================
prereg loss 0.5564388 reg_l1 22.998648 reg_l2 11.576055
loss 5.1561685
STEP 69 ================================
prereg loss 0.55547476 reg_l1 22.992062 reg_l2 11.575308
loss 5.1538873
STEP 70 ================================
prereg loss 0.5543395 reg_l1 22.984608 reg_l2 11.574732
loss 5.151261
STEP 71 ================================
prereg loss 0.55334663 reg_l1 22.977776 reg_l2 11.574179
loss 5.148902
STEP 72 ================================
prereg loss 0.5523059 reg_l1 22.971132 reg_l2 11.573725
loss 5.146532
STEP 73 ================================
prereg loss 0.5512692 reg_l1 22.964827 reg_l2 11.573464
loss 5.1442347
STEP 74 ================================
prereg loss 0.55015117 reg_l1 22.957754 reg_l2 11.573264
loss 5.141702
STEP 75 ================================
prereg loss 0.549309 reg_l1 22.950388 reg_l2 11.572821
loss 5.139387
STEP 76 ================================
prereg loss 0.5482276 reg_l1 22.9425 reg_l2 11.572316
loss 5.136728
STEP 77 ================================
prereg loss 0.5477268 reg_l1 22.93589 reg_l2 11.571862
loss 5.134905
STEP 78 ================================
prereg loss 0.5463355 reg_l1 22.929432 reg_l2 11.571668
loss 5.132222
STEP 79 ================================
prereg loss 0.5454313 reg_l1 22.92257 reg_l2 11.571463
loss 5.1299453
STEP 80 ================================
prereg loss 0.5443565 reg_l1 22.91538 reg_l2 11.570755
loss 5.1274323
STEP 81 ================================
prereg loss 0.5435406 reg_l1 22.907463 reg_l2 11.569775
loss 5.1250334
STEP 82 ================================
prereg loss 0.54251134 reg_l1 22.899937 reg_l2 11.568972
loss 5.122499
STEP 83 ================================
prereg loss 0.5418505 reg_l1 22.892601 reg_l2 11.568291
loss 5.120371
STEP 84 ================================
prereg loss 0.54086566 reg_l1 22.885517 reg_l2 11.567913
loss 5.1179695
STEP 85 ================================
prereg loss 0.53953797 reg_l1 22.878662 reg_l2 11.567734
loss 5.1152706
STEP 86 ================================
prereg loss 0.5387768 reg_l1 22.871918 reg_l2 11.567709
loss 5.1131606
STEP 87 ================================
prereg loss 0.5373701 reg_l1 22.86708 reg_l2 11.567905
loss 5.1107864
STEP 88 ================================
prereg loss 0.5363899 reg_l1 22.861134 reg_l2 11.567966
loss 5.108617
STEP 89 ================================
prereg loss 0.5353379 reg_l1 22.853909 reg_l2 11.567408
loss 5.1061196
STEP 90 ================================
prereg loss 0.53446317 reg_l1 22.846064 reg_l2 11.566412
loss 5.103676
STEP 91 ================================
prereg loss 0.5335964 reg_l1 22.83789 reg_l2 11.565348
loss 5.101175
STEP 92 ================================
prereg loss 0.53247696 reg_l1 22.831991 reg_l2 11.564409
loss 5.098875
STEP 93 ================================
prereg loss 0.5318444 reg_l1 22.825249 reg_l2 11.563233
loss 5.0968943
STEP 94 ================================
prereg loss 0.5309412 reg_l1 22.817577 reg_l2 11.562158
loss 5.0944567
STEP 95 ================================
prereg loss 0.52993053 reg_l1 22.810326 reg_l2 11.561407
loss 5.0919957
STEP 96 ================================
prereg loss 0.52874714 reg_l1 22.804258 reg_l2 11.560854
loss 5.0895987
STEP 97 ================================
prereg loss 0.52776134 reg_l1 22.797514 reg_l2 11.560261
loss 5.0872645
STEP 98 ================================
prereg loss 0.52672607 reg_l1 22.789679 reg_l2 11.55941
loss 5.084662
STEP 99 ================================
prereg loss 0.5259353 reg_l1 22.783089 reg_l2 11.558433
loss 5.082553
STEP 100 ================================
prereg loss 0.5248565 reg_l1 22.776358 reg_l2 11.557574
loss 5.080128
STEP 101 ================================
prereg loss 0.5240504 reg_l1 22.76932 reg_l2 11.556656
loss 5.077914
STEP 102 ================================
prereg loss 0.52302027 reg_l1 22.761728 reg_l2 11.555698
loss 5.075366
STEP 103 ================================
prereg loss 0.52245086 reg_l1 22.753922 reg_l2 11.554876
loss 5.0732355
STEP 104 ================================
prereg loss 0.5210301 reg_l1 22.747679 reg_l2 11.554388
loss 5.0705657
STEP 105 ================================
prereg loss 0.52020097 reg_l1 22.742031 reg_l2 11.553816
loss 5.0686073
STEP 106 ================================
prereg loss 0.5190392 reg_l1 22.733746 reg_l2 11.552811
loss 5.0657883
STEP 107 ================================
prereg loss 0.5184248 reg_l1 22.724844 reg_l2 11.5515785
loss 5.063394
STEP 108 ================================
prereg loss 0.51742524 reg_l1 22.717815 reg_l2 11.550632
loss 5.0609884
STEP 109 ================================
prereg loss 0.5163324 reg_l1 22.711578 reg_l2 11.550085
loss 5.058648
STEP 110 ================================
prereg loss 0.5153772 reg_l1 22.705233 reg_l2 11.549522
loss 5.0564237
STEP 111 ================================
prereg loss 0.5143026 reg_l1 22.698673 reg_l2 11.548856
loss 5.0540376
STEP 112 ================================
prereg loss 0.513369 reg_l1 22.6917 reg_l2 11.548074
loss 5.051709
STEP 113 ================================
prereg loss 0.5123204 reg_l1 22.684921 reg_l2 11.547289
loss 5.049305
STEP 114 ================================
prereg loss 0.51134056 reg_l1 22.677736 reg_l2 11.5463915
loss 5.046888
STEP 115 ================================
prereg loss 0.5105828 reg_l1 22.670513 reg_l2 11.545278
loss 5.044686
STEP 116 ================================
prereg loss 0.50958943 reg_l1 22.663227 reg_l2 11.544298
loss 5.0422354
STEP 117 ================================
prereg loss 0.50867 reg_l1 22.656164 reg_l2 11.543391
loss 5.0399027
STEP 118 ================================
prereg loss 0.50771034 reg_l1 22.649137 reg_l2 11.542756
loss 5.037538
STEP 119 ================================
prereg loss 0.50660354 reg_l1 22.642021 reg_l2 11.542206
loss 5.035008
STEP 120 ================================
prereg loss 0.5056632 reg_l1 22.635456 reg_l2 11.541517
loss 5.032755
STEP 121 ================================
prereg loss 0.50470716 reg_l1 22.628317 reg_l2 11.540673
loss 5.0303707
STEP 122 ================================
prereg loss 0.50374424 reg_l1 22.621386 reg_l2 11.539862
loss 5.0280213
STEP 123 ================================
prereg loss 0.50267875 reg_l1 22.614204 reg_l2 11.539097
loss 5.02552
STEP 124 ================================
prereg loss 0.5018875 reg_l1 22.607334 reg_l2 11.53814
loss 5.023354
STEP 125 ================================
prereg loss 0.5008457 reg_l1 22.600142 reg_l2 11.537247
loss 5.020874
STEP 126 ================================
prereg loss 0.4998992 reg_l1 22.592724 reg_l2 11.536432
loss 5.018444
STEP 127 ================================
prereg loss 0.4988008 reg_l1 22.5857 reg_l2 11.535923
loss 5.0159407
STEP 128 ================================
prereg loss 0.4977193 reg_l1 22.57915 reg_l2 11.535419
loss 5.0135493
STEP 129 ================================
prereg loss 0.49673674 reg_l1 22.57287 reg_l2 11.534711
loss 5.0113106
STEP 130 ================================
prereg loss 0.49578676 reg_l1 22.565016 reg_l2 11.533762
loss 5.00879
STEP 131 ================================
prereg loss 0.49495825 reg_l1 22.557041 reg_l2 11.532689
loss 5.0063667
STEP 132 ================================
prereg loss 0.49414277 reg_l1 22.549965 reg_l2 11.53187
loss 5.004136
STEP 133 ================================
prereg loss 0.49303356 reg_l1 22.543762 reg_l2 11.53126
loss 5.0017858
STEP 134 ================================
prereg loss 0.49192888 reg_l1 22.53693 reg_l2 11.530806
loss 4.9993153
STEP 135 ================================
prereg loss 0.4908949 reg_l1 22.52977 reg_l2 11.530195
loss 4.996849
STEP 136 ================================
prereg loss 0.48989466 reg_l1 22.522491 reg_l2 11.52938
loss 4.9943933
STEP 137 ================================
prereg loss 0.48894587 reg_l1 22.515379 reg_l2 11.528339
loss 4.992022
STEP 138 ================================
prereg loss 0.48806712 reg_l1 22.507616 reg_l2 11.527168
loss 4.9895906
STEP 139 ================================
prereg loss 0.48742217 reg_l1 22.499247 reg_l2 11.526173
loss 4.9872713
STEP 140 ================================
prereg loss 0.48615488 reg_l1 22.493027 reg_l2 11.525493
loss 4.9847603
STEP 141 ================================
prereg loss 0.48543778 reg_l1 22.486242 reg_l2 11.524787
loss 4.9826865
STEP 142 ================================
prereg loss 0.48437083 reg_l1 22.478792 reg_l2 11.524065
loss 4.9801292
STEP 143 ================================
prereg loss 0.48350334 reg_l1 22.471664 reg_l2 11.523511
loss 4.977836
STEP 144 ================================
prereg loss 0.48211518 reg_l1 22.465366 reg_l2 11.523197
loss 4.9751887
STEP 145 ================================
prereg loss 0.48108706 reg_l1 22.459503 reg_l2 11.522721
loss 4.972988
STEP 146 ================================
prereg loss 0.47998065 reg_l1 22.452532 reg_l2 11.521718
loss 4.970487
STEP 147 ================================
prereg loss 0.47924656 reg_l1 22.44444 reg_l2 11.520482
loss 4.968135
STEP 148 ================================
prereg loss 0.4782276 reg_l1 22.436577 reg_l2 11.5194235
loss 4.9655433
STEP 149 ================================
prereg loss 0.47732192 reg_l1 22.42945 reg_l2 11.518628
loss 4.963212
STEP 150 ================================
prereg loss 0.47651982 reg_l1 22.421852 reg_l2 11.518104
loss 4.960891
STEP 151 ================================
prereg loss 0.47526714 reg_l1 22.41483 reg_l2 11.517768
loss 4.958233
STEP 152 ================================
prereg loss 0.47432432 reg_l1 22.408594 reg_l2 11.517163
loss 4.9560432
STEP 153 ================================
prereg loss 0.4732836 reg_l1 22.401785 reg_l2 11.516181
loss 4.953641
STEP 154 ================================
prereg loss 0.47242987 reg_l1 22.393473 reg_l2 11.515028
loss 4.951124
STEP 155 ================================
prereg loss 0.47141266 reg_l1 22.385532 reg_l2 11.513988
loss 4.948519
STEP 156 ================================
prereg loss 0.47069648 reg_l1 22.378605 reg_l2 11.512951
loss 4.9464173
STEP 157 ================================
prereg loss 0.46975067 reg_l1 22.3711 reg_l2 11.512139
loss 4.9439707
STEP 158 ================================
prereg loss 0.46869582 reg_l1 22.363405 reg_l2 11.511701
loss 4.9413767
STEP 159 ================================
prereg loss 0.46745276 reg_l1 22.356804 reg_l2 11.511452
loss 4.938813
STEP 160 ================================
prereg loss 0.4665038 reg_l1 22.350117 reg_l2 11.511017
loss 4.9365273
STEP 161 ================================
prereg loss 0.4654772 reg_l1 22.343962 reg_l2 11.5101595
loss 4.9342694
STEP 162 ================================
prereg loss 0.4646142 reg_l1 22.336124 reg_l2 11.509197
loss 4.9318395
STEP 163 ================================
prereg loss 0.46358252 reg_l1 22.32744 reg_l2 11.508359
loss 4.9290705
STEP 164 ================================
prereg loss 0.46247977 reg_l1 22.319931 reg_l2 11.507685
loss 4.926466
STEP 165 ================================
prereg loss 0.46173579 reg_l1 22.313137 reg_l2 11.507151
loss 4.924363
STEP 166 ================================
prereg loss 0.46041057 reg_l1 22.306374 reg_l2 11.506832
loss 4.921685
STEP 167 ================================
prereg loss 0.4595526 reg_l1 22.299488 reg_l2 11.506331
loss 4.9194503
STEP 168 ================================
prereg loss 0.45853913 reg_l1 22.292244 reg_l2 11.50532
loss 4.916988
STEP 169 ================================
prereg loss 0.45765692 reg_l1 22.283504 reg_l2 11.50412
loss 4.9143577
STEP 170 ================================
prereg loss 0.45674592 reg_l1 22.275673 reg_l2 11.502964
loss 4.911881
STEP 171 ================================
prereg loss 0.45588374 reg_l1 22.269054 reg_l2 11.502081
loss 4.9096947
STEP 172 ================================
prereg loss 0.45466653 reg_l1 22.26232 reg_l2 11.501425
loss 4.9071307
STEP 173 ================================
prereg loss 0.4535792 reg_l1 22.255518 reg_l2 11.500906
loss 4.904683
STEP 174 ================================
prereg loss 0.4525585 reg_l1 22.248974 reg_l2 11.500299
loss 4.9023533
STEP 175 ================================
prereg loss 0.45163548 reg_l1 22.241364 reg_l2 11.499556
loss 4.899908
STEP 176 ================================
prereg loss 0.45072448 reg_l1 22.233192 reg_l2 11.498817
loss 4.897363
STEP 177 ================================
prereg loss 0.44965202 reg_l1 22.226282 reg_l2 11.498143
loss 4.894909
STEP 178 ================================
prereg loss 0.4487768 reg_l1 22.21858 reg_l2 11.497386
loss 4.892493
STEP 179 ================================
prereg loss 0.44777364 reg_l1 22.209957 reg_l2 11.496539
loss 4.889765
STEP 180 ================================
prereg loss 0.4468922 reg_l1 22.203016 reg_l2 11.495697
loss 4.8874955
STEP 181 ================================
prereg loss 0.4458518 reg_l1 22.196276 reg_l2 11.495001
loss 4.885107
STEP 182 ================================
prereg loss 0.44491735 reg_l1 22.188461 reg_l2 11.494383
loss 4.8826094
STEP 183 ================================
prereg loss 0.443805 reg_l1 22.181225 reg_l2 11.493822
loss 4.88005
STEP 184 ================================
prereg loss 0.44281375 reg_l1 22.174717 reg_l2 11.493193
loss 4.8777575
STEP 185 ================================
prereg loss 0.44175884 reg_l1 22.167809 reg_l2 11.492465
loss 4.8753204
STEP 186 ================================
prereg loss 0.44080076 reg_l1 22.160196 reg_l2 11.491733
loss 4.87284
STEP 187 ================================
prereg loss 0.43982846 reg_l1 22.152502 reg_l2 11.491069
loss 4.870329
STEP 188 ================================
prereg loss 0.43879014 reg_l1 22.144901 reg_l2 11.490459
loss 4.8677707
STEP 189 ================================
prereg loss 0.43797654 reg_l1 22.137909 reg_l2 11.48994
loss 4.865558
STEP 190 ================================
prereg loss 0.43678564 reg_l1 22.130373 reg_l2 11.489589
loss 4.86286
STEP 191 ================================
prereg loss 0.43590534 reg_l1 22.12345 reg_l2 11.489079
loss 4.860595
STEP 192 ================================
prereg loss 0.43487436 reg_l1 22.115425 reg_l2 11.488097
loss 4.8579597
STEP 193 ================================
prereg loss 0.43410525 reg_l1 22.10719 reg_l2 11.48701
loss 4.855543
STEP 194 ================================
prereg loss 0.43292755 reg_l1 22.099318 reg_l2 11.48618
loss 4.8527913
STEP 195 ================================
prereg loss 0.43188322 reg_l1 22.091879 reg_l2 11.4855175
loss 4.8502593
STEP 196 ================================
prereg loss 0.43093455 reg_l1 22.085186 reg_l2 11.484956
loss 4.847972
STEP 197 ================================
prereg loss 0.4298163 reg_l1 22.0784 reg_l2 11.484465
loss 4.845496
STEP 198 ================================
prereg loss 0.42886695 reg_l1 22.070724 reg_l2 11.4838295
loss 4.843012
STEP 199 ================================
prereg loss 0.427977 reg_l1 22.06227 reg_l2 11.482896
loss 4.840431
STEP 200 ================================
prereg loss 0.4271148 reg_l1 22.05314 reg_l2 11.481888
loss 4.8377433
STEP 201 ================================
prereg loss 0.42631412 reg_l1 22.046143 reg_l2 11.480937
loss 4.8355427
STEP 202 ================================
prereg loss 0.42526224 reg_l1 22.038965 reg_l2 11.4802475
loss 4.8330555
STEP 203 ================================
prereg loss 0.42421585 reg_l1 22.031576 reg_l2 11.479709
loss 4.830531
STEP 204 ================================
prereg loss 0.4230369 reg_l1 22.02435 reg_l2 11.479452
loss 4.827907
STEP 205 ================================
prereg loss 0.42185998 reg_l1 22.017473 reg_l2 11.479185
loss 4.8253546
STEP 206 ================================
prereg loss 0.42082176 reg_l1 22.0112 reg_l2 11.478701
loss 4.823062
STEP 207 ================================
prereg loss 0.419883 reg_l1 22.004282 reg_l2 11.477932
loss 4.8207397
STEP 208 ================================
prereg loss 0.41914415 reg_l1 21.996143 reg_l2 11.477189
loss 4.8183727
STEP 209 ================================
prereg loss 0.41791475 reg_l1 21.988699 reg_l2 11.4766445
loss 4.8156548
STEP 210 ================================
prereg loss 0.41706443 reg_l1 21.980516 reg_l2 11.475894
loss 4.8131676
STEP 211 ================================
prereg loss 0.41607475 reg_l1 21.972347 reg_l2 11.474931
loss 4.8105445
STEP 212 ================================
prereg loss 0.4152403 reg_l1 21.964933 reg_l2 11.474014
loss 4.808227
STEP 213 ================================
prereg loss 0.414347 reg_l1 21.957685 reg_l2 11.473306
loss 4.8058844
STEP 214 ================================
prereg loss 0.41325575 reg_l1 21.950481 reg_l2 11.472842
loss 4.803352
STEP 215 ================================
prereg loss 0.41199493 reg_l1 21.942692 reg_l2 11.472427
loss 4.8005333
STEP 216 ================================
prereg loss 0.41109186 reg_l1 21.935484 reg_l2 11.4718895
loss 4.7981887
STEP 217 ================================
prereg loss 0.41000485 reg_l1 21.928486 reg_l2 11.471389
loss 4.795702
STEP 218 ================================
prereg loss 0.40905076 reg_l1 21.920263 reg_l2 11.470719
loss 4.7931037
STEP 219 ================================
prereg loss 0.4081215 reg_l1 21.910889 reg_l2 11.46972
loss 4.7902994
STEP 220 ================================
prereg loss 0.40732855 reg_l1 21.903717 reg_l2 11.468581
loss 4.788072
STEP 221 ================================
prereg loss 0.40659747 reg_l1 21.89774 reg_l2 11.467765
loss 4.7861457
STEP 222 ================================
prereg loss 0.40525916 reg_l1 21.891178 reg_l2 11.467305
loss 4.783495
STEP 223 ================================
prereg loss 0.4041492 reg_l1 21.88401 reg_l2 11.466811
loss 4.780951
STEP 224 ================================
prereg loss 0.4031556 reg_l1 21.876707 reg_l2 11.466086
loss 4.778497
STEP 225 ================================
prereg loss 0.40212458 reg_l1 21.86985 reg_l2 11.465328
loss 4.7760944
STEP 226 ================================
prereg loss 0.40111533 reg_l1 21.861843 reg_l2 11.464517
loss 4.773484
STEP 227 ================================
prereg loss 0.40033087 reg_l1 21.853209 reg_l2 11.463488
loss 4.7709727
STEP 228 ================================
prereg loss 0.3993847 reg_l1 21.844566 reg_l2 11.462574
loss 4.7682977
STEP 229 ================================
prereg loss 0.39849058 reg_l1 21.837461 reg_l2 11.461807
loss 4.7659826
STEP 230 ================================
prereg loss 0.39739123 reg_l1 21.830536 reg_l2 11.461375
loss 4.763499
STEP 231 ================================
prereg loss 0.39637142 reg_l1 21.82442 reg_l2 11.460923
loss 4.7612557
STEP 232 ================================
prereg loss 0.39533603 reg_l1 21.81684 reg_l2 11.460225
loss 4.758704
STEP 233 ================================
prereg loss 0.39437333 reg_l1 21.809673 reg_l2 11.459357
loss 4.756308
STEP 234 ================================
prereg loss 0.39334258 reg_l1 21.802351 reg_l2 11.45856
loss 4.753813
STEP 235 ================================
prereg loss 0.39252535 reg_l1 21.794851 reg_l2 11.457747
loss 4.7514954
STEP 236 ================================
prereg loss 0.39155668 reg_l1 21.788456 reg_l2 11.457075
loss 4.749248
STEP 237 ================================
prereg loss 0.39039636 reg_l1 21.783184 reg_l2 11.45668
loss 4.747033
STEP 238 ================================
prereg loss 0.3892876 reg_l1 21.776352 reg_l2 11.456276
loss 4.744558
STEP 239 ================================
prereg loss 0.38830674 reg_l1 21.769299 reg_l2 11.455651
loss 4.7421665
STEP 240 ================================
prereg loss 0.38737777 reg_l1 21.761423 reg_l2 11.454695
loss 4.7396626
STEP 241 ================================
prereg loss 0.38647828 reg_l1 21.753681 reg_l2 11.453632
loss 4.7372146
STEP 242 ================================
prereg loss 0.38548306 reg_l1 21.746183 reg_l2 11.452523
loss 4.73472
STEP 243 ================================
prereg loss 0.38449684 reg_l1 21.739216 reg_l2 11.451235
loss 4.73234
STEP 244 ================================
prereg loss 0.38354796 reg_l1 21.732567 reg_l2 11.449887
loss 4.730061
STEP 245 ================================
prereg loss 0.38273558 reg_l1 21.725262 reg_l2 11.448546
loss 4.727788
STEP 246 ================================
prereg loss 0.38172263 reg_l1 21.71803 reg_l2 11.4474945
loss 4.725329
STEP 247 ================================
prereg loss 0.38087982 reg_l1 21.710745 reg_l2 11.446796
loss 4.723029
STEP 248 ================================
prereg loss 0.37956253 reg_l1 21.70477 reg_l2 11.446379
loss 4.720516
STEP 249 ================================
prereg loss 0.3786615 reg_l1 21.698454 reg_l2 11.445702
loss 4.7183523
STEP 250 ================================
prereg loss 0.37771168 reg_l1 21.689966 reg_l2 11.444552
loss 4.715705
STEP 251 ================================
prereg loss 0.3770024 reg_l1 21.682312 reg_l2 11.443501
loss 4.7134647
STEP 252 ================================
prereg loss 0.37587255 reg_l1 21.67521 reg_l2 11.442766
loss 4.710915
STEP 253 ================================
prereg loss 0.37479165 reg_l1 21.667633 reg_l2 11.44203
loss 4.708318
STEP 254 ================================
prereg loss 0.37412778 reg_l1 21.660282 reg_l2 11.4413185
loss 4.7061844
STEP 255 ================================
prereg loss 0.37269986 reg_l1 21.65493 reg_l2 11.440996
loss 4.7036858
STEP 256 ================================
prereg loss 0.37187853 reg_l1 21.64904 reg_l2 11.440555
loss 4.701687
STEP 257 ================================
prereg loss 0.3706405 reg_l1 21.641607 reg_l2 11.439496
loss 4.6989617
STEP 258 ================================
prereg loss 0.36992925 reg_l1 21.633757 reg_l2 11.438323
loss 4.6966805
STEP 259 ================================
prereg loss 0.368725 reg_l1 21.626385 reg_l2 11.437434
loss 4.6940017
STEP 260 ================================
prereg loss 0.36810145 reg_l1 21.618734 reg_l2 11.436452
loss 4.6918488
STEP 261 ================================
prereg loss 0.36717352 reg_l1 21.61127 reg_l2 11.435499
loss 4.689428
STEP 262 ================================
prereg loss 0.3661734 reg_l1 21.604332 reg_l2 11.434869
loss 4.68704
STEP 263 ================================
prereg loss 0.3648455 reg_l1 21.59831 reg_l2 11.434526
loss 4.6845074
STEP 264 ================================
prereg loss 0.3638877 reg_l1 21.59206 reg_l2 11.434058
loss 4.6823
STEP 265 ================================
prereg loss 0.36282092 reg_l1 21.584467 reg_l2 11.433106
loss 4.6797147
STEP 266 ================================
prereg loss 0.3619402 reg_l1 21.577574 reg_l2 11.4321165
loss 4.6774554
STEP 267 ================================
prereg loss 0.36087856 reg_l1 21.571112 reg_l2 11.431239
loss 4.675101
STEP 268 ================================
prereg loss 0.36000696 reg_l1 21.563076 reg_l2 11.430028
loss 4.672622
STEP 269 ================================
prereg loss 0.35940763 reg_l1 21.55447 reg_l2 11.428736
loss 4.6703014
STEP 270 ================================
prereg loss 0.35830572 reg_l1 21.546728 reg_l2 11.427828
loss 4.6676517
STEP 271 ================================
prereg loss 0.35726553 reg_l1 21.54013 reg_l2 11.427287
loss 4.665292
STEP 272 ================================
prereg loss 0.3562346 reg_l1 21.533901 reg_l2 11.426711
loss 4.663015
STEP 273 ================================
prereg loss 0.35522813 reg_l1 21.52668 reg_l2 11.425808
loss 4.660564
STEP 274 ================================
prereg loss 0.35432434 reg_l1 21.519663 reg_l2 11.424809
loss 4.658257
STEP 275 ================================
prereg loss 0.35323158 reg_l1 21.513138 reg_l2 11.423952
loss 4.655859
STEP 276 ================================
prereg loss 0.35231835 reg_l1 21.50585 reg_l2 11.423056
loss 4.653488
STEP 277 ================================
prereg loss 0.3513547 reg_l1 21.498503 reg_l2 11.422209
loss 4.6510553
STEP 278 ================================
prereg loss 0.3504765 reg_l1 21.491308 reg_l2 11.421347
loss 4.648738
STEP 279 ================================
prereg loss 0.34958753 reg_l1 21.48361 reg_l2 11.420538
loss 4.6463094
STEP 280 ================================
prereg loss 0.3485226 reg_l1 21.476368 reg_l2 11.4198885
loss 4.6437964
STEP 281 ================================
prereg loss 0.34768248 reg_l1 21.46958 reg_l2 11.419234
loss 4.6415987
STEP 282 ================================
prereg loss 0.34654185 reg_l1 21.462286 reg_l2 11.418591
loss 4.638999
STEP 283 ================================
prereg loss 0.3455802 reg_l1 21.45462 reg_l2 11.417908
loss 4.636504
STEP 284 ================================
prereg loss 0.34456953 reg_l1 21.448175 reg_l2 11.417279
loss 4.634205
STEP 285 ================================
prereg loss 0.34356973 reg_l1 21.44171 reg_l2 11.416684
loss 4.6319118
STEP 286 ================================
prereg loss 0.34254584 reg_l1 21.4352 reg_l2 11.416052
loss 4.629586
STEP 287 ================================
prereg loss 0.34159318 reg_l1 21.42799 reg_l2 11.415234
loss 4.6271915
STEP 288 ================================
prereg loss 0.34069636 reg_l1 21.419487 reg_l2 11.414154
loss 4.6245937
STEP 289 ================================
prereg loss 0.33989704 reg_l1 21.411985 reg_l2 11.413042
loss 4.6222944
STEP 290 ================================
prereg loss 0.33896476 reg_l1 21.405005 reg_l2 11.412305
loss 4.619966
STEP 291 ================================
prereg loss 0.33793023 reg_l1 21.398672 reg_l2 11.411766
loss 4.617665
STEP 292 ================================
prereg loss 0.336877 reg_l1 21.391445 reg_l2 11.411165
loss 4.615166
STEP 293 ================================
prereg loss 0.3360661 reg_l1 21.383871 reg_l2 11.410288
loss 4.6128407
STEP 294 ================================
prereg loss 0.3350062 reg_l1 21.3765 reg_l2 11.409549
loss 4.6103063
STEP 295 ================================
prereg loss 0.3340548 reg_l1 21.369543 reg_l2 11.40873
loss 4.6079636
STEP 296 ================================
prereg loss 0.3331271 reg_l1 21.362377 reg_l2 11.407653
loss 4.6056027
STEP 297 ================================
prereg loss 0.3323006 reg_l1 21.355576 reg_l2 11.40656
loss 4.603416
STEP 298 ================================
prereg loss 0.33144572 reg_l1 21.348028 reg_l2 11.405843
loss 4.6010513
STEP 299 ================================
prereg loss 0.3302887 reg_l1 21.340519 reg_l2 11.405397
loss 4.598393
STEP 300 ================================
prereg loss 0.3292097 reg_l1 21.33393 reg_l2 11.4048
loss 4.595996
STEP 301 ================================
prereg loss 0.3283938 reg_l1 21.326973 reg_l2 11.403978
loss 4.5937886
STEP 302 ================================
prereg loss 0.32740483 reg_l1 21.31946 reg_l2 11.403281
loss 4.591297
STEP 303 ================================
prereg loss 0.3265659 reg_l1 21.311583 reg_l2 11.402576
loss 4.5888824
STEP 304 ================================
prereg loss 0.325589 reg_l1 21.303757 reg_l2 11.401689
loss 4.5863404
STEP 305 ================================
prereg loss 0.32470647 reg_l1 21.296938 reg_l2 11.400864
loss 4.584094
STEP 306 ================================
prereg loss 0.32377893 reg_l1 21.289928 reg_l2 11.40018
loss 4.5817647
STEP 307 ================================
prereg loss 0.32269827 reg_l1 21.281935 reg_l2 11.39956
loss 4.5790854
STEP 308 ================================
prereg loss 0.32176012 reg_l1 21.274273 reg_l2 11.398889
loss 4.576615
STEP 309 ================================
prereg loss 0.32081205 reg_l1 21.268158 reg_l2 11.398257
loss 4.574444
STEP 310 ================================
prereg loss 0.31974456 reg_l1 21.261217 reg_l2 11.397693
loss 4.571988
STEP 311 ================================
prereg loss 0.31880063 reg_l1 21.254145 reg_l2 11.397029
loss 4.5696297
STEP 312 ================================
prereg loss 0.317888 reg_l1 21.246319 reg_l2 11.396293
loss 4.5671515
STEP 313 ================================
prereg loss 0.31712124 reg_l1 21.238525 reg_l2 11.395514
loss 4.564826
STEP 314 ================================
prereg loss 0.31610125 reg_l1 21.231243 reg_l2 11.394881
loss 4.56235
STEP 315 ================================
prereg loss 0.3151956 reg_l1 21.224213 reg_l2 11.394149
loss 4.560038
STEP 316 ================================
prereg loss 0.31430653 reg_l1 21.216413 reg_l2 11.393124
loss 4.5575895
STEP 317 ================================
prereg loss 0.3136204 reg_l1 21.209032 reg_l2 11.392273
loss 4.555427
STEP 318 ================================
prereg loss 0.31229934 reg_l1 21.202784 reg_l2 11.391785
loss 4.552856
STEP 319 ================================
prereg loss 0.31146446 reg_l1 21.196331 reg_l2 11.3911085
loss 4.5507307
STEP 320 ================================
prereg loss 0.31057996 reg_l1 21.188194 reg_l2 11.390033
loss 4.5482187
STEP 321 ================================
prereg loss 0.30970496 reg_l1 21.180746 reg_l2 11.389135
loss 4.545854
STEP 322 ================================
prereg loss 0.3088872 reg_l1 21.172853 reg_l2 11.38842
loss 4.543458
STEP 323 ================================
prereg loss 0.30791843 reg_l1 21.165695 reg_l2 11.387718
loss 4.5410576
STEP 324 ================================
prereg loss 0.30728436 reg_l1 21.157787 reg_l2 11.3870735
loss 4.5388417
STEP 325 ================================
prereg loss 0.30583096 reg_l1 21.152002 reg_l2 11.386891
loss 4.5362315
STEP 326 ================================
prereg loss 0.30502847 reg_l1 21.145567 reg_l2 11.386652
loss 4.534142
STEP 327 ================================
prereg loss 0.3039075 reg_l1 21.138147 reg_l2 11.385752
loss 4.531537
STEP 328 ================================
prereg loss 0.30323318 reg_l1 21.130844 reg_l2 11.384834
loss 4.5294023
STEP 329 ================================
prereg loss 0.30217093 reg_l1 21.123175 reg_l2 11.384206
loss 4.526806
STEP 330 ================================
prereg loss 0.3012065 reg_l1 21.11504 reg_l2 11.383535
loss 4.5242147
STEP 331 ================================
prereg loss 0.3005005 reg_l1 21.107311 reg_l2 11.382779
loss 4.5219626
STEP 332 ================================
prereg loss 0.29930252 reg_l1 21.102203 reg_l2 11.382289
loss 4.5197434
STEP 333 ================================
prereg loss 0.2984733 reg_l1 21.096292 reg_l2 11.381641
loss 4.517732
STEP 334 ================================
prereg loss 0.29757914 reg_l1 21.088224 reg_l2 11.380579
loss 4.5152245
STEP 335 ================================
prereg loss 0.2967739 reg_l1 21.080818 reg_l2 11.379564
loss 4.5129375
STEP 336 ================================
prereg loss 0.29575616 reg_l1 21.07407 reg_l2 11.378756
loss 4.5105705
STEP 337 ================================
prereg loss 0.29486507 reg_l1 21.06713 reg_l2 11.3778715
loss 4.5082912
STEP 338 ================================
prereg loss 0.2939486 reg_l1 21.06021 reg_l2 11.376977
loss 4.5059905
STEP 339 ================================
prereg loss 0.2931302 reg_l1 21.05276 reg_l2 11.376075
loss 4.503682
STEP 340 ================================
prereg loss 0.29211706 reg_l1 21.045446 reg_l2 11.375241
loss 4.5012064
STEP 341 ================================
prereg loss 0.29122663 reg_l1 21.039164 reg_l2 11.374416
loss 4.4990597
STEP 342 ================================
prereg loss 0.290331 reg_l1 21.033525 reg_l2 11.373629
loss 4.497036
STEP 343 ================================
prereg loss 0.28943294 reg_l1 21.026585 reg_l2 11.372696
loss 4.49475
STEP 344 ================================
prereg loss 0.28849262 reg_l1 21.019674 reg_l2 11.371692
loss 4.492428
STEP 345 ================================
prereg loss 0.28774908 reg_l1 21.013914 reg_l2 11.370724
loss 4.4905324
STEP 346 ================================
prereg loss 0.28672567 reg_l1 21.0081 reg_l2 11.370015
loss 4.4883456
STEP 347 ================================
prereg loss 0.2858712 reg_l1 21.002598 reg_l2 11.369563
loss 4.4863906
STEP 348 ================================
prereg loss 0.28471914 reg_l1 20.996943 reg_l2 11.369298
loss 4.4841075
STEP 349 ================================
prereg loss 0.28389338 reg_l1 20.989813 reg_l2 11.368658
loss 4.4818563
STEP 350 ================================
prereg loss 0.28314176 reg_l1 20.982908 reg_l2 11.36759
loss 4.4797235
STEP 351 ================================
prereg loss 0.2822622 reg_l1 20.976912 reg_l2 11.366623
loss 4.477645
STEP 352 ================================
prereg loss 0.2814381 reg_l1 20.970133 reg_l2 11.365612
loss 4.475465
STEP 353 ================================
prereg loss 0.2806236 reg_l1 20.962612 reg_l2 11.364401
loss 4.473146
STEP 354 ================================
prereg loss 0.27979583 reg_l1 20.955608 reg_l2 11.363242
loss 4.470917
STEP 355 ================================
prereg loss 0.27906016 reg_l1 20.949402 reg_l2 11.362261
loss 4.4689407
STEP 356 ================================
prereg loss 0.27803382 reg_l1 20.943323 reg_l2 11.361613
loss 4.4666986
STEP 357 ================================
prereg loss 0.2770036 reg_l1 20.938261 reg_l2 11.361218
loss 4.464656
STEP 358 ================================
prereg loss 0.27591118 reg_l1 20.932442 reg_l2 11.360954
loss 4.4624
STEP 359 ================================
prereg loss 0.27500156 reg_l1 20.925858 reg_l2 11.360513
loss 4.460173
STEP 360 ================================
prereg loss 0.2741226 reg_l1 20.919073 reg_l2 11.359643
loss 4.4579372
STEP 361 ================================
prereg loss 0.2733326 reg_l1 20.912493 reg_l2 11.358643
loss 4.455831
STEP 362 ================================
prereg loss 0.27242997 reg_l1 20.90601 reg_l2 11.357711
loss 4.453632
STEP 363 ================================
prereg loss 0.27167976 reg_l1 20.899187 reg_l2 11.3566
loss 4.4515176
STEP 364 ================================
prereg loss 0.27094117 reg_l1 20.892563 reg_l2 11.355251
loss 4.449454
STEP 365 ================================
prereg loss 0.27027592 reg_l1 20.885944 reg_l2 11.354036
loss 4.447465
STEP 366 ================================
prereg loss 0.26928455 reg_l1 20.879183 reg_l2 11.353249
loss 4.4451213
STEP 367 ================================
prereg loss 0.26850417 reg_l1 20.87336 reg_l2 11.352773
loss 4.4431763
STEP 368 ================================
prereg loss 0.2674155 reg_l1 20.86888 reg_l2 11.35255
loss 4.4411917
STEP 369 ================================
prereg loss 0.26653033 reg_l1 20.863111 reg_l2 11.352047
loss 4.4391527
STEP 370 ================================
prereg loss 0.26570576 reg_l1 20.855637 reg_l2 11.351215
loss 4.436833
STEP 371 ================================
prereg loss 0.26470745 reg_l1 20.849781 reg_l2 11.350504
loss 4.434664
STEP 372 ================================
prereg loss 0.26386204 reg_l1 20.84406 reg_l2 11.34976
loss 4.4326744
STEP 373 ================================
prereg loss 0.26303983 reg_l1 20.837303 reg_l2 11.348609
loss 4.430501
STEP 374 ================================
prereg loss 0.26235345 reg_l1 20.829567 reg_l2 11.347278
loss 4.428267
STEP 375 ================================
prereg loss 0.26147756 reg_l1 20.823055 reg_l2 11.346153
loss 4.426089
STEP 376 ================================
prereg loss 0.26096797 reg_l1 20.816828 reg_l2 11.345073
loss 4.4243336
STEP 377 ================================
prereg loss 0.26008523 reg_l1 20.810339 reg_l2 11.344191
loss 4.422153
STEP 378 ================================
prereg loss 0.2590632 reg_l1 20.804205 reg_l2 11.343674
loss 4.419904
STEP 379 ================================
prereg loss 0.25798348 reg_l1 20.7986 reg_l2 11.343382
loss 4.4177036
STEP 380 ================================
prereg loss 0.25713968 reg_l1 20.793116 reg_l2 11.342935
loss 4.415763
STEP 381 ================================
prereg loss 0.25633496 reg_l1 20.786175 reg_l2 11.342126
loss 4.41357
STEP 382 ================================
prereg loss 0.25551787 reg_l1 20.779184 reg_l2 11.341348
loss 4.411355
STEP 383 ================================
prereg loss 0.25461268 reg_l1 20.773376 reg_l2 11.340669
loss 4.4092884
STEP 384 ================================
prereg loss 0.25393513 reg_l1 20.766426 reg_l2 11.339714
loss 4.407221
STEP 385 ================================
prereg loss 0.25295144 reg_l1 20.760263 reg_l2 11.338854
loss 4.4050045
STEP 386 ================================
prereg loss 0.25211465 reg_l1 20.754436 reg_l2 11.33803
loss 4.4030023
STEP 387 ================================
prereg loss 0.25132254 reg_l1 20.748285 reg_l2 11.337217
loss 4.40098
STEP 388 ================================
prereg loss 0.25049007 reg_l1 20.742073 reg_l2 11.336406
loss 4.398905
STEP 389 ================================
prereg loss 0.24953239 reg_l1 20.735867 reg_l2 11.335738
loss 4.3967056
STEP 390 ================================
prereg loss 0.24870862 reg_l1 20.72981 reg_l2 11.334988
loss 4.394671
STEP 391 ================================
prereg loss 0.2479588 reg_l1 20.723415 reg_l2 11.334024
loss 4.392642
STEP 392 ================================
prereg loss 0.24706683 reg_l1 20.717106 reg_l2 11.333224
loss 4.390488
STEP 393 ================================
prereg loss 0.24633071 reg_l1 20.71125 reg_l2 11.332459
loss 4.388581
STEP 394 ================================
prereg loss 0.24549009 reg_l1 20.70393 reg_l2 11.331614
loss 4.3862762
STEP 395 ================================
prereg loss 0.244766 reg_l1 20.69761 reg_l2 11.330773
loss 4.384288
STEP 396 ================================
prereg loss 0.2439846 reg_l1 20.691854 reg_l2 11.330037
loss 4.3823557
STEP 397 ================================
prereg loss 0.24319133 reg_l1 20.686293 reg_l2 11.329404
loss 4.38045
STEP 398 ================================
prereg loss 0.242261 reg_l1 20.681162 reg_l2 11.328949
loss 4.3784933
STEP 399 ================================
prereg loss 0.2414331 reg_l1 20.67523 reg_l2 11.3284235
loss 4.376479
STEP 400 ================================
prereg loss 0.24056865 reg_l1 20.66889 reg_l2 11.327591
loss 4.3743467
STEP 401 ================================
prereg loss 0.23979786 reg_l1 20.663502 reg_l2 11.326668
loss 4.3724985
STEP 402 ================================
prereg loss 0.23898724 reg_l1 20.657763 reg_l2 11.325849
loss 4.37054
STEP 403 ================================
prereg loss 0.23817803 reg_l1 20.651495 reg_l2 11.325218
loss 4.3684773
STEP 404 ================================
prereg loss 0.23728849 reg_l1 20.645498 reg_l2 11.324694
loss 4.3663883
STEP 405 ================================
prereg loss 0.23659605 reg_l1 20.63991 reg_l2 11.324106
loss 4.3645782
STEP 406 ================================
prereg loss 0.23567607 reg_l1 20.63485 reg_l2 11.323624
loss 4.362646
STEP 407 ================================
prereg loss 0.23492555 reg_l1 20.62932 reg_l2 11.322963
loss 4.36079
STEP 408 ================================
prereg loss 0.23410201 reg_l1 20.622849 reg_l2 11.321875
loss 4.358672
STEP 409 ================================
prereg loss 0.23338726 reg_l1 20.616661 reg_l2 11.320756
loss 4.35672
STEP 410 ================================
prereg loss 0.23256193 reg_l1 20.61121 reg_l2 11.319778
loss 4.354804
STEP 411 ================================
prereg loss 0.23183748 reg_l1 20.60515 reg_l2 11.318461
loss 4.3528676
STEP 412 ================================
prereg loss 0.23120824 reg_l1 20.598476 reg_l2 11.31686
loss 4.3509035
STEP 413 ================================
prereg loss 0.23039086 reg_l1 20.59257 reg_l2 11.315498
loss 4.348905
STEP 414 ================================
prereg loss 0.22981875 reg_l1 20.58617 reg_l2 11.314146
loss 4.347053
STEP 415 ================================
prereg loss 0.22889265 reg_l1 20.580036 reg_l2 11.312783
loss 4.3449
STEP 416 ================================
prereg loss 0.2281503 reg_l1 20.57502 reg_l2 11.311539
loss 4.3431544
STEP 417 ================================
prereg loss 0.22728151 reg_l1 20.570164 reg_l2 11.310531
loss 4.3413143
STEP 418 ================================
prereg loss 0.2264165 reg_l1 20.564423 reg_l2 11.309443
loss 4.339301
STEP 419 ================================
prereg loss 0.22566953 reg_l1 20.558157 reg_l2 11.308346
loss 4.337301
STEP 420 ================================
prereg loss 0.22486356 reg_l1 20.55228 reg_l2 11.307332
loss 4.3353195
STEP 421 ================================
prereg loss 0.22415116 reg_l1 20.546705 reg_l2 11.306299
loss 4.3334923
STEP 422 ================================
prereg loss 0.22338116 reg_l1 20.54126 reg_l2 11.305343
loss 4.331633
STEP 423 ================================
prereg loss 0.22267117 reg_l1 20.535688 reg_l2 11.304296
loss 4.3298087
STEP 424 ================================
prereg loss 0.22196221 reg_l1 20.529055 reg_l2 11.303136
loss 4.327773
STEP 425 ================================
prereg loss 0.22114465 reg_l1 20.523458 reg_l2 11.3021
loss 4.3258367
STEP 426 ================================
prereg loss 0.22044112 reg_l1 20.517977 reg_l2 11.301109
loss 4.3240366
STEP 427 ================================
prereg loss 0.21954334 reg_l1 20.512562 reg_l2 11.300292
loss 4.322056
STEP 428 ================================
prereg loss 0.21872416 reg_l1 20.50728 reg_l2 11.299514
loss 4.3201804
STEP 429 ================================
prereg loss 0.21794866 reg_l1 20.501787 reg_l2 11.298639
loss 4.318306
STEP 430 ================================
prereg loss 0.21721342 reg_l1 20.495264 reg_l2 11.297539
loss 4.3162665
STEP 431 ================================
prereg loss 0.21649598 reg_l1 20.48974 reg_l2 11.296445
loss 4.314444
STEP 432 ================================
prereg loss 0.2158903 reg_l1 20.48444 reg_l2 11.295321
loss 4.3127785
STEP 433 ================================
prereg loss 0.21510571 reg_l1 20.478632 reg_l2 11.2944565
loss 4.310832
STEP 434 ================================
prereg loss 0.21416554 reg_l1 20.472725 reg_l2 11.293776
loss 4.3087106
STEP 435 ================================
prereg loss 0.21337885 reg_l1 20.467863 reg_l2 11.292873
loss 4.3069515
STEP 436 ================================
prereg loss 0.21265693 reg_l1 20.463 reg_l2 11.291699
loss 4.305257
STEP 437 ================================
prereg loss 0.21185248 reg_l1 20.457117 reg_l2 11.290608
loss 4.303276
STEP 438 ================================
prereg loss 0.21120998 reg_l1 20.451302 reg_l2 11.289348
loss 4.3014703
STEP 439 ================================
prereg loss 0.21056592 reg_l1 20.444094 reg_l2 11.288058
loss 4.299385
STEP 440 ================================
prereg loss 0.20975736 reg_l1 20.438313 reg_l2 11.287024
loss 4.29742
STEP 441 ================================
prereg loss 0.20908988 reg_l1 20.432962 reg_l2 11.286018
loss 4.2956824
STEP 442 ================================
prereg loss 0.20835382 reg_l1 20.426878 reg_l2 11.284873
loss 4.29373
STEP 443 ================================
prereg loss 0.20758015 reg_l1 20.421257 reg_l2 11.283862
loss 4.2918315
STEP 444 ================================
prereg loss 0.20681012 reg_l1 20.416359 reg_l2 11.282931
loss 4.290082
STEP 445 ================================
prereg loss 0.20610048 reg_l1 20.410646 reg_l2 11.281767
loss 4.28823
STEP 446 ================================
prereg loss 0.20547752 reg_l1 20.403503 reg_l2 11.280531
loss 4.2861786
STEP 447 ================================
prereg loss 0.20490408 reg_l1 20.398441 reg_l2 11.279471
loss 4.2845926
STEP 448 ================================
prereg loss 0.20399809 reg_l1 20.393307 reg_l2 11.278663
loss 4.2826595
STEP 449 ================================
prereg loss 0.20341255 reg_l1 20.387941 reg_l2 11.277879
loss 4.281001
STEP 450 ================================
prereg loss 0.20243667 reg_l1 20.383043 reg_l2 11.277427
loss 4.279045
STEP 451 ================================
prereg loss 0.20171121 reg_l1 20.377928 reg_l2 11.276687
loss 4.277297
STEP 452 ================================
prereg loss 0.2010726 reg_l1 20.371681 reg_l2 11.275579
loss 4.275409
STEP 453 ================================
prereg loss 0.20025064 reg_l1 20.365738 reg_l2 11.274692
loss 4.2733984
STEP 454 ================================
prereg loss 0.19956262 reg_l1 20.360619 reg_l2 11.2738495
loss 4.2716866
STEP 455 ================================
prereg loss 0.19886935 reg_l1 20.354887 reg_l2 11.272616
loss 4.269847
STEP 456 ================================
prereg loss 0.19832726 reg_l1 20.348461 reg_l2 11.271255
loss 4.268019
STEP 457 ================================
prereg loss 0.19756621 reg_l1 20.34224 reg_l2 11.270176
loss 4.266014
STEP 458 ================================
prereg loss 0.19689338 reg_l1 20.335854 reg_l2 11.26906
loss 4.264064
STEP 459 ================================
prereg loss 0.1961758 reg_l1 20.329687 reg_l2 11.267722
loss 4.262113
STEP 460 ================================
prereg loss 0.19547303 reg_l1 20.323915 reg_l2 11.2665
loss 4.2602563
STEP 461 ================================
prereg loss 0.19485731 reg_l1 20.318861 reg_l2 11.265394
loss 4.2586293
STEP 462 ================================
prereg loss 0.194034 reg_l1 20.313168 reg_l2 11.264499
loss 4.2566676
STEP 463 ================================
prereg loss 0.1932093 reg_l1 20.3078 reg_l2 11.2638
loss 4.2547693
STEP 464 ================================
prereg loss 0.19241051 reg_l1 20.302538 reg_l2 11.263307
loss 4.2529182
STEP 465 ================================
prereg loss 0.19164363 reg_l1 20.297565 reg_l2 11.262766
loss 4.251157
STEP 466 ================================
prereg loss 0.19092074 reg_l1 20.291193 reg_l2 11.261964
loss 4.2491593
STEP 467 ================================
prereg loss 0.19025114 reg_l1 20.284998 reg_l2 11.261048
loss 4.247251
STEP 468 ================================
prereg loss 0.18957967 reg_l1 20.27935 reg_l2 11.260075
loss 4.2454495
STEP 469 ================================
prereg loss 0.18896054 reg_l1 20.27318 reg_l2 11.258915
loss 4.2435966
STEP 470 ================================
prereg loss 0.18832783 reg_l1 20.26709 reg_l2 11.257703
loss 4.241746
STEP 471 ================================
prereg loss 0.18770793 reg_l1 20.261335 reg_l2 11.256481
loss 4.239975
STEP 472 ================================
prereg loss 0.18705879 reg_l1 20.255915 reg_l2 11.2554
loss 4.238242
STEP 473 ================================
prereg loss 0.18620005 reg_l1 20.251358 reg_l2 11.254598
loss 4.2364717
STEP 474 ================================
prereg loss 0.18553351 reg_l1 20.245169 reg_l2 11.253636
loss 4.234567
STEP 475 ================================
prereg loss 0.18488167 reg_l1 20.2387 reg_l2 11.252386
loss 4.232622
STEP 476 ================================
prereg loss 0.18425925 reg_l1 20.233337 reg_l2 11.251219
loss 4.230927
STEP 477 ================================
prereg loss 0.18364172 reg_l1 20.228046 reg_l2 11.250197
loss 4.2292514
STEP 478 ================================
prereg loss 0.18298876 reg_l1 20.221735 reg_l2 11.24917
loss 4.227336
STEP 479 ================================
prereg loss 0.1822973 reg_l1 20.215393 reg_l2 11.248312
loss 4.225376
STEP 480 ================================
prereg loss 0.18166861 reg_l1 20.209429 reg_l2 11.247478
loss 4.2235546
STEP 481 ================================
prereg loss 0.18118174 reg_l1 20.203808 reg_l2 11.246675
loss 4.2219434
STEP 482 ================================
prereg loss 0.1801835 reg_l1 20.198938 reg_l2 11.2462635
loss 4.219971
STEP 483 ================================
prereg loss 0.1794772 reg_l1 20.194204 reg_l2 11.245703
loss 4.218318
STEP 484 ================================
prereg loss 0.17878619 reg_l1 20.188433 reg_l2 11.244584
loss 4.216473
STEP 485 ================================
prereg loss 0.1780405 reg_l1 20.183084 reg_l2 11.24367
loss 4.2146573
STEP 486 ================================
prereg loss 0.17752594 reg_l1 20.1779 reg_l2 11.242837
loss 4.213106
STEP 487 ================================
prereg loss 0.1767111 reg_l1 20.172426 reg_l2 11.24167
loss 4.2111964
STEP 488 ================================
prereg loss 0.17651038 reg_l1 20.16594 reg_l2 11.240425
loss 4.209698
STEP 489 ================================
prereg loss 0.17540805 reg_l1 20.16071 reg_l2 11.239728
loss 4.20755
STEP 490 ================================
prereg loss 0.17499888 reg_l1 20.154373 reg_l2 11.238726
loss 4.2058735
STEP 491 ================================
prereg loss 0.17446098 reg_l1 20.147686 reg_l2 11.237157
loss 4.203998
STEP 492 ================================
prereg loss 0.17379473 reg_l1 20.14249 reg_l2 11.235991
loss 4.202293
STEP 493 ================================
prereg loss 0.17324144 reg_l1 20.137299 reg_l2 11.235116
loss 4.200701
STEP 494 ================================
prereg loss 0.17252862 reg_l1 20.130247 reg_l2 11.233849
loss 4.1985784
STEP 495 ================================
prereg loss 0.1721236 reg_l1 20.12336 reg_l2 11.232729
loss 4.1967955
STEP 496 ================================
prereg loss 0.17137134 reg_l1 20.117943 reg_l2 11.232121
loss 4.19496
STEP 497 ================================
prereg loss 0.17057543 reg_l1 20.112684 reg_l2 11.231738
loss 4.1931124
STEP 498 ================================
prereg loss 0.16965069 reg_l1 20.107758 reg_l2 11.231457
loss 4.191202
STEP 499 ================================
prereg loss 0.16881093 reg_l1 20.103138 reg_l2 11.231241
loss 4.1894383
STEP 500 ================================
prereg loss 0.16810118 reg_l1 20.098516 reg_l2 11.230821
loss 4.1878047
STEP 501 ================================
prereg loss 0.16740924 reg_l1 20.093302 reg_l2 11.230003
loss 4.18607
STEP 502 ================================
prereg loss 0.16691446 reg_l1 20.086885 reg_l2 11.228932
loss 4.184292
STEP 503 ================================
prereg loss 0.16631995 reg_l1 20.080631 reg_l2 11.227881
loss 4.182446
STEP 504 ================================
prereg loss 0.16582327 reg_l1 20.073912 reg_l2 11.226681
loss 4.180606
STEP 505 ================================
prereg loss 0.16534752 reg_l1 20.067472 reg_l2 11.225322
loss 4.178842
STEP 506 ================================
prereg loss 0.16478886 reg_l1 20.060793 reg_l2 11.224059
loss 4.1769476
STEP 507 ================================
prereg loss 0.164162 reg_l1 20.055641 reg_l2 11.222923
loss 4.1752906
STEP 508 ================================
prereg loss 0.16343711 reg_l1 20.050428 reg_l2 11.221735
loss 4.1735225
STEP 509 ================================
prereg loss 0.162809 reg_l1 20.044722 reg_l2 11.220477
loss 4.1717534
STEP 510 ================================
prereg loss 0.16232969 reg_l1 20.038456 reg_l2 11.219368
loss 4.170021
STEP 511 ================================
prereg loss 0.16150385 reg_l1 20.033455 reg_l2 11.218691
loss 4.168195
STEP 512 ================================
prereg loss 0.16082624 reg_l1 20.028717 reg_l2 11.218081
loss 4.1665697
STEP 513 ================================
prereg loss 0.16030118 reg_l1 20.022987 reg_l2 11.217494
loss 4.164899
STEP 514 ================================
prereg loss 0.15954606 reg_l1 20.017427 reg_l2 11.217066
loss 4.1630316
STEP 515 ================================
prereg loss 0.15896028 reg_l1 20.011272 reg_l2 11.216424
loss 4.161215
STEP 516 ================================
prereg loss 0.15843588 reg_l1 20.00486 reg_l2 11.215418
loss 4.159408
STEP 517 ================================
prereg loss 0.15792158 reg_l1 19.99861 reg_l2 11.2145
loss 4.157644
STEP 518 ================================
prereg loss 0.15732996 reg_l1 19.992966 reg_l2 11.213718
loss 4.155923
STEP 519 ================================
prereg loss 0.1566531 reg_l1 19.98658 reg_l2 11.212679
loss 4.1539693
STEP 520 ================================
prereg loss 0.15610002 reg_l1 19.980993 reg_l2 11.2115755
loss 4.1522985
STEP 521 ================================
prereg loss 0.15546373 reg_l1 19.976105 reg_l2 11.210619
loss 4.150685
STEP 522 ================================
prereg loss 0.15487628 reg_l1 19.969748 reg_l2 11.209624
loss 4.148826
STEP 523 ================================
prereg loss 0.1543619 reg_l1 19.962378 reg_l2 11.208529
loss 4.146837
STEP 524 ================================
prereg loss 0.15393835 reg_l1 19.956436 reg_l2 11.20777
loss 4.1452255
STEP 525 ================================
prereg loss 0.1530651 reg_l1 19.952976 reg_l2 11.207417
loss 4.1436605
STEP 526 ================================
prereg loss 0.15239248 reg_l1 19.947006 reg_l2 11.20662
loss 4.1417937
STEP 527 ================================
prereg loss 0.15205878 reg_l1 19.940649 reg_l2 11.205597
loss 4.1401887
STEP 528 ================================
prereg loss 0.15126362 reg_l1 19.935072 reg_l2 11.204944
loss 4.138278
STEP 529 ================================
prereg loss 0.15062073 reg_l1 19.929296 reg_l2 11.204117
loss 4.1364803
STEP 530 ================================
prereg loss 0.15012003 reg_l1 19.922884 reg_l2 11.203215
loss 4.134697
STEP 531 ================================
prereg loss 0.14945859 reg_l1 19.917631 reg_l2 11.2024555
loss 4.1329846
STEP 532 ================================
prereg loss 0.14887589 reg_l1 19.91194 reg_l2 11.201456
loss 4.1312637
STEP 533 ================================
prereg loss 0.1484153 reg_l1 19.905891 reg_l2 11.200236
loss 4.1295934
STEP 534 ================================
prereg loss 0.14778464 reg_l1 19.90039 reg_l2 11.199274
loss 4.127863
STEP 535 ================================
prereg loss 0.1471707 reg_l1 19.894545 reg_l2 11.198454
loss 4.1260796
STEP 536 ================================
prereg loss 0.14660545 reg_l1 19.889154 reg_l2 11.197668
loss 4.1244364
STEP 537 ================================
prereg loss 0.14603832 reg_l1 19.88383 reg_l2 11.196888
loss 4.122804
STEP 538 ================================
prereg loss 0.14550544 reg_l1 19.877903 reg_l2 11.195936
loss 4.121086
STEP 539 ================================
prereg loss 0.14496458 reg_l1 19.871222 reg_l2 11.195014
loss 4.119209
STEP 540 ================================
prereg loss 0.14443868 reg_l1 19.8654 reg_l2 11.194251
loss 4.117519
STEP 541 ================================
prereg loss 0.14376597 reg_l1 19.861025 reg_l2 11.193669
loss 4.115971
STEP 542 ================================
prereg loss 0.14321618 reg_l1 19.855839 reg_l2 11.19291
loss 4.114384
STEP 543 ================================
prereg loss 0.14271529 reg_l1 19.84915 reg_l2 11.191977
loss 4.1125455
STEP 544 ================================
prereg loss 0.14211318 reg_l1 19.843756 reg_l2 11.191174
loss 4.110864
STEP 545 ================================
prereg loss 0.1415446 reg_l1 19.838493 reg_l2 11.190352
loss 4.1092434
STEP 546 ================================
prereg loss 0.14099321 reg_l1 19.832417 reg_l2 11.189376
loss 4.1074767
STEP 547 ================================
prereg loss 0.14042255 reg_l1 19.82686 reg_l2 11.188401
loss 4.1057944
STEP 548 ================================
prereg loss 0.13991912 reg_l1 19.82081 reg_l2 11.187295
loss 4.104081
STEP 549 ================================
prereg loss 0.13945054 reg_l1 19.815691 reg_l2 11.186274
loss 4.1025887
STEP 550 ================================
prereg loss 0.13872272 reg_l1 19.810041 reg_l2 11.185542
loss 4.100731
STEP 551 ================================
prereg loss 0.13823974 reg_l1 19.804165 reg_l2 11.184679
loss 4.099073
STEP 552 ================================
prereg loss 0.13782062 reg_l1 19.797436 reg_l2 11.183547
loss 4.0973077
STEP 553 ================================
prereg loss 0.13730274 reg_l1 19.79096 reg_l2 11.182637
loss 4.0954947
STEP 554 ================================
prereg loss 0.13683167 reg_l1 19.784956 reg_l2 11.1818695
loss 4.093823
STEP 555 ================================
prereg loss 0.136209 reg_l1 19.778744 reg_l2 11.181199
loss 4.0919576
STEP 556 ================================
prereg loss 0.13552824 reg_l1 19.774212 reg_l2 11.180731
loss 4.0903707
STEP 557 ================================
prereg loss 0.13493825 reg_l1 19.769325 reg_l2 11.180372
loss 4.0888033
STEP 558 ================================
prereg loss 0.13430335 reg_l1 19.763786 reg_l2 11.180025
loss 4.087061
STEP 559 ================================
prereg loss 0.13374718 reg_l1 19.757614 reg_l2 11.179404
loss 4.08527
STEP 560 ================================
prereg loss 0.13324091 reg_l1 19.751665 reg_l2 11.178533
loss 4.083574
STEP 561 ================================
prereg loss 0.13272965 reg_l1 19.74588 reg_l2 11.177619
loss 4.081906
STEP 562 ================================
prereg loss 0.13223436 reg_l1 19.740158 reg_l2 11.176599
loss 4.080266
STEP 563 ================================
prereg loss 0.13181125 reg_l1 19.733837 reg_l2 11.175438
loss 4.078579
STEP 564 ================================
prereg loss 0.13138568 reg_l1 19.727522 reg_l2 11.174295
loss 4.07689
STEP 565 ================================
prereg loss 0.1309248 reg_l1 19.72062 reg_l2 11.173202
loss 4.075049
STEP 566 ================================
prereg loss 0.13043328 reg_l1 19.71546 reg_l2 11.172142
loss 4.0735254
STEP 567 ================================
prereg loss 0.12995751 reg_l1 19.710588 reg_l2 11.171064
loss 4.0720754
STEP 568 ================================
prereg loss 0.12943791 reg_l1 19.70415 reg_l2 11.170112
loss 4.0702677
STEP 569 ================================
prereg loss 0.12893145 reg_l1 19.698376 reg_l2 11.169235
loss 4.0686064
STEP 570 ================================
prereg loss 0.12849851 reg_l1 19.691973 reg_l2 11.168581
loss 4.066893
STEP 571 ================================
prereg loss 0.12768781 reg_l1 19.68786 reg_l2 11.168313
loss 4.06526
STEP 572 ================================
prereg loss 0.12706497 reg_l1 19.683699 reg_l2 11.167735
loss 4.0638046
STEP 573 ================================
prereg loss 0.1265627 reg_l1 19.677994 reg_l2 11.166864
loss 4.0621614
STEP 574 ================================
prereg loss 0.12594417 reg_l1 19.671719 reg_l2 11.1661005
loss 4.060288
STEP 575 ================================
prereg loss 0.12555228 reg_l1 19.665659 reg_l2 11.1652355
loss 4.0586843
STEP 576 ================================
prereg loss 0.12502733 reg_l1 19.660133 reg_l2 11.164357
loss 4.057054
STEP 577 ================================
prereg loss 0.124531895 reg_l1 19.654009 reg_l2 11.163548
loss 4.0553336
STEP 578 ================================
prereg loss 0.124203034 reg_l1 19.647495 reg_l2 11.162698
loss 4.0537024
STEP 579 ================================
prereg loss 0.12368351 reg_l1 19.640945 reg_l2 11.161819
loss 4.0518727
STEP 580 ================================
prereg loss 0.12327658 reg_l1 19.635077 reg_l2 11.161008
loss 4.050292
STEP 581 ================================
prereg loss 0.12272662 reg_l1 19.6299 reg_l2 11.160383
loss 4.0487065
STEP 582 ================================
prereg loss 0.12220303 reg_l1 19.623896 reg_l2 11.159545
loss 4.0469823
STEP 583 ================================
prereg loss 0.12175556 reg_l1 19.617523 reg_l2 11.1586
loss 4.04526
STEP 584 ================================
prereg loss 0.12119266 reg_l1 19.612495 reg_l2 11.157835
loss 4.0436916
STEP 585 ================================
prereg loss 0.120743774 reg_l1 19.606956 reg_l2 11.156983
loss 4.0421352
STEP 586 ================================
prereg loss 0.12028941 reg_l1 19.599995 reg_l2 11.15589
loss 4.0402884
STEP 587 ================================
prereg loss 0.11980225 reg_l1 19.59393 reg_l2 11.154912
loss 4.038588
STEP 588 ================================
prereg loss 0.119440764 reg_l1 19.58854 reg_l2 11.154013
loss 4.0371485
STEP 589 ================================
prereg loss 0.11881404 reg_l1 19.582954 reg_l2 11.153358
loss 4.035405
STEP 590 ================================
prereg loss 0.118258335 reg_l1 19.576954 reg_l2 11.152789
loss 4.033649
STEP 591 ================================
prereg loss 0.117776155 reg_l1 19.571987 reg_l2 11.152059
loss 4.0321736
STEP 592 ================================
prereg loss 0.117360905 reg_l1 19.565517 reg_l2 11.151226
loss 4.0304646
STEP 593 ================================
prereg loss 0.116848126 reg_l1 19.56014 reg_l2 11.150499
loss 4.0288763
STEP 594 ================================
prereg loss 0.11645367 reg_l1 19.555023 reg_l2 11.149593
loss 4.0274587
STEP 595 ================================
prereg loss 0.11611661 reg_l1 19.548388 reg_l2 11.148866
loss 4.025794
STEP 596 ================================
prereg loss 0.115399756 reg_l1 19.543169 reg_l2 11.148482
loss 4.0240335
STEP 597 ================================
prereg loss 0.11491096 reg_l1 19.538532 reg_l2 11.147676
loss 4.0226173
STEP 598 ================================
prereg loss 0.114694044 reg_l1 19.532532 reg_l2 11.146606
loss 4.0212
STEP 599 ================================
prereg loss 0.11406416 reg_l1 19.526432 reg_l2 11.145895
loss 4.0193505
STEP 600 ================================
prereg loss 0.11370927 reg_l1 19.519451 reg_l2 11.145009
loss 4.0175996
STEP 601 ================================
prereg loss 0.11331844 reg_l1 19.513668 reg_l2 11.14382
loss 4.0160522
STEP 602 ================================
prereg loss 0.11289083 reg_l1 19.507805 reg_l2 11.142817
loss 4.014452
STEP 603 ================================
prereg loss 0.11239044 reg_l1 19.502094 reg_l2 11.141954
loss 4.0128093
STEP 604 ================================
prereg loss 0.11192321 reg_l1 19.496061 reg_l2 11.141
loss 4.0111356
STEP 605 ================================
prereg loss 0.111827604 reg_l1 19.490171 reg_l2 11.140344
loss 4.009862
STEP 606 ================================
prereg loss 0.1110183 reg_l1 19.485868 reg_l2 11.140276
loss 4.008192
STEP 607 ================================
prereg loss 0.11040077 reg_l1 19.480036 reg_l2 11.139624
loss 4.006408
STEP 608 ================================
prereg loss 0.110266544 reg_l1 19.473513 reg_l2 11.138716
loss 4.004969
STEP 609 ================================
prereg loss 0.10945779 reg_l1 19.469435 reg_l2 11.138274
loss 4.003345
STEP 610 ================================
prereg loss 0.10901761 reg_l1 19.462833 reg_l2 11.1374855
loss 4.001584
STEP 611 ================================
prereg loss 0.10885459 reg_l1 19.455914 reg_l2 11.136354
loss 4.000037
STEP 612 ================================
prereg loss 0.10815276 reg_l1 19.45117 reg_l2 11.135655
loss 3.9983869
STEP 613 ================================
prereg loss 0.10800441 reg_l1 19.444904 reg_l2 11.134837
loss 3.9969852
STEP 614 ================================
prereg loss 0.10735304 reg_l1 19.438295 reg_l2 11.133882
loss 3.995012
STEP 615 ================================
prereg loss 0.1069141 reg_l1 19.432617 reg_l2 11.133029
loss 3.9934375
STEP 616 ================================
prereg loss 0.10646462 reg_l1 19.427896 reg_l2 11.132284
loss 3.992044
STEP 617 ================================
prereg loss 0.1060566 reg_l1 19.421127 reg_l2 11.131276
loss 3.9902823
STEP 618 ================================
prereg loss 0.10568341 reg_l1 19.414875 reg_l2 11.130333
loss 3.9886584
STEP 619 ================================
prereg loss 0.10535101 reg_l1 19.408865 reg_l2 11.129535
loss 3.987124
STEP 620 ================================
prereg loss 0.10478417 reg_l1 19.403065 reg_l2 11.128891
loss 3.9853973
STEP 621 ================================
prereg loss 0.10426282 reg_l1 19.397675 reg_l2 11.1283245
loss 3.9837978
STEP 622 ================================
prereg loss 0.10380357 reg_l1 19.392231 reg_l2 11.127673
loss 3.98225
STEP 623 ================================
prereg loss 0.10340237 reg_l1 19.386162 reg_l2 11.126858
loss 3.9806347
STEP 624 ================================
prereg loss 0.10301293 reg_l1 19.379162 reg_l2 11.1260195
loss 3.9788454
STEP 625 ================================
prereg loss 0.102715865 reg_l1 19.372974 reg_l2 11.125195
loss 3.977311
STEP 626 ================================
prereg loss 0.10217978 reg_l1 19.368134 reg_l2 11.1246395
loss 3.9758065
STEP 627 ================================
prereg loss 0.10171111 reg_l1 19.363369 reg_l2 11.12413
loss 3.9743848
STEP 628 ================================
prereg loss 0.10129164 reg_l1 19.357489 reg_l2 11.123432
loss 3.9727895
STEP 629 ================================
prereg loss 0.10090782 reg_l1 19.351128 reg_l2 11.122595
loss 3.9711335
STEP 630 ================================
prereg loss 0.10051082 reg_l1 19.344261 reg_l2 11.12178
loss 3.9693632
STEP 631 ================================
prereg loss 0.10012075 reg_l1 19.339144 reg_l2 11.120909
loss 3.9679496
STEP 632 ================================
prereg loss 0.0998224 reg_l1 19.333733 reg_l2 11.119865
loss 3.966569
STEP 633 ================================
prereg loss 0.09939282 reg_l1 19.328157 reg_l2 11.119026
loss 3.9650245
STEP 634 ================================
prereg loss 0.09905925 reg_l1 19.322308 reg_l2 11.118221
loss 3.963521
STEP 635 ================================
prereg loss 0.09871541 reg_l1 19.31485 reg_l2 11.117171
loss 3.9616854
STEP 636 ================================
prereg loss 0.09825279 reg_l1 19.309158 reg_l2 11.116376
loss 3.9600844
STEP 637 ================================
prereg loss 0.09796093 reg_l1 19.303417 reg_l2 11.115573
loss 3.9586444
STEP 638 ================================
prereg loss 0.09751374 reg_l1 19.297752 reg_l2 11.114811
loss 3.9570642
STEP 639 ================================
prereg loss 0.09688753 reg_l1 19.292562 reg_l2 11.114368
loss 3.9554002
STEP 640 ================================
prereg loss 0.096570924 reg_l1 19.286882 reg_l2 11.113855
loss 3.9539475
STEP 641 ================================
prereg loss 0.09621004 reg_l1 19.28061 reg_l2 11.113106
loss 3.9523318
STEP 642 ================================
prereg loss 0.09564684 reg_l1 19.274324 reg_l2 11.112608
loss 3.9505117
STEP 643 ================================
prereg loss 0.0953262 reg_l1 19.26887 reg_l2 11.112023
loss 3.9491
STEP 644 ================================
prereg loss 0.09497404 reg_l1 19.263237 reg_l2 11.110966
loss 3.9476216
STEP 645 ================================
prereg loss 0.09453086 reg_l1 19.257204 reg_l2 11.110111
loss 3.9459717
STEP 646 ================================
prereg loss 0.09426068 reg_l1 19.25095 reg_l2 11.1093025
loss 3.9444506
STEP 647 ================================
prereg loss 0.09383539 reg_l1 19.244812 reg_l2 11.108243
loss 3.942798
STEP 648 ================================
prereg loss 0.093427755 reg_l1 19.239061 reg_l2 11.107363
loss 3.94124
STEP 649 ================================
prereg loss 0.09308181 reg_l1 19.233618 reg_l2 11.106543
loss 3.9398053
STEP 650 ================================
prereg loss 0.09276774 reg_l1 19.226702 reg_l2 11.105463
loss 3.9381082
STEP 651 ================================
prereg loss 0.092402354 reg_l1 19.220844 reg_l2 11.104591
loss 3.9365714
STEP 652 ================================
prereg loss 0.09222149 reg_l1 19.215094 reg_l2 11.103866
loss 3.9352403
STEP 653 ================================
prereg loss 0.09163787 reg_l1 19.209423 reg_l2 11.1033535
loss 3.9335225
STEP 654 ================================
prereg loss 0.09150869 reg_l1 19.203596 reg_l2 11.1030445
loss 3.9322278
STEP 655 ================================
prereg loss 0.09082566 reg_l1 19.199318 reg_l2 11.103195
loss 3.930689
STEP 656 ================================
prereg loss 0.090242416 reg_l1 19.194382 reg_l2 11.1027
loss 3.9291189
STEP 657 ================================
prereg loss 0.090155944 reg_l1 19.188492 reg_l2 11.101988
loss 3.9278543
STEP 658 ================================
prereg loss 0.08954786 reg_l1 19.18268 reg_l2 11.101628
loss 3.926084
STEP 659 ================================
prereg loss 0.089133546 reg_l1 19.175915 reg_l2 11.100792
loss 3.9243164
STEP 660 ================================
prereg loss 0.089061484 reg_l1 19.169947 reg_l2 11.099681
loss 3.9230509
STEP 661 ================================
prereg loss 0.08845539 reg_l1 19.165737 reg_l2 11.098965
loss 3.921603
STEP 662 ================================
prereg loss 0.08822569 reg_l1 19.16017 reg_l2 11.098015
loss 3.9202595
STEP 663 ================================
prereg loss 0.087917 reg_l1 19.153719 reg_l2 11.0966625
loss 3.9186609
STEP 664 ================================
prereg loss 0.08746159 reg_l1 19.147188 reg_l2 11.095638
loss 3.9168992
STEP 665 ================================
prereg loss 0.08717867 reg_l1 19.14196 reg_l2 11.094626
loss 3.9155707
STEP 666 ================================
prereg loss 0.086866885 reg_l1 19.135899 reg_l2 11.093358
loss 3.9140465
STEP 667 ================================
prereg loss 0.08643571 reg_l1 19.130247 reg_l2 11.092424
loss 3.9124854
STEP 668 ================================
prereg loss 0.08606588 reg_l1 19.125679 reg_l2 11.091626
loss 3.9112017
STEP 669 ================================
prereg loss 0.085627094 reg_l1 19.1211 reg_l2 11.09065
loss 3.909847
STEP 670 ================================
prereg loss 0.0852393 reg_l1 19.115437 reg_l2 11.08969
loss 3.9083269
STEP 671 ================================
prereg loss 0.08481488 reg_l1 19.111073 reg_l2 11.088813
loss 3.9070294
STEP 672 ================================
prereg loss 0.08441383 reg_l1 19.105934 reg_l2 11.087797
loss 3.9056005
STEP 673 ================================
prereg loss 0.08407533 reg_l1 19.1002 reg_l2 11.086603
loss 3.9041157
STEP 674 ================================
prereg loss 0.08365549 reg_l1 19.095299 reg_l2 11.085533
loss 3.9027154
STEP 675 ================================
prereg loss 0.08331468 reg_l1 19.090117 reg_l2 11.084324
loss 3.901338
STEP 676 ================================
prereg loss 0.08297301 reg_l1 19.084402 reg_l2 11.082943
loss 3.8998535
STEP 677 ================================
prereg loss 0.08260515 reg_l1 19.07818 reg_l2 11.081534
loss 3.8982413
STEP 678 ================================
prereg loss 0.082250684 reg_l1 19.073153 reg_l2 11.080094
loss 3.896881
STEP 679 ================================
prereg loss 0.08194617 reg_l1 19.068037 reg_l2 11.07862
loss 3.8955536
STEP 680 ================================
prereg loss 0.08164508 reg_l1 19.062962 reg_l2 11.077219
loss 3.8942373
STEP 681 ================================
prereg loss 0.081341624 reg_l1 19.057806 reg_l2 11.075849
loss 3.8929029
STEP 682 ================================
prereg loss 0.081080854 reg_l1 19.051432 reg_l2 11.0743885
loss 3.8913672
STEP 683 ================================
prereg loss 0.080845274 reg_l1 19.045124 reg_l2 11.0730095
loss 3.8898702
STEP 684 ================================
prereg loss 0.080554545 reg_l1 19.040087 reg_l2 11.071815
loss 3.888572
STEP 685 ================================
prereg loss 0.08019715 reg_l1 19.034334 reg_l2 11.070496
loss 3.887064
STEP 686 ================================
prereg loss 0.07984254 reg_l1 19.028662 reg_l2 11.069281
loss 3.885575
STEP 687 ================================
prereg loss 0.0794576 reg_l1 19.024828 reg_l2 11.068164
loss 3.8844233
STEP 688 ================================
prereg loss 0.07908445 reg_l1 19.019686 reg_l2 11.067052
loss 3.8830216
STEP 689 ================================
prereg loss 0.07867547 reg_l1 19.014511 reg_l2 11.065995
loss 3.8815777
STEP 690 ================================
prereg loss 0.07839446 reg_l1 19.008904 reg_l2 11.064915
loss 3.880175
STEP 691 ================================
prereg loss 0.078041896 reg_l1 19.00367 reg_l2 11.0639515
loss 3.8787758
STEP 692 ================================
prereg loss 0.07761898 reg_l1 18.999258 reg_l2 11.063146
loss 3.8774707
STEP 693 ================================
prereg loss 0.07727726 reg_l1 18.993849 reg_l2 11.062003
loss 3.876047
STEP 694 ================================
prereg loss 0.077056386 reg_l1 18.987406 reg_l2 11.060713
loss 3.8745377
STEP 695 ================================
prereg loss 0.07674187 reg_l1 18.98168 reg_l2 11.05953
loss 3.8730779
STEP 696 ================================
prereg loss 0.07648748 reg_l1 18.97652 reg_l2 11.058144
loss 3.8717916
STEP 697 ================================
prereg loss 0.07624271 reg_l1 18.971033 reg_l2 11.056778
loss 3.8704493
STEP 698 ================================
prereg loss 0.07597919 reg_l1 18.965199 reg_l2 11.055496
loss 3.869019
STEP 699 ================================
prereg loss 0.075670026 reg_l1 18.959078 reg_l2 11.054163
loss 3.8674855
STEP 700 ================================
prereg loss 0.07541892 reg_l1 18.95366 reg_l2 11.052782
loss 3.8661509
STEP 701 ================================
prereg loss 0.075075164 reg_l1 18.949326 reg_l2 11.051606
loss 3.8649404
STEP 702 ================================
prereg loss 0.074732184 reg_l1 18.944128 reg_l2 11.05038
loss 3.8635578
STEP 703 ================================
prereg loss 0.074435905 reg_l1 18.938911 reg_l2 11.049176
loss 3.8622184
STEP 704 ================================
prereg loss 0.07409336 reg_l1 18.9335 reg_l2 11.048103
loss 3.8607934
STEP 705 ================================
prereg loss 0.07379486 reg_l1 18.928549 reg_l2 11.046992
loss 3.8595047
STEP 706 ================================
prereg loss 0.07350315 reg_l1 18.923359 reg_l2 11.045914
loss 3.858175
STEP 707 ================================
prereg loss 0.07327266 reg_l1 18.917906 reg_l2 11.044856
loss 3.856854
STEP 708 ================================
prereg loss 0.072899856 reg_l1 18.912287 reg_l2 11.043986
loss 3.8553572
STEP 709 ================================
prereg loss 0.07246798 reg_l1 18.90754 reg_l2 11.04331
loss 3.853976
STEP 710 ================================
prereg loss 0.072163254 reg_l1 18.903286 reg_l2 11.04243
loss 3.8528206
STEP 711 ================================
prereg loss 0.071887225 reg_l1 18.898226 reg_l2 11.041252
loss 3.8515325
STEP 712 ================================
prereg loss 0.071513146 reg_l1 18.892756 reg_l2 11.04019
loss 3.8500643
STEP 713 ================================
prereg loss 0.07123892 reg_l1 18.888016 reg_l2 11.039024
loss 3.8488421
STEP 714 ================================
prereg loss 0.07105894 reg_l1 18.88191 reg_l2 11.037574
loss 3.8474412
STEP 715 ================================
prereg loss 0.07075967 reg_l1 18.87683 reg_l2 11.036342
loss 3.8461256
STEP 716 ================================
prereg loss 0.07055544 reg_l1 18.871326 reg_l2 11.035166
loss 3.8448207
STEP 717 ================================
prereg loss 0.07032691 reg_l1 18.865494 reg_l2 11.03374
loss 3.8434255
STEP 718 ================================
prereg loss 0.07000724 reg_l1 18.860123 reg_l2 11.032605
loss 3.842032
STEP 719 ================================
prereg loss 0.069752626 reg_l1 18.85506 reg_l2 11.03151
loss 3.8407648
STEP 720 ================================
prereg loss 0.069476716 reg_l1 18.848999 reg_l2 11.030225
loss 3.8392766
STEP 721 ================================
prereg loss 0.06911567 reg_l1 18.844412 reg_l2 11.029187
loss 3.8379982
STEP 722 ================================
prereg loss 0.06881534 reg_l1 18.838919 reg_l2 11.028165
loss 3.836599
STEP 723 ================================
prereg loss 0.06856747 reg_l1 18.833847 reg_l2 11.027025
loss 3.835337
STEP 724 ================================
prereg loss 0.06819879 reg_l1 18.829643 reg_l2 11.0261
loss 3.8341274
STEP 725 ================================
prereg loss 0.06790279 reg_l1 18.824345 reg_l2 11.025072
loss 3.8327718
STEP 726 ================================
prereg loss 0.06759872 reg_l1 18.818558 reg_l2 11.023902
loss 3.8313105
STEP 727 ================================
prereg loss 0.06731676 reg_l1 18.813232 reg_l2 11.022706
loss 3.8299632
STEP 728 ================================
prereg loss 0.06706971 reg_l1 18.808502 reg_l2 11.021452
loss 3.8287702
STEP 729 ================================
prereg loss 0.0668187 reg_l1 18.802921 reg_l2 11.020203
loss 3.827403
STEP 730 ================================
prereg loss 0.06660362 reg_l1 18.795902 reg_l2 11.018934
loss 3.8257842
STEP 731 ================================
prereg loss 0.06634035 reg_l1 18.7907 reg_l2 11.017807
loss 3.8244808
STEP 732 ================================
prereg loss 0.06596853 reg_l1 18.786613 reg_l2 11.016856
loss 3.8232913
STEP 733 ================================
prereg loss 0.06570499 reg_l1 18.781363 reg_l2 11.015697
loss 3.8219776
STEP 734 ================================
prereg loss 0.06548374 reg_l1 18.775566 reg_l2 11.014437
loss 3.8205972
STEP 735 ================================
prereg loss 0.06525089 reg_l1 18.770187 reg_l2 11.013271
loss 3.8192885
STEP 736 ================================
prereg loss 0.06504578 reg_l1 18.764317 reg_l2 11.012023
loss 3.8179092
STEP 737 ================================
prereg loss 0.06489941 reg_l1 18.758484 reg_l2 11.010744
loss 3.8165963
STEP 738 ================================
prereg loss 0.06463296 reg_l1 18.753153 reg_l2 11.009667
loss 3.8152635
STEP 739 ================================
prereg loss 0.06437466 reg_l1 18.747478 reg_l2 11.0085
loss 3.8138704
STEP 740 ================================
prereg loss 0.06418734 reg_l1 18.742676 reg_l2 11.007296
loss 3.8127224
STEP 741 ================================
prereg loss 0.06384013 reg_l1 18.738419 reg_l2 11.006338
loss 3.811524
STEP 742 ================================
prereg loss 0.06353083 reg_l1 18.732853 reg_l2 11.005339
loss 3.8101015
STEP 743 ================================
prereg loss 0.06325892 reg_l1 18.727655 reg_l2 11.004212
loss 3.80879
STEP 744 ================================
prereg loss 0.06292259 reg_l1 18.722681 reg_l2 11.003208
loss 3.8074586
STEP 745 ================================
prereg loss 0.06265484 reg_l1 18.717527 reg_l2 11.002028
loss 3.8061602
STEP 746 ================================
prereg loss 0.0624711 reg_l1 18.711931 reg_l2 11.000774
loss 3.8048575
STEP 747 ================================
prereg loss 0.06220784 reg_l1 18.706621 reg_l2 10.999659
loss 3.8035321
STEP 748 ================================
prereg loss 0.061988574 reg_l1 18.700632 reg_l2 10.998448
loss 3.802115
STEP 749 ================================
prereg loss 0.061813686 reg_l1 18.695143 reg_l2 10.997154
loss 3.8008423
STEP 750 ================================
prereg loss 0.061551083 reg_l1 18.689466 reg_l2 10.996048
loss 3.7994444
STEP 751 ================================
prereg loss 0.061315082 reg_l1 18.684536 reg_l2 10.994793
loss 3.7982223
STEP 752 ================================
prereg loss 0.06108415 reg_l1 18.679789 reg_l2 10.993505
loss 3.797042
STEP 753 ================================
prereg loss 0.060807396 reg_l1 18.675352 reg_l2 10.992328
loss 3.795878
STEP 754 ================================
prereg loss 0.060563184 reg_l1 18.66935 reg_l2 10.991041
loss 3.794433
STEP 755 ================================
prereg loss 0.06035797 reg_l1 18.663774 reg_l2 10.989795
loss 3.793113
STEP 756 ================================
prereg loss 0.06011704 reg_l1 18.659428 reg_l2 10.988711
loss 3.7920027
STEP 757 ================================
prereg loss 0.059958976 reg_l1 18.653868 reg_l2 10.98766
loss 3.7907326
STEP 758 ================================
prereg loss 0.059559688 reg_l1 18.648617 reg_l2 10.986918
loss 3.789283
STEP 759 ================================
prereg loss 0.059317 reg_l1 18.643108 reg_l2 10.985928
loss 3.7879388
STEP 760 ================================
prereg loss 0.05918885 reg_l1 18.6376 reg_l2 10.984747
loss 3.7867088
STEP 761 ================================
prereg loss 0.058879606 reg_l1 18.633154 reg_l2 10.983799
loss 3.7855105
STEP 762 ================================
prereg loss 0.058669567 reg_l1 18.62765 reg_l2 10.982699
loss 3.7841995
STEP 763 ================================
prereg loss 0.05851475 reg_l1 18.621973 reg_l2 10.98145
loss 3.7829094
STEP 764 ================================
prereg loss 0.05823002 reg_l1 18.61657 reg_l2 10.980384
loss 3.781544
STEP 765 ================================
prereg loss 0.05796058 reg_l1 18.611261 reg_l2 10.979254
loss 3.7802129
STEP 766 ================================
prereg loss 0.057763997 reg_l1 18.606081 reg_l2 10.978019
loss 3.7789803
STEP 767 ================================
prereg loss 0.057489984 reg_l1 18.600994 reg_l2 10.976937
loss 3.7776887
STEP 768 ================================
prereg loss 0.057296835 reg_l1 18.59572 reg_l2 10.975759
loss 3.7764409
STEP 769 ================================
prereg loss 0.05710854 reg_l1 18.590084 reg_l2 10.974434
loss 3.7751255
STEP 770 ================================
prereg loss 0.056877725 reg_l1 18.584087 reg_l2 10.973263
loss 3.7736952
STEP 771 ================================
prereg loss 0.056700714 reg_l1 18.578632 reg_l2 10.972084
loss 3.7724273
STEP 772 ================================
prereg loss 0.056508243 reg_l1 18.57336 reg_l2 10.970855
loss 3.7711804
STEP 773 ================================
prereg loss 0.05629367 reg_l1 18.569674 reg_l2 10.96972
loss 3.7702284
STEP 774 ================================
prereg loss 0.056044348 reg_l1 18.563978 reg_l2 10.968631
loss 3.76884
STEP 775 ================================
prereg loss 0.055801995 reg_l1 18.557762 reg_l2 10.967508
loss 3.7673545
STEP 776 ================================
prereg loss 0.055599462 reg_l1 18.552374 reg_l2 10.96632
loss 3.7660742
STEP 777 ================================
prereg loss 0.05532141 reg_l1 18.548498 reg_l2 10.965292
loss 3.765021
STEP 778 ================================
prereg loss 0.055127706 reg_l1 18.543114 reg_l2 10.964233
loss 3.7637503
STEP 779 ================================
prereg loss 0.05495829 reg_l1 18.536747 reg_l2 10.963268
loss 3.762308
STEP 780 ================================
prereg loss 0.054573365 reg_l1 18.532675 reg_l2 10.96258
loss 3.7611084
STEP 781 ================================
prereg loss 0.054338854 reg_l1 18.527987 reg_l2 10.961517
loss 3.7599363
STEP 782 ================================
prereg loss 0.054351315 reg_l1 18.523512 reg_l2 10.960333
loss 3.7590537
STEP 783 ================================
prereg loss 0.05405825 reg_l1 18.518625 reg_l2 10.95947
loss 3.7577834
STEP 784 ================================
prereg loss 0.053820133 reg_l1 18.51288 reg_l2 10.95827
loss 3.7563963
STEP 785 ================================
prereg loss 0.053753607 reg_l1 18.50669 reg_l2 10.956951
loss 3.7550914
STEP 786 ================================
prereg loss 0.053471655 reg_l1 18.501646 reg_l2 10.95595
loss 3.7538009
STEP 787 ================================
prereg loss 0.053270422 reg_l1 18.497284 reg_l2 10.954815
loss 3.7527273
STEP 788 ================================
prereg loss 0.05319368 reg_l1 18.491793 reg_l2 10.9534855
loss 3.751552
STEP 789 ================================
prereg loss 0.052848905 reg_l1 18.487204 reg_l2 10.952493
loss 3.7502897
STEP 790 ================================
prereg loss 0.052646514 reg_l1 18.481844 reg_l2 10.9513235
loss 3.7490153
STEP 791 ================================
prereg loss 0.052592665 reg_l1 18.476017 reg_l2 10.950005
loss 3.7477963
STEP 792 ================================
prereg loss 0.05230085 reg_l1 18.47119 reg_l2 10.948997
loss 3.7465389
STEP 793 ================================
prereg loss 0.05209586 reg_l1 18.46633 reg_l2 10.947912
loss 3.7453618
STEP 794 ================================
prereg loss 0.051925838 reg_l1 18.461409 reg_l2 10.946761
loss 3.7442076
STEP 795 ================================
prereg loss 0.051677708 reg_l1 18.456131 reg_l2 10.945795
loss 3.742904
STEP 796 ================================
prereg loss 0.051457167 reg_l1 18.451113 reg_l2 10.944572
loss 3.7416797
STEP 797 ================================
prereg loss 0.051276006 reg_l1 18.445072 reg_l2 10.943393
loss 3.7402904
STEP 798 ================================
prereg loss 0.051047944 reg_l1 18.44006 reg_l2 10.942329
loss 3.73906
STEP 799 ================================
prereg loss 0.050866604 reg_l1 18.434925 reg_l2 10.941092
loss 3.7378516
STEP 800 ================================
prereg loss 0.050697755 reg_l1 18.429844 reg_l2 10.939853
loss 3.7366667
STEP 801 ================================
prereg loss 0.050530836 reg_l1 18.424612 reg_l2 10.938644
loss 3.7354534
STEP 802 ================================
prereg loss 0.050341025 reg_l1 18.41927 reg_l2 10.937435
loss 3.734195
STEP 803 ================================
prereg loss 0.050196897 reg_l1 18.413671 reg_l2 10.936195
loss 3.7329311
STEP 804 ================================
prereg loss 0.050026055 reg_l1 18.408081 reg_l2 10.93501
loss 3.7316422
STEP 805 ================================
prereg loss 0.049853284 reg_l1 18.40356 reg_l2 10.933811
loss 3.7305655
STEP 806 ================================
prereg loss 0.049673036 reg_l1 18.398516 reg_l2 10.932613
loss 3.7293763
STEP 807 ================================
prereg loss 0.049499933 reg_l1 18.39299 reg_l2 10.931486
loss 3.7280982
STEP 808 ================================
prereg loss 0.049303893 reg_l1 18.387924 reg_l2 10.930393
loss 3.726889
STEP 809 ================================
prereg loss 0.049098417 reg_l1 18.383278 reg_l2 10.929294
loss 3.725754
STEP 810 ================================
prereg loss 0.049169768 reg_l1 18.37791 reg_l2 10.928261
loss 3.724752
STEP 811 ================================
prereg loss 0.048724495 reg_l1 18.373487 reg_l2 10.927729
loss 3.723422
STEP 812 ================================
prereg loss 0.048388924 reg_l1 18.369335 reg_l2 10.926676
loss 3.722256
STEP 813 ================================
prereg loss 0.048431817 reg_l1 18.364235 reg_l2 10.925533
loss 3.721279
STEP 814 ================================
prereg loss 0.048125524 reg_l1 18.359026 reg_l2 10.924755
loss 3.7199306
STEP 815 ================================
prereg loss 0.047886018 reg_l1 18.353989 reg_l2 10.923536
loss 3.718684
STEP 816 ================================
prereg loss 0.047914132 reg_l1 18.348547 reg_l2 10.922205
loss 3.7176235
STEP 817 ================================
prereg loss 0.047664322 reg_l1 18.343668 reg_l2 10.921219
loss 3.716398
STEP 818 ================================
prereg loss 0.047486454 reg_l1 18.338846 reg_l2 10.919938
loss 3.7152557
STEP 819 ================================
prereg loss 0.04748598 reg_l1 18.332764 reg_l2 10.918594
loss 3.7140388
STEP 820 ================================
prereg loss 0.04722118 reg_l1 18.327131 reg_l2 10.917623
loss 3.7126474
STEP 821 ================================
prereg loss 0.046987314 reg_l1 18.322727 reg_l2 10.91639
loss 3.7115328
STEP 822 ================================
prereg loss 0.046835266 reg_l1 18.318354 reg_l2 10.915113
loss 3.710506
STEP 823 ================================
prereg loss 0.046580452 reg_l1 18.312994 reg_l2 10.914048
loss 3.7091794
STEP 824 ================================
prereg loss 0.046361733 reg_l1 18.3076 reg_l2 10.912805
loss 3.7078817
STEP 825 ================================
prereg loss 0.046170067 reg_l1 18.302483 reg_l2 10.911535
loss 3.7066665
STEP 826 ================================
prereg loss 0.04595282 reg_l1 18.297611 reg_l2 10.910352
loss 3.705475
STEP 827 ================================
prereg loss 0.045791276 reg_l1 18.292664 reg_l2 10.908905
loss 3.7043242
STEP 828 ================================
prereg loss 0.045692697 reg_l1 18.287807 reg_l2 10.907496
loss 3.7032542
STEP 829 ================================
prereg loss 0.0455617 reg_l1 18.28269 reg_l2 10.906231
loss 3.7020998
STEP 830 ================================
prereg loss 0.04547699 reg_l1 18.276928 reg_l2 10.904741
loss 3.7008626
STEP 831 ================================
prereg loss 0.045288023 reg_l1 18.271364 reg_l2 10.903488
loss 3.6995609
STEP 832 ================================
prereg loss 0.045188274 reg_l1 18.266443 reg_l2 10.90227
loss 3.6984768
STEP 833 ================================
prereg loss 0.045120247 reg_l1 18.262785 reg_l2 10.900846
loss 3.6976774
STEP 834 ================================
prereg loss 0.04485998 reg_l1 18.257923 reg_l2 10.899762
loss 3.6964445
STEP 835 ================================
prereg loss 0.04464866 reg_l1 18.252884 reg_l2 10.898512
loss 3.6952255
STEP 836 ================================
prereg loss 0.044548307 reg_l1 18.24874 reg_l2 10.897128
loss 3.6942961
STEP 837 ================================
prereg loss 0.04429378 reg_l1 18.24462 reg_l2 10.896006
loss 3.6932178
STEP 838 ================================
prereg loss 0.044074148 reg_l1 18.23952 reg_l2 10.894623
loss 3.691978
STEP 839 ================================
prereg loss 0.044015244 reg_l1 18.23517 reg_l2 10.89319
loss 3.6910493
STEP 840 ================================
prereg loss 0.043822598 reg_l1 18.230865 reg_l2 10.891994
loss 3.6899958
STEP 841 ================================
prereg loss 0.04366179 reg_l1 18.226171 reg_l2 10.89041
loss 3.6888962
STEP 842 ================================
prereg loss 0.043612923 reg_l1 18.220524 reg_l2 10.888883
loss 3.6877177
STEP 843 ================================
prereg loss 0.04351412 reg_l1 18.215626 reg_l2 10.887522
loss 3.6866393
STEP 844 ================================
prereg loss 0.043467354 reg_l1 18.210434 reg_l2 10.885953
loss 3.685554
STEP 845 ================================
prereg loss 0.04339942 reg_l1 18.205954 reg_l2 10.884525
loss 3.68459
STEP 846 ================================
prereg loss 0.043310586 reg_l1 18.200602 reg_l2 10.883167
loss 3.683431
STEP 847 ================================
prereg loss 0.043230984 reg_l1 18.196148 reg_l2 10.881718
loss 3.6824605
STEP 848 ================================
prereg loss 0.04306051 reg_l1 18.192034 reg_l2 10.880482
loss 3.6814673
STEP 849 ================================
prereg loss 0.042920344 reg_l1 18.18707 reg_l2 10.879219
loss 3.6803346
STEP 850 ================================
prereg loss 0.04276623 reg_l1 18.18231 reg_l2 10.877848
loss 3.6792283
STEP 851 ================================
prereg loss 0.042598996 reg_l1 18.178381 reg_l2 10.876579
loss 3.678275
STEP 852 ================================
prereg loss 0.04247316 reg_l1 18.174065 reg_l2 10.8752775
loss 3.6772861
STEP 853 ================================
prereg loss 0.042415656 reg_l1 18.168453 reg_l2 10.873789
loss 3.6761062
STEP 854 ================================
prereg loss 0.04228215 reg_l1 18.163189 reg_l2 10.872427
loss 3.6749198
STEP 855 ================================
prereg loss 0.042198457 reg_l1 18.158773 reg_l2 10.870952
loss 3.673953
STEP 856 ================================
prereg loss 0.042223718 reg_l1 18.15448 reg_l2 10.869365
loss 3.6731198
STEP 857 ================================
prereg loss 0.04204922 reg_l1 18.150608 reg_l2 10.868051
loss 3.6721709
STEP 858 ================================
prereg loss 0.04194591 reg_l1 18.144543 reg_l2 10.866534
loss 3.6708546
STEP 859 ================================
prereg loss 0.04196882 reg_l1 18.139048 reg_l2 10.864939
loss 3.6697783
STEP 860 ================================
prereg loss 0.04179354 reg_l1 18.13439 reg_l2 10.863663
loss 3.6686716
STEP 861 ================================
prereg loss 0.04164917 reg_l1 18.130968 reg_l2 10.862156
loss 3.6678429
STEP 862 ================================
prereg loss 0.041568276 reg_l1 18.12591 reg_l2 10.860648
loss 3.6667502
STEP 863 ================================
prereg loss 0.041422714 reg_l1 18.120777 reg_l2 10.859287
loss 3.6655781
STEP 864 ================================
prereg loss 0.041333854 reg_l1 18.11599 reg_l2 10.857843
loss 3.664532
STEP 865 ================================
prereg loss 0.041233197 reg_l1 18.111576 reg_l2 10.856442
loss 3.6635485
STEP 866 ================================
prereg loss 0.041134097 reg_l1 18.107098 reg_l2 10.855069
loss 3.6625538
STEP 867 ================================
prereg loss 0.041042313 reg_l1 18.10213 reg_l2 10.853677
loss 3.6614685
STEP 868 ================================
prereg loss 0.040950656 reg_l1 18.097628 reg_l2 10.852305
loss 3.6604762
STEP 869 ================================
prereg loss 0.040869232 reg_l1 18.09311 reg_l2 10.850834
loss 3.659491
STEP 870 ================================
prereg loss 0.040747266 reg_l1 18.088089 reg_l2 10.849465
loss 3.658365
STEP 871 ================================
prereg loss 0.04063863 reg_l1 18.083696 reg_l2 10.8481045
loss 3.657378
STEP 872 ================================
prereg loss 0.040520538 reg_l1 18.080181 reg_l2 10.846654
loss 3.6565566
STEP 873 ================================
prereg loss 0.040390242 reg_l1 18.075613 reg_l2 10.845293
loss 3.6555128
STEP 874 ================================
prereg loss 0.04029678 reg_l1 18.070683 reg_l2 10.843933
loss 3.6544333
STEP 875 ================================
prereg loss 0.040267173 reg_l1 18.065516 reg_l2 10.842501
loss 3.6533704
STEP 876 ================================
prereg loss 0.040188894 reg_l1 18.061243 reg_l2 10.841178
loss 3.6524374
STEP 877 ================================
prereg loss 0.04011127 reg_l1 18.05667 reg_l2 10.83983
loss 3.6514452
STEP 878 ================================
prereg loss 0.040067617 reg_l1 18.05124 reg_l2 10.838453
loss 3.650316
STEP 879 ================================
prereg loss 0.03995967 reg_l1 18.046158 reg_l2 10.837208
loss 3.6491914
STEP 880 ================================
prereg loss 0.039854325 reg_l1 18.041716 reg_l2 10.835831
loss 3.6481974
STEP 881 ================================
prereg loss 0.039721668 reg_l1 18.038244 reg_l2 10.834535
loss 3.6473706
STEP 882 ================================
prereg loss 0.03958789 reg_l1 18.0339 reg_l2 10.833267
loss 3.6463678
STEP 883 ================================
prereg loss 0.039517142 reg_l1 18.028612 reg_l2 10.831827
loss 3.6452396
STEP 884 ================================
prereg loss 0.039335605 reg_l1 18.023449 reg_l2 10.83057
loss 3.6440253
STEP 885 ================================
prereg loss 0.039219987 reg_l1 18.019604 reg_l2 10.829209
loss 3.6431408
STEP 886 ================================
prereg loss 0.03918344 reg_l1 18.014717 reg_l2 10.827743
loss 3.6421268
STEP 887 ================================
prereg loss 0.039051704 reg_l1 18.009995 reg_l2 10.826417
loss 3.6410508
STEP 888 ================================
prereg loss 0.039004534 reg_l1 18.00485 reg_l2 10.824934
loss 3.6399746
STEP 889 ================================
prereg loss 0.038974572 reg_l1 17.999979 reg_l2 10.8235
loss 3.6389704
STEP 890 ================================
prereg loss 0.038952384 reg_l1 17.994547 reg_l2 10.822177
loss 3.6378617
STEP 891 ================================
prereg loss 0.038907275 reg_l1 17.989288 reg_l2 10.820703
loss 3.636765
STEP 892 ================================
prereg loss 0.038784295 reg_l1 17.985525 reg_l2 10.819429
loss 3.6358893
STEP 893 ================================
prereg loss 0.038663004 reg_l1 17.981047 reg_l2 10.818275
loss 3.6348722
STEP 894 ================================
prereg loss 0.038531348 reg_l1 17.975933 reg_l2 10.817008
loss 3.633718
STEP 895 ================================
prereg loss 0.038394403 reg_l1 17.971169 reg_l2 10.81585
loss 3.6326282
STEP 896 ================================
prereg loss 0.03827245 reg_l1 17.967016 reg_l2 10.814661
loss 3.6316757
STEP 897 ================================
prereg loss 0.038167365 reg_l1 17.962719 reg_l2 10.813371
loss 3.6307113
STEP 898 ================================
prereg loss 0.03803718 reg_l1 17.957584 reg_l2 10.812111
loss 3.6295543
STEP 899 ================================
prereg loss 0.03795419 reg_l1 17.953453 reg_l2 10.810701
loss 3.6286447
STEP 900 ================================
prereg loss 0.037906464 reg_l1 17.948988 reg_l2 10.809281
loss 3.6277041
STEP 901 ================================
prereg loss 0.03779168 reg_l1 17.944548 reg_l2 10.807985
loss 3.6267014
STEP 902 ================================
prereg loss 0.037724923 reg_l1 17.93941 reg_l2 10.806562
loss 3.6256068
STEP 903 ================================
prereg loss 0.037703585 reg_l1 17.934212 reg_l2 10.805059
loss 3.6245458
STEP 904 ================================
prereg loss 0.037622232 reg_l1 17.930214 reg_l2 10.803748
loss 3.623665
STEP 905 ================================
prereg loss 0.037570916 reg_l1 17.925789 reg_l2 10.80234
loss 3.6227288
STEP 906 ================================
prereg loss 0.037499018 reg_l1 17.920362 reg_l2 10.801006
loss 3.6215715
STEP 907 ================================
prereg loss 0.037393935 reg_l1 17.91618 reg_l2 10.799828
loss 3.62063
STEP 908 ================================
prereg loss 0.03729349 reg_l1 17.912548 reg_l2 10.798499
loss 3.6198032
STEP 909 ================================
prereg loss 0.037153844 reg_l1 17.90804 reg_l2 10.79731
loss 3.6187618
STEP 910 ================================
prereg loss 0.037049275 reg_l1 17.902636 reg_l2 10.79612
loss 3.6175764
STEP 911 ================================
prereg loss 0.036986522 reg_l1 17.89767 reg_l2 10.794748
loss 3.616521
STEP 912 ================================
prereg loss 0.03686518 reg_l1 17.893064 reg_l2 10.793524
loss 3.6154783
STEP 913 ================================
prereg loss 0.036784265 reg_l1 17.88848 reg_l2 10.792285
loss 3.61448
STEP 914 ================================
prereg loss 0.03678162 reg_l1 17.883099 reg_l2 10.790909
loss 3.6134014
STEP 915 ================================
prereg loss 0.036662094 reg_l1 17.878496 reg_l2 10.789727
loss 3.6123614
STEP 916 ================================
prereg loss 0.036593515 reg_l1 17.873602 reg_l2 10.788428
loss 3.6113138
STEP 917 ================================
prereg loss 0.036518943 reg_l1 17.869066 reg_l2 10.787031
loss 3.6103323
STEP 918 ================================
prereg loss 0.03640377 reg_l1 17.864788 reg_l2 10.78574
loss 3.6093614
STEP 919 ================================
prereg loss 0.036323175 reg_l1 17.859478 reg_l2 10.784354
loss 3.6082187
STEP 920 ================================
prereg loss 0.03623579 reg_l1 17.854866 reg_l2 10.783057
loss 3.607209
STEP 921 ================================
prereg loss 0.036142588 reg_l1 17.85092 reg_l2 10.781798
loss 3.6063266
STEP 922 ================================
prereg loss 0.036059167 reg_l1 17.846199 reg_l2 10.780392
loss 3.605299
STEP 923 ================================
prereg loss 0.03599274 reg_l1 17.841026 reg_l2 10.779054
loss 3.604198
STEP 924 ================================
prereg loss 0.03591212 reg_l1 17.836596 reg_l2 10.777801
loss 3.6032312
STEP 925 ================================
prereg loss 0.035882194 reg_l1 17.831453 reg_l2 10.776473
loss 3.6021729
STEP 926 ================================
prereg loss 0.035828635 reg_l1 17.82613 reg_l2 10.775204
loss 3.6010547
STEP 927 ================================
prereg loss 0.035764687 reg_l1 17.822037 reg_l2 10.773956
loss 3.600172
STEP 928 ================================
prereg loss 0.035678487 reg_l1 17.817163 reg_l2 10.772607
loss 3.599111
STEP 929 ================================
prereg loss 0.035559695 reg_l1 17.813084 reg_l2 10.771382
loss 3.5981765
STEP 930 ================================
prereg loss 0.035465047 reg_l1 17.80818 reg_l2 10.770195
loss 3.597101
STEP 931 ================================
prereg loss 0.03539343 reg_l1 17.803057 reg_l2 10.768929
loss 3.596005
STEP 932 ================================
prereg loss 0.035272487 reg_l1 17.799574 reg_l2 10.76774
loss 3.5951874
STEP 933 ================================
prereg loss 0.035188325 reg_l1 17.79537 reg_l2 10.7665205
loss 3.5942626
STEP 934 ================================
prereg loss 0.035125148 reg_l1 17.790388 reg_l2 10.765169
loss 3.5932028
STEP 935 ================================
prereg loss 0.035061695 reg_l1 17.785866 reg_l2 10.763892
loss 3.5922348
STEP 936 ================================
prereg loss 0.03499196 reg_l1 17.78111 reg_l2 10.762526
loss 3.5912142
STEP 937 ================================
prereg loss 0.03494243 reg_l1 17.77632 reg_l2 10.761175
loss 3.5902064
STEP 938 ================================
prereg loss 0.03485033 reg_l1 17.771807 reg_l2 10.759941
loss 3.5892117
STEP 939 ================================
prereg loss 0.034779355 reg_l1 17.767336 reg_l2 10.758589
loss 3.5882466
STEP 940 ================================
prereg loss 0.034733687 reg_l1 17.763332 reg_l2 10.757256
loss 3.5874002
STEP 941 ================================
prereg loss 0.034677953 reg_l1 17.760408 reg_l2 10.756
loss 3.5867598
STEP 942 ================================
prereg loss 0.03458137 reg_l1 17.7561 reg_l2 10.754654
loss 3.5858014
STEP 943 ================================
prereg loss 0.03450667 reg_l1 17.751175 reg_l2 10.753334
loss 3.5847416
STEP 944 ================================
prereg loss 0.034414586 reg_l1 17.746897 reg_l2 10.752116
loss 3.5837939
STEP 945 ================================
prereg loss 0.034382306 reg_l1 17.743057 reg_l2 10.750812
loss 3.5829937
STEP 946 ================================
prereg loss 0.034372058 reg_l1 17.738441 reg_l2 10.749508
loss 3.5820603
STEP 947 ================================
prereg loss 0.03432397 reg_l1 17.733374 reg_l2 10.74832
loss 3.5809987
STEP 948 ================================
prereg loss 0.034291476 reg_l1 17.728722 reg_l2 10.746905
loss 3.580036
STEP 949 ================================
prereg loss 0.03420834 reg_l1 17.7248 reg_l2 10.745671
loss 3.5791683
STEP 950 ================================
prereg loss 0.034142308 reg_l1 17.721062 reg_l2 10.744481
loss 3.5783546
STEP 951 ================================
prereg loss 0.034092277 reg_l1 17.716398 reg_l2 10.743071
loss 3.5773718
STEP 952 ================================
prereg loss 0.033974297 reg_l1 17.712315 reg_l2 10.741858
loss 3.5764375
STEP 953 ================================
prereg loss 0.033931904 reg_l1 17.708239 reg_l2 10.740662
loss 3.5755796
STEP 954 ================================
prereg loss 0.03387712 reg_l1 17.702991 reg_l2 10.739247
loss 3.5744755
STEP 955 ================================
prereg loss 0.033806566 reg_l1 17.698132 reg_l2 10.737887
loss 3.573433
STEP 956 ================================
prereg loss 0.033748005 reg_l1 17.694824 reg_l2 10.736511
loss 3.572713
STEP 957 ================================
prereg loss 0.033740602 reg_l1 17.69121 reg_l2 10.735098
loss 3.5719826
STEP 958 ================================
prereg loss 0.03369349 reg_l1 17.686405 reg_l2 10.733782
loss 3.5709746
STEP 959 ================================
prereg loss 0.033659503 reg_l1 17.681314 reg_l2 10.732427
loss 3.5699224
STEP 960 ================================
prereg loss 0.03363162 reg_l1 17.677816 reg_l2 10.7310095
loss 3.5691948
STEP 961 ================================
prereg loss 0.03358861 reg_l1 17.673433 reg_l2 10.729668
loss 3.5682755
STEP 962 ================================
prereg loss 0.033523135 reg_l1 17.669708 reg_l2 10.728387
loss 3.5674648
STEP 963 ================================
prereg loss 0.03344758 reg_l1 17.66545 reg_l2 10.727089
loss 3.5665374
STEP 964 ================================
prereg loss 0.033393074 reg_l1 17.66113 reg_l2 10.725808
loss 3.5656195
STEP 965 ================================
prereg loss 0.033329602 reg_l1 17.656685 reg_l2 10.724585
loss 3.5646665
STEP 966 ================================
prereg loss 0.033291575 reg_l1 17.651802 reg_l2 10.723263
loss 3.563652
STEP 967 ================================
prereg loss 0.033261884 reg_l1 17.646935 reg_l2 10.722022
loss 3.5626488
STEP 968 ================================
prereg loss 0.03319034 reg_l1 17.642187 reg_l2 10.720881
loss 3.5616276
STEP 969 ================================
prereg loss 0.033137057 reg_l1 17.638947 reg_l2 10.719612
loss 3.5609264
STEP 970 ================================
prereg loss 0.033080056 reg_l1 17.634892 reg_l2 10.718351
loss 3.5600584
STEP 971 ================================
prereg loss 0.03302505 reg_l1 17.631052 reg_l2 10.717093
loss 3.5592356
STEP 972 ================================
prereg loss 0.032989353 reg_l1 17.625877 reg_l2 10.715717
loss 3.5581648
STEP 973 ================================
prereg loss 0.032930054 reg_l1 17.621439 reg_l2 10.714409
loss 3.557218
STEP 974 ================================
prereg loss 0.032880664 reg_l1 17.61758 reg_l2 10.713053
loss 3.556397
STEP 975 ================================
prereg loss 0.0328797 reg_l1 17.61351 reg_l2 10.711636
loss 3.5555816
STEP 976 ================================
prereg loss 0.032820705 reg_l1 17.609863 reg_l2 10.710387
loss 3.5547934
STEP 977 ================================
prereg loss 0.03278258 reg_l1 17.604628 reg_l2 10.708925
loss 3.553708
STEP 978 ================================
prereg loss 0.03279996 reg_l1 17.600618 reg_l2 10.707503
loss 3.5529237
STEP 979 ================================
prereg loss 0.032773264 reg_l1 17.596695 reg_l2 10.706238
loss 3.5521123
STEP 980 ================================
prereg loss 0.032727826 reg_l1 17.593264 reg_l2 10.704874
loss 3.5513804
STEP 981 ================================
prereg loss 0.0326994 reg_l1 17.589346 reg_l2 10.70354
loss 3.5505686
STEP 982 ================================
prereg loss 0.032648604 reg_l1 17.58421 reg_l2 10.702328
loss 3.5494905
STEP 983 ================================
prereg loss 0.03262838 reg_l1 17.580143 reg_l2 10.7009945
loss 3.548657
STEP 984 ================================
prereg loss 0.03251751 reg_l1 17.576345 reg_l2 10.699857
loss 3.5477865
STEP 985 ================================
prereg loss 0.03246255 reg_l1 17.572527 reg_l2 10.698727
loss 3.546968
STEP 986 ================================
prereg loss 0.032408383 reg_l1 17.568691 reg_l2 10.697443
loss 3.5461469
STEP 987 ================================
prereg loss 0.032305527 reg_l1 17.563736 reg_l2 10.696266
loss 3.5450528
STEP 988 ================================
prereg loss 0.03224621 reg_l1 17.559807 reg_l2 10.694966
loss 3.5442076
STEP 989 ================================
prereg loss 0.032247096 reg_l1 17.555588 reg_l2 10.693605
loss 3.5433648
STEP 990 ================================
prereg loss 0.032216944 reg_l1 17.551428 reg_l2 10.692316
loss 3.5425026
STEP 991 ================================
prereg loss 0.032203488 reg_l1 17.54757 reg_l2 10.691022
loss 3.5417173
STEP 992 ================================
prereg loss 0.032189056 reg_l1 17.543488 reg_l2 10.689598
loss 3.5408866
STEP 993 ================================
prereg loss 0.032146417 reg_l1 17.539137 reg_l2 10.688307
loss 3.539974
STEP 994 ================================
prereg loss 0.032116286 reg_l1 17.534441 reg_l2 10.687026
loss 3.5390043
STEP 995 ================================
prereg loss 0.032111708 reg_l1 17.530401 reg_l2 10.685672
loss 3.538192
STEP 996 ================================
prereg loss 0.03202389 reg_l1 17.52723 reg_l2 10.684471
loss 3.5374699
STEP 997 ================================
prereg loss 0.031977296 reg_l1 17.523117 reg_l2 10.683233
loss 3.5366006
STEP 998 ================================
prereg loss 0.031995412 reg_l1 17.518476 reg_l2 10.6818285
loss 3.5356905
STEP 999 ================================
prereg loss 0.031878356 reg_l1 17.514816 reg_l2 10.680647
loss 3.5348418
STEP 1000 ================================
prereg loss 0.031807408 reg_l1 17.509747 reg_l2 10.679257
loss 3.5337567
2022-06-27T04:31:13.002

julia> serialize("sparse16-after-2500-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse16-after-2500-steps-opt.ser", opt)

julia> open("sparse16-after-2500-steps-matrix.json", "w") do f
         JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end

julia> close(io)
```

This is good enough, we've even pushed this training further
in terms of loss than for the non-sparse version.

The `sparse16-after-2500-steps` is becoming the new baseline
and the new starting point.
