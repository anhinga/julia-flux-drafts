# Verifying refactor

Reproducing first 16 steps of [../rough-sketches/run-3](../rough-sketches/run-3)

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1")

julia> include("prepare.jl")
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> steps!(16)
2022-06-10T13:08:08.056
STEP 1 ================================
prereg loss 2.4201107 regularization 3279.3413 reg_novel 1046.6927
loss 6.7461452
STEP 2 ================================
prereg loss 1.9696658 regularization 3275.0347 reg_novel 967.8846
loss 6.2125854
STEP 3 ================================
prereg loss 1.6089424 regularization 3270.124 reg_novel 894.13464
loss 5.7732015
STEP 4 ================================
prereg loss 1.3480141 regularization 3264.6677 reg_novel 825.1346
loss 5.4378166
STEP 5 ================================
prereg loss 1.1890152 regularization 3258.814 reg_novel 760.6182
loss 5.2084475
STEP 6 ================================
prereg loss 1.111058 regularization 3252.5957 reg_novel 700.3695
loss 5.0640235
STEP 7 ================================
prereg loss 1.1007833 regularization 3246.0486 reg_novel 644.1925
loss 4.9910245
STEP 8 ================================
prereg loss 1.1458588 regularization 3239.1934 reg_novel 591.929
loss 4.976981
STEP 9 ================================
prereg loss 1.2170291 regularization 3232.0234 reg_novel 543.43097
loss 4.9924836
STEP 10 ================================
prereg loss 1.281207 regularization 3224.5703 reg_novel 498.5328
loss 5.00431
STEP 11 ================================
prereg loss 1.3243695 regularization 3216.8616 reg_novel 457.03268
loss 4.998264
STEP 12 ================================
prereg loss 1.3417207 regularization 3208.9502 reg_novel 418.72858
loss 4.9693995
STEP 13 ================================
prereg loss 1.3329804 regularization 3200.8062 reg_novel 383.41687
loss 4.917204
STEP 14 ================================
prereg loss 1.3121157 regularization 3192.4255 reg_novel 350.8905
loss 4.855432
STEP 15 ================================
prereg loss 1.2646813 regularization 3183.8496 reg_novel 320.96182
loss 4.769493
STEP 16 ================================
prereg loss 1.2151241 regularization 3175.061 reg_novel 293.4462
loss 4.683632
2022-06-10T13:31:10.271
```
