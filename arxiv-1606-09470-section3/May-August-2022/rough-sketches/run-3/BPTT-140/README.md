# The BPTT-140 series of experiments with run-3 regularization

A rather troubled series of runs.

First I was looking at a right regularization value and bumping
into areas where "dot" neurons were giving outputs which were too high.

Eventually I ended up capping "dot" at 4.0 instead of 100.0
(but I should really replace it with something similar to
`x, y => dot(softmax(x), y)`)

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
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
2588

julia> steps!(1)
2022-06-06T14:14:32.080
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 423.65842
2022-06-06T14:15:49.545

julia> steps!(15)
2022-06-06T14:16:04.890
STEP 1 ================================
prereg loss 427.6269 regularization 321.46832 reg_novel 28.091711
loss 427.97644
STEP 2 ================================
prereg loss 430.31848 regularization 320.299 reg_novel 28.090366
loss 430.66687
STEP 3 ================================
prereg loss 427.51968 regularization 319.11453 reg_novel 28.104265
loss 427.8669
STEP 4 ================================
prereg loss 412.36584 regularization 317.90036 reg_novel 28.105253
loss 412.71185
STEP 5 ================================
prereg loss 425.85406 regularization 316.6488 reg_novel 28.100288
loss 426.19882
STEP 6 ================================
ERROR: InterruptException:

[...]

julia> # Let's bring regylarization 100 fold up (although previous BPTT-140 suggests 1000 fold)

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
2588

julia> steps!(16)
2022-06-06T14:22:00.779
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 455.59406
STEP 2 ================================
prereg loss 424.11948 regularization 320.83206 reg_novel 28.828873
loss 456.2315
STEP 3 ================================
prereg loss 432.18283 regularization 319.02277 reg_novel 29.533377
loss 464.11465
STEP 4 ================================
prereg loss 434.09607 regularization 317.24084 reg_novel 30.259354
loss 465.8504
STEP 5 ================================
prereg loss 413.78412 regularization 315.45245 reg_novel 30.994705
loss 445.36035
STEP 6 ================================
prereg loss 417.45145 regularization 313.66803 reg_novel 31.746508
loss 448.85
STEP 7 ================================
prereg loss 419.24222 regularization 311.87943 reg_novel 32.509987
loss 450.46268
STEP 8 ================================
prereg loss 404.23868 regularization 310.06683 reg_novel 33.265957
loss 435.27863
STEP 9 ================================
prereg loss 371.35977 regularization 308.2686 reg_novel 34.04556
loss 402.22067
STEP 10 ================================
prereg loss 370.35693 regularization 306.48068 reg_novel 34.82916
loss 401.03983
STEP 11 ================================
prereg loss 367.06894 regularization 304.6786 reg_novel 35.61834
loss 397.57242
STEP 12 ================================
prereg loss 330.875 regularization 302.90033 reg_novel 36.42571
loss 361.20145
STEP 13 ================================
prereg loss 1.4990321e6 regularization 301.1426 reg_novel 37.241123
loss 1.4990622e6
STEP 14 ================================
prereg loss 384.59732 regularization 299.35632 reg_novel 38.03891
loss 414.57098
STEP 15 ================================
prereg loss 370.35944 regularization 297.61115 reg_novel 38.842026
loss 400.1594
STEP 16 ================================
prereg loss 386.3658 regularization 295.89005 reg_novel 39.648766
loss 415.99448
2022-06-06T14:36:38.941

julia> # I can't say that I like this; do I want a stronger regularization, or a weaker one?

julia> # Let's try a weaker one first, making this 10 times weaker

julia> count(sparse)
2588

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
2588

julia> steps!(16)
2022-06-06T14:38:45.026
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 426.56165
STEP 2 ================================
prereg loss 424.1191 regularization 320.83206 reg_novel 28.828873
loss 427.35626
STEP 3 ================================
prereg loss 432.15283 regularization 319.0228 reg_novel 29.533346
loss 435.3726
STEP 4 ================================
prereg loss 434.0765 regularization 317.23712 reg_novel 30.259096
loss 437.27914
STEP 5 ================================
prereg loss 381.11377 regularization 315.44528 reg_novel 30.993464
loss 384.29922
STEP 6 ================================
prereg loss 403.27826 regularization 313.65442 reg_novel 31.734434
loss 406.44653
STEP 7 ================================
prereg loss 396.0335 regularization 311.8626 reg_novel 32.48901
loss 399.18463
STEP 8 ================================
prereg loss 400.11456 regularization 310.0436 reg_novel 33.256645
loss 403.24826
STEP 9 ================================
prereg loss 3.2955945e6 regularization 308.2253 reg_novel 34.035828
loss 3.2955975e6
ERROR: InterruptException:

[...]

julia> # I DON"T LIKE THIS EITHER; LET'S MAKE REG AS STRONG AS IN run-1.1/BPTT-140

julia>

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
2588

julia> steps!(16)
2022-06-06T14:48:14.155
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 745.9182
STEP 2 ================================
prereg loss 424.11877 regularization 320.83206 reg_novel 28.828875
loss 744.9797
STEP 3 ================================
prereg loss 432.19675 regularization 319.02243 reg_novel 29.533363
loss 751.2487
STEP 4 ================================
prereg loss 434.06088 regularization 317.23923 reg_novel 30.259314
loss 751.3304
STEP 5 ================================
prereg loss 417.31372 regularization 315.44208 reg_novel 30.991697
loss 732.7868
STEP 6 ================================
prereg loss 421.26874 regularization 313.6047 reg_novel 31.716137
loss 734.90515
STEP 7 ================================
prereg loss 425.4768 regularization 311.77887 reg_novel 32.449444
loss 737.2881
STEP 8 ================================
prereg loss 416.02585 regularization 309.9546 reg_novel 33.197784
loss 726.0137
STEP 9 ================================
prereg loss 410.81952 regularization 308.13055 reg_novel 33.95942
loss 718.984
STEP 10 ================================
prereg loss 402.65152 regularization 306.3086 reg_novel 34.730858
loss 708.9949
STEP 11 ================================
prereg loss 401.52338 regularization 304.48648 reg_novel 35.51621
loss 706.0454
STEP 12 ================================
prereg loss 386.59796 regularization 302.68994 reg_novel 36.3141
loss 689.3242
STEP 13 ================================
prereg loss 390.9645 regularization 300.92014 reg_novel 37.124836
loss 691.92175
STEP 14 ================================
prereg loss 392.31677 regularization 299.17407 reg_novel 37.94413
loss 691.5288
STEP 15 ================================
prereg loss 345.5391 regularization 297.4661 reg_novel 38.767956
loss 643.04395
STEP 16 ================================
prereg loss 338.1652 regularization 295.79196 reg_novel 39.614582
loss 633.99677
2022-06-06T15:03:26.735

julia> steps!(100)
2022-06-06T15:04:41.022
STEP 1 ================================
prereg loss 319.2743 regularization 294.13284 reg_novel 40.46291
loss 613.44763
STEP 2 ================================
prereg loss 315.34033 regularization 292.49103 reg_novel 41.30953
loss 607.8727
STEP 3 ================================
prereg loss 298.01895 regularization 290.85568 reg_novel 42.151283
loss 588.91675
STEP 4 ================================
prereg loss 1.0395232e6 regularization 289.21637 reg_novel 42.979694
loss 1.03981244e6
STEP 5 ================================
prereg loss 2.7192082e6 regularization 287.5705 reg_novel 43.78019
loss 2.7194958e6
STEP 6 ================================
prereg loss 2.5782162e6 regularization 285.9576 reg_novel 44.605213
loss 2.5785022e6
STEP 7 ================================
prereg loss 894484.44 regularization 284.35626 reg_novel 45.42717
loss 894768.8
STEP 8 ================================
prereg loss 122906.836 regularization 282.7724 reg_novel 46.24439
loss 123189.66
ERROR: InterruptException:

[...]

julia> # LET'S REDO THE ORIGINAL, WEAK REGULARIZARION, if not, we'll try a non-sparse version

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> steps!(16)
2022-06-06T15:13:53.635
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 423.65842
STEP 2 ================================
prereg loss 427.6269 regularization 321.46832 reg_novel 28.091711
loss 427.97644
STEP 3 ================================
prereg loss 430.31848 regularization 320.299 reg_novel 28.090366
loss 430.66687
STEP 4 ================================
prereg loss 427.51968 regularization 319.11453 reg_novel 28.104265
loss 427.8669
STEP 5 ================================
prereg loss 412.36584 regularization 317.90036 reg_novel 28.105253
loss 412.71185
STEP 6 ================================
prereg loss 425.85406 regularization 316.6488 reg_novel 28.100288
loss 426.19882
STEP 7 ================================
prereg loss 413.1412 regularization 315.4076 reg_novel 28.112255
loss 413.4847
STEP 8 ================================
prereg loss 2.7188005e6 regularization 314.1521 reg_novel 28.145014
loss 2.7188008e6
STEP 9 ================================
prereg loss 411.1945 regularization 312.9356 reg_novel 28.16598
loss 411.53558
STEP 10 ================================
prereg loss 418.21747 regularization 311.71942 reg_novel 28.19939
loss 418.55737
STEP 11 ================================
prereg loss 418.3477 regularization 310.50623 reg_novel 28.241089
loss 418.68643
STEP 12 ================================
prereg loss 411.08652 regularization 309.3048 reg_novel 28.28656
loss 411.4241
STEP 13 ================================
prereg loss 419.302 regularization 308.12195 reg_novel 28.33066
loss 419.63846
STEP 14 ================================
prereg loss 417.1363 regularization 306.9603 reg_novel 28.368498
loss 417.47162
STEP 15 ================================
prereg loss 414.11496 regularization 305.8269 reg_novel 28.398483
loss 414.4492
STEP 16 ================================
prereg loss 416.66547 regularization 304.72128 reg_novel 28.41645
loss 416.9986
2022-06-06T15:28:51.519
```

Now I am trying a non-sparse BPTT version and seeing a huge slowdown,
because the system is bumping into memory limits and garbage collecting all the time.

So instead of under 10 min per iteration, we are having over 30 min per iteration.

That prompted my failed inquiry earlier this evening trying to see if Enzyme is
read to handle this (it is not, see `enzyme-tests`).

Thinking about exploring a JAX solution; this might be a good enough reason to explore
porting this to JAX.

```

julia> # non-sparse version next, if not we can do that with the original initialization

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to endpoint of the previous run, is ready to train, use 'steps!(N)'
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

julia> count(non_sparse)
20308

julia> steps!(16)
2022-06-06T15:32:36.428
STEP 1 ================================
prereg loss 428.78357 regularization 332.0412 reg_novel 19.78745
loss 429.1354
STEP 2 ================================
prereg loss 410.71783 regularization 334.944 reg_novel 32.225925
loss 411.085
STEP 3 ================================
prereg loss 401.8305 regularization 330.3983 reg_novel 22.736206
loss 402.18365
STEP 4 ================================
prereg loss 404.7552 regularization 334.7599 reg_novel 15.30199
loss 405.10526
STEP 5 ================================
prereg loss 402.81564 regularization 339.12015 reg_novel 11.461526
loss 403.16623
STEP 6 ================================
prereg loss 372.93668 regularization 340.8107 reg_novel 9.8923235
loss 373.28738
STEP 7 ================================
prereg loss 390.96182 regularization 340.12097 reg_novel 9.75109
loss 391.3117
STEP 8 ================================
prereg loss 391.40976 regularization 337.74457 reg_novel 10.577131
loss 391.7581
STEP 9 ================================
prereg loss 388.38354 regularization 334.5175 reg_novel 12.019366
loss 388.73007
STEP 10 ================================
prereg loss 376.58328 regularization 330.82028 reg_novel 13.567652
loss 376.92767
STEP 11 ================================
prereg loss 397.73224 regularization 327.03265 reg_novel 15.130631
loss 398.0744
STEP 12 ================================
prereg loss 379.46106 regularization 323.5869 reg_novel 16.740952
loss 379.8014
STEP 13 ================================
prereg loss 363.88922 regularization 320.2539 reg_novel 18.382223
loss 364.22784
STEP 14 ================================
prereg loss 358.91388 regularization 317.13507 reg_novel 19.963652
loss 359.25098
STEP 15 ================================
prereg loss 368.69666 regularization 314.44977 reg_novel 21.448103
loss 369.03256
STEP 16 ================================
prereg loss 360.5386 regularization 312.14606 reg_novel 22.546343
loss 360.8733
2022-06-07T00:47:28.647
```

Now finally capping `dot` output to 4.0, and going back to sparse experiments.

```
julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> count(sparse)
2588

julia> steps!(16)
2022-06-07T02:22:18.590
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 423.65842
STEP 2 ================================
prereg loss 427.6269 regularization 321.46832 reg_novel 28.091711
loss 427.97644
STEP 3 ================================
prereg loss 430.31848 regularization 320.299 reg_novel 28.090366
loss 430.66687
STEP 4 ================================
prereg loss 427.51968 regularization 319.11453 reg_novel 28.104265
loss 427.8669
STEP 5 ================================
prereg loss 412.36584 regularization 317.90036 reg_novel 28.105253
loss 412.71185
STEP 6 ================================
prereg loss 425.85406 regularization 316.6488 reg_novel 28.100288
loss 426.19882
STEP 7 ================================
prereg loss 413.1412 regularization 315.4076 reg_novel 28.112255
loss 413.4847
STEP 8 ================================
prereg loss 1961.3588 regularization 314.1521 reg_novel 28.145014
loss 1961.701
STEP 9 ================================
prereg loss 411.70404 regularization 312.9391 reg_novel 28.171673
loss 412.04517
STEP 10 ================================
prereg loss 416.13055 regularization 311.7256 reg_novel 28.20892
loss 416.4705
STEP 11 ================================
prereg loss 414.51556 regularization 310.51608 reg_novel 28.252293
loss 414.85434
STEP 12 ================================
prereg loss 404.97467 regularization 309.31952 reg_novel 28.297043
loss 405.3123
STEP 13 ================================
prereg loss 410.546 regularization 308.1426 reg_novel 28.33778
loss 410.88248
STEP 14 ================================
prereg loss 404.6591 regularization 306.9884 reg_novel 28.370085
loss 404.99445
STEP 15 ================================
prereg loss 404.3888 regularization 305.8638 reg_novel 28.392881
loss 404.72305
STEP 16 ================================
prereg loss 397.6868 regularization 304.7679 reg_novel 28.402445
loss 398.01996
2022-06-07T02:39:01.768

julia>
```

Continuing:

```
julia> # NO WE WANTED STRONG REGULARIZATION HERE, 1000 FOLD of the base run

julia>

julia> # and plenty of bugs in the experiments above, because we did this

julia> #     l += 1.0f0 * regularization + 0.001f0 * reg_novel

julia> # while we really meant to keep these two coefs in sync

julia>

julia> # still doing this at this particular run, but might decide to change

julia> # this later

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> steps!(16)
2022-06-07T02:56:14.718
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 745.9182
STEP 2 ================================
prereg loss 424.11877 regularization 320.83206 reg_novel 28.828875
loss 744.9797
STEP 3 ================================
prereg loss 432.19675 regularization 319.02243 reg_novel 29.533363
loss 751.2487
STEP 4 ================================
prereg loss 434.06088 regularization 317.23923 reg_novel 30.259314
loss 751.3304
STEP 5 ================================
prereg loss 417.31372 regularization 315.44208 reg_novel 30.991697
loss 732.7868
STEP 6 ================================
prereg loss 421.26874 regularization 313.6047 reg_novel 31.716137
loss 734.90515
STEP 7 ================================
prereg loss 425.4768 regularization 311.77887 reg_novel 32.449444
loss 737.2881
STEP 8 ================================
prereg loss 416.02585 regularization 309.9546 reg_novel 33.197784
loss 726.0137
STEP 9 ================================
prereg loss 410.81952 regularization 308.13055 reg_novel 33.95942
loss 718.984
STEP 10 ================================
prereg loss 402.65152 regularization 306.3086 reg_novel 34.730858
loss 708.9949
STEP 11 ================================
prereg loss 401.52338 regularization 304.48648 reg_novel 35.51621
loss 706.0454
STEP 12 ================================
prereg loss 386.59796 regularization 302.68994 reg_novel 36.3141
loss 689.3242
STEP 13 ================================
prereg loss 390.9645 regularization 300.92014 reg_novel 37.124836
loss 691.92175
STEP 14 ================================
prereg loss 392.31677 regularization 299.17407 reg_novel 37.94413
loss 691.5288
STEP 15 ================================
prereg loss 345.5391 regularization 297.4661 reg_novel 38.767956
loss 643.04395
STEP 16 ================================
prereg loss 338.1652 regularization 295.79196 reg_novel 39.614582
loss 633.99677
2022-06-07T03:11:40.392

julia> steps!(100)
2022-06-07T03:11:46.404
STEP 1 ================================
prereg loss 319.2743 regularization 294.13284 reg_novel 40.46291
loss 613.44763
STEP 2 ================================
prereg loss 315.34033 regularization 292.49103 reg_novel 41.30953
loss 607.8727
STEP 3 ================================
prereg loss 298.01895 regularization 290.85568 reg_novel 42.151283
loss 588.91675
STEP 4 ================================
prereg loss 878.3498 regularization 289.21637 reg_novel 42.979694
loss 1167.6091
STEP 5 ================================
prereg loss 921.1369 regularization 287.55817 reg_novel 43.793938
loss 1208.7389
STEP 6 ================================
prereg loss 1163.6638 regularization 285.85757 reg_novel 44.57175
loss 1449.5659
STEP 7 ================================
prereg loss 305.9798 regularization 284.175 reg_novel 45.340427
loss 590.20013
STEP 8 ================================
prereg loss 369.34155 regularization 282.5187 reg_novel 46.108486
loss 651.9064
STEP 9 ================================
prereg loss 374.75992 regularization 280.88693 reg_novel 46.877075
loss 655.6937
STEP 10 ================================
prereg loss 364.50424 regularization 279.2795 reg_novel 47.64459
loss 643.8314
STEP 11 ================================
prereg loss 375.36505 regularization 277.6979 reg_novel 48.414417
loss 653.1113
STEP 12 ================================
prereg loss 374.88507 regularization 276.1448 reg_novel 49.186512
loss 651.0791
STEP 13 ================================
prereg loss 375.20496 regularization 274.62476 reg_novel 49.962315
loss 649.87964
STEP 14 ================================
prereg loss 373.06454 regularization 273.1226 reg_novel 50.740913
loss 646.2379
STEP 15 ================================
prereg loss 377.36493 regularization 271.63736 reg_novel 51.52447
loss 649.05383
STEP 16 ================================
prereg loss 366.3551 regularization 270.16675 reg_novel 52.31185
loss 636.57416
STEP 17 ================================
prereg loss 371.85782 regularization 268.7143 reg_novel 53.10393
loss 640.62524
STEP 18 ================================
prereg loss 373.5169 regularization 267.25983 reg_novel 53.896854
loss 640.8306
STEP 19 ================================
prereg loss 367.46982 regularization 265.81287 reg_novel 54.6914
loss 633.3374
STEP 20 ================================
prereg loss 369.156 regularization 264.3706 reg_novel 55.48628
loss 633.5821
STEP 21 ================================
prereg loss 369.05145 regularization 262.9413 reg_novel 56.281242
loss 632.0491
STEP 22 ================================
prereg loss 370.44614 regularization 261.54507 reg_novel 57.07754
loss 632.0483
STEP 23 ================================
prereg loss 367.6914 regularization 260.1688 reg_novel 57.874546
loss 627.9181
STEP 24 ================================
prereg loss 366.3039 regularization 258.81305 reg_novel 58.669006
loss 625.1756
STEP 25 ================================
prereg loss 366.22955 regularization 257.47638 reg_novel 59.459007
loss 623.7654
STEP 26 ================================
prereg loss 367.06723 regularization 256.1509 reg_novel 60.24148
loss 623.2784
STEP 27 ================================
prereg loss 364.1078 regularization 254.83475 reg_novel 61.016598
loss 619.00354
STEP 28 ================================
prereg loss 364.6977 regularization 253.52725 reg_novel 61.789146
loss 618.28674
STEP 29 ================================
prereg loss 361.11514 regularization 252.22623 reg_novel 62.559666
loss 613.40393
STEP 30 ================================
prereg loss 358.63104 regularization 250.93216 reg_novel 63.327656
loss 609.6265
STEP 31 ================================
prereg loss 345.1297 regularization 249.65103 reg_novel 64.095436
loss 594.84485
STEP 32 ================================
prereg loss 283.28094 regularization 248.3786 reg_novel 64.86689
loss 531.7244
STEP 33 ================================
prereg loss 274.38812 regularization 247.11397 reg_novel 65.63994
loss 521.56775
STEP 34 ================================
prereg loss 269.01196 regularization 245.85959 reg_novel 66.41544
loss 514.938
STEP 35 ================================
prereg loss 264.987 regularization 244.62117 reg_novel 67.19552
loss 509.67535
STEP 36 ================================
prereg loss 242.00273 regularization 243.3995 reg_novel 67.974495
loss 485.4702
STEP 37 ================================
prereg loss 259.80124 regularization 242.2014 reg_novel 68.74517
loss 502.07138
STEP 38 ================================
prereg loss 312.8375 regularization 241.02643 reg_novel 69.50724
loss 553.9334
STEP 39 ================================
prereg loss 210.71956 regularization 239.86331 reg_novel 70.26172
loss 450.65314
STEP 40 ================================
prereg loss 883.3624 regularization 238.71387 reg_novel 71.010735
loss 1122.1473
STEP 41 ================================
prereg loss 283.54422 regularization 237.5701 reg_novel 71.7517
loss 521.18604
STEP 42 ================================
prereg loss 341.49228 regularization 236.44095 reg_novel 72.49498
loss 578.00574
STEP 43 ================================
prereg loss 947.4356 regularization 235.3279 reg_novel 73.236336
loss 1182.8368
STEP 44 ================================
prereg loss 2562.437 regularization 234.19243 reg_novel 73.98109
loss 2796.7034
STEP 45 ================================
prereg loss 1969.2858 regularization 233.05672 reg_novel 74.752075
loss 2202.4172
STEP 46 ================================
prereg loss 2733.757 regularization 231.93617 reg_novel 75.50153
loss 2965.7688
STEP 47 ================================
prereg loss 1469.8374 regularization 230.81946 reg_novel 76.24396
loss 1700.7332
STEP 48 ================================
prereg loss 347.13596 regularization 229.7168 reg_novel 76.97555
loss 576.92975
STEP 49 ================================
prereg loss 348.78326 regularization 228.63124 reg_novel 77.70075
loss 577.4922
STEP 50 ================================
prereg loss 349.49878 regularization 227.55592 reg_novel 78.42049
loss 577.1331
STEP 51 ================================
prereg loss 347.40756 regularization 226.48703 reg_novel 79.14245
loss 573.97375
STEP 52 ================================
prereg loss 347.6982 regularization 225.42477 reg_novel 79.86607
loss 573.2029
STEP 53 ================================
prereg loss 345.62793 regularization 224.37314 reg_novel 80.58941
loss 570.08167
STEP 54 ================================
prereg loss 345.65527 regularization 223.3347 reg_novel 81.31177
loss 569.0713
STEP 55 ================================
prereg loss 343.76285 regularization 222.31052 reg_novel 82.03652
loss 566.1554
STEP 56 ================================
prereg loss 341.5178 regularization 221.28656 reg_novel 82.76241
loss 562.8871
STEP 57 ================================
prereg loss 338.8303 regularization 220.28629 reg_novel 83.49087
loss 559.2001
STEP 58 ================================
prereg loss 336.5436 regularization 219.30798 reg_novel 84.22114
loss 555.9358
STEP 59 ================================
prereg loss 336.1386 regularization 218.34119 reg_novel 84.94755
loss 554.56476
STEP 60 ================================
prereg loss 340.91174 regularization 217.38533 reg_novel 85.66829
loss 558.38275
STEP 61 ================================
prereg loss 333.65747 regularization 216.43666 reg_novel 86.377556
loss 550.18054
STEP 62 ================================
prereg loss 338.40076 regularization 215.49245 reg_novel 87.08071
loss 553.9803
STEP 63 ================================
prereg loss 333.51013 regularization 214.57071 reg_novel 87.778725
loss 548.16864
STEP 64 ================================
prereg loss 334.5579 regularization 213.65533 reg_novel 88.47038
loss 548.3017
STEP 65 ================================
prereg loss 328.75232 regularization 212.74982 reg_novel 89.1563
loss 541.5913
STEP 66 ================================
prereg loss 318.60568 regularization 211.85008 reg_novel 89.83591
loss 530.5456
STEP 67 ================================
prereg loss 304.97336 regularization 210.95496 reg_novel 90.5102
loss 516.0188
STEP 68 ================================
prereg loss 327.66357 regularization 210.07887 reg_novel 91.17913
loss 537.8336
STEP 69 ================================
prereg loss 328.44467 regularization 209.21608 reg_novel 91.84117
loss 537.75256
STEP 70 ================================
prereg loss 334.10965 regularization 208.35164 reg_novel 92.49625
loss 542.5538
STEP 71 ================================
prereg loss 324.75293 regularization 207.4931 reg_novel 93.136696
loss 532.3392
STEP 72 ================================
prereg loss 312.5121 regularization 206.63643 reg_novel 93.763176
loss 519.2423
STEP 73 ================================
prereg loss 316.2528 regularization 205.79536 reg_novel 94.386024
loss 522.1426
STEP 74 ================================
prereg loss 328.02222 regularization 204.96667 reg_novel 95.00447
loss 533.08386
STEP 75 ================================
prereg loss 326.63055 regularization 204.15933 reg_novel 95.62618
loss 530.8855
STEP 76 ================================
prereg loss 305.00598 regularization 203.3555 reg_novel 96.25386
loss 508.45773
STEP 77 ================================
prereg loss 320.9068 regularization 202.55197 reg_novel 96.8761
loss 523.55566
STEP 78 ================================
prereg loss 319.72766 regularization 201.7655 reg_novel 97.48974
loss 521.59064
STEP 79 ================================
prereg loss 313.81732 regularization 200.98836 reg_novel 98.09561
loss 514.9038
STEP 80 ================================
prereg loss 310.0234 regularization 200.21237 reg_novel 98.695305
loss 510.33447
STEP 81 ================================
prereg loss 310.52206 regularization 199.44731 reg_novel 99.29003
loss 510.06866
STEP 82 ================================
prereg loss 313.5647 regularization 198.69646 reg_novel 99.87165
loss 512.361
STEP 83 ================================
prereg loss 313.7414 regularization 197.95393 reg_novel 100.44068
loss 511.79578
STEP 84 ================================
prereg loss 316.6863 regularization 197.21898 reg_novel 101.00385
loss 514.0063
STEP 85 ================================
prereg loss 307.18002 regularization 196.49068 reg_novel 101.56004
loss 503.77228
STEP 86 ================================
prereg loss 315.66312 regularization 195.75558 reg_novel 102.11195
loss 511.5208
STEP 87 ================================
prereg loss 302.00647 regularization 195.0228 reg_novel 102.663536
loss 497.13193
STEP 88 ================================
prereg loss 300.65036 regularization 194.29341 reg_novel 103.216354
loss 495.047
STEP 89 ================================
prereg loss 312.15518 regularization 193.58238 reg_novel 103.76922
loss 505.84134
STEP 90 ================================
prereg loss 312.43124 regularization 192.89636 reg_novel 104.3211
loss 505.43195
STEP 91 ================================
prereg loss 297.2506 regularization 192.2264 reg_novel 104.86655
loss 489.58188
STEP 92 ================================
prereg loss 298.02954 regularization 191.55664 reg_novel 105.40567
loss 489.6916
STEP 93 ================================
prereg loss 298.31418 regularization 190.88939 reg_novel 105.94348
loss 489.3095
STEP 94 ================================
prereg loss 297.2604 regularization 190.21957 reg_novel 106.4762
loss 487.58646
STEP 95 ================================
prereg loss 295.48117 regularization 189.56108 reg_novel 107.00068
loss 485.14923
STEP 96 ================================
prereg loss 303.76868 regularization 188.92386 reg_novel 107.520065
loss 492.80005
STEP 97 ================================
prereg loss 279.6875 regularization 188.2863 reg_novel 108.04283
loss 468.08185
STEP 98 ================================
prereg loss 274.2028 regularization 187.64517 reg_novel 108.56612
loss 461.95654
STEP 99 ================================
prereg loss 279.16132 regularization 187.01454 reg_novel 109.08281
loss 466.28494
STEP 100 ================================
prereg loss 295.38712 regularization 186.3926 reg_novel 109.59338
loss 481.88928
2022-06-07T04:51:41.394
```

Doing package update because of the Enzyme work I am doing in parallel.

```
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

julia> # want to update Enzyme

julia> # running "update Enzyme"

(@v1.7) pkg> status
      Status `C:\Users\anhin\.julia\environments\v1.7\Project.toml`
  [31c24e10] Distributions v0.25.61
  [7da242da] Enzyme v0.10.0
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

julia> # actually, the recompilation is so substantial, that it's easier to run "update"

julia> # which does update Distributions, ReverseDiff, and JLD

(@v1.7) pkg> status
      Status `C:\Users\anhin\.julia\environments\v1.7\Project.toml`
  [31c24e10] Distributions v0.25.62
  [7da242da] Enzyme v0.10.0
  [587475ba] Flux v0.13.3
  [de31a74c] FunctionalCollections v0.5.0
  [f67ccb44] HDF5 v0.16.9
  [7073ff75] IJulia v1.23.3
  [86fae568] ImageView v0.11.1
  [916415d5] Images v0.25.2
  [4138dd39] JLD v0.13.2
  [033835bb] JLD2 v0.4.22
  [0f8b85d8] JSON3 v1.9.5
  [37e2e3b7] ReverseDiff v1.14.1
  [5e47fb64] TestImages v1.7.0
  [e88e6eb3] Zygote v0.6.40
  [37e2e46d] LinearAlgebra

(@v1.7) pkg>
```

And finally:

```
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.3 (2022-05-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> # l += 0.1f0 * regularization + 0.1f0 * reg_novel

julia> steps!(16)
2022-06-07T08:53:10.704
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 458.3755
STEP 2 ================================
prereg loss 427.627 regularization 321.46838 reg_novel 28.091665
loss 462.583
STEP 3 ================================
prereg loss 430.3181 regularization 320.2989 reg_novel 28.090267
loss 465.15704
STEP 4 ================================
prereg loss 427.26254 regularization 319.11417 reg_novel 28.104069
loss 461.98438
STEP 5 ================================
prereg loss 421.87628 regularization 317.92584 reg_novel 28.12501
loss 456.48138
STEP 6 ================================
prereg loss 409.08228 regularization 316.73242 reg_novel 28.156563
loss 443.57117
STEP 7 ================================
prereg loss 385.49423 regularization 315.5436 reg_novel 28.205154
loss 419.8691
STEP 8 ================================
prereg loss 375.78195 regularization 314.32462 reg_novel 28.25028
loss 410.03943
STEP 9 ================================
prereg loss 408.61395 regularization 313.12643 reg_novel 28.30521
loss 442.7571
STEP 10 ================================
prereg loss 400.93347 regularization 311.91513 reg_novel 28.35519
loss 434.9605
STEP 11 ================================
prereg loss 392.01227 regularization 310.71143 reg_novel 28.406246
loss 425.92404
STEP 12 ================================
prereg loss 392.05676 regularization 309.52298 reg_novel 28.455702
loss 425.85464
STEP 13 ================================
prereg loss 363.62772 regularization 308.35596 reg_novel 28.500854
loss 397.3134
STEP 14 ================================
prereg loss 401.808 regularization 307.20828 reg_novel 28.538307
loss 435.3827
STEP 15 ================================
prereg loss 2213.4485 regularization 306.08395 reg_novel 28.576927
loss 2246.9146
STEP 16 ================================
prereg loss 3329.4856 regularization 304.99133 reg_novel 28.598448
loss 3362.8445
2022-06-07T09:09:43.347

julia> include("train-v0-0-1.jl")
Computing gradient

TreeADAM included

DEFINED: opt
SKIPPED: adam_step!
The network is set to 'sparse', ready to train, use 'steps!(N)'
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0865232), "norm-5"=>Dict("true"=>0.0195538, "norm"=>…
  "norm-5"    => Dict("dict"=>Dict("accum-1"=>Dict("false"=>0.100292), "accum-4"=>Dict("dict"=>-0.0832906), "dot-2"=>Di…
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
  "accum-2"   => Dict("dict"=>Dict("norm-5"=>Dict("dict-2"=>0.125853), "compare-4"=>Dict("dict"=>-0.0675133, "norm"=>-0…
  "compare-1" => Dict("dict"=>Dict("compare-2"=>Dict("dict"=>0.0473879), "accum-4"=>Dict("dict-2"=>0.138899), "dot-2"=>…
  "dot-4"     => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.0387086), "norm-5"=>Dict("false"=>0.0476152), "accum-1"…
  "dot-5"     => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>0.0739928), "compare-4"=>Dict("dict"=>0.0627812), "norm-2"=>…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.453306), "norm-5"=>Dict("norm"=>-0.022435), "accum-4"=>…

julia> #     l += 1.0f0 * regularization + 1.0f0 * reg_novel

julia> steps!(16)
2022-06-07T09:11:06.063
STEP 1 ================================
prereg loss 423.30774 regularization 322.58237 reg_novel 28.09511
loss 773.9852
STEP 2 ================================
prereg loss 427.6269 regularization 321.46838 reg_novel 28.091667
loss 777.18695
STEP 3 ================================
prereg loss 430.31287 regularization 320.2978 reg_novel 28.089651
loss 778.7003
STEP 4 ================================
prereg loss 425.47488 regularization 319.11206 reg_novel 28.10277
loss 772.6897
STEP 5 ================================
prereg loss 425.14664 regularization 317.93164 reg_novel 28.131193
loss 771.2095
STEP 6 ================================
prereg loss 426.34564 regularization 316.7451 reg_novel 28.17155
loss 771.26227
STEP 7 ================================
prereg loss 398.81183 regularization 315.5805 reg_novel 28.231945
loss 742.62427
STEP 8 ================================
prereg loss 363.2983 regularization 314.4066 reg_novel 28.27854
loss 705.9834
STEP 9 ================================
prereg loss 388.8734 regularization 313.20132 reg_novel 28.315197
loss 730.3899
STEP 10 ================================
prereg loss 348.65402 regularization 311.99722 reg_novel 28.356789
loss 689.00806
STEP 11 ================================
prereg loss 347.4522 regularization 310.7889 reg_novel 28.395248
loss 686.63635
STEP 12 ================================
prereg loss 317.3538 regularization 309.5687 reg_novel 28.419222
loss 655.3417
STEP 13 ================================
prereg loss 370.00558 regularization 308.36487 reg_novel 28.4462
loss 706.81665
STEP 14 ================================
prereg loss 340.4912 regularization 307.24374 reg_novel 28.498781
loss 676.23376
STEP 15 ================================
prereg loss 385.0519 regularization 306.1533 reg_novel 28.540628
loss 719.74585
STEP 16 ================================
prereg loss 352.49988 regularization 305.0873 reg_novel 28.565638
loss 686.15283
2022-06-07T09:27:15.306

julia> steps!(100)
2022-06-07T09:27:51.740
STEP 1 ================================
prereg loss 775.1314 regularization 304.05246 reg_novel 28.57998
loss 1107.7639
STEP 2 ================================
prereg loss 1906.288 regularization 302.97757 reg_novel 28.556194
loss 2237.8218
STEP 3 ================================
prereg loss 320.07983 regularization 301.92477 reg_novel 28.546482
loss 650.5511
STEP 4 ================================
prereg loss 349.55746 regularization 300.87564 reg_novel 28.5381
loss 678.9712
STEP 5 ================================
prereg loss 363.63998 regularization 299.83902 reg_novel 28.532894
loss 692.0119
STEP 6 ================================
prereg loss 333.05743 regularization 298.809 reg_novel 28.531956
loss 660.3984
STEP 7 ================================
prereg loss 364.12573 regularization 297.78668 reg_novel 28.53497
loss 690.4474
STEP 8 ================================
prereg loss 403.6483 regularization 296.76898 reg_novel 28.540401
loss 728.95764
STEP 9 ================================
prereg loss 383.69067 regularization 295.75836 reg_novel 28.551708
loss 708.00073
STEP 10 ================================
prereg loss 1982.1908 regularization 294.7628 reg_novel 28.565763
loss 2305.5193
STEP 11 ================================
prereg loss 339.5839 regularization 293.7827 reg_novel 28.577059
loss 661.94366
STEP 12 ================================
prereg loss 377.6527 regularization 292.82425 reg_novel 28.587082
loss 699.064
STEP 13 ================================
prereg loss 356.81274 regularization 291.88632 reg_novel 28.59619
loss 677.2953
STEP 14 ================================
prereg loss 351.71198 regularization 290.9653 reg_novel 28.605598
loss 671.28284
STEP 15 ================================
prereg loss 311.7363 regularization 290.0577 reg_novel 28.615032
loss 630.40906
STEP 16 ================================
prereg loss 1469.1381 regularization 289.1616 reg_novel 28.623623
loss 1786.9233
STEP 17 ================================
prereg loss 1378.0997 regularization 288.29404 reg_novel 28.612345
loss 1695.0061
STEP 18 ================================
prereg loss 361.89746 regularization 287.446 reg_novel 28.606564
loss 677.9501
STEP 19 ================================
prereg loss 290.33643 regularization 286.60745 reg_novel 28.601189
loss 605.54504
STEP 20 ================================
prereg loss 2172.6162 regularization 285.7736 reg_novel 28.596462
loss 2486.9863
STEP 21 ================================
prereg loss 333.4249 regularization 284.95612 reg_novel 28.591885
loss 646.9729
STEP 22 ================================
prereg loss 365.6792 regularization 284.15012 reg_novel 28.58908
loss 678.4184
STEP 23 ================================
prereg loss 327.43896 regularization 283.34973 reg_novel 28.58768
loss 639.37634
STEP 24 ================================
prereg loss 1254.5237 regularization 282.55246 reg_novel 28.587587
loss 1565.6637
STEP 25 ================================
prereg loss 1244.3258 regularization 281.75412 reg_novel 28.592148
loss 1554.6721
STEP 26 ================================
prereg loss 874.5867 regularization 280.954 reg_novel 28.601362
loss 1184.1421
STEP 27 ================================
prereg loss 1374.1588 regularization 280.15213 reg_novel 28.618961
loss 1682.9299
STEP 28 ================================
prereg loss 794.5054 regularization 279.34805 reg_novel 28.644676
loss 1102.498
STEP 29 ================================
prereg loss 1579.3542 regularization 278.5424 reg_novel 28.677576
loss 1886.5742
STEP 30 ================================
prereg loss 1684.3146 regularization 277.73135 reg_novel 28.719196
loss 1990.7651
STEP 31 ================================
prereg loss 1071.2855 regularization 276.92197 reg_novel 28.770031
loss 1376.9775
STEP 32 ================================
prereg loss 843.63324 regularization 276.1159 reg_novel 28.826258
loss 1148.5754
STEP 33 ================================
prereg loss 371.18292 regularization 275.31342 reg_novel 28.884333
loss 675.3807
STEP 34 ================================
prereg loss 1668.998 regularization 274.5173 reg_novel 28.94285
loss 1972.4583
STEP 35 ================================
prereg loss 1404.0186 regularization 273.7264 reg_novel 29.000345
loss 1706.7454
STEP 36 ================================
prereg loss 1957.288 regularization 272.95975 reg_novel 29.079138
loss 2259.327
STEP 37 ================================
prereg loss 2254.343 regularization 272.20325 reg_novel 29.156595
loss 2555.703
STEP 38 ================================
prereg loss 1659.3794 regularization 271.4633 reg_novel 29.228569
loss 1960.0713
STEP 39 ================================
prereg loss 1228.8047 regularization 270.73227 reg_novel 29.29727
loss 1528.8342
STEP 40 ================================
prereg loss 1782.7546 regularization 270.0443 reg_novel 29.393927
loss 2082.1929
STEP 41 ================================
prereg loss 1987.786 regularization 269.3629 reg_novel 29.478424
loss 2286.6274
STEP 42 ================================
prereg loss 2091.089 regularization 268.6937 reg_novel 29.554977
loss 2389.338
STEP 43 ================================
prereg loss 2486.6938 regularization 268.0293 reg_novel 29.623766
loss 2784.347
STEP 44 ================================
prereg loss 1979.824 regularization 267.34833 reg_novel 29.69
loss 2276.8623
STEP 45 ================================
prereg loss 2648.084 regularization 266.68237 reg_novel 29.716438
loss 2944.483
STEP 46 ================================
prereg loss 1677.3776 regularization 266.0214 reg_novel 29.742128
loss 1973.1411
STEP 47 ================================
prereg loss 1808.7527 regularization 265.36685 reg_novel 29.767656
loss 2103.8872
STEP 48 ================================
prereg loss 1794.6825 regularization 264.7239 reg_novel 29.791376
loss 2089.1978
STEP 49 ================================
prereg loss 1625.9438 regularization 264.08627 reg_novel 29.817314
loss 1919.8474
STEP 50 ================================
prereg loss 1591.1375 regularization 263.44925 reg_novel 29.845072
loss 1884.4318
STEP 51 ================================
prereg loss 1819.2485 regularization 262.8033 reg_novel 29.87555
loss 2111.9275
STEP 52 ================================
prereg loss 1536.1685 regularization 262.15796 reg_novel 29.90664
loss 1828.233
STEP 53 ================================
prereg loss 1577.2961 regularization 261.5186 reg_novel 29.939734
loss 1868.7544
STEP 54 ================================
prereg loss 1822.8186 regularization 260.88474 reg_novel 29.974443
loss 2113.6777
STEP 55 ================================
prereg loss 1890.3488 regularization 260.25916 reg_novel 30.014
loss 2180.6218
STEP 56 ================================
prereg loss 1766.6553 regularization 259.6378 reg_novel 30.055748
loss 2056.3489
STEP 57 ================================
prereg loss 1374.2668 regularization 259.0299 reg_novel 30.099882
loss 1663.3966
STEP 58 ================================
prereg loss 1649.5729 regularization 258.43066 reg_novel 30.142076
loss 1938.1456
STEP 59 ================================
prereg loss 1781.7709 regularization 257.83017 reg_novel 30.181421
loss 2069.7825
STEP 60 ================================
prereg loss 1389.7218 regularization 257.23514 reg_novel 30.220413
loss 1677.1774
STEP 61 ================================
prereg loss 1802.6259 regularization 256.6546 reg_novel 30.25659
loss 2089.537
STEP 62 ================================
prereg loss 871.88776 regularization 256.08795 reg_novel 30.291798
loss 1158.2676
STEP 63 ================================
prereg loss 1714.9109 regularization 255.52882 reg_novel 30.324877
loss 2000.7646
STEP 64 ================================
prereg loss 1410.547 regularization 254.97516 reg_novel 30.35904
loss 1695.8812
STEP 65 ================================
prereg loss 1321.8517 regularization 254.42601 reg_novel 30.396805
loss 1606.6746
STEP 66 ================================
prereg loss 1078.23 regularization 253.8746 reg_novel 30.432901
loss 1362.5375
STEP 67 ================================
prereg loss 1404.3806 regularization 253.32945 reg_novel 30.46394
loss 1688.1741
STEP 68 ================================
prereg loss 1370.6592 regularization 252.78873 reg_novel 30.493967
loss 1653.9419
STEP 69 ================================
prereg loss 1172.239 regularization 252.25621 reg_novel 30.524286
loss 1455.0195
STEP 70 ================================
prereg loss 1155.4893 regularization 251.7329 reg_novel 30.552046
loss 1437.7742
STEP 71 ================================
prereg loss 974.5714 regularization 251.22153 reg_novel 30.580471
loss 1256.3734
STEP 72 ================================
prereg loss 943.585 regularization 250.71446 reg_novel 30.610338
loss 1224.9098
STEP 73 ================================
prereg loss 1180.8921 regularization 250.20863 reg_novel 30.63928
loss 1461.74
STEP 74 ================================
prereg loss 1039.8105 regularization 249.70201 reg_novel 30.66841
loss 1320.1809
STEP 75 ================================
prereg loss 1149.1138 regularization 249.18907 reg_novel 30.694832
loss 1428.9977
STEP 76 ================================
prereg loss 1118.2827 regularization 248.68533 reg_novel 30.721241
loss 1397.6893
STEP 77 ================================
prereg loss 1096.7057 regularization 248.18863 reg_novel 30.745544
loss 1375.6399
STEP 78 ================================
prereg loss 1032.7346 regularization 247.69603 reg_novel 30.766928
loss 1311.1975
STEP 79 ================================
prereg loss 1242.3309 regularization 247.21014 reg_novel 30.786366
loss 1520.3274
STEP 80 ================================
prereg loss 1095.045 regularization 246.73503 reg_novel 30.803392
loss 1372.5835
STEP 81 ================================
prereg loss 919.90546 regularization 246.26315 reg_novel 30.81878
loss 1196.9874
STEP 82 ================================
prereg loss 1227.3152 regularization 245.79686 reg_novel 30.832352
loss 1503.9443
STEP 83 ================================
prereg loss 904.222 regularization 245.333 reg_novel 30.844109
loss 1180.399
STEP 84 ================================
prereg loss 833.0088 regularization 244.8724 reg_novel 30.857903
loss 1108.7391
STEP 85 ================================
prereg loss 1041.5199 regularization 244.4217 reg_novel 30.871965
loss 1316.8136
STEP 86 ================================
prereg loss 992.97577 regularization 243.97337 reg_novel 30.888437
loss 1267.8376
STEP 87 ================================
prereg loss 943.63715 regularization 243.5301 reg_novel 30.905117
loss 1218.0724
STEP 88 ================================
prereg loss 968.7199 regularization 243.09149 reg_novel 30.9206
loss 1242.7319
STEP 89 ================================
prereg loss 908.6359 regularization 242.65837 reg_novel 30.938833
loss 1182.2332
STEP 90 ================================
prereg loss 914.5285 regularization 242.2297 reg_novel 30.955162
loss 1187.7134
STEP 91 ================================
prereg loss 881.9008 regularization 241.80984 reg_novel 30.970154
loss 1154.6808
STEP 92 ================================
prereg loss 972.38275 regularization 241.3999 reg_novel 30.984781
loss 1244.7675
STEP 93 ================================
prereg loss 868.8218 regularization 240.99521 reg_novel 30.999296
loss 1140.8163
STEP 94 ================================
prereg loss 852.8934 regularization 240.60028 reg_novel 31.011784
loss 1124.5054
STEP 95 ================================
prereg loss 889.6281 regularization 240.2068 reg_novel 31.024141
loss 1160.8591
STEP 96 ================================
prereg loss 1033.4083 regularization 239.81647 reg_novel 31.037184
loss 1304.262
STEP 97 ================================
prereg loss 1250.0061 regularization 239.43115 reg_novel 31.04832
loss 1520.4856
STEP 98 ================================
prereg loss 1105.885 regularization 239.05595 reg_novel 31.063944
loss 1376.0049
STEP 99 ================================
prereg loss 955.3507 regularization 238.68355 reg_novel 31.08233
loss 1225.1166
STEP 100 ================================
prereg loss 936.0323 regularization 238.3145 reg_novel 31.100529
loss 1205.4473
2022-06-07T11:09:32.582

julia>
```

We either need to improve activations for these more realistic training data,
e.g. replacing `dot` with `x, y => dot(softmax(x), y)`, or a similar change,
or we need to focus on a non-recurrent setup first.
