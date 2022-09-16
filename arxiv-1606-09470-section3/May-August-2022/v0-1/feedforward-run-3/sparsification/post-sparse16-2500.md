# Testing and building upon sparse16-after-2500-steps model

Sparsity structure:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>5.88375f-6), "accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-3.28732f-6), "dot-2-1"=>Dict("dot"=>-0.166926), "accum-2…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272), "norm-1-2"=>Dict("norm"=>-3.48981f-6)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-3-2"=>Dict("dot"=>-5.78366f-6), "dot-4-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>-5.21862f-6)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-1"=>Dict("dot"=>5.51001f-6), "dot-3-2…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> count_interval(trainable["network_matrix"], -0.001f0, 0.001f0)
58

julia> count_interval(trainable["network_matrix"], -0.0001f0, 0.0001f0)
37

julia> count_interval(trainable["network_matrix"], -0.00001f0, 0.00001f0)
10

julia> count_neg_interval(trainable["network_matrix"], -1.0f0, 1.0f0)
2

julia> count_neg_interval(trainable["network_matrix"], -0.9f0, 0.9f0)
4

julia> count_neg_interval(trainable["network_matrix"], -0.8f0, 0.8f0)
4

julia> count_neg_interval(trainable["network_matrix"], -0.7f0, 0.7f0)
7

julia> count_neg_interval(trainable["network_matrix"], -0.6f0, 0.6f0)
8

julia> count_neg_interval(trainable["network_matrix"], -0.5f0, 0.5f0)
10

julia> count_neg_interval(trainable["network_matrix"], -0.4f0, 0.4f0)
19

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03182579 reg_l1 17.506548 reg_l2 10.677891
loss 3.5331354
3.5331354f0
```

Generalization (rather perfect):

```
$ diff test.jl test-original.jl
36c36
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
47c47
<     s::String = "tets string."
---
>     s::String = "test string."
56c56
< trainable["network_matrix"] = deserialize("sparse16-after-2500-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")
```

```
julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>5.88375f-6), "accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-3.28732f-6), "dot-2-1"=>Dict("dot"=>-0.166926), "accum-2…  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272), "norm-1-2"=>Dict("norm"=>-3.48981f-6)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-3-2"=>Dict("dot"=>-5.78366f-6), "dot-4-1"…  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>-5.21862f-6)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-1"=>Dict("dot"=>5.51001f-6), "dot-3-2…  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.031260885 reg_l1 17.506548 reg_l2 10.677891
loss 3.5325706
3.5325706f0
```

```
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(driving input) timer: 3.0
(getting on output) left: -0.08300162 right: 5.895536e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(driving input) timer: 4.0
(getting on output) left: -0.0066833887 right: -0.00015061733
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(driving input) timer: 5.0
(getting on output) left: 0.008235299 right: -0.00016214754
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(driving input) timer: 6.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(driving input) timer: 7.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(driving input) timer: 8.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(driving input) timer: 9.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(driving input) timer: 10.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(driving input) timer: 11.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(driving input) timer: 12.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(driving input) timer: 13.0
(getting on output) left: 0.00823554 right: -0.00017364937
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(driving input) timer: 14.0
(getting on output) left: 0.008235589 right: -7.341107e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(driving input) timer: 15.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(driving input) timer: 16.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(driving input) timer: 17.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(driving input) timer: 18.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(driving input) timer: 19.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(driving input) timer: 20.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(driving input) timer: 21.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(driving input) timer: 22.0
(getting on output) left: 0.008235589 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(driving input) timer: 23.0
(getting on output) left: 0.008232956 right: -7.3408584e-5
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(driving input) timer: 24.0
(getting on output) left: 0.94056636 right: 0.0009920545
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 24.0
(driving input) timer: 25.0
(getting on output) left: 1.0033191 right: -0.0005868999
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 25.0
(driving input) timer: 26.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 26.0
(driving input) timer: 27.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 27.0
(driving input) timer: 28.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 28.0
(driving input) timer: 29.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 29.0
(driving input) timer: 30.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 30.0
(driving input) timer: 31.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 31.0
(driving input) timer: 32.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 32.0
(driving input) timer: 33.0
(getting on output) left: 1.0033201 right: -0.00073390314
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 33.0
(driving input) timer: 34.0
(getting on output) left: 1.0033202 right: -0.000413337
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
(driving input) timer: 35.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 35.0
(driving input) timer: 36.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 36.0
(driving input) timer: 37.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 37.0
(driving input) timer: 38.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 38.0
(driving input) timer: 39.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 39.0
(driving input) timer: 40.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 40.0
(driving input) timer: 41.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 41.0
(driving input) timer: 42.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 42.0
(driving input) timer: 43.0
(getting on output) left: 1.0033202 right: -0.00041333048
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 43.0
(driving input) timer: 44.0
(getting on output) left: 1.0033203 right: 3.469389e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 44.0
(driving input) timer: 45.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 45.0
(driving input) timer: 46.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 46.0
(driving input) timer: 47.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 47.0
(driving input) timer: 48.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 48.0
(driving input) timer: 49.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 49.0
(driving input) timer: 50.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 50.0
(driving input) timer: 51.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 51.0
(driving input) timer: 52.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 52.0
(driving input) timer: 53.0
(getting on output) left: 1.0033203 right: 3.4701425e-5
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 53.0
(driving input) timer: 54.0
(getting on output) left: 1.0033207 right: 0.0021435216
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 54.0
(driving input) timer: 55.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 55.0
(driving input) timer: 56.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 56.0
(driving input) timer: 57.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 57.0
(driving input) timer: 58.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 58.0
(driving input) timer: 59.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 59.0
(driving input) timer: 60.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 60.0
(driving input) timer: 61.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 61.0
(driving input) timer: 62.0
(getting on output) left: 1.0033207 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 62.0
(driving input) timer: 63.0
(getting on output) left: 1.0033181 right: 0.0021435504
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 63.0
(driving input) timer: 64.0
(getting on output) left: 1.9356514 right: 0.008207415
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 64.0
(driving input) timer: 65.0
(getting on output) left: 1.9984046 right: 0.003980049
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 65.0
(driving input) timer: 66.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 66.0
(driving input) timer: 67.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 67.0
(driving input) timer: 68.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 68.0
(driving input) timer: 69.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 69.0
(driving input) timer: 70.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 70.0
(driving input) timer: 71.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 71.0
(driving input) timer: 72.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 72.0
(driving input) timer: 73.0
(getting on output) left: 1.9984057 right: 0.0035864722
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 73.0
(driving input) timer: 74.0
(getting on output) left: 1.9984059 right: 0.0048921118
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 74.0
(driving input) timer: 75.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 75.0
(driving input) timer: 76.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 76.0
(driving input) timer: 77.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 77.0
(driving input) timer: 78.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 78.0
(driving input) timer: 79.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 79.0
(driving input) timer: 80.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 80.0
(driving input) timer: 81.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 81.0
(driving input) timer: 82.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 82.0
(driving input) timer: 83.0
(getting on output) left: 1.9984059 right: 0.0048921285
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 83.0
(driving input) timer: 84.0
(getting on output) left: 1.998406 right: 0.006325229
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 84.0
(driving input) timer: 85.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 85.0
(driving input) timer: 86.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 86.0
(driving input) timer: 87.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 87.0
(driving input) timer: 88.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 88.0
(driving input) timer: 89.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 89.0
(driving input) timer: 90.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 90.0
(driving input) timer: 91.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 91.0
(driving input) timer: 92.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 92.0
(driving input) timer: 93.0
(getting on output) left: 1.998406 right: 0.0063252468
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 93.0
(driving input) timer: 94.0
(getting on output) left: 1.9984063 right: 0.007885811
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 94.0
(driving input) timer: 95.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 95.0
(driving input) timer: 96.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 96.0
(driving input) timer: 97.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 97.0
(driving input) timer: 98.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 98.0
(driving input) timer: 99.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 99.0
(driving input) timer: 100.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 100.0
(driving input) timer: 101.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 101.0
(driving input) timer: 102.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 102.0
(driving input) timer: 103.0
(getting on output) left: 1.9984063 right: 0.010694812
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 103.0
(driving input) timer: 104.0
(getting on output) left: 1.9984064 right: 0.012382827
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 104.0
(driving input) timer: 105.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 105.0
(driving input) timer: 106.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 106.0
(driving input) timer: 107.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 107.0
(driving input) timer: 108.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 108.0
(driving input) timer: 109.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 109.0
(driving input) timer: 110.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 110.0
(driving input) timer: 111.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 111.0
(driving input) timer: 112.0
(getting on output) left: 1.9984064 right: 0.015378978
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 112.0
(driving input) timer: 113.0
(getting on output) left: 1.9984064 right: 0.9927326
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 113.0
(driving input) timer: 114.0
(getting on output) left: 1.998407 right: 0.97667825
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 114.0
(driving input) timer: 115.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 115.0
(driving input) timer: 116.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 116.0
(driving input) timer: 117.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 117.0
(driving input) timer: 118.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 118.0
(driving input) timer: 119.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 119.0
(driving input) timer: 120.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 120.0
(driving input) timer: 121.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 121.0
(driving input) timer: 122.0
(getting on output) left: 1.998407 right: 0.9876882
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 122.0
(driving input) timer: 123.0
(getting on output) left: 1.998407 right: 1.9671019
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 123.0
(driving input) timer: 124.0
(getting on output) left: 1.9984078 right: 1.9629389
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 124.0
(driving input) timer: 125.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 125.0
(driving input) timer: 126.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 126.0
(driving input) timer: 127.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 127.0
(driving input) timer: 128.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 128.0
(driving input) timer: 129.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 129.0
(driving input) timer: 130.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 130.0
(driving input) timer: 131.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 131.0
(driving input) timer: 132.0
(getting on output) left: 1.9984078 right: 1.98722
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 132.0
(driving input) timer: 133.0
(getting on output) left: 1.9984078 right: 2.969604
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 133.0
(driving input) timer: 134.0
(getting on output) left: 1.9984092 right: 2.983963
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 134.0
(driving input) timer: 135.0
(getting on output) left: 1.9984092 right: 3.0281985
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 135.0
(driving input) timer: 136.0
(getting on output) left: 1.9984092 right: 3.0281985
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 136.0
(driving input) timer: 137.0
(getting on output) left: 1.9984092 right: 3.0281985
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 137.0
(driving input) timer: 138.0
(getting on output) left: 1.9984092 right: 3.0281985
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 138.0
(driving input) timer: 139.0
(getting on output) left: 1.9984092 right: 3.0281985
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 139.0
prereg loss 0.031260885 reg_l1 17.506548 reg_l2 10.677891
loss 3.5325706
```

---


---

Let's go back to

```
$ diff test.jl test-original.jl
36c36
<     l += 0.2f0 * reg_l1 # + 0.001f0 * reg_l2
---
>     l += 0.01f0 * reg_l1 # + 0.001f0 * reg_l2
56c56
< trainable["network_matrix"] = deserialize("sparse16-after-2500-steps-matrix.ser")
---
> trainable["network_matrix"] = deserialize("2500-steps-matrix.ser")

```

Check how sparsification would affect the results:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>5.88375f-6), "accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-3.28732f-6), "dot-2-1"=>Dict("dot"=>-0.166926), "accum-2…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272), "norm-1-2"=>Dict("norm"=>-3.48981f-6)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-3-2"=>Dict("dot"=>-5.78366f-6), "dot-4-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>-5.21862f-6)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-1"=>Dict("dot"=>5.51001f-6), "dot-3-2…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03182579 reg_l1 17.506548 reg_l2 10.677891
loss 3.5331354
3.5331354f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.00001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 24 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.000140903), "norm-2-1"=>Dict("norm"=>-0.000341628), "no…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926), "accum-2-2"=>Dict("dict"=>-5.40142f-5)), "dic…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-4-1"=>Dict("dot"=>-1.11046f-5)), "dict-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-2"=>Dict("dot"=>-1.1269f-5)), "dict-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> count(sparse)
99

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 24 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.000140903), "norm-2-1"=>Dict("norm"=>-0.000341628), "no…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926), "accum-2-2"=>Dict("dict"=>-5.40142f-5)), "dic…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-4-1"=>Dict("dot"=>-1.11046f-5)), "dict-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-2"=>Dict("dot"=>-1.1269f-5)), "dict-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.031823676 reg_l1 17.506495 reg_l2 10.677891
loss 3.5331225
3.5331225f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.0001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 21 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326)), "…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
72

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 21 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326)), "…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03181451 reg_l1 17.505331 reg_l2 10.677891
loss 3.5328808
3.5328808f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.00202319), "dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
51

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.00202319), "dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.032015312 reg_l1 17.5008 reg_l2 10.677889
loss 3.5321753
3.5321753f0

julia> count(sparse)
51
```

So, in reality, we can cut from 109 to 51 meaningful weights without any problem.

And let's see how long to cut before retraining further. We'll end up cutting to 41
before starting to retrain:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/sparsification")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 25 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>5.88375f-6), "accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-3.28732f-6), "dot-2-1"=>Dict("dot"=>-0.166926), "accum-2…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272), "norm-1-2"=>Dict("norm"=>-3.48981f-6)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-3-2"=>Dict("dot"=>-5.78366f-6), "dot-4-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-5-2"     => Dict("dict-2"=>Dict("compare-4-2"=>Dict("true"=>-5.21862f-6)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-1"=>Dict("dot"=>5.51001f-6), "dot-3-2…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03182579 reg_l1 17.506548 reg_l2 10.677891
loss 3.5331354
3.5331354f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.00001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 24 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.000140903), "norm-2-1"=>Dict("norm"=>-0.000341628), "no…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926), "accum-2-2"=>Dict("dict"=>-5.40142f-5)), "dic…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-4-1"=>Dict("dot"=>-1.11046f-5)), "dict-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-2"=>Dict("dot"=>-1.1269f-5)), "dict-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> count(sparse)
99

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 24 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("norm-3-2"=>Dict("norm"=>0.000140903), "norm-2-1"=>Dict("norm"=>-0.000341628), "no…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926), "accum-2-2"=>Dict("dict"=>-5.40142f-5)), "dic…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326), "c…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "norm-2-2"=>Dict("norm"=>-7.62393f-5), "com…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882), "dot-4-1"=>Dict("dot"=>-1.11046f-5)), "dict-1"…
  "norm-3-2"    => Dict("dict"=>Dict("accum-2-2"=>Dict("dict"=>-1.12947f-5), "accum-1-2"=>Dict("dict"=>0.292919)))
  "dot-4-1"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>7.36614f-5), "dot-3-2"=>Dict("dot"=>-1.1269f-5)), "dict-1…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-2-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>3.92443f-5)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))
  "norm-1-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>6.03532f-5)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.031823676 reg_l1 17.506495 reg_l2 10.677891
loss 3.5331225
3.5331225f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.0001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 21 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326)), "…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
72

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 21 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("input"=>Dict("char"=>0.000114912)), "dict-1"=>Dict("eos"=>Dict("char"=>-0.00020…
  "norm-5-2"    => Dict("dict"=>Dict("compare-4-2"=>Dict("false"=>-0.000166086), "norm-2-1"=>Dict("norm"=>-0.000341628)…
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "norm-4-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.000141194)))
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.000159859), "norm-2-1"=>Dict("norm"=>0.00202319), "dot…
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "norm-1-2"    => Dict("dict"=>Dict("eos"=>Dict("char"=>0.000127477)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "norm-3-1"=>Dict("norm"=>-0.000154198), "com…
  "norm-2-2"    => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.000184272)))
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-4-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.000500476), "norm-3-1"=>Dict("norm"=>0.000106326)), "…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03181451 reg_l1 17.505331 reg_l2 10.677891
loss 3.5328808
3.5328808f0

julia> sparse = sparsecopy(trainable["network_matrix"], 0.001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.00202319), "dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
51

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.00202319), "dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.032015312 reg_l1 17.5008 reg_l2 10.677889
loss 3.5321753
3.5321753f0

julia> count(sparse)
51

julia> count_interval(trainable["network_matrix"], -0.001f0, 0.001f0)
0

julia> count_interval(trainable["network_matrix"], -0.002f0, 0.002f0)
1

julia> count_interval(trainable["network_matrix"], -0.003f0, 0.003f0)
3

julia> count_interval(trainable["network_matrix"], -0.004f0, 0.004f0)
3

julia> count_interval(trainable["network_matrix"], -0.005f0, 0.005f0)
3

julia> count_interval(trainable["network_matrix"], -0.006f0, 0.006f0)
3

julia> count_interval(trainable["network_matrix"], -0.007f0, 0.007f0)
3

julia> count_interval(trainable["network_matrix"], -0.009f0, 0.009f0)
3

julia> count_interval(trainable["network_matrix"], -0.01f0, 0.01f0)
3

julia> # ok let's cut 3 more, see what happens

julia> sparse = sparsecopy(trainable["network_matrix"], 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
48

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.03876851 reg_l1 17.494648 reg_l2 10.6778755
loss 3.5376983
3.5376983f0

julia> # a bit more visible loss, but still very low

julia> count_interval(trainable["network_matrix"], -0.02f0, 0.02f0)
1

julia> count_interval(trainable["network_matrix"], -0.03f0, 0.03f0)
1

julia> count_interval(trainable["network_matrix"], -0.04f0, 0.04f0)
1

julia> count_interval(trainable["network_matrix"], -0.05f0, 0.05f0)
1

julia> count_interval(trainable["network_matrix"], -0.06f0, 0.06f0)
2

julia> # let's cut one more

julia> sparse = sparsecopy(trainable["network_matrix"], 0.05f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(sparse)
47

julia> trainable["network_matrix"] = sparse
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.084821664 reg_l1 17.476336 reg_l2 10.677542
loss 3.5800889
3.5800889f0

julia> # this is more noticeable, let's keep this, but try more

julia> count_interval(trainable["network_matrix"], -0.05f0, 0.05f0)
0

julia> count_interval(trainable["network_matrix"], -0.06f0, 0.06f0)
1

julia> count_interval(trainable["network_matrix"], -0.07f0, 0.07f0)
2

julia> count_interval(trainable["network_matrix"], -0.08f0, 0.08f0)
5

julia> count_interval(trainable["network_matrix"], -0.09f0, 0.09f0)
6

julia> count_interval(trainable["network_matrix"], -0.1f0, 0.1f0)
6

julia> sparse_06 = sparsecopy(sparse, 0.06f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> sparse_07 = sparsecopy(sparse, 0.07f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> sparse_08 = sparsecopy(sparse, 0.08f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.72474…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> sparse_09 = sparsecopy(sparse, 0.09f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.703505), "dot-5-1"=>Dict("dot"=>-0.263721), "compa…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.72474…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.084821664 reg_l1 17.476336 reg_l2 10.677542
loss 3.5800889
3.5800889f0

julia> count(trainable["network_matrix"])
47

julia> trainable["network_matrix"] = sparse_06
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664), "compare-4-2"=>Dict("false"=>0.0610959))…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(trainable["network_matrix"])
46

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.07521725 reg_l1 17.424942 reg_l2 10.6749
loss 3.5602057
3.5602057f0

julia> trainable["network_matrix"] = sparse_07
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(trainable["network_matrix"])
45

julia> reset_dicts!()

julia> trainable["network_matrix"] = sparse_07
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453), "dot-3-2"=>Dict("dot"=>0.0705307)), "dict-1"=>…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.15411034 reg_l1 17.363846 reg_l2 10.671167
loss 3.6268797
3.6268797f0

julia> trainable["network_matrix"] = sparse_08
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("dot-3-2"=>Dict("dot"=>-0.0829952), "compare-5-1"=>Dict("true"=>0.703505), "dot-…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.72474…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(trainable["network_matrix"])
42

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.12735501 reg_l1 17.14514 reg_l2 10.655207
loss 3.5563831
3.5563831f0

julia> trainable["network_matrix"] = sparse_09
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 15 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.537252), "input"=>Dict("char"=>-0.593143)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.08082)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4550…
  "norm-3-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.123041)))
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.703505), "dot-5-1"=>Dict("dot"=>-0.263721), "compa…
  "dot-3-1"     => Dict("dict-2"=>Dict("dot-2-1"=>Dict("dot"=>-0.166926)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.151…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.73453)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.72474…
  "dot-5-1"     => Dict("dict-2"=>Dict("compare-4-1"=>Dict("false"=>0.158664)), "dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.471662), "compare-3-1"=>Dict("false"=>-0.130565)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.223333), "eos"=>Dict("char"=>-0.274708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.162548), "compare-1-2"=>Dict("false"=>0.452412)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.318882)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3210…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.292919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43987)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.156941), "dot-2-1"=>Dict("dot"=>-0.187197), "accum-2-2"…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.44843)))

julia> count(trainable["network_matrix"])
41

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.49032277 reg_l1 17.062145 reg_l2 10.648317
loss 3.902752
3.902752f0

julia> count_interval(trainable["network_matrix"], -0.1f0, 0.1f0)
0

julia> count_interval(trainable["network_matrix"], -0.2f0, 0.2f0)
12

julia> count_interval(trainable["network_matrix"], -0.3f0, 0.3f0)
19

julia> count_interval(trainable["network_matrix"], -0.15f0, 0.15f0)
3

julia> count_interval(trainable["network_matrix"], -0.18f0, 0.18f0)
9

julia> # let's train this one

julia> count(trainable["network_matrix"])
41

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-27T13:38:23.449
STEP 1 ================================
prereg loss 0.49032277 reg_l1 17.062145 reg_l2 10.648317
loss 3.902752
STEP 2 ================================
prereg loss 0.45193085 reg_l1 17.055143 reg_l2 10.6551895
loss 3.8629594
STEP 3 ================================
prereg loss 0.3244663 reg_l1 17.048016 reg_l2 10.660875
loss 3.7340693
STEP 4 ================================
prereg loss 0.24732774 reg_l1 17.040846 reg_l2 10.666567
loss 3.655497
STEP 5 ================================
prereg loss 0.20270209 reg_l1 17.033512 reg_l2 10.672334
loss 3.6094046
STEP 6 ================================
prereg loss 0.15080987 reg_l1 17.0259 reg_l2 10.67807
loss 3.5559897
STEP 7 ================================
prereg loss 0.09695798 reg_l1 17.017992 reg_l2 10.683636
loss 3.5005565
STEP 8 ================================
prereg loss 0.05885269 reg_l1 17.009703 reg_l2 10.688795
loss 3.4607933
STEP 9 ================================
prereg loss 0.041828487 reg_l1 17.000898 reg_l2 10.69329
loss 3.442008
STEP 10 ================================
prereg loss 0.03623286 reg_l1 16.991453 reg_l2 10.696829
loss 3.4345236
STEP 11 ================================
prereg loss 0.030979842 reg_l1 16.98124 reg_l2 10.699188
loss 3.4272277
STEP 12 ================================
prereg loss 0.024889087 reg_l1 16.970154 reg_l2 10.70023
loss 3.4189198
STEP 13 ================================
prereg loss 0.023003312 reg_l1 16.958117 reg_l2 10.6998825
loss 3.4146266
STEP 14 ================================
prereg loss 0.028757427 reg_l1 16.945066 reg_l2 10.698142
loss 3.4177706
STEP 15 ================================
prereg loss 0.039615385 reg_l1 16.930971 reg_l2 10.69503
loss 3.4258096
STEP 16 ================================
prereg loss 0.04902463 reg_l1 16.915829 reg_l2 10.690599
loss 3.4321904
STEP 17 ================================
prereg loss 0.052719526 reg_l1 16.899689 reg_l2 10.684906
loss 3.4326575
STEP 18 ================================
prereg loss 0.0516121 reg_l1 16.882608 reg_l2 10.678011
loss 3.428134
STEP 19 ================================
prereg loss 0.04917815 reg_l1 16.86465 reg_l2 10.669965
loss 3.4221084
STEP 20 ================================
prereg loss 0.047620423 reg_l1 16.845905 reg_l2 10.660828
loss 3.4168017
STEP 21 ================================
prereg loss 0.046232045 reg_l1 16.826458 reg_l2 10.650673
loss 3.4115236
STEP 22 ================================
prereg loss 0.042952575 reg_l1 16.806417 reg_l2 10.639627
loss 3.404236
STEP 23 ================================
prereg loss 0.03718114 reg_l1 16.7859 reg_l2 10.627846
loss 3.3943613
STEP 24 ================================
prereg loss 0.0306139 reg_l1 16.765034 reg_l2 10.615517
loss 3.3836207
STEP 25 ================================
prereg loss 0.025571978 reg_l1 16.743929 reg_l2 10.602832
loss 3.374358
STEP 26 ================================
prereg loss 0.022927066 reg_l1 16.722708 reg_l2 10.589984
loss 3.3674686
STEP 27 ================================
prereg loss 0.021698805 reg_l1 16.70147 reg_l2 10.577145
loss 3.3619926
STEP 28 ================================
prereg loss 0.020527868 reg_l1 16.68031 reg_l2 10.564469
loss 3.3565898
STEP 29 ================================
prereg loss 0.019222314 reg_l1 16.659319 reg_l2 10.552081
loss 3.3510861
STEP 30 ================================
prereg loss 0.018724317 reg_l1 16.638586 reg_l2 10.540077
loss 3.3464415
STEP 31 ================================
prereg loss 0.019822601 reg_l1 16.618183 reg_l2 10.528531
loss 3.3434594
STEP 32 ================================
prereg loss 0.022146802 reg_l1 16.598173 reg_l2 10.517496
loss 3.3417814
STEP 33 ================================
prereg loss 0.024477327 reg_l1 16.578602 reg_l2 10.507015
loss 3.3401976
STEP 34 ================================
prereg loss 0.025927037 reg_l1 16.55949 reg_l2 10.497125
loss 3.337825
STEP 35 ================================
prereg loss 0.026603742 reg_l1 16.540854 reg_l2 10.487856
loss 3.3347745
STEP 36 ================================
prereg loss 0.027117833 reg_l1 16.5227 reg_l2 10.479236
loss 3.3316576
STEP 37 ================================
prereg loss 0.027682083 reg_l1 16.505007 reg_l2 10.471276
loss 3.3286834
STEP 38 ================================
prereg loss 0.02786551 reg_l1 16.487755 reg_l2 10.463977
loss 3.3254163
STEP 39 ================================
prereg loss 0.027191238 reg_l1 16.470905 reg_l2 10.457292
loss 3.3213723
STEP 40 ================================
prereg loss 0.025761101 reg_l1 16.454412 reg_l2 10.451161
loss 3.3166437
STEP 41 ================================
prereg loss 0.024158904 reg_l1 16.438206 reg_l2 10.445494
loss 3.3118002
STEP 42 ================================
prereg loss 0.022829095 reg_l1 16.422232 reg_l2 10.440186
loss 3.3072755
STEP 43 ================================
prereg loss 0.02171661 reg_l1 16.406416 reg_l2 10.435129
loss 3.3029997
STEP 44 ================================
prereg loss 0.020550149 reg_l1 16.390694 reg_l2 10.430219
loss 3.2986891
STEP 45 ================================
prereg loss 0.019324472 reg_l1 16.374998 reg_l2 10.425371
loss 3.2943242
STEP 46 ================================
prereg loss 0.018349985 reg_l1 16.359262 reg_l2 10.420505
loss 3.2902024
STEP 47 ================================
prereg loss 0.017863555 reg_l1 16.343431 reg_l2 10.415569
loss 3.2865498
STEP 48 ================================
prereg loss 0.017746422 reg_l1 16.327452 reg_l2 10.4105015
loss 3.2832367
STEP 49 ================================
prereg loss 0.017690506 reg_l1 16.311285 reg_l2 10.405258
loss 3.2799475
STEP 50 ================================
prereg loss 0.01756105 reg_l1 16.294884 reg_l2 10.399789
loss 3.2765377
STEP 51 ================================
prereg loss 0.017474733 reg_l1 16.278244 reg_l2 10.394054
loss 3.2731235
STEP 52 ================================
prereg loss 0.017553387 reg_l1 16.261333 reg_l2 10.388017
loss 3.26982
STEP 53 ================================
prereg loss 0.017722705 reg_l1 16.244158 reg_l2 10.381647
loss 3.266554
STEP 54 ================================
prereg loss 0.017815458 reg_l1 16.22672 reg_l2 10.374949
loss 3.2631595
STEP 55 ================================
prereg loss 0.017792247 reg_l1 16.209036 reg_l2 10.367939
loss 3.2595994
STEP 56 ================================
prereg loss 0.017768873 reg_l1 16.191141 reg_l2 10.360651
loss 3.2559972
STEP 57 ================================
prereg loss 0.017837923 reg_l1 16.173052 reg_l2 10.353142
loss 3.2524483
STEP 58 ================================
prereg loss 0.01795395 reg_l1 16.154821 reg_l2 10.3454685
loss 3.2489183
STEP 59 ================================
prereg loss 0.018029694 reg_l1 16.136484 reg_l2 10.337692
loss 3.2453265
STEP 60 ================================
prereg loss 0.018078085 reg_l1 16.118084 reg_l2 10.3298645
loss 3.241695
STEP 61 ================================
prereg loss 0.018194264 reg_l1 16.09966 reg_l2 10.32203
loss 3.2381265
STEP 62 ================================
prereg loss 0.018423682 reg_l1 16.08126 reg_l2 10.314241
loss 3.234676
STEP 63 ================================
prereg loss 0.01870983 reg_l1 16.062922 reg_l2 10.306532
loss 3.2312942
STEP 64 ================================
prereg loss 0.018979736 reg_l1 16.044676 reg_l2 10.298946
loss 3.227915
STEP 65 ================================
prereg loss 0.019227095 reg_l1 16.026556 reg_l2 10.291525
loss 3.2245383
STEP 66 ================================
prereg loss 0.019479005 reg_l1 16.008577 reg_l2 10.284304
loss 3.2211945
STEP 67 ================================
prereg loss 0.019717384 reg_l1 15.990759 reg_l2 10.277314
loss 3.2178693
STEP 68 ================================
prereg loss 0.01988241 reg_l1 15.973112 reg_l2 10.2705765
loss 3.214505
STEP 69 ================================
prereg loss 0.019944914 reg_l1 15.95563 reg_l2 10.264093
loss 3.211071
STEP 70 ================================
prereg loss 0.0199351 reg_l1 15.938318 reg_l2 10.257861
loss 3.207599
STEP 71 ================================
prereg loss 0.019889759 reg_l1 15.921157 reg_l2 10.251857
loss 3.2041214
STEP 72 ================================
prereg loss 0.01980985 reg_l1 15.904135 reg_l2 10.246056
loss 3.2006369
STEP 73 ================================
prereg loss 0.019682089 reg_l1 15.887226 reg_l2 10.240428
loss 3.1971273
STEP 74 ================================
prereg loss 0.019521957 reg_l1 15.870411 reg_l2 10.23495
loss 3.1936042
STEP 75 ================================
prereg loss 0.019364322 reg_l1 15.853667 reg_l2 10.2295885
loss 3.1900978
STEP 76 ================================
prereg loss 0.019223267 reg_l1 15.836956 reg_l2 10.2243185
loss 3.1866145
STEP 77 ================================
prereg loss 0.01908659 reg_l1 15.820265 reg_l2 10.219114
loss 3.1831396
STEP 78 ================================
prereg loss 0.018948112 reg_l1 15.803556 reg_l2 10.213942
loss 3.1796594
STEP 79 ================================
prereg loss 0.018824987 reg_l1 15.786817 reg_l2 10.208768
loss 3.1761885
STEP 80 ================================
prereg loss 0.01873325 reg_l1 15.77002 reg_l2 10.203568
loss 3.1727371
STEP 81 ================================
prereg loss 0.01866949 reg_l1 15.753153 reg_l2 10.198309
loss 3.1693003
STEP 82 ================================
prereg loss 0.018622845 reg_l1 15.736202 reg_l2 10.192978
loss 3.1658633
STEP 83 ================================
prereg loss 0.018593607 reg_l1 15.719162 reg_l2 10.187567
loss 3.162426
STEP 84 ================================
prereg loss 0.018588586 reg_l1 15.702033 reg_l2 10.182072
loss 3.1589952
STEP 85 ================================
prereg loss 0.018603927 reg_l1 15.68481 reg_l2 10.176498
loss 3.155566
STEP 86 ================================
prereg loss 0.018628277 reg_l1 15.6675 reg_l2 10.170856
loss 3.1521282
STEP 87 ================================
prereg loss 0.018658191 reg_l1 15.650115 reg_l2 10.165157
loss 3.1486812
STEP 88 ================================
prereg loss 0.018701611 reg_l1 15.632668 reg_l2 10.159412
loss 3.145235
STEP 89 ================================
prereg loss 0.01876243 reg_l1 15.61517 reg_l2 10.153636
loss 3.1417964
STEP 90 ================================
prereg loss 0.01883555 reg_l1 15.597639 reg_l2 10.147844
loss 3.1383634
STEP 91 ================================
prereg loss 0.018914828 reg_l1 15.5800905 reg_l2 10.142057
loss 3.134933
STEP 92 ================================
prereg loss 0.01899848 reg_l1 15.562539 reg_l2 10.136293
loss 3.1315062
STEP 93 ================================
prereg loss 0.019083833 reg_l1 15.545 reg_l2 10.130577
loss 3.1280837
STEP 94 ================================
prereg loss 0.019163612 reg_l1 15.527491 reg_l2 10.124926
loss 3.1246617
STEP 95 ================================
prereg loss 0.019229526 reg_l1 15.510015 reg_l2 10.119353
loss 3.1212323
STEP 96 ================================
prereg loss 0.019280747 reg_l1 15.492584 reg_l2 10.113875
loss 3.1177976
STEP 97 ================================
prereg loss 0.019319434 reg_l1 15.475208 reg_l2 10.10849
loss 3.1143613
STEP 98 ================================
prereg loss 0.019345447 reg_l1 15.457884 reg_l2 10.103205
loss 3.1109223
STEP 99 ================================
prereg loss 0.019357063 reg_l1 15.440615 reg_l2 10.098014
loss 3.10748
STEP 100 ================================
prereg loss 0.019355448 reg_l1 15.423398 reg_l2 10.092921
loss 3.1040351
2022-06-27T13:44:03.743

julia> serialize("sparse17-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse17-after-100-steps-opt.ser", opt)
```

This is a nice convergence, and we are down to 41 parameters.

Interestingly enough, these 8 parameters seem to not be important at all
(perhaps, some things are getting disconnected as we prune):

```
julia> count_interval(trainable["network_matrix"], -0.1f0, 0.1f0)
8

julia> count_interval(trainable["network_matrix"], -0.01f0, 0.01f0)
0

julia> count_interval(trainable["network_matrix"], -0.05f0, 0.05f0)
1

julia> count_interval(trainable["network_matrix"], -0.07f0, 0.07f0)
5

julia> count_interval(trainable["network_matrix"], -0.2f0, 0.2f0)
19

julia> sparse18 = sparsecopy(trainable["network_matrix"], 0.1f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse18
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
33

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.019343078 reg_l1 14.893552 reg_l2 10.051794
loss 2.9980536
2.9980536f0
```

So, we seem to be down to no more than 33 parameters.

But cutting to 22 parameters creates a big loss, let's not do that, but find something
in between. Cutting to 27 parameters actually looks nice:

```
julia> count_interval(trainable["network_matrix"], -0.1f0, 0.1f0)
8

julia> count_interval(trainable["network_matrix"], -0.01f0, 0.01f0)
0

julia> count_interval(trainable["network_matrix"], -0.05f0, 0.05f0)
1

julia> count_interval(trainable["network_matrix"], -0.07f0, 0.07f0)
5

julia> count_interval(trainable["network_matrix"], -0.2f0, 0.2f0)
19

julia> sparse18 = sparsecopy(trainable["network_matrix"], 0.1f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse18
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
33

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.019343078 reg_l1 14.893552 reg_l2 10.051794
loss 2.9980536
2.9980536f0

julia> sparse_18_1 = sparsecopy(trainable["network_matrix"], 0.2f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 11 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "compare-5-2"=>Dict("false"=>0.452573)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.451591)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse_18_1
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 11 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "compare-5-2"=>Dict("false"=>0.452573)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.451591)), "dict-1"=>Dict("compare-1-2"=>Dict("fal…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
22

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 50.71021 reg_l1 13.284479 reg_l2 9.808733
loss 53.367104
53.367104f0

julia> trainable["network_matrix"] = sparse18
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count_interval(trainable["network_matrix"], -0.2f0, 0.2f0)
11

julia> count_interval(trainable["network_matrix"], -0.15f0, 0.15f0)
6

julia> count_interval(trainable["network_matrix"], -0.12f0, 0.12f0)
2

julia> count_interval(trainable["network_matrix"], -0.11f0, 0.11f0)
0

julia> sparse_18_0_12 = sparsecopy(trainable["network_matrix"], 0.12f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse_18_0_12
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 13 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "dot-5-1"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>-0.125796)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("accum-1-2"=>Dict("dict"=>-0.123333), "eos"=>Dict("char"=>-0.174708)), "dict-1"=…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
31

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.019343078 reg_l1 14.669011 reg_l2 10.026585
loss 2.9531455
2.9531455f0

julia> count_interval(trainable["network_matrix"], -0.15f0, 0.15f0)
4

julia> count_interval(trainable["network_matrix"], -0.13f0, 0.13f0)
2

julia> sparse_18_0_13 = sparsecopy(trainable["network_matrix"], 0.13f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.174708)), "dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse_18_0_13
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667), "compare-3-1"=>Dict("false"=>-0.134369)), "d…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.174708)), "dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
29

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.019343078 reg_l1 14.419881 reg_l2 9.995549
loss 2.9033194
2.9033194f0

julia> count_interval(trainable["network_matrix"], -0.14f0, 0.14f0)
2

julia> sparse_18_0_14 = sparsecopy(trainable["network_matrix"], 0.14f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.174708)), "dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> trainable["network_matrix"] = sparse_18_0_14
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.538981), "input"=>Dict("char"=>-0.585397)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.09401)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.4698…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.717506), "dot-5-1"=>Dict("dot"=>-0.163721), "compa…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.744218)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.7344…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.472667)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.174708)), "dict-1"=>Dict("eos"=>Dict("char"=>0.248892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.163473), "compare-1-2"=>Dict("false"=>0.451591)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.323593)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.3257…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.192919)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.43914)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.313095)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.43449)))

julia> count(trainable["network_matrix"])
27

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.78494686 reg_l1 14.149865 reg_l2 9.959094
loss 3.61492
3.61492f0

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-27T14:13:45.649
STEP 1 ================================
prereg loss 0.78494686 reg_l1 14.149865 reg_l2 9.959094
loss 3.61492
STEP 2 ================================
prereg loss 0.6646826 reg_l1 14.144866 reg_l2 9.958766
loss 3.493656
STEP 3 ================================
prereg loss 0.63376105 reg_l1 14.133311 reg_l2 9.948374
loss 3.4604235
STEP 4 ================================
prereg loss 0.6530575 reg_l1 14.125848 reg_l2 9.942731
loss 3.4782271
STEP 5 ================================
prereg loss 0.66210616 reg_l1 14.124764 reg_l2 9.944425
loss 3.487059
STEP 6 ================================
prereg loss 0.6465158 reg_l1 14.126916 reg_l2 9.949706
loss 3.471899
STEP 7 ================================
prereg loss 0.61561036 reg_l1 14.129809 reg_l2 9.955775
loss 3.4415722
STEP 8 ================================
prereg loss 0.5799277 reg_l1 14.131653 reg_l2 9.9604845
loss 3.4062583
STEP 9 ================================
prereg loss 0.5468845 reg_l1 14.131995 reg_l2 9.963239
loss 3.3732836
STEP 10 ================================
prereg loss 0.52089214 reg_l1 14.131373 reg_l2 9.96472
loss 3.3471668
STEP 11 ================================
prereg loss 0.50268984 reg_l1 14.130551 reg_l2 9.965953
loss 3.3288002
STEP 12 ================================
prereg loss 0.48944354 reg_l1 14.130244 reg_l2 9.967901
loss 3.3154924
STEP 13 ================================
prereg loss 0.47681913 reg_l1 14.130791 reg_l2 9.971014
loss 3.3029773
STEP 14 ================================
prereg loss 0.4616531 reg_l1 14.131993 reg_l2 9.975069
loss 3.2880516
STEP 15 ================================
prereg loss 0.44320208 reg_l1 14.13336 reg_l2 9.979476
loss 3.269874
STEP 16 ================================
prereg loss 0.42277178 reg_l1 14.134334 reg_l2 9.983536
loss 3.2496386
STEP 17 ================================
prereg loss 0.40253013 reg_l1 14.134483 reg_l2 9.986673
loss 3.2294269
STEP 18 ================================
prereg loss 0.38441536 reg_l1 14.133699 reg_l2 9.988706
loss 3.2111554
STEP 19 ================================
prereg loss 0.36937395 reg_l1 14.132239 reg_l2 9.989919
loss 3.195822
STEP 20 ================================
prereg loss 0.35697046 reg_l1 14.130589 reg_l2 9.990889
loss 3.1830883
STEP 21 ================================
prereg loss 0.34561506 reg_l1 14.129258 reg_l2 9.99226
loss 3.1714668
STEP 22 ================================
prereg loss 0.33348778 reg_l1 14.1286335 reg_l2 9.994506
loss 3.1592145
STEP 23 ================================
prereg loss 0.31954163 reg_l1 14.128815 reg_l2 9.997743
loss 3.1453047
STEP 24 ================================
prereg loss 0.30392236 reg_l1 14.129608 reg_l2 10.001736
loss 3.1298442
STEP 25 ================================
prereg loss 0.28763646 reg_l1 14.130642 reg_l2 10.006029
loss 3.113765
STEP 26 ================================
prereg loss 0.2718732 reg_l1 14.131505 reg_l2 10.010128
loss 3.0981743
STEP 27 ================================
prereg loss 0.25748402 reg_l1 14.131922 reg_l2 10.01369
loss 3.0838683
STEP 28 ================================
prereg loss 0.24473737 reg_l1 14.131824 reg_l2 10.016644
loss 3.0711021
STEP 29 ================================
prereg loss 0.23331828 reg_l1 14.131344 reg_l2 10.019181
loss 3.0595872
STEP 30 ================================
prereg loss 0.2225497 reg_l1 14.130729 reg_l2 10.021631
loss 3.0486956
STEP 31 ================================
prereg loss 0.21177305 reg_l1 14.130208 reg_l2 10.024304
loss 3.0378149
STEP 32 ================================
prereg loss 0.20073242 reg_l1 14.129896 reg_l2 10.027338
loss 3.0267117
STEP 33 ================================
prereg loss 0.18969467 reg_l1 14.129734 reg_l2 10.030655
loss 3.0156415
STEP 34 ================================
prereg loss 0.17919137 reg_l1 14.129548 reg_l2 10.034022
loss 3.005101
STEP 35 ================================
prereg loss 0.1696319 reg_l1 14.12913 reg_l2 10.037161
loss 2.9954581
STEP 36 ================================
prereg loss 0.1610553 reg_l1 14.128355 reg_l2 10.039891
loss 2.9867263
STEP 37 ================================
prereg loss 0.15316987 reg_l1 14.12724 reg_l2 10.042215
loss 2.978618
STEP 38 ================================
prereg loss 0.1455481 reg_l1 14.125931 reg_l2 10.044322
loss 2.9707344
STEP 39 ================================
prereg loss 0.13785785 reg_l1 14.12465 reg_l2 10.046478
loss 2.9627879
STEP 40 ================================
prereg loss 0.13000062 reg_l1 14.123564 reg_l2 10.048913
loss 2.9547133
STEP 41 ================================
prereg loss 0.122154064 reg_l1 14.122735 reg_l2 10.051699
loss 2.946701
STEP 42 ================================
prereg loss 0.11464812 reg_l1 14.122066 reg_l2 10.054731
loss 2.9390612
STEP 43 ================================
prereg loss 0.10775133 reg_l1 14.121358 reg_l2 10.057771
loss 2.932023
STEP 44 ================================
prereg loss 0.10151755 reg_l1 14.120416 reg_l2 10.060571
loss 2.9256005
STEP 45 ================================
prereg loss 0.09581181 reg_l1 14.119104 reg_l2 10.06297
loss 2.9196327
STEP 46 ================================
prereg loss 0.09044546 reg_l1 14.117422 reg_l2 10.064976
loss 2.91393
STEP 47 ================================
prereg loss 0.08530407 reg_l1 14.115484 reg_l2 10.066731
loss 2.908401
STEP 48 ================================
prereg loss 0.08038432 reg_l1 14.113445 reg_l2 10.068438
loss 2.9030733
STEP 49 ================================
prereg loss 0.07575932 reg_l1 14.11143 reg_l2 10.070247
loss 2.8980455
STEP 50 ================================
prereg loss 0.07149861 reg_l1 14.109472 reg_l2 10.072195
loss 2.893393
STEP 51 ================================
prereg loss 0.067604296 reg_l1 14.107525 reg_l2 10.074203
loss 2.8891094
STEP 52 ================================
prereg loss 0.06399126 reg_l1 14.105474 reg_l2 10.076126
loss 2.8850863
STEP 53 ================================
prereg loss 0.060541116 reg_l1 14.103236 reg_l2 10.077847
loss 2.8811884
STEP 54 ================================
prereg loss 0.057183065 reg_l1 14.100789 reg_l2 10.079333
loss 2.8773408
STEP 55 ================================
prereg loss 0.053929277 reg_l1 14.0981865 reg_l2 10.080655
loss 2.8735666
STEP 56 ================================
prereg loss 0.05084436 reg_l1 14.095513 reg_l2 10.08193
loss 2.8699472
STEP 57 ================================
prereg loss 0.047985848 reg_l1 14.092838 reg_l2 10.083257
loss 2.8665535
STEP 58 ================================
prereg loss 0.045364548 reg_l1 14.090174 reg_l2 10.084647
loss 2.8633995
STEP 59 ================================
prereg loss 0.042957816 reg_l1 14.087449 reg_l2 10.086031
loss 2.8604476
STEP 60 ================================
prereg loss 0.04072689 reg_l1 14.084573 reg_l2 10.087283
loss 2.8576415
STEP 61 ================================
prereg loss 0.038650345 reg_l1 14.081464 reg_l2 10.088299
loss 2.854943
STEP 62 ================================
prereg loss 0.036734592 reg_l1 14.078098 reg_l2 10.089052
loss 2.8523543
STEP 63 ================================
prereg loss 0.034994803 reg_l1 14.07453 reg_l2 10.089597
loss 2.8499007
STEP 64 ================================
prereg loss 0.033421643 reg_l1 14.070845 reg_l2 10.090045
loss 2.8475907
STEP 65 ================================
prereg loss 0.03197261 reg_l1 14.067128 reg_l2 10.090495
loss 2.8453984
STEP 66 ================================
prereg loss 0.030590381 reg_l1 14.063411 reg_l2 10.090988
loss 2.8432724
STEP 67 ================================
prereg loss 0.02924253 reg_l1 14.059672 reg_l2 10.091489
loss 2.841177
STEP 68 ================================
prereg loss 0.02793277 reg_l1 14.05585 reg_l2 10.091919
loss 2.839103
STEP 69 ================================
prereg loss 0.026690463 reg_l1 14.05188 reg_l2 10.092202
loss 2.8370664
STEP 70 ================================
prereg loss 0.02554795 reg_l1 14.047746 reg_l2 10.092311
loss 2.835097
STEP 71 ================================
prereg loss 0.02451364 reg_l1 14.043463 reg_l2 10.092271
loss 2.8332064
STEP 72 ================================
prereg loss 0.023570677 reg_l1 14.039075 reg_l2 10.092147
loss 2.8313859
STEP 73 ================================
prereg loss 0.022692263 reg_l1 14.034624 reg_l2 10.091989
loss 2.829617
STEP 74 ================================
prereg loss 0.021862986 reg_l1 14.030111 reg_l2 10.091807
loss 2.8278854
STEP 75 ================================
prereg loss 0.021087065 reg_l1 14.025522 reg_l2 10.09157
loss 2.8261917
STEP 76 ================================
prereg loss 0.020377472 reg_l1 14.02081 reg_l2 10.091225
loss 2.8245394
STEP 77 ================================
prereg loss 0.019736059 reg_l1 14.015963 reg_l2 10.090737
loss 2.8229287
STEP 78 ================================
prereg loss 0.019149346 reg_l1 14.010982 reg_l2 10.090117
loss 2.8213456
STEP 79 ================================
prereg loss 0.018592417 reg_l1 14.005907 reg_l2 10.089409
loss 2.819774
STEP 80 ================================
prereg loss 0.01804521 reg_l1 14.000781 reg_l2 10.088661
loss 2.8182015
STEP 81 ================================
prereg loss 0.01750215 reg_l1 13.995623 reg_l2 10.087908
loss 2.8166265
STEP 82 ================================
prereg loss 0.01697375 reg_l1 13.990431 reg_l2 10.08714
loss 2.81506
STEP 83 ================================
prereg loss 0.01647707 reg_l1 13.985167 reg_l2 10.086318
loss 2.8135104
STEP 84 ================================
prereg loss 0.016022708 reg_l1 13.979801 reg_l2 10.085393
loss 2.8119829
STEP 85 ================================
prereg loss 0.015612286 reg_l1 13.974309 reg_l2 10.084345
loss 2.8104742
STEP 86 ================================
prereg loss 0.015238451 reg_l1 13.968703 reg_l2 10.083182
loss 2.8089793
STEP 87 ================================
prereg loss 0.014894242 reg_l1 13.963013 reg_l2 10.081947
loss 2.8074968
STEP 88 ================================
prereg loss 0.014572911 reg_l1 13.957264 reg_l2 10.080673
loss 2.8060257
STEP 89 ================================
prereg loss 0.014271212 reg_l1 13.951475 reg_l2 10.079375
loss 2.8045664
STEP 90 ================================
prereg loss 0.013985785 reg_l1 13.945639 reg_l2 10.078046
loss 2.8031137
STEP 91 ================================
prereg loss 0.013713761 reg_l1 13.939737 reg_l2 10.076658
loss 2.8016613
STEP 92 ================================
prereg loss 0.013452708 reg_l1 13.933762 reg_l2 10.075197
loss 2.8002052
STEP 93 ================================
prereg loss 0.0132018365 reg_l1 13.927709 reg_l2 10.073662
loss 2.7987437
STEP 94 ================================
prereg loss 0.012962723 reg_l1 13.9215975 reg_l2 10.072072
loss 2.7972825
STEP 95 ================================
prereg loss 0.012735584 reg_l1 13.915439 reg_l2 10.070449
loss 2.7958233
STEP 96 ================================
prereg loss 0.012521484 reg_l1 13.909245 reg_l2 10.068805
loss 2.7943704
STEP 97 ================================
prereg loss 0.012321727 reg_l1 13.9030075 reg_l2 10.067135
loss 2.7929232
STEP 98 ================================
prereg loss 0.012139557 reg_l1 13.896714 reg_l2 10.065416
loss 2.7914824
STEP 99 ================================
prereg loss 0.011976432 reg_l1 13.890349 reg_l2 10.063632
loss 2.7900465
STEP 100 ================================
prereg loss 0.011831867 reg_l1 13.883911 reg_l2 10.061781
loss 2.788614
2022-06-27T14:18:16.670

julia> serialize("sparse18-after-100-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse18-after-100-steps-opt.ser", opt)

julia> open("sparse18-after-100-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end
```

Here is the sequence leading to sparse19 model after 500 steps:

```
julia> count_interval(trainable["network_matrix"], -0.13f0, 0.13f0)
3

julia> count_interval(trainable["network_matrix"], -0.14f0, 0.14f0)
3

julia> count_interval(trainable["network_matrix"], -0.1f0, 0.1f0)
3

julia> count_interval(trainable["network_matrix"], -0.08f0, 0.08f0)
2

julia> count_interval(trainable["network_matrix"], -0.06f0, 0.06f0)
0

julia> count_interval(trainable["network_matrix"], -0.07f0, 0.07f0)
1

julia> count_interval(trainable["network_matrix"], -0.08f0, 0.08f0)
2

julia> count_interval(trainable["network_matrix"], -0.13f0, 0.13f0)
3

julia> count_interval(trainable["network_matrix"], -0.15f0, 0.15f0)
4

julia> count_interval(trainable["network_matrix"], -0.16f0, 0.16f0)
4

julia> count_interval(trainable["network_matrix"], -0.16f0, 0.17f0)
4

julia> count_interval(trainable["network_matrix"], -0.17f0, 0.17f0)
4

julia> count_interval(trainable["network_matrix"], -0.18f0, 0.18f0)
4

julia> count_interval(trainable["network_matrix"], -0.19f0, 0.19f0)
5

julia> count_interval(trainable["network_matrix"], -0.2f0, 0.2f0)
6

julia> count_interval(trainable["network_matrix"], -0.21f0, 0.21f0)
6

julia> count_interval(trainable["network_matrix"], -0.22f0, 0.22f0)
7

julia> sparse_19_0_07 = sparsecopy(trainable["network_matrix"], 0.07f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.0747079)), "dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.0929191)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> trainable["network_matrix"] = sparse_19_0_07
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-0.0747079)), "dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.0929191)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> count(trainable["network_matrix"])
26

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.011702288 reg_l1 13.813698 reg_l2 10.055818
loss 2.774442
2.774442f0

julia> sparse_19_0_08 = sparsecopy(trainable["network_matrix"], 0.08f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.0929191)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> trainable["network_matrix"] = sparse_19_0_08
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 12 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "norm-3-2"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.0929191)))
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> count(trainable["network_matrix"])
25

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.011702288 reg_l1 13.73899 reg_l2 10.050237
loss 2.7595003
2.7595003f0

julia> sparse_19_0_14 = sparsecopy(trainable["network_matrix"], 0.14f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 11 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> trainable["network_matrix"] = sparse_19_0_14
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 11 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…
  "dot-2-1"     => Dict("dict-1"=>Dict("eos"=>Dict("char"=>0.148892)))
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))

julia> count(trainable["network_matrix"])
24

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.011702288 reg_l1 13.6460705 reg_l2 10.041602
loss 2.7409165
2.7409165f0

julia> sparse_19_0_15 = sparsecopy(trainable["network_matrix"], 0.15f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse_19_0_15
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
23

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.011702288 reg_l1 13.497178 reg_l2 10.019434
loss 2.711138
2.711138f0

julia> sparse_19_0_19 = sparsecopy(trainable["network_matrix"], 0.19f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.490925)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> trainable["network_matrix"] = sparse_19_0_19
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-3-1" => Dict("dict-2"=>Dict("compare-1-2"=>Dict("false"=>0.490925)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> count(trainable["network_matrix"])
22

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 17.562979 reg_l1 13.314265 reg_l2 9.985977
loss 20.225832
20.225832f0

julia> trainable["network_matrix"] = sparse_19_0_15
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 10 entries:
  "norm-2-1"    => Dict("dict"=>Dict("accum-1-2"=>Dict("dict"=>0.557708), "input"=>Dict("char"=>-0.687314)))
  "dot-2-2"     => Dict("dict-2"=>Dict("eos"=>Dict("char"=>-1.12866)), "dict-1"=>Dict("accum-1-2"=>Dict("dict"=>-0.5028…
  "output"      => Dict("dict-2"=>Dict("compare-5-1"=>Dict("true"=>0.764284), "compare-5-2"=>Dict("false"=>0.404984)), …
  "compare-1-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.4785)), "dict-1"=>Dict("const_1"=>Dict("const_1"=>-…
  "compare-5-1" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>-0.72441)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>0.71462…
  "compare-3-1" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>-0.182913), "compare-1-2"=>Dict("false"=>0.490925)), "d…
  "dot-3-2"     => Dict("dict-1"=>Dict("dot-2-1"=>Dict("dot"=>0.213096)))
  "compare-5-2" => Dict("dict-2"=>Dict("dot-2-2"=>Dict("dot"=>0.247073)), "dict-1"=>Dict("dot-2-2"=>Dict("dot"=>-0.2492…
  "accum-1-2"   => Dict("dict-2"=>Dict("input"=>Dict("char"=>1.3542)))
  "compare-4-2" => Dict("dict-2"=>Dict("norm-2-1"=>Dict("norm"=>0.491465)), "dict-1"=>Dict("norm-2-1"=>Dict("norm"=>-0.…

julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 0.011702288 reg_l1 13.497178 reg_l2 10.019434
loss 2.711138
2.711138f0

julia> count(trainable["network_matrix"])
23

julia> # OK, so we can do 23, but if we do 22, we'll have to start really high

julia> # so let's do 23 and 100 steps

julia> opt = TreeADAM(trainable["network_matrix"], 0.001f0)
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[0.9, 0.999], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}(), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}())

julia> steps!(100)
2022-06-27T15:10:43.731
STEP 1 ================================
prereg loss 0.011702288 reg_l1 13.497178 reg_l2 10.019434
loss 2.711138
STEP 2 ================================
prereg loss 0.041841023 reg_l1 13.4981785 reg_l2 10.023064
loss 2.7414768
STEP 3 ================================
prereg loss 0.012752748 reg_l1 13.48936 reg_l2 10.013783
loss 2.7106247
STEP 4 ================================
prereg loss 0.02999337 reg_l1 13.4888315 reg_l2 10.014991
loss 2.7277596
STEP 5 ================================
prereg loss 0.036308113 reg_l1 13.490214 reg_l2 10.0179615
loss 2.734351
STEP 6 ================================
prereg loss 0.024146188 reg_l1 13.489976 reg_l2 10.018571
loss 2.7221413
STEP 7 ================================
prereg loss 0.01234177 reg_l1 13.487547 reg_l2 10.016078
loss 2.7098513
STEP 8 ================================
prereg loss 0.011636903 reg_l1 13.483939 reg_l2 10.012025
loss 2.7084248
STEP 9 ================================
prereg loss 0.018165091 reg_l1 13.480261 reg_l2 10.008075
loss 2.7142174
STEP 10 ================================
prereg loss 0.020895021 reg_l1 13.477369 reg_l2 10.005405
loss 2.716369
STEP 11 ================================
prereg loss 0.016917555 reg_l1 13.475267 reg_l2 10.003969
loss 2.711971
STEP 12 ================================
prereg loss 0.01185096 reg_l1 13.473453 reg_l2 10.003075
loss 2.7065415
STEP 13 ================================
prereg loss 0.011167507 reg_l1 13.471284 reg_l2 10.001839
loss 2.7054243
STEP 14 ================================
prereg loss 0.014754491 reg_l1 13.468323 reg_l2 9.99957
loss 2.708419
STEP 15 ================================
prereg loss 0.018144874 reg_l1 13.464598 reg_l2 9.996206
loss 2.7110643
STEP 16 ================================
prereg loss 0.018060008 reg_l1 13.460518 reg_l2 9.992252
loss 2.7101636
STEP 17 ================================
prereg loss 0.015210945 reg_l1 13.456617 reg_l2 9.98845
loss 2.7065344
STEP 18 ================================
prereg loss 0.012464373 reg_l1 13.453327 reg_l2 9.985408
loss 2.7031298
STEP 19 ================================
prereg loss 0.0118780555 reg_l1 13.450731 reg_l2 9.98329
loss 2.7020242
STEP 20 ================================
prereg loss 0.013066541 reg_l1 13.448561 reg_l2 9.9818125
loss 2.7027788
STEP 21 ================================
prereg loss 0.013999245 reg_l1 13.446365 reg_l2 9.980438
loss 2.7032723
STEP 22 ================================
prereg loss 0.013418317 reg_l1 13.443729 reg_l2 9.978634
loss 2.7021642
STEP 23 ================================
prereg loss 0.012079613 reg_l1 13.440485 reg_l2 9.976147
loss 2.7001767
STEP 24 ================================
prereg loss 0.011627948 reg_l1 13.436774 reg_l2 9.973125
loss 2.6989827
STEP 25 ================================
prereg loss 0.012657323 reg_l1 13.432978 reg_l2 9.970014
loss 2.699253
STEP 26 ================================
prereg loss 0.014084425 reg_l1 13.429496 reg_l2 9.967299
loss 2.6999836
STEP 27 ================================
prereg loss 0.014436209 reg_l1 13.426562 reg_l2 9.96526
loss 2.6997488
STEP 28 ================================
prereg loss 0.0134344725 reg_l1 13.424144 reg_l2 9.963847
loss 2.6982632
STEP 29 ================================
prereg loss 0.012080879 reg_l1 13.421977 reg_l2 9.962749
loss 2.6964765
STEP 30 ================================
prereg loss 0.0114512 reg_l1 13.4197235 reg_l2 9.96155
loss 2.695396
STEP 31 ================================
prereg loss 0.0116648665 reg_l1 13.417131 reg_l2 9.959959
loss 2.6950912
STEP 32 ================================
prereg loss 0.012017189 reg_l1 13.414159 reg_l2 9.957938
loss 2.694849
STEP 33 ================================
prereg loss 0.011928221 reg_l1 13.410963 reg_l2 9.955717
loss 2.694121
STEP 34 ================================
prereg loss 0.011535798 reg_l1 13.407806 reg_l2 9.953626
loss 2.693097
STEP 35 ================================
prereg loss 0.011364533 reg_l1 13.404896 reg_l2 9.951916
loss 2.6923437
STEP 36 ================================
prereg loss 0.011617799 reg_l1 13.402282 reg_l2 9.950622
loss 2.6920743
STEP 37 ================================
prereg loss 0.011941165 reg_l1 13.399855 reg_l2 9.949565
loss 2.6919122
STEP 38 ================================
prereg loss 0.011870129 reg_l1 13.397409 reg_l2 9.948464
loss 2.6913521
STEP 39 ================================
prereg loss 0.011383445 reg_l1 13.394785 reg_l2 9.947101
loss 2.6903405
STEP 40 ================================
prereg loss 0.010889004 reg_l1 13.391944 reg_l2 9.945437
loss 2.689278
STEP 41 ================================
prereg loss 0.010726977 reg_l1 13.388973 reg_l2 9.943624
loss 2.6885216
STEP 42 ================================
prereg loss 0.0108054355 reg_l1 13.3860235 reg_l2 9.941882
loss 2.6880102
STEP 43 ================================
prereg loss 0.010798363 reg_l1 13.383192 reg_l2 9.940359
loss 2.6874368
STEP 44 ================================
prereg loss 0.010595312 reg_l1 13.380476 reg_l2 9.939049
loss 2.6866906
STEP 45 ================================
prereg loss 0.01040525 reg_l1 13.377763 reg_l2 9.937797
loss 2.685958
STEP 46 ================================
prereg loss 0.010435962 reg_l1 13.374925 reg_l2 9.936409
loss 2.685421
STEP 47 ================================
prereg loss 0.01061553 reg_l1 13.37189 reg_l2 9.934761
loss 2.6849937
STEP 48 ================================
prereg loss 0.010703734 reg_l1 13.3686905 reg_l2 9.932888
loss 2.684442
STEP 49 ================================
prereg loss 0.010586449 reg_l1 13.365445 reg_l2 9.930936
loss 2.6836755
STEP 50 ================================
prereg loss 0.010376463 reg_l1 13.362271 reg_l2 9.929079
loss 2.6828308
STEP 51 ================================
prereg loss 0.0102360835 reg_l1 13.359212 reg_l2 9.927391
loss 2.6820784
STEP 52 ================================
prereg loss 0.010193619 reg_l1 13.356214 reg_l2 9.925822
loss 2.6814363
STEP 53 ================================
prereg loss 0.010166951 reg_l1 13.35316 reg_l2 9.924223
loss 2.680799
STEP 54 ================================
prereg loss 0.010124282 reg_l1 13.349949 reg_l2 9.92246
loss 2.680114
STEP 55 ================================
prereg loss 0.01013532 reg_l1 13.346561 reg_l2 9.920498
loss 2.6794477
STEP 56 ================================
prereg loss 0.010246972 reg_l1 13.34307 reg_l2 9.918425
loss 2.6788611
STEP 57 ================================
prereg loss 0.010381692 reg_l1 13.3396 reg_l2 9.916386
loss 2.6783016
STEP 58 ================================
prereg loss 0.010414318 reg_l1 13.336238 reg_l2 9.914493
loss 2.677662
STEP 59 ================================
prereg loss 0.010323114 reg_l1 13.332993 reg_l2 9.912759
loss 2.6769216
STEP 60 ================================
prereg loss 0.010203968 reg_l1 13.329792 reg_l2 9.911094
loss 2.6761625
STEP 61 ================================
prereg loss 0.010146026 reg_l1 13.326542 reg_l2 9.909382
loss 2.6754546
STEP 62 ================================
prereg loss 0.010148572 reg_l1 13.323187 reg_l2 9.9075575
loss 2.6747859
STEP 63 ================================
prereg loss 0.01016668 reg_l1 13.319739 reg_l2 9.905649
loss 2.6741145
STEP 64 ================================
prereg loss 0.010186771 reg_l1 13.31627 reg_l2 9.903747
loss 2.6734407
STEP 65 ================================
prereg loss 0.0102229295 reg_l1 13.312852 reg_l2 9.901941
loss 2.6727934
STEP 66 ================================
prereg loss 0.010264792 reg_l1 13.309513 reg_l2 9.900262
loss 2.6721675
STEP 67 ================================
prereg loss 0.010273797 reg_l1 13.306222 reg_l2 9.898657
loss 2.671518
STEP 68 ================================
prereg loss 0.010233064 reg_l1 13.302922 reg_l2 9.897043
loss 2.6708176
STEP 69 ================================
prereg loss 0.010174628 reg_l1 13.299563 reg_l2 9.895365
loss 2.6700873
STEP 70 ================================
prereg loss 0.010139174 reg_l1 13.296152 reg_l2 9.893623
loss 2.6693697
STEP 71 ================================
prereg loss 0.0101304315 reg_l1 13.292721 reg_l2 9.891884
loss 2.6686747
STEP 72 ================================
prereg loss 0.010125106 reg_l1 13.289308 reg_l2 9.890202
loss 2.6679866
STEP 73 ================================
prereg loss 0.0101141445 reg_l1 13.285928 reg_l2 9.888596
loss 2.6672997
STEP 74 ================================
prereg loss 0.01011298 reg_l1 13.282557 reg_l2 9.887032
loss 2.6666243
STEP 75 ================================
prereg loss 0.010127727 reg_l1 13.279155 reg_l2 9.885447
loss 2.665959
STEP 76 ================================
prereg loss 0.010140551 reg_l1 13.27571 reg_l2 9.883809
loss 2.6652827
STEP 77 ================================
prereg loss 0.010132681 reg_l1 13.272229 reg_l2 9.882137
loss 2.6645787
STEP 78 ================================
prereg loss 0.010106169 reg_l1 13.268748 reg_l2 9.880476
loss 2.6638558
STEP 79 ================================
prereg loss 0.010075666 reg_l1 13.265291 reg_l2 9.878869
loss 2.6631339
STEP 80 ================================
prereg loss 0.010049376 reg_l1 13.261863 reg_l2 9.877323
loss 2.662422
STEP 81 ================================
prereg loss 0.010029847 reg_l1 13.258432 reg_l2 9.8758
loss 2.6617162
STEP 82 ================================
prereg loss 0.010022205 reg_l1 13.254968 reg_l2 9.874259
loss 2.6610157
STEP 83 ================================
prereg loss 0.010030669 reg_l1 13.251457 reg_l2 9.872679
loss 2.6603222
STEP 84 ================================
prereg loss 0.010046734 reg_l1 13.247919 reg_l2 9.871079
loss 2.6596305
STEP 85 ================================
prereg loss 0.010051658 reg_l1 13.244386 reg_l2 9.869503
loss 2.6589289
STEP 86 ================================
prereg loss 0.010033532 reg_l1 13.240883 reg_l2 9.86798
loss 2.6582103
STEP 87 ================================
prereg loss 0.010000473 reg_l1 13.237404 reg_l2 9.866507
loss 2.6574812
STEP 88 ================================
prereg loss 0.009968841 reg_l1 13.233927 reg_l2 9.86505
loss 2.6567543
STEP 89 ================================
prereg loss 0.009948627 reg_l1 13.230419 reg_l2 9.863581
loss 2.6560326
STEP 90 ================================
prereg loss 0.0099402545 reg_l1 13.226877 reg_l2 9.862084
loss 2.6553156
STEP 91 ================================
prereg loss 0.009940095 reg_l1 13.223311 reg_l2 9.86058
loss 2.6546025
STEP 92 ================================
prereg loss 0.009941889 reg_l1 13.21975 reg_l2 9.859105
loss 2.653892
STEP 93 ================================
prereg loss 0.009937043 reg_l1 13.2162075 reg_l2 9.857671
loss 2.6531787
STEP 94 ================================
prereg loss 0.00992057 reg_l1 13.212676 reg_l2 9.85627
loss 2.6524558
STEP 95 ================================
prereg loss 0.009896271 reg_l1 13.20914 reg_l2 9.854876
loss 2.6517243
STEP 96 ================================
prereg loss 0.009872509 reg_l1 13.205584 reg_l2 9.85347
loss 2.6509893
STEP 97 ================================
prereg loss 0.0098553095 reg_l1 13.2020035 reg_l2 9.852052
loss 2.650256
STEP 98 ================================
prereg loss 0.009843644 reg_l1 13.198408 reg_l2 9.850639
loss 2.6495252
STEP 99 ================================
prereg loss 0.009834419 reg_l1 13.194816 reg_l2 9.84925
loss 2.6487978
STEP 100 ================================
prereg loss 0.009826124 reg_l1 13.191227 reg_l2 9.847887
loss 2.6480715
2022-06-27T15:14:58.167

julia> # this training had a rather difficult start, let's do another 100 steps

julia> steps!(100)
2022-06-27T15:15:25.064
STEP 1 ================================
prereg loss 0.009818266 reg_l1 13.187632 reg_l2 9.846536
loss 2.6473446
STEP 2 ================================
prereg loss 0.009808704 reg_l1 13.18402 reg_l2 9.84518
loss 2.646613
STEP 3 ================================
prereg loss 0.009796834 reg_l1 13.18039 reg_l2 9.843813
loss 2.645875
STEP 4 ================================
prereg loss 0.009783041 reg_l1 13.176749 reg_l2 9.842445
loss 2.645133
STEP 5 ================================
prereg loss 0.009769212 reg_l1 13.173101 reg_l2 9.841093
loss 2.6443896
STEP 6 ================================
prereg loss 0.009756445 reg_l1 13.1694565 reg_l2 9.839762
loss 2.6436477
STEP 7 ================================
prereg loss 0.009746789 reg_l1 13.165803 reg_l2 9.838442
loss 2.6429074
STEP 8 ================================
prereg loss 0.009740962 reg_l1 13.162137 reg_l2 9.837125
loss 2.6421685
STEP 9 ================================
prereg loss 0.009738674 reg_l1 13.158449 reg_l2 9.8358
loss 2.6414285
STEP 10 ================================
prereg loss 0.009736129 reg_l1 13.154749 reg_l2 9.8344755
loss 2.6406858
STEP 11 ================================
prereg loss 0.009729604 reg_l1 13.151048 reg_l2 9.833163
loss 2.6399393
STEP 12 ================================
prereg loss 0.009718557 reg_l1 13.1473465 reg_l2 9.831872
loss 2.639188
STEP 13 ================================
prereg loss 0.009705549 reg_l1 13.143648 reg_l2 9.830603
loss 2.6384351
STEP 14 ================================
prereg loss 0.009694523 reg_l1 13.139944 reg_l2 9.829343
loss 2.6376834
STEP 15 ================================
prereg loss 0.009687356 reg_l1 13.136223 reg_l2 9.828083
loss 2.6369321
STEP 16 ================================
prereg loss 0.009683388 reg_l1 13.132486 reg_l2 9.826822
loss 2.6361806
STEP 17 ================================
prereg loss 0.0096801035 reg_l1 13.128743 reg_l2 9.825573
loss 2.6354287
STEP 18 ================================
prereg loss 0.009674228 reg_l1 13.125006 reg_l2 9.824343
loss 2.6346755
STEP 19 ================================
prereg loss 0.009664303 reg_l1 13.121266 reg_l2 9.823133
loss 2.6339176
STEP 20 ================================
prereg loss 0.009652209 reg_l1 13.117525 reg_l2 9.821937
loss 2.6331573
STEP 21 ================================
prereg loss 0.0096404515 reg_l1 13.113771 reg_l2 9.820745
loss 2.6323948
STEP 22 ================================
prereg loss 0.009631215 reg_l1 13.110006 reg_l2 9.819555
loss 2.6316326
STEP 23 ================================
prereg loss 0.009623789 reg_l1 13.106234 reg_l2 9.818375
loss 2.6308706
STEP 24 ================================
prereg loss 0.009617027 reg_l1 13.102454 reg_l2 9.817204
loss 2.6301079
STEP 25 ================================
prereg loss 0.009609741 reg_l1 13.098677 reg_l2 9.816053
loss 2.6293452
STEP 26 ================================
prereg loss 0.009601106 reg_l1 13.09489 reg_l2 9.814914
loss 2.6285791
STEP 27 ================================
prereg loss 0.009591738 reg_l1 13.091099 reg_l2 9.813781
loss 2.6278117
STEP 28 ================================
prereg loss 0.009581822 reg_l1 13.087295 reg_l2 9.812651
loss 2.6270409
STEP 29 ================================
prereg loss 0.009572496 reg_l1 13.083483 reg_l2 9.811528
loss 2.626269
STEP 30 ================================
prereg loss 0.009563245 reg_l1 13.079668 reg_l2 9.810422
loss 2.6254969
STEP 31 ================================
prereg loss 0.009554196 reg_l1 13.075848 reg_l2 9.809326
loss 2.6247237
STEP 32 ================================
prereg loss 0.009545979 reg_l1 13.0720215 reg_l2 9.808243
loss 2.6239505
STEP 33 ================================
prereg loss 0.009538642 reg_l1 13.068186 reg_l2 9.807165
loss 2.6231759
STEP 34 ================================
prereg loss 0.009531111 reg_l1 13.064345 reg_l2 9.806093
loss 2.6224
STEP 35 ================================
prereg loss 0.009522579 reg_l1 13.060491 reg_l2 9.805031
loss 2.621621
STEP 36 ================================
prereg loss 0.009513323 reg_l1 13.056636 reg_l2 9.803981
loss 2.6208405
STEP 37 ================================
prereg loss 0.009503535 reg_l1 13.052777 reg_l2 9.802943
loss 2.620059
STEP 38 ================================
prereg loss 0.0094938455 reg_l1 13.048912 reg_l2 9.801918
loss 2.6192763
STEP 39 ================================
prereg loss 0.009485621 reg_l1 13.045038 reg_l2 9.800898
loss 2.6184933
STEP 40 ================================
prereg loss 0.009478169 reg_l1 13.041156 reg_l2 9.799886
loss 2.6177094
STEP 41 ================================
prereg loss 0.009470843 reg_l1 13.037267 reg_l2 9.798881
loss 2.6169243
STEP 42 ================================
prereg loss 0.0094629815 reg_l1 13.03337 reg_l2 9.797888
loss 2.616137
STEP 43 ================================
prereg loss 0.0094536245 reg_l1 13.029471 reg_l2 9.796907
loss 2.6153479
STEP 44 ================================
prereg loss 0.009444241 reg_l1 13.025567 reg_l2 9.7959385
loss 2.6145577
STEP 45 ================================
prereg loss 0.009434873 reg_l1 13.021655 reg_l2 9.794978
loss 2.613766
STEP 46 ================================
prereg loss 0.009426559 reg_l1 13.017734 reg_l2 9.794024
loss 2.6129735
STEP 47 ================================
prereg loss 0.009418813 reg_l1 13.013806 reg_l2 9.793079
loss 2.61218
STEP 48 ================================
prereg loss 0.009411573 reg_l1 13.0098715 reg_l2 9.792143
loss 2.6113858
STEP 49 ================================
prereg loss 0.009403654 reg_l1 13.005932 reg_l2 9.791218
loss 2.6105902
STEP 50 ================================
prereg loss 0.009395219 reg_l1 13.001986 reg_l2 9.790304
loss 2.6097922
STEP 51 ================================
prereg loss 0.009386303 reg_l1 12.998032 reg_l2 9.789401
loss 2.6089926
STEP 52 ================================
prereg loss 0.009378013 reg_l1 12.994073 reg_l2 9.788502
loss 2.6081927
STEP 53 ================================
prereg loss 0.009369834 reg_l1 12.990105 reg_l2 9.787613
loss 2.6073909
STEP 54 ================================
prereg loss 0.009362308 reg_l1 12.986129 reg_l2 9.786735
loss 2.6065881
STEP 55 ================================
prereg loss 0.009354701 reg_l1 12.982148 reg_l2 9.785865
loss 2.6057842
STEP 56 ================================
prereg loss 0.0093468595 reg_l1 12.978163 reg_l2 9.7850065
loss 2.6049795
STEP 57 ================================
prereg loss 0.00933925 reg_l1 12.97417 reg_l2 9.784158
loss 2.6041734
STEP 58 ================================
prereg loss 0.009331489 reg_l1 12.97017 reg_l2 9.783315
loss 2.6033654
STEP 59 ================================
prereg loss 0.009323555 reg_l1 12.966162 reg_l2 9.782484
loss 2.602556
STEP 60 ================================
prereg loss 0.00931564 reg_l1 12.96215 reg_l2 9.781661
loss 2.6017456
STEP 61 ================================
prereg loss 0.009307423 reg_l1 12.958131 reg_l2 9.780849
loss 2.6009336
STEP 62 ================================
prereg loss 0.009299731 reg_l1 12.954107 reg_l2 9.78005
loss 2.6001213
STEP 63 ================================
prereg loss 0.009292141 reg_l1 12.950073 reg_l2 9.779258
loss 2.5993068
STEP 64 ================================
prereg loss 0.009284754 reg_l1 12.946035 reg_l2 9.778473
loss 2.598492
STEP 65 ================================
prereg loss 0.009276873 reg_l1 12.94199 reg_l2 9.777698
loss 2.5976748
STEP 66 ================================
prereg loss 0.009268793 reg_l1 12.937939 reg_l2 9.776936
loss 2.5968566
STEP 67 ================================
prereg loss 0.009260489 reg_l1 12.933883 reg_l2 9.776183
loss 2.596037
STEP 68 ================================
prereg loss 0.009252165 reg_l1 12.92982 reg_l2 9.77544
loss 2.595216
STEP 69 ================================
prereg loss 0.009244452 reg_l1 12.925749 reg_l2 9.774705
loss 2.5943942
STEP 70 ================================
prereg loss 0.009237078 reg_l1 12.921675 reg_l2 9.77398
loss 2.5935721
STEP 71 ================================
prereg loss 0.009229535 reg_l1 12.91759 reg_l2 9.773263
loss 2.5927474
STEP 72 ================================
prereg loss 0.009221362 reg_l1 12.913501 reg_l2 9.772558
loss 2.5919216
STEP 73 ================================
prereg loss 0.0092131505 reg_l1 12.909406 reg_l2 9.771864
loss 2.5910945
STEP 74 ================================
prereg loss 0.009205227 reg_l1 12.905304 reg_l2 9.771175
loss 2.5902662
STEP 75 ================================
prereg loss 0.009197661 reg_l1 12.901195 reg_l2 9.770494
loss 2.5894368
STEP 76 ================================
prereg loss 0.00918993 reg_l1 12.897078 reg_l2 9.769826
loss 2.5886054
STEP 77 ================================
prereg loss 0.009182161 reg_l1 12.892958 reg_l2 9.769167
loss 2.5877738
STEP 78 ================================
prereg loss 0.00917458 reg_l1 12.888829 reg_l2 9.7685175
loss 2.5869405
STEP 79 ================================
prereg loss 0.009166461 reg_l1 12.884696 reg_l2 9.7678795
loss 2.5861058
STEP 80 ================================
prereg loss 0.009158741 reg_l1 12.880555 reg_l2 9.767248
loss 2.58527
STEP 81 ================================
prereg loss 0.009151069 reg_l1 12.876409 reg_l2 9.7666235
loss 2.5844328
STEP 82 ================================
prereg loss 0.009143259 reg_l1 12.872256 reg_l2 9.766012
loss 2.5835946
STEP 83 ================================
prereg loss 0.009135503 reg_l1 12.8680935 reg_l2 9.765409
loss 2.5827541
STEP 84 ================================
prereg loss 0.0091277575 reg_l1 12.863929 reg_l2 9.764818
loss 2.5819137
STEP 85 ================================
prereg loss 0.009120229 reg_l1 12.8597555 reg_l2 9.764234
loss 2.5810714
STEP 86 ================================
prereg loss 0.00911275 reg_l1 12.855578 reg_l2 9.763659
loss 2.5802286
STEP 87 ================================
prereg loss 0.009105261 reg_l1 12.85139 reg_l2 9.763092
loss 2.5793831
STEP 88 ================================
prereg loss 0.009097404 reg_l1 12.8471985 reg_l2 9.762536
loss 2.578537
STEP 89 ================================
prereg loss 0.009089472 reg_l1 12.843001 reg_l2 9.761991
loss 2.57769
STEP 90 ================================
prereg loss 0.009081765 reg_l1 12.838797 reg_l2 9.761456
loss 2.576841
STEP 91 ================================
prereg loss 0.00907447 reg_l1 12.834586 reg_l2 9.760926
loss 2.5759916
STEP 92 ================================
prereg loss 0.009067095 reg_l1 12.830368 reg_l2 9.7604065
loss 2.5751407
STEP 93 ================================
prereg loss 0.009059264 reg_l1 12.826143 reg_l2 9.759897
loss 2.574288
STEP 94 ================================
prereg loss 0.009051919 reg_l1 12.821915 reg_l2 9.759398
loss 2.5734348
STEP 95 ================================
prereg loss 0.009043803 reg_l1 12.817678 reg_l2 9.758909
loss 2.5725794
STEP 96 ================================
prereg loss 0.0090360865 reg_l1 12.8134365 reg_l2 9.75843
loss 2.5717235
STEP 97 ================================
prereg loss 0.009029002 reg_l1 12.809185 reg_l2 9.757956
loss 2.5708659
STEP 98 ================================
prereg loss 0.009021577 reg_l1 12.804932 reg_l2 9.757493
loss 2.5700078
STEP 99 ================================
prereg loss 0.009014072 reg_l1 12.800669 reg_l2 9.75704
loss 2.5691478
STEP 100 ================================
prereg loss 0.009006554 reg_l1 12.7964 reg_l2 9.7565975
loss 2.5682867
2022-06-27T15:19:39.745

julia> # this is going well, let's do 300 more steps for the total of 500

julia> steps!(300)
2022-06-27T15:20:09.204
STEP 1 ================================
prereg loss 0.008998883 reg_l1 12.792127 reg_l2 9.756164
loss 2.5674243
STEP 2 ================================
prereg loss 0.008991209 reg_l1 12.787847 reg_l2 9.755738
loss 2.5665605
STEP 3 ================================
prereg loss 0.008983809 reg_l1 12.783558 reg_l2 9.755323
loss 2.5656955
STEP 4 ================================
prereg loss 0.008976383 reg_l1 12.779265 reg_l2 9.754915
loss 2.5648296
STEP 5 ================================
prereg loss 0.008968886 reg_l1 12.774966 reg_l2 9.7545185
loss 2.5639622
STEP 6 ================================
prereg loss 0.008961306 reg_l1 12.770661 reg_l2 9.754131
loss 2.5630934
STEP 7 ================================
prereg loss 0.008953679 reg_l1 12.76635 reg_l2 9.753756
loss 2.5622237
STEP 8 ================================
prereg loss 0.008946477 reg_l1 12.762032 reg_l2 9.753385
loss 2.5613527
STEP 9 ================================
prereg loss 0.008938729 reg_l1 12.7577095 reg_l2 9.753024
loss 2.5604808
STEP 10 ================================
prereg loss 0.008931148 reg_l1 12.753378 reg_l2 9.752676
loss 2.5596068
STEP 11 ================================
prereg loss 0.008923635 reg_l1 12.749041 reg_l2 9.752336
loss 2.5587318
STEP 12 ================================
prereg loss 0.008916205 reg_l1 12.744697 reg_l2 9.752006
loss 2.5578556
STEP 13 ================================
prereg loss 0.00890878 reg_l1 12.74035 reg_l2 9.751683
loss 2.5569787
STEP 14 ================================
prereg loss 0.008901446 reg_l1 12.735993 reg_l2 9.751369
loss 2.5561001
STEP 15 ================================
prereg loss 0.008893883 reg_l1 12.733439 reg_l2 9.751066
loss 2.5555818
STEP 16 ================================
prereg loss 0.008886387 reg_l1 12.73087 reg_l2 9.7507715
loss 2.5550604
STEP 17 ================================
prereg loss 0.008878932 reg_l1 12.728119 reg_l2 9.750484
loss 2.5545027
STEP 18 ================================
prereg loss 0.008871486 reg_l1 12.725196 reg_l2 9.750206
loss 2.5539107
STEP 19 ================================
prereg loss 0.0088643 reg_l1 12.722122 reg_l2 9.749933
loss 2.553289
STEP 20 ================================
prereg loss 0.008856703 reg_l1 12.718912 reg_l2 9.749665
loss 2.5526392
STEP 21 ================================
prereg loss 0.0088492315 reg_l1 12.715575 reg_l2 9.749407
loss 2.5519643
STEP 22 ================================
prereg loss 0.008841744 reg_l1 12.712126 reg_l2 9.749154
loss 2.551267
STEP 23 ================================
prereg loss 0.008834568 reg_l1 12.708576 reg_l2 9.748908
loss 2.55055
STEP 24 ================================
prereg loss 0.008827092 reg_l1 12.704934 reg_l2 9.748668
loss 2.549814
STEP 25 ================================
prereg loss 0.008819775 reg_l1 12.701209 reg_l2 9.748436
loss 2.5490618
STEP 26 ================================
prereg loss 0.008812467 reg_l1 12.697407 reg_l2 9.748212
loss 2.5482938
STEP 27 ================================
prereg loss 0.008805063 reg_l1 12.693535 reg_l2 9.747993
loss 2.547512
STEP 28 ================================
prereg loss 0.008797877 reg_l1 12.689602 reg_l2 9.747783
loss 2.5467184
STEP 29 ================================
prereg loss 0.00879051 reg_l1 12.685614 reg_l2 9.747581
loss 2.5459132
STEP 30 ================================
prereg loss 0.00878312 reg_l1 12.68157 reg_l2 9.747386
loss 2.545097
STEP 31 ================================
prereg loss 0.008775706 reg_l1 12.678344 reg_l2 9.747201
loss 2.5444446
STEP 32 ================================
prereg loss 0.008768405 reg_l1 12.675344 reg_l2 9.747023
loss 2.5438373
STEP 33 ================================
prereg loss 0.008761089 reg_l1 12.672192 reg_l2 9.74685
loss 2.5431995
STEP 34 ================================
prereg loss 0.008753899 reg_l1 12.668903 reg_l2 9.746687
loss 2.5425348
STEP 35 ================================
prereg loss 0.008746465 reg_l1 12.665488 reg_l2 9.7465315
loss 2.5418441
STEP 36 ================================
prereg loss 0.008738893 reg_l1 12.66196 reg_l2 9.746381
loss 2.541131
STEP 37 ================================
prereg loss 0.00873162 reg_l1 12.65833 reg_l2 9.74624
loss 2.5403976
STEP 38 ================================
prereg loss 0.008724432 reg_l1 12.654606 reg_l2 9.746104
loss 2.5396457
STEP 39 ================================
prereg loss 0.008717263 reg_l1 12.6508 reg_l2 9.7459755
loss 2.5388772
STEP 40 ================================
prereg loss 0.008709861 reg_l1 12.64892 reg_l2 9.745857
loss 2.5384939
STEP 41 ================================
prereg loss 0.008640137 reg_l1 12.647162 reg_l2 9.745744
loss 2.5380726
STEP 42 ================================
prereg loss 0.008647132 reg_l1 12.647531 reg_l2 9.744991
loss 2.5381534
STEP 43 ================================
prereg loss 0.00871768 reg_l1 12.64701 reg_l2 9.743782
loss 2.5381198
STEP 44 ================================
prereg loss 0.008755087 reg_l1 12.646168 reg_l2 9.742884
loss 2.5379887
STEP 45 ================================
prereg loss 0.00869374 reg_l1 12.645272 reg_l2 9.7426195
loss 2.537748
STEP 46 ================================
prereg loss 0.008828824 reg_l1 12.644055 reg_l2 9.742512
loss 2.53764
STEP 47 ================================
prereg loss 0.008794247 reg_l1 12.642327 reg_l2 9.742201
loss 2.5372598
STEP 48 ================================
prereg loss 0.009019785 reg_l1 12.640714 reg_l2 9.741758
loss 2.5371625
STEP 49 ================================
prereg loss 0.008781274 reg_l1 12.639025 reg_l2 9.7415285
loss 2.5365863
STEP 50 ================================
prereg loss 0.008464258 reg_l1 12.637355 reg_l2 9.741789
loss 2.5359354
STEP 51 ================================
prereg loss 0.008290239 reg_l1 12.635568 reg_l2 9.74233
loss 2.535404
STEP 52 ================================
prereg loss 0.008060601 reg_l1 12.633298 reg_l2 9.7425785
loss 2.5347204
STEP 53 ================================
prereg loss 0.008056543 reg_l1 12.631244 reg_l2 9.742367
loss 2.5343053
STEP 54 ================================
prereg loss 0.008034548 reg_l1 12.631166 reg_l2 9.74204
loss 2.534268
STEP 55 ================================
prereg loss 0.007930889 reg_l1 12.630844 reg_l2 9.74174
loss 2.5340998
STEP 56 ================================
prereg loss 0.00783586 reg_l1 12.630136 reg_l2 9.741507
loss 2.533863
STEP 57 ================================
prereg loss 0.0078036212 reg_l1 12.62898 reg_l2 9.741218
loss 2.5335996
STEP 58 ================================
prereg loss 0.007804215 reg_l1 12.62731 reg_l2 9.740725
loss 2.533266
STEP 59 ================================
prereg loss 0.00786058 reg_l1 12.625193 reg_l2 9.740034
loss 2.5328991
STEP 60 ================================
prereg loss 0.00795976 reg_l1 12.623131 reg_l2 9.739284
loss 2.532586
STEP 61 ================================
prereg loss 0.008019814 reg_l1 12.62079 reg_l2 9.738566
loss 2.532178
STEP 62 ================================
prereg loss 0.008048946 reg_l1 12.618708 reg_l2 9.737837
loss 2.5317905
STEP 63 ================================
prereg loss 0.008104039 reg_l1 12.617755 reg_l2 9.736975
loss 2.531655
STEP 64 ================================
prereg loss 0.008351618 reg_l1 12.61636 reg_l2 9.735765
loss 2.5316236
STEP 65 ================================
prereg loss 0.008409713 reg_l1 12.614572 reg_l2 9.734636
loss 2.5313241
STEP 66 ================================
prereg loss 0.008566622 reg_l1 12.612572 reg_l2 9.733766
loss 2.531081
STEP 67 ================================
prereg loss 0.008599583 reg_l1 12.61072 reg_l2 9.733213
loss 2.5307436
STEP 68 ================================
prereg loss 0.008568991 reg_l1 12.609261 reg_l2 9.732939
loss 2.5304213
STEP 69 ================================
prereg loss 0.008501255 reg_l1 12.607743 reg_l2 9.732528
loss 2.53005
STEP 70 ================================
prereg loss 0.008542583 reg_l1 12.606342 reg_l2 9.732062
loss 2.5298111
STEP 71 ================================
prereg loss 0.008439772 reg_l1 12.60538 reg_l2 9.731875
loss 2.5295157
STEP 72 ================================
prereg loss 0.008411362 reg_l1 12.6044 reg_l2 9.731927
loss 2.5292914
STEP 73 ================================
prereg loss 0.008332843 reg_l1 12.603238 reg_l2 9.732144
loss 2.5289805
STEP 74 ================================
prereg loss 0.00824221 reg_l1 12.602138 reg_l2 9.732295
loss 2.5286696
STEP 75 ================================
prereg loss 0.008197306 reg_l1 12.600898 reg_l2 9.732227
loss 2.5283768
STEP 76 ================================
prereg loss 0.008184391 reg_l1 12.599543 reg_l2 9.73203
loss 2.528093
STEP 77 ================================
prereg loss 0.008177983 reg_l1 12.598615 reg_l2 9.731877
loss 2.527901
STEP 78 ================================
prereg loss 0.008169127 reg_l1 12.597483 reg_l2 9.73181
loss 2.5276659
STEP 79 ================================
prereg loss 0.008172479 reg_l1 12.596137 reg_l2 9.731615
loss 2.5274
STEP 80 ================================
prereg loss 0.008234221 reg_l1 12.594778 reg_l2 9.731195
loss 2.52719
STEP 81 ================================
prereg loss 0.008308636 reg_l1 12.5931225 reg_l2 9.730909
loss 2.5269332
STEP 82 ================================
prereg loss 0.008286591 reg_l1 12.591618 reg_l2 9.73076
loss 2.5266101
STEP 83 ================================
prereg loss 0.00821129 reg_l1 12.591084 reg_l2 9.730689
loss 2.5264282
STEP 84 ================================
prereg loss 0.008154737 reg_l1 12.59021 reg_l2 9.730639
loss 2.5261967
STEP 85 ================================
prereg loss 0.008150116 reg_l1 12.588869 reg_l2 9.730461
loss 2.525924
STEP 86 ================================
prereg loss 0.008120406 reg_l1 12.587742 reg_l2 9.730397
loss 2.5256686
STEP 87 ================================
prereg loss 0.008078191 reg_l1 12.58652 reg_l2 9.730508
loss 2.5253823
STEP 88 ================================
prereg loss 0.008056189 reg_l1 12.585182 reg_l2 9.730555
loss 2.5250926
STEP 89 ================================
prereg loss 0.008075423 reg_l1 12.583327 reg_l2 9.730354
loss 2.524741
STEP 90 ================================
prereg loss 0.008128187 reg_l1 12.58235 reg_l2 9.730028
loss 2.5245981
STEP 91 ================================
prereg loss 0.008303542 reg_l1 12.581593 reg_l2 9.730012
loss 2.5246222
STEP 92 ================================
prereg loss 0.008239319 reg_l1 12.580727 reg_l2 9.730198
loss 2.5243847
STEP 93 ================================
prereg loss 0.008019254 reg_l1 12.579531 reg_l2 9.730386
loss 2.5239253
STEP 94 ================================
prereg loss 0.008171923 reg_l1 12.578011 reg_l2 9.730351
loss 2.5237741
STEP 95 ================================
prereg loss 0.008057986 reg_l1 12.576653 reg_l2 9.730335
loss 2.5233886
STEP 96 ================================
prereg loss 0.008087762 reg_l1 12.575351 reg_l2 9.730512
loss 2.523158
STEP 97 ================================
prereg loss 0.008190899 reg_l1 12.5746 reg_l2 9.730793
loss 2.5231109
STEP 98 ================================
prereg loss 0.007974525 reg_l1 12.573768 reg_l2 9.730976
loss 2.5227282
STEP 99 ================================
prereg loss 0.007866359 reg_l1 12.572491 reg_l2 9.730978
loss 2.5223646
STEP 100 ================================
prereg loss 0.0079426 reg_l1 12.571111 reg_l2 9.730924
loss 2.5221648
STEP 101 ================================
prereg loss 0.0078993095 reg_l1 12.569785 reg_l2 9.730919
loss 2.5218563
STEP 102 ================================
prereg loss 0.00793368 reg_l1 12.568587 reg_l2 9.730823
loss 2.521651
STEP 103 ================================
prereg loss 0.008112415 reg_l1 12.567189 reg_l2 9.730531
loss 2.5215504
STEP 104 ================================
prereg loss 0.008129709 reg_l1 12.565666 reg_l2 9.730174
loss 2.521263
STEP 105 ================================
prereg loss 0.008062027 reg_l1 12.564272 reg_l2 9.729919
loss 2.5209165
STEP 106 ================================
prereg loss 0.00829986 reg_l1 12.562799 reg_l2 9.729815
loss 2.5208597
STEP 107 ================================
prereg loss 0.008209284 reg_l1 12.561885 reg_l2 9.729956
loss 2.5205863
STEP 108 ================================
prereg loss 0.0082837455 reg_l1 12.5610485 reg_l2 9.730164
loss 2.5204935
STEP 109 ================================
prereg loss 0.008362434 reg_l1 12.559866 reg_l2 9.730309
loss 2.5203357
STEP 110 ================================
prereg loss 0.008084134 reg_l1 12.558751 reg_l2 9.730594
loss 2.5198343
STEP 111 ================================
prereg loss 0.008025561 reg_l1 12.557641 reg_l2 9.731005
loss 2.519554
STEP 112 ================================
prereg loss 0.007936236 reg_l1 12.556856 reg_l2 9.731457
loss 2.5193076
STEP 113 ================================
prereg loss 0.0079225125 reg_l1 12.555876 reg_l2 9.731726
loss 2.5190976
STEP 114 ================================
prereg loss 0.0079905605 reg_l1 12.554681 reg_l2 9.73196
loss 2.5189269
STEP 115 ================================
prereg loss 0.007966018 reg_l1 12.553766 reg_l2 9.732251
loss 2.5187194
STEP 116 ================================
prereg loss 0.007867103 reg_l1 12.552814 reg_l2 9.732548
loss 2.5184298
STEP 117 ================================
prereg loss 0.007851596 reg_l1 12.551722 reg_l2 9.732726
loss 2.5181959
STEP 118 ================================
prereg loss 0.007888553 reg_l1 12.550268 reg_l2 9.732769
loss 2.5179422
STEP 119 ================================
prereg loss 0.007958185 reg_l1 12.549243 reg_l2 9.73295
loss 2.5178068
STEP 120 ================================
prereg loss 0.00808484 reg_l1 12.548326 reg_l2 9.733167
loss 2.51775
STEP 121 ================================
prereg loss 0.007978031 reg_l1 12.547099 reg_l2 9.733363
loss 2.5173979
STEP 122 ================================
prereg loss 0.007954469 reg_l1 12.545743 reg_l2 9.733475
loss 2.517103
STEP 123 ================================
prereg loss 0.007906944 reg_l1 12.544685 reg_l2 9.733716
loss 2.516844
STEP 124 ================================
prereg loss 0.007927729 reg_l1 12.543708 reg_l2 9.733979
loss 2.5166693
STEP 125 ================================
prereg loss 0.0079430435 reg_l1 12.542588 reg_l2 9.734268
loss 2.516461
STEP 126 ================================
prereg loss 0.0079318965 reg_l1 12.541281 reg_l2 9.734536
loss 2.5161881
STEP 127 ================================
prereg loss 0.007976095 reg_l1 12.540365 reg_l2 9.734737
loss 2.5160491
STEP 128 ================================
prereg loss 0.008000526 reg_l1 12.539712 reg_l2 9.735126
loss 2.515943
STEP 129 ================================
prereg loss 0.0079978295 reg_l1 12.538752 reg_l2 9.735597
loss 2.515748
STEP 130 ================================
prereg loss 0.00787885 reg_l1 12.537483 reg_l2 9.736122
loss 2.5153754
STEP 131 ================================
prereg loss 0.0078997975 reg_l1 12.536481 reg_l2 9.73657
loss 2.515196
STEP 132 ================================
prereg loss 0.00782033 reg_l1 12.535993 reg_l2 9.737094
loss 2.515019
STEP 133 ================================
prereg loss 0.007847898 reg_l1 12.53516 reg_l2 9.73761
loss 2.51488
STEP 134 ================================
prereg loss 0.007810998 reg_l1 12.533998 reg_l2 9.738169
loss 2.5146105
STEP 135 ================================
prereg loss 0.0077439533 reg_l1 12.532876 reg_l2 9.7386675
loss 2.5143192
STEP 136 ================================
prereg loss 0.0077468073 reg_l1 12.531987 reg_l2 9.739017
loss 2.5141442
STEP 137 ================================
prereg loss 0.0077848276 reg_l1 12.531222 reg_l2 9.739444
loss 2.5140293
STEP 138 ================================
prereg loss 0.007842225 reg_l1 12.530101 reg_l2 9.739903
loss 2.5138626
STEP 139 ================================
prereg loss 0.00777114 reg_l1 12.528814 reg_l2 9.740393
loss 2.513534
STEP 140 ================================
prereg loss 0.0077213915 reg_l1 12.52766 reg_l2 9.740795
loss 2.5132535
STEP 141 ================================
prereg loss 0.007828018 reg_l1 12.526494 reg_l2 9.741064
loss 2.5131269
STEP 142 ================================
prereg loss 0.007790233 reg_l1 12.525459 reg_l2 9.741458
loss 2.5128822
STEP 143 ================================
prereg loss 0.007892714 reg_l1 12.524494 reg_l2 9.741913
loss 2.5127914
STEP 144 ================================
prereg loss 0.007843703 reg_l1 12.523462 reg_l2 9.742428
loss 2.5125363
STEP 145 ================================
prereg loss 0.007763702 reg_l1 12.522168 reg_l2 9.74289
loss 2.5121973
STEP 146 ================================
prereg loss 0.007931552 reg_l1 12.5213585 reg_l2 9.743257
loss 2.5122032
STEP 147 ================================
prereg loss 0.007784079 reg_l1 12.520623 reg_l2 9.743817
loss 2.5119088
STEP 148 ================================
prereg loss 0.007851572 reg_l1 12.519667 reg_l2 9.744426
loss 2.511785
STEP 149 ================================
prereg loss 0.007833736 reg_l1 12.518562 reg_l2 9.745053
loss 2.5115461
STEP 150 ================================
prereg loss 0.007768393 reg_l1 12.517448 reg_l2 9.745621
loss 2.5112581
STEP 151 ================================
prereg loss 0.0077767707 reg_l1 12.516706 reg_l2 9.746336
loss 2.5111182
STEP 152 ================================
prereg loss 0.007739719 reg_l1 12.515956 reg_l2 9.74709
loss 2.510931
STEP 153 ================================
prereg loss 0.0076610986 reg_l1 12.515103 reg_l2 9.74787
loss 2.5106819
STEP 154 ================================
prereg loss 0.0076342938 reg_l1 12.514201 reg_l2 9.74855
loss 2.5104747
STEP 155 ================================
prereg loss 0.0076671187 reg_l1 12.513032 reg_l2 9.7490835
loss 2.5102735
STEP 156 ================================
prereg loss 0.007755487 reg_l1 12.51224 reg_l2 9.749735
loss 2.5102036
STEP 157 ================================
prereg loss 0.007829842 reg_l1 12.511466 reg_l2 9.750443
loss 2.5101233
STEP 158 ================================
prereg loss 0.007664093 reg_l1 12.510609 reg_l2 9.751178
loss 2.509786
STEP 159 ================================
prereg loss 0.007625436 reg_l1 12.509685 reg_l2 9.751817
loss 2.5095623
STEP 160 ================================
prereg loss 0.00760347 reg_l1 12.510715 reg_l2 9.752488
loss 2.5097463
STEP 161 ================================
prereg loss 0.007645197 reg_l1 12.511774 reg_l2 9.753136
loss 2.51
STEP 162 ================================
prereg loss 0.0076725096 reg_l1 12.512654 reg_l2 9.753813
loss 2.5102034
STEP 163 ================================
prereg loss 0.007640016 reg_l1 12.5133 reg_l2 9.754449
loss 2.5103002
STEP 164 ================================
prereg loss 0.0076311883 reg_l1 12.513668 reg_l2 9.7549925
loss 2.510365
STEP 165 ================================
prereg loss 0.007712941 reg_l1 12.513617 reg_l2 9.755485
loss 2.5104363
STEP 166 ================================
prereg loss 0.007742342 reg_l1 12.513829 reg_l2 9.7561865
loss 2.5105083
STEP 167 ================================
prereg loss 0.007819234 reg_l1 12.514332 reg_l2 9.756933
loss 2.5106857
STEP 168 ================================
prereg loss 0.0077083693 reg_l1 12.514444 reg_l2 9.75767
loss 2.5105972
STEP 169 ================================
prereg loss 0.0077397292 reg_l1 12.514176 reg_l2 9.7583475
loss 2.510575
STEP 170 ================================
prereg loss 0.0076447222 reg_l1 12.513745 reg_l2 9.759195
loss 2.5103939
STEP 171 ================================
prereg loss 0.0076619773 reg_l1 12.513818 reg_l2 9.760063
loss 2.5104256
STEP 172 ================================
prereg loss 0.0076494496 reg_l1 12.513856 reg_l2 9.760928
loss 2.5104206
STEP 173 ================================
prereg loss 0.007608205 reg_l1 12.513681 reg_l2 9.761719
loss 2.5103445
STEP 174 ================================
prereg loss 0.0076374356 reg_l1 12.5141735 reg_l2 9.762433
loss 2.5104723
STEP 175 ================================
prereg loss 0.0076502846 reg_l1 12.514679 reg_l2 9.763339
loss 2.5105863
STEP 176 ================================
prereg loss 0.00767008 reg_l1 12.515436 reg_l2 9.764295
loss 2.5107574
STEP 177 ================================
prereg loss 0.0075639114 reg_l1 12.5159645 reg_l2 9.765227
loss 2.5107567
STEP 178 ================================
prereg loss 0.0075372607 reg_l1 12.516082 reg_l2 9.766049
loss 2.5107539
STEP 179 ================================
prereg loss 0.0075274003 reg_l1 12.516351 reg_l2 9.766958
loss 2.5107975
STEP 180 ================================
prereg loss 0.007560284 reg_l1 12.516303 reg_l2 9.767886
loss 2.5108209
STEP 181 ================================
prereg loss 0.0075181434 reg_l1 12.516444 reg_l2 9.768817
loss 2.510807
STEP 182 ================================
prereg loss 0.007470236 reg_l1 12.517045 reg_l2 9.769655
loss 2.5108793
STEP 183 ================================
prereg loss 0.007483812 reg_l1 12.517363 reg_l2 9.7703705
loss 2.5109563
STEP 184 ================================
prereg loss 0.0075399815 reg_l1 12.51749 reg_l2 9.771032
loss 2.511038
STEP 185 ================================
prereg loss 0.0076679466 reg_l1 12.517391 reg_l2 9.771899
loss 2.5111463
STEP 186 ================================
prereg loss 0.0077427053 reg_l1 12.517606 reg_l2 9.772793
loss 2.5112638
STEP 187 ================================
prereg loss 0.007549239 reg_l1 12.51847 reg_l2 9.773672
loss 2.5112433
STEP 188 ================================
prereg loss 0.007637358 reg_l1 12.518862 reg_l2 9.774494
loss 2.5114098
STEP 189 ================================
prereg loss 0.0075118467 reg_l1 12.518981 reg_l2 9.775485
loss 2.5113082
STEP 190 ================================
prereg loss 0.007656615 reg_l1 12.519373 reg_l2 9.776643
loss 2.511531
STEP 191 ================================
prereg loss 0.0077278432 reg_l1 12.520095 reg_l2 9.777819
loss 2.511747
STEP 192 ================================
prereg loss 0.0074297544 reg_l1 12.520859 reg_l2 9.778973
loss 2.5116017
STEP 193 ================================
prereg loss 0.0073736883 reg_l1 12.521187 reg_l2 9.780054
loss 2.511611
STEP 194 ================================
prereg loss 0.0074162884 reg_l1 12.5213175 reg_l2 9.781027
loss 2.51168
STEP 195 ================================
prereg loss 0.007347703 reg_l1 12.521549 reg_l2 9.781874
loss 2.5116577
STEP 196 ================================
prereg loss 0.0074915197 reg_l1 12.521884 reg_l2 9.782608
loss 2.5118685
STEP 197 ================================
prereg loss 0.0076160813 reg_l1 12.522026 reg_l2 9.783304
loss 2.5120213
STEP 198 ================================
prereg loss 0.007506223 reg_l1 12.522425 reg_l2 9.784028
loss 2.511991
STEP 199 ================================
prereg loss 0.0075615975 reg_l1 12.522683 reg_l2 9.784776
loss 2.5120983
STEP 200 ================================
prereg loss 0.0075064246 reg_l1 12.522711 reg_l2 9.78567
loss 2.5120485
STEP 201 ================================
prereg loss 0.007577705 reg_l1 12.522914 reg_l2 9.786513
loss 2.5121605
STEP 202 ================================
prereg loss 0.0076248925 reg_l1 12.52334 reg_l2 9.7873535
loss 2.5122929
STEP 203 ================================
prereg loss 0.0076566627 reg_l1 12.523794 reg_l2 9.78824
loss 2.5124154
STEP 204 ================================
prereg loss 0.007668046 reg_l1 12.524078 reg_l2 9.789421
loss 2.5124838
STEP 205 ================================
prereg loss 0.0075976616 reg_l1 12.524241 reg_l2 9.790663
loss 2.512446
STEP 206 ================================
prereg loss 0.007483386 reg_l1 12.52517 reg_l2 9.791919
loss 2.5125177
STEP 207 ================================
prereg loss 0.007458484 reg_l1 12.526136 reg_l2 9.79333
loss 2.5126858
STEP 208 ================================
prereg loss 0.007415064 reg_l1 12.52677 reg_l2 9.7947235
loss 2.512769
STEP 209 ================================
prereg loss 0.0073232865 reg_l1 12.527164 reg_l2 9.7961035
loss 2.512756
STEP 210 ================================
prereg loss 0.0072655794 reg_l1 12.527497 reg_l2 9.797393
loss 2.5127652
STEP 211 ================================
prereg loss 0.0072505665 reg_l1 12.528255 reg_l2 9.798541
loss 2.5129018
STEP 212 ================================
prereg loss 0.007261548 reg_l1 12.528814 reg_l2 9.799578
loss 2.5130243
STEP 213 ================================
prereg loss 0.0073072277 reg_l1 12.528987 reg_l2 9.800544
loss 2.5131047
STEP 214 ================================
prereg loss 0.007345834 reg_l1 12.529248 reg_l2 9.801456
loss 2.5131955
STEP 215 ================================
prereg loss 0.0073722918 reg_l1 12.529401 reg_l2 9.802309
loss 2.5132525
STEP 216 ================================
prereg loss 0.007501655 reg_l1 12.5299425 reg_l2 9.803351
loss 2.5134902
STEP 217 ================================
prereg loss 0.0074795685 reg_l1 12.530326 reg_l2 9.804447
loss 2.5135448
STEP 218 ================================
prereg loss 0.0074124206 reg_l1 12.530636 reg_l2 9.805573
loss 2.5135396
STEP 219 ================================
prereg loss 0.0073712426 reg_l1 12.53142 reg_l2 9.806875
loss 2.5136552
STEP 220 ================================
prereg loss 0.0073573245 reg_l1 12.531786 reg_l2 9.808147
loss 2.5137146
STEP 221 ================================
prereg loss 0.0073126014 reg_l1 12.532265 reg_l2 9.809428
loss 2.5137656
STEP 222 ================================
prereg loss 0.007283 reg_l1 12.532955 reg_l2 9.810687
loss 2.513874
STEP 223 ================================
prereg loss 0.007279838 reg_l1 12.533623 reg_l2 9.811862
loss 2.5140045
STEP 224 ================================
prereg loss 0.007298718 reg_l1 12.533987 reg_l2 9.812962
loss 2.5140963
STEP 225 ================================
prereg loss 0.007335556 reg_l1 12.533964 reg_l2 9.81403
loss 2.5141284
STEP 226 ================================
prereg loss 0.007425813 reg_l1 12.5342865 reg_l2 9.815087
loss 2.5142832
STEP 227 ================================
prereg loss 0.007535238 reg_l1 12.53512 reg_l2 9.816392
loss 2.5145593
STEP 228 ================================
prereg loss 0.007447459 reg_l1 12.5357485 reg_l2 9.817724
loss 2.5145972
STEP 229 ================================
prereg loss 0.007364361 reg_l1 12.536012 reg_l2 9.819063
loss 2.5145667
STEP 230 ================================
prereg loss 0.0072415643 reg_l1 12.536504 reg_l2 9.820574
loss 2.5145423
STEP 231 ================================
prereg loss 0.0072233765 reg_l1 12.537569 reg_l2 9.822015
loss 2.5147371
STEP 232 ================================
prereg loss 0.007206875 reg_l1 12.538363 reg_l2 9.823401
loss 2.5148797
STEP 233 ================================
prereg loss 0.007186913 reg_l1 12.538728 reg_l2 9.824726
loss 2.5149324
STEP 234 ================================
prereg loss 0.0071788426 reg_l1 12.538966 reg_l2 9.82598
loss 2.514972
STEP 235 ================================
prereg loss 0.007188493 reg_l1 12.539268 reg_l2 9.827178
loss 2.515042
STEP 236 ================================
prereg loss 0.007213879 reg_l1 12.53983 reg_l2 9.82833
loss 2.5151799
STEP 237 ================================
prereg loss 0.00727575 reg_l1 12.540142 reg_l2 9.829437
loss 2.5153043
STEP 238 ================================
prereg loss 0.0074686375 reg_l1 12.540694 reg_l2 9.830754
loss 2.5156076
STEP 239 ================================
prereg loss 0.0074148388 reg_l1 12.54151 reg_l2 9.832132
loss 2.5157168
STEP 240 ================================
prereg loss 0.0072340756 reg_l1 12.541952 reg_l2 9.833544
loss 2.5156245
STEP 241 ================================
prereg loss 0.0071789986 reg_l1 12.542588 reg_l2 9.835104
loss 2.5156968
STEP 242 ================================
prereg loss 0.00716128 reg_l1 12.543228 reg_l2 9.836591
loss 2.5158072
STEP 243 ================================
prereg loss 0.007148219 reg_l1 12.543876 reg_l2 9.838026
loss 2.5159235
STEP 244 ================================
prereg loss 0.0071410397 reg_l1 12.544369 reg_l2 9.839396
loss 2.5160148
STEP 245 ================================
prereg loss 0.007147907 reg_l1 12.544442 reg_l2 9.840687
loss 2.5160363
STEP 246 ================================
prereg loss 0.007169799 reg_l1 12.545018 reg_l2 9.84192
loss 2.5161734
STEP 247 ================================
prereg loss 0.0072022607 reg_l1 12.545723 reg_l2 9.843123
loss 2.5163467
STEP 248 ================================
prereg loss 0.0072857295 reg_l1 12.54614 reg_l2 9.8443
loss 2.5165138
STEP 249 ================================
prereg loss 0.0074413223 reg_l1 12.546398 reg_l2 9.845716
loss 2.516721
STEP 250 ================================
prereg loss 0.007395261 reg_l1 12.54677 reg_l2 9.847201
loss 2.5167494
STEP 251 ================================
prereg loss 0.007232306 reg_l1 12.5477 reg_l2 9.848738
loss 2.5167723
STEP 252 ================================
prereg loss 0.0071629635 reg_l1 12.548567 reg_l2 9.8504505
loss 2.5168765
STEP 253 ================================
prereg loss 0.007132921 reg_l1 12.549126 reg_l2 9.852107
loss 2.5169582
STEP 254 ================================
prereg loss 0.0071016443 reg_l1 12.549924 reg_l2 9.853722
loss 2.5170863
STEP 255 ================================
prereg loss 0.0070802867 reg_l1 12.550497 reg_l2 9.855261
loss 2.5171797
STEP 256 ================================
prereg loss 0.007079236 reg_l1 12.551056 reg_l2 9.8567095
loss 2.5172904
STEP 257 ================================
prereg loss 0.007096554 reg_l1 12.551531 reg_l2 9.858088
loss 2.5174026
STEP 258 ================================
prereg loss 0.00712639 reg_l1 12.552078 reg_l2 9.859428
loss 2.5175421
STEP 259 ================================
prereg loss 0.0071886135 reg_l1 12.552535 reg_l2 9.860729
loss 2.5176957
STEP 260 ================================
prereg loss 0.007364027 reg_l1 12.552869 reg_l2 9.862252
loss 2.517938
STEP 261 ================================
prereg loss 0.007306117 reg_l1 12.553482 reg_l2 9.863841
loss 2.5180025
STEP 262 ================================
prereg loss 0.007146895 reg_l1 12.554433 reg_l2 9.865472
loss 2.5180335
STEP 263 ================================
prereg loss 0.0070813354 reg_l1 12.555384 reg_l2 9.867264
loss 2.518158
STEP 264 ================================
prereg loss 0.007060034 reg_l1 12.556047 reg_l2 9.868992
loss 2.5182695
STEP 265 ================================
prereg loss 0.007035272 reg_l1 12.556384 reg_l2 9.870679
loss 2.5183122
STEP 266 ================================
prereg loss 0.007015464 reg_l1 12.55706 reg_l2 9.872301
loss 2.5184276
STEP 267 ================================
prereg loss 0.0070154485 reg_l1 12.557936 reg_l2 9.873835
loss 2.5186026
STEP 268 ================================
prereg loss 0.007034566 reg_l1 12.5585575 reg_l2 9.875296
loss 2.5187461
STEP 269 ================================
prereg loss 0.0070674196 reg_l1 12.558779 reg_l2 9.87672
loss 2.5188231
STEP 270 ================================
prereg loss 0.0071424665 reg_l1 12.559063 reg_l2 9.878109
loss 2.5189552
STEP 271 ================================
prereg loss 0.0073217065 reg_l1 12.560162 reg_l2 9.879729
loss 2.5193539
STEP 272 ================================
prereg loss 0.0072571826 reg_l1 12.561035 reg_l2 9.881416
loss 2.5194643
STEP 273 ================================
prereg loss 0.0071065295 reg_l1 12.561564 reg_l2 9.88315
loss 2.5194194
STEP 274 ================================
prereg loss 0.007023668 reg_l1 12.562155 reg_l2 9.885052
loss 2.5194545
STEP 275 ================================
prereg loss 0.0070034377 reg_l1 12.562922 reg_l2 9.886883
loss 2.519588
STEP 276 ================================
prereg loss 0.006983603 reg_l1 12.563846 reg_l2 9.888671
loss 2.5197527
STEP 277 ================================
prereg loss 0.0069650183 reg_l1 12.564507 reg_l2 9.89039
loss 2.5198662
STEP 278 ================================
prereg loss 0.0069629448 reg_l1 12.565171 reg_l2 9.892028
loss 2.5199974
STEP 279 ================================
prereg loss 0.006979658 reg_l1 12.565798 reg_l2 9.893601
loss 2.5201392
STEP 280 ================================
prereg loss 0.0070111966 reg_l1 12.566184 reg_l2 9.895134
loss 2.520248
STEP 281 ================================
prereg loss 0.007093642 reg_l1 12.566616 reg_l2 9.896628
loss 2.520417
STEP 282 ================================
prereg loss 0.0072743134 reg_l1 12.5673065 reg_l2 9.89836
loss 2.5207357
STEP 283 ================================
prereg loss 0.0072064907 reg_l1 12.568362 reg_l2 9.9001665
loss 2.520879
STEP 284 ================================
prereg loss 0.0070498986 reg_l1 12.569095 reg_l2 9.902016
loss 2.5208688
STEP 285 ================================
prereg loss 0.006961903 reg_l1 12.569592 reg_l2 9.904034
loss 2.5208805
STEP 286 ================================
prereg loss 0.0069408626 reg_l1 12.570542 reg_l2 9.905976
loss 2.5210493
STEP 287 ================================
prereg loss 0.0069223335 reg_l1 12.571554 reg_l2 9.907873
loss 2.521233
STEP 288 ================================
prereg loss 0.0069042468 reg_l1 12.572365 reg_l2 9.909701
loss 2.5213773
STEP 289 ================================
prereg loss 0.006902119 reg_l1 12.572742 reg_l2 9.911441
loss 2.5214508
STEP 290 ================================
prereg loss 0.0069192904 reg_l1 12.573017 reg_l2 9.913115
loss 2.5215228
STEP 291 ================================
prereg loss 0.006951615 reg_l1 12.5740795 reg_l2 9.91475
loss 2.5217676
STEP 292 ================================
prereg loss 0.0070365868 reg_l1 12.574824 reg_l2 9.916344
loss 2.5220015
STEP 293 ================================
prereg loss 0.0072236606 reg_l1 12.575373 reg_l2 9.918176
loss 2.522298
STEP 294 ================================
prereg loss 0.0071543073 reg_l1 12.576058 reg_l2 9.920089
loss 2.522366
STEP 295 ================================
prereg loss 0.006995601 reg_l1 12.576706 reg_l2 9.922049
loss 2.522337
STEP 296 ================================
prereg loss 0.0069046267 reg_l1 12.577807 reg_l2 9.924174
loss 2.5224662
STEP 297 ================================
prereg loss 0.0068844385 reg_l1 12.578635 reg_l2 9.926226
loss 2.5226114
STEP 298 ================================
prereg loss 0.00686702 reg_l1 12.579377 reg_l2 9.92823
loss 2.5227425
STEP 299 ================================
prereg loss 0.0068489616 reg_l1 12.580217 reg_l2 9.930167
loss 2.5228925
STEP 300 ================================
prereg loss 0.0068467367 reg_l1 12.58075 reg_l2 9.932012
loss 2.522997
2022-06-27T15:32:51.506

julia> serialize("sparse19-after-500-steps-matrix.ser", trainable["network_matrix"])

julia> serialize("sparse19-after-500-steps-opt.ser", opt)

julia> open("sparse19-after-500-steps-matrix.json", "w") do f
           JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
       end
```

#### Unsuccessful exploration of training past the first 500 steps is in the `post-500-steps-for-sparse19.md` file.

## Work with the new baseline is continuing in the `post-sparse19-500.md` file.
