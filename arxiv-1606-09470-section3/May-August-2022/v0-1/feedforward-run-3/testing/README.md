# More systematic testing after completion of the sparsification experiment

`test.jl` - upgraded, with `test_this!` function added.

Noting Julia configuration:

```
Version 1.7.3

  [31c24e10] Distributions v0.25.62
  [7da242da] Enzyme v0.10.0
  [587475ba] Flux v0.13.3
  [de31a74c] FunctionalCollections v0.5.0
  [f67ccb44] HDF5 v0.16.10
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
```

Some test results:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3/testing")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
test_this! (generic function with 1 method)

julia> test_this!("sparse20-after-100-steps-matrix.ser", "test string.", 140)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: test string.
length of test string: 12
temporal window: 140

prereg loss 0.0016100588 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.0005265353 average per element loss 5.75021e-6
0.0016100588f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "test string.", 150)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: test string.
length of test string: 12
temporal window: 150

prereg loss 0.6776272 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.11265322 average per element loss 0.0022587574
0.6776272f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "teest string.", 140)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: teest string.
length of test string: 13
temporal window: 140

prereg loss 0.0013412877 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.0005265353 average per element loss 4.7903136e-6
0.0013412877f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "teest string.", 150)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: teest string.
length of test string: 13
temporal window: 150

prereg loss 0.0014515945 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.0005265353 average per element loss 4.8386482e-6
0.0014515945f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "tets string.", 140)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: tets string.
length of test string: 12
temporal window: 140

prereg loss 0.0014382894 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.0005265353 average per element loss 5.136748e-6
0.0014382894f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "tets. str.ingtb str.ingtb", 140)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: tets. str.ingtb str.ingtb
length of test string: 25
temporal window: 140

prereg loss 1.6212574 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.944003 average per element loss 0.005790205
1.6212574f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "tets. str.ingtb str.ingtb", 180)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: tets. str.ingtb str.ingtb
length of test string: 25
temporal window: 180

prereg loss 8.995915 reg_l1 12.706874 reg_l2 10.178641
max per element loss 0.96731496 average per element loss 0.024988653
8.995915f0

julia> test_this!("sparse20-after-100-steps-matrix.ser", "tets. str.ingtb str.ingtb", 250)


model: sparse20-after-100-steps-matrix.ser
number of soft links: 19
test string: tets. str.ingtb str.ingtb
length of test string: 25
temporal window: 250

prereg loss 50.602097 reg_l1 12.706874 reg_l2 10.178641
max per element loss 1.735625 average per element loss 0.101204194
50.602097f0
```
