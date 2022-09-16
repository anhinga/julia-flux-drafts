# A bit of testing

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/feedforward-run-3")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> steps!(2)
2022-06-23T11:56:41.187
STEP 1 ================================
prereg loss 434.98627 reg_l1 11.908083 reg_l2 0.150648
loss 434.99817
STEP 2 ================================
prereg loss 417.8722 reg_l1 11.291319 reg_l2 0.14108413
loss 417.88348
2022-06-23T11:58:37.883
```

Yes, this does indeed reproduce.

Now let's test generalization:

```
julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> include("test.jl")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 41 entries:
  "dot-1-1"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0373403),
  [...]
  
julia> reset_dicts!()

julia> loss_k(trainable, 140)
prereg loss 5.016074 reg_l1 119.898636 reg_l2 35.95802
loss 5.135973
5.135973f0
```

So, we moved the duplicate character one position up, replacing `"test string."` with `"tets string."`
and the duplicate character is detected just fine at the moment of time where we expect it to be detected,
but the downstream error at the subsequent time is somewhat larger, more visible, resulting in a larger,
although still moderate, total loss:

```
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.03162676 right: 0.0022939611
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(driving input) timer: 2.0
(getting on output) left: -0.010527162 right: -0.01611366
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(driving input) timer: 3.0
(getting on output) left: 0.0042322706 right: -0.010507682
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(driving input) timer: 4.0
(getting on output) left: 0.0012383538 right: 3.9528823e-7
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(driving input) timer: 5.0
(getting on output) left: 0.008493041 right: -0.006646416
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(driving input) timer: 6.0
(getting on output) left: 0.0043333373 right: -0.0022611518
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(driving input) timer: 7.0
(getting on output) left: 0.0085146865 right: -0.0016428402
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(driving input) timer: 8.0
(getting on output) left: 0.0077711996 right: -0.0024548648
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(driving input) timer: 9.0
(getting on output) left: 0.0066113127 right: -0.0032500948
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(driving input) timer: 10.0
(getting on output) left: 0.0054158717 right: -0.004007792
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(driving input) timer: 11.0
(getting on output) left: 0.004187524 right: -0.004732237
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(driving input) timer: 12.0
(getting on output) left: 0.019868065 right: -0.0007903932
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(driving input) timer: 13.0
(getting on output) left: -0.0032312926 right: 0.0054480117
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(driving input) timer: 14.0
(getting on output) left: 0.006323075 right: 0.010820383
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(driving input) timer: 15.0
(getting on output) left: -0.007679085 right: -0.0025672351
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(driving input) timer: 16.0
(getting on output) left: 0.0041404385 right: 0.0021515836
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(driving input) timer: 17.0
(getting on output) left: 0.00041037425 right: 0.0006958833
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(driving input) timer: 18.0
(getting on output) left: -0.00064952066 right: -0.000100645935
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(driving input) timer: 19.0
(getting on output) left: -0.0017329119 right: -0.0009018844
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(driving input) timer: 20.0
(getting on output) left: -0.0028397734 right: -0.001711516
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(driving input) timer: 21.0
(getting on output) left: -0.003969599 right: -0.0025332596
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(driving input) timer: 22.0
(getting on output) left: -0.007185456 right: 0.000113854185
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(driving input) timer: 23.0
(getting on output) left: 0.072947964 right: -0.019087031
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(driving input) timer: 24.0
(getting on output) left: 1.0229231 right: -0.010449778
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 24.0
(driving input) timer: 25.0
(getting on output) left: 1.0459604 right: 0.0033113125
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 25.0
(driving input) timer: 26.0
(getting on output) left: 1.0216912 right: 0.013193019
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 26.0
(driving input) timer: 27.0
(getting on output) left: 1.0329444 right: 0.014201372
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 27.0
(driving input) timer: 28.0
(getting on output) left: 1.0463002 right: 0.016003214
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 28.0
(driving input) timer: 29.0
(getting on output) left: 1.0592836 right: 0.01796051
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 29.0
(driving input) timer: 30.0
(getting on output) left: 1.0719016 right: 0.020063026
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 30.0
(driving input) timer: 31.0
(getting on output) left: 1.0841606 right: 0.022300962
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 31.0
(driving input) timer: 32.0
(getting on output) left: 1.0822589 right: 0.035878085
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 32.0
(driving input) timer: 33.0
(getting on output) left: 1.0558043 right: 0.03831727
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 33.0
(driving input) timer: 34.0
(getting on output) left: 1.0970638 right: 0.039405722
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
(driving input) timer: 35.0
(getting on output) left: 1.1256356 right: 0.033098903
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 35.0
(driving input) timer: 36.0
(getting on output) left: 1.1317152 right: 0.037236035
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 36.0
(driving input) timer: 37.0
(getting on output) left: 1.1396714 right: 0.039256267
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 37.0
(driving input) timer: 38.0
(getting on output) left: 1.1492842 right: 0.041854855
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 38.0
(driving input) timer: 39.0
(getting on output) left: 1.1585597 right: 0.044490173
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 39.0
(driving input) timer: 40.0
(getting on output) left: 1.1675019 right: 0.04715325
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 40.0
(driving input) timer: 41.0
(getting on output) left: 1.174628 right: 0.049649302
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 41.0
(driving input) timer: 42.0
(getting on output) left: 1.1649657 right: 0.061876014
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 42.0
(driving input) timer: 43.0
(getting on output) left: 1.1429008 right: 0.06595834
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 43.0
(driving input) timer: 44.0
(getting on output) left: 1.1500081 right: 0.066083774
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 44.0
(driving input) timer: 45.0
(getting on output) left: 1.1903297 right: 0.06668828
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 45.0
(driving input) timer: 46.0
(getting on output) left: 1.1714904 right: 0.06626591
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 46.0
(driving input) timer: 47.0
(getting on output) left: 1.1762484 right: 0.069201574
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 47.0
(driving input) timer: 48.0
(getting on output) left: 1.1795208 right: 0.071686655
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 48.0
(driving input) timer: 49.0
(getting on output) left: 1.1824425 right: 0.07411686
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 49.0
(driving input) timer: 50.0
(getting on output) left: 1.1850152 right: 0.07648291
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 50.0
(driving input) timer: 51.0
(getting on output) left: 1.1872405 right: 0.07877618
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 51.0
(driving input) timer: 52.0
(getting on output) left: 1.1824961 right: 0.08808693
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 52.0
(driving input) timer: 53.0
(getting on output) left: 1.1281754 right: 0.09029125
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 53.0
(driving input) timer: 54.0
(getting on output) left: 1.1824538 right: 0.099408776
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 54.0
(driving input) timer: 55.0
(getting on output) left: 1.1679065 right: 0.09177622
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 55.0
(driving input) timer: 56.0
(getting on output) left: 1.1548172 right: 0.09178788
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 56.0
(driving input) timer: 57.0
(getting on output) left: 1.1589717 right: 0.0924656
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 57.0
(driving input) timer: 58.0
(getting on output) left: 1.162414 right: 0.09283609
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 58.0
(driving input) timer: 59.0
(getting on output) left: 1.165506 right: 0.09302828
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 59.0
(driving input) timer: 60.0
(getting on output) left: 1.1682411 right: 0.09303678
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 60.0
(driving input) timer: 61.0
(getting on output) left: 1.1706123 right: 0.09285685
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 61.0
(driving input) timer: 62.0
(getting on output) left: 1.1261681 right: 0.1132549
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 62.0
(driving input) timer: 63.0
(getting on output) left: 1.1302494 right: 0.053447217
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 63.0
(driving input) timer: 64.0
(getting on output) left: 2.0799649 right: 0.12089459
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 64.0
(driving input) timer: 65.0
(getting on output) left: 2.1351693 right: 0.077885784
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 65.0
(driving input) timer: 66.0
(getting on output) left: 2.165115 right: 0.08387119
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 66.0
(driving input) timer: 67.0
(getting on output) left: 2.1714456 right: 0.08535173
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 67.0
(driving input) timer: 68.0
(getting on output) left: 2.178908 right: 0.08703249
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 68.0
(driving input) timer: 69.0
(getting on output) left: 2.1856167 right: 0.088590026
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 69.0
(driving input) timer: 70.0
(getting on output) left: 2.1915712 right: 0.09001106
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 70.0
(driving input) timer: 71.0
(getting on output) left: 2.1967695 right: 0.09128222
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 71.0
(driving input) timer: 72.0
(getting on output) left: 2.1886723 right: 0.10059496
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 72.0
(driving input) timer: 73.0
(getting on output) left: 2.1895509 right: 0.12038001
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 73.0
(driving input) timer: 74.0
(getting on output) left: 2.1792746 right: 0.10927994
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 74.0
(driving input) timer: 75.0
(getting on output) left: 2.1692374 right: 0.107191324
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 75.0
(driving input) timer: 76.0
(getting on output) left: 2.171421 right: 0.10804462
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 76.0
(driving input) timer: 77.0
(getting on output) left: 2.1684415 right: 0.107507914
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 77.0
(driving input) timer: 78.0
(getting on output) left: 2.1666858 right: 0.107084654
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 78.0
(driving input) timer: 79.0
(getting on output) left: 2.1640508 right: 0.10641665
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 79.0
(driving input) timer: 80.0
(getting on output) left: 2.1605203 right: 0.10549768
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 80.0
(driving input) timer: 81.0
(getting on output) left: 2.1560764 right: 0.104323134
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 81.0
(driving input) timer: 82.0
(getting on output) left: 2.163618 right: 0.11051006
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 82.0
(driving input) timer: 83.0
(getting on output) left: 2.1676528 right: 0.12498589
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 83.0
(driving input) timer: 84.0
(getting on output) left: 2.1774707 right: 0.114182994
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 84.0
(driving input) timer: 85.0
(getting on output) left: 2.1708949 right: 0.11611359
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 85.0
(driving input) timer: 86.0
(getting on output) left: 2.1783323 right: 0.11558246
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 86.0
(driving input) timer: 87.0
(getting on output) left: 2.1866777 right: 0.11381018
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 87.0
(driving input) timer: 88.0
(getting on output) left: 2.1964526 right: 0.11229418
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 88.0
(driving input) timer: 89.0
(getting on output) left: 2.2056372 right: 0.110696234
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 89.0
(driving input) timer: 90.0
(getting on output) left: 2.2142158 right: 0.10901521
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 90.0
(driving input) timer: 91.0
(getting on output) left: 2.2221715 right: 0.1072511
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 91.0
(driving input) timer: 92.0
(getting on output) left: 2.2318356 right: 0.11313327
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 92.0
(driving input) timer: 93.0
(getting on output) left: 2.1974018 right: 0.124027796
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 93.0
(driving input) timer: 94.0
(getting on output) left: 2.2281713 right: 0.11130891
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 94.0
(driving input) timer: 95.0
(getting on output) left: 2.2184083 right: 0.118845865
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 95.0
(driving input) timer: 96.0
(getting on output) left: 2.2149422 right: 0.118612364
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 96.0
(driving input) timer: 97.0
(getting on output) left: 2.2172608 right: 0.11559864
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 97.0
(driving input) timer: 98.0
(getting on output) left: 2.221875 right: 0.11277109
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 98.0
(driving input) timer: 99.0
(getting on output) left: 2.2257166 right: 0.10986798
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 99.0
(driving input) timer: 100.0
(getting on output) left: 2.2286878 right: 0.10690855
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 100.0
(driving input) timer: 101.0
(getting on output) left: 2.2307572 right: 0.10390216
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 101.0
(driving input) timer: 102.0
(getting on output) left: 2.2342386 right: 0.10858804
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 102.0
(driving input) timer: 103.0
(getting on output) left: 2.198985 right: 0.11752933
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 103.0
(driving input) timer: 104.0
(getting on output) left: 2.211664 right: 0.10652604
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 104.0
(driving input) timer: 105.0
(getting on output) left: 2.210854 right: 0.12495246
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 105.0
(driving input) timer: 106.0
(getting on output) left: 2.2177975 right: 0.13162823
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 106.0
(driving input) timer: 107.0
(getting on output) left: 2.2212331 right: 0.13167492
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 107.0
(driving input) timer: 108.0
(getting on output) left: 2.2215748 right: 0.131243
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 108.0
(driving input) timer: 109.0
(getting on output) left: 2.2208812 right: 0.1310091
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 109.0
(driving input) timer: 110.0
(getting on output) left: 2.2191029 right: 0.13099557
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 110.0
(driving input) timer: 111.0
(getting on output) left: 2.2159522 right: 0.13154635
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 111.0
(driving input) timer: 112.0
(getting on output) left: 2.2448196 right: 0.12003485
(getting on output) left: 2.0 right: 0.0
(driving input) timer: 112.0
(driving input) timer: 113.0
(getting on output) left: 2.1766362 right: 0.9715332
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 113.0
(driving input) timer: 114.0
(getting on output) left: 2.1860206 right: 0.9607267
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 114.0
(driving input) timer: 115.0
(getting on output) left: 2.2021713 right: 0.9547861
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 115.0
(driving input) timer: 116.0
(getting on output) left: 2.2109084 right: 0.95997745
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 116.0
(driving input) timer: 117.0
(getting on output) left: 2.2120407 right: 0.93429196
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 117.0
(driving input) timer: 118.0
(getting on output) left: 2.2172759 right: 0.94338745
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 118.0
(driving input) timer: 119.0
(getting on output) left: 2.2213244 right: 0.9522716
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 119.0
(driving input) timer: 120.0
(getting on output) left: 2.224095 right: 0.96156603
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 120.0
(driving input) timer: 121.0
(getting on output) left: 2.2255404 right: 0.9715013
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 121.0
(driving input) timer: 122.0
(getting on output) left: 2.2352445 right: 0.9381918
(getting on output) left: 2.0 right: 1.0
(driving input) timer: 122.0
(driving input) timer: 123.0
(getting on output) left: 2.23194 right: 1.9575629
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 123.0
(driving input) timer: 124.0
(getting on output) left: 2.2838662 right: 1.9239875
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 124.0
(driving input) timer: 125.0
(getting on output) left: 2.2850952 right: 1.9351227
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 125.0
(driving input) timer: 126.0
(getting on output) left: 2.2617 right: 1.9096935
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 126.0
(driving input) timer: 127.0
(getting on output) left: 2.242563 right: 1.9092952
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 127.0
(driving input) timer: 128.0
(getting on output) left: 2.2405694 right: 1.9306122
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 128.0
(driving input) timer: 129.0
(getting on output) left: 2.2449367 right: 1.9477128
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 129.0
(driving input) timer: 130.0
(getting on output) left: 2.2479439 right: 1.9652528
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 130.0
(driving input) timer: 131.0
(getting on output) left: 2.2499359 right: 1.9834547
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 131.0
(driving input) timer: 132.0
(getting on output) left: 2.2410526 right: 1.9292579
(getting on output) left: 2.0 right: 2.0
(driving input) timer: 132.0
(driving input) timer: 133.0
(getting on output) left: 2.2145185 right: 2.9886315
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 133.0
(driving input) timer: 134.0
(getting on output) left: 2.2453966 right: 2.9526973
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 134.0
(driving input) timer: 135.0
(getting on output) left: 2.2221668 right: 3.0034082
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 135.0
(driving input) timer: 136.0
(getting on output) left: 2.2267087 right: 3.0017881
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 136.0
(driving input) timer: 137.0
(getting on output) left: 2.2271452 right: 2.9964254
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 137.0
(driving input) timer: 138.0
(getting on output) left: 2.2275484 right: 2.992019
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 138.0
(driving input) timer: 139.0
(getting on output) left: 2.2272408 right: 2.9869714
(getting on output) left: 2.0 right: 3.0
(driving input) timer: 139.0
prereg loss 5.016074 reg_l1 119.898636 reg_l2 35.95802
loss 5.135973
```
