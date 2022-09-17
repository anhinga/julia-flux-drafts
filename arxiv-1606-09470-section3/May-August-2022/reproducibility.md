# Reproducibility and compatibility

Efforts has been made to make these experiments as reproducible as possible.

Each subdirectory is self-contained (use the version of the code contained
in that subdirectory to reproduce its runs; I feel that doing it this way 
is more reliable and transparent than using version control tags or branches).

Random numbers have been generated using a seed and they have been cached
and stored, in case the reliance on a seed turns to be insufficient.

The jobs were run in Julia console on a Windows 10 laptop using laptop CPU;
the edited content of console has been copied into markdown files with
appropriate comments which should be sufficient to reproduce the runs
if necessary.

Julia version and versions of packages have been frozen in early May, and the
May-July experiments have been produced with those versions (which has been noted elsewhere).

In particular, I have used `Zygote.jl v0.6.40`. Then I've upgraded the packages in
late July/early August, but it has turned out that `Zygote.jl v0.6.41` has introduced a bug
which has not been fixed as of `Zygote.jl v0.6.43`, so I have downgraded Zygote back
to v0.6.40. The following configuration has been used in August:

```
Julia 1.7.3

(@v1.7) pkg> status
      Status `C:\Users\anhin\.julia\environments\v1.7\Project.toml`
  [7d9f7c33] Accessors v0.1.18
  [d360d2e6] ChainRulesCore v1.15.3
  [31c24e10] Distributions v0.25.66
  [7da242da] Enzyme v0.10.4
  [587475ba] Flux v0.13.5
  [de31a74c] FunctionalCollections v0.5.0
  [f67ccb44] HDF5 v0.16.10
  [7073ff75] IJulia v1.23.3
  [86fae568] ImageView v0.11.1
  [916415d5] Images v0.25.2
  [4138dd39] JLD v0.13.2
  [033835bb] JLD2 v0.4.22
  [0f8b85d8] JSON3 v1.9.5
  [b12ccfe2] PyCallChainRules v0.4.0
  [37e2e3b7] ReverseDiff v1.14.1
  [f2b01f46] Roots v2.0.2
  [13530c0b] Semagrams v0.2.0
  [8254be44] SymbolicRegression v0.10.0
  [5e47fb64] TestImages v1.7.0
  [e88e6eb3] Zygote v0.6.40
  [37e2e46d] LinearAlgebra
```

I have tested reproducibility a number of times and I have found 
the reproducibility to be "robust, but imperfect".

More specifically, the system is sufficiently robust to noise, and
the qualitative trajectories were reproduced correctly each time.

However, some numerical noise making runs slightly different has been
present fairly often. This is on my TODO list to investigate further,
but the presence of this noise leads me to call the reproducibility here
to be "imperfect".

---

The software also works with Julia 1.8.0

Here is the change which makes it work with Zygote 0.6.41-0.6.46:

In `rough-sketches/variadic-dp-v0-0-1.jl` or in various instances of `dmm-lite.jl` in `v0-1` one should make the following change:

Instead of

```julia
function up_movement!(all_neurons::Dict{String, Neuron})
    for neuron in values(all_neurons)
        # println(neuron)
        # println("INPUT DICT: ", neuron.input_dict)
        # println("AFTER FUNCTION APPLICATION: ", neuron.f(neuron.input_dict))
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end
```

or

```julia
function up_movement!(dmm_lite::DMM_Lite_)
    for neuron in values(dmm_lite["neurons"])
        # println(neuron)
        # println("INPUT DICT: ", neuron.input_dict)
        # println("AFTER FUNCTION APPLICATION: ", neuron.f(neuron.input_dict))
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end
```

one should use this pattern

```julia
function up_movement!(dmm_lite::DMM_Lite_)
    for k in keys(dmm_lite["neurons"])
        neuron = dmm_lite["neurons"][k]
        # println(neuron)
        # println("INPUT DICT: ", neuron.input_dict)
        # println("AFTER FUNCTION APPLICATION: ", neuron.f(neuron.input_dict))
        neuron.output_dict = neuron.f(neuron.input_dict)
    end
end
```

I was not able to easily produce a compact example of this regression which would be needed to file
a regression issue with `Zygote.jl` (in fact, when one uses `in values(dict)` in short Zygote
examples things do tend to break,
so it might be more appropriate to think that this syntactic sugar was never properly supported,
and worked by accident in our case).
