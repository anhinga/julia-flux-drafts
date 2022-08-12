Experiments around a DMM prototyped in Section 3 of "Programming Patterns in Dataflow Matrix Machines and Generalized Recurrent Neural Nets", https://arxiv.org/abs/1606.09470

(A version of duplicate characters or duplicate words detector. We think this is a nice toy problem to work as a playground for initial "program synthesis via DMMs" experiments.)

### Failed experiments trying to gradients with respect to variables stored in dictionaries (July 2021) 

The information about experiments which involve taking gradients through dictionaries is now located in the
`dictionary-experiments` subdirectory.

I concluded that this is not doable without adding new custom rules (either to Zygote itself or to ChainRules/ChainRulesCore).
The `dictionary-experiments` subdirectory documents various unsuccessful attempts to differentiate through dictionaries
without custom rules. (Of course, I have not tried to merge the pull request allowing mutable arrays; but even if this can be
made to work in this fashion, it would be a slow and inefficient way to do things; adding new custom rules is the
correct way to do these things.)

### Successful experiments taking gradients with respect to variables stored in dictionaries and nested dictionaries (April 2022)

In February 2022 I was able to do that in JAX: https://github.com/anhinga/jax-pytree-example

So I was able to formulate a better question to the creators of Zygote.jl and to get an advice which allowed to solve
the problems I had in July 2021: https://github.com/FluxML/Flux.jl/issues/628 (also there was an interesting remark in that discussion 
that named tuples might be a better fit if all keys are strings (and if we want immutable dictionaries), this is something to ponder;
named tuples should be way faster)

It turns out that
   * the key is to avoid using the legacy interface based on **Params** (that interface is not flexible enough)
   * many improvements have happened since July 2021, and a lot of new adjoints are now available out of the box.

```julia
julia> using Flux

julia> function sum_relu_dict(d)
                  s = 0f0
                  for k in keys(d)
                      s += relu(d[k])
                  end
                  s
              end
sum_relu_dict (generic function with 1 method)

julia> pars = Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Dict{Any, Float32} with 3 entries:
  "y" => 4.0
  8   => -3.0
  :x  => 0.0

julia> sum_relu_dict(pars)
4.0f0

julia> gradient(sum_relu_dict, pars)
(Dict{Any, Any}("y" => 1.0f0, 8 => 0.0f0, :x => 1.0f0),)
```

and now with nested dictionaries:

```julia
julia> function sum_relu_nested(d::Number)
         relu(d)
       end
sum_relu_nested (generic function with 1 method)

julia> function sum_relu_nested(d)
         s = 0f0
         for k in keys(d)
           s += sum_relu_nested(d[k])
         end
         s
       end
sum_relu_nested (generic function with 2 methods)

julia> sum_relu_nested(pars)
4.0f0

julia> gradient(sum_relu_nested, pars)
(Dict{Any, Any}("y" => 1.0f0, 8 => 0.0f0, :x => 1.0f0),)

julia> pars = convert(Dict{Any, Any}, Dict(:x=>0f0, "y"=>4f0, 8=>-3f0))
Dict{Any, Any} with 3 entries:
  :x  => 0.0
  8   => -3.0
  "y" => 4.0

julia> pars["heh"] = Dict(4 => -2f0, "oh oh" => 5f0)
Dict{Any, Float32} with 2 entries:
  4       => -2.0
  "oh oh" => 5.0

julia> pars
Dict{Any, Any} with 4 entries:
  :x    => 0.0
  "heh" => Dict{Any, Float32}(4=>-2.0, "oh oh"=>5.0)
  8     => -3.0
  "y"   => 4.0

julia> sum_relu_nested(pars)
9.0f0

julia> gradient(sum_relu_nested, pars)
(Dict{Any, Any}(:x => 1.0f0, "heh" => Dict{Any, Any}(4 => 0.0f0, "oh oh" => 1.0f0), 8 => 0.0f0, "y" => 1.0f0),)
```

Moreover, taking gradients through dictionary creation:

```julia
julia> using Flux

julia> pars = Dict("x" => 0f0, "y" => 4f0, "8" => -3f0)
Dict{String, Float32} with 3 entries:
  "8" => -3.0
  "x" => 0.0
  "y" => 4.0

julia> function my_map(my_f, my_dict)
           new_dict = Dict()
           for k in keys(my_dict)
               new_dict[k] = my_f(my_dict[k])
           end
           new_dict
       end
my_map (generic function with 1 method)

julia> function my_sum(my_dict)
           s = 0f0
           for k in keys(my_dict)
               s += my_dict[k]
           end
           s
       end
my_sum (generic function with 1 method)

julia> my_sum(my_map(relu, pars))
4.0f0

julia> my_map(relu, pars)
Dict{Any, Any} with 3 entries:
  "y" => 4.0
  "x" => 0.0
  "8" => 0.0

julia> gradient(pars -> my_sum(my_map(relu, pars)), pars)
(Dict{Any, Any}("8" => 0.0f0, "x" => 0.0f0, "y" => 1.0f0),)

julia> pars2 = Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Dict{Any, Float32} with 3 entries:
  "y" => 4.0
  8   => -3.0
  :x  => 0.0

julia> my_map(relu, pars2)
Dict{Any, Any} with 3 entries:
  :x  => 0.0
  8   => 0.0
  "y" => 4.0

julia> my_sum(my_map(relu, pars2))
4.0f0

julia> gradient(pars -> my_sum(my_map(relu, pars)), pars2)
(Dict{Any, Any}("y" => 1.0f0, 8 => 0.0f0, :x => 0.0f0),)
```

Actually, the last example does not quite demonstrate the loss of type info, because `my_map` already loses type info.

Let's redo that:

```julia
julia> pars
Dict{String, Float32} with 3 entries:
  "8" => -3.0
  "x" => 0.0
  "y" => 4.0
  
julia> function my_map_typed(my_f, my_dict::Dict{T, T1}) where {T, T1}
                         new_dict = Dict{T, T1}()
                        for k in keys(my_dict)
                             new_dict[k] = my_f(my_dict[k])
                         end
                         new_dict
                     end
my_map_typed (generic function with 1 method)

julia> my_map_typed(relu, pars)
Dict{String, Float32} with 3 entries:
  "y" => 4.0
  "x" => 0.0
  "8" => 0.0

julia> my_sum(my_map_typed(relu, pars))
4.0f0

julia> gradient(pars -> my_sum(my_map_typed(relu, pars)), pars)
(Dict{Any, Any}("8" => 0.0f0, "x" => 0.0f0, "y" => 1.0f0),)
```
