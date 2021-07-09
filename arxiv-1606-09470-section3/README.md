Experiments around a DMM prototyped in Section 3 of "Programming Patterns in Dataflow Matrix Machines and Generalized Recurrent Neural Nets", https://arxiv.org/abs/1606.09470

(A version of duplicate characters or duplicate words detector. We think this is a nice toy problem to work as a playground for initial "program synthesis via DMMs" experiments.)

---

Our first experiment, `gradient-of-relu-on-dict.ipynb` was actually performed on June 22. You can see that taking a gradient of a computation involving 
ReLU applied to an array is easy in Julia Flux.

However, taking a gradient of a computation involving ReLU applied to a dictionary turned out to be non-trivial, because there are some problems with gradients of iterators
in the current version of Flux/Zygote.

I successfully found a workaround here, and the gradient is computed, but I don't like this particular solution for a number of reasons:
it involves an entirely unnecessary mutation and it is not compositional.

We are going to try again.

---

The straight functional refactor does not work. We document it as `unsuccessful-refactor-of-gradient-of-relu-on-dict.ipynb`.

This is also an extremely small Jupyter notebook, but it just took me 3 reloads to render it on GitHub (use `nbviewer` if necessary).

The first problem here is that `map` is not provided for the dictionary, and the explanations for that are quite unsatisfying.

So, here is a horrible mutable workaround which works:

```julia
function my_map(my_f, my_dict)
    new_dict = deepcopy(my_dict)
    map!(my_f, values(new_dict))
    new_dict
end
```

But then we bumping into this diagnostics: https://github.com/FluxML/Zygote.jl/issues/408

So here we at least have a compositional solution, although still with encapsulated mutations (which are unfortunate),
but it does not work with the current Zygote. 

OK, the question of how to make this without mutations is a deeper Julia-related question (we have to implement `map`,
if Julia would not want to provide one of the box, and in some sense, all those immutable operations do mutate
newly allocated memory, so in some sense one can argue that `my_map` is already that), but we need to make another
at making gradients for a compositional solution work.

---

So, I tried a few things in `second-unsuccessful-refactor-of-gradient-of-relu-on-dict.ipynb`.

I tried to add type annotation to `my_map`, and I tried to replace it with

```julia
function my_map(my_f, my_dict::Dict{Any, Float32})
    new_dict = Dict{Any, Float32}()
    for k in keys(my_dict)
        new_dict[k] = f(my_dict[k])
    end
    new_dict
end
```

and I tried to incorporate adjoints from this unintegrated pull request

https://github.com/FluxML/Zygote.jl/pull/412

(not that I expected it to work, since I don't have explicit get or get!, but
I don't really know the internals of the system, so I thought to give it a chance),

but I was still getting

`MethodError: no method matching getindex(::Dict{Any, Any})`

diagnostics.
