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
