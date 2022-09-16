# v0.1

_(Jun 9 - Aug 17, 2022)_ **main results are here**

Compared to `rough-sketches`, I am aiming for more clearly structured code.

Differences between "DMM Lite" (here) and DMM 1.0 (2016-2017, https://github.com/jsa-aerial/DMM):

 * streams of flat dictionaries in DMM Lite instead of streams of trees (nested dictionaries) in DMM 1.0
 * the network matrix is outside the network in DMM Lite instead of being controlled by one of the neurons in DMM 1.0
 * fixed number of active neurons in DMM Lite instead of the number of active neurons being controlled by the network matrix in DMM 1.0
 * the network matrix is a tensor of rank 4 in DMM Lite vs tensor of rank 6 in DMM 1.0

**Refactoring is complete**

---

### Next steps to do

Experiment-wise, I am aiming to step back from a fully recurrent setup for a while.

Instead, as the handcrafted system is a _"feedforward transducer with locally recurrent elements"_,
I am going to try to synthesize something similar starting from a larger unstructured
_"feedforward transducer with locally recurrent elements"_.

---

### Subdirectories

#### preliminary runs:

`verifying-refactor` - correctness of the refactor

_(Jun 10, 2022)_

`rec-with-id-4-instead-of-5` - adding `id_transform` to the recurrent setup

_(Jun 12, 2022)_

`first-feedforward-run` - first _"feedforward transducer with locally recurrent elements"_ run

_(Jun 14-15, 2022)_

`feedforward-run-1.1` - similar, but with L1 regularization (`id_transform` still is not a good fit)

_(Jun 16, 2022)_

`feedforward-run-2` - first run with skip connections instead of `id_transform`, but the initialization is quite suboptimal

_(Jun 17-18, 2022)_

#### good runs:

`feedforward-run-3` - a nice convergence finally, with skip connections and well tuned initialization

   * `Testing.md` - The model this run produces is capable of some generalization, it's not completely overfit to the training sequence.

   * `sparsification` - post-feedforward-run-3 sparsification experiments

   * `testing` - testing the resulting models for generalization
   
 `compare-free` - a similar experiment without `compare` neurons (in progress)
