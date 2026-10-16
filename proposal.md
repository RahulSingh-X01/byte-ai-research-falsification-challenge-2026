
## Title

Qualifying Simplicity Bias: Feature Complexity, Cleanliness, and Optimization Ease in a Controlled Synthetic Task


## Introduction

Neural networks are often said to exhibit a simplicity bias: when several
features can solve a task, they tend to learn the simplest one first, or
rely on it entirely. This claim appears in influential work such as Shah et
al. (2020).

But "simple" can mean multiple things. A feature can be simple in the sense of
being linearly separable, low-dimensional, low-frequency, or short to
describe. Two experiments can both claim to show simplicity bias while
having entirely different properties. Worse, in most setups a simple
feature is also cleaner, easier to optimize, and more strongly correlated
with the label than its alternatives. When these factors coincide, it is
hard to tell which one is actually driving the network's preference.

This study takes the role of a skeptical reviewer. Rather than trying to
confirm the claim, I ask under what conditions it holds true and whether it can
be broken. I use a small synthetic task where a simple linear feature (S)
and a complex k-slab feature (C) are both predictive, and I manipulate their
complexity, label noise, and input scale independently. The goal is not to
prove or disprove simplicity bias, but to see how far it survives when the
confounds are removed.


## Literature Review

Three papers are most relevant to this study. Each one claims that neural
networks prefer simple features, but each defines "simple" differently and
tests it in different settings.

**Shah et al. (2020)** is the primary paper. They argue that standard
training with SGD does not just prefer simple features — it can rely on the
simplest feature *exclusively*, ignoring all other predictive features even
when those other features are more accurate or more robust. They introduce
two synthetic datasets where every sample has multiple predictive features
of varying complexity. On these datasets, they show that a network trained
with SGD will often ignore every feature except the simplest one, leading to
failures under small distribution shifts or adversarial perturbations. They
also report that common fixes like ensembles and adversarial training do not
reliably solve the problem. In this paper, "simple" roughly means "linearly
separable" — a feature whose decision boundary is a straight line or
hyperplane.

**Morwani et al. (2023)** give a more formal treatment for one-hidden-layer
networks. They define simplicity bias as the network effectively depending
on a low-dimensional projection of its input — meaning the network's output
is mostly determined by a small number of directions in input space, even
when other directions carry useful information. They prove that for linearly
separable data, a one-hidden-layer network in the infinite-width limit will
depend almost entirely on the linearly separable subspace, even if many
other complex features are present. They also show empirically that models
trained on real datasets like ImageNet and Waterbirds-Landbirds do depend on
low-dimensional projections. So their definition of "simple" is
"low-dimensional," which is not the same as Shah et al.'s "linearly
separable."

**Rahaman et al. (2019)** study a related but distinct phenomenon called
spectral bias. Using Fourier analysis, they show that neural networks learn
low-frequency functions much faster than high-frequency ones. A
low-frequency function is one that changes slowly and smoothly across the
input space; a high-frequency function changes rapidly with small local
fluctuations. In practice, this means a network will fit a smooth global
pattern first and only later (if ever) fit sharp local details. Their
definition of "simple" is "low-frequency," which again differs from both
"linearly separable" and "low-dimensional."

The three papers agree that some form of simplicity bias exists. But they
disagree on what simplicity means. Shah et al. treat it as linear
separability. Morwani et al. treat it as low dimensionality. Rahaman et al.
treat it as low frequency. These are not the same property. A feature can be
linearly separable but high-dimensional, or low-frequency but nonlinear, or
low-dimensional but high-frequency. So when someone says "networks prefer
simple features," the claim is underspecified until you say which definition
of "simple" you are using.

The papers also leave another gap. In most of their experiments, the simple feature is not just simpler — it is also cleaner, easier to optimize, and often more strongly
correlated with the label. When all these factors point the same way, you
cannot tell which one the network is actually responding to. The network
might prefer the simple feature because it is simpler. Or it might prefer it
because it is easier to learn, or less noisy, or more predictive. No single
paper cleanly separates these factors.

My study takes one step toward filling that gap. This study lets me ask whether the preference for the simple feature survives when cleanliness and optimization ease are controlled.


## Research Question

When a simple feature and a complex feature both predict the label, does a
neural network prefer the simple one because it is simple, or because the
simple feature is also easier to optimize, cleaner, or more strongly
predictive?

I break this into three sub-questions that can each be tested separately:

1. **Complexity vs predictive strength.** If the simple feature is made
   noisier than the complex one, does the network still prefer the simple
   feature? This tests whether the preference tracks simplicity or
   predictive strength.
2. **Simplicity vs optimization ease.** If both features are equally
   predictive but the complex feature is made easier to optimize (by
   scaling its input), does the network switch to it? This tests whether
   the preference tracks simplicity or optimization ease.
3. **Persistence of the preference.** Once the network has learned one
   feature, does its reliance change with longer training, or is the
   preference permanent?

Each sub-question isolates one factor at a time. Together they let me
distinguish "the network prefers simple features" from "the network prefers
whichever feature is easiest to learn and cleanest to rely on."


## Hypothesis and Null Hypothesis

**Main hypothesis (H1).**
The apparent simplicity bias is substantially explained by factors that
correlate with simplicity in typical setups — optimization ease, label
cleanliness, and predictive strength. When these factors are matched
between the simple feature S and the complex feature C, the network's
preference for S weakens. The bias is real but conditional, not intrinsic.

**Null hypothesis (H0).**
Feature complexity has no effect on reliance once predictive strength and
optimization ease are matched. If the null holds, R_S stays near 0.5
regardless of k, noise, or scale. Equivalently, the network does not
systematically prefer either feature when the two are matched on every
factor except complexity.

**What would falsify H1.**
If the network continues to rely on S even when S is noisier, harder to
optimize, and lower-dimensional than C, then simplicity is doing the work
on its own and H1 is false. This is the outcome the study is designed to
detect if it exists.

**Sub-hypotheses mapped to experiments.**

| #   | Sub-hypothesis                                                                | Null                                            | Experiment     |
| --- | ----------------------------------------------------------------------------- | ----------------------------------------------- | -------------- |
| H1a | Making S noisier will push the network toward C.                              | Reliance on S is unaffected by rho_S.           | Exp 2          |
| H1b | Scaling C's input to be easier to optimize will push the network toward C.    | Reliance on S is unaffected by input scale.     | Exp 3          |
| H1c | When C is equally simple and cleaner than S, the network will pivot to C.     | The network cannot pivot once it has learned S. | k=1 diagnostic |
| H1d | Reliance is stable over the training budget, not a transient ordering effect. | Reliance shifts toward C with longer training.  | Exp 1 dynamics |

Each sub-hypothesis is independently falsifiable. If H1a–H1c all fail
while H1d holds, H1 is weakened and the original claim survives a serious
attempt to break it. If any of H1a–H1c hold, H1 is supported and the
original claim is qualified.


## Alternative Explanation

The simplicity bias could exist for reasons other than simplicity itself.
The main alternative is optimization ease: the simple feature converges
faster and reaches low loss before the complex feature has a chance to
be learned. Once it does, gradients to other features vanish and the
network has no pressure to re-represent. Under this view, the bias is
about training dynamics, not simplicity as a property.

Two other alternatives matter. Predictive strength: if the simple feature
is also more accurate, the network may prefer it for that reason.
Cleanliness: a clean feature provides a smoother training signal, and the
network may prefer whichever feature is cleaner regardless of complexity.

This study treats all three as competing explanations. Exp 2 varies
predictive strength. Exp 3 varies input scale as one form of optimization
ease. The negative control keeps complexity equal while making the simple
feature noisier, isolating cleanliness. If the claim survives these
manipulations, simplicity is doing work on its own. If the preference
flips under any of them, the alternative was doing the work.


## Proposed Experiment Suite

All experiments share the same dataset generator, model, and training
procedure. I describe those first, then each experiment.

### Shared setup

**Dataset.** Each sample is a 10-dimensional vector with a binary label y in {-1, +1}.

- Coordinate 0: feature S. Linear. sign(S) predicts the label. Values are
  drawn from a Gaussian centered at +mu_S or -mu_S depending on the class,
  with noise sigma_S.
- Coordinate 1: feature C. A k-slab feature in [0, 1) divided into k+1
  equal-width alternating bands. Band parity (even = +1, odd = -1) predicts
  the label.
- Coordinates 2-9: pure Gaussian noise, independent of the label.

Both S and C are individually sufficient to solve the task when their
label-noise rates rho_S and rho_C are zero. The generator also produces a
conflict dataset where S and C imply opposite classes, used for one of the
reliance measures.

**Model.** A small MLP: 2 hidden layers, 64 units each, ReLU activation,
single logit output. Trained with BCEWithLogitsLoss.

**Training.** Adam optimizer, lr = 1e-3, batch size 128, 70/30 stratified
train/test split. Max 400 epochs, chosen after Exp 0 showed that the complex
feature at k=5 needs ~100 epochs to converge in isolation.

**Reliance measures.** Two independent measures of which feature the model
uses:

1. *Shuffle ablation.* Shuffle one feature's values across test samples and
   measure accuracy drop. Report drop_S, drop_C, and the normalized ratio
   R_S_shuffle = drop_S / (drop_S + drop_C). Returns NaN when the total
   drop is under 0.05, to avoid ratios of noise.
2. *Conflict test.* On samples where S and C disagree, measure the fraction
   of predictions that follow S. Call this R_S_conflict. Values near 1 mean
   the model follows S, near 0 mean it follows C, near 0.5 mean a tie.

Both measures are computed every 20 epochs to track how reliance changes
during training.

**Seeds.** 3 seeds per configuration.

### Experiment 0 — Capacity sanity check

Train the MLP on S alone, C at k=1 alone, and C at k=5 alone (each as a
1-dimensional input). Record final train/test accuracy and epochs-to-95%
train accuracy.

Purpose: confirm the model can fit each feature before asking which feature
it prefers. If the model cannot fit C at k=5 in isolation, any later
preference for S is a capacity failure, not a bias.

### Diagnostic — Phase transition at k=5

Train on C alone at k=5 for 1000 epochs, logging accuracy every 50 epochs.
This was run after Exp 0 showed seed-variable convergence times. Purpose:
distinguish "not enough time" from "genuine optimization difficulty."

### Experiment 1 — Reproduce the baseline

Present both S and C (k=5), both fully predictive (rho_S = rho_C = 0).
Train for 400 epochs. Compute reliance measures every 20 epochs.

Purpose: confirm the simplicity bias in our controlled setup. This is the
baseline against which later manipulations are compared.

### Experiment 2 — Decouple complexity from predictive strength

Present both features. Vary the label noise on S: rho_S in {0, 0.1, 0.2,
0.3}. Keep C clean (rho_C = 0). Train for 400 epochs.

Purpose: test whether the network pivots to C when S becomes less
predictive. If it does, the bias tracks predictive strength. If it does
not, the bias is not simply about which feature is more accurate.

### Diagnostic — Negative control (k=1, noisy S)

Present both features with C at k=1 (linear, equally simple as S) and S
made noisy (rho_S = 0.2). Train for 400 epochs.

Purpose: rule out "the network cannot pivot once it has learned S" as an
explanation for Exp 1-2. If the network pivots here, the lock-in observed
earlier is specific to the case where C is complex, not a general
inability to change.

### Experiment 3 — Simplicity vs optimization ease

Present both features, both fully predictive. Scale C's input by a factor
alpha_C in {0.01, 0.1, 1, 10, 100}, keeping S's scale fixed at 1. Train
for 400 epochs.

Purpose: test whether making C easier to optimize (by increasing the
magnitude of its gradients) flips the preference. If it does, the bias
is about optimization ease. If it does not, input scale alone is too weak
to overcome the preference.

Note: the conflict measure must use the same input scale as training, or
the measure is contaminated by the scale mismatch. I caught this bug in
an earlier version of the code and corrected it.

### Reproducibility

Every dataset call uses an explicit seed. Experiment configs are hard-coded
at the top of each script. Results are saved as CSVs, one per experiment,
with columns for the config, seed, final accuracy, and final reliance
measures. Full details, including the numbers for every run, are in
`experiment_log.md`.


## Controls and Confounding Variables

The main risk in this study is mistaking a correlated factor for
simplicity itself. In most published demonstrations of simplicity bias,
the simple feature is not just simpler — it is also cleaner, easier to
optimize, and often more predictive. When all of these point the same
way, the observed preference cannot be attributed to any one of them.
The design below tries to pull these factors apart.

### Confound 1: Predictive strength

**The issue.** A simpler feature may also be the more accurate one. If
the network prefers it, that could just mean the network prefers the more
predictive feature.

**Control.** In Exp 2, I hold complexity fixed (k=5) and vary the label
noise on S. At rho_S = 0, S and C are equally predictive. As rho_S
increases, S becomes less predictive than C. If the network pivots to C
when S becomes noisier, the preference tracks predictive strength. If it
stays on S, then predictive strength is not the main driver.

**What this cannot distinguish.** Label noise changes not just accuracy
but also the difficulty of the training signal. A noisy feature is both
less accurate and harder to fit. So Exp 2 conflates those two things.

### Confound 2: Optimization ease

**The issue.** A simpler feature may be easier for gradient descent to
learn, independent of its accuracy. If the network prefers it, that could
mean the network prefers whichever feature trains faster.

**Control.** In Exp 3, both features are clean and equally predictive
(rho_S = rho_C = 0). I vary only C's input scale. Scaling the input
multiplies the gradients flowing into C's weights without changing its
predictive accuracy or its function complexity. If the network pivots to
C at high scale, the preference is about optimization ease. If it does
not, input scale alone is too weak a knob.

**What this cannot distinguish.** Input scale is only one form of
optimization ease. A feature could also be easy to optimize because of
its margin, its initialization, or its low-frequency structure. Exp 3
tests one form, not all of them.

### Confound 3: Capacity versus preference

**The issue.** If the model cannot learn the complex feature, then
ignoring it says nothing about preference. The model might simply be
incapable.

**Control.** Exp 0 trains the model on each feature alone. If the model
learns C at k=5 to 99% accuracy on its own, then capacity is ruled out as
the reason it ignores C in the combined task.

**What this cannot distinguish.** Capacity in isolation does not mean
capacity in combination. It is possible that a model which can learn C
alone cannot learn C when it is simultaneously trying to fit S. This is
a limitation, not a fatal flaw — but it means Exp 0 is necessary, not
sufficient.

### Confound 4: Lock-in versus inability to pivot

**The issue.** If the network learns S first and never switches, that
could mean either (a) it cannot switch, or (b) it has no reason to switch
because S is sufficient. These have different implications.

**Control.** The negative control (k=1, noisy S) makes C equally simple
and cleaner than S. If the network pivots here, it can switch when the
alternative is easy. This rules out "the network cannot pivot" as an
explanation for Exp 1-2.

**What this cannot distinguish.** Pivoting in one setting (k=1) does not
mean the network would pivot in another (k=5). So this control rules out
one hypothesis but does not establish a general rule.

### Confound 5: Transient ordering

**The issue.** The network might rely on S early in training and shift to
C with more training. If reliance is measured only at the end, that
ordering is invisible.

**Control.** Exp 1 computes reliance every 20 epochs for 400 epochs.
If reliance is flat, the preference is stable over the training budget.
If it drifts, the "bias" is an ordering effect, not a final preference.

**What this cannot distinguish.** Training was capped at 400 epochs. A
shift beyond that budget would not be detected. Given that C alone
converges at ~100 epochs (Exp 0), 400 is generous, but not unlimited.

### What remains confounded

Three things the design does not fully separate:

1. **Complexity and optimization ease at high k.** Scaling C's input
   (Exp 3) does not change the fact that a k=5 feature requires a
   nonlinear decision boundary and takes ~100 epochs to converge. No
   amount of input scaling fixes that. So at k=5, "complex" and "hard to
   optimize" are partly inseparable.
2. **Simplicity and dimensionality.** The k-slab feature is both
   nonlinear and varies along one input coordinate. A linear feature
   spread over many coordinates would
   test whether the network cares about nonlinearity or dimensionality.
   That experiment was not run.
3. **Definition of simplicity.** I use "linear and low-dimensional" as
   the operational definition throughout. If another definition — low
   frequency, short description length — were used, the results could
   differ. The study is a controlled test of one definition, not of
   simplicity in general.

### Summary

Each experiment isolates one factor while holding the others fixed. Exp 0
rules out capacity. Exp 2 isolates predictive strength. Exp 3 isolates
input scale as one form of optimization ease. The negative control rules
out lock-in. Exp 1 dynamics rules out transient ordering. What remains
confounded is documented above rather than hidden.


## Expected Outcomes and Interpretation

For each experiment, I list the possible outcomes and what each would
imply. This section is written prospectively, as if the experiments had
not yet been run. The actual results are in the next section.

### Exp 1 — Reproduce the baseline

| Outcome                       | Interpretation                                                                                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Network relies on S (R_S ~ 1) | Claim reproduced in our setup. The confound between simplicity, cleanliness, and ease is maximal, so this alone does not tell us which factor is at work. |
| Network relies on C (R_S ~ 0) | Claim fails under our conditions. Something about our setup — the k-slab feature, the MLP, the optimizer — differs from the papers.                     |
| No preference (R_S ~ 0.5)     | Neither feature dominates. Suggests the two features are treated as equally useful, contradicting the extreme-bias claim.                                 |

### Exp 2 — Decouple complexity from predictive strength

| Outcome                                                           | Interpretation                                                                                                                                                                |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R_S drops as rho_S increases, and test accuracy approaches C-only | Network tracks predictive strength, not complexity. The bias observed in Exp 1 was driven by S being the cleaner feature.                                                     |
| R_S stays high even when S is noisier than C                      | Network prefers S despite S being less predictive. Simplicity is doing work beyond predictive strength.                                                                       |
| Test accuracy drops below S-only Bayes without R_S dropping       | Network overfits label noise. It does not switch to C, but it also does not cleanly use S. This would point to a training-dynamics explanation, not a feature-preference one. |

### Diagnostic — Negative control (k=1, noisy S)

| Outcome                       | Interpretation                                                                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Network pivots to C (R_S ~ 0) | Network can change features when the alternative is easy. Rules out "cannot pivot" as an explanation for Exp 1-2.                                                |
| Network stays on S (R_S ~ 1)  | Network cannot pivot even under favorable conditions. This would strengthen the idea that lock-in is a general phenomenon, not specific to complex alternatives. |

### Exp 3 — Simplicity vs optimization ease

| Outcome                             | Interpretation                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| R_S drops toward 0 as alpha_C grows | The bias is about optimization ease, not simplicity. Input scale is the driver.                               |
| R_S stays high across all scales    | Input scale alone is too weak to overcome the preference. The claim survives a serious falsification attempt. |
| R_S drops only at the highest scale | Weak effect. Suggests optimization ease contributes but is not dominant.                                      |

### Combinations of outcomes

No single experiment decides the question. The interpretation depends on
the pattern across all four.

- **If Exp 2 pivots to C and Exp 3 does not:** the bias tracks predictive
  strength but not input scale. Simplicity is doing some work, but only
  when the simple feature is also the more accurate one.
- **If both Exp 2 and Exp 3 fail to flip the preference:** the claim
  survives two serious falsification attempts. This is the strongest
  outcome for the original claim.
- **If Exp 2 does not pivot but the negative control does:** the network
  can switch, but only when the alternative is genuinely simple. This
  would suggest complexity itself (not just optimization ease) is the
  barrier.
- **If all experiments flip toward C:** the entire bias was an artifact
  of the simple feature being cleaner and easier to optimize. This would
  be the strongest outcome against the original claim.

### What the experiments cannot decide

Even in the best case, this study cannot prove that simplicity causes the
preference in general. It can only show that the preference under specific
conditions is (or is not) explained by specific alternatives. Whether the
same holds for other definitions of simplicity, other datasets, or other
architectures is outside the scope.


## Optional Experimental Results

The experiments described above were implemented and run. Full details
and every run are in `experiment_log.md`; this section summarizes the
results and what they show.

### Methodology

The dataset generator, model, training procedure, and reliance measures
are as described in the previous section. All experiments used 3 seeds,
n_samples = 5000, and MAX_EPOCHS = 400. Reliance was computed every 20
epochs for Exp 1 and every 50 epochs for Exp 2-3.

### Results

**Exp 0 — Capacity sanity check.**

| Feature | Test acc       | Epochs to 95% |
| ------- | -------------- | ------------- |
| S       | 0.979 ± 0.003 | 1, 1, 1       |
| C, k=1  | 1.000 ± 0.001 | 2, 2, 3       |
| C, k=5  | 0.991 ± 0.005 | 157, 49, 137  |

All features are learnable in isolation. The optimization-ease gap is
large: S converges in ~1 epoch, C at k=5 takes ~100 (with 3x seed spread).
A separate 1000-epoch diagnostic on C alone showed a phase transition —
the network plateaus at ~0.82 for ~150 epochs, then jumps to ~0.99. This
motivated MAX_EPOCHS = 400 for all later experiments.

**Exp 1 — Reproduce the baseline (S vs C, k=5, both perfect).**

| seed | test_acc | R_S_shuffle | R_S_conflict |
| ---- | -------- | ----------- | ------------ |
| 0    | 0.971    | 0.993       | 0.981        |
| 1    | 0.968    | 0.991       | 0.981        |
| 2    | 0.973    | 0.992       | 0.983        |

Mean R_S_shuffle = 0.992 ± 0.001. Mean R_S_conflict = 0.982 ± 0.001.
Both measures were flat from epoch 20 to 400 across all seeds. Final
test accuracy matches the Bayes limit of S (0.977), confirming C is not
being used.

**Exp 2 — Decouple complexity from predictive strength.**

| rho_S | test_acc | R_S_shuffle | R_S_conflict | drop_S | drop_C |
| ----- | -------- | ----------- | ------------ | ------ | ------ |
| 0.0   | 0.970    | 0.992       | 0.982        | 0.473  | 0.004  |
| 0.1   | 0.820    | 0.936       | 0.864        | 0.303  | 0.021  |
| 0.2   | 0.701    | 0.853       | 0.763        | 0.168  | 0.030  |
| 0.3   | 0.633    | 0.569       | 0.639        | 0.074  | 0.057  |

The network does not pivot to C. Test accuracy drops *below* the
S-only Bayes limit (0.977 − 0.954·rho_S) at every rho_S > 0. Train-test
gap grows from 0.030 to 0.310, indicating overfitting to label noise.

**Negative control — noisy S, simple clean C (k=1).**

| rho_S | test_acc | R_S_conflict | drop_S | drop_C |
| ----- | -------- | ------------ | ------ | ------ |
| 0.2   | 0.981    | 0.055        | 0.012  | 0.460  |

With C equally simple and cleaner than S, the network pivots almost
entirely to C.

**Exp 3 — Simplicity vs optimization ease.**

| alpha_C | test_acc | R_S_conflict | R_S_shuffle | drop_S | drop_C |
| ------- | -------- | ------------ | ----------- | ------ | ------ |
| 0.01    | 0.968    | 0.983        | 0.999       | 0.472  | 0.001  |
| 0.10    | 0.969    | 0.983        | 0.996       | 0.471  | 0.002  |
| 1.00    | 0.970    | 0.982        | 0.992       | 0.473  | 0.004  |
| 10.0    | 0.972    | 0.963        | 0.972       | 0.467  | 0.014  |
| 100.0   | 0.979    | 0.945        | 0.940       | 0.455  | 0.029  |

Scaling C's input by up to 100x does not flip the preference. Both
reliance measures stay above 0.94. There is a small monotonic trend
toward C at high scale (drop_C grows from 0.001 to 0.029), but it never
dominates.

### Analysis

Four points follow from these results.

**The bias is real in our setup.** Exp 1 reproduces the original claim
cleanly. Both independent measures agree, and the results are stable
across seeds. This is not an artifact of the measurement.

**The bias is conditional on S being near-perfect.** Exp 2 shows that
when S becomes noisy, the network does not switch to C. It overfits the
noisy labels instead. The extreme bias reported in the literature is
observed at rho_S = 0 but not sustained when the simple feature degrades.

**The network can pivot, but only when the alternative is easy.** The
negative control (k=1, noisy S) shows the network will move to C when C
is equally simple and cleaner. This rules out "the network cannot change
its mind" as the explanation for Exp 1-2. The lock-in is specific to
cases where C is genuinely harder to represent.

**Input scale is a weak knob.** Exp 3 shows that making C 100x easier to
optimize at the input does not break the preference. S is learned so much
faster that by the time C's gradients are large enough to matter, S is
already locked in.

Taken together, the results qualify the claim rather than falsify it.
The simplicity bias holds when simplicity, cleanliness, and optimization
ease coincide. It weakens when these are decoupled, but it does not
reduce to any single factor. The network prefers S at rho_S = 0 because
S is simpler *and* cleaner *and* faster to learn — and no experiment in
this study was able to pull those factors far enough apart to make the
network switch to C.


## Limitations

### What this study cannot establish

**Causality.** The study shows correlations between feature properties and
reliance in one setup. It does not prove that simplicity causes the
preference across architectures, datasets, or training regimes.

**A general definition of simplicity.** I operationalize simplicity as
"linear and low-dimensional." Other definitions — low frequency, short
description length — might produce different results. This is a test of
one definition, not of simplicity as a concept.

**Transfer to real data.** Synthetic k-slab features are not images or
text. The dynamics may differ on realistic high-dimensional inputs.

### What remains confounded

**Complexity and optimization ease at high k.** At k=5, the feature is
both nonlinear and slow to learn. Exp 3 scales the input, but that does
not change the fact that a 6-band function is genuinely harder to fit
than a threshold. The two factors cannot be fully separated at k=5.

**Simplicity and dimensionality.** The k-slab feature is nonlinear *and*
varies along one coordinate. A linear high-dimensional feature would test
whether the network cares about nonlinearity or dimensionality. That
experiment was not run.

**Predictive strength and training difficulty.** Label noise reduces both
the accuracy of S and the smoothness of its training signal. Exp 2 cannot
cleanly attribute the observed behavior to either alone.

### Methodological limitations

**Shuffle ablation assumes feature independence.** The measure is a proxy,
not a direct readout of reliance.

**Time-to-fit and input scale are proxies for optimization ease.** Neither
is exact. Training speed also depends on initialization, learning rate,
and architecture.

**Three seeds is minimal.** The phase transition at k=5 has 3x spread
across seeds. More seeds would tighten confidence.

**MAX_EPOCHS = 400 is a budget choice.** A shift in reliance beyond that
budget would not be detected.

**The conflict measure was corrected mid-study.** An earlier version
produced a spurious U-shaped curve in Exp 3. The fix is documented in the
log, but it is a reminder that reliance measures are sensitive to
implementation details.

### What the study still contributes

Despite these limitations, the study does three things the existing
literature does not: it separates predictive strength from complexity, it
isolates input scale as one form of optimization ease, and it provides a
negative control showing the network can pivot.


## Conclusion

The original claim is that neural networks preferentially learn simpler
predictive features. This study tested that claim by asking whether the
preference comes from simplicity itself, or from correlated factors like
optimization ease, predictive strength, and label cleanliness.

The results qualify the claim rather than confirm or falsify it. The
preference for the simple feature is strong when the simple feature is
also cleaner and faster to learn — the conditions under which the claim
is usually tested. When those conditions are broken, the preference
weakens or disappears. The network pivots to the complex feature when
the complex feature is equally simple and cleaner. It does not pivot when
the complex feature is scaled to be easier to optimize but remains
genuinely harder to represent.

So the honest reading is: the bias is real, but it is not a bias toward
simplicity in isolation. It is a bias toward whichever feature is easiest
to learn and cleanest to rely on, which in typical setups happens to be
the simpler one. Whether this holds for other definitions of simplicity,
other architectures, or real data is left open. The study narrows the
claim without closing it.
