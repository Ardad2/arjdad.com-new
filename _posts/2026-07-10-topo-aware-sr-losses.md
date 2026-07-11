---
layout: post
title: Topology Aware (Merge Trees and Persistence Diagrams) losses for scalar field neural interpolation (Wind Field SR)
date: 2026-07-10 14:24:00
description: Notes on topology-aware losses for wind-field super-resolution.
tags: topology tda super-resolution machine-learning
categories: research
---

# Candidate Loss Summary

## Kissi et al. (2025) Approach (Topology-Aware Neural Interpolation)

TTK's PD distances are not differentiable, so Kissi et al. work around this as follows:

1. Use topology extraction (TTK) to identify meaningful critical vertices, persistence pairs, and target critical values.
2. Use differentiable neural losses (PyTorch) evaluated at those selected locations.
3. Combine geometric losses with topology-derived losses.

This is the inspiration for Candidate C and, eventually, Candidate E. (Candidate C showed better results and was eventually picked.)

## Shared Setup

All candidates start from the pretrained PhIRE CNN for wind-field SR. The network predicts normalized vector components $\hat u,\hat v$. For scalar topology and physical losses, I denormalize the vector field and compute scalar wind speed:

$$
u_{\mathrm{phys}}=\sigma_u u_{\mathrm{norm}}+\mu_u,\quad
v_{\mathrm{phys}}=\sigma_v v_{\mathrm{norm}}+\mu_v
$$

$$
s(x,y)=\sqrt{u(x,y)^2+v(x,y)^2}.
$$

The base reconstruction loss is

$$
L_{uv}=\mathrm{MSE}([\hat u_{\mathrm{norm}},\hat v_{\mathrm{norm}}],
[u_{\mathrm{norm}},v_{\mathrm{norm}}]).
$$

This keeps the model faithful to the vector field, not only the scalar speed magnitude. Scalar speed alone doesn't determine direction (e.g. $(10,0)$ and $(0,10)$ have the same speed but very different fields).

## Candidate UV: Vector-Only Control (Ablation)

$$
L_{\mathrm{UV}}=L_{uv}.
$$

This is used as a control/ablation to check whether any gains from the other candidates are simply an artifact of fine-tuning itself. It improves some direct fidelity metrics, but does not reproduce the PD gains seen with Candidate C.

## Candidate B: Scalar-Field Proxy Losses

$$
L_B=L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}+0.25L_{\mathrm{levelset}}.
$$

where

$$
L_{\mathrm{speed}}=\mathrm{MSE}(s_{\mathrm{SR}},s_{\mathrm{GT}})
$$

preserves scalar speed, and

$$
L_{\mathrm{grad}}=\mathrm{MSE}(|\nabla s_{\mathrm{SR}}|,|\nabla s_{\mathrm{GT}}|)
$$

preserves local ridges/fronts.

The soft level-set term is a differentiable mask over high-speed wind regions, so the model is rewarded for placing high-wind regions in the right _places_, not just matching speed values pointwise, while still supplying usable gradients:

$$
M_\tau(s)=\sigma(k(s-\tau)),\quad k=10,\quad \tau\in\{5,10,15\}\text{ m/s},
$$

$$
L_{\mathrm{levelset}}=
\frac{1}{|T|}\sum_{\tau\in T}
\mathrm{MSE}(M_\tau(s_{\mathrm{SR}}),M_\tau(s_{\mathrm{GT}})).
$$

- A hard superlevel set at threshold $\tau$ (the region $\{x : s(x)\geq\tau\}$) is a step function, so it's not differentiable. $M_\tau$ softens it with a sigmoid: near 1 well above $\tau$, near 0 well below, smooth in between. So it can be trained with gradient descent.
- $L_{\mathrm{levelset}}$ averages this soft-mask MSE across $\tau\in\{5,10,15\}$ m/s.

**Note:** this was written from the superlevel side; I later realized TTK's default sublevel convention was switched on. That's not an inconsistency in the loss itself, since $\sigma(-z)=1-\sigma(z)$, the soft sublevel mask is exactly $1-M_\tau(s)$, so the two MSE losses are identical as long as SR and GT use the same convention. What does change is the critical-point interpretation: high-speed local maxima are births of high-speed components under superlevel, but under the sublevel convention those same maxima are high critical values, often on the death side of the persistence pair.

This is topology-inspired because PDs and MTs study how thresholded regions appear, merge, and disappear.

## Candidate C: High-Speed Extrema / Critical-Value Proxy (**Final Choice**)

$$
L_C=L_B+0.001L_{\mathrm{crit}}.
$$

Inspired by Kissi et al., Candidate C treats prominent high-speed local maxima as proxies for persistence-relevant critical values. I select GT pixels that are local maxima in a $3\times3$ neighborhood and exceed an adaptive high-speed threshold:

$$
s_{\mathrm{GT}}(x,y)\geq \mu_s+\sigma_s.
$$

At those fixed GT-selected locations, I penalize scalar-speed error:

$$
L_{\mathrm{crit}}=
\frac{
\sum_{x,y}M_{\max}(x,y)
(s_{\mathrm{SR}}(x,y)-s_{\mathrm{GT}}(x,y))^2
}{
\max(\sum_{x,y}M_{\max}(x,y),1)
}.
$$

In a superlevel view, these are births of high-speed components; in the sublevel convention, they are still important high critical values, often appearing on the death side of persistence pairs.

Candidate C was the strongest practical proxy. It improved PD robustly and gave encouraging MT behavior in the 168-sample pilot, but the MT gains were less consistent at larger training scales.

## Candidate D: Differentiable PD Loss

Candidate D used a small residual refiner on top of the frozen CNN output:

$$
\hat u_D=\hat u_{\mathrm{CNN}}+\eta R_\theta(\hat u_{\mathrm{CNN}}),\quad \eta=0.1.
$$

The form was

$$
L_{Dpd}=L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}+0.001L_{\mathrm{crit}}
+0.001904L_{\mathrm{PD}}.
$$

Here $L_{\mathrm{PD}}$ was a differentiable Wasserstein-style persistence-diagram loss computed in PyTorch on cubical-complex persistence diagrams.

Intuition: put a differentiable PD loss directly inside training, rather than only using TTK MT distance after training.

This was feasible, but it did not improve the final TTK PD/MT metrics. Still exploring the reasons, could maybe due to mismatch between the type of PD distances used for training vs evaluation (Wasserstein vs Bottleneck) or something else. Did not investigate further due to time restriction.

## Candidate E: TTK Critical-Pair Supervision

Candidate E follows the Kissi et al. idea more closely. The key idea is not to differentiate through TTK distance directly. Instead:

1. Use TTK offline to identify meaningful persistence-pair vertices.
2. Treat those vertices as fixed spatial indices during training.
3. Use ordinary differentiable scalar-value losses at those locations.

TTK gives persistence pairs

$$
P=\{(b_i,d_i)\},
$$

where $b_i,d_i$ are birth/death vertex locations.

Read targets directly from GT speed:

$$
s^*(b_i)=s_{\mathrm{GT}}(b_i),\quad
s^*(d_i)=s_{\mathrm{GT}}(d_i),
$$

$$
p_i^*=|s_{\mathrm{GT}}(b_i)-s_{\mathrm{GT}}(d_i)|.
$$

The critical-value loss is

$$
L_{\mathrm{TTK-CV}}=
\frac{1}{2|P|}
\sum_{(b_i,d_i)\in P}
\left[
(\hat s(b_i)-s^*(b_i))^2+
(\hat s(d_i)-s^*(d_i))^2
\right],
$$

and the persistence-gap loss is

$$
L_{\mathrm{TTK-pers}}=
\frac{1}{|P|}
\sum_{(b_i,d_i)\in P}
\left(
|\hat s(b_i)-\hat s(d_i)|-p_i^*
\right)^2.
$$

The E objective was

$$
L_{E}=L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}
+0.001L_{\mathrm{crit}}+0.25L_{\mathrm{levelset}}
+0.04L_{\mathrm{TTK-CV}}+0.02L_{\mathrm{TTK-pers}}.
$$

Intuition: this is a more faithful topology-derived loss than Candidate C, because TTK selects the birth/death vertices. However, it still only supervises selected scalar values and persistence gaps. It does not encode saddle relationships, branch ancestry, component merge order, or full merge-tree hierarchy.

It did not improve PD as well as Candidate C.

| Method            | Training / setup         | PD mean | MT mean | PD < CNN | MT < CNN |
| ----------------- | ------------------------ | ------: | ------: | -------: | -------: |
| CNN               | baseline                 | 27.4063 |  5.8678 |       -- |       -- |
| GAN               | baseline                 | 20.8641 |  8.3481 |  166/168 |   20/168 |
| C-168             | TF PhIRE fine-tune       | 27.0021 |  5.7141 |  120/168 |  102/168 |
| C-672             | TF PhIRE fine-tune       | 23.9580 |  6.0765 |  168/168 |   54/168 |
| C-1344            | TF PhIRE fine-tune       | 22.8623 |  6.1236 |  168/168 |   49/168 |
| C-2688            | TF PhIRE fine-tune       | 22.4944 |  6.0803 |  168/168 |   52/168 |
| E2-low-672 fixed  | PyTorch residual refiner | 27.1011 |  5.7522 |  130/168 |  113/168 |
| E2-low-1344 fixed | PyTorch residual refiner | 26.9905 |  5.7102 |  128/168 |  120/168 |
| E2-low-2688 fixed | PyTorch residual refiner | 26.4934 |  5.6989 |  142/168 |  118/168 |

## Summary of Topology Results

Lower is better for both PD and MT. "PD < CNN" and "MT < CNN" count how often the candidate improved over the pretrained CNN on the fixed 168-sample benchmark.

### 168-Sample Pilot / Topology-Candidate Comparison

| Method       | Main idea                                            | PD mean | MT mean | PD < CNN | MT < CNN |
| ------------ | ---------------------------------------------------- | ------: | ------: | -------: | -------: |
| CNN baseline | Pretrained fidelity-oriented CNN                     | 27.4063 |  5.8678 |       -- |       -- |
| GAN baseline | Pretrained adversarial model                         | 20.8641 |  8.3481 |  166/168 |   20/168 |
| Candidate UV | Vector-only fine-tuning, $L_{uv}$                    | 30.5098 |  6.3862 |    0/168 |   26/168 |
| Candidate B  | Speed + gradient + soft level-set losses             | 26.1794 |  6.0186 |  136/168 |   66/168 |
| Candidate C  | Candidate B + high-speed local-maxima proxy          | 27.0021 |  5.7141 |  120/168 |  102/168 |
| Candidate D  | Residual-refiner audit; original PD weight was zero  | 27.9510 |  5.9837 |    1/168 |   50/168 |
| Candidate E  | TTK birth/death vertices + fixed-index scalar losses | 28.1715 |  6.0618 |    1/168 |   35/168 |

The 168-sample pilot made Candidate C look most promising for MT: it improved mean MT relative to CNN. However, this result needed a stronger expanded-data check, since the same 168 samples were also used for fine-tuning.

### Expanded 672-Sample Ablation Ladder

All expanded variants below were trained on the same 672 non-overlapping seasonal samples and evaluated on the original fixed 168-sample benchmark.

| Method          | Objective summary                                                   | PD mean | MT mean | PD < CNN | MT < CNN |
| --------------- | ------------------------------------------------------------------- | ------: | ------: | -------: | -------: |
| CNN baseline    | Released CNN                                                        | 27.4063 |  5.8678 |       -- |       -- |
| GAN baseline    | Released GAN                                                        | 20.8641 |  8.3481 |  166/168 |   20/168 |
| UV-expanded-672 | $L_{uv}$                                                            | 29.8747 |  6.2891 |    6/168 |   43/168 |
| B-expanded-672  | $L_{uv}+L_{\mathrm{speed}}+L_{\mathrm{grad}}+L_{\mathrm{levelset}}$ | 23.7094 |  6.3124 |  167/168 |   36/168 |
| C-expanded-672  | Candidate B + $L_{\mathrm{crit}}$                                   | 23.9580 |  6.0765 |  168/168 |   54/168 |
| D-expanded-672  | Candidate C + active differentiable PD loss                         | 28.7656 |  6.2210 |    0/168 |   23/168 |
| E-expanded-672  | Candidate C + TTK critical-value/persistence-gap losses             | 28.6697 |  6.1622 |    0/168 |   24/168 |

The expanded ablation changed the story. Candidate B and Candidate C strongly improved PD, showing that scalar-speed, gradient, level-set, and high-speed-extrema proxies can influence persistence-style structure. However, the MT improvements were much weaker and did not beat the CNN baseline on mean MT. Candidate D and E were feasible but did not improve final TTK PD/MT, suggesting that differentiable PD losses or fixed birth/death value constraints are not automatically aligned with the final topology evaluators.

---

## Candidate E Repair Audit and Repaired E2 Ablation History

This section records the post-hoc repair/audit sequence for Candidate E. It should be read separately from the original submitted paper story: Candidate C remains the main submitted topology-inspired fine-tuning result, while the repaired E2 runs are follow-up experiments designed to understand whether TTK-derived critical-pair supervision can be made useful once implementation and objective-balance issues are corrected.

### Why the original Candidate E was not a clean test

The original Candidate E idea was scientifically reasonable: use TTK offline to identify persistence-pair birth/death vertices, then use differentiable neural losses at those fixed spatial locations. However, the first E-family implementation mixed several confounders:

| Issue                           | What happened originally                                                                                                                 | Repair / correction                                                                                                                                  | Why it matters                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Vector loss normalization       | Some PyTorch E/D scripts used physical-unit `[u,v]` arrays in `L_uv` instead of the normalized vector convention used by PhIRE training. | Repaired scripts normalize `[u,v]` before computing `L_uv`; speed/topology losses remain in physical units.                                          | Keeps the base reconstruction term comparable to Candidate B/C and the pretrained CNN objective.                                   |
| VTI / vertex mapping            | Early VTI writing used a flattening convention inconsistent with VTK point IDs.                                                          | Repaired VTI writing uses C-order flattening so point ID `vid = ix + iy * W` matches `iy = vid // W`, `ix = vid % W`.                                | Ensures TTK birth/death vertex IDs refer to the intended pixels in the SR/GT scalar field.                                         |
| Target critical values          | Early E used diagram-encoded birth/death coordinates as targets rather than reading actual GT speed at the selected vertex IDs.          | Repaired E2 uses `s_GT(b_i)` and `s_GT(d_i)` directly as target scalar values.                                                                       | Makes the fixed-index critical-value loss physically meaningful.                                                                   |
| Loss scale                      | The original high TTK weights, especially `0.04 L_TTKCV + 0.02 L_TTKpers`, were too aggressive after repairs.                            | Low-lambda repaired E2 uses `0.004 L_TTKCV + 0.002 L_TTKpers`.                                                                                       | Avoids large scalar-field distortions while still giving TTK constraints influence.                                                |
| Architecture confound           | Early repaired E2 was implemented as a PyTorch residual refiner on frozen CNN outputs.                                                   | Later E2 was implemented inside the native PhIRE/TensorFlow fine-tuning path.                                                                        | Separates "does the repaired loss work?" from "does a residual post-processor behave differently from full generator fine-tuning?" |
| Topology extraction reliability | TTK MT extraction could segfault under repeated full reruns, especially with many threads.                                               | The topology pipeline was patched to skip completed outputs, retry only missing fields, hard-gate incomplete extraction, and run with `--threads 1`. | Makes full 168-sample PD/MT evaluation reproducible for repaired candidates.                                                       |

### Repaired E2 loss definitions

The repaired E2 terms keep the same conceptual structure as Candidate E but correct the target convention and implementation details. For each TTK persistence pair `(b_i, d_i)`, where `b_i` and `d_i` are fixed vertex indices selected from GT topology:

$$
L_{\mathrm{TTKCV}} =
\frac{1}{2|P|}
\sum_{(b_i,d_i)\in P}
\left[
(\hat{s}(b_i)-s_{\mathrm{GT}}(b_i))^2 +
(\hat{s}(d_i)-s_{\mathrm{GT}}(d_i))^2
\right],
$$

and

$$
L_{\mathrm{TTKpers}} =
\frac{1}{|P|}
\sum_{(b_i,d_i)\in P}
\left(
|\hat{s}(b_i)-\hat{s}(d_i)| -
|s_{\mathrm{GT}}(b_i)-s_{\mathrm{GT}}(d_i)|
\right)^2.
$$

The repaired low-lambda E2 terms use

$$
0.004L_{\mathrm{TTKCV}} + 0.002L_{\mathrm{TTKpers}}.
$$

### Stage 1 — Original E / expanded E negative result

The first expanded E run used the Candidate C stack plus TTK fixed-index critical-value and persistence-gap terms:

$$
L_E =
L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}+0.001L_{\mathrm{crit}}
+0.04L_{\mathrm{TTKCV}}+0.02L_{\mathrm{TTKpers}}.
$$

On the 672-sample expanded setup, this did not improve the final TTK topology metrics:

| Method         | Training path          | Main difference                     | PD mean | MT mean | PD < CNN | MT < CNN |
| -------------- | ---------------------- | ----------------------------------- | ------: | ------: | -------: | -------: |
| CNN baseline   | pretrained PhIRE CNN   | no fine-tuning                      | 27.4063 |  5.8678 |       -- |       -- |
| C-expanded-672 | TF PhIRE fine-tune     | Candidate B + `L_crit`              | 23.9580 |  6.0765 |  168/168 |   54/168 |
| E-expanded-672 | initial E-family setup | Candidate C + high-weight TTK terms | 28.6697 |  6.1622 |    0/168 |   24/168 |

This result should not be interpreted as "TTK supervision does not work." It was later found to be confounded by implementation details and loss-scale issues.

### Stage 2 — Repaired E2 constraints and PyTorch residual refiner

After repairing the VTI coordinate mapping, target convention, and normalized `L_uv`, the first successful repaired tests used a PyTorch residual refiner on top of frozen CNN outputs. This was not an apples-to-apples PhIRE fine-tuning comparison, but it was a useful diagnostic because it showed that repaired TTK critical-pair supervision carries a real MT-oriented signal.

| Method            | Architecture             | Training scale | PD mean | MT mean | PD < CNN | MT < CNN | MT-GAN cases recovered |
| ----------------- | ------------------------ | -------------: | ------: | ------: | -------: | -------: | ---------------------: |
| CNN baseline      | pretrained CNN           |             -- | 27.4063 |  5.8678 |       -- |       -- |                     -- |
| E2-low-672 fixed  | PyTorch residual refiner |            672 | 27.1011 |  5.7522 |  130/168 |  113/168 |                   6/20 |
| E2-low-1344 fixed | PyTorch residual refiner |           1344 | 26.9905 |  5.7102 |  128/168 |  120/168 |                  11/20 |
| E2-low-2688 fixed | PyTorch residual refiner |           2688 | 26.4934 |  5.6989 |  142/168 |  118/168 |                  11/20 |

Interpretation: the repaired residual E2 family did not close the PD gap to GAN, but it consistently improved MT relative to CNN. This was the first strong sign that TTK critical-pair constraints were more merge-tree-oriented than persistence-diagram-oriented in this setup.

### Stage 3 — Native PhIRE/TensorFlow C+E2-low

To remove the residual-refiner architecture confound, repaired E2 was then implemented in the native PhIRE/TensorFlow generator fine-tuning path. The first native run kept Candidate C's `L_crit` term and added repaired low-lambda TTK terms:

$$
L_{\mathrm{C+E2}} =
L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}
+0.001L_{\mathrm{crit}}
+0.004L_{\mathrm{TTKCV}}
+0.002L_{\mathrm{TTKpers}}.
$$

This run is best named **TF C+E2-low-672**, not pure Candidate E, because it includes Candidate C's local-maxima proxy.

| Method          | Architecture           | PD mean | MT mean | PD < CNN | MT < CNN | PD beats GAN | MT beats GAN | MT-GAN cases recovered |
| --------------- | ---------------------- | ------: | ------: | -------: | -------: | -----------: | -----------: | ---------------------: |
| CNN baseline    | pretrained CNN         | 27.4063 |  5.8678 |       -- |       -- |           -- |           -- |                     -- |
| GAN baseline    | pretrained GAN         | 20.8641 |  8.3481 |  166/168 |   20/168 |           -- |           -- |                     -- |
| TF C+E2-low-672 | native PhIRE fine-tune | 24.9844 |  5.7076 |  162/168 |   96/168 |        4/168 |      160/168 |                  14/20 |

Interpretation: adding repaired E2 inside the native PhIRE fine-tuning path produced a balanced result. It preserved much of Candidate C's PD improvement while substantially improving MT relative to the CNN baseline. However, because `L_crit` was still present, this run could not prove whether the MT gain came from the repaired TTK terms or from interaction with Candidate C's local-maxima proxy.

### Stage 4 — Clean B+E2-low ablation with `L_crit` removed and scaled to 2688

The final clean E2 ablation removed Candidate C's `L_crit` term and kept only the Candidate B stack plus repaired low-lambda TTK terms:

$$
L_{\mathrm{B+E2}} =
L_{uv}+0.01L_{\mathrm{speed}}+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}
+0.004L_{\mathrm{TTKCV}}
+0.002L_{\mathrm{TTKpers}}.
$$

In the implementation, the raw `L_crit` op was still built for diagnostics, but its weight was exactly zero:

$$
\lambda_{\mathrm{crit}} = 0.
$$

This makes **TF B+E2-low** the cleanest repaired E2 ablation, because it tests whether the repaired TTK fixed-index losses themselves carry the MT signal. After the initial 672-sample B+E2 run succeeded, the same ablation was scaled to 1344 and 2688 training samples to compare directly with the Candidate C scale ladder.

| Method           | Architecture           | Loss summary                    |     PD mean |    MT mean | PD < CNN | MT < CNN | PD beats GAN | MT beats GAN | MT-GAN cases recovered |
| ---------------- | ---------------------- | ------------------------------- | ----------: | ---------: | -------: | -------: | -----------: | -----------: | ---------------------: |
| CNN baseline     | pretrained CNN         | --                              |     27.4063 |     5.8678 |       -- |       -- |           -- |           -- |                     -- |
| GAN baseline     | pretrained GAN         | --                              |     20.8641 |     8.3481 |  166/168 |   20/168 |           -- |           -- |                     -- |
| C-expanded-672   | native PhIRE fine-tune | B + `0.001 L_crit`              |     23.9580 |     6.0765 |  168/168 |   54/168 |       20/168 |      155/168 |                   7/20 |
| C-expanded-1344  | native PhIRE fine-tune | B + `0.001 L_crit`              |     22.8623 |     6.1236 |  168/168 |   49/168 |           -- |           -- |                   9/20 |
| C-expanded-2688  | native PhIRE fine-tune | B + `0.001 L_crit`              |     22.4944 |     6.0803 |  168/168 |   52/168 |       20/168 |           -- |                  10/20 |
| TF C+E2-low-672  | native PhIRE fine-tune | C + low-lambda TTK              |     24.9844 |     5.7076 |  162/168 |   96/168 |        4/168 |      160/168 |                  14/20 |
| TF B+E2-low-672  | native PhIRE fine-tune | B + low-lambda TTK, no `L_crit` |     24.7596 |     5.7161 |  162/168 |   99/168 |        4/168 |      164/168 |                  16/20 |
| TF B+E2-low-1344 | native PhIRE fine-tune | B + low-lambda TTK, no `L_crit` |     24.4965 | **5.6514** |  160/168 |   97/168 |        4/168 |      166/168 |              **18/20** |
| TF B+E2-low-2688 | native PhIRE fine-tune | B + low-lambda TTK, no `L_crit` | **23.9876** |     5.6774 |  166/168 |   90/168 |        8/168 |      166/168 |              **18/20** |

### Cheap scalar/domain metrics for TF B+E2-low scale ladder

Before the expensive TTK runs, the cheap scalar/domain evaluations indicated that all B+E2 scale-up runs were safe to evaluate topologically. All three B+E2 scales improved direct fidelity, scalar speed error, WPD metrics, gradient error, exceedance behavior, and component-count proxy behavior relative to the CNN baseline. The main repeated caveat is that PSD slope error worsened relative to CNN.

| Metric                   |        CNN |      GAN | TF B+E2-low-672 | TF B+E2-low-1344 | TF B+E2-low-2688 | Direction        |
| ------------------------ | ---------: | -------: | --------------: | ---------------: | ---------------: | ---------------- |
| PSNRuv (dB)              |    31.1925 |  29.1380 |         32.1146 |          32.4480 |      **32.6889** | higher is better |
| Speed MAE (m/s)          |     0.6941 |   0.9026 |          0.6178 |           0.5884 |       **0.5684** | lower is better  |
| Speed RMSE (m/s)         |     1.1078 |   1.3775 |          1.0240 |           0.9852 |       **0.9629** | lower is better  |
| WPD MAE                  |   231.6709 | 310.7328 |        202.2449 |         191.8304 |     **183.4170** | lower is better  |
| WPD W1                   |    45.2713 |  85.6191 |         19.5757 |          19.3128 |      **16.0868** | lower is better  |
| WPD bias abs             |    35.3439 |  78.7236 |         13.1372 |      **11.5541** |          12.6851 | lower is better  |
| Gradient MAE             |     0.3491 |   0.3806 |          0.3212 |           0.3180 |       **0.3120** | lower is better  |
| Gradient W1              |     0.2329 |   0.0564 |          0.1492 |           0.1654 |           0.1543 | lower is better  |
| Gradient kurtosis abs Δ  |     3.7004 |   4.2010 |          3.0060 |       **2.9293** |           2.9747 | lower is better  |
| PSD log-L2               |     0.8335 |   0.5139 |      **0.8096** |           0.8560 |           0.8509 | lower is better  |
| PSD slope abs Δ          | **0.9150** |   0.9482 |          1.0791 |           1.1891 |           1.2029 | lower is better  |
| Exceedance abs Δ s>5     |     0.0042 |   0.0082 |          0.0027 |           0.0029 |       **0.0020** | lower is better  |
| Exceedance abs Δ s>10    |     0.0066 |   0.0243 |          0.0037 |           0.0027 |       **0.0026** | lower is better  |
| Exceedance abs Δ s>15    |     0.0062 |   0.0096 |          0.0018 |           0.0023 |       **0.0018** | lower is better  |
| Exceedance abs Δ p90     |     0.0103 |   0.0123 |          0.0032 |           0.0031 |       **0.0028** | lower is better  |
| Component-count curve L1 |   115.5278 | 124.6567 |         84.1935 |          91.1538 |      **88.0060** | lower is better  |

Pairwise improvement counts relative to CNN also remained strong:

| Metric                   | TF B+E2-low-672 improved / 168 | TF B+E2-low-1344 improved / 168 | TF B+E2-low-2688 improved / 168 |
| ------------------------ | -----------------------------: | ------------------------------: | ------------------------------: |
| PSNRuv                   |                            168 |                             168 |                             168 |
| Speed MAE                |                            168 |                             168 |                             168 |
| Speed RMSE               |                            168 |                             168 |                             168 |
| WPD MAE                  |                            168 |                             168 |                             168 |
| WPD W1                   |                            156 |                             139 |                             157 |
| WPD bias abs             |                            152 |                             124 |                             140 |
| Gradient MAE             |                            168 |                             168 |                             168 |
| Gradient W1              |                            162 |                             155 |                             162 |
| Exceedance abs Δ s>5     |                            116 |                             118 |                             132 |
| Exceedance abs Δ s>10    |                            117 |                             123 |                             134 |
| Exceedance abs Δ s>15    |                            137 |                             124 |                             141 |
| Exceedance abs Δ p90     |                            155 |                             136 |                             148 |
| Component-count curve L1 |                            155 |                             153 |                             155 |

### B+E2-low scale-ladder interpretation

The B+E2-low scale ladder gives a clearer conclusion than the original Candidate E result:

1. **The repaired TTK fixed-index terms are not a one-off 672-sample artifact.** Scaling to 1344 and 2688 preserved the MT-oriented behavior.
2. **Candidate C remains stronger for PD at large scale.** C-2688 reaches PD mean 22.4944, while B+E2-low-2688 reaches 23.9876.
3. **B+E2-low is consistently stronger for MT.** All B+E2-low scales have lower mean MT than Candidate C at the corresponding or nearby scales.
4. **The best B+E2 PD result is at 2688.** PD improves from 24.7596 at 672 to 23.9876 at 2688.
5. **The best B+E2 MT mean is at 1344.** MT improves from 5.7161 at 672 to 5.6514 at 1344, then remains strong at 5.6774 for 2688.
6. **MT-GAN recovery is stable at larger scales.** B+E2-low recovers 16/20 original MT-GAN cases at 672 and 18/20 at both 1344 and 2688.

A compact comparison against Candidate C is:

| Scale | Candidate C PD | Candidate C MT | B+E2-low PD | B+E2-low MT | Main readout                                       |
| ----: | -------------: | -------------: | ----------: | ----------: | -------------------------------------------------- |
|   672 |    **23.9580** |         6.0765 |     24.7596 |  **5.7161** | C better PD; B+E2 better MT                        |
|  1344 |    **22.8623** |         6.1236 |     24.4965 |  **5.6514** | C better PD; B+E2 much better MT                   |
|  2688 |    **22.4944** |         6.0803 |     23.9876 |  **5.6774** | C better PD; B+E2 better MT and improved PD vs 672 |

### Final interpretation of the E repair sequence

The repaired E2 sequence changes the interpretation of Candidate E:

1. The original E result was negative, but it was not a clean test of the TTK critical-pair idea.
2. After correcting the target convention, coordinate mapping, vector-loss normalization, and loss scale, repaired E2 produced stable MT gains.
3. The PyTorch residual refiner showed the first MT-oriented signal, but it had an architecture confound.
4. Native PhIRE TF C+E2-low showed that the repaired losses work inside the actual generator fine-tuning path.
5. Native PhIRE TF B+E2-low removed `L_crit` and still recovered the MT gain, showing that the repaired TTK fixed-index losses themselves are sufficient to drive merge-tree improvement.
6. Scaling B+E2-low to 1344 and 2688 strengthened this conclusion: the MT signal persisted, and the PD mean improved with scale.

A concise advisor-facing summary:

> The repaired E2 audit shows that the original negative Candidate E result was confounded by implementation and loss-scale issues. After fixing the VTI/vertex mapping, using GT speed values at TTK-selected birth/death vertices, restoring normalized `L_uv`, and lowering the TTK weights, E2 becomes a strong MT-oriented signal. The clean B+E2 ablation removes Candidate C's `L_crit` term but still improves mean MT from 5.8678 to 5.7161 at 672 samples, 5.6514 at 1344 samples, and 5.6774 at 2688 samples, while recovering 16/20, 18/20, and 18/20 original MT-GAN cases respectively. This suggests that the repaired TTK fixed-index supervision itself is responsible for the merge-tree improvement.

### What not to overclaim

Do not claim that repaired E2 supersedes Candidate C as the submitted result. Candidate C remains the main submitted controlled result because it has the original matched UV/B/C scale ladder and strong PD behavior. The repaired E2 family should be framed as a post-submission/future-work audit showing that hierarchy-aware or critical-pair-aware supervision is promising, especially for merge-tree structure.

A safe paper/future-work statement is:

> A repaired post-submission audit of TTK critical-pair supervision suggests that fixed-index birth/death value and persistence-gap losses provide a merge-tree-oriented training signal. In a clean B+E2 ablation, removing the local-maxima proxy did not remove the MT improvement, and scaling the ablation to 1344 and 2688 samples preserved the effect. Future work should extend this pointwise critical-pair supervision to neighborhood-aware and branch-aware losses that more directly encode merge-tree structure.

---

## Priority 3 Update — Native TF C+E2-low scale-up to 1344/2688

This update records the next step after the clean B+E2-low scale ladder: scaling the native PhIRE/TensorFlow **C+E2-low** objective to 1344 and 2688 samples. Unlike B+E2-low, this family keeps Candidate C's local-maxima critical-value proxy enabled and adds the repaired low-lambda TTK fixed-index terms.

The objective is

$$
L_{\mathrm{C+E2}}
= L_{uv}
+0.01L_{\mathrm{speed}}
+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}
+0.001L_{\mathrm{crit}}
+0.004L_{\mathrm{TTKCV}}
+0.002L_{\mathrm{TTKpers}}.
$$

The two scale-up scripts were generated from the completed `candidateE2_tf_lowlambda_expanded672` script:

| Script                                                                 | Method name                             | Training samples | Key point                                                      |
| ---------------------------------------------------------------------- | --------------------------------------- | ---------------: | -------------------------------------------------------------- |
| `scripts/run_candidateE2_tf_lowlambda_expanded1344_ttkcrit_refiner.py` | `candidateE2_tf_lowlambda_expanded1344` |             1344 | keeps `LAMBDA_CRIT=0.001`; native PhIRE/TF generator fine-tune |
| `scripts/run_candidateE2_tf_lowlambda_expanded2688_ttkcrit_refiner.py` | `candidateE2_tf_lowlambda_expanded2688` |             2688 | keeps `LAMBDA_CRIT=0.001`; native PhIRE/TF generator fine-tune |

Safety checks performed before running included `py_compile`, AST/constant extraction, self-block checks, completed-run protection checks, sibling cross-protection, and dry-run precondition checks. The important distinction is that these are **C+E2-low** runs, not the `L_crit`-disabled B+E2 ablation.

### C+E2-low-1344 cheap scalar/domain evaluation

The 1344 C+E2-low run completed training and paired inference on the fixed 168-sample benchmark. The cheap evaluation was healthy enough for TTK: GT alignment was exact, 168 common samples were present, and most scalar/domain metrics improved relative to the CNN baseline.

| Metric                   |        CNN |        GAN | TF C+E2-low-1344 | Direction        |
| ------------------------ | ---------: | ---------: | ---------------: | ---------------- |
| PSNRuv (dB)              |    31.1925 |    29.1380 |      **32.3659** | higher is better |
| Speed MAE (m/s)          |     0.6941 |     0.9026 |       **0.5925** | lower is better  |
| Speed RMSE (m/s)         |     1.1078 |     1.3775 |       **0.9947** | lower is better  |
| WPD MAE                  |   231.6709 |   310.7328 |     **193.1359** | lower is better  |
| WPD W1                   |    45.2713 |    85.6191 |      **14.7417** | lower is better  |
| WPD bias abs             |    35.3439 |    78.7236 |       **4.7898** | lower is better  |
| Gradient MAE             |     0.3491 |     0.3806 |       **0.3173** | lower is better  |
| Gradient W1              |     0.2329 | **0.0564** |           0.1530 | lower is better  |
| Gradient kurtosis abs Δ  |     3.7004 |     4.2010 |       **2.8428** | lower is better  |
| PSD log-L2               |     0.8335 | **0.5139** |           0.8302 | lower is better  |
| PSD slope abs Δ          | **0.9150** |     0.9482 |           1.1317 | lower is better  |
| Exceedance abs Δ s>5     |     0.0042 |     0.0082 |       **0.0030** | lower is better  |
| Exceedance abs Δ s>10    |     0.0066 |     0.0243 |       **0.0030** | lower is better  |
| Exceedance abs Δ s>15    |     0.0062 |     0.0096 |       **0.0013** | lower is better  |
| Exceedance abs Δ p90     |     0.0103 |     0.0123 |       **0.0018** | lower is better  |
| Component-count curve L1 |   115.5278 |   124.6567 |      **88.8919** | lower is better  |

Pairwise improvement counts relative to CNN:

| Metric                   | TF C+E2-low-1344 improved / 168 | TF C+E2-low-1344 worsened / 168 |
| ------------------------ | ------------------------------: | ------------------------------: |
| PSNRuv                   |                             168 |                               0 |
| Speed MAE                |                             168 |                               0 |
| Speed RMSE               |                             168 |                               0 |
| WPD MAE                  |                             168 |                               0 |
| WPD W1                   |                             165 |                               3 |
| WPD bias abs             |                             155 |                              13 |
| Gradient MAE             |                             168 |                               0 |
| Gradient W1              |                             165 |                               3 |
| Gradient kurtosis abs Δ  |                              79 |                              89 |
| PSD log-L2               |                              98 |                              70 |
| PSD slope abs Δ          |                               3 |                             165 |
| Exceedance abs Δ s>5     |                             117 |                              51 |
| Exceedance abs Δ s>10    |                             123 |                              45 |
| Exceedance abs Δ s>15    |                             141 |                              26 |
| Exceedance abs Δ p90     |                             160 |                               8 |
| Component-count curve L1 |                             156 |                              12 |

The main cheap-eval caveat is the same as for B+E2: PSD slope worsens relative to CNN, even though direct fidelity, scalar speed, WPD, gradient, exceedance, and component-count proxy metrics improve strongly.

### C+E2-low-1344 true TTK topology result

The TTK topology pipeline completed cleanly for C+E2-low-1344: 336 VTI files, 336 PD files, and all merge-tree ports completed for both GT and SR fields. The final true topology result was:

| Method           | PD mean | MT mean | PD < CNN | MT < CNN | PD beats GAN | MT beats GAN | MT-GAN cases recovered | Winner distribution after candidate                        |
| ---------------- | ------: | ------: | -------: | -------: | -----------: | -----------: | ---------------------: | ---------------------------------------------------------- |
| CNN baseline     | 27.4063 |  5.8678 |       -- |       -- |           -- |           -- |                     -- | PD: CNN 2 / GAN 166; MT: CNN 148 / GAN 20                  |
| GAN baseline     | 20.8641 |  8.3481 |  166/168 |   20/168 |           -- |           -- |                     -- | --                                                         |
| TF C+E2-low-1344 | 24.3389 |  5.7479 |  164/168 |   90/168 |        6/168 |      161/168 |                  16/20 | PD: C+E2 4 / CNN 2 / GAN 162; MT: C+E2 87 / CNN 77 / GAN 4 |

### Comparison: Candidate C vs B+E2 vs C+E2 at 1344

The 1344 result directly tests whether adding Candidate C's `L_crit` back into the repaired E2 objective helps or hurts the clean B+E2 behavior.

| Method        | Objective summary               |     PD mean |    MT mean | PD < CNN | MT < CNN | MT-GAN cases recovered | Readout                                 |
| ------------- | ------------------------------- | ----------: | ---------: | -------: | -------: | ---------------------: | --------------------------------------- |
| C-1344        | B + `0.001 L_crit`              | **22.8623** |     6.1236 |  168/168 |   49/168 |                   9/20 | strongest PD at this scale              |
| B+E2-low-1344 | B + low-lambda TTK, no `L_crit` |     24.4965 | **5.6514** |  160/168 |   97/168 |              **18/20** | strongest MT at this scale              |
| C+E2-low-1344 | B + `L_crit` + low-lambda TTK   |     24.3389 |     5.7479 |  164/168 |   90/168 |                  16/20 | slightly better PD than B+E2, weaker MT |

Interpretation so far:

1. Adding `L_crit` back into the repaired E2 stack slightly improves PD relative to B+E2-low-1344 (`24.3389` vs `24.4965`).
2. Adding `L_crit` back weakens MT relative to B+E2-low-1344 (`5.7479` vs `5.6514`) and recovers fewer MT-GAN cases (`16/20` vs `18/20`).
3. Candidate C remains the strongest PD-oriented result at 1344, but C+E2 and B+E2 are much stronger for MT.
4. The clean B+E2 result still gives the clearest evidence that the repaired TTK fixed-index terms drive the merge-tree improvement.

### Current status of C+E2-low-2688

At the time of this documentation update, C+E2-low-2688 had been scripted and validated but was still being run. Its result should be added once the cheap evaluation and TTK topology pipeline complete.

A placeholder comparison table is:

| Method        | PD mean | MT mean | PD < CNN | MT < CNN | MT-GAN cases recovered |
| ------------- | ------: | ------: | -------: | -------: | ---------------------: |
| C-2688        | 22.4944 |  6.0803 |  168/168 |   52/168 |                  10/20 |
| B+E2-low-2688 | 23.9876 |  5.6774 |  166/168 |   90/168 |                  18/20 |
| C+E2-low-2688 | pending | pending |  pending |  pending |                pending |
