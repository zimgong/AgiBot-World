# Decoder Design

* **DDPM**, **Flow Matching (FM/ODE)**, **MeanFlow (MF/JVP)**, and **SDE sampling** (for FM **and** MF),
* how/when to use each (and how to **mix** them safely),
* all **key formulas** and **references**, with minimal math but enough for implementation,
* **config usage** (based on your Base config),
* **code patches** that:

  1. add **MF‑SDE** sampling,
  2. fix a couple of **bugs** and **inconsistencies** in the provided decoder file, and
  3. make config **field names** in the model compatible with your Base config.

---

# GO‑1 Action Decoder: Diffusion & Flow Variants

This decoder turns vision‑language context + robot state into a distribution over **action trajectories** $z \in \mathbb{R}^{H \times D}$ (H: chunk length, D: action_dim). We support four decoders:

1. **DDPM** (discrete diffusion) — classic denoising diffusion with a noise scheduler.
2. **Flow Matching (FM/ODE)** — trains a **deterministic velocity field** ($v_\theta$) on a straight interpolation path and **integrates an ODE** to sample. ([arXiv][1])
3. **MeanFlow (MF/JVP)** — trains an **average velocity** ($u_\theta$) with the **MeanFlow identity** using JVP; integrates with an **average‑velocity step**. ([arXiv][2])
4. **SDE samplers** (FM‑SDE and MF‑SDE) — **stochastic** Euler–Maruyama samplers that **preserve the ODE marginals** by adding an **Ito correction** to the drift, enabling exploration for RL while keeping the behavior‑cloned density unchanged. ([arXiv][3])

---

## Notation (shared by all)

* We use the **linear path** between a standard Gaussian prior ($e \sim \mathcal{N}(0,I)$) and demonstration action ($a$):
  $$z_t = (1-t) \cdot e + t \cdot a, \quad v(z_t,t) \equiv \frac{dz_t}{dt} = a - e$$
* At **training** time, we sample ($t \sim \text{Unif}[0,1]$), construct ($z_t$) and the target velocity ($v=a-e$).
* Conditioning (vision+language+state) is fed through the Action Expert with tokens ($[t, r, f, \text{state}, \text{action tokens}]$) (see code).

---

## 1) DDPM decoder

### Training objective

Predict either the **clean sample** ($a$) or **noise** ($\epsilon$) depending on `prediction_type`:

```python
# target = action_gts if pred_type=="sample" else noise
L_DDPM = E[||model_out - target||²]
```

This is standard diffusion training with `DDPMScheduler.add_noise` and a regression loss. (Implements the DDPM/score‑based umbrella; reverse‑SDE ↔ probability‑flow ODE correspondence is classic.) ([arXiv][3])

### Inference

We use `DPMSolverMultistepScheduler` to iteratively denoise:
$$x_{t-1} = \text{step}_{\text{DPM}}(x_t; \text{model\_out}, t)$$

**When to use**: strongest choice when you want **matched training/inference** to the diffusion literature and don't need continuous‑time control of exploration.

---

## 2) Flow Matching (FM/ODE) decoder

**Flow Matching** (a.k.a. rectified flow / stochastic interpolants) **regresses the instantaneous velocity** on a fixed path; here the straight line $((1-t)e + ta)$. Train with:
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,e,a}\big[|v_\theta(z_t,t) - (a-e)|^2\big]$$
(conditional flow matching reduces variance; same objective in our linear path). ([arXiv][1])

**ODE sampling**:
$$z_{t_{i+1}} = z_{t_i} + v_\theta(z_{t_i},t_i) \cdot \Delta t$$

**When to use**: best for **fast**, deterministic sampling. Works great for **BC**; can be switched to **SDE** at inference time **without retraining** while **preserving marginals** (next section). ([arXiv][3])

---

## 3) MeanFlow (MF/JVP) decoder

**MeanFlow** learns the **average velocity** over an interval $([t,r])$:
$$u(z_t,r,t) \approx \frac{1}{r-t}\int_{t}^{r} v(z_s,s) \, ds$$

The **MeanFlow identity** links ($u$) to the instantaneous ($v$):
$$u(z_t,r,t) = v(z_t,t) - (t-r) \cdot \frac{d}{dt}u(z_t,r,t)$$
which implies:
$$v(z_t,t) = u(z_t,r,t) + (t-r) \cdot \frac{d}{dt}u(z_t,r,t)$$
We compute $\frac{d}{dt}u$ efficiently by **JVP**. ([arXiv][2])

**Training**: regress the LHS to the RHS target from the identity (see code's `calc_action_meanflow_loss`).

**ODE sampling** (MeanFlow step):
on a grid $(0=t_0<\dots<t_K=1)$, set ($r=t_{i+1}$) and update
$$z_{t_{i+1}} = z_{t_i} + u_\theta(z_{t_i}, r=t_{i+1}, t=t_i) \cdot \Delta t$$

**When to use**: powerful when you want low‑step sampling and a **one‑step** or few‑step generator; the JVP‑based identity can improve stability/accuracy.

---

## 4) SDE sampling (for exploration) — FM‑SDE & MF‑SDE

We want **exploration** during downstream RL **without changing the BC‑learned marginals**. For any **isotropic** ($\sigma(t)$), the **reverse SDE**
$$dx = \tilde{v}_\theta(x,t) \, dt + \sigma(t) \, dW_t$$
shares the **same marginals** as the **probability‑flow ODE** if
$$\tilde{v}_\theta(x,t) = v_\theta(x,t) + \frac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x)$$
(ODE↔SDE **marginal equivalence**). ([arXiv][3])

With the **linear path** ($z_t=(1-t)e + ta$), the **score–velocity identity** is:
$$\nabla_x \log p_t(x) = \frac{t \cdot v_\theta(x,t) - x}{1-t}$$
(derivable for affine paths; see flow‑matching guides). Substituting yields the **marginal‑preserving drift**:
$$\tilde{v}_\theta(x,t) = v_\theta(x,t) + \frac{1}{2} \sigma(t)^2 \frac{t \cdot v_\theta(x,t) - x}{1-t}$$
This is the **correction** we use in code. (Your earlier draft had opposite signs; the corrected version above preserves marginals.) ([Department of Computer Science][4])

**Discretization (Euler–Maruyama, per step (i))**:
$$\begin{aligned}
\mu_i &= x_i + \tilde{v}_\theta(x_i,t_i) \cdot \Delta t_i \\
x_{i+1} &= \mu_i + \sigma(t_i)\sqrt{\Delta t_i} \cdot \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0,I)
\end{aligned}$$

**Per‑step Gaussian log‑prob** (for SAC):
$$\log \pi_i(x_{i+1}|x_i) = -\frac{1}{2}\big[D \log(2\pi\sigma(t_i)^2\Delta t_i) + |x_{i+1}-\mu_i|^2/(\sigma(t_i)^2\Delta t_i)\big]$$
To make the **entropy term** only update the **learnable noise** ($\sigma$), compute it **from ($\epsilon_i$)**:
$$\log \pi_i = -\frac{1}{2}\big[D \log(2\pi) + |\epsilon_i|^2\big] - D \log\big(\sigma(t_i)\sqrt{\Delta t_i}\big)$$
which **removes mean‑dependence** and routes gradients to **$\sigma$ only** (as in Gaussian SAC). (The Q‑term still updates ($v_\theta$) through the reparameterized action.)

For ($\sigma(t)$) we use your **churn‑style** schedule
$$\sigma(t) = s_{\text{churn}}\sqrt{1-\frac{t}{t+\varepsilon}}$$
(or "fixed"), common in EDM samplers. ([Hugging Face][5])

### FM‑SDE (implemented)

We directly use ($v_\theta=$)**`_v_theta`** (FM head) in the formula above. See `flowmatch_sde_sample`.

### **MF‑SDE** (new)

We need ($v_\theta$) but the MF head predicts ($u_\theta$). Two minimal, architecture‑preserving options:

1. **Boundary trick (no JVP):** use the **limit ($r \to t$)**, where ($u(z_t,r,t) \to v(z_t,t)$). Our MF head can be called with **`r=t`**, so we set
   $$\hat{v}_\theta(x,t) := u_\theta(x, r=t, t)$$
   and plug ($\hat{v}_\theta$) into the same drift correction above. (This is what the code below does.) ([GitHub][6])

2. **Identity‑based (with JVP):** use ($\hat{v} = u + (t-r)\frac{d}{dt}u$) with a small ($r>t$) and compute ($\frac{d}{dt}u$) by JVP; then plug ($\hat{v}$) into the drift. (Heavier; optional.)

---

## Mixing: which training ↔ inference combos are safe?

| Train        | Inference      | Safe? | Notes                                                                                                                                                   |
| ------------ | -------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DDPM**     | DDPM           |     ✅ | Native.                                                                                                                                                 |
| **FM (ODE)** | **FM‑ODE**     |     ✅ | Deterministic; fastest.                                                                                                                                 |
| **FM (ODE)** | **FM‑SDE**     |     ✅ | **Same marginals** (by construction), adds exploration for RL. ([arXiv][3])                                                                             |
| **MF (JVP)** | **MF‑ODE**     |     ✅ | Native MeanFlow sampler. ([arXiv][2])                                                                                                                   |
| **MF (JVP)** | **MF‑SDE**     |     ✅ | Using **boundary (r=t)** to get ($\hat{v}$). In practice works well; exactness depends on how well the network respects the ($r \to t$) limit. ([GitHub][6]) |
| **MF (JVP)** | **FM‑SDE/ODE** |    ⚠️ | You can reuse **`_v_theta = _u_theta(·, r=t)`** as an FM proxy. This is a practical approximation.                                                      |

---

## Config: how to select decoders & SDE

Using your **Base config** fields:

```python
# DDPM (default)
decoder_type = "DDPM"

# Flow Matching (deterministic ODE)
decoder_type = "flow_matching"
flow_sde_use = False

# Flow Matching with SDE (for RL/exploration)
decoder_type = "flow_matching"
flow_sde_use = True
flow_sde_mode = "churn"     # or "fixed"
flow_sde_s_churn = 0.7
flow_sde_eps = 1e-4
flow_sde_learn_sigma = True # enable learnable noise; train via RL
flow_sde_return_logprob = True
flow_sde_logprob_sigma_only = True  # entropy grads → σ only

# MeanFlow (deterministic ODE)
decoder_type = "mean_flow"

# MeanFlow with SDE (boundary trick, new sampler)
decoder_type = "mean_flow"
flow_sde_use = True
flow_sde_return_logprob = True
```

At **runtime** you can override sampler per call:
`model(..., sampler="sde", return_logprob=True)`.

---

## API surfaces (code snippets)

### FM‑ODE sampling

```python
z = torch.randn(B, H, D)
for i in range(K):
    t = t_grid[i]
    v = _v_theta(z, t, state, ctrl, attn, kv)
    z = z + v * dt
```

### FM‑SDE sampling (implemented)

```python
v = _v_theta(z, t, ...)
vtilde = v + 0.5 * sigma(t)**2 * (t * v - z) / (1 - t)
mu = z + vtilde * dt
z  = mu + sigma(t) * sqrt(dt) * eps
# logprob (epsilon form) sums over dims; grads -> sigma only
```

### MF‑ODE sampling

```python
u = _u_theta(z, t, r=t_next, ...)
z = z + u * dt
```

### MF‑SDE sampling (new; boundary trick)

```python
v_hat = _u_theta(z, t, r=t, ...)          # use r=t to emulate v
vtilde = v_hat + 0.5 * sigma(t)**2 * (t * v_hat - z) / (1 - t)
mu = z + vtilde * dt
z  = mu + sigma(t) * sqrt(dt) * eps
# same epsilon-based logprob if needed
```

---

## References (short list)

* **Score‑SDE & probability‑flow ODE equivalence**: Song et al., *ICLR 2021*; includes exact ODE↔SDE correspondence and predictor–corrector. ([arXiv][3])
* **Flow Matching / Rectified Flow**: Lipman et al., *Flow Matching for Generative Modeling*; Liu et al., *Rectified Flow*. ([arXiv][1])
* **MeanFlow identity** (average vs instantaneous velocity): Geng et al., *Mean Flows for One‑step Generative Modeling*; boundary property ($r \to t$). ([arXiv][2])
* **Churn‑style stochasticity** (EDM samplers): Karras et al.; diffusers `EDMEulerScheduler` doc. ([Hugging Face][5])
* **Unified FM resources**: "Let us Flow Together" book & site. ([Department of Computer Science][4])

---

## Practical tips

* **Training**: keep **FM‑ODE** or **MF** training unchanged (BC). Turn on **SDE only at inference** (for RL/data collection). This guarantees marginal‑equivalence for FM and a solid approximation for MF (boundary trick).
* **Entropy gradients**: set `flow_sde_logprob_sigma_only=True` to route **entropy** to **learnable σ** only (SAC‑style); the **Q‑term** still updates ($v_\theta$) through the sampled rollouts.
* **Numerics**: avoid squashing (tanh) inside the **per‑step** Gaussian; if you need bounded actions, squash **once at the final step** and adjust downstream (SAC usually works with pre‑squash entropy).
* **FlashAttention + JVP**: If you see instabilities with MF+JVP, set the AE attn implementation to **`eager`** in training; FM and SDE are unaffected by FA.

---


[1]: https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "[2210.02747] Flow Matching for Generative Modeling"
[2]: https://arxiv.org/abs/2505.13447?utm_source=chatgpt.com "[2505.13447] Mean Flows for One-step Generative Modeling"
[3]: https://arxiv.org/abs/2011.13456?utm_source=chatgpt.com "Score-Based Generative Modeling through Stochastic Differential Equations"
[4]: https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf?utm_source=chatgpt.com "Let us Flow Together"
[5]: https://huggingface.co/docs/diffusers/api/schedulers/edm_euler?utm_source=chatgpt.com "EDMEulerScheduler - Hugging Face"
[6]: https://github.com/magicknight/MeanFlow?utm_source=chatgpt.com "GitHub - magicknight/MeanFlow"