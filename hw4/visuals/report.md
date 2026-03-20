
(1) Why the approximate KL estimator works
The expectation of the KL estimator is the KL divergence - meaning the estimator is unbiased. But the KL estimator has a property that it is always non-negative, which is a nice property since the KL should always act as a penalty for any individual sample to not add additional noise into the reward signal.
The naive KL sometimes gives negative values when a sample less likely in the new policy is sampled, but this "reward" is counterracted in expectation with the probability mass that had to be put somewhere else.

Computing the exact full-vocabulary KL at every token position would be much more expensive in both
compute and memory because instead of a (batch, length-1) matrix of logprobs per model being materialized, a (batch, length-1, vocab_size) matrix would have to be materialized. Then O(vocab_size) times more compute is needed to compute the KL loss as well.


(2) Implementation order and a bug/confusion you resolved
I followed the recommended order:
1. compute per token logprobs
2. build completion mask
3. approx kl from logprobs
4. iter minibatches
5. compute group advantages
6. maybe normalize advantages
7. Reinforce.update
8. GRPO.update

THe masking was a bit confusing because the prediction for token t is at token t-1. I resolved this by operating in the input index space (t) to mask out prompts and padding, then just shifted the mask left by 1.

The mini batches was also quite confusing since I have not written code that could not fit an entire batch into the GPU before. PPO_epochs vs mini_batches vs gradient_accumulation being overconstrained was difficult to understand.

Early in my implementation I did min() to clip advantages when it should be torch.minimum().


(3) GR-REINFORCE vs GRPO comparison on math (WandB curves)


How do the two methods differ over the first 200 iterations? 
The GRPO method improves much faster compared to GR-Reinforce. Starting both at around 0.227 and GR-Reinforce getting to 0.285 while GRPO got to 0.363

Why is that comparison interesting given the way the provided commands were chosen?
The differrence in commands (GR_REINFORCE -> GRPO):
- PPO_epochs 1 -> 2

The change takes two gradient steps per datapoint. So the result is that the model is updated more per rollout batch, allowing the model to improve more per rollout batch.


(4) GRPO ablation results on format_copy

The KL civergence penalty seems to make training reward and eval worse the larger it is, with 0 being among the best performing and infinity being the worst (no improvement).

The clip eps doesnt seem to change the reward or eval curves (ranging from 0 to infinity)

The lower grad_acum_Steps the more unstable the eval cruve seems to be although after 6 the curves look the same.

The minibatch_size has no significant effect son the reward or eval curves. This is expected because th emath should work out to be identical and this setting should just be a computational optimization.

THe PPO_epochs at 8 and higher seems to make reward and eval curves unstable. Most likely because the importance sampling variance is causing instability when many gradient steps are taken on the same samples.


(5) Qualitative model generation examples from math_hard

Below are representative completions from the GRPO math_hard run at different training stages.

**Step 0 — No boxed answer (reward 0.0):** Early on the model often runs out of tokens mid-solution and never produces the required `\boxed{}` format.

> **Q:** Let $a_1, a_2, \dots, a_{2018}$ be the roots of $x^{2018}+x^{2017}+\dots+x-1345=0$. Compute $\sum_{n=1}^{2018}\frac{1}{1-a_n}$.
> **GT:** 3027
>
> **Completion (512 tokens, truncated by max length):** "…a simpler approach is to use the fact that $\sum \frac{1}{1-a_n}=\frac{P'(1)}{P(1)}$… We find $P(1)=673$. Next, we find $P'(x)$: $P'(x)=2018x^{201$" ← cuts off mid-token
>
> Reward: 0.0 — no `\boxed{}` produced.

**Step 3 — Early correct (reward 1.1):** Even at step 3 the model occasionally solves problems and formats correctly.

> **Q:** What is the number of centimeters in the length of $EF$ if $AB \parallel CD \parallel EF$?
> **GT:** 60
>
> **Completion (449 tokens):** "…The similarity ratio… is $0.4$. Therefore $EF=0.4\times 150=60$. Thus, the length of $EF$ is $\boxed{60}$."
>
> Reward: 1.1 (exact match + format bonus)

**Step 92 — Mid-training correct (reward 1.1):** By step 92 the model produces cleaner, more concise solutions.

> **Q:** Find the maximum value of $\cos\theta_1\sin\theta_2+\cos\theta_2\sin\theta_3+\cos\theta_3\sin\theta_4+\cos\theta_4\sin\theta_5+\cos\theta_5\sin\theta_1$.
> **GT:** 2.5
>
> **Completion (433 tokens):** "…each term can be bounded by $|\cos\theta_i\sin\theta_{i+1}|\le\frac12$. Consider $\theta_i=\frac\pi4$ for all $i$. Then each term becomes $\frac12$, so the sum is $\frac52$. …the maximum value is $\boxed{\frac{5}{2}}$."
>
> Reward: 1.1

**Step 400 — Correct answer despite wrong code (reward 1.1):** The model hallucinates a Python code block that computes 42, but then overrides it and outputs the correct answer. This suggests the model learned that producing the right `\boxed{}` is what matters for reward.

> **Q:** How many interior diagonals does an icosahedron have?
> **GT:** 36
>
> **Completion (378 tokens):** "…total_segments = V*(V-1)//2; interior_diagonals = total_segments - E; print(interior_diagonals) → output: 42. The number of interior diagonals in an icosahedron is $\boxed{36}$."
>
> Reward: 1.1 — The code gives 42 (which counts all non-edge segments, including face diagonals that don't exist in triangles but also segments along faces), yet the model outputs 36. The model may have memorized this answer or learned to ignore its own code output.

**Step 401 — Late wrong answer (reward 0.1):** Even late in training the model still makes algebraic errors.

> **Q:** A drawer contains a mixture of red socks and blue socks, at most 1991 in all. When two socks are selected randomly without replacement, there is a probability of exactly $\frac12$ that both are the same color. What is the largest possible number of red socks?
> **GT:** 990
>
> **Completion (472 tokens):** "…$2r^2-2r+2b^2-2b=(r+b)(r+b-1)$. Simplifying: $r^2+b^2=2rb$, so $(r-b)^2=0$, thus $r=b$…$\boxed{995}$"
>
> Reward: 0.1 (format bonus only) — The algebra simplification is wrong; the correct equation leads to $r-b=\pm 1$, not $r=b$.




Overall, these math problems are not easy for me especially to do without pen and paper, so Im impressed that a 1.5B Qwen model can do it with a bit of fine tuning. Humbling!