# GRPO Hyperparameter Study Visualizations

Run the hyperparameter study first (`scripts/GRPO_hyperparameter_study.py`), then generate visualizations:

```bash
cd hw4
uv run --with matplotlib --with numpy python visuals/visualize_hyperparameters.py
```

This produces one figure per hyperparameter (5 total):

- **grpo_ppo_epochs.png**
- **grpo_minibatch_size.png**
- **grpo_kl_coef.png**
- **grpo_clip_eps.png**
- **grpo_grad_accum_steps.png**

Each figure has 5 subplots showing training curves over steps (color-coded by hyperparameter value):

1. Eval exact match (format_copy)
2. Rollout mean reward
3. Approximate KL divergence
4. PPO ratio clipped fraction
5. Gradient norm after clipping
