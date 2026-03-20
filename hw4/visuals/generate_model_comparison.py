"""Generate comparison rollouts from Base, GR-REINFORCE, and GRPO models.

Runs on Modal with access to the training volume where checkpoints live.

Usage:
    uv run modal run visuals/generate_model_comparison.py

Then download the JSON result:
    uv run modal volume get hw4-llm-rl-volume \
        /comparisons/model_comparison.json visuals/model_comparison.json

Then visualize locally:
    uv run python visuals/visualize_model_comparison.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "hw4-llm-rl-comparison"
VOLUME_PATH = "/vol"
PROJECT_DIR = "/root/project"
NETRC_PATH = Path("~/.netrc").expanduser()

volume = modal.Volume.from_name("hw4-llm-rl-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["remote"])
)
image = image.run_commands(
    "uv pip install --system --index-url https://download.pytorch.org/whl/cu124 'torch>=2.5,<2.7'"
)
if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)
image = image.add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())

app = modal.App(APP_NAME)

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
GRPO_RUN_DIR = "/vol/runs/modal_math_hard_grpo"
REINFORCE_RUN_DIR = "/vol/runs/modal_math_hard_reinforce"
NUM_QUESTIONS = 6
NUM_ROLLOUTS = 5
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 512
OUTPUT_PATH = "/vol/comparisons/model_comparison.json"

QUESTION_INDICES = [0, 42, 100, 200, 300, 400]


def find_latest_adapter(run_dir: str) -> str | None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    step_dirs = sorted(ckpt_dir.glob("step_*"), key=lambda p: p.name)
    if not step_dirs:
        return None
    adapter_dir = step_dirs[-1] / "adapter"
    if adapter_dir.exists():
        return str(adapter_dir)
    return None


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
    env={"PYTHONPATH": PROJECT_DIR, "PYTHONUNBUFFERED": "1",
         "HF_HOME": f"{VOLUME_PATH}/hf", "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets"},
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=65536,
)
def generate_comparison() -> None:
    import torch
    from hw4.models.load import load_inference_model_and_tokenizer, tokenize_chat_prompts
    from hw4.tasks.math_hard import MathHardTask
    from hw4.tasks.base import TaskExample
    from hw4.utils.answer_parsing import extract_last_boxed_content, extract_number_from_boxed_answer

    device = torch.device("cuda")
    dtype = torch.bfloat16

    task = MathHardTask(seed=0, train_levels=(5,), eval_subset_size=512)
    pool = task._get_eval_pool("test_subset")

    questions = []
    for idx in QUESTION_INDICES:
        if idx < len(pool):
            row = pool[idx]
            questions.append({
                "idx": idx,
                "problem": row["problem"],
                "gt": row["gt"],
                "subject": row.get("subject", "unknown"),
                "solution": row.get("solution", ""),
            })
    print(f"Selected {len(questions)} questions")

    grpo_adapter = find_latest_adapter(GRPO_RUN_DIR)
    reinforce_adapter = find_latest_adapter(REINFORCE_RUN_DIR)
    print(f"GRPO adapter: {grpo_adapter}")
    print(f"REINFORCE adapter: {reinforce_adapter}")

    models_config = [
        ("Base", None),
        ("GR-REINFORCE", reinforce_adapter),
        ("GRPO", grpo_adapter),
    ]

    for q in questions:
        q["rollouts"] = {}

    for model_name, adapter_path in models_config:
        print(f"\n--- Loading {model_name} ---")
        loaded = load_inference_model_and_tokenizer(
            MODEL_NAME, device=device, dtype=dtype, adapter_path=adapter_path,
        )
        model, tokenizer = loaded.model, loaded.tokenizer

        for qi, q in enumerate(questions):
            print(f"  Q{qi+1}/{len(questions)}: generating {NUM_ROLLOUTS} rollouts...")
            messages = task._build_messages(q["problem"])
            messages_batch = [messages] * NUM_ROLLOUTS

            input_ids, attention_mask = tokenize_chat_prompts(
                tokenizer, messages_batch, add_generation_prompt=True,
                max_prompt_tokens=512, device=device,
            )

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_len = int(input_ids.shape[1])
            completion_ids = out[:, prompt_len:]
            pad_id = int(tokenizer.pad_token_id)

            rollouts = []
            for row_ids in completion_ids:
                if (row_ids == pad_id).any():
                    n = int((row_ids != pad_id).sum().item())
                    row_ids = row_ids[:n]
                completion = tokenizer.decode(row_ids, skip_special_tokens=True)

                example = TaskExample(
                    meta={"gt": q["gt"], "level": 5, "question": q["problem"],
                          "subject": q["subject"], "gt_source": "boxed_or_direct_number"},
                    messages=messages,
                    task_name="math_hard",
                )
                reward, info = task.reward(example, completion)

                boxed_content = extract_last_boxed_content(completion)
                pred_boxed = extract_number_from_boxed_answer(completion)

                rollouts.append({
                    "completion": completion,
                    "reward": reward,
                    "boxed_content": boxed_content,
                    "predicted_boxed": pred_boxed,
                    "exact_boxed": bool(info.get("math_hard/is_exact_match_using_number_parsed_from_boxed_answer", 0)),
                    "exact_relaxed": bool(info.get("math_hard/is_exact_match_using_relaxed_last_number_parser", 0)),
                    "has_boxed_keyword": bool(info.get("math_hard/completion_contains_literal_backslash_boxed_open_brace", 0)),
                })

            q["rollouts"][model_name] = rollouts

        del model, loaded
        torch.cuda.empty_cache()

    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"questions": questions}, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_PATH}")
    volume.commit()


@app.local_entrypoint()
def main():
    generate_comparison.remote()
