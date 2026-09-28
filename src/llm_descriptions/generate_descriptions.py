"""
generate_descriptions.py -- LLM description generation using the paper's
exact Section 3.2 template and prompt. Three backends:
  --backend api      : real OpenAI-compatible API call (needs OPENAI_API_KEY)
  --backend local     : local open-source instruct model on this GPU (no key needed)
  --backend offline    : deterministic stub text, for pipeline testing only

Supports three datasets: cifar10, cifar100, robotcar.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robotcar"))
from cifar_hierarchy import get_hierarchy

PROMPT_TEMPLATE = (
    "Please use the Template y to generate a description for class {cls}, "
    "ensuring the description does not exceed 300 words.\n\n"
    'Template y: "A description of the class {cls}, {{based on its appearance '
    'characteristics}} and {{behavioral traits or functional features}}."'
    "{disambiguation}"
    "\n\nOutput ONLY the description itself -- do not repeat this prompt, "
    "the template text, or add any preamble/explanation."
)

PEOPLE_CLASSES_DISAMBIGUATION = {
    "baby": " (Note: this refers to a human infant/baby person, not an animal.)",
    "boy": " (Note: this refers to a human boy/male child, not an animal.)",
    "girl": " (Note: this refers to a human girl/female child, not an animal.)",
    "man": " (Note: this refers to an adult human male person, not an animal.)",
    "woman": " (Note: this refers to an adult human female person, not an animal.)",
    "Cyc": " (Note: describe the CYCLIST as a road user -- their presence, vulnerability, and behavior in traffic -- not the bicycle as a mechanical object. Focus on the person riding, not the vehicle.)",
}

CLASS_NAME_EXPANSIONS = {
    "Ped": "pedestrian",
    "Cyc": "cyclist",
    "vulnerable_road_user": "vulnerable road user (pedestrians and cyclists)",
}


def truncate_to_word_limit(text: str, max_words: int = 300) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def clean_response(text: str, cls: str) -> str:
    text = text.strip()
    for marker in ['Template y:', 'Template y :', f'"A description of the class {cls}']:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    text = text.strip('"').strip()
    return text


def build_prompt(cls: str) -> str:
    prompt_cls = CLASS_NAME_EXPANSIONS.get(cls, cls)
    disambiguation = PEOPLE_CLASSES_DISAMBIGUATION.get(cls, "")
    return PROMPT_TEMPLATE.format(cls=prompt_cls, disambiguation=disambiguation)


def generate_one_description_api(client, cls: str, model: str, max_retries: int = 3) -> str:
    prompt = build_prompt(cls)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return clean_response(resp.choices[0].message.content.strip(), cls)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to generate description for {cls!r}: {last_err}")


def generate_one_description_local(model, tokenizer, cls: str, device: str, max_new_tokens: int = 450) -> str:
    import torch
    prompt = build_prompt(cls)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    result = clean_response(result, cls)
    return truncate_to_word_limit(result, max_words=300)


def generate_offline_stub(cls: str) -> str:
    return (
        f"A description of the class {cls}, based on its appearance "
        f"characteristics and behavioral traits or functional features. "
        f"[OFFLINE STUB -- not a real LLM description]"
    )


def get_dataset_classes(dataset: str, data_root: str):
    if dataset == "robotcar":
        from robotcar_hierarchy import get_robotcar_hierarchy
        fine_names, coarse_names, _ = get_robotcar_hierarchy()
    else:
        fine_names, coarse_names, _ = get_hierarchy(dataset, data_root)
    return fine_names, coarse_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "robotcar"], default="cifar100")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--backend", choices=["api", "local", "offline"], default="local")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--local_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--regenerate", nargs="*", default=[])
    args = parser.parse_args()

    fine_names, coarse_names = get_dataset_classes(args.dataset, args.data_root)
    all_classes = list(fine_names) + list(coarse_names)
    print(f"{args.dataset}: {len(fine_names)} fine + {len(coarse_names)} coarse = {len(all_classes)} classes "
          f"(backend={args.backend})")

    client = None
    model = tokenizer = None
    device = "cuda"
    if args.backend == "api":
        import openai
        client = openai.OpenAI()
    elif args.backend == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {args.local_model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.local_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.local_model, torch_dtype=torch.bfloat16, device_map=device
        )
        model.eval()
        print("Model loaded.")

    descriptions = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            descriptions = json.load(f)
        print(f"Resuming: {len(descriptions)} already present")

    for cls in args.regenerate:
        descriptions.pop(cls, None)

    for i, cls in enumerate(all_classes):
        if cls in descriptions:
            continue
        if args.backend == "offline":
            desc = generate_offline_stub(cls)
        elif args.backend == "local":
            desc = generate_one_description_local(model, tokenizer, cls, device)
        else:
            desc = generate_one_description_api(client, cls, args.model)

        descriptions[cls] = desc
        print(f"[{i+1}/{len(all_classes)}] {cls}: {desc[:100]}...")
        if args.backend == "api":
            time.sleep(args.sleep)
        with open(args.out, "w") as f:
            json.dump(descriptions, f, indent=2)

    print(f"\nDone. {len(descriptions)} descriptions saved to {args.out}")


if __name__ == "__main__":
    main()
