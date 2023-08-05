# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    # data_file: str, # path to the file containing testing data
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    data_path = "/home/zyliang/llama2-benchmark/datasets/test/dev.jsonl"
    with open(data_path, "r") as data_file:
        for count, line in enumerate(data_file):
            data = json.loads(line)
            question_str = f"question: {data['question']}"
            options = data["options"]
            options_str = f"options: A: {options['A']}, B: {options['B']}, C: {options['C']}, D: {options['D']}, E: {options['E']}."
            prompts = [f"{question_str} {options_str}", "The correct answer to the multiple choice question above is "]
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"[Question {count+1}]")
            print(prompts)
            print("[Answer]")
            for index, result in enumerate(results):
                print(f"[Part {index+1}]")
                print(result["generation"])
            print("\n")


if __name__ == "__main__":
    fire.Fire(main)
