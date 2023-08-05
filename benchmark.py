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
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    # with open(data_file, "r") as data_file:
    #     for line in data_file:
    prompts = ["Test1", "Test2", "Test3"]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")
    print(results)


if __name__ == "__main__":
    fire.Fire(main)