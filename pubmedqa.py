# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json
from datetime import datetime

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1,
    max_batch_size: int = 4,
):
    start_time = datetime.now() # record start time

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    total = 0
    correct = 0
    
    ground_truth = json.load(open("/home/zyliang/llama2-benchmark/datasets/PubMedQA/test_ground_truth.json"))
    questions = json.load(open("/home/zyliang/llama2-benchmark/datasets/PubMedQA/test_set.json"))
    for id, question in questions.items():
        context_str = " ".join(question["CONTEXTS"])
        question_str = question["QUESTION"]
        prompts = [f"Question: Cardiomyocyte proliferation gradually declines during embryogenesis resulting in severely limited regenerative capacities in the adult heart. Understanding the developmental processes controlling cardiomyocyte proliferation may thus identify new therapeutic targets to modulate the cell-cycle activity of cardiomyocytes in the adult heart. This study aims to determine the mechanism by which fibroblast growth factor 10 (FGF10) controls foetal cardiomyocyte proliferation and to test the hypothesis that FGF10 promotes the proliferative capacity of adult cardiomyocytes. \
Analysis of Fgf10(-/-) hearts and primary cardiomyocyte cultures reveals that altered ventricular morphology is associated with impaired proliferation of right but not left-ventricular myocytes. Decreased FOXO3 phosphorylation associated with up-regulated p27(kip) (1) levels was observed specifically in the right ventricle of Fgf10(-/-) hearts. In addition, cell-type-specific expression analysis revealed that Fgf10 and its receptor, Fgfr2b, are expressed in cardiomyocytes and not cardiac fibroblasts, consistent with a cell-type autonomous role of FGF10 in regulating regional specific myocyte proliferation in the foetal heart. Furthermore, we demonstrate that in vivo overexpression of Fgf10 in adult mice promotes cardiomyocyte but not cardiac fibroblast cell-cycle re-entry. \
Does fGF10 promote regional foetal cardiomyocyte proliferation and adult cardiomyocyte cell-cycle re-entry?\
\nA. yes, B. no, C. maybe.\nAnswer: A\nQuestion: {context_str} {question_str}\nA. yes, B. no, C. maybe.\nAnswer:"]
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        result = results[-1]["generation"]
        print(prompts)
        print(result, "\n")
        # total += 1
        # if result == data["answer_idx"]:
        #     correct += 1
    # print(f"Total: {total}")
    # print(f"Correct: {correct}")
    # print(f"Accuracy: {correct / total}")
    end_time = datetime.now()
    print(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    fire.Fire(main)
