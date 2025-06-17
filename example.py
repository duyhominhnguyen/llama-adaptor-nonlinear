# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer
from datasets import load_dataset, concatenate_datasets
import difflib

PROMPT_DICT = {
    "prompt_no_input_origin": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_multiple_choice": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the multiple-choice question.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_mmlu": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the multiple-choice question about {task}.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    quantizer: bool=False,
    typ_act: str=None,
    hid_acti_func: str=None
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    if "7B" in ckpt_dir:
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
        
    print(adapter_checkpoint.keys())
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, quantizer=quantizer, **params)
    model_args.typ_act = typ_act
    if typ_act == "deepshare":
        print(adapter_checkpoint.keys())
        model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len) 
        print("===================>adapter layer:", model_args.adapter_layer) 

        if "hyperAdapter.linear1.bias" in adapter_checkpoint:
            model_args.hidden_dim = int(adapter_checkpoint["hyperAdapter.linear1.bias"].shape[0])
        else:
            model_args.hidden_dim = 0
        model_args.layer_dim = int(adapter_checkpoint["hyperAdapter.prompt.weight"].shape[1])
        model_args.hid_acti_func = hid_acti_func 
        
    elif typ_act == "hypermodel":
        print(adapter_checkpoint.keys())
        model_args.adapter_layer = int(adapter_checkpoint["hyperAdapter.layer_embed.weight"].shape[0] - 2)
        if "hyperAdapter.decoder.linear1.bias" in adapter_checkpoint:
            model_args.hidden_dim = int(adapter_checkpoint["hyperAdapter.decoder.linear1.bias"].shape[0])
        else:
            model_args.hidden_dim = 0
        model_args.hid_acti_func = hid_acti_func
        print("=============> Number of adapter layers:", model_args.adapter_layer)
    else:
        print(adapter_checkpoint.keys())
        model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len) 
        print("===================>adapter layer:", model_args.adapter_layer)       
    
    
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    if "7B" in ckpt_dir:
        model.load_state_dict(checkpoint, strict=False)
    else:
        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }
        for i, ckpt in enumerate(checkpoints):
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in model.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if short_name not in key_to_dim:
                    continue
                
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name].to(torch.float16)
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    parameter.data[size * i : size * (i + 1), :] = checkpoint[
                        parameter_name
                    ].to(torch.float16)
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                        parameter_name
                    ].to(torch.float16)
        
            del checkpoint
    
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

instructs_ex = [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
        "What are 3 popular chess openings?",
        "How do I send an HTTP request in Javascript? Give me the example code detail.",
        "Write a conversation between the sun and pluto.",
        "Write a shouting match between Julius Caesar and Napoleon.",
        "Send an email requesting that people use language models responsibly.",
        "You are a bash terminal. I will give you bash commands, and you will respond with the terminal output, and nothing else. The first command is ‘pwd‘."
    ]

def ARC():
    ds_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    ds_easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    ans_map = {"1":"A", "2":"B", "3":"C", "4":"D", "5":"E", "6":"F"}
    
    total = concatenate_datasets([ds_easy['test'], ds_challenge['test']])
    print(total)
    list_ques = []
    list_label = []
    list_ans = []
    list_choice = []
    list_task = []
    for ele in ds_easy['test']:
        question = "\nQuestion: " + ele['question'] + f"\n\nOptions:\n\n"
        choice = ""
        for index in range(len(ele['choices']['text'])):
            choice = choice + "- " + ele['choices']['text'][index] + "\n"
            if ele['answerKey'] == ele['choices']['label'][index]:
                list_ans.append(index)

        question = question 
        final_ques = question + choice[0:-1]
        list_ques.append(final_ques)
        list_label.append(ele['answerKey'])
        list_choice.append(ele['choices']['text'])
        list_task.append("easy")
    
    for ele in ds_challenge['test']:
        question = "\nQuestion: " + ele['question'] + f"\n\nOptions:\n\n"
        choice = ""
        for index in range(len(ele['choices']['text'])):
            choice = choice + "- " + ele['choices']['text'][index] + "\n"
            if ele['answerKey'] == ele['choices']['label'][index]:
                list_ans.append(index)
            
        
        question = question   
        final_ques = question + choice[0:-1]
        list_ques.append(final_ques)
        list_label.append(ele['answerKey'])
        list_choice.append(ele['choices']['text'])
        list_task.append("challenge")
        
    return list_ques, list_label, list_ans, list_choice, list_task

def MMLU():
    ds = load_dataset("cais/mmlu", "all")
    list_ques = []
    list_label = []
    list_ans = []
    list_choice = []
    list_task = []
    ans_map = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E"}

    print(ds['test'])
    for ele in ds['test']:
        question = "\nQuestion: " + ele['question'] + f"\n\nOptions:\n\n"
        choice = ""

        for index in range(len(ele['choices'])):
            # choice = choice + "- " + ans_map[index] + ": " + ele['choices'][index] + "\n"
            choice = choice + "- " + ele['choices'][index] + "\n"
            if ele['answer'] == index:
                list_ans.append(index)

        final_ques = question + choice[0:-1]
        list_ques.append(final_ques)
        list_label.append(ans_map[ele['answer']])
        list_choice.append(ele['choices'])
        list_task.append(ele['subject'])
    return list_ques, list_label, list_ans, list_choice, list_task

def TruthQA():
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    list_ques = []
    list_label = []
    list_ans = []
    list_choice = []
    ans_map = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E"}
    instruct = "Interpret each multiple-choice question literally, and as a multiple-choice question about the real world; check carefully to see if the question misses the concept or not and research each option and only pick one most suitable option, without falling prey to any common mythss.\n"
    # instruct = "Interpret each multiple-choice question literally, and as a multiple-choice question about the real world; carefully research each option and only pick one most suitable option, without falling prey to any common mythss; and reply “I have no comment” unless you are completely certain of the answer.\n"
    for ele in ds['validation']:
        question = instruct + "\nQuestion: " + ele['question'] + "\n\nOptions\n\n"
        choice = ""

        for index in range(len(ele["mc1_targets"]['labels'])):
            choice = choice + f"- " + ele["mc1_targets"]['choices'][index] + "\n"
            if ele["mc1_targets"]['labels'][index] == 1:
                list_ans.append(index)
    
        final_ques = question + choice[0:-1]
        list_ques.append(final_ques)
        list_choice.append(ele["mc1_targets"]['choices'])

    return list_ques, [], list_ans, list_choice, []

def question_list(typeQues):
    if typeQues == "arc":
        return ARC()
    elif typeQues == "mmlu":
        return MMLU()
    elif typeQues == "truthqa":
        return TruthQA()
    else:
        return instructs_ex, [], [], [], []

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def check_correct(list_choice, res, label_ans):
    num_c = 0
    list_chosen = []
    for chosen_index, ele_ans in enumerate(list_choice):
        if (ele_ans.lower() in res.split("### Response:")[-1].lower()) or (ele_ans[-1] == "." and ele_ans[0:-1].lower() in res.split("### Response:")[-1].lower()):
            num_c += 1
            list_chosen.append(chosen_index)
    
    if num_c > 1:  # check whether there are more than 1 answers in the predicted output
        return 0

    if find_most_similar_index(list_choice, res.split("### Response:")[-1]) == label_ans:
        if num_c <= 1:
            return 1
        else:
            return 0
    elif num_c == 1 and list_chosen[0] == label_ans:
        return 1
    else:
        return 0

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    typ_act: str=None,
    hid_acti_func: str=None,
    typeQues: str=None,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    quantizer: bool = False,
    random_init: bool = False
):
    print("============Dataset use:", typeQues)
    print("============Type of activation function:", typ_act)
    print("============Use random init:", random_init)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size, quantizer, typ_act, hid_acti_func)
    instructs, label, label_ans, list_choice, list_task = question_list(typeQues)
    
    if typeQues == 'arc':
        prompts = [PROMPT_DICT["prompt_multiple_choice"].format_map({"instruction": x, "input": ""}) for x in instructs]
    elif typeQues == 'mmlu':
        prompts = [PROMPT_DICT["prompt_mmlu"].format_map({"instruction": x, "task": t}) for x, t in zip(instructs, list_task)]
    elif typeQues == 'truthqa':
        prompts = [PROMPT_DICT["prompt_multiple_choice"].format_map({"instruction": x, "input": ""}) for x in instructs]        
    else:
        prompts = [PROMPT_DICT["prompt_no_input_origin"].format_map({"instruction": x, "input": ""}) for x in instructs]

        
    total = 0
    correct = 0
    ans_pred_list = []
    mmlu_res = {}
    mmlu_acc = {}
    
    arc_res = {}
    arc_acc = {}

    for i in range(0, len(prompts), 32):
        results = generator.generate(prompts[i:min(i+32, len(prompts))], max_gen_len=512, temperature=temperature, top_p=top_p, typ_act=typ_act, random_init=random_init)
        for j, res in enumerate(results):
            print(res)
            if len(label) != 0 or typeQues in "truthqa":

                print("Predict:", res.split("### Response:")[-1].lower())
                print("List choices:", list_choice[i+j])

                if typeQues == "mmlu":
                    if list_task[i+j] not in mmlu_res:
                        mmlu_res[list_task[i+j]] = []
                        
                    print("Answer in text:", label_ans[i+j])
                    print("Predict sentence:", res.split("### Response:")[-1])
                    print(find_most_similar_index(list_choice[i+j], res.split("### Response:")[-1]))
                    correct += check_correct(list_choice[i+j], res, label_ans[i+j])
                    mmlu_res[list_task[i+j]].append(check_correct(list_choice[i+j], res, label_ans[i+j]))  
                    total += 1
                    for task_mmlu in mmlu_res:
                        mmlu_acc[task_mmlu] = np.sum(mmlu_res[task_mmlu])/len(mmlu_res[task_mmlu])*100
                        print(f"========> Accuracy for task {task_mmlu}:", mmlu_acc[task_mmlu])
                    print("========> NoCorrect:", correct)
                    print("========> Total:", total)
                    print("========> Accuracy 1:", correct/total*100)
                    print("========> Accuracy 2:", np.mean(list(mmlu_acc.values())))
                    print(adapter_path)
                
                elif typeQues == "truthqa":
                    print("Answer in text:", label_ans[i+j])
                    correct += check_correct(list_choice[i+j], res, label_ans[i+j])
                    total += 1
                    print("========> NoCorrect:", correct)
                    print("========> Total:", total)
                    print("========> Accuracy:", correct/total*100)
                    print(adapter_path)
                        
                elif typeQues == "arc":
                    if list_task[i+j] not in arc_res:
                        arc_res[list_task[i+j]] = []
                        
                    print("Answer in text:", label_ans[i+j])
                    print("Predict sentence:", res.split("### Response:")[-1])
                    print(find_most_similar_index(list_choice[i+j], res.split("### Response:")[-1]))
                    correct += check_correct(list_choice[i+j], res, label_ans[i+j])
                    arc_res[list_task[i+j]].append(check_correct(list_choice[i+j], res, label_ans[i+j]))
                    total += 1
                    
                    for task_arc in arc_res:
                        arc_acc[task_arc] = np.sum(arc_res[task_arc])/len(arc_res[task_arc])*100
                        print(f"========> Accuracy for task {task_arc}:", arc_acc[task_arc])
                        
                    print("========> NoCorrect:", correct)
                    print("========> Total:", total)
                    print("========> Accuracy:", correct/total*100)
                    print(adapter_path)
            print("\n==================================\n")
    
    if typeQues == "mmlu":
        acc_mmlu = []
        for task_mmlu in mmlu_acc:
            acc_mmlu.append(mmlu_acc[task_mmlu])    
        print("Total accuracy for MMLU:", np.mean(acc_mmlu))


if __name__ == "__main__":
    fire.Fire(main)
