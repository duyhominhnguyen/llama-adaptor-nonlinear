import json

import torch

from llama import ModelArgs, Tokenizer, Transformer
from pathlib import Path

def Llama7B_adapter(args, **kwargs):

    llama_model_path = args.llama_model_path
    model_name = "7B"
    
    if "7B" in args.llama_model_path:
        checkpoints = torch.load(llama_model_path + "/consolidated.00.pth", map_location="cpu")
    else:
        checkpoints = sorted(Path(args.llama_model_path).glob("*.pth"))
        
    print(llama_model_path)

    with open(llama_model_path + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_len=args.adapter_len,
        adapter_layer=args.adapter_layer,
        **params
    )
    
    model_args.typ_act = args.typ_act
    model_args.hid_acti_func = args.hid_acti_func
    model_args.hidden_dim = args.hidden_dim
    model_args.typ_gate = args.typ_gate
    
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if "7B" in args.llama_model_path:
        model_llama_adapter.load_state_dict(checkpoints, strict=False)
        # for parameter_name, parameter in model_llama_adapter.named_parameters():
        #     print("=================", parameter_name)
        #     print(parameter.dtype)
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
            for parameter_name, parameter in model_llama_adapter.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if short_name not in key_to_dim:
                    # print(parameter_name)
                    continue
                
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name].to(torch.float16)
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    # print(checkpoint[
                    #     parameter_name
                    # ].float())
                    parameter.data[size * i : size * (i + 1), :] = checkpoint[
                        parameter_name
                    ].to(torch.float16)
                elif key_to_dim[short_name] == -1:
                    # print(checkpoint[
                    #     parameter_name
                    # ].float().dtype)
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                        parameter_name
                    ].to(torch.float16)
                # parameter.data = parameter.data.float()
                # print(parameter.data.dtype)
            del checkpoint

    for name, param in model_llama_adapter.named_parameters():
        if "adapter" not in name and "hyperAdapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

    for name, param in model_llama_adapter.layers[-1 * args.adapter_layer :].named_parameters():
        if "gate" in name or "adapter" in name or "hyperAdapter" in name:
            param.data = param.data.float()
            param.requires_grad = True

    return model_llama_adapter


# set recommended archs
Llama7B_adapter = Llama7B_adapter
