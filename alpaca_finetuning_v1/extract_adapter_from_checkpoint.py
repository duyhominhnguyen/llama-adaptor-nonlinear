import torch
import argparse

parser = argparse.ArgumentParser("Audio Captioning", add_help=False)
parser.add_argument(
    "--folder",
    default=None,
    type=str,
    help="folder saving weight",
)
args = parser.parse_args()
 
folder = args.folder
print(folder)

if "13B" in folder:
    num_layer = 40
else:
    num_layer = 32
    
model = torch.load(f"{folder}/checkpoint-1.pth", map_location="cpu")
new_model = dict()
weight_list = ["layers." + str(i) + ".attention.gate" for i in range(num_layer)]
old_weight_list = ["layers." + str(i) + ".attention.gate" for i in range(num_layer)]

if "deepshare" in folder:
    weight_list = weight_list + ["adapter_query.weight"]
    for name in model["model"]:
        if "hyperAdapter" in name:
            weight_list.append(name)
elif "hypermodel" not in folder:
    weight_list = weight_list + ["adapter_query.weight"]
else:
    for name in model["model"]:
        if "hyperAdapter" in name:
            weight_list.append(name)

print(weight_list)
if "hypermodel" not in folder:
    print(model["model"]["adapter_query.weight"].shape)

for i in range(len(weight_list)):
    new_model[weight_list[i]] = model["model"][weight_list[i]]
    if "hyperAdapter" in weight_list[i]:
        print(weight_list[i], model["model"][weight_list[i]].shape)

torch.save(new_model, f"{folder}/adapter_adapter_len10_layer30_epoch2.pth")
