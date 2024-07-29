# v0.8.2
# REF: https://github.com/pytorch/serve/blob/v0.8.2/kubernetes/kserve/kserve_wrapper/__main__.py
# REF: https://github.com/pytorch/serve/blob/v0.8.2/kubernetes/kserve/kserve_wrapper/TorchserveModel.py
# REF: https://github.com/kserve/kserve/blob/release-0.8/python/kserve/kserve/model.py

import json

separator = "="
keys = {}

with open("config.properties") as f:
        for line in f:
            if separator in line:
                # Find the name and value by splitting the string
                name, value = line.split(separator, 1)

                # Assign key value pair to dict
                # strip() removes white space from the ends of strings
                keys[name.strip()] = value.strip()

keys["model_snapshot"] = json.loads(keys["model_snapshot"])
inference_address, management_address, grpc_inference_port, model_store = (
    keys["inference_address"],
    keys["management_address"],
    keys["grpc_inference_port"],
    keys["model_store"],
)

models = keys["model_snapshot"]["models"]
model_names = []

# Get all the model_names
for model, value in models.items():
    model_names.append(model)

print(keys["model_snapshot"])
print(models)
print(inference_address)
print(management_address)
print(grpc_inference_port)
print(model_store)
print(model_names)
