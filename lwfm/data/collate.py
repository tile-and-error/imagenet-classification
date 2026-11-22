import torch 

def classification_collate(samples):
    return {
        "inputs": {
            "image": torch.stack([
                s["inputs"]["image"] for s in samples
            ])
        },
        "targets": {
            "class": torch.stack([
                s["targets"]["class"] for s in samples
            ])
        },
        "meta": {
            "sample_id": [
                s["meta"]["sample_id"] for s in samples
            ]
        }
    }