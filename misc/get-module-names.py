import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

model_name = "./DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.pad_token_id = tokenizer.pad_token_id
config.num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=True
)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name)