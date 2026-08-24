import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

MODEL  = "vankey/DocShield-9B"

print("Loading model...")
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype= torch.bfloat16,
    device_map= "auto"
)

model.eval()

print("DocShield loaded Successfully!")
print(f"Device: {model.device}")