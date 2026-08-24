import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from document_preprocessing import prepare_document_for_model
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

SYSTEM_PROMPT = (
    "你是一个图像鉴伪专家，擅长结合视觉，文字结合伪造特征分析手段鉴别输入图像的真假。"
    "分析过程中，你会逐步分析，抽丝剥茧，找到图像伪造的蛛丝马迹，最终给出专业的鉴别结果及分析。"
)
USER_PROMPT = "请分析这张文档图片是否存在伪造或篡改风险，并输出一份专业、精炼、准确的防伪分析报告。"

#image_paths = prepare_document_for_model("data/converted/Forgery-test-page1.png")
image_path = "data/converted/Forgery-test_page1.png"

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": USER_PROMPT},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

gen = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
print(processor.batch_decode(gen, skip_special_tokens=False)[0])

# print("DocShield loaded Successfully!")
# print(f"Device: {model.device}")