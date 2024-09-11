from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import pandas as pd
import csv

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

materials_list = [
    'rock',
    'leaf',
    'water',
    'wood',
    'plastic-bag',
    'ceramic',
    'metal',
    'dirt',
    'cloth',
    'plastic',
    'tile',
    'gravel',
    'paper',
    'drywall',
    'glass',
    'grass',
    'carpet'
]
url_csv = pd.read_csv('./extracted_VGGSound.csv')
url_csv = url_csv.values.tolist()

with open('./extracted_VGGSound.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row['label'] for row in reader]

unique_sounds = set(column)
print(unique_sounds, "\n", len(unique_sounds))

max_new_tokens = 200
unique_sounds = list(unique_sounds)
print(unique_sounds)

predicted_materials = []
predicted_sounds = []

for url in url_csv:
  # print(url)
    id, _, label, type, url = url
    torch.cuda.empty_cache()
    image = Image.open(requests.get(url, stream=True).raw)
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is the main material of this video? Please choose from the ones on the {materials_list} and tell me. If there are no materials in {materials_list}, say None. \nAnswer:"},
          {"type": "image"},
        ],
    },
]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    
#   prompt = f"Question: <image>\nWhat is the main material of this video? Please choose from the ones on the {materials_list} and tell me. If there are no materials in {materials_list}, say None.\nAnswer:"
#   outputs1 = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    output1 = model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(processor.decode(output1[0], skip_special_tokens=True))

    output2 = model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(processor.decode(output2[0], skip_special_tokens=True))
    print("Predicted:", output1[0]["generated_text"].split("Answer: ")[1], url)
    predicted_materials.append(output1[0]["generated_text"].split("Answer: ")[1])

    # prompt = f"Question: <image>\nThis is a video thumbnail, what do you think this video will make? You should choose from the ones on the {unique_sounds}.\nAnswer:"
    # outputs2 = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    print(output2)
    print(label, "|", output2[0]["generated_text"].split("Answer: ")[1], end="\n\n")
    predicted_sounds.append(output2[0]["generated_text"].split("Answer: ")[1])

df = pd.read_csv("./extracted_VGGSound.csv")

def create(list):
    required_length = len(df) - len(list)

    extended_list = list + [None] * required_length
    return list

df["predicted_materials"] = create(predicted_materials)
df["predicted_sounds"] = create(predicted_sounds)

df.to_csv("updated.csv", index=False)


# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is shown in this image?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))