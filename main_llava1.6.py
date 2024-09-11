from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import pandas as pd
import csv
import re
import pandas as pd
import csv
import math
import numpy as np

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:1")

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

max_new_tokens = 100
unique_sounds = list(unique_sounds)

predicted_materials = []
predicted_sounds = []
gpt_score = []

for url in url_csv[:5]:
  # print(url)
    id, _, label, type, url = url
    torch.cuda.empty_cache()
    image = Image.open(requests.get(url, stream=True).raw)
    conversation1 = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is the main material of this video? Please choose from the ones on the {materials_list} and just say that material in one or two words. Don't speak descriptively, speak in short answers."}, # If there are no materials in {materials_list}, say None
          {"type": "image"},
        ],
    },
]
    
    conversation2 = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Question: This is a video thumbnail, what do you think this video will make? You should choose from the ones on the {unique_sounds}. Don't speak descriptively, speak in short answers."},
          {"type": "image"},
        ],
    },
]

    prompt1 = processor.apply_chat_template(conversation1, add_generation_prompt=True)
    inputs1 = processor(images=image, text=prompt1, return_tensors="pt").to("cuda:1")
    
    output1 = model.generate(**inputs1, max_new_tokens=max_new_tokens)
    decoded_output1 = processor.decode(output1[0], skip_special_tokens=True)
    print(decoded_output1)

    prompt2 = processor.apply_chat_template(conversation2, add_generation_prompt=True)
    inputs2 = processor(images=image, text=prompt2, return_tensors="pt").to("cuda:1")

    output2 = model.generate(**inputs2, max_new_tokens=max_new_tokens)
    decoded_output2 = processor.decode(output2[0], skip_special_tokens=True)
    print(decoded_output2)
    
    assistant_response1 = decoded_output1.split("ASSISTANT:")[-1].strip()
    assistant_response2 = decoded_output2.split("ASSISTANT:")[-1].strip()

    print("predicted_materials.append: ", assistant_response1)
    predicted_materials.append(assistant_response1)

    # prompt = f"Question: <image>\nThis is a
    #  video thumbnail, what do you think this video will make? You should choose from the ones on the {unique_sounds}.\nAnswer:"

    print(label, "|", assistant_response2, end="\n\n")
    predicted_sounds.append(assistant_response2)

df = pd.read_csv("./extracted_VGGSound.csv")

def create(list):
    required_length = len(df) - len(list)

    extended_list = list + [None] * required_length
    return extended_list

df["predicted_materials"] = create(predicted_materials)
df["predicted_sounds"] = create(predicted_sounds)

df.to_csv("updated.csv", index=False)


def contains_any_word(sentence1, sentence2):
    def clean_and_split(sentence):
        if isinstance(sentence, float) and math.isnan(sentence):
            return set()  # NaN일 경우 빈 세트를 반환
        cleaned_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        return set(cleaned_sentence.split())
    
    set1 = clean_and_split(sentence1)
    set2 = clean_and_split(sentence2)

    common_words = set1.intersection(set2)

    return bool(common_words)

updated = pd.read_csv('./updated.csv')
updated = updated.values.tolist()

success_count = 0

for row in updated:
    # print(url)
    id, time, label, type, url, predicted_material, predicted_sound = row
    if contains_any_word(label, predicted_sound):
        success_count += 1
  # if str(label) in str(predicted_sound).lower()
print(success_count)

acc = success_count / len(updated)

print(acc)

# for row in updated:
#   # print(url)
#     id, time, label, type, url, predicted_material, predicted_sound = row
#     conversation3 = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": f"Assume you are a child who has assignment that you have to write the answer about the similarity of the words. Assignment words '{label}' and '{predicted_sound}'. The answer should be from 0 up to 1. your answer is?"},
#         ],
#     },
# ]
#     prompt3 = processor.apply_chat_template(conversation3, add_generation_prompt=True)
#     inputs3 = processor(text=prompt3, return_tensors="pt").to("cuda:1")
    
#     output3 = model.generate(**inputs3, max_new_tokens=max_new_tokens)
#     decoded_output3 = processor.decode(output3[0], skip_special_tokens=True)
#     assistant_response3 = decoded_output3.split("ASSISTANT:")[-1].strip()
#     import pdb; pdb.set_trace();
#     gpt_score.append(assistant_response2)

# print("GPT score: ", np.mean(gpt_score))