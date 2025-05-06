# Meme Understanding with CLIP and BLIP-2

import os
import torch
from PIL import Image
import json
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load BLIP-2
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

with open("real_meme_dataset_clip_blip.json", "r") as f:
    meme_data = json.load(f)

# Folder where meme images are stored
image_folder = "./memes"

results = []

for meme in meme_data:
    image_path = os.path.join(image_folder, meme["image_filename"])
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    raw_image = Image.open(image_path).convert("RGB")
    image_input_clip = clip_preprocess(raw_image).unsqueeze(0).to(device)

    text_inputs = clip.tokenize(meme["captions"]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs)
        logits_per_image, _ = clip_model(image_input_clip, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    predicted_index = int(probs.argmax())
    correct = (predicted_index == meme["correct_caption_index"])

    blip_inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**blip_inputs)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    prompt = f"Explain why this meme is funny: {meme['correct_caption']}"
    blip_inputs_prompt = blip_processor(raw_image, prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_prompt = blip_model.generate(**blip_inputs_prompt)
    blip_caption_with_prompt = blip_processor.decode(out_prompt[0], skip_special_tokens=True)

    results.append({
        "meme_id": meme["id"],
        "clip_prediction_index": predicted_index,
        "clip_predicted_caption": meme["captions"][predicted_index],
        "clip_correct": correct,
        "clip_probs": [float(p) for p in probs],
        "blip_caption": blip_caption,
        "blip_caption_with_prompt": blip_caption_with_prompt,
        "correct_caption": meme["correct_caption"]
    })

clip_accuracy = sum([r["clip_correct"] for r in results]) / len(results)
print(f"\nCLIP Accuracy: {clip_accuracy:.2%}\n")

for r in results[:5]:
    print(f"Meme ID: {r['meme_id']}")
    print(f"CLIP Prediction: {r['clip_predicted_caption']}")
    print(f"BLIP Caption (Image Only): {r['blip_caption']}")
    print(f"BLIP Caption (With Prompt): {r['blip_caption_with_prompt']}")
    print(f"Correct Caption: {r['correct_caption']}")
    print("--" * 30)

with open("meme_analysis_results.json", "w") as out_file:
    json.dump(results, out_file, indent=2)

print("Analysis complete. Results saved to meme_analysis_results.json")
