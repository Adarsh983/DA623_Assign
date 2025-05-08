# Meme Understanding with CLIP and BLIP-2

This project uses state-of-the-art vision-language models—**CLIP** and **BLIP-2**—to understand memes. The goal is to:

- **Evaluate** how well CLIP matches meme images with the correct captions.
- **Generate** meme captions using BLIP-2 (with and without prompt).
- **Explain** why a meme is funny using BLIP-2's reasoning capabilities.

---

## Dataset Format

The project uses a JSON file named `real_meme_dataset_clip_blip.json`, where each entry is structured as:

```json
{
  "id": 1,
  "image_filename": "meme1.jpg",
  "captions": ["caption A", "caption B", "caption C"],
  "correct_caption_index": 2,
  "correct_caption": "caption C"
}
```

All corresponding meme images must be stored in a `./memes` folder.

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/meme-understanding-clip-blip2.git
cd meme-understanding-clip-blip2
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat   # On Windows
```

### Step 3: Install Requirements

Make sure `pip` is up to date:

```bash
python -m pip install --upgrade pip
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## `requirements.txt`

```txt
torch
clip @ git+https://github.com/openai/CLIP.git
transformers
pillow
matplotlib
```

Note: If you're using a system with CUDA support, consider installing the GPU version of PyTorch from https://pytorch.org.

---

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open `Meme_Understanding_CLIP_BLIP2.ipynb` and run the cells.

---

## Output

The notebook generates:

- CLIP prediction vs ground truth comparison
- BLIP-2 generated captions (free and prompted)
- BLIP-2 humor explanations
- Accuracy of CLIP
- A JSON log of results saved as `meme_analysis_results.json`

---

## Model Details

- **CLIP**: `ViT-B/32` from OpenAI. Uses contrastive loss between image and text embeddings.
- **BLIP-2**: `Salesforce/blip2-flan-t5-xl`. Uses frozen image encoder and Flan-T5 decoder for multi-modal generation tasks.

---

## Sample Output

```
CLIP Accuracy: 30.00%

Meme ID: 1
CLIP Prediction: "When you realize it's Monday again"
BLIP Caption: "A person with a funny face"
BLIP Caption with Prompt: "When Monday hits you harder than expected"
BLIP Reasoning: "The image and caption together depict the struggle of facing a new work week."
```

---

## Project Structure

```
.
├── Meme_Understanding_CLIP_BLIP2.ipynb
├── README.md
├── clip_blip_meme.py
├── meme_analysis_results.json
├── memes
├── real_meme_dataset_clip_blip.json
├── requirements.txt
└── README.md

```

---

## Contact

For questions or collaborations, reach out to `g.mani@iitg.ac.in`.