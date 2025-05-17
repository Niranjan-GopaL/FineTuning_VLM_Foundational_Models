import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import inspect
import types
import os


from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load image_dir and metadata CSV
    IMAGE_DIR = args.image_dir
    df = pd.read_csv(args.csv_path)

    # load base model then inject adapters
    base_model_id = "Salesforce/blip-vqa-base"
    repo_name = "GiganticTiger/fine_tuned_blip_lora"
    processor = BlipProcessor.from_pretrained(repo_name)
    model = BlipForQuestionAnswering.from_pretrained(base_model_id)
    lora_model = PeftModel.from_pretrained(model, repo_name)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_model.to(device)
    
    # 1. Get original method
    _orig_forward = BlipForQuestionAnswering.forward

    # 2. Allowed kwargs from the method signature
    _allowed = set(inspect.signature(_orig_forward).parameters.keys())

    # 3. Define patched forward
    def _patched_forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
        return _orig_forward(self, *args, **filtered_kwargs)

    # 4. Bind patched method to instance
    lora_model.base_model.forward = types.MethodType(_patched_forward, lora_model.base_model)

    # Send to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_model.to(device)
    lora_model.eval()

    generated_answers = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Predictions"):
        image_path = os.path.join(IMAGE_DIR, row["image_name"])
        question = row["question"]
        answer = row["answer"]

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            image,
            question,
            return_tensors="pt",
            padding=True, truncation=True, max_length=32
        ).to(device)

        # Ensure answer is one word and in English (basic post-processing)
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
        generated_ids = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        answer = str(generated_ids).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    print(generated_answers)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
