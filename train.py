import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from tqdm import tqdm
import requests
import argparse

HOME = os.getcwd()
print(HOME)



parser = argparse.ArgumentParser(
                    prog='Image Captioning',
                    description='Train Image Captioning')

parser.add_argument('-m', '--model', type=str, default="swin")      # option that takes a value
parser.add_argument('-d', '--dataset', type=str, default="200")      # option that takes a value
parser.add_argument('-e', '--epoch', type=int, default=3)      # option that takes a value
parser.add_argument('-b', '--batch', type=int, default=3)      # option that takes a value

args = parser.parse_args()


dataset_info ={
    "200" : "pebipebriadi/coco_traffic_200",
    "2000" : "pebipebriadi/coco_traffic_2000",
    "15557" : "pebipebriadi/coco_traffic_15557",
}

hf_dataset = dataset_info[args.dataset]

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_DISABLED"] = "true"

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# Load image and text Model
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

# tutorial
# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
# https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/


# (e.g. ViT, BEiT, DeiT, Swin) and any pre-trained language model as the decoder (e.g. RoBERTa, GPT2, BERT, DistilBERT).

# Encoder model
model_info = {
    "vit" : "google/vit-base-patch16-224-in21k",
    "swin" : "microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft",
    "beit" : "microsoft/beit-base-patch16-224",
    "deit" : "facebook/deit-base-patch16-224",
}

num_epochs = args.epoch
batch_size = args.batch

image_encoder_model = model_info[args.model]

# Decoder model
text_decoder_model = "gpt2"


# Create image captioning model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
     image_encoder_model, text_decoder_model).to(device)

# Image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)

# Corresponding Tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


output_dir = image_encoder_model.replace("/","-") + "-gpt-model-" + hf_dataset.replace("/","_")
model.save_pretrained("pretrained_model/" + output_dir)
feature_extractor.save_pretrained("pretrained_model/" +output_dir)
tokenizer.save_pretrained("pretrained_model/" +output_dir)


import datasets

ds = datasets.load_dataset(hf_dataset, "2017", data_dir="./data/", trust_remote_code=True)
print(ds)

print(os.path.exists(ds["train"][0]["image_path"]))
ds["train"][0]

import urllib.parse as parse

def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                      padding="max_length",
                      max_length=max_target_length).input_ids

    return labels

# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    model_inputs = {}

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                if check_url(image_file):
                    img = Image.open(requests.get(image_file, stream=True).raw)
                elif os.path.exists(image_file):
                    img = Image.open(image_file)

                # convert img to RGB mode
                if img.mode != "RGB":
                    img = img.convert("RGB")

                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="pt")

    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']

    captions = examples['caption']

    model_inputs = {}
    # This contains image path column
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)

    return model_inputs


processed_dataset = ds.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=ds['train'].column_names
)


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    num_train_epochs=num_epochs,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="image_captioning_output/"+output_dir,
    # report_to="none", # turn off wandb
)


import evaluate
metric = evaluate.load("rouge")


import numpy as np
import nltk

nltk.download('punkt_tab')

ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    processing_class=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)


if __name__ == "__main__":    
    trainer.train()