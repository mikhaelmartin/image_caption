# Accesssing images from the web
import urllib.parse as parse
import os
import requests
import torch
from PIL import Image
from IPython.display import display
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

HOME = os.getcwd()
print(HOME)

import os
os.environ['HF_HOME'] = f'{HOME}/hf_home/'


device = "cuda" if torch.cuda.is_available() else "cpu"
image_encoder_model = "microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft"
text_decoder_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
     image_encoder_model, text_decoder_model).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)

# Verify url
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# Load an image
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    

# Image inference
def get_caption(model, image_processor, tokenizer, image_path):

    # Preprocessing the Image
    # img = image_processor(image, return_tensors="pt").to(device)
    img = image_processor([image_path])
    img_tensor = torch.Tensor(img).to(device)

    # Generating captions
    output = model.generate(img_tensor)

    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption



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



if __name__ == "main":

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Japanese_car_accident_blur.jpg/800px-Japanese_car_accident_blur.jpg"

    display(load_image(url))

    get_caption(
        model,
        feature_extraction_fn,
        tokenizer,
        url
    )