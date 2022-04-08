import torch
import clip
from PIL import Image
import json

clip_preprocessed_images = dict()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('../../../data/coco_id2file.json', 'r') as f:
    coco_id2file = json.load(f)

count_img = 0

print('len imgs', len(coco_id2file))  # 358

for img in coco_id2file:

    count_img += 1
    if count_img % 10 == 0:
        print(count_img)

    image_path = '../data/photobook_coco_images/' + coco_id2file[img]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # torch.Size([1, 3, 224, 224])

    with torch.no_grad():
        image_features = model.encode_image(image)

        clip_preprocessed_images[img] = image_features[0].tolist()

with open('../../../data/clip.json', 'w') as f:
    json.dump(clip_preprocessed_images, f)

