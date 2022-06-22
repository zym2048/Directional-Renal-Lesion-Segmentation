import os
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bounding_boxes(image):
    img = plt.imread(os.path.join(imagesDir, image[0]))
    fig, ax = plt.subplots()
    ax.imshow(img)
    for box in image[1]:
        x, y, w, h = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    # plt.show()
    plt.axis("off")
    plt.savefig(os.path.join(saveDir, image[0]), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    imagesDir = ".."
    labelPath = "..\\Kits\\Kits_coco\\annotations\\val.json"
    saveDir = "..\\Kits\\Kits_yolox_output\\yolox_s\\vis_res\\val_label"

    with open(labelPath) as f:
        label = json.load(f)

    labelImages = label['images']
    labelAnno = label['annotations']

    images = {}
    # key: image_id, file_name: str, bbox: [[], [], ...]
    # images = {image_id1: [file_name1, [[bbox1], [bbox2], ...], image_id2:...}
    for image in labelImages:
        images[image['id']] = [image['file_name'], []]

    for anno in labelAnno:
        images[anno['image_id']][1].append(anno['bbox'])

    for image_id in images:
        image = images[image_id]
        draw_bounding_boxes(image)
        print(f"{image_id} finish")

    print("Finish all")