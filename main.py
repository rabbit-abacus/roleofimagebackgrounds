import os
import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from torchvision import transforms
from argparse import ArgumentParser

standard_transform = transforms.Compose(
    [transforms.Resize(256, Image.NEAREST), transforms.CenterCrop(256)]
)

test_transform = transforms.Compose(
    [transforms.Resize(256, Image.NEAREST), transforms.CenterCrop(224)]
)


# Helper function for directories
def make_if_not_exists(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)


# Takes in PIL image and mask, returns blacked PIL image (affects mask==255).
def blackout(img, mask):
    np_mask = np.array(mask)
    np_img = np.array(img)
    if len(np_img.shape) == 2:
        np_img = np.stack((np_img,) * 3, axis=-1)
    np_img[np_mask == 255] = 0  # Set mask white to black
    return Image.fromarray(np_img)


# Takes in two PIL images and mask, returns PIL image where mask==255 is set to
# image 2, and otherwise it is set to image 1.
def combine(img1, img2, mask):
    np_mask = np.array(mask)
    np_img1 = np.array(img1)
    np_img2 = np.array(img2)
    np_img1[np_mask == 255] = np_img2[np_mask == 255]  # Set mask black to img2
    return Image.fromarray(np_img1)


# Takes in PIL image and a rectangle surrounding the entire foreground,
# finds segmented foreground only. Returns whether segmentation worked
# and PIL image.
def get_fg_mask(img, xmin, ymin, xmax, ymax):
    np_img = np.array(img)
    mask = np.zeros(np_img.shape[:2], np.uint8)
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)
    rect = (xmin, ymin, xmax - xmin, ymax - ymin)
    try:
        cv2.grabCut(np_img, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
    except:
        return None, False

    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Check if grabCut failed
    mask_proportion = np.sum(fg_mask) / ((xmax - xmin) * (ymax - ymin))
    MIN_MASK_AMOUNT = 0.1
    if mask_proportion < MIN_MASK_AMOUNT:
        success = False
    else:
        success = True

    return fg_mask, success


# Takes in PIL image and a rectangle surrounding the entire foreground,
# finds tile from background and repeats that tile. Returns PIL image.
def get_bg_tiled(img, all_xmin, all_ymin, all_xmax, all_ymax):
    width, height = img.size
    xmin_margin = all_xmin
    ymin_margin = all_ymin
    xmax_margin = width - all_xmax
    ymax_margin = height - all_ymax
    max_x_area = max(xmin_margin, xmax_margin) * height
    max_y_area = max(ymin_margin, ymax_margin) * width
    try:
        assert max(max_x_area, max_y_area) > 0
    except:
        raise ValueError("No background rectangles left in this image")
    use_horizontal = max_x_area > max_y_area
    if use_horizontal:
        tile_ymin, tile_ymax = 0, height
        if xmin_margin > xmax_margin:
            tile_xmin, tile_xmax = 0, all_xmin
        else:
            tile_xmin, tile_xmax = all_xmax, width
    else:
        tile_xmin, tile_xmax = 0, width
        if ymin_margin > ymax_margin:
            tile_ymin, tile_ymax = 0, all_ymin
        else:
            tile_ymin, tile_ymax = all_ymax, height
    tile = img.crop([tile_xmin, tile_ymin, tile_xmax, tile_ymax])
    tile_w, tile_h = tile.size
    bg_tiled = Image.new("RGB", (width, height))
    for i in range(0, width, tile_w):
        for j in range(0, height, tile_h):
            bg_tiled.paste(tile, (i, j))
    return bg_tiled


# Gets path to the image file
def get_image_path(directory, synset, image_num):
    return f"{directory}/{synset}/ILSVRC2012_val_000{image_num}.JPEG"


# Gets path to the annotation file
def get_annotation_path(directory, synset, image_num):
    return f"{directory}/ILSVRC2012_val_000{image_num}.xml"


# Tests if an image is a good image. Returns 0 if it is, and 1-6 if it isn't.
def is_good_image(in_dir, ann_dir, synset, image_num):
    # Set full paths
    image_path = get_image_path(in_dir, synset, image_num)
    annotation_path = get_annotation_path(ann_dir, synset, image_num)
    # Check if the image exists
    if not os.path.exists(image_path):
        return 1

    # Check if the annotation exists
    if not os.path.exists(annotation_path):
        return 2

    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Load annotation
    annotation_file = open(annotation_path)
    root = ET.fromstring(annotation_file.read())
    annotation_file.close()

    # Make sure that the width and height attributes match
    annotated_width = int(root.find("size").find("width").text)
    annotated_height = int(root.find("size").find("height").text)
    if annotated_width != width or annotated_height != height:
        return 3

    # For simplicity, only consider images with exactly 1 bounding box
    all_bounding_boxes = root.findall("object")
    if len(all_bounding_boxes) != 1:
        return 4

    # For Only-BG: Ensure that the removed bounding box is less than FRACTION
    # of the resulting (post center-crop) image. This prevents (nearly) all-black images
    FRACTION = 0.9

    bounding_box = all_bounding_boxes[0]
    xmin = int(bounding_box.find("bndbox").find("xmin").text)
    ymin = int(bounding_box.find("bndbox").find("ymin").text)
    xmax = int(bounding_box.find("bndbox").find("xmax").text)
    ymax = int(bounding_box.find("bndbox").find("ymax").text)
    # Create the mask
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([xmin, ymin, xmax, ymax], fill="white")

    post_crop = test_transform(mask)
    num_masked_pixels = np.sum(post_crop) // (255 * 3)

    area_ratio = num_masked_pixels / 224 ** 2
    if area_ratio > FRACTION:
        return 5

    # For Only-FG, ensure that the bounding box is mostly (at least FRACTION_2)
    # inside the center crop.
    FRACTION_2 = 0.5

    post_resize_only = transforms.Resize(256, Image.NEAREST)(mask)
    num_masked_pixels_resize = np.sum(post_resize_only) // (255 * 3)
    if num_masked_pixels < FRACTION_2 * num_masked_pixels_resize:
        # Size of bounding box inside center crop is too small relative
        # to the size of bounding box (post-resize-only)
        return 6

    # 0 means the image is good!
    return 0


# Loads a random image from the directory (for mixed background
def get_rand_image(img_dir):
    all_imgs = os.listdir(img_dir)
    num_imgs = len(all_imgs)
    img_num = np.random.randint(num_imgs)
    img_file = all_imgs[img_num]

    synset, image_num = img_file.strip(".JPEG").split("_")
    return synset, image_num


# As is, this is coded to work for just the validation set for now.
def main():

    parser = ArgumentParser()
    parser.add_argument("--in_dir", help='ImageNet directory path')
    parser.add_argument(
        "--out_dir", help="Where you want to output synthetic datasets to"
    )
    parser.add_argument("--ann_dir", help='Annotations directory path')
    parser.add_argument("--synset", default="n02279972", help='Which synset to convert')  # default: monarch butterfly

    args = parser.parse_args()

    # Hard coded to work on the val set for now
    in_dir = f"{args.in_dir}/val"
    out_dir = args.out_dir
    ann_dir = f"{args.ann_dir}/val"
    synset = args.synset

    in_ims_dir = f"{in_dir}/{synset}"
    im_files = os.listdir(in_ims_dir)
    num_ims_processed = 0
    IMS_PER_PRINTOUT = 10
    start = time.time()

    # First, make original, only_bg_b, only_bg_t for each image
    for im_file in im_files:
        if num_ims_processed % IMS_PER_PRINTOUT == 0:
            end = time.time()
            print(
                f"Took {end-start} to process last {IMS_PER_PRINTOUT} images up to {num_ims_processed}."
            )
            start = time.time()

        # This line only works for standard imagenet .JPEG images and filename conventions
        image_num = im_file[-10:-5]

        check_good_image = is_good_image(in_dir, ann_dir, synset, image_num)
        if check_good_image != 0:
#             print(
#                 f"{synset}, {image_num} is not a good image, had error {check_good_image}. Skipping it."
#             )
            num_ims_processed += 1
            continue

        # Set full paths
        image_path = get_image_path(in_dir, synset, image_num)
        annotation_path = get_annotation_path(ann_dir, synset, image_num)

        ims = {}
        # Load image
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        ims["original"] = img.copy()

        # Load annotation
        annotation_file = open(annotation_path)
        root = ET.fromstring(annotation_file.read())
        annotation_file.close()

        # Identify all bounding boxes
        all_bounding_boxes = root.findall("object")
        bounding_box = all_bounding_boxes[0]
        xmin = int(bounding_box.find("bndbox").find("xmin").text)
        ymin = int(bounding_box.find("bndbox").find("ymin").text)
        xmax = int(bounding_box.find("bndbox").find("xmax").text)
        ymax = int(bounding_box.find("bndbox").find("ymax").text)

        # Create the mask
        mask = Image.new("RGB", (width, height), (0, 0, 0))
        fill_color = "white"
        draw = ImageDraw.Draw(mask)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)

        ims["only_bg_b"] = blackout(img, mask)
        bg_tiled = get_bg_tiled(img, xmin, ymin, xmax, ymax)
        ims["only_bg_t"] = combine(img, bg_tiled, mask)

        # Try segmenting the foreground, just to see if it works.
        # Don't save anything if it doesn't work.
        fg_mask, success = get_fg_mask(img, xmin, ymin, xmax, ymax)
        if not success:
            num_ims_processed += 1
            continue

        # If segmentation did work, save some images
        for dirname, im in ims.items():
            # Hard coded to just do the validation sets
            full_dirname = f"{out_dir}/{dirname}/val/{synset}"
            make_if_not_exists(full_dirname)
            im.save(f"{full_dirname}/{synset}_{image_num}.JPEG")

        num_ims_processed += 1

    # Does all images requiring segmentation, and keeps a consistent segmentation mask.
    # This code just shows an example with mixed_same.
    # Change the background class to make mixed_rand and mixed_next.
    # Need to redo segmentations from scratch (as opposed to loading only_fg/no_fg)
    # to avoid JPEG compression issues.
    good_ims_dir = f"{out_dir}/original/val/{synset}"
    good_im_files = os.listdir(good_ims_dir)

    bgclass_synsets = {}
    bgclass_synsets["same"] = synset

    num_ims_processed = 0
    start = time.time()
    for im_file in good_im_files:
        if num_ims_processed % IMS_PER_PRINTOUT == 0:
            end = time.time()
            print(
                f"Took {end-start} to process last {IMS_PER_PRINTOUT} images up to {num_ims_processed}."
            )
            start = time.time()

        image_num = im_file[-10:-5]

        # Set full paths
        image_path = get_image_path(in_dir, synset, image_num)
        annotation_path = get_annotation_path(ann_dir, synset, image_num)

        ims = {}
        mixed_ims = {}
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Load annotation
        annotation_file = open(annotation_path)
        root = ET.fromstring(annotation_file.read())
        annotation_file.close()

        # Identify all bounding boxes
        all_bounding_boxes = root.findall("object")
        bounding_box = all_bounding_boxes[0]
        xmin = int(bounding_box.find("bndbox").find("xmin").text)
        ymin = int(bounding_box.find("bndbox").find("ymin").text)
        xmax = int(bounding_box.find("bndbox").find("xmax").text)
        ymax = int(bounding_box.find("bndbox").find("ymax").text)

        # Get FG mask. Try it a few times just in case.
        success = False
        segmentation_fails = False
        failures_counter = 0
        while (not success) and (not segmentation_fails):
            # Try it just one time
            fg_mask, success = get_fg_mask(img, xmin, ymin, xmax, ymax)
            failures_counter += 1
            if failures_counter > 10:
                segmentation_fails = True

        if segmentation_fails:
            print(
                f"for {synset}, {image_num}, segmentation fails. May need to remove this image from other datasets later."
            )
            continue

        np_img = np.array(img)
        ims["only_fg"] = Image.fromarray(np_img * fg_mask[:, :, np.newaxis])
        ims["no_fg"] = Image.fromarray(np_img * (1 - fg_mask[:, :, np.newaxis]))

        fg_resized = standard_transform(ims["only_fg"])
        bg_mask = np.all(np.array(fg_resized) == [0, 0, 0], axis=-1) * 255

        for bgtype, bg_synset in bgclass_synsets.items():
            # Hard coded to work on the val set for now
            bg_dir = f"{out_dir}/original/val/{bg_synset}"
            _, bg_image_num = get_rand_image(bg_dir)
            bg_image_path = get_image_path(in_dir, bg_synset, bg_image_num)
            bg_annotation_path = get_annotation_path(ann_dir, bg_synset, bg_image_num)

            # Load image
            img = Image.open(bg_image_path).convert("RGB")
            width, height = img.size

            # Load annotation
            annotation_file = open(bg_annotation_path)
            root = ET.fromstring(annotation_file.read())
            annotation_file.close()

            # Identify all bounding boxes
            all_bounding_boxes = root.findall("object")
            bounding_box = all_bounding_boxes[0]
            xmin = int(bounding_box.find("bndbox").find("xmin").text)
            ymin = int(bounding_box.find("bndbox").find("ymin").text)
            xmax = int(bounding_box.find("bndbox").find("xmax").text)
            ymax = int(bounding_box.find("bndbox").find("ymax").text)

            # Create the mask, as before
            mask = Image.new("RGB", (width, height), (0, 0, 0))
            fill_color = "white"
            draw = ImageDraw.Draw(mask)
            draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)

            bg_tiled = get_bg_tiled(img, xmin, ymin, xmax, ymax)
            only_bg_t = combine(img, bg_tiled, mask)
            bg_resized = standard_transform(only_bg_t)
            mixed_ims[f"mixed_{bgtype}"] = (
                combine(fg_resized, bg_resized, bg_mask),
                bg_synset,
                bg_image_num,
            )

        for dirname, im in ims.items():
            # Hard coded to work on the val set for now
            full_dirname = f"{out_dir}/{dirname}/val/{synset}"
            make_if_not_exists(full_dirname)
            im.save(f"{full_dirname}/{synset}_{image_num}.JPEG")

        for dirname, im_info in mixed_ims.items():
            actual_im, im_bg_synset, im_bg_image_num = im_info
            # Hard coded to work on the val set for now
            full_dirname = f"{out_dir}/{dirname}/val/{synset}"
            make_if_not_exists(full_dirname)
            actual_im.save(
                f"{full_dirname}/fg_{synset}_{image_num}_bg_{im_bg_synset}_{im_bg_image_num}.JPEG"
            )

        num_ims_processed += 1


if __name__ == "__main__":
    main()
