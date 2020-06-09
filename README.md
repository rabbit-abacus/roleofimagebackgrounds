For NeurIPS 2020 Submission "Noise or Signal: The Role of Image Backgrounds in Object Recognition"

A .tar.gz containing the test sets of the 8 variations of ImageNet-9 is included in the release as a binary (Click on the "Release" tab to see it).

A python script showing how to generate nearly all types of synthetic datasets described in the paper for a single ImageNet class is shown here. It does not generate Mixed-Rand or Mixed-Next, but the transformations used are the same as for Mixed-Same.

Run *main.py -h* for options.
Example command: *python main.py --in_dir PATH_TO_IMAGENET_FOLDER --out_dir MY_OUT_DIRECTORY --ann_dir PATH_TO_IMAGENET_ANNOTATIONS_FOLDER*
