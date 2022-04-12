# ReCLIP: A Strong Zero-shot Baseline for Referring Expression Comprehension
This repository contains the code for the paper [ReCLIP: A Strong Zero-shot Baseline for Referring Expression Comprehension](https://arxiv.org/abs/2204.05991)
(ACL 2022).

## Setup
This code has been tested on Ubuntu 18.04. We recommend creating a new environment with Python 3.6+ to install the appropriate versions of dependencies for this project. First, install `pytorch`, `torchvision`, and `cudatoolkit` following the instructions in `https://pytorch.org/get-started/locally/`. Then run `pip install -r requirements.txt`. Download the [ALBEF pre-trained checkpoint](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) and place it at the path `albef/checkpoint.pth`.

## Data Download
Download preprocessed data files via `gsutil cp gs://reclip-sanjays/reclip_data.tar.gz`, and extract the data using `tar -xvzf reclip_data.tar.gz`. This data
does not include images.
Download the images for RefCOCO/g/+ from [http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/train2014.zip). Download the images for RefGTA from [the original dataset release](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2). NOTE: As stated in the original RefGTA dataset release, the images in RefGTA may only be used "in non-commercial and research uses."

## Results with CLIP/ALBEF/MDETR
The following format can be used to run experiments:
```
python main.py --input_file INPUT_FILE --image_root IMAGE_ROOT --method {parse/baseline/gradcam/random} --gradcam_alpha 0.5 0.5 --box_method_aggregator sum {--clip_model RN50x16,ViT-B/32} {--albef_path albef --albef_mode itm/itc --albef_block_num 8/11} {--mdetr mdetr_efficientnetB3/mdetr_efficientnetB3_refcocoplus/mdetr_effcientnetB3_refcocog} {--box_representation_method crop,blur/crop/blur/shade} {--detector_file PATH_TO_DETECTOR_FILE} {--cache_path PATH_TO_CACHE_DIRECTORY} {--output_file PATH_TO_OUTPUT_FILE}
```
(`/` is used above to denote different options for a given argument.)

`--input_file`: should be in `.jsonl` format (we provide these files for the datasets discussed in our paper; see the Data Download information above).

`--image_root`: the top-level directory containing all images in the dataset. For RefCOCO/g/+, this is the `train2014` directory. For RefGTA, this directory contains three subdirectories called `black_wearing`, `dont_specify`, `white_wearing`.

`--detector_file`: if not specified, ground-truth proposals are used. For RefCOCO/g/+, the detection files are in `reclip_data.tar.gz` and have the format `{refcoco/refcocog/refcoco+}_dets_dict.json`. For RefGTA, the detections are in `reclip_data.tar.gz` and have the format `refgta_{val/test}_{gt/unidet_dt/unidet_all_dt}_output2.json`.

For ALBEF, we use ALBEF block num 8 for ITM (following the ALBEF paper) and block num 11 for ITC. Note that several arguments are only required for a particular "method," but they can still be included in the command when using a different method.

Choices for `method`: "parse" is the full version of ReCLIP that includes isolated proposal scoring and the heuristic-based relation handling system. "baseline" is the version of ReCLIP using only isolated proposal scoring. "gradcam" uses GradCAM, and "random" selects one of the proposals uniformly at random. (default: "parse")

Choices for `clip_model`: The choices are the same as the model names used in the CLIP repository except that the model names can be concatenated with a comma between consecutive names. (default: "RN50x16,ViT-B/32")

Choices for `box_representation_method`: This argument dictates which of the following methods is used to score proposals: CPT-adapted, cropping, blurring, or some combination of these. For CPT-adapted, choose "shade". To use more than one method, concatenate them with a comma between consecutive methods. (default: "crop,blur")

To see explanations of other arguments see the `main.py` file.

## Results with UNITER
We recommend creating a new environment for UNITER experiments. See `UNITER/requirements.txt` for the dependencies/versions that we used for these experiments. Note that the lines commented out should still be installed, but it may be easier/better to install them in a different manner than simply installing all packages at once via `pip`. In particular, we recommend first following the instructions in `https://pytorch.org/get-started/locally` to install `pytorch`, `torchvision`, and `cudatoolkit`. Then we recommend cloning `https://github.com/NVIDIA/apex` and following the instructions within that repository to install `apex`. Then we recommend installing horovod via `pip install horovod`. Then we recommend running `pip install -r requirements.txt`. Download the pre-trained UNITER model from [https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-large.pt](https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-large.pt) and place it inside `UNITER/downloads/pretrained/`. To train a model on RefCOCO+, edit `UNITER/configs/train-refcoco+-large-1gpu.json` to have the correct data paths and desired output path. The necessary data files are provided in `reclip_data.tar.gz`. Run the following command within the `UNITER/` directory to train the model:
```
python train_re.py --config config/train-refcoco+-large-1gpu.json --output_dir OUTPUT_DIR --simple_format
```
where `OUTPUT_DIR` is the desired output directory. (Training on RefCOCOg can be done in a similar manner.) Alternatively, you can download our UNITER models trained on RefCOCO+/RefCOCOg:
```
gsutil cp gs://reclip-sanjays/uniter_large_refcoco+_py10100feats.tar.gz .
gsutil cp gs://reclip-sanjays/uniter_large_refcocog_py10100feats.tar.gz .
```

To evaluate, run `bash scripts/eval_{refcoco+/refcocog/refgta}.sh OUTPUT_DIR`. Again, you will probably need to modify the data paths in `eval_{refcoco+/refcocog/refgta}.sh`.

## Synthetic relations experiment on CLEVR-like images
To obtain the accuracies for the relations task on synthetic CLEVR-like image (Section 3.2 in our paper), download the data via `gsutil cp gs://reclip-sanjays/clevr-dataset-gen.tar.gz .` and extract the data using `tar -xvzf clevr-dataset-gen`. Then run `python generic_clip_pairs.py --input_file clevr-dataset-gen/spatial_2obj_text_pairs.json --image_root clevr-dataset-gen/output/images --gpu 0 --clip_model RN50x16` to obtain results on the spatial text pair task using the CLIP RN50x16 model. Results for the spatial image pair and non-spatial image/text pair tasks can be obtained by replacing the JSON file name appropriately, and results for the other CLIP models can be obtained by replacing "RN50x16" with the appropriate model name. Results for the ALBEF model can be obtained by specifying the ALBEF path (which should be "albef"), and to obtain results with ALBEF ITC you can add the `--albef_itc` flag.

## Other
We used UniDet to detect objects for RefGTA. We provide the outputs in `reclip_data.tar.gz`, but if you would like to run the pipeline yourself, you can clone `UniDet` [https://github.com/xingyizhou/UniDet](https://github.com/xingyizhou/UniDet) and use our script in `UniDet/extract_boxes.py` on the outputs to obtain the desired detections.

We provide input features for UNITER in `reclip_data.tar.gz`, but if you would like to run the feature extraction yourself, you can clone `py-bottom-up-attention` [https://github.com/airsplay/py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention) and use our script in `py-bottom-up-attention/extract_features.py` to obtain the features for ground-truth/detected proposals. You should compile the repository (following the directions given in the repository) before running the script.

## Acknowledgements
The code in the `albef` directory is taken from the [ALBEF repository](https://github.com/salesforce/ALBEF/tree/main). The code in `clip_mm_explain` is taken from [https://github.com/hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability). The code in `UNITER` is a slightly modified version of [https://github.com/ChenRocks/UNITER](https://github.com/ChenRocks/UNITER). The script `py-bottom-up-attention/extract_features.py` is adapted from code in [https://github.com/airsplay/py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention). The file `clevr-dataset-gen/bounding_box.py` is adapted from [https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py](https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py).

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{subramanian-etal-2022-reclip,
    title = "ReCLIP: A Strong Zero-shot Baseline for Referring Expression Comprehension",
    author = "Subramanian, Sanjay  and
      Merrill, Will  and
       Darrell, Trevor and
      Gardner, Matt  and
      Singh, Sameer  and
      Rohrbach, Anna",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics"
}
```
