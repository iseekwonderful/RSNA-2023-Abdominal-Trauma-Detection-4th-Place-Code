# RSNA 2023 Abdominal Trauma Detection

Below you can find an outline of how to reproduce our solution for the `RSNA 2023 Abdominal Trauma Detection`.
If you run into any trouble with the setup/code or have any questions please contact us:
* @sheep: sss3barry@gmail.com


## 1.INSTALLATION
- Ubuntu 20.04.5 LTS
- CUDA 11.8
- Python 3.9.17 
- training requires a 4 card with 32G or 48G VRAM.
- Please install anaconda first and follow the cmd below to setup environment
```
$ conda create --name rsna_env python==3.9
$ conda activate rsna_env
$ conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
$ conda install -y matplotlib seaborn pandas scikit-learn tensorboard
$ pip install opencv-python-headless tqdm timm nibabel scikit-image albumentations python-gdcm pydicom SimpleITK path transformers
$ pip install -U pyyaml==5.1.1
$ pip install packages/segmentation_models.pytorch/ 
```

## 2.DATA
1. The convertion of dicom to pngs and the prediction of TotalSegmentator takes very long time and could be downloaded from
	* https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427427
	* https://www.kaggle.com/datasets/steamedsheep/ts-body-and-organ
	* As for the TotalSegmentator's mask prediction, I use following CMD for each series:
  ``` shell
  #!/bin/bash

# Base directory
BASE_DIR="/media/data/rsna/input/train_images"

# Loop through each subdirectory
for dir in ${BASE_DIR}/*/*; do
    if [ -d "${dir}" ]; then
        # Extract the directory structure to replicate for the output
        RELATIVE_PATH=${dir#${BASE_DIR}/}

        # Create output directory if it doesn't exist
        mkdir -p "segmentations/${RELATIVE_PATH}"
        mkdir -p "segmentations_organ/${RELATIVE_PATH}"

        # Call TotalSegmentator
        TotalSegmentator -i "${dir}/" -o "segmentations/${RELATIVE_PATH}" -ta body
        TotalSegmentator -i "${dir}/" -o "segmentations_organ/$output" --fast --roi_subset spleen kidney_right kidney_left liver small_bowel
    fi
done
	
  ```
2. go to `pp` folder and run all py file
```python
python 0.2d_mask.py  
python 1.segment_train.py  
python 2.body_segment_train.py  
python 3.rescale.py
```

## 3.TRAIN
* The `organ` folder is the project folder and model weights and logs will be save to `results` folder, plsease run follow cmd to train 4 models.
```
./train.sh
```

## 3.INFERENCE

* Since this is a code competition and the inference env already made public at kaggle, please refer to [submission notebook](https://www.kaggle.com/code/steamedsheep/candidate-3-submit-debug-1-4m-try-fix?scriptVersionId=146535393)

## Directory Structure
```
rsna_rep/
    ts/
    results/
    organ/
        basic_train.py
        batch_train.sh
        main.py
        eval.py
        train_net.py
        dist_train.py
    pp/
        0.2d_mask.py
        1.segment_train.py
        3.rescale.py
        2.body_segment_train.py
    input/
    packages/
```