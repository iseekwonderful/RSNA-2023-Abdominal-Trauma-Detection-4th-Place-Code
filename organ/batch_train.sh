#!/bin/bash

python main.py train -i covid_v2l_lm_32.yaml -j covid/v2l_lm_32.yaml
python main.py train -i covid_v2l_lm_32_0002.yaml -j covid/v2l_lm_32_0002.yaml
python main.py train -i covid_v2l_lm_32_cutmix_025.yaml -j covid/v2l_lm_32_cutmix_025.yaml
