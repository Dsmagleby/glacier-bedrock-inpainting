# glacier-bedrock-inpainting

## Censor feedback

The training sampling algorithm might be creating a bias in the model.
The reason behind this idea, is that the algorithm only samples patches with a middle region,
that is on average equal to or greater than the lower glacier elevation. This might lead
the model to think that for every patch the middle region has to be a certain elevation and if
the surrounding region on average is lower then the middle region, then we might be enforcing
that the patches are all centered on the top of a parabolic surface shape. This could be causing more
negative predictions.

Solutions: rethink the training data sampling algorithm AND/OR randomize the glacier position in the test
data, such that it is not always in the center. The latter makes sense, since the training glaciers are 
already randomized in such a way. Ideally both.

## Deepfillv2 fork:

https://github.com/nipponjo/deepfillv2-pytorch
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```

## Partial convolutional model forks:

https://github.com/MathiasGruber/PConv-Keras and
https://github.com/ezavarygin/PConv2D_Keras
```
@misc{https://doi.org/10.48550/arxiv.1804.07723,
  doi = {10.48550/ARXIV.1804.07723},
  url = {https://arxiv.org/abs/1804.07723},
  author = {Liu, Guilin and Reda, Fitsum A. and Shih, Kevin J. and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  title = {Image Inpainting for Irregular Holes Using Partial Convolutions},
  publisher = {arXiv},
  year = {2018},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

## Unet fork:

https://github.com/pietz/unet-keras


## How to run

1. Load PConv python environment found in Pconv/pconv_env.yml
2. Navigate to DEM_files and run script in one of the folders to download DEM tiles, required EarthData account
    1. optionally define your own region from https://search.earthdata.nasa.gov/search/ and download that, create a new folder for it
3. Run create_mosaic.py, this creates a mosaic and mosaic glacier mask
    1. define input folder, output folder and RGI region, run with --help for options
4. Run create_test.py, this creates a images and glacier masks.
    1. define input (mosaic) and outdir, default is dataset/, which creates dataset/test with images and masks
    2. also create a folder with all visible masks per image (used for training segmentation model)
    3. this procedure is very slow and shouldbe rewritten if you need more than 1000 glaciers.
5. Run create_train.py, this creates training images
    1. define input (mosaic) and outdir, default is dataset/, which creates dataset/val and train with images
    2. define RGI region to match mosaic region, default is region 11
    3. optionally define # of samples, default 25.000
6. Train UNet segmentation model, run mask_segmentation_train.py
    1. define input (test images, masks are automatically found from image paths)
    2. optionally calibrate model, see options with --help
7. Run mask_segmentation_test.py, this creates training and validation masks
    1. define input, root dataset, val and train folder are automatically picked.
    2. after creating masks for all images, any images, with no resonable glacier mask is deleted, normal ratio 1:1000
8. Train either Deepfillv2 by running deepfillv2_train.py or PConv by running PConv_train.py
    1. for Deepfillv2 define input etc. in Deepfillv2/configs/train.yml
    2. finished model can be found in Deepfillv2/callbacks/checkpoints/

    3. for PConv define input model parameters, use --help to see all options, default values are fine
    4. finished model can be found in Pconv/callbacks/best_model/
9. Create inpainted glaciers by running deepfillv2_test.py or PConv_test.py
    1. use --weights to point to trained model.
