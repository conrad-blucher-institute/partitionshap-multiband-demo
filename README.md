# `PartitionShap` demo: channel-wise explanations of raster-based machine learning models
Exploring PartitionShap on raster datasets of an arbitrary number of bands/channels

These notebooks use our [SHAP fork](https://github.com/conrad-blucher-institute/shap) with modifications to make it easier to use and visualize multi-channel explanations. 
The EuroSAT notebooks use [TorchSat](https://github.com/sshuair/torchsat), a library for working with PyTorch models whose inputs are rasters with arbitrary number of channels. 

#### Citation

    @inproceedings{krell2022explaining,
      title={Explaining Complex 3D Atmospheric CNNs Using SHAP-Based Channel-wise XAI Techniques with Interactive 3D Visualization},
      author={Krell, Evan and Kamangir, Hamid and Friesen, Josh and Judge, Julianna and Collins, Waylon G and King, Scott Alan and Tissot, Philippe E},
      booktitle={102nd American Meteorological Society Annual Meeting},
      year={2022},
      organization={AMS}
    }

### Notebooks

1. [`PartitionSHAP`: ImageNet (RGB) demo](PartitionSHAP_ImageNet.ipynb)
2. [`PartitionSHAP`: EuroSAT (RGB) demo](PartitionSHAP_EuroSAT_RGB.ipynb)
   * pretrained weights: https://drive.google.com/file/d/14BYvrjem4dbmkmibmmSD2cDq2ZvStfVG/view?usp=sharing
3. [`PartitionSHAP`: EuroSAT (13-band) demo](PartitionSHAP_EuroSAT_13bands.ipynb)
   * pretrained weights: https://drive.google.com/file/d/1gYtOpYdMCCMxCkEhRzK1dDyE6tOMIOKa/view?usp=sharing
4. [`PartitionSHAP`: Tornado (4-band) demo](PartitionSHAP_tornado.ipynb)

### 3D SHAP viewer tool (prototype)

A simple prototype of the viewer is included for completeness. 
Further development of this tool occurs in the [xai-raster-vis-tools](https://github.com/conrad-blucher-institute/xai-raster-vis-tools) repo.

    python SHAP_3D_viewer.py --help
    Usage: SHAP_3D_viewer.py [options]

    Options:
      -h, --help            show this help message and exit
      -f FILE, --file=FILE  Path to 3D SHAP values (.npz)
      -d DATA_NAME, --data_name=DATA_NAME
                            Name of SHAP values in the input SHAP values (.npz)
                            file.
      -e, --show_edges      Show edges of grid elements

[<img src="play_video.png">](https://youtu.be/kNFY6ff996E)


### Todo

- [ ] Evan: Use test instead of validation data for XAI runs in notebooks -> not currently repoducable since will choose a random validation case
- [ ] Josh: Add colab notebook with RGB & RGB+NIR model train, test, PartitionShap results (other XAI not is not for this repo!)
