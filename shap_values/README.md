# Saved SHAP values

The saved SHAP values from the notebooks are stored here. These may be used as input to visualization programs. 

The template for filenames is: `shap_datasource_bands_single-or-multiband_feature-removal-method.npz`

For example: `shap_imagenet_rgb_multiband_inpaint-telea.npz`
- Data source: ImageNet
- Bands: RGB (3)
- Explanations: multiband (a.k.a. channel-wise)
- Feature removal: inpaint telea

Each file SHAP values for 2 instances, as shown in the notebooks. The are saved with the following code:

    def save_shap(shap_values, filename, Class=0):
      # Save selected SHAP values
      image0 = shap_values[0, :, :, :, Class].values
      image1 = shap_values[1, :, :, :, Class].values
      np.savez_compressed(filename, array_0=image0, array_1=image1)
      
And may be loaded with the following code:

    shap_values = np.load(filename)
    shap_values_1 = shap_values["array_0"]   # Instance 1
    shap_values_2 = shap_values["array_1"]   # Instance 2

