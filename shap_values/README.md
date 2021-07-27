# Saved SHAP values

The saved SHAP values from the notebooks are stored here. These may be used as input to visualization programs. 

The visualization tool supports both SHAP output `.pickle` files and generic numpy `.npz` files.
The advantage of the `.pickle` is that it has metadata in addition to the SHAP values. 
The disadvantage is that it requires the SHAP library to be installed and loaded.
The advantage of the `.npz` is that it is not restricted to SHAP values. Any XAI output in a 3D numpy array may be used. 

## Suggested filename template

The template for filenames is: `shap_datasource_bands_single-or-multiband_feature-removal-method.filetype`

For example: `shap_imagenet_rgb_multiband_inpaint-telea.pickle`
- Data source: ImageNet
- Bands: RGB (3)
- Explanations: multiband (a.k.a. channel-wise)
- Feature removal: inpaint telea
- Filetype: pickle   (choose `.pickle` or `.npz`)

## SHAP `.pickle` file

The following saves the SHAP values and metadata output by SHAP explainer functions:

    
    with open(filename, 'wb') as f:
        pickle.dump(shap_values, f)

And may be loaded with:

    import shap
    shap_values = pickle.load(open(pickleFile, "rb"))

## Numpy `.npz` file

The following saves SHAP values for the top prediction (class 0) from the first 2 instances:

    def save_shap(shap_values, filename, Class=0):
      # Save selected SHAP values
      image0 = shap_values[0, :, :, :, Class].values
      image1 = shap_values[1, :, :, :, Class].values
      np.savez_compressed(filename, array_0=image0, array_1=image1)
      
And may be loaded with:

    shap_values = np.load(filename)
    shap_values_1 = shap_values["array_0"]   # Instance 1
    shap_values_2 = shap_values["array_1"]   # Instance 2

