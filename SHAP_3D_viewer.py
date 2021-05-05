# Visualize 3D SHAP values
#
# PartitionShap, among other programs, may assign SHAP values
# to each (x, y, z) cell in a 3D model input.
# For example, an image classification model may have RGB inputs
# and we are interested in the SHAP contribution of superpixels
# within each color channel

import numpy as np
import pyvista as pv
from optparse import OptionParser
from matplotlib.colors import ListedColormap

def buildGrid(values, origin=(0, 0, 0), spacing=(10, 10, 10)):
    # Spatial reference
    grid = pv.UniformGrid()

    # Grid dimensions (shape + 1)
    grid.dimensions = np.array(values.shape) + 1

    # Spatial reference params
    grid.origin = origin
    grid.spacing = spacing

    # Grid data
    grid.cell_arrays["values"] = values.flatten(order="F")

    return grid


def main():
    parser = OptionParser()
    parser.add_option("-f", "--file",
            help="Path to 3D SHAP values (.npz)",
            default="shap_values/shap_eurosat_13band_multiband_blur-100x100.npz")
    parser.add_option("-d", "--data_name",
            help="Name of SHAP values in the input SHAP values (.npz) file.",
            default="array_0")
    parser.add_option("-e", "--show_edges",
            help="Show edges of grid elements",
            default=False, action="store_true")
    options, args = parser.parse_args()
    inFile = options.file
    dataName = options.data_name
    showEdges = options.show_edges

    inNPZ = None  # Numpy archive
    values = None # SHAP values


    # Check: can open Numpy file?
    try:
        inNPZ = np.load(inFile)
    except:
        print("Could not load input SHAP values file {}. Ensure valid numpy '.npz'".format(
            inFile))
        exit(1)

    # Check: can read data from name?
    try:
        values = inNPZ[dataName]
    except:
        print("{} is not a file in the numpy (.npz) archive".format(dataName))
        exit(1)

    # Check: is data 3D?
    if (len(values.shape) != 3):
        print("Only supports 3 dimensions. Detected shape of {}".format(values.shape))
        exit(1)

    # Calc min, max
    minValue = np.min(values)
    maxValue = np.max(values)

    print("")
    print("SHAP 3D viewer")
    print("--------------")
    print("values file: {}".format(inFile))
    print("  data name: {}".format(dataName))
    print("      shape: {}".format(values.shape))
    print("      range: ({:.4f}, {:.4f})".format(minValue, maxValue))
    print("")

    percent = 0.5



    # Create grid
    grid = buildGrid(values)

    tgrid = grid.threshold_percent([0.4, 0.6], invert = True)

    p = pv.Plotter()
    # Very faint grid mesh
    p.add_mesh(grid,
               style="wireframe",
               opacity=0.075,
               cmap="seismic",
               )

    #p.add_mesh_clip_plane(grid,
    #                      cmap="seismic",
    #                      assign_to_axis='z',
    #                      invert=True)


    p.add_mesh_threshold(grid,
                         invert=True,
                         pointa=(0.1, 0.9),
                         pointb=(0.45, 0.9),
                         title = "Lower threshold",
                         cmap="seismic")

    p.add_mesh_threshold(grid,
                         pointa=(0.55, 0.9),
                         pointb=(0.9, 0.9),
                         invert=False,
                         title = "Higher threshold",
                         cmap="seismic")

    p.show()

    return 0

if __name__ == "__main__":
    main()



