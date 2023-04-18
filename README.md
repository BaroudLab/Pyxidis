# FL-Image-Analysis

This repository contains the main code to reproduce image and data analysis as well as figure pannels from the article "Spatiotemporal dynamics of cytokines expression dictate fetal liver hematopoiesis"

# Installation

We recommand to create a specific python environment to avoid any conflict in package versions.
```
conda create --name FL_image_analysis python=3.8
```

You need to install via the repository. In the command prompt enter:
```
git clone https://github.com/BaroudLab/FL-Image-Analysis.git
```
This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:
```
pip install .
```
The library is now installed on your computer.

# Description of repository content
This repository contains all steps to reproduce image analysis pipeline described in figure 2 of the article. Data to run these example can be dowloaded using ... (PUT LINK HERE to repository).
### Image analysis steps - in folder **Image_processing**
- 0. Training Neural Network for cell classification based on membrane staining:

**0-Training_network.ipynb**: Notebook to train squeezenet neural network (an example dataset is given in (PUT LINK HERE to repository) folder  **Example_dataset_CD45**)

- 1. Cut images in pieces for parallel segmentation using CellPose on GPUs / reconstruction of segmented image:

**1-Saucisson_preprocessing.ipynb**: Notebook to cut image in pieces for easier segmentation with Cellpose and reconstruction of the segmented image.
**scripts** contains useful functions for this step.

- 2. Classify cells using Neural Network:

**2-Classifying_cells.ipynb**: Notebook to use neural network on a set of images (a bunch of image to try is given in (PUT LINK HERE to repository) folder **Example_cell_classification_CD45**).

- 3. Threshold for nuclear staining and gather all data:

**3-Generating_final_data_file.ipynb**: Classification of nuclear signal with thresholding and concatenating with data from Neural Network classification.

- 4. Basic data analysis and distance to structures of the liver:

**4-Example_image_analysis.ipynb**: Removing disrupted regions and some typical spatial analysis.
**scripts** contains useful functions for this step.

- 5. Graph representation of the tissue and analysis of cells neighborhood:

**5-Contact_on_graph_vs_random.ipynb**: Building graph and extracting information on neighbor composition.

### Visualisation of initial image and classification

To evaluate good accuracy of segmentation and classification, it is convinient to have a graphical tool to overlap image and classification result. For this we use napari interface. The folder called "Visualization" contains 2 little scripts to to so on a 2D and 3D example images. They can be found here ... (PUT LINK HERE to repository), in the folders **2D_visualization** and **3D_visualization**.

### Description of data 

The folder **Example_dataset_CD45** contains example of a training / validation dataset to train neural network for CD45. It can be used with the notebook **0-Training_network.ipynb**. The resulting trained Neural Network is available in the folder **trained_networks**, along with other networks trained for other stainings and used for the image analysis in the article.

The folder **Example_cell_classification_CD45** contains example of classification of 400 cells with trained neural network for CD45 membrane staining. It can be used with the notebook **2-Classifying_cells.ipynb**

The folders **2D_visualization** and **3D_visualization** contains example to overlayed segmentation and classification over the original image with napari for a 2D and a 3D image (data + image)

The folder **Example_image** contains all files to run the entire image analysis pipeline on an example image (FL E12.5 with LHX2, CD45 and KIT)

  - 221212_6.tif: original image
  - saucisson_00: pieces of original image with only nucleus / segmentation with Cellpose on each piece
  - 221212_6_labeled.tif: image with segmentation after reconstruction
  - 221212_6_labeled.csv: cell positions and average intensity on different channels in nucleus masks
  - CD45_NN_classification.csv: result of neural network classification of CD45
  - Kit_NN_classification: result of neural network classification of KIT
  - 221212_6_classified.csv: result of cell classification
  - border_selection: manual selection of border cells with Coloriage
  - vessels_selection: manual selection of vessel with Coloriage
  - cells_to_remove: autofluorescent cells and disrupted regions manually selected with Coloriage
  - 221212_6_classified_corrected.csv: result of cell classification after removing autofluorescent cells and disrupted regions
  - 221212_6_napari.tif: file to drag and drop in napari (different channel order)
  
### Code to reproduce figure - in folder **Code_figures**
Scripts to reproduce main pannels from the article. Use data ... (PUT LINK HERE to repository) in folder **Data_figures** to go with it.
