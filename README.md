# Pyxidis

Here we present **Pyxidis**, an integrated pipeline to build a graph-based representation of large 2D and 3D tissue microscopy images. It incorporates a tool to allows segmentation of large images (cutting them in small tiles for parallel computation and reconstruction of the image), a deep-learning based approach for cell classification as well as a graph-based analysis of spatial structures. An interactive selection tool is also provided to manually anotate regions.
Pyxidis is described and used in the article "Spatiotemporal dynamics of cytokines expression dictate fetal liver hematopoiesis". This analysis pipeline and type of data analysis can be extended to large variety of images of biological tissues.

This repository contains a step by step tutorial to use this pipeline as well as example data to run main operations. Example images can be dowloaded from our Zenodo repository (DOI:10.5281/zenodo.7867025).

<p align="center">
<img src="images/general_illustration.png" width="60%" height="60%">
</p>

# Installation

We recommand to create a specific python environment to avoid any conflict in package versions.
```
conda create --name FL_image_analysis python=3.8
```

You need to install via the repository. In the command prompt enter:
```
git clone https://github.com/BaroudLab/Pyxidis.git
```
This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:
```
pip install .
```
The library is now installed on your computer.

For some application, you will need to install a napari plugin for griottes. You can do it directly in napari interface (Plugin > Install packages > napari-griottes). More documentation is provided here: https://github.com/BaroudLab/napari-griottes.
A part of the analysis presented here rely on the use of *Griottes* tool. More documentation is provided here: https://github.com/BaroudLab/Griottes and in the corresponding article: [Griottes: a generalist tool for network generation from segmented tissue images](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01376-2).

# Description of repository content
This repository contains all steps to reproduce image analysis pipeline (More description can be find in Fig2 and Methods of the article). Some light csv files to run some notebooks on examples are provided directly in the Github repository. Images files can be downloaded from Zenodo (DOI:10.5281/zenodo.7867025).

### The repository contains:
- **Notebooks** containing:
  - **Image_processing**: step by step tutorial to reproduce the entire image analysis pipeline described in figure 2 of the article as well as the main type of data analysis done in the article. csv and txt files are provided in **Data** to reproduce data analysis steps (notebooks 4 and 5), images are on Zenodo.
  - **Visualization**: To evaluate good accuracy of segmentation and classification, it is convinient to have a graphical tool to overlap image and classification result. For this we use napari interface. Here, you can find notebooks to visualize result of classification overlayed with original 2D or 3D images using napari. csv files are also provided for these notebooks in a folder called **Data** and images on Zenodo.
  
- **src** containing useful packages that are automatically installed in the installation step:
  - *plot_data*: some function to allow easy visualization of classified data as dot plots.
  - *coloriage*: a graphical tool to manually select cells in a graph representation of the tissue.
  - *saucisson*: a set of functions to cut a big image in small pieces to allow parallel segmentation followed by reconstruction of the entire segmented image.  
  
<p align="center">
<img src="images/illustration_saucisson.png" width="50%" height="50%">
<img src="images/illustration_coloriage.png" width="40%" height="40%">
</p>

### Image analysis steps (**Image_processing**)
- Training Neural Network for cell classification based on membrane staining: **0-Training_network.ipynb**.
- Cut images in pieces for parallel segmentation using CellPose on GPUs / rebuild segmented image: **1-Saucisson_preprocessing.ipynb**.
- Classify cells using Neural Network: **2-Classifying_cells.ipynb**.

<p align="center">
<img src="images/NN_classification.png" width="60%" height="60%">
</p>

- Threshold for nuclear staining and gather all data: **3-Generating_final_data_file.ipynb**.
- Removing disrupted regions and some typical spatial analysis: **4-Example_image_analysis.ipynb**.
- Graph representation of the tissue and analysis of cells neighborhood: **5-Contact_on_graph_vs_random.ipynb**.

<p align="center">
<img src="images/graph_representation.png" width="30%" height="30%">
</p>

# Description of provided data
### Data for visualization (in Notebooks>Visualization>Data)
Contain .csv files to overlayed segmentation and classification over the original image with napari for a 2D and a 3D image for visual inspection
  - data_example_2D.csv
  - data_example_3D.csv
  
### Data for image processing notebooks (in Notebooks>Image_processing>Data)
  - 221212_6_labeled.csv: cell positions and average intensity on different channels in nucleus masks.
  - CD45_NN_classification.csv: result of neural network classification of CD45.
  - Kit_NN_classification: result of neural network classification of KIT.
  - 221212_6_classified.csv: result of cell classification.
  - 221212_6_classified_corrected.csv: result of cell classification after removing autofluorescent cells and disrupted regions.
  - border_selection.txt: manual selection of border cells with Coloriage.
  - vessels_selection.txt: manual selection of vessel with Coloriage.
  - autofluorescent_cells.txt: autofluorescent cells to remove detected using neural network.
  - manual_selection_cells_to_remove.txt: disrupted regions manually selected with Coloriage.

### Data on Zenodo (DOI:10.5281/zenodo.7867025)
  - 221212_6.tif: original image (containing 2 membrane stainings (CD45 and KIT) and a nuclear staining (LHX2)
  - 221212_6_labeled.tif: image with segmentation after reconstruction.
  - 221212_6_napari.tif: file to drag and drop in napari (different channel order).
  - example_image_2D_napari.tif: tif files to overlay with classification to go with data_example_2D.csv
  - example_image_3D_napari.tif: tif files to overlay with classification to go with data_example_3D.csv
