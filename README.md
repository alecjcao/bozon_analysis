# Bozon Analysis

Application for running data analysis in the Kaufman lab strontium experiment. Communicates with the Bozon Manager application during automatic running, but can also execute analysis in a standalone manner. Requires access to the heap and jilafile to access and save data.

## Install packages

### Conda

An environment.yml file is provided. Create a new environment with

```shell
conda env create --name <env-name> -f environment.yml
```

Activate the environment with

```shell
conda activate <env-name>
```

### Pip

A requirements.txt file is provided. Create a new environment with

```shell
python -m virtualenv <env-name>
```

Activate the environment with

```shell
source venv/bin/activate
```

Install the packages with

```shell
pip install -r requirements.txt
```

## Running the program

In the environment, run

```shell
python main.py
```

This should open up a user interface. It may take a few seconds to initialize.

For manual analysis, set the date and file number.

If it is the first run being analyzed, set the crop region and the offset. After this, the crop region only needs to be reset if the array has moved significantly on the image. The offset will be tracked as each run is analyzed, but may need to be reset if there's more than a couple pixels of motion of the array from one run to another (or if the analysis fails to track the motion). The crop can also be disabled.

To process the images and save an xarray dataset, run process images only. If an appropriate analysis script already exists, set the analysis script and then run analysis. The interface will only display the result of the image processing. The result of the image processing and the analysis are saved to the heap.

The default image processing method is to mask the image with Gaussian kernels; for denser arrays with signifcant overlap between atoms on the image, the image processing method can be switched to deconvolution instead. By default, the atoms will be assumed to be only located on the loading or target sites depending on whether there was rearrangement or not; enabling all sites will change it to analyze on all 48x48 (85x85) sites when cropped (uncropped).