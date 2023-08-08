
## Run Locally

Clone the project

```bash
  git clone https://github.com/LaviSchlyter/4D-Flow-CNN-Segmentation
```

Go to the project directory

```bash
  cd 4D-Flow-CNN-Segmentation
```

Install dependencies

```bash
  conda create --name <env> --file requirements.txt
```

Preprocess the data by running 

```bash
# This is to prepare the Freiburg data
  python data_freiburg_dicom_to_numpy.py
  python data_freiburg_numpy_to_hdf5.py

# For bern the data was within the data folder (not publiblicly available) converted to numpy similarly to the Freiburg converted
# Go into the Bern_numpy_to_hdf5.ipynb notebook and depending on what type of training you want run different cells 
```

Train the network

```bash
  # In the `expriments` folder set the various parameters needed in unet.py
  # Run the bash file which launches the `train.py` onto the cluster if available
  sbatch train.sh
```

Running inference\
The data does not have test data but inference runs the best saved model on the data from Bern that was not yet manually segmented

```bash
  # In the `expriments` folder set the various parameters needed in `exp_inference.py`
  # Run the bash file which launches the `bern_inference.py` onto the cluster if available
  sbatch vern_inference.sh
```

Visualize 3D segmentation \
In the `visualization` folder you have a notebook that takes the results from the inference network and where you can visualize the results in 3D

