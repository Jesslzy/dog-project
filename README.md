[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/dog_web_app.jpeg "Web App"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, I have build a neural network model using transfer learning from the InceptionV3 model architecture. The model can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify that human faces are present. If neither a human nor a dog is presented in the supplied image, the algorithm will let you know.

![Sample Output][image1]

After training and tuning the neural network, I have achieved a 78% of accuracy in classifying the breed of a dog.

## Table of Contents

  * [Project Instructions](#Project Instructions)
  * [File Description](#File-Description)
  * [Results](#Results)
  * [License](#license)

## Project Instructions

### Get Data Instructions

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/dogImages`. 
2. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.
3. Download the [InceptionV3 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for additional dog dataset. Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

### Requirements

#### Run dog_app.ipynb
For running dog_app.ipynb, GPU mode is recommended considering the complexity of neural network models. For instructions to get GPU support locally, please se below options. Apart from GPU, there should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*

1. __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

2. **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

3. **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
4. **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

#### Run flask app
To run flask app, clone this repo, cd to the app folder in your console and run following:
```
python dog_classifier.py
```

## File Descrption
- `app` <br>
  &nbsp;| - `templates` <br>
  &nbsp;| --- `master_dog.html`: main page of web app <br>
  &nbsp;| --- `go_dog.html`: extended section of web app to display classification results <br>
  &nbsp;| - `dog_classifier.py`: Dog breed classifier app <br>
- `bottleneck_features` <br>
  &nbsp;| &nbsp;&nbsp; Directory to store bottleneck features for the `dog_app` notebook (empty in git) <br>
- `data` <br>
  &nbsp;| - `dog_names.json`: json file containing names of dog breeds available to be classified <br>
- `haarcascades` <br>
  &nbsp;| - `haarcascades_frontalface_alt.xml`: model weights for human face detector model <br>
- `images` <br>
  &nbsp;| &nbsp;&nbsp; 13 `jpg/png` image files used in the `dog_app` notebook and README <br>
- `requirements` <br>
  &nbsp;| - `dog-linux-gpu.yml`: GPU `dog_app` notebook anaconda python environment export <br>
  &nbsp;| - `dog-linux.yml`: CPU `dog_app` notebook anaconda python environment export <br>
  &nbsp;| - `dog-mac-gpu.yml`: GPU `dog_app` notebook anaconda python environment export for mac user<br>
  &nbsp;| - `dog-mac.yml`: CPU `dog_app` notebook anaconda python environment export for mac user<br>
  &nbsp;| - `dog-windows-gpu.yml`: GPU `dog_app` notebook anaconda python environment export for windows user<br>
  &nbsp;| - `dog-windows.yml`: CPU `dog_app` notebook anaconda python environment export for windows user<br>
  &nbsp;| - `requirements-gpu.txt`: GPU `dog_app` notebook pip python environment export <br>
  &nbsp;| - `requirements.txt`: CPU `dog_app` notebook pip python environment export <br>  
- `saved_models` <br>
  &nbsp;| - `dogBreedXception.h5`: re-trained imagenet model for dog breed classification <br>
- `static` <br>
  &nbsp;| &nbsp;&nbsp; 7 `jpg/jpeg` image files user uploaded for classification <br>
- `README.md`: readme file
- `.gitignore`: file/folders to ignore
- `dog_app.ipynb`: notebook used for exploration, dog breed classification model creation and training.
- `extract_bottleneck_features.py`: python file used for extracting bottleneck features.


## Results
Depending on the nature of the project, the result is shown and can be tested in web page.
![Web App][image2]
---
## Licensing, Authors, and Acknowledgements
For this project, credit must give to Udacity for the data. Any enquiry with the Licensing for the data should directly go to Udacity. Otherwise, feel free to use the code here as you would like! 
