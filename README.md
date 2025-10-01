\# Fashion-MNIST Classifier



This is a project where I trained a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset.  

The goal was to practice with PyTorch and get hands-on experience with image classification.



---



\## Project files

\- `train\_fmnist.py` → script to train and test the model  

\- `requirements.txt` → list of required packages  

\- `model.pt` → saved model weights after training  



---



\## How to run

Create a virtual environment and install the requirements:



```bash

python -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt



* Then run training:



python train\_fmnist.py





The Fashion-MNIST dataset will be downloaded automatically when you run the script for the first time.



* Results:



After 5 epochs the model reached about 91.0% accuracy on the test set.



The trained weights are saved as model.pt.



* Notes:



The model runs on CPU by default (if a GPU is available it will use it automatically).

