## Jupyter folder

In this folder you can find the Jupyter notebooks used to train, convert and export Keras and NNoM models.

In order to run the files please install the required packages that can be found in the requirements.txt file (`pip install -r requirements.txt`). 
Train and test datasets can be downloaded from here: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Some notebooks have been used on Google Colab, please avoid running cells marked as Colab on local Jupyter servers.

Folder structure and file explanation:
<ul>
  <li><b>CNN_Model_Comparison</b>: notebook used to compare several neural net, saving key metrics and a computed weight basing on a simple mathematical model.</li>
  <li><b>CNN_Post_Training_Quantization_Model</b>: notebook to quantize, convert and export a Keras model (.h5) to a Tensorflow lite model. Both fp16 and int8 models with fp32 and int8 inputs are exported.</li>
  <li><b>CNN_Quantization_Aware_Model</b>: notebook to re-train the neural network with a quatization-aware approach.</li>
  <li><b>CNN_Results_Comparison</b>: notebook to plot the results obtained by the model comparison.</li>
  <li><b>CNN_Training</b>: notebook to implement the state of the art model "SermaNet" with a linear Keras implementation.</li>
  <li><b>CNN_Training_Optimized_Model</b>: notebook to implement the best neural network as of memory and validation loss from the model comparison. Keras implementation and metrics are proposed.</li>
</ul>

The folders presents the exported models in "Models", the scripts to quantize and to convert the model with NNoM in "NNoM" and some test data to validate the NN with CubeAI in "Test_data"
