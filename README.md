# Signal Modulation Classification Using Machine Learning
## Morad Shefa, Gerry Zhang, Steve Croft
If you want to skip all the readings and want to see what we provide and how you can use our code feel free to skip to the final section. Also, you can reach me at moradshefa@berkeley.edu
## Background
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  A deep convolutional neural network architecture is used for signal modulation classification. There are different reasons why signal modulation classification can be important. For example, radio-frequency interference (RFI) is a major problem in radio astronomy. This is especially prevalent in SETI where RFI plagues collected data and can exhibit characteristics we look for in SETI signals. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  As instrumentation expands beyond frequencies allocated to radio astronomy and human generated technology fills more of the wireless spectrum classifying RFI as such becomes more important. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  Modulation schemes are methods of encoding information onto a high frequency carrier wave, that are more practical for transmission. Human-generated RFI tends to utilize one of a limited number of modulation schemes. Most of these methods modulate the amplitude, frequency, or phase of the carrier wave. Thus one way of classifying RFI is to classify it as a certain modulation scheme.


<img align="left" width="581" height="327" src="https://user-images.githubusercontent.com/22650698/44496778-e247e580-a62a-11e8-9050-31ad4b8184ff.png">

<br><br>Examples of how information can be transmitted by changing the shape of a carrier wave. Picture credit: Tait Radio Academy<br><br><br><br><br><br><br><br>


## Methods and Materials
* Datasets provided by the Army Rapid Capabilities Office’s Artificial Intelligence Signal Classification challenge
* Simulated signals of 24 different modulations: 16PSK, 2FSK_5KHz, 2FSK_75KHz, 8PSK, AM_DSB, AM_SSB, APSK16_c34,  APSK32_c34, BPSK, CPFSK_5KHz, CPFSK_75KHz, FM_NB, FM_WB,  GFSK_5KHz, GFSK_75KHz, GMSK, MSK, NOISE, OQPSK, PI4QPSK, QAM16,  QAM32, QAM64, QPSK
* 6 different signal to noise ratios (SNR): -10 dB, -6 dB, -2 dB, 2 dB, 6 dB, 10 dB
* Used deep convolutional neural networks for classification
* CNNs are widely used and have advanced performance in computer vision
* Convolutions with learned filters are used to extract features in the data
* Hierarchical classification: Classify into subgroups then use another classifier to identify modulation
* Data augmentation: Perturbing the data during training to avoid overfit
* Ensemble training: Train multiple models and average predictions
* Residual Connections: Allow for deeper networks by avoiding vanishing gradients

<img align="left" width="444" height="211" src="https://user-images.githubusercontent.com/22650698/44498003-348c0500-a631-11e8-87ee-67e76453b903.png">

<br><br>
Inception Layers: 
* Layers with filters of different dimensions
* 1x1 convolutions to reduce dimension
<br><br>Picture credit:  GoogLeNet
<br><br><br><br><br>


<img align="left" width="444" height="444" src="https://user-images.githubusercontent.com/22650698/44546315-669e7500-a6cc-11e8-997f-69378706b6f0.png">

<br><br>
Dimensionality reduction using t-distributed stochastic neighbor embedding (t-SNE) and principal component analysis (PCA) to visualize feature extraction and diagnose problems of the architecture. 
<br><br>These t-SNE plots helped us to evaluate our models on unlabelled test data that was distributed differently than training data.
<br><br>Dimensionality reduction after extracting features of 16PSK (red), 2FSK_5kHz (green),AM_DSB (blue)
<br><br><br><br><br><br><br><br>


<img align="left" width="444" height="211" src="https://user-images.githubusercontent.com/22650698/44546552-24296800-a6cd-11e8-8417-aaa4f244e42e.png">

<br>
Example of a vanilla convolutional neural network. <br>Picture credit: MDPI: "A Framework for Designing the Architectures of Deep Convolutional Neural Networks"
<br><br><br><br><br><br><br>


<img align="left" width="444" height="444" src="https://user-images.githubusercontent.com/22650698/44546670-85513b80-a6cd-11e8-93be-83c6feefab2d.png">

Embedding: 
* Extracting output of final inception layer; 100 per modulation (dimension: 5120)
* Reducing dimension using principal component analysis (dimension: 50)
* Reducing dimension using t-distributed neighbor embedding (dimension: 2)

<br>*Embedding of 24 modulations using one of our models. As we can see different modulations map to different clusters even in 2-dimensional space indicating that our model does well in extracting features that are specific to the different modulation schemes. The axis have no physical meaning. They merely represent the space found by t-SNE in which close points in high dimension stay close in lower dimension.*
<br><br>

## Results
Results for one of our models without hierarchical inference.

<img align="left" width="259" height="294" src="https://user-images.githubusercontent.com/22650698/44547003-8d5dab00-a6ce-11e8-9ec8-5509adc178c8.png">

CNNs are able to achieve high accuracy in classification of signal modulations across different SNR values
<br><br><br><br><br><br><br><br><br><br><br><br><br>

<br><img src="https://user-images.githubusercontent.com/22650698/44547386-cba79a00-a6cf-11e8-9a28-e179aba3c12c.png"  width="330" height="330" /> <img src="https://user-images.githubusercontent.com/22650698/44547206-34424700-a6cf-11e8-8f20-6680c34529cc.png"  width="315" height="315" /> 


<img align="left" width="360" height="340" src="https://user-images.githubusercontent.com/22650698/44547394-d4986b80-a6cf-11e8-8f6d-d587482d9281.png">
<br><br>Confusion matrices for different SNR values (-10 dB, -6 dB, 10 dB) 
<br>As SNR increases accuracies increase and more predicted labels are true labels causing stronger diagonals

<br><br><br><br><br><br><br><br><br>

## Conclusion
* The ability of CNNs to classify signal modulations at high accuracy shows great promise in the future of using CNNs and other machine learning methods to classify RFI
* Future work can focus on extending these methods to classify modulations in real data
* One can use machine learning methods  to extend these models to real data
  * Use domain adaption to find performing model for a target distribution that is different from the source distribution/ training data
  * Label real data and train a classifier

<img align="left" width="390" height="300" src="https://user-images.githubusercontent.com/22650698/44547872-4624e980-a6d1-11e8-849f-0ae5f8f9fe6b.png">
Adapting models to domains that are related but different than what was trained on is a common challenge for machine learning systems. Picture credit: Oxford Robotics Institute
<br><br><br><br><br><br><br><br><br><br><br><br>

<img align="left" width="390" height="170" src="https://user-images.githubusercontent.com/22650698/44547982-a2880900-a6d1-11e8-915d-75943d41cdb6.png">
When the target distribution is different classification performance can suffer. Domain adaptation methods aim at finding a space in which the discrepancy is low. Picture credit: Science Direct: “Unsupervised domain adaptation techniques based on auto-encoder for non-stationary EEG-based emotion recognition”
<br><br><br>

## Provided
Unfortunately, as part of the army challenge rules we are not allowed to distribute any of the provided datasets. 
However, we will provide:
* a notebook that we used to experiment with different models and that is able to achieve 
our results with our data (morad_scatch.ipynb)
* a notebook that builds a similar model but simplified to classify handwritten digits on the mnist dataset that achieves 99.43% accuracy (mnist_example.ipynb)
* the notebook we used to get the t-SNE embeddings on training and unlabelled test data to evaluate models (tsne_clean.ipynb)
* simplified code that can be used to get your own t-SNE embeddings on your own Keras models and plot them interactively using Bokeh if you desire (tsne_utils.py) 
* a notebook that uses tsne_utils.py and one of our models to get embeddings for signal modulation data on training data only (tsne_train_only.ipynb)
* the mnist model (mnist_model.h5)
* a notebook to do t-SNE on the mnist data and model (mnist_tsne.ipynb)

<br><img align="left" width="470" height="420" src="https://user-images.githubusercontent.com/22650698/44552995-f77f4b80-a6e0-11e8-9267-2e95bd400fbb.png">

Simple embedding of our small mnist model (no legend, no prediction probability). As we can see the data maps decently into 10 different clusters.
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

<br><img align="left" width="470" height="420" src="https://user-images.githubusercontent.com/22650698/44553335-f69ae980-a6e1-11e8-82d9-29231cc6791a.png">

Embedding showing the legend and the predicted probability for each class of points. The points over which we hover is predicted to be a 1 with probability 0.822.
