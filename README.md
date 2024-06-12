<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Malaria Cell Classification using CNN and Transfer Learning:</h1>

<h2>Overview</h2>
<p>This project aims to classify cell images as either infected with Malaria or uninfected using Convolutional Neural Networks (CNN) and Transfer Learning techniques. The project utilizes the dataset provided by the National Institutes of Health (NIH) comprising thousands of cell images.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#requirements">Requirements</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#training">Training</a></li>
  <li><a href="#evaluation">Evaluation</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
  <li><a href="#acknowledgements">Acknowledgements</a></li>
  <li><a href="#references">References</a></li>
</ul>
<h2 id="project-structure">Project Structure</h2>
<pre>
Malaria-Cell-Classification/
├── data/
│   ├── cell_images/
│   │   ├── Parasitized/
│   │   ├── Uninfected/
│   ├── train/
│   ├── val/
│   ├── test/
├── models/
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
├── requirements.txt
├── README.md
└── config.yaml
</pre>
<h2 id="dataset">Dataset</h2>
<p>The dataset used in this project can be downloaded from the official NIH dataset repository or Kaggle. It contains cell images categorized into two classes:</p>
<li><a href="https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data">Malaria Cell Classification Kaggle Dataset </a></li>

<ul>
  <li><code>Parasitized</code>: Cells infected with Malaria parasites.</li>
  <li><code>Uninfected</code>: Healthy cells without any infection.</li>
</ul>
<p>The dataset is split into training, validation, and testing sets.</p>

<h2 id="requirements">Requirements</h2>
<ul>
  <li>Python 3.8+</li>
  <li>TensorFlow 2.5+</li>
  <li>Keras</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>Scikit-learn</li>
  <li>Pandas</li>
</ul>

<h2 id="installation">Installation</h2>
<ol>
  <li>Clone this repository:
    <pre><code>git clone https://github.com/yourusername/Malaria-Cell-Classification.git
cd Malaria-Cell-Classification
</code></pre>
  </li>
  <li>Create and activate a virtual environment (optional but recommended):
    <pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
</code></pre>
  </li>
  <li>Install the required packages:
    <pre><code>pip install -r requirements.txt
</code></pre>
  </li>
</ol>

<h2 id="usage">Usage</h2>
<ol>
  <li>Prepare the data:
    <pre><code>python src/data_loader.py
</code></pre>
  </li>
  <li>Train the model:
    <pre><code>python src/train.py
</code></pre>
  </li>
  <li>Evaluate the model:
    <pre><code>python src/evaluate.py
</code></pre>
  </li>
</ol>

<h2 id="model-architecture">Model Architecture</h2>
<p>We utilize a CNN model architecture with Transfer Learning. The pre-trained models such as VGG16, ResNet50, or InceptionV3 from the ImageNet dataset are used as the base model, and additional custom layers are added for fine-tuning on the Malaria cell dataset.</p>

<h2 id="training">Training</h2>
<p>Training scripts and Jupyter notebooks are provided for step-by-step guidance on training the model. The key steps include:</p>
<ol>
  <li>Data preprocessing and augmentation.</li>
  <li>Loading the pre-trained model.</li>
  <li>Adding custom layers for classification.</li>
  <li>Compiling the model with appropriate loss functions and optimizers.</li>
  <li>Training the model with the training dataset and validating it with the validation set.</li>
</ol>

<h2 id="evaluation">Evaluation</h2>
<p>The evaluation script and notebooks provide detailed analysis of the model's performance on the test dataset. Key metrics include:</p>
<ul>
  <li>Accuracy</li>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1-score</li>
</ul>
<p>Additionally, confusion matrices and ROC curves are plotted for a comprehensive evaluation.</p>

<h2>Results</h2>
    
  <h3>Model Architecture</h3>
  <ul>
  <li><strong>Convolutional Neural Network (CNN)</strong>: A custom CNN model was designed and trained from scratch for the classification of malaria-infected and uninfected cells.</li>
  <li><strong>Transfer Learning</strong>: Pre-trained models such as VGG16, ResNet50, and InceptionV3 were fine-tuned on the malaria cell dataset to leverage learned features from large-scale datasets.    </li>
  </ul>
    
  <h3>Performance Metrics</h3>
  <ul>
  <li><strong>Accuracy</strong>: The accuracy achieved by the custom CNN model and transfer learning models were compared. Typically, transfer learning models outperformed the custom CNN in terms of accuracy due to their ability to generalize better with pre-learned features.</li>
  <li><strong>Precision, Recall, F1-Score</strong>: These metrics were used to evaluate the classification performance more comprehensively. Transfer learning models generally showed higher precision, recall, and F1-score.</li>
  </ul>
    
  <h3>Training and Validation Loss</h3>
  <ul>
    <li>The custom CNN model showed higher training and validation loss compared to the transfer learning models, indicating potential overfitting or underfitting.</li>
    <li>Transfer learning models had lower training and validation losses, suggesting better generalization.</li>
  </ul>
    
  <h3>Confusion Matrix</h3>
  <ul>
    <li>The confusion matrix indicated that transfer learning models had fewer false positives and false negatives compared to the custom CNN model.</li>
  </ul>
    
  <h2>Conclusion</h2>
    <h3>Effectiveness of Transfer Learning</h3>
    <ul>
      <li>Transfer learning significantly improved the performance of malaria cell classification. Pre-trained models like Mobilenet and VGG! achieved higher accuracy and better overall performance metrics compared to a custom CNN trained from scratch.</li>
    </ul>
    <h3>Generalization</h3>
    <ul>
      <li>Transfer learning models demonstrated superior generalization abilities. They were more effective in correctly identifying both infected and uninfected cells, as evidenced by higher precision, recall, and F1-scores.</li>
    </ul>
    <h3>Model Choice</h3>
    <ul>
      <li>Among the transfer learning models, ResNet50 and InceptionV3 were particularly effective, showing the highest accuracy and best performance in handling the classification task.</li>
    </ul>
    <h3>Recommendations</h3>
    <ul>
      <li>For practical applications, transfer learning with models like Mobilenet or VGG19 is recommended for malaria cell classification due to their robust performance and ability to generalize well across different datasets.</li>
      <li>Further improvements can be achieved by fine-tuning these models and possibly combining them with ensemble techniques to enhance classification performance.</li>
    </ul>

<h2 id="acknowledgements">Acknowledgements</h2>
<ul>
  <li>National Institutes of Health (NIH) for providing the dataset.</li>
  <li>TensorFlow and Keras for the deep learning frameworks.</li>
</ul>

<h2 id="references">References</h2>
<ul>
  <li><a href="https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip">Malaria Dataset by NIH</a></li>
  <li><a href="https://www.tensorflow.org/">TensorFlow Documentation</a></li>
  <li><a href="https://keras.io/">Keras Documentation</a></li>
</ul>

</body>
</html>
