**Text Classification Using Autoencoders and Neural Networks**
*The project was done as part of the academic studies
Project Overview
This project focuses on binary text classification using autoencoders and deep neural networks. The task is to predict whether texts are labeled as 0 or 1, based on the text content. Data is provided in three sets: training, validation, and test.

Steps
Data Preprocessing:

Read the text files.
Vectorize the texts (using n-grams, TF-IDF, or BOW) without using transformer-based models.
Ensure all texts are in the same vector space.
Autoencoder:

Build and train an autoencoder to compress the vectorized texts.
Extract latent representations of the texts.
Classification:

Use a fully connected neural network with Keras.
The network should have multiple layers with ReLU activations and a sigmoid output.
Train the model using binary_crossentropy as the loss function and accuracy as the evaluation metric.
