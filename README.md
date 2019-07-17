## Preprocess images
## Information about model
The model consists of 5 CNN layers, 2 RNN (Bi-LSTM) layers and the CTC loss and decoding layer and can handle a full page of text image
* The input image is a gray-value image and has a size of 128x32
* 5 CNN layers map the input image to a feature sequence of size 32x256
* 2 LSTM layers with 256 units propagate information through the sequence and map the sequence to a matrix of size 32x148. Each matrix-element represents a score for one of the 148 characters at one of the 32 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with beam search decoding
* Batch size is set to 64
