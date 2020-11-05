# Target

- Smarthone virtual keyboard
- continuous finger motion to enter a word

# Related work

- Deep Visual-Semantic Alignments for Generating Image Descriptions / Andrej Karpathy, Li Fei-Fei / CVPR 2015
- Show, attend and tell: Neural image caption generation with visual attention / K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R.S. Zemel, Y. Bengio / ICML 2015

# Approach

- Deep learning tecniques for error-tolerant input
- the task is similar to image captioning hence similar architecture can be used

## Gestures generator

- it is possible to make generator of user's gestures for the samples of text from the internet and than use its outut to train a model of any complexity

### GAN as gestures generator

- gestures collection application would help to get the real user gestures dataset
- generator net produces a curve for the input word
- discriminator net gets word and corresponding curve as an input and should distinguish real user input curves from curves produced by the generator net

## Character-level RNN (LSTM) for phrase prediction

- predicts phrase character by character
- takes previous words to account
- can be trained on any text data e. g. web dumps
- can predict quotes, commas, semicolons, dots etc.
- see also [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Neural network architecture

- input: full gesture points sequience, repeated N times
  - code of the underlaying letter can be added as an additional coordinate for each point of sequence
- output: letters, one letter per one repeat

## Gestures CNN layers

- gesture points are time-ordered, hence it is reasonable to consider input for CNN layer as ordered array of 2D points
  - first filter gets points 1-10, second filter gets points 2-11 etc.
  - 3D points could be used if underlaying letter code is added to gesture points
