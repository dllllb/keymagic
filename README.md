A neural net for SWYPE-like gestures recognition, trained on synthetic data.

# Approach

The task is similar to image captioning hence similar architecture can be used.

## Neural network architecture

Input: full gesture points sequience, repeated N times. Code of the underlaying letter can be added as an additional coordinate for each point of sequence. Output: character, one character per one repeat.

## Gestures CNN layers

Gesture points are time-ordered, hence it is reasonable to consider input for CNN layer as ordered array of 2D points: first filter gets points 1-10, second filter gets points 2-11 etc. 3D points could be used if underlaying letter code is added to gesture points.

# Related work

- Deep Visual-Semantic Alignments for Generating Image Descriptions / Andrej Karpathy, Li Fei-Fei / CVPR 2015
- Show, attend and tell: Neural image caption generation with visual attention / K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R.S. Zemel, Y. Bengio / ICML 2015
