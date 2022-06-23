A neural net for SWYPE-like gestures recognition, trained on synthetic data.

Synthetic data generation [demo notebook](strokes.ipynb)
Simple model train [demo notebook](cnn-train.ipynb)
Simple model test [demo notebook](cnn-test.ipynb)

# Neural network architecture

CNN encder + RNN decoder. Input: gesture points sequience, output: characters, one character per RNN output repeat. Underlaying letter code is added to each gesture point as its 3-rd component.

Gesture points are time-ordered, hence it is reasonable to consider input for CNN layer as ordered array of 2D points: first filter gets points 1-10, second filter gets points 2-11 etc.

# Related papers

- Deep Visual-Semantic Alignments for Generating Image Descriptions / Andrej Karpathy, Li Fei-Fei / CVPR 2015
- Show, attend and tell: Neural image caption generation with visual attention / K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R.S. Zemel, Y. Bengio / ICML 2015
- How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers / Kseniia Palin, Anna Maria Feit, Sunjun Kim, Per Ola Kristensson, Antti Oulasvirta / MobileHCI 2019 / https://userinterfaces.aalto.fi/typing37k/
