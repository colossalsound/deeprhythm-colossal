# DeepRhythm: High-Speed Tempo Prediction

DeepRhythm is a convolutional neural network designed for rapid, precise tempo prediction for modern music. It runs on anything that supports PyTorch.

Audio is batch-processed using a vectorized `Harmonic Constant-Q Modulation` (HCQM), drastically reducing computation time by avoiding the usual bottlenecks encountered in feature extraction.

This repo contains modifications/bugfixes/adaptations (etc.) made to the [original repo](https://github.com/bleugreen/deeprhythm) for use at Colossal.

## References

[1] Hadrien Foroughmand and Geoffroy Peeters, “Deep-Rhythm for Global Tempo Estimation in Music”, in Proceedings of the 20th International Society for Music Information Retrieval Conference, Delft, The Netherlands, Nov. 2019, pp. 636–643. doi: 10.5281/zenodo.3527890.

[2] K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.
