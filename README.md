Interesting and reproducible research works should be conserved.
Unfortunately, too many model repositories provide different ways to use.
It is an obstacle for people who just want to use them right away, especially for those without luxury to (re)train big deep learning models.
This repository aims to wrap a collection of deep neural network models into a simple and consistent API.

## Installation

```
pip install git+https://github.com/ariaghora/ezpznet
```
It depends mainly on pytorch and torchvision. 

## Pretrained weights
Each model will download its own pretrained weight (once at the first time) at initialization.
I host them in my personal OneDrive storage. 
Let me know if you have better options.

## Available models

### SketchGAN

Simplify rough outline sketch.

```python
from ezpznet.sketchgan import SketchGAN, load_image

image = load_image(image_path)
net = SketchGAN()
pred = net.predict(image)
plt.imshow(pred.squeeze(), cmap="gray")
```

<p align="center">
  <img src="https://i.ibb.co/QM8Z9Tb/shinji.png" width=600/>
<p>

(Art by [Shinji](https://artbyshinji.tumblr.com))

  **References**
  - PyTorch implementation is adopted from [bobbens](https://github.com/bobbens/sketch_simplification/)
  - Paper: Simo-Serra et al., Mastering Sketching: Adversarial Augmentation for Structured Prediction

---
  
  
### AnimeGAN

  Apply anime-ish effect to images.

  ```python
  from ezpznet.animegan import AnimeGAN, load_image

  net = AnimeGAN(style="webtoon")
  image = load_image(image_path)
  pred = net.predict(image)
  plt.imshow(pred)
  ```
  When `style="webtoon"`:

  <p align="center">
    <img src="https://i.ibb.co/c871Qjc/elon.png" width=600/>
  <p>

  When `style="hayao"`:

  <p align="center">
    <img src="https://i.ibb.co/wNjc8x6/hayao.png" width=500/>
  <p>


  There are 4 styles available: `webtoon` (default), `shinkai`, `hayao`, and `paprika`.
  
  **References**
  - PyTorch implementation is adopted from [bryandlee](https://github.com/bryandlee/animegan2-pytorch)
  - Paper: Chen et al., AnimeGAN: A Novel Lightweight GAN for Photo Animation
  
