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

- SketchGAN, AnimeGAN, SRGAN

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

  net = AnimeGAN(style="facepaint")
  image = load_image(image_path)
  pred = net.predict(image)
  plt.imshow(pred)
  ```
  When `style="facepaint"`:

  <p align="center">
    <img src="https://i.ibb.co/c871Qjc/elon.png" width=600/>
  <p>

  When `style="hayao"`:

  <p align="center">
    <img src="https://i.ibb.co/wNjc8x6/hayao.png" width=500/>
  <p>


  There are 4 styles available: `facepaint` (default), `webtoon`, `shinkai`, `hayao`, and `paprika`.
  
  **References**
  - PyTorch implementation is adopted from [bryandlee](https://github.com/bryandlee/animegan2-pytorch)
  - Paper: Chen et al., AnimeGAN: A Novel Lightweight GAN for Photo Animation

---
    
### SRGAN
    
  GAN for super-resolution: upscale the resolution of an image and still keeping the detail, minimizing pixellated parts.
    
  ```python
  from ezpznet.srgan import SRGAN, load_image

  srgan = SRGAN()
  image = load_image(image_path)
  pred = srgan.predict(image)

  pred = ((pred + 1) / 2).squeeze().permute(1, 2, 0)
  pred = (pred * 255).numpy().astype(np.uint8)
  plt.imshow(pred)
  ```

  <p align="center">
    <img src="https://i.ibb.co/FkpkdBK/sr-comparison.png" width=700/>
  <p>
  
  **References**
  - PyTorch implementation is adopted from [dongheehand](https://github.com/dongheehand/SRGAN-PyTorch)
  - Paper: Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    
