Interesting and reproducible research works should be conserved.
This repository wraps a collection of deep neural network models into a simple and uniform API.

## Available models

### SketchGAN

### AnimeGAN

```python
from ezpznet.animegan import AnimeGAN, load_image

net = AnimeGAN(style="webtoon")
image = load_image(image_path)
pred = net.predict(image)
plt.imshow(pred)
```
When `style="webtoon"`:

<p align="center">
  <img src="https://i.ibb.co/c871Qjc/elon.png" width=500/>
<p>

When `style="hayao"`:

<p align="center">
  <img src="https://i.ibb.co/wNjc8x6/hayao.png" width=500/>
<p>
  

There are 4 styles available: `webtoon` (default), `shinkai`, `hayao`, and `paprika`.
