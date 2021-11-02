Interesting and reproducible research works should be conserved.
This repository wraps a collection of deep neural network models into a simple and uniform API.

## Available models

- SketchGAN

- AnimeGAN

```python
from ezpznet.animegan import AnimeGAN, load_image

net = AnimeGAN(style="webtoon")
image = load_image(image_path)
pred = net.predict(image)
plt.imshow(pred)
```
