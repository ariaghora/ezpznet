import matplotlib.pyplot as plt

from ezpznet.animegan import AnimeGAN, load_image as animegan_load_image
from ezpznet.sketchgan import SketchGAN, load_image as sketchgan_load_image

img = sketchgan_load_image("/Users/ghora/Codes/ezpznet/large.jpg")
net = SketchGAN()
pred = net.predict(img)
plt.imshow(pred.squeeze(), cmap="gray")
plt.show()

net = AnimeGAN(style="hayao")
img = animegan_load_image("/Users/ghora/Downloads/winter.png")
result = net.predict(img)
plt.imshow(result)
plt.show()
