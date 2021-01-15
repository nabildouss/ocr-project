try:
    from PIL import Image
except ImportError:
    import Image

from torch import nn
import pytesseract
import numpy

class Tesseract(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, x):
        imgs = x.cpu().numpy()
        imgs = numpy.transpose(imgs, (1, 0, 2, 3))[0] * 255
        y = [pytesseract.image_to_string(Image.fromarray(image, 'F').convert('RGB')) for image in imgs]
        return y