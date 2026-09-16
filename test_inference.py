from inference import get_generator
from PIL import Image

# Point this to any test image on your machine
img = Image.open("test_image.png")

generator = get_generator()
caption = generator.generate_caption(img)
print("Generated caption:", caption)