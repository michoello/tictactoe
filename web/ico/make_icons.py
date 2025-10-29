from PIL import Image, ImageDraw, ImageFont
import string
from colorsys import hsv_to_rgb
import random
import os

def make_icon(char: str, color: str, filename: str = "icon.ico", size: int = 256):
    """
    Generate an .ico file with a colored background and a negative character on top.

    Args:
        char (str): Single character to display (only first character is used).
        color (str): Background color (any CSS or hex color accepted by Pillow).
        filename (str): Output .ico file name.
        size (int): Image size (default 256x256).
    """
    if not char:
        raise ValueError("Character must not be empty.")
    char = char[0]

    # Create base image
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)

    letter_color = tuple(255 - c for c in color)

    # Choose font
    try:
        font_path = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"
        font = ImageFont.truetype(font_path, size)
    except IOError:
        font = ImageFont.load_default()

    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (size - text_w) / 2
    text_y = (size - text_h) / 2 - 35

    # Draw the character
    draw.text((text_x, text_y), char, fill=letter_color, font=font)

    # Save as .ico
    img.save(filename, format="ICO")
    print(f"Icon for {char} saved as {filename}")


def random_distinct_color(min_saturation=0.9, min_value=0.85):
    """
    Generate a random, visually distinct RGB color.
    Returns a tuple (R, G, B) with values 0-255.
    """
    h = random.random()  # Hue: 0.0-1.0
    s = random.uniform(min_saturation, 1.0)  # Saturation: avoid gray
    v = random.uniform(min_value, 1.0)       # Brightness/value

    r, g, b = hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def random_dark_color():
    return tuple(random.randint(50, 250) for _ in range(3))  # RGB 0–150


for c in string.ascii_uppercase:
   color = random_distinct_color()
   filename = f"icons/{c}.ico"
   make_icon(c, color, filename)
