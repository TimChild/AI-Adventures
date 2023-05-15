import requests
import shutil
from IPython.display import display, Image
import openai
import os


# TODO: remove this from here
import openai

with open(r'D:\GitHub\ai_adventures\API_KEY', "r") as f:
    key = f.read()

openai.api_key = key


class ImageGenerator:
    def __init__(self):
        self._last_response = None
        pass

    def generate_image(self, prompt, save_path=None):
        """

        Args:
            prompt ():
            save_path ():

        Note:
            SIZES = ["256x256", "512x512", "1024x1024"]
            PRICES = [0.016, 0.018, 0.02]

        Returns:
            url of generated image

        """
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size='1024x1024',
        )
        self._last_response = response

        image_url = response['data'][0]['url']

        if save_path:
            self.save_image_from_url(image_url, save_path)

        return image_url

    def save_image_from_url(self, image_url, save_path):
        response = requests.get(image_url, stream=True)
        with open(save_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)

    def display_image(self, image_url):
        display(Image(url=image_url))

    def display_image_from_file(self, filepath):
        display(Image(filename=filepath))

    def generate_and_save_image(self, prompt, save_path, show=True):
        image_url = self.generate_image(prompt, save_path)
        self.save_image_from_url(image_url, save_path)
        if show:
            self.display_image_from_file(save_path)
