import glob
import io
import os

import click

from google.cloud import vision
from google.cloud.vision import types


@click.command()
@click.option('--directory', type=click.Path(exists=True),  prompt=True, help='screenshot directory path.')
def main(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    files_path = glob.glob(directory)
    if files_path is None:
        raise IndexError()
    client = vision.ImageAnnotatorClient()

    for file_path in files_path:
        with io.open(file_path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        response = client.label_detection(image=image)
        labels = response.label_annotations

        print('Labels:')
        for label in labels:
            print(label.description)


if __name__ == "__main__":
    main()
