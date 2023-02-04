"""
This is a script that will generate prompts for an image generator model.
then it will generate the images and save them to a folder.
"""
import os
import re
import openai
import requests
import random
from langchain import PromptTemplate
from langchain.llms import OpenAI

NUM_DESCRIPTIONS = 100
NUM_STYLES = 100
NUM_IMAGES = 25
NUM_IMAGES_PER_PROMPT = 1
# Set the API key
with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
    key = f.read().strip()
    openai.api_key = key
    os.environ['OPENAI_API_KEY'] = key

IMAGE_TEMPLATE = """
You are a large language model that is going to be generating a list of descriptions for beautiful images, 
these descriptions will be used to generate images using the DALL-E model. The descriptions will be
long and describe the image in detail, and they will describe somthing that humans will find beautiful. 
The description will include a lot of physical details about everything that is present in the image
and it will describe the foreground, background and everything that is visable in the image.

You will format your responses as a list separated by linebreaks, and won't include numbers. 
Your response will have one image description per line. for example:

description
description
description

LIST OF {num} IMAGE DESCRIPTIONS:
"""
STYLE_TEMPLATE = """
You are a large language model that is going to be generating a list of descriptions for beautiful artistic styles, 
these descriptions will be used to generate images using the DALL-E model. The descriptions will be
long and describe the style in detail, and they will describe somthing that humans will find beautiful. 
The description will include a lot of details about a particular style of art including different styles of painting,
photography, sculpture, glass art, illustration, and digital art. The description should use words that describe
all of the visual aspects of the style including specific colors, specific techniques, visual details, emotions, 
and specific details about the composition.

You will format your responses as a list separated by linebreaks, and won't include numbers. 
Your response will have one image description per line. For Example:

description
description
description

LIST OF {num} STYLE DESCRIPTIONS:
"""


def init_llm():
    """Initialize the LLM."""
    llm = OpenAI(temperature=0.9,
                 max_tokens=1024,
                 top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0.6)
    return llm


def generate_list(llm, num, template):
    """Generate a list of image prompts that will be used to generate images."""
    prompt = PromptTemplate(input_variables=['num'],
                            template=template)
    response = llm(prompt.format(num=num))
    response_list = response.splitlines()
    # remove empty strings from list
    response_list = [x for x in response_list if x]
    return response_list


def save_image(image, filename):
    """Save the image to a file."""
    with open(filename, "wb") as f:
        f.write(image)


def main():
    llm = init_llm()
    description_list = generate_list(llm, NUM_DESCRIPTIONS, IMAGE_TEMPLATE)
    style_list = generate_list(llm, NUM_STYLES, STYLE_TEMPLATE)
    print(description_list, style_list)
    num_images = NUM_IMAGES
    images_per_prompt = NUM_IMAGES_PER_PROMPT
    for i in range(num_images):
        description = random.choice(description_list)
        style = random.choice(style_list)
        prompt = f"{description} {style}"
        print(prompt)
        prompt = re.sub(r'[^a-zA-Z ]', '', prompt)
        image_dict = openai.Image.create(prompt=prompt,
                                         n=images_per_prompt,
                                         size="1024x1024")
        for j in range(images_per_prompt):
            url = image_dict['data'][j]['url']
            image = requests.get(url).content
            pathname = f"/Users/alexthe5th/Pictures/AI Art/{j}_{prompt}"[:250]
            filename = f"{pathname}.png"
            save_image(image, filename)


if __name__ == '__main__':
    main()
