import gradio as gr

from flair.data import Sentence
from flair.models import SequenceTagger

# Load the model
model = SequenceTagger.load("qanastek/pos-french")

def getPartOfSpeechFR(content):

    # George Washington est allé à Washington

    sentence = Sentence(content)

    # predict tags
    model.predict(sentence)

    # print predicted pos tags
    res = sentence.to_tagged_string()

    return res

iface = gr.Interface(
    title="🥖 French Part Of Speech Tagging",
    fn=getPartOfSpeechFR,
    inputs="textbox",
    outputs="textbox",
)
iface.launch()