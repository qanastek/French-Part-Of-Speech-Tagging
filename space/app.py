import time
import streamlit as st
from annotated_text import annotated_text

from flair.data import Sentence
from flair.models import SequenceTagger

checkpoints = [
    "qanastek/pos-french",
]

colors = {'DET': '#b9d9a6', 'NFP': '#eddc92', 'ADJFP': '#95e9d7', 'AUX': '#e797db', 'VPPMS': '#9ff48b', 'ADV': '#ed92b4', 'PREP': '#decfa1', 'PDEMMS': '#ada7d7', 'NMS': '#85fad8', 'COSUB': '#8ba4f4', 'PINDMS': '#e7a498', 'PPOBJMS': '#e5c79a', 'VERB': '#eb94b6', 'DETFS': '#e698ae', 'NFS': '#d9d1a6', 'YPFOR': '#96e89f', 'VPPFS': '#e698c6', 'PUNCT': '#ddbfa2', 'DETMS': '#f788cd', 'PROPN': '#f19c8d', 'ADJMS': '#8ed5f0', 'PPER3FS': '#c4d8a6', 'ADJFS': '#e39bdc', 'COCO': '#8df1e2', 'NMP': '#d7f787', 'PREL': '#f986f0', 'PPER1S': '#878df8', 'ADJMP': '#83fe80', 'VPPMP': '#a6d8c9', 'DINTMS': '#d9a6cc', 'PPER3MS': '#a1deda', 'PPER3MP': '#8fefe1', 'PREF': '#e3c79b', 'ADJ': '#fb81fe', 'DINTFS': '#d5fe81', 'CHIF': '#8084ff', 'XFAMIL': '#dd80fe', 'PRELFS': '#9ce3e3', 'SYM': '#9fbddf', 'NOUN': '#dea1b5', 'MOTINC': '#93b8ec', 'PINDFS': '#f787a5', 'PPOBJMP': '#dca3d2', 'NUM': '#b2e897', 'PREFP': '#e39cd0', 'PDEMFS': '#d8a7cb', 'VPPFP': '#83d9fb', 'PPER3FP': '#a1ddaa', 'PPOBJFS': '#e9ca95', 'PINDMP': '#e897e3', 'PRON': '#e29dcc', 'PPOBJFP': '#86f9dc', 'PART': '#aa96e8', 'PDEMMP': '#b2d7a8', 'PRELMS': '#e39bde', 'PDEMFP': '#b1e599', 'PRELFP': '#bbe39b', 'INTJ': '#bde996', 'PREFS': '#b39be4', 'PINDFP': '#e2e897', 'PRELMP': '#a5c0da', 'PINTFS': '#ceff80', 'PPER2S': '#d5a2dd', 'VPPRE': '#e78af4', '<START>': '#e6a899', '<STOP>': '#9adde5'}

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_model(model_name):
    return SequenceTagger.load(model_name) # Load the model

def getPos(s: Sentence):
    texts = []
    labels = []
    for t in s.tokens:
        for label in t.annotation_layers.keys():
            texts.append(t.text)
            labels.append(t.get_labels(label)[0].value)          
    return texts, labels

def getDictFromPOS(texts, labels):
    return [{ "text": t, "label": l } for t, l in zip(texts, labels)]

def getAnnotatedFromPOS(texts, labels):
    return [(t,l,colors[l]) for t, l in zip(texts, labels)]

def main():

    st.title("🥖 French Part-Of-Speech Tagging")

    checkpoint = st.selectbox("Choose model", checkpoints)
    model = get_model(checkpoint)
    
    default_text = "George Washington est allé à Washington"
    input_text = st.text_area(
        label="Original text",
        value=default_text,
    )

    start = None
    if st.button("🧠 Compute"):
        start = time.time()
        with st.spinner("Search for Part-Of-Speech Tags 🔍"):
            
            # Build Sentence
            s = Sentence(input_text)

            # predict tags
            model.predict(s)

            try:

                texts, labels = getPos(s)
                
                st.header("Labels:")
                anns = getAnnotatedFromPOS(texts, labels)
                annotated_text(*anns)

                st.header("JSON:")
                st.json(getDictFromPOS(texts, labels))

            except Exception as e:
                st.error("Some error occured!" + str(e))
                st.stop()

    st.write("---")

    st.markdown(
        "Built by [Yanis Labrak](https://www.linkedin.com/in/yanis-labrak-8a7412145/) 🚀"
    )
    st.markdown(
        "_Source code made with [FlairNLP](https://github.com/flairNLP/flair)_"
    )

    if start is not None:
        st.text(f"prediction took {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
