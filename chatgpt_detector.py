!pip install gradio transformers

import os
import gradio as gr
from transformers import pipeline

auth_token = os.environ.get("access_token")
pipeline_en = pipeline(task="text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta",use_auth_token=auth_token)
#pipeline_zh = pipeline(task="text-classification", model="Hello-SimpleAI/chatgpt-detector-single-chinese",use_auth_token=auth_token)



def predict_en(text):
    res = pipeline_en(text)[0]
    return res['label'],res['score']

def predict_zh(text):
    res = pipeline_zh(text)[0]
    return res['label'],res['score']




with gr.Blocks() as demo:
    gr.Markdown("""
                ## ChatGPT Detector 
                detect whether a piece of text is ChatGPT generated, using PLM-based classifiers ;
                
                
                """)
    with gr.Tab("English"):
        gr.Markdown("""
                    Note: Providing more text to the `Text` box can make the prediction more accurate!
                    """)
        t1 = gr.Textbox(lines=5, label='Text',value="Quantum computing is a method of performing calculations using quantum-mechanical phenomena, such as superposition and entanglement. In a classical computer, information is stored in bits, which can have one of two values, 0 or 1. In a quantum computer, the basic unit of information is a quantum bit, or qubit, which can exist in multiple states simultaneously. This property allows quantum computers to perform certain types of calculations much faster than classical computers. However, quantum computers are not yet able to perform all types of calculations faster than classical computers, and they are also more difficult to build and control.")
        button1 = gr.Button("🤖 Predict!")
        label1 = gr.Textbox(lines=1, label='Predicted Label 🎃')
        score1 = gr.Textbox(lines=1, label='Prob')

    button1.click(predict_en, inputs=[t1], outputs=[label1,score1])

    # Page Count
    gr.Markdown("""
                <center><a href='https://clustrmaps.com/site/1bsdc'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=080808&w=a&t=tt&d=NXQdnwxvIm27veMbB5F7oHNID09nhSvkBRZ_Aji9eIA&co=ffffff&ct=808080'/></a></center>
                """)

demo.launch()
