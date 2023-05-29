import os

import gradio as gr
import google.generativeai as palm

"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""

global global_palm
global_palm = None


def get_palm(api_key):
    global global_palm
    if global_palm != None:
        return global_palm
    palm.configure(api_key=api_key)
    global_palm = palm
    return palm


def get_sample_result(prompt):
    defaults = {
        "model": "models/text-bison-001",
        "temperature": 0.7,
        "candidate_count": 1,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "stop_sequences": [],
        "safety_settings": [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1},
            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2},
        ],
    }
    api_key = os.getenv("PALM_API_KEY")
    palm = get_palm(api_key)
    print(api_key)
    response = palm.generate_text(**defaults, prompt=prompt)
    print(response.result)
    return response.result


def sample_prompt(prompt):
    return get_sample_result(prompt=prompt)


with gr.Blocks() as ui:
    with gr.Tab("Sample prompts"):
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=4, interactive=True)
            ai = gr.Button("AI")
            gr.Markdown(value="## Result")
            result = gr.Markdown(value="")

        ai.click(fn=sample_prompt, inputs=[prompt], outputs=[result])


if __name__ == "__main__":
    ui.launch(debug=True)
