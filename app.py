import pandas as pd
import gradio as gr
from main import process_file_and_question

pal = '''
问题输入要贴合excel表述，和excel无关的话题会出现异常！
'''

with gr.Blocks(title="AI-Excel数据处理与分析", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        """
        <div style="background-color: #e0f7fa; padding: 20px; border-radius: 8px; text-align: center;">
            <strong style="font-size: 18px;">🤖DeepSeek+Excel数据处理</strong>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="上传excel文件（仅支持csv和xlsx格式）", file_types=[".csv", ".xlsx"])
            data_display = gr.Dataframe(label="数据预览（前100行）")
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="输入您的问题", lines=9, placeholder=pal)
            submit_button = gr.Button("提交", variant="primary")
            output_text = gr.Textbox(label="文本输出")
            output_file = gr.File(label="下载文件")
            output_image = gr.Image(label="图片输出")

    def on_file_upload(file):
        if file is None:
            return None
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name, nrows=100)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name, nrows=100)
            else:
                return None
            return df
        except Exception:
            return None

    file_upload.change(on_file_upload, inputs=file_upload, outputs=data_display)

    def on_submit(file, question):
        if file is None or not question:
            return "请上传excel文件并输入问题", None, None
        text, file_path, image_path = process_file_and_question(file.name, question)
        return text, file_path, image_path


    file_upload.change(on_file_upload, inputs=file_upload, outputs=data_display)
    submit_button.click(on_submit, inputs=[file_upload, question_input],
                        outputs=[output_text, output_file, output_image])

demo.launch()
