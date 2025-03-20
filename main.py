import os
import tempfile
import pandas as pd
import pandasai as pai
import matplotlib.pyplot as plt
from pandasai_openai import OpenAI

plt.rcParams['font.sans-serif'] = ['SimHei']

os.makedirs("./exports/temp", exist_ok=True)
api_token = os.getenv('OPENAI_KEY')


llm = OpenAI(api_token=api_token, api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
llm.model = 'deepseek-v3'  # qwen-max-0125 deepseek-v3
pai.config.set({"llm": llm})


def process_file_and_question(file_path, question):
    """
    处理上传的CSV文件和用户问题，返回文本、文件路径或图片路径。
    """
    try:
        if not file_path.endswith('.csv') and not file_path.endswith('.xlsx'):
            return "仅支持csv和xlsx文件", None, None

        if file_path.endswith('.csv'):
            df = pai.read_csv(file_path)
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir="./exports/temp")
            df_x = pd.read_excel(file_path)
            df_x.to_csv(temp_file.name, index=False)
            df = pai.read_csv(temp_file.name)

        response = df.chat(question)
        if response.error:
            return response.error, None, None
        if response.type == 'dataframe':
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', dir="./exports/temp")
            df = response.value
            df.to_excel(temp_file.name, index=False)
            return "数据处理完成，请下载结果", temp_file.name, None
        elif response.type == 'chart':
            image_path = response.value
            return "图片生成完成", None, image_path
        else:
            return response.value, None, None

    except:
        return f"系统异常，请重试...", None, None
