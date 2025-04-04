# AI-Excel 数据处理与分析
![企业微信截图_17423706687874](https://github.com/user-attachments/assets/c839561e-2023-4e1f-b37f-445da545ab86)

在线测试
您可以通过以下链接在线体验本项目：
https://huggingface.co/spaces/viaho/pandas-ai-excel
---

## 简介

**AI-Excel 数据处理与分析** 是一个开源项目，旨在帮助用户通过自然语言轻松处理和分析 Excel 数据。该项目基于 **DeepSeek** 提供的免费接口(国内非官方，比如阿里云，华为等)，结合 **PandasAI** 库，支持对 Excel 文件（CSV 和 XLSX 格式）的高效操作。用户只需上传文件并提出问题，即可获得数据分析结果、图表或处理后的文件。

项目特别为中国用户设计，充分利用 **DeepSeek** 在中文环境下的自然语言处理能力，确保操作简单、结果准确。DeepSeek 的免费接口让用户无需额外成本即可享受高效的 Excel 数据处理体验。

---

## 功能特点

- **自然语言交互**：用中文提问即可操作 Excel 数据，例如“哪一行数据最高？”或“生成一个销量趋势图”。
- **支持多种格式**：兼容 `.csv` 和 `.xlsx` 文件，自动处理文件读取和转换。
- **免费高效**：DeepSeek 提供免费 API，结合 PandasAI，确保数据处理快速且无成本。
- **数据预览**：上传文件后，自动展示前 100 行数据，便于快速了解内容。
- **多样化输出**：支持文本结果、图表生成或导出处理后的 Excel 文件。
- **开源可扩展**：代码完全开源，欢迎用户根据需求进行定制开发。

---

## 安装和使用说明

### 1. 克隆项目

```bash
git clone https://github.com/via007/pandas-ai-excel.git
cd pandas-ai-excel
```

### 2. 安装依赖

项目需要 Python 3.10+，建议使用虚拟环境。

```bash
pip install -r requirements.txt
```

### 3. 配置 DeepSeek API

运行前需设置 DeepSeek 的 API 密钥，通过环境变量 `OPENAI_KEY` 加载。

```bash
export OPENAI_KEY="your-deepseek-api-key"
```

### 4. 启动项目

```bash
python app.py
```

启动后，默认在 `http://127.0.0.1:7860` 生成 Gradio 界面，可通过 `share=True` 创建公共链接。

---

## 示例

### 1. 上传文件
- 支持 `.csv` 或 `.xlsx` 文件。
- 上传后左侧显示前 100 行数据预览。

### 2. 输入问题
- 在右侧输入与数据相关的问题，例如：
  - “哪位学生的成绩最高？”
  - “按销量排序并导出”
  - “绘制月度收入柱状图”

### 3. 获取结果
- 根据问题返回：
  - **文本**：如分析结果。
  - **图表**：如柱状图或折线图。
  - **文件**：处理后的 Excel 文件可下载。

---

## 贡献指南

欢迎参与项目改进！您可以：
- 提交 Issue 反馈问题或建议功能。
- 提交 Pull Request 优化代码或文档。
- Star 仓库并分享给更多人。

提交 Pull Request 前，请确保代码风格一致并通过测试。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)，欢迎自由使用和修改。
