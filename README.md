# AI-Excel 数据处理与分析
![企业微信截图_17423706687874](https://github.com/user-attachments/assets/c839561e-2023-4e1f-b37f-445da545ab86)

在线测试
您可以通过以下链接在线体验本项目：
https://huggingface.co/spaces/viaho/pandas-ai-excel
---

## 简介

**AI-Excel 数据处理与分析** 是一个开源项目，旨在帮助用户通过自然语言轻松处理和分析 Excel 数据。项目基于 **PandasAI**，通过 OpenAI 兼容接口连接主流 LLM（默认 DeepSeek），支持对 Excel 文件（CSV 和 XLSX 格式）的高效操作。用户只需上传文件并提出问题，即可获得数据分析结果、图表或处理后的文件。

项目特别为中文场景优化，提供多文件多表管理、关系视图（Join）与数据概览报告等能力，兼顾易用性与可扩展性。

---

## 功能特点

- **自然语言交互**：用中文提问即可操作 Excel 数据，例如“哪一行数据最高？”或“生成一个销量趋势图”。
- **多文件多表**：支持同时上传多个 CSV/XLSX，Excel 可读取全部 Sheet，并统一管理。
- **数据预览**：上传后展示前 100 行预览，便于快速了解数据。
- **数据概览**：自动生成字段类型、缺失率、唯一值、示例值等概览报告，可导出。
- **关系视图**：可选择两张表进行 Join，预览结果并保存为新数据集。
- **多样化输出**：文本、图表、导出结果（xlsx/csv/json/png）。
- **灵活 LLM 配置**：内置配置页支持 DeepSeek/OpenAI/阿里云/Gemini/自定义兼容接口。

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

### 3. 配置 LLM（可选）

你可以在页面“配置”页填写 API Key，也可以用环境变量作为默认配置。

**DeepSeek（默认）**
```bash
export LLM_PROVIDER="DeepSeek"
export OPENAI_KEY="your-deepseek-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-chat"
```

**Gemini（OpenAI 兼容）**
```bash
export LLM_PROVIDER="Gemini(OpenAI兼容)"
export OPENAI_KEY="your-gemini-api-key"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export OPENAI_MODEL="gemini-1.5-flash"
```

### 4. 启动项目

```bash
python app.py
```

启动后，默认在 `http://127.0.0.1:7860` 生成 Gradio 界面，可通过 `share=True` 创建公共链接。

---

## 示例

### 1. 上传文件
- 支持 `.csv` 或 `.xlsx` 文件，多文件上传。
- Excel 可选择读取所有 Sheet。
- 上传后显示前 100 行数据预览。

### 2. 输入问题
- 在右侧输入与数据相关的问题，例如：
  - “哪位学生的成绩最高？”
  - “按销量排序并导出”
  - “绘制月度收入柱状图”

### 3. 获取结果
- 根据问题返回：
  - **文本**：如分析结果。
  - **图表**：如柱状图或折线图。
  - **文件**：处理后的结果文件可下载。

### 4. 数据概览与关系视图
- “数据概览”页可生成字段概览并导出。
- “关系与视图”页支持双表 Join，预览结果并保存为新数据集。

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
