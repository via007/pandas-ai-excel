import os

import gradio as gr
import pandas as pd

from main import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    build_join_dataset,
    build_join_preview,
    build_profile_report,
    configure_llm,
    export_dataframe,
    get_dataset_columns,
    load_datasets,
    process_question,
    suggest_join_columns,
)

PLACEHOLDER = """
问题输入要贴合Excel表述，和Excel无关的话题会出现异常！
支持多文件、多表：可先上传多个CSV/XLSX，再选择需要分析的数据集。
"""

OUTPUT_HINTS = {
    "自动": "",
    "表格": "请以表格形式输出结果。",
    "图表": "请生成图表输出结果。",
    "文本": "请用简洁文本总结输出结果。",
}

CHART_HINTS = {
    "折线图": "请生成折线图。",
    "柱状图": "请生成柱状图。",
    "饼图": "请生成饼图。",
    "散点图": "请生成散点图。",
    "直方图": "请生成直方图。",
}

PROVIDERS = {
    "OpenAI": {
        "api_base": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "阿里云 DashScope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-max-0125", "qwen-plus", "qwen-turbo"],
    },
    "Gemini(OpenAI兼容)": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
    },
    "自定义": {
        "api_base": "",
        "models": [],
    },
}

DEFAULT_PROVIDER = DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else "自定义"
DEFAULT_API_KEY = DEFAULT_API_KEY

_provider_defaults = PROVIDERS.get(DEFAULT_PROVIDER, {})
if not DEFAULT_API_BASE:
    DEFAULT_API_BASE = _provider_defaults.get("api_base", "")
if not DEFAULT_MODEL:
    default_models = _provider_defaults.get("models", [])
    DEFAULT_MODEL = default_models[0] if default_models else ""


def _append_hint(question: str, hint: str) -> str:
    question = (question or "").strip()
    if not question:
        return hint
    return f"{question} {hint}"


def _dataset_updates(dataset_names):
    return gr.update(choices=dataset_names, value=dataset_names)


def _single_dataset_update(dataset_names):
    return gr.update(choices=dataset_names, value=dataset_names[0] if dataset_names else None)


def on_provider_change(provider: str):
    info = PROVIDERS.get(provider, {})
    api_base = info.get("api_base", "")
    models = info.get("models", [])
    model_value = models[0] if models else ""
    return gr.update(value=api_base), gr.update(choices=models, value=model_value)


def on_apply_config(provider: str, api_base: str, model: str, api_key: str):
    config = {
        "provider": provider,
        "api_base": api_base,
        "model": model,
        "api_key": api_key,
    }

    if provider == "自定义" and not api_base:
        return config, "自定义模式需要填写 Base URL"

    ok, message = configure_llm(api_key, api_base, model)
    return config, message


def on_files_upload(files, load_all_sheets):
    if not files:
        empty_update = gr.update(choices=[], value=[])
        empty_single = gr.update(choices=[], value=None)
        empty_columns = gr.update(choices=[], value=None)
        return {}, None, empty_update, empty_single, None, "请上传文件", empty_single, empty_single, empty_single, empty_columns, empty_columns

    file_paths = [f.name for f in files]
    datasets, summary_df, errors = load_datasets(file_paths, load_all_sheets)
    dataset_names = list(datasets.keys())

    preview_name = dataset_names[0] if dataset_names else None
    preview_df = datasets[preview_name].head(100) if preview_name else None
    left_name = dataset_names[0] if dataset_names else None
    right_name = dataset_names[1] if len(dataset_names) > 1 else left_name
    left_columns = get_dataset_columns(datasets, left_name) if left_name else []
    right_columns = get_dataset_columns(datasets, right_name) if right_name else []

    status_parts = [f"已加载 {len(dataset_names)} 个数据集"]
    if errors:
        status_parts.append(" | ".join(errors))

    return (
        datasets,
        summary_df,
        _dataset_updates(dataset_names),
        gr.update(choices=dataset_names, value=preview_name),
        preview_df,
        "；".join(status_parts),
        _single_dataset_update(dataset_names),
        _single_dataset_update(dataset_names),
        _single_dataset_update(dataset_names),
        gr.update(choices=left_columns, value=left_columns[0] if left_columns else None),
        gr.update(choices=right_columns, value=right_columns[0] if right_columns else None),
    )


def on_preview_change(preview_name, datasets):
    if not preview_name or preview_name not in datasets:
        return None
    return datasets[preview_name].head(100)


def on_profile_generate(dataset_name, datasets):
    overview, profile_df = build_profile_report(datasets, dataset_name)
    if overview is None:
        return "请先选择数据集", None

    overview_text = (
        f"数据集：{dataset_name}  |  行数：{overview['rows']}  |  列数：{overview['cols']}  |  "
        f"重复行：{overview['duplicates']}  |  内存：{overview['memory_mb']} MB"
    )
    return overview_text, profile_df


def on_profile_export(profile_df, export_formats):
    if profile_df is None or export_formats is None:
        return [], "请先生成数据概览"
    if not export_formats:
        return [], "请选择导出格式"
    if isinstance(profile_df, list):
        profile_df = pd.DataFrame(profile_df)
    if isinstance(profile_df, dict):
        profile_df = pd.DataFrame(profile_df)
    if profile_df is None or profile_df.empty:
        return [], "没有可导出的概览数据"
    files = export_dataframe(profile_df, export_formats)
    return files, "已生成报告文件"


def on_dataset_change(dataset_name, datasets):
    cols = get_dataset_columns(datasets, dataset_name)
    return gr.update(choices=cols, value=cols[0] if cols else None)


def on_auto_match(left_name, right_name, datasets):
    left_cols = get_dataset_columns(datasets, left_name)
    right_cols = get_dataset_columns(datasets, right_name)
    left_col, right_col = suggest_join_columns(left_cols, right_cols)
    if not left_col or not right_col:
        return None, None, "未找到可自动匹配字段"
    return left_col, right_col, f"已匹配字段：{left_col} = {right_col}"


def on_preview_join(datasets, left_name, right_name, left_col, right_col, join_type, preview_rows):
    preview_df, stats, join_config = build_join_preview(
        datasets,
        left_name,
        right_name,
        left_col,
        right_col,
        join_type,
        preview_rows,
    )
    return preview_df, stats, join_config


def on_save_join(datasets, join_config, new_name):
    new_datasets, final_name, error = build_join_dataset(datasets, join_config, new_name)
    if error:
        return (
            datasets,
            gr.update(choices=list(datasets.keys()), value=list(datasets.keys())),
            gr.update(choices=list(datasets.keys()), value=list(datasets.keys())[0] if datasets else None),
            gr.update(choices=list(datasets.keys()), value=list(datasets.keys())[0] if datasets else None),
            gr.update(choices=list(datasets.keys()), value=list(datasets.keys())[0] if datasets else None),
            gr.update(choices=list(datasets.keys()), value=list(datasets.keys())[0] if datasets else None),
            error,
        )

    dataset_names = list(new_datasets.keys())
    return (
        new_datasets,
        gr.update(choices=dataset_names, value=dataset_names),
        gr.update(choices=dataset_names, value=dataset_names[0] if dataset_names else None),
        gr.update(choices=dataset_names, value=dataset_names[0] if dataset_names else None),
        gr.update(choices=dataset_names, value=dataset_names[0] if dataset_names else None),
        gr.update(choices=dataset_names, value=dataset_names[0] if dataset_names else None),
        f"已保存为新数据集：{final_name}",
    )


def on_submit(datasets, selected_names, question, output_mode, export_formats, llm_config):
    if not datasets:
        return "请先上传数据文件", [], None
    if not selected_names:
        return "请先选择要分析的数据集", [], None
    if not question or not question.strip():
        return "请输入问题", [], None

    hint = OUTPUT_HINTS.get(output_mode, "")
    final_question = _append_hint(question, hint) if hint else question

    return process_question(datasets, selected_names, final_question, export_formats or [], llm_config)


with gr.Blocks(title="AI-Excel数据处理与分析", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        """
        <div style="background-color: #e0f7fa; padding: 20px; border-radius: 8px; text-align: center;">
            <strong style="font-size: 18px;">🤖 DeepSeek + Excel 数据处理</strong>
        </div>
        """
    )

    datasets_state = gr.State({})
    config_state = gr.State({
        "provider": DEFAULT_PROVIDER,
        "api_base": DEFAULT_API_BASE,
        "model": DEFAULT_MODEL,
        "api_key": DEFAULT_API_KEY,
    })
    join_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("数据分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.Files(label="上传文件（支持csv/xlsx，多文件）", file_types=[".csv", ".xlsx"])
                    load_all_sheets = gr.Checkbox(label="Excel读取所有Sheet", value=True)
                    upload_status = gr.Markdown()
                    dataset_summary = gr.Dataframe(label="数据集列表", interactive=False)
                    preview_dataset = gr.Dropdown(label="预览数据集", choices=[])
                    data_display = gr.Dataframe(label="数据预览（前100行）", interactive=False)

                with gr.Column(scale=1):
                    active_datasets = gr.Dropdown(label="选择用于分析的数据集", multiselect=True, choices=[])
                    output_mode = gr.Dropdown(label="输出偏好", choices=["自动", "表格", "图表", "文本"], value="自动")
                    export_formats = gr.CheckboxGroup(
                        label="导出格式（表格: xlsx/csv/json，图表: png）",
                        choices=["xlsx", "csv", "json", "png"],
                        value=["xlsx", "csv"],
                    )
                    question_input = gr.Textbox(label="输入您的问题", lines=8, placeholder=PLACEHOLDER)

                    with gr.Row():
                        btn_line = gr.Button("折线图")
                        btn_bar = gr.Button("柱状图")
                        btn_pie = gr.Button("饼图")
                        btn_scatter = gr.Button("散点图")
                        btn_hist = gr.Button("直方图")

                    submit_button = gr.Button("提交", variant="primary")
                    output_text = gr.Textbox(label="文本输出")
                    output_files = gr.Files(label="下载文件")
                    output_image = gr.Image(label="图片输出")

        with gr.Tab("数据概览"):
            gr.Markdown("选择数据集并生成概览报告。")
            profile_dataset = gr.Dropdown(label="选择数据集", choices=[])
            generate_profile = gr.Button("生成概览", variant="primary")
            profile_overview = gr.Markdown()
            profile_table = gr.Dataframe(label="字段概览", interactive=False, type="pandas")

            profile_export_formats = gr.CheckboxGroup(
                label="导出概览报告",
                choices=["xlsx", "csv", "json"],
                value=["xlsx"],
            )
            export_profile = gr.Button("导出报告")
            profile_files = gr.Files(label="报告文件")
            profile_status = gr.Markdown()

        with gr.Tab("关系与视图"):
            gr.Markdown("选择两个数据集并配置连接字段，可预览连接效果并保存为新数据集。")
            with gr.Row():
                left_dataset = gr.Dropdown(label="左表", choices=[])
                right_dataset = gr.Dropdown(label="右表", choices=[])

            with gr.Row():
                left_column = gr.Dropdown(label="左表字段", choices=[])
                right_column = gr.Dropdown(label="右表字段", choices=[])

            auto_match = gr.Button("自动匹配字段")
            auto_match_status = gr.Markdown()

            with gr.Row():
                join_type = gr.Dropdown(label="连接方式", choices=["inner", "left", "right", "outer"], value="inner")
                preview_rows = gr.Slider(label="预览行数", minimum=10, maximum=200, value=50, step=10)

            preview_join = gr.Button("预览连接", variant="primary")
            relation_stats = gr.Markdown()
            join_preview = gr.Dataframe(label="连接预览", interactive=False)

            new_dataset_name = gr.Textbox(label="保存为新数据集名称")
            save_join = gr.Button("保存为新数据集")
            save_status = gr.Markdown()

        with gr.Tab("配置"):
            gr.Markdown("填写 API Key 并选择模型后保存配置。默认使用 OpenAI 兼容接口。")
            provider = gr.Dropdown(
                label="LLM 提供方",
                choices=list(PROVIDERS.keys()),
                value=DEFAULT_PROVIDER,
            )
            api_base = gr.Textbox(label="Base URL", value=DEFAULT_API_BASE)
            model = gr.Dropdown(
                label="模型",
                choices=PROVIDERS[DEFAULT_PROVIDER]["models"],
                value=DEFAULT_MODEL,
                allow_custom_value=True,
            )
            api_key = gr.Textbox(label="API Key", type="password", value=DEFAULT_API_KEY)
            save_config = gr.Button("保存配置", variant="primary")
            config_status = gr.Markdown()

    file_upload.change(
        on_files_upload,
        inputs=[file_upload, load_all_sheets],
        outputs=[
            datasets_state,
            dataset_summary,
            active_datasets,
            preview_dataset,
            data_display,
            upload_status,
            profile_dataset,
            left_dataset,
            right_dataset,
            left_column,
            right_column,
        ],
    )
    load_all_sheets.change(
        on_files_upload,
        inputs=[file_upload, load_all_sheets],
        outputs=[
            datasets_state,
            dataset_summary,
            active_datasets,
            preview_dataset,
            data_display,
            upload_status,
            profile_dataset,
            left_dataset,
            right_dataset,
            left_column,
            right_column,
        ],
    )

    preview_dataset.change(
        on_preview_change,
        inputs=[preview_dataset, datasets_state],
        outputs=data_display,
    )

    profile_dataset.change(
        on_profile_generate,
        inputs=[profile_dataset, datasets_state],
        outputs=[profile_overview, profile_table],
    )
    generate_profile.click(
        on_profile_generate,
        inputs=[profile_dataset, datasets_state],
        outputs=[profile_overview, profile_table],
    )
    export_profile.click(
        on_profile_export,
        inputs=[profile_table, profile_export_formats],
        outputs=[profile_files, profile_status],
    )

    left_dataset.change(
        on_dataset_change,
        inputs=[left_dataset, datasets_state],
        outputs=left_column,
    )
    right_dataset.change(
        on_dataset_change,
        inputs=[right_dataset, datasets_state],
        outputs=right_column,
    )
    auto_match.click(
        on_auto_match,
        inputs=[left_dataset, right_dataset, datasets_state],
        outputs=[left_column, right_column, auto_match_status],
    )
    preview_join.click(
        on_preview_join,
        inputs=[datasets_state, left_dataset, right_dataset, left_column, right_column, join_type, preview_rows],
        outputs=[join_preview, relation_stats, join_state],
    )
    save_join.click(
        on_save_join,
        inputs=[datasets_state, join_state, new_dataset_name],
        outputs=[
            datasets_state,
            active_datasets,
            preview_dataset,
            profile_dataset,
            left_dataset,
            right_dataset,
            save_status,
        ],
    )

    provider.change(
        on_provider_change,
        inputs=provider,
        outputs=[api_base, model],
    )
    save_config.click(
        on_apply_config,
        inputs=[provider, api_base, model, api_key],
        outputs=[config_state, config_status],
    )

    btn_line.click(lambda q: _append_hint(q, CHART_HINTS["折线图"]), inputs=question_input, outputs=question_input)
    btn_bar.click(lambda q: _append_hint(q, CHART_HINTS["柱状图"]), inputs=question_input, outputs=question_input)
    btn_pie.click(lambda q: _append_hint(q, CHART_HINTS["饼图"]), inputs=question_input, outputs=question_input)
    btn_scatter.click(lambda q: _append_hint(q, CHART_HINTS["散点图"]), inputs=question_input, outputs=question_input)
    btn_hist.click(lambda q: _append_hint(q, CHART_HINTS["直方图"]), inputs=question_input, outputs=question_input)

    submit_button.click(
        on_submit,
        inputs=[datasets_state, active_datasets, question_input, output_mode, export_formats, config_state],
        outputs=[output_text, output_files, output_image],
    )


demo.launch()
