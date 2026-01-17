import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pandasai as pai
from openai import AuthenticationError
from pandasai_openai import OpenAI

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # macOS; Windows can use SimHei
plt.rcParams['axes.unicode_minus'] = False

EXPORT_DIR = "./exports"
TEMP_DIR = os.path.join(EXPORT_DIR, "temp")
CHART_DIR = os.path.join(EXPORT_DIR, "charts")

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "DeepSeek")
DEFAULT_API_KEY = os.getenv("OPENAI_KEY", "")

PROVIDER_DEFAULTS = {
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "阿里云 DashScope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max-0125"),
    "Gemini(OpenAI兼容)": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-1.5-flash"),
}

default_base, default_model = PROVIDER_DEFAULTS.get(DEFAULT_PROVIDER, ("", ""))
DEFAULT_API_BASE = os.getenv("OPENAI_BASE_URL") or default_base
DEFAULT_MODEL = os.getenv("OPENAI_MODEL") or default_model

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

def _build_llm(api_key: Optional[str], api_base: Optional[str], model: Optional[str]):
    if not api_key:
        return None
    llm = OpenAI(api_token=api_key, api_base=api_base or DEFAULT_API_BASE)
    if model:
        llm.model = model
    return llm


def configure_llm(api_key: Optional[str], api_base: Optional[str], model: Optional[str]) -> Tuple[bool, str]:
    if not api_key:
        return False, "API Key 为空，请在配置页填写"
    llm = _build_llm(api_key, api_base, model)
    if not llm:
        return False, "LLM 配置失败，请检查 API Key 与 Base URL"
    pai.config.set({"llm": llm})
    return True, "LLM 配置已更新"


default_llm = _build_llm(DEFAULT_API_KEY, DEFAULT_API_BASE, DEFAULT_MODEL)
if default_llm is not None:
    pai.config.set({"llm": default_llm})


def _make_unique_name(base: str, existing: set) -> str:
    name = base
    counter = 2
    while name in existing:
        name = f"{base}_{counter}"
        counter += 1
    return name


def _sanitize_table_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "dataset"


def _collect_table_names(datasets: Dict[str, pai.DataFrame]) -> set:
    table_names = set()
    for df in datasets.values():
        table_name = getattr(df, "_table_name", None)
        if table_name:
            table_names.add(table_name)
    return table_names


def add_dataset(
    datasets: Dict[str, pai.DataFrame],
    display_name: str,
    df_raw: pd.DataFrame,
) -> Tuple[Dict[str, pai.DataFrame], str]:
    new_datasets = dict(datasets)
    display_name = _make_unique_name(display_name, set(new_datasets.keys()))

    table_names = _collect_table_names(new_datasets)
    table_base = _sanitize_table_name(display_name)
    table_name = _make_unique_name(table_base, table_names)
    table_names.add(table_name)

    new_datasets[display_name] = pai.DataFrame(df_raw, _table_name=table_name)
    return new_datasets, display_name


def load_datasets(file_paths: List[str], load_all_sheets: bool = True) -> Tuple[Dict[str, pai.DataFrame], pd.DataFrame, List[str]]:
    datasets: Dict[str, pai.DataFrame] = {}
    summary_rows = []
    errors = []
    table_names = set()

    for file_path in file_paths:
        if not file_path:
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.csv':
                df_raw = pd.read_csv(file_path)
                display_name = _make_unique_name(base_name, set(datasets.keys()))
                table_base = _sanitize_table_name(base_name)
                table_name = _make_unique_name(table_base, table_names)
                table_names.add(table_name)
                datasets[display_name] = pai.DataFrame(df_raw, _table_name=table_name)
            elif ext == '.xlsx':
                if load_all_sheets:
                    sheets = pd.read_excel(file_path, sheet_name=None)
                    if isinstance(sheets, dict):
                        for sheet_name, df_raw in sheets.items():
                            display_name = f"{base_name}::{sheet_name}"
                            display_name = _make_unique_name(display_name, set(datasets.keys()))
                            table_base = _sanitize_table_name(f"{base_name}_{sheet_name}")
                            table_name = _make_unique_name(table_base, table_names)
                            table_names.add(table_name)
                            datasets[display_name] = pai.DataFrame(df_raw, _table_name=table_name)
                    else:
                        df_raw = sheets
                        display_name = _make_unique_name(base_name, set(datasets.keys()))
                        table_base = _sanitize_table_name(base_name)
                        table_name = _make_unique_name(table_base, table_names)
                        table_names.add(table_name)
                        datasets[display_name] = pai.DataFrame(df_raw, _table_name=table_name)
                else:
                    df_raw = pd.read_excel(file_path, sheet_name=0)
                    display_name = _make_unique_name(base_name, set(datasets.keys()))
                    table_base = _sanitize_table_name(base_name)
                    table_name = _make_unique_name(table_base, table_names)
                    table_names.add(table_name)
                    datasets[display_name] = pai.DataFrame(df_raw, _table_name=table_name)
            else:
                errors.append(f"不支持的文件类型: {os.path.basename(file_path)}")
        except Exception as exc:  # pragma: no cover - guardrail for file parsing
            errors.append(f"读取失败: {os.path.basename(file_path)} ({exc})")

    for name, df in datasets.items():
        summary_rows.append({
            "dataset": name,
            "rows": len(df),
            "cols": len(df.columns),
        })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return datasets, summary_df, errors


def get_dataset_columns(datasets: Dict[str, pai.DataFrame], dataset_name: str) -> List[str]:
    if not dataset_name or dataset_name not in datasets:
        return []
    return [str(col) for col in datasets[dataset_name].columns]


def build_profile_report(
    datasets: Dict[str, pai.DataFrame],
    dataset_name: str,
) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    if not dataset_name or dataset_name not in datasets:
        return None, None

    df = datasets[dataset_name]
    rows = len(df)
    cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    missing = df.isna().sum()
    if rows:
        missing_pct = (missing / rows * 100).round(2)
    else:
        missing_pct = pd.Series([0] * len(missing), index=missing.index)

    unique = df.nunique(dropna=True)
    example = df.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else "")

    profile_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": missing.values,
        "missing_pct": missing_pct.values,
        "unique": unique.values,
        "example": example.values,
    })

    overview = {
        "rows": rows,
        "cols": cols,
        "duplicates": duplicate_rows,
        "memory_mb": round(memory_mb, 2),
    }
    return overview, profile_df


def export_dataframe(df: pd.DataFrame, export_formats: List[str]) -> List[str]:
    return _export_dataframe(df, export_formats)


def _normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    return re.sub(r"[^a-z0-9]+", "", name)


def suggest_join_columns(left_cols: List[str], right_cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    left_map = { _normalize_column_name(col): col for col in left_cols }
    right_map = { _normalize_column_name(col): col for col in right_cols }
    common = set(left_map.keys()) & set(right_map.keys())

    if not common:
        return None, None

    preferred = [key for key in common if key in {"id", "uuid"} or key.endswith("id")]
    if preferred:
        key = sorted(preferred, key=len)[0]
    else:
        key = sorted(common, key=len)[0]

    return left_map[key], right_map[key]


def build_join_preview(
    datasets: Dict[str, pai.DataFrame],
    left_name: str,
    right_name: str,
    left_col: str,
    right_col: str,
    join_type: str,
    preview_rows: int,
) -> Tuple[Optional[pd.DataFrame], str, Optional[dict]]:
    if not left_name or left_name not in datasets:
        return None, "请选择左表", None
    if not right_name or right_name not in datasets:
        return None, "请选择右表", None

    left_df = datasets[left_name]
    right_df = datasets[right_name]

    if left_col not in left_df.columns:
        return None, f"左表字段不存在: {left_col}", None
    if right_col not in right_df.columns:
        return None, f"右表字段不存在: {right_col}", None

    left_values = left_df[left_col].dropna().unique()
    right_values = right_df[right_col].dropna().unique()
    left_set = set(left_values)
    right_set = set(right_values)
    intersection = len(left_set & right_set)
    denom = min(len(left_set), len(right_set)) or 1
    overlap_ratio = intersection / denom

    merged = pd.merge(
        left_df,
        right_df,
        how=join_type,
        left_on=left_col,
        right_on=right_col,
        suffixes=("_left", "_right"),
    )

    stats = (
        f"左表行数: {len(left_df)} | 右表行数: {len(right_df)} | 连接行数: {len(merged)} | "
        f"匹配值数量: {intersection} | 匹配比例: {overlap_ratio:.2%}"
    )

    join_config = {
        "left_name": left_name,
        "right_name": right_name,
        "left_col": left_col,
        "right_col": right_col,
        "join_type": join_type,
    }

    preview = merged.head(preview_rows)
    return preview, stats, join_config


def build_join_dataset(
    datasets: Dict[str, pai.DataFrame],
    join_config: Optional[dict],
    display_name: str,
) -> Tuple[Dict[str, pai.DataFrame], Optional[str], Optional[str]]:
    if not join_config:
        return datasets, None, "请先预览连接结果"
    if not display_name:
        return datasets, None, "请填写新数据集名称"

    left_df = datasets.get(join_config["left_name"])
    right_df = datasets.get(join_config["right_name"])
    if left_df is None or right_df is None:
        return datasets, None, "选择的数据集不存在"

    merged = pd.merge(
        left_df,
        right_df,
        how=join_config["join_type"],
        left_on=join_config["left_col"],
        right_on=join_config["right_col"],
        suffixes=("_left", "_right"),
    )

    new_datasets, final_name = add_dataset(datasets, display_name, merged)
    return new_datasets, final_name, None


def _export_dataframe(df: pd.DataFrame, export_formats: List[str]) -> List[str]:
    files: List[str] = []

    if 'xlsx' in export_formats:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', dir=TEMP_DIR)
        df.to_excel(temp_file.name, index=False)
        files.append(temp_file.name)

    if 'csv' in export_formats:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=TEMP_DIR)
        df.to_csv(temp_file.name, index=False)
        files.append(temp_file.name)

    if 'json' in export_formats:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=TEMP_DIR)
        df.to_json(temp_file.name, orient='records', force_ascii=False)
        files.append(temp_file.name)

    return files


def _export_chart(response) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=CHART_DIR)
    response.save(temp_file.name)
    return temp_file.name


def process_question(
    datasets: Dict[str, pai.DataFrame],
    selected_names: List[str],
    question: str,
    export_formats: List[str],
    llm_config: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str], str]:
    if not selected_names:
        return "请先选择数据集", [], None

    dfs = [datasets[name] for name in selected_names if name in datasets]
    if not dfs:
        return "未找到已选择的数据集", [], None

    if llm_config is not None:
        ok, message = configure_llm(
            llm_config.get("api_key"),
            llm_config.get("api_base"),
            llm_config.get("model"),
        )
        if not ok:
            return message, [], None

    try:
        response = pai.chat(question, *dfs)
    except AuthenticationError:
        return "API Key 无效，请检查 OPENAI_KEY 配置", [], None
    except Exception as exc:
        return f"处理失败：{exc}", [], None

    if response.type == 'error':
        return response.value, [], None
    if response.type == 'dataframe':
        export_formats = export_formats or []
        files = _export_dataframe(response.value, export_formats)
        return "数据处理完成，请下载结果", files, None
    if response.type == 'chart':
        image_path = _export_chart(response)
        files = [image_path] if 'png' in (export_formats or []) else []
        return "图片生成完成", files, image_path

    return str(response.value), [], None
