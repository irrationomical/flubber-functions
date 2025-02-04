import pandas as pd
import numpy as np
import argparse
import json
import os

from pandas.api.types import is_numeric_dtype, is_string_dtype

def load_data(file_path):
    """
    Loads a CSV or Excel file into a pandas DataFrame.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv']:
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df

def analyze_column(col_name, series, desc_dict):
    """
    Analyze a single column in the DataFrame.
    
    Returns a dictionary with:
      - Field Name
      - Definition (from the provided description dictionary)
      - Current Data Type (from pandas)
      - Recommended Data Type (based on our simple heuristics)
      - Data Category (e.g., Categorical, Quantitative, Nested JSON)
      - Additional Info (unique values, summary statistics, etc.)
    """
    # Retrieve the field definition if available
    definition = desc_dict.get(col_name, "No description provided.")
    
    # Get the current dtype (as a string)
    current_dtype = series.dtype
    additional_info = {}
    
    # Count missing values
    missing = series.isna().sum()
    additional_info["Missing values"] = int(missing)

    # Check for nested JSON if the column is a string
    nested_flag = False
    sample_json = None
    if is_string_dtype(series):
        # Sample a few non-null values
        for val in series.dropna().head(10):
            if isinstance(val, str):
                val_strip = val.strip()
                if (val_strip.startswith("{") and val_strip.endswith("}")) or \
                   (val_strip.startswith("[") and val_strip.endswith("]")):
                    try:
                        parsed = json.loads(val_strip)
                        nested_flag = True
                        sample_json = parsed
                        break
                    except Exception:
                        continue

    if nested_flag:
        data_category = "Nested JSON"
        recommended_dtype = str(current_dtype)
        additional_info["Sample JSON"] = sample_json
    else:
        # If the column is numeric, decide if it's boolean, discrete or continuous.
        if is_numeric_dtype(series):
            # Get unique non-null values
            unique_values = series.dropna().unique()
            num_unique = len(unique_values)
            # Boolean: if only two unique values (e.g. 0 and 1)
            if num_unique == 2 and set(unique_values) <= {0, 1}:
                data_category = "Boolean"
                recommended_dtype = "bool"
            else:
                # Check if all non-null numeric values are integers
                if series.dropna().apply(lambda x: float(x).is_integer()).all():
                    # For integer values, check the range to recommend small, regular, or big int.
                    min_val = series.min()
                    max_val = series.max()
                    if min_val >= -32768 and max_val <= 32767:
                        recommended_dtype = "smallint"
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        recommended_dtype = "int"
                    else:
                        recommended_dtype = "bigint"
                    # Categorize as discrete if the number of unique values is small
                    if num_unique < 20:
                        data_category = "Discrete Numeric"
                        additional_info["Unique values"] = sorted(list(unique_values))
                    else:
                        data_category = "Quantitative (Discrete)"
                        additional_info["Summary"] = series.describe().to_dict()
                else:
                    # Otherwise treat as continuous
                    data_category = "Quantitative (Continuous)"
                    recommended_dtype = "float"
                    additional_info["Summary"] = series.describe().to_dict()
        elif is_string_dtype(series):
            # For string types, check the number of unique values to decide if it's categorical
            unique_values = series.dropna().unique()
            num_unique = len(unique_values)
            if num_unique < 20:
                data_category = "Categorical"
                recommended_dtype = "category"
                additional_info["Unique values"] = list(unique_values)
            else:
                data_category = "Text"
                recommended_dtype = "text"
                additional_info["Unique count"] = num_unique
        else:
            # Fallback for other types (including datetime)
            if pd.api.types.is_datetime64_any_dtype(series):
                data_category = "Datetime"
                recommended_dtype = "datetime"
            else:
                data_category = "Unknown"
                recommended_dtype = str(current_dtype)
    
    # NEW FUNCTIONALITY:
    # For fields that aren't categorized as "Categorical" OR if the column name includes "id"
    # (case-insensitive), add a random sample of non-null values.
    if (data_category != "Categorical") or ("id" in col_name.lower()):
        non_null_values = series.dropna()
        if not non_null_values.empty:
            sample_count = min(5, len(non_null_values))
            additional_info["Random Samples"] = non_null_values.sample(n=sample_count).tolist()
    
    return {
        "Field Name": col_name,
        "Definition": definition,
        "Current Data Type": str(current_dtype),
        "Recommended Data Type": recommended_dtype,
        "Data Category": data_category,
        "Additional Info": additional_info
    }

def generate_markdown_report(analysis_results, output_file):
    """
    Generates a Markdown report from the analysis results.
    Each field gets its own section with a table-of-contents at the top.
    """
    def convert_to_serializable(obj):
        """Helper function to convert NumPy types to Python native types"""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        return obj

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Data Documentation Report\n\n")
        
        # Create a Table of Contents
        f.write("## Table of Contents\n")
        for res in analysis_results:
            field_name = res["Field Name"]
            anchor = field_name.lower().replace(" ", "-")
            f.write(f"- [{field_name}](#{anchor})\n")
        f.write("\n---\n")
        
        # Write detailed sections for each field
        for res in analysis_results:
            field_name = res["Field Name"]
            anchor = field_name.lower().replace(" ", "-")
            f.write(f"## {field_name}\n\n")
            f.write(f"**Definition:** {res['Definition']}\n\n")
            f.write(f"**Current Data Type:** `{res['Current Data Type']}`\n\n")
            f.write(f"**Recommended Data Type:** `{res['Recommended Data Type']}`\n\n")
            f.write(f"**Data Category:** {res['Data Category']}\n\n")
            
            # Additional information section
            f.write("**Additional Information:**\n")
            additional_info = res["Additional Info"]
            if additional_info:
                for key, value in additional_info.items():
                    # If the additional information is a dictionary or list, format it as a code block
                    if isinstance(value, dict) or isinstance(value, list):
                        serializable_value = convert_to_serializable(value)
                        f.write(f"- **{key}:**\n```json\n{json.dumps(serializable_value, indent=2)}\n```\n")
                    else:
                        f.write(f"- **{key}:** {convert_to_serializable(value)}\n")
            else:
                f.write("None\n")
            f.write("\n---\n")
    print(f"Documentation generated and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate thorough documentation on a CSV or Excel dataset."
    )
    parser.add_argument("data_file", help="Path to the CSV or Excel data file.")
    parser.add_argument("desc_file", help="Path to the CSV or Excel file containing field descriptions.")
    parser.add_argument("desc_column", help="Name of the column in the description file that contains field descriptions.")
    parser.add_argument("--output", default="data_documentation.md", help="Output Markdown file.")
    
    args = parser.parse_args()

    # Load the main data file
    data = load_data(args.data_file)
    
    # Load the field descriptions file.
    # (This file should include a column for field names and a column with definitions.)
    desc_df = load_data(args.desc_file)
    
    # Create a mapping from field name to description.
    # We assume that the description file contains a column named "Field" for field names.
    if "Field" not in desc_df.columns:
        # If no "Field" column exists, assume the first column holds the field names.
        field_col = desc_df.columns[0]
    else:
        field_col = "Field"
    
    desc_mapping = {}
    for _, row in desc_df.iterrows():
        field_name = row[field_col]
        definition = row[args.desc_column]
        desc_mapping[field_name] = definition

    # Analyze each column in the data file
    analysis_results = []
    for col in data.columns:
        series = data[col]
        res = analyze_column(col, series, desc_mapping)
        analysis_results.append(res)

    # Generate a multi-section Markdown report
    generate_markdown_report(analysis_results, args.output)

if __name__ == "__main__":
    main()