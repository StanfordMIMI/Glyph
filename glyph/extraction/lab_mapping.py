"""Resolve published model lab IDs without overwriting duplicate code mappings."""
import pandas as pd


def _unit(value):
    return "" if pd.isna(value) else "".join(str(value).lower().split()).replace("µ", "u").replace("μ", "u")


def _name_and_unit(name):
    text = str(name).strip()
    depth = 0
    if text.endswith(")"):
        for index in range(len(text) - 1, -1, -1):
            depth += (text[index] == ")") - (text[index] == "(")
            if depth == 0:
                return text[:index].strip(), text[index + 1:-1]
    return text, ""


def mapping_codes(mapping):
    mapping = mapping.copy()
    mapping["mapping_code"] = mapping["loinc_code"]
    if "selected_code" in mapping and "selected_code_system" in mapping:
        local = mapping.selected_code_system.eq("STANFORD_LAB")
        mapping.loc[local, "mapping_code"] = mapping.loc[local, "selected_code"]
    mapping = mapping.dropna(subset=["mapping_code", "lab_id"])
    mapping = mapping[mapping.mapping_code.astype(str).str.strip().ne("")]
    mapping["mapping_code"] = mapping.mapping_code.astype(str).str.removeprefix("LOINC/")
    return mapping


def resolve_lab_ids(labs, mapping, stats):
    stats = stats[stats.retained].set_index("lab_id", drop=False)
    mapping = mapping_codes(mapping)
    mapping = mapping[mapping.lab_id.isin(stats.index)]
    candidates = {str(code): list(dict.fromkeys(group.lab_id.astype(int)))
                  for code, group in mapping.groupby("mapping_code")}
    overrides = labs["lab_id"].tolist() if "lab_id" in labs else [None] * len(labs)
    overrides = [None if pd.isna(value) else int(value) for value in overrides]
    codes = labs.loinc_code.astype(str).str.removeprefix("LOINC/") if "loinc_code" in labs else pd.Series("", index=labs.index)
    unit_column = next((c for c in ("unit", "units", "Units") if c in labs), None)
    units = labs[unit_column].map(_unit) if unit_column else pd.Series("", index=labs.index)
    names = labs.lab_name.fillna("").astype(str).str.lower() if "lab_name" in labs else pd.Series("", index=labs.index)
    keys = list(zip(codes, units, names, overrides))
    resolved = {}
    for key in dict.fromkeys(keys):
        code, unit, name, override = key
        if override is not None:
            if override not in stats.index:
                raise ValueError("Explicit lab_id is not supported by this encoder")
            resolved[key] = override
            continue
        ids = candidates.get(code, [])
        if unit:
            ids = [i for i in ids if _unit(stats.loc[i].get("training_unit", _name_and_unit(stats.loc[i, "name"])[1])) == unit]
        if name:
            ids = [i for i in ids if name in {str(stats.loc[i, "name"]).lower(),
                   str(stats.loc[i].get("measurement_name", _name_and_unit(stats.loc[i, "name"])[0])).lower()}]
        if len(ids) > 1:
            raise ValueError(f"LOINC {code} has multiple model labs; supply unit, lab_name, or an explicit lab_id")
        if code in candidates and not ids:
            raise ValueError(f"LOINC {code} does not match the supplied unit/name; no unit conversion is assumed")
        resolved[key] = ids[0] if ids else None
    return [resolved[key] for key in keys]
