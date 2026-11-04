import argparse
import json
from pathlib import Path
import torch
from glyph.extraction import extract_embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract raw FM embeddings separately from Glyph analysis")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--encoder", action="append", required=True, metavar="COLUMN=MODEL")
    parser.add_argument("--model-kwargs", help="JSON file mapping input columns to encoder keyword arguments")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    mapping = dict(item.split("=", 1) for item in args.encoder)
    output = Path(args.output_dir)
    if output.exists():
        parser.error("output-dir must be a new directory")
    kwargs = json.loads(Path(args.model_kwargs).read_text()) if args.model_kwargs else None
    extracted = extract_embeddings(args.metadata, mapping, model_kwargs=kwargs,
                                   device=args.device, batch_size=args.batch_size)
    output.mkdir(parents=True, exist_ok=False)
    for modality in mapping:
        ids, splits, values = [], [], []
        for split, data in extracted.items():
            ids.extend(data["sample_ids"])
            splits.extend([split] * len(data["sample_ids"]))
            values.append(data["embeddings"][modality])
        name = modality.replace("_path", "")
        torch.save({"sample_id": ids, "split": splits, name + "_embedding": torch.cat(values),
                    "encoder": mapping[modality], "schema_version": 1},
                   output / (name + "_embeddings.pt"))


if __name__ == "__main__":
    main()
