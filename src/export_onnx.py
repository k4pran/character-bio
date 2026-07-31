import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer


class TokenClassifierOnnx(torch.nn.Module):
    """Expose only the two Android inputs and the logits output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def run_onnx(path: Path, inputs: dict[str, torch.Tensor]) -> np.ndarray:
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {
        name: value.detach().cpu().numpy()
        for name, value in inputs.items()
    }
    return session.run(["logits"], ort_inputs)[0]


def copy_label_maps(model_dir: Path, output_dir: Path) -> None:
    for filename in ("label2id.json", "id2label.json"):
        source = model_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained token classifier to Android-ready ONNX files."
    )
    parser.add_argument("--model_dir", default="models/character-ner")
    parser.add_argument(
        "--output_dir",
        default="exported/character-ner-onnx-updated",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--sample_text",
        default="aren vale walked through london with mira thorn",
        help="Text used for export tracing and numerical validation.",
    )
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Create only the full-precision model.onnx file.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    fp32_path = output_dir / "model.onnx"
    int8_path = output_dir / "model.int8.onnx"

    if not model_dir.exists():
        raise FileNotFoundError(f"Trained model directory not found: {model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stale_external_data = output_dir / "model.onnx.data"
    if stale_external_data.exists():
        stale_external_data.unlink()

    print(f"Loading trained model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        # The optimized SDPA path can bake the traced attention-mask shape into
        # a legacy ONNX graph. Eager attention preserves dynamic sequence axes.
        attn_implementation="eager",
    )
    model.to("cpu")
    model.eval()
    wrapper = TokenClassifierOnnx(model)

    # Trace with real padding so the attention-mask operations cannot be
    # constant-folded as an all-ones mask.
    inputs = tokenizer(
        [args.sample_text, "daphne waited"],
        padding=True,
        return_tensors="pt",
    )
    export_inputs = {
        "input_ids": inputs["input_ids"].to(dtype=torch.long),
        "attention_mask": inputs["attention_mask"].to(dtype=torch.long),
    }

    with torch.inference_mode():
        pytorch_logits = wrapper(**export_inputs).cpu().numpy()

    print(f"Exporting FP32 ONNX model to: {fp32_path}")
    batch_size = torch.export.Dim("batch_size", min=1)
    # DistilBERT inputs always contain at least [CLS] and [SEP].
    sequence_length = torch.export.Dim("sequence_length", min=2, max=512)
    torch.onnx.export(
        wrapper,
        (export_inputs["input_ids"], export_inputs["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=(
            {0: batch_size, 1: sequence_length},
            {0: batch_size, 1: sequence_length},
        ),
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=True,
        external_data=False,
    )

    onnx_model = onnx.load(str(fp32_path))
    onnx.checker.check_model(onnx_model)
    fp32_logits = run_onnx(fp32_path, export_inputs)
    np.testing.assert_allclose(
        fp32_logits,
        pytorch_logits,
        rtol=1e-3,
        atol=1e-4,
    )
    fp32_max_abs_diff = float(np.max(np.abs(fp32_logits - pytorch_logits)))

    # Prove that the declared dynamic batch and sequence axes work with shapes
    # different from the single tracing example.
    dynamic_inputs = tokenizer(
        "the guard followed alice through london",
        return_tensors="pt",
    )
    dynamic_export_inputs = {
        "input_ids": dynamic_inputs["input_ids"].to(dtype=torch.long),
        "attention_mask": dynamic_inputs["attention_mask"].to(dtype=torch.long),
    }
    with torch.inference_mode():
        dynamic_pytorch_logits = wrapper(**dynamic_export_inputs).cpu().numpy()
    dynamic_fp32_logits = run_onnx(fp32_path, dynamic_export_inputs)
    np.testing.assert_allclose(
        dynamic_fp32_logits,
        dynamic_pytorch_logits,
        rtol=1e-3,
        atol=1e-4,
    )
    fp32_max_abs_diff = max(
        fp32_max_abs_diff,
        float(np.max(np.abs(dynamic_fp32_logits - dynamic_pytorch_logits))),
    )

    quantized_metrics = None
    if not args.skip_quantization:
        print(f"Quantizing weights to INT8: {int8_path}")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        onnx.checker.check_model(onnx.load(str(int8_path)))
        int8_logits = run_onnx(int8_path, dynamic_export_inputs)

        if int8_logits.shape != dynamic_pytorch_logits.shape:
            raise RuntimeError(
                f"INT8 output shape {int8_logits.shape} does not match "
                f"PyTorch shape {dynamic_pytorch_logits.shape}"
            )
        if not np.isfinite(int8_logits).all():
            raise RuntimeError("INT8 output contains NaN or infinite values")

        quantized_metrics = {
            "size_mb": round(model_size_mb(int8_path), 3),
            "max_abs_diff_from_pytorch": float(
                np.max(np.abs(int8_logits - dynamic_pytorch_logits))
            ),
            "mean_abs_diff_from_pytorch": float(
                np.mean(np.abs(int8_logits - dynamic_pytorch_logits))
            ),
            "token_label_agreement_with_pytorch": float(
                np.mean(
                    np.argmax(int8_logits, axis=-1)
                    == np.argmax(dynamic_pytorch_logits, axis=-1)
                )
            ),
        }

    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    copy_label_maps(model_dir, output_dir)

    metadata = {
        "source_model": str(model_dir),
        "task": "token-classification",
        "opset": args.opset,
        "inputs": {
            "input_ids": "int64[batch_size, sequence_length]",
            "attention_mask": "int64[batch_size, sequence_length]",
        },
        "outputs": {
            "logits": "float32[batch_size, sequence_length, num_labels]",
        },
        "num_labels": model.config.num_labels,
        "id2label": {
            str(key): value for key, value in model.config.id2label.items()
        },
        "validation_sample": args.sample_text,
        "dynamic_shape_validation": {
            "batch_size": int(dynamic_export_inputs["input_ids"].shape[0]),
            "sequence_length": int(dynamic_export_inputs["input_ids"].shape[1]),
        },
        "fp32": {
            "filename": fp32_path.name,
            "size_mb": round(model_size_mb(fp32_path), 3),
            "max_abs_diff_from_pytorch": fp32_max_abs_diff,
        },
        "int8": (
            {"filename": int8_path.name, **quantized_metrics}
            if quantized_metrics is not None
            else None
        ),
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    with (output_dir / "export_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nExport validation passed")
    print(f"FP32 size: {model_size_mb(fp32_path):.2f} MB")
    print(f"FP32 maximum absolute difference: {fp32_max_abs_diff:.8f}")
    if quantized_metrics is not None:
        print(f"INT8 size: {model_size_mb(int8_path):.2f} MB")
        print(
            "INT8 token-label agreement with PyTorch: "
            f"{quantized_metrics['token_label_agreement_with_pytorch']:.2%}"
        )
    print(f"Android bundle written to: {output_dir}")


if __name__ == "__main__":
    main()
