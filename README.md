# Character and Location NER Prototype

Small Python prototype for training a DistilBERT token classifier to detect character names and named locations in messy audiobook/STT text.

## Approach

This uses **NER** / **token classification**.

Each token gets a BIO label:

```text
B-CHARACTER = first token of a character name
I-CHARACTER = later token in the same character name
B-LOCATION  = first token of a named location
I-LOCATION  = later token in the same named location
O           = not a labelled entity
```

Example:

```text
aren          B-CHARACTER
vale          I-CHARACTER
said          O
nothing       O
```

## CPU setup

Use this only when training on the CPU. For the RX 9070 XT, skip this section
and use the Radeon setup below so that `pip install torch` does not install the
wrong build.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch
pip install -r requirements.txt
```

`torch` is installed separately because CPU, CUDA, and ROCm builds use different wheels. The ROCm setup below installs the AMD GPU build instead of the default PyPI build.

## Radeon GPU setup (native Windows ROCm)

This project has been verified with native Windows ROCm; WSL and Docker are not
needed. The working machine used:

```text
GPU:       AMD Radeon RX 9070 XT
Driver:    32.0.21036.18
Windows:   build 26200.8875
Python:    3.12.13 (64-bit)
ROCm:      7.2.1 / HIP 7.2.53211-158bd99533
PyTorch:   2.9.1+rocm7.2.1
```

Use Python 3.12: the tested PyTorch wheel is a `cp312` wheel and will not install
on Python 3.13 or 3.14. Install/update the AMD Radeon driver first, following
AMD's [Windows PyTorch installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).

The `.venv-rocm` environment in this working checkout is already configured.
To use it now:

```powershell
.\.venv-rocm\Scripts\Activate.ps1
```

The remaining installation steps are for recreating that environment on this
machine or setting up a fresh clone.

In a new PowerShell window at the repository root, create a separate environment:

```powershell
py -3.12 -m venv .venv-rocm
.\.venv-rocm\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, run this once in that PowerShell window and
then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install the native Windows ROCm 7.2.1 SDK and PyTorch wheel. Keep the backticks
as the final character on each continued line:

```powershell
python -m pip install --no-cache-dir `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"

python -m pip install -r requirements.txt
python -m pip check
```

Do not run `pip install torch` in this environment afterward: the ordinary
PyPI build can replace the ROCm build. This project does not require
`torchvision` or `torchaudio`.

Verify device detection and an actual calculation on the GPU:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('HIP:', torch.version.hip); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda'); y=x@x; torch.cuda.synchronize(); print('calculation:', y.device)"
```

Expected output includes:

```text
torch: 2.9.1+rocm7.2.1
HIP: 7.2.53211-158bd99533
available: True
device: AMD Radeon RX 9070 XT
calculation: cuda:0
```

ROCm deliberately uses PyTorch's existing `torch.cuda` API and reports the
device as `cuda:0`; that does not mean an NVIDIA CUDA build is being used. The
non-empty `torch.version.hip` value confirms the ROCm build.

Start training and require the GPU so the command fails instead of silently
falling back to the CPU:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --epochs 1 --batch_size 8 --max_length 64 --device gpu
```

At startup, look for:

```text
pytorch backend: ROCm/HIP
gpu available through torch.cuda API: True
training device: GPU - AMD Radeon RX 9070 XT
trainer device: cuda:0
```

The verified smoke run completed DistilBERT forward and backward attention
kernels, four optimizer steps, validation, test evaluation, and model saving on
the RX 9070 XT. ROCm reported its AOTriton attention backend for both forward
and backward passes.

If `available` is `False`, check all of the following before training:

- `python --version` says 3.12 and `where.exe python` points inside `.venv-rocm`.
- `python -m pip show torch` reports version `2.9.1+rocm7.2.1`, not a plain CPU build.
- The RX 9070 XT is enabled in Device Manager and the AMD driver is current.
- No later `pip install torch` command replaced the ROCm wheel.

## Data layout

Training recursively searches under `data/` for every `.conll` file:

```text
*.conll
```

Example:

```text
data/
  label2id.json

  synthetic/
    train.conll

  real/
    manually_checked_examples.conll

  litbank/
    105_persuasion.conll
    1342_pride_and_prejudice.conll

  locations/
    fewnerd.conll
    wikigold.conll
    wnut17.conll
```

All discovered `.conll` files are logged and merged into one full dataset.

Since there are no separate `validation.conll` or `test.conll` files, the merged training data is automatically split into:

```text
80% train
10% validation
10% test
```

This means the validation/test sets grow automatically as more examples are added.

### Location data

`data/locations/` contains a deterministic, deduplicated mix of three open
datasets: manually labelled Few-NERD and WikiGold for broad clean coverage, and
WNUT 2017 for rare/noisy social text. Source provenance and licenses are kept
on every CoNLL example. See [the location corpus notes](data/locations/README.md)
for the selection rationale, licenses and rebuild command.

To rebuild the corpus:

```powershell
python src/build_location_corpus.py
```

## Train

Quick smoke test:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --epochs 1 --batch_size 8 --max_length 64
```

Normal test run:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --model_name distilbert/distilbert-base-uncased --epochs 5 --batch_size 16 --max_length 128
```

Device selection defaults to `--device auto`, which uses a CUDA/ROCm GPU when PyTorch can see one and falls back to CPU otherwise. To force CPU:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --device cpu
```

To require a GPU and fail fast if one is not visible:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --device gpu
```

Optional explicit split command:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --model_name distilbert/distilbert-base-uncased --split_from_train --validation_ratio 0.1 --test_ratio 0.1 --epochs 5 --batch_size 16 --max_length 128
```

Training prints:

```text
train row count
validation row count
test row count
validation precision / recall / F1
test precision / recall / F1
```

## Predict

```powershell
python src/predict.py --model_dir models/character-ner --text "aren vale said nothing and the guard opened the door while mira thorn waited"
```

Expected:

```text
Detected characters:
- aren vale
- mira thorn
```

Negative test:

```powershell
python src/predict.py --model_dir models/character-ner --text "the guard and the prisoner walked through the hall"
```

Expected: no detected characters.

## CoNLL / TSV data format

Manual training data should use a simple tabular format:

```text
token<TAB>label
```

Blank lines separate examples.

Comment lines beginning with `#` can be used for metadata.

Example:

```text
# id = real-0001
# source = bad_stt
# text = as if woken from a deep sleep daphne jumps up to hug me
as	O
if	O
woken	O
from	O
a	O
deep	O
sleep	O
daphne	B-CHARACTER
jumps	O
up	O
to	O
hug	O
me	O
```

Another example:

```text
# id = real-0002
# source = bad_stt
# text = she just can't stand the idea of doing something wrong don't tease her daphne whispers in my ear
she	O
just	O
can't	O
stand	O
the	O
idea	O
of	O
doing	O
something	O
wrong	O
don't	O
tease	O
her	O
daphne	B-CHARACTER
whispers	O
in	O
my	O
ear	O
```

## Label IDs

`data/label2id.json`:

```json
{
  "O": 0,
  "B-CHARACTER": 1,
  "I-CHARACTER": 2,
  "B-LOCATION": 3,
  "I-LOCATION": 4,
  "B-QUOTE": 5,
  "I-QUOTE": 6
}
```

## Notes

Keep training data close to real STT:

```text
lowercase
unpunctuated
short windows
explicit names only
pronouns and generic people are O
```

Examples that should usually be `O`:

```text
he
she
her
me
the guard
the prisoner
the old man
the captain
```

Story-specific non-human characters can still be labelled as characters:

```text
white	B-CHARACTER
rabbit	I-CHARACTER
```

But generic animals should stay `O`:

```text
white	O
rabbit	O
```

## Output

The final trained model is saved to:

```text
models/character-ner
```

Use this folder with `predict.py`.

Checkpoint folders are intermediate training snapshots and are usually not needed for prediction once the final model has been saved.

## Export to ONNX for Android

The ONNX export uses the final model in `models/character-ner`, not one of the
intermediate checkpoint folders. Activate the same Python 3.12 environment used
for training and install the export-only dependencies:

```powershell
.\.venv-rocm\Scripts\Activate.ps1
python -m pip install -r requirements-onnx.txt
```

Export and validate the model:

```powershell
python src/export_onnx.py `
  --model_dir models/character-ner `
  --output_dir exported/character-ner-onnx-updated
```

The exporter performs all of these steps:

- Exports ONNX opset 18 with dynamic batch and sequence dimensions.
- Gives the graph stable names: `input_ids`, `attention_mask`, and `logits`.
- Checks the graph with ONNX's model checker.
- Runs the FP32 model in ONNX Runtime and compares its logits with PyTorch.
- Tests a second input shape with a two-sentence batch.
- Produces and validates a dynamically quantized INT8 model for mobile CPU use.
- Copies the tokenizer, vocabulary, labels, and model configuration into the bundle.

The generated folder contains:

```text
exported/character-ner-onnx-updated/
  model.int8.onnx       # about 64 MB; recommended starting point for Android
  model.onnx            # 253.29 MB; full-precision reference/fallback
  config.json
  export_metadata.json  # tensor contract, versions, sizes, validation results
  id2label.json
  label2id.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
```

The verified export used `onnx 1.22.0` and `onnxruntime 1.27.0`. Its FP32
maximum absolute difference from PyTorch was below `0.000012`. The INT8 model
had the same predicted token labels as PyTorch on the export validation batch.
Quantization can still affect accuracy on other data, so compare FP32 and INT8
on representative audiobook/STT samples before shipping.

### Android tensor contract

Use [ONNX Runtime for Android](https://onnxruntime.ai/docs/tutorials/mobile/)
and put the chosen `.onnx` file plus the tokenizer/label files in the app's
assets. The model expects:

```text
input_ids      int64 [batch_size, sequence_length]
attention_mask int64 [batch_size, sequence_length]
logits         float32 [batch_size, sequence_length, 7]
```

Tokenize with DistilBERT's uncased WordPiece vocabulary in `vocab.txt`, adding
`[CLS]` and `[SEP]`, and use the same truncation length used for training
(`128` in the documented commands). Create `attention_mask` with `1` for real
tokens and `0` for padding. DistilBERT does not need `token_type_ids`.

For every token, take `argmax` over the final logits dimension and translate the
result through `id2label.json`. Ignore `[CLS]`, `[SEP]`, padding, and continuation
WordPiece tokens when rebuilding BIO entity spans. The existing Python
`predict.py` shows the corresponding span-merging behavior.
