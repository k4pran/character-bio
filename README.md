# Character NER Prototype

Small Python prototype for training a DistilBERT token classifier to detect character names in messy audiobook/STT text.

## Approach

This uses **NER** / **token classification**.

Each token gets a BIO label:

```text
B-CHARACTER = first token of a character name
I-CHARACTER = later token in the same character name
O           = not a character
```

Example:

```text
aren          B-CHARACTER
vale          I-CHARACTER
said          O
nothing       O
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

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
```

All discovered `.conll` files are logged and merged into one full dataset.

Since there are no separate `validation.conll` or `test.conll` files, the merged training data is automatically split into:

```text
80% train
10% validation
10% test
```

This means the validation/test sets grow automatically as more examples are added.

## Train

Quick smoke test:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --epochs 1 --batch_size 8 --max_length 64
```

Normal test run:

```powershell
python src/train.py --data_dir data --output_dir models/character-ner --model_name distilbert/distilbert-base-uncased --epochs 5 --batch_size 16 --max_length 128
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
  "I-CHARACTER": 2
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
