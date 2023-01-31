# pb_speaker_adaptation


## Installation

### Conda env
You can either install by copying the conda yaml file with the following command:

```bash
conda env create -f uvapb.yml
```
the corresponding conda env `uvapb` will be created.

### Setup.py
You can also install by running the following command:
`pip install .`

and nlgeval with:
`pip install git+https://github.com/Maluuba/nlg-eval.git`

Note: comment out [this line](https://github.com/Maluuba/nlg-eval/blob/7f7993035a2f4729a15d20040fd904933ea58767/nlgeval/__init__.py#L289) in the library:
```
ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
```

### Additional setup
remember to download spacy with 
`python -m spacy download en_core_web_lg`
