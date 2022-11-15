# pb_speaker_adaptation

Meeting logs: https://docs.google.com/document/d/1ToHEaCEceFcVoE96sbZ5yTIKDOaP6xYvxSMA7ATsoz8/edit#


## Installation

### Conda env
You can either instal by coopying the conda yaml file with the following command:

```bash
conda env create -f uvapb.yml
```
the corresponding conda env `uvapb` will be created.

### Setup.py
You can also install by running the following command:
`pip install .`

and nlgeval with:
`pip install git+https://github.com/Maluuba/nlg-eval.git`

### Additional setup
remember to download spacy with 
`python -m spacy download en_core_web_lg`
