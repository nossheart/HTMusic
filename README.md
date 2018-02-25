# HTMusic
HTMusic is an experimental project for the generation of musical compositions (MIDI) based on the hierarchical temporal memory algorithm developed by Numenta.

## Dependencies
 - [Python 2.7](https://www.python.org/downloads/)
 - [NuPIC](https://github.com/numenta/nupic)
 - [pretty_midi](https://github.com/craffel/pretty-midi)
 - [tqdm](https://github.com/tqdm/tqdm)

## Installing from source
 1. Clone this repo
 2. `pip install -r requirements.txt` from repository root - install dependencies

## Run
To run this application you need to put MIDI files to "input" folder or assign a folder using the parameter `--input_dir`:
```
python train.py
```
Trained model will be saved to "model" directory.
To generate MIDI file after training:
```
python generate.py
```



