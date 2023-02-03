# Introduction
HARP Sonification Data Processing. Used for converting magnetospheric ULF waves to sound.
Default processing codes can be found in default_process.py. 
Run with default options or set your own start time, end time, probe, etc. 
in the call to process_data at the bottom of the file.
You can also run the batch_process.py to batch process selected events in the event_list folder.
Examples of output can be found in the outputs folder.

---
# Installation

## Requirements
You need Visual C++ 2014 or a higher version. Download:
https://visualstudio.microsoft.com/visual-cpp-build-tools

### Install with venv
Init your virtual environment
```shell
python3 venv venv
```
Install all the libraries
```shell
pip install requirements.txt
```
---
## Run program
/!\ The ogg sound type can create error. Don't hesitate to remove it.
### Execute the process with default parameter.
```shell
default_process.py
```
### With orbit file. 
Orbit file must be composed of 2 column separated with one space.

example:
```txt
2008-02-01/21:10:00 2008-02-02/21:06:00
```
The format for the date must be this one: **%Y-%m-%d/%H:%M:%S**

```shell
default_process.py --orbit_file THEMISE_Orbits_Feb2008_Nov2020.txt
```

###With orbit file and light mode _HIGHLY RECOMMENDED_
```shell
default_process.py --orbit_file THEMISE_Orbits_Feb2008_Nov2020.txt --light
```