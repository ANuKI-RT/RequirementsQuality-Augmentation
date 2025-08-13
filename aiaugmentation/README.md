# Requirements Augmentation

This project contains the implementation of the different model interfaces used to augment the given requirements.
4 approaches have been used to augment requirements:
<ul>
    <li>eda</li>
    <li>gpt</li>
    <li>para</li>
    <li>rtt</li>
</ul>

The main.py has predefined functions to replicate experiments that only have to be uncommented to execute certain tasks.
Generally, an input first has to be read by the system by using the read_raw_data() method. The output can then be parsed to one of the chosen models.
The generated results can be passed to a optional function for removal of duplicates before being exported as a new created file either as .json or .txt

As an input, a .txt file is expected, were each requirement is one line. [END]-Tokens will be removed, content enclosed between [SEP]-Tokens will be ignored.

## Folder structure
The folders 
<ul>
    <li>eda</li>
    <li>gpt</li>
    <li>para</li>
    <li>rtt</li>
</ul>
contain the model-interfaces and code surrounding the application of these models. In this project, fully implemented models are not shipped due to memory-usage considerations.
It is therefore necessary to train the GPT-Model manually and download the desired Fairseq models from https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md

The folder data contains the input-text as well as the output data. The name of the folders should not be changed as those are accessed by other methods. 
Code for reading .txt or .json files as well as the ability to create such files is also located there.

## Installation
Consider using conda for the installation:
    conda env create --name envname --file=environment.yml
This takes care of everything except for fairseq and the parrot paraphraser. 

To setup fairseq, it is recommended to follow the steps described on https://github.com/facebookresearch/fairseq

<ol>
    <li>Clone the fairseq project from github into the rtt-folder:
        git clone https://github.com/pytorch/fairseq ./rtt/fairseq
    </li>
    <li>afterwards, install fairseq from the cloned project folder:
        pip install --editable ./rtt/fairseq
    on MacOS:
        CFLAGS="-stdlib=libc++" pip install --editable ./rtt/fairseq
    </li>
    <li>lastly download the two models that are used for the rtt from https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md and unpack them, make sure they are locacted in the rtt folder:
        curl https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz | tar xvjf -
        curl https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz | tar xvjf -
    </li>
</ol>
Now, language models for german to english and vice versa are installed. For russian translation, refer to https://github.com/facebookresearch/fairseq

In this version of the software, the setup for parror paraphraser has been added to the environment.yml. In case this doesn't work, do as instructed below:

To setup parrot paraphraser, the package has to be downloaded directly from github
    pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

## Versions and Dependencies
Warning! Fairseq is incompatible on windows OS. It is required to use OS systems based on Linux, such as MacOS
Python 3.9.17 has been used, compatibility-issues occured with python 3.11