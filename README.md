# Overview

This repository contains source code to augment natural language requirements through four different methods. These methods are
- [Easy-Data-Augmentation](https://aclanthology.org/D19-1670/)
- Locally fine-tuned GPT-2 model
- [Fairseq language model](https://github.com/facebookresearch/fairseq)
- [Parrot Paraphraser](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser)

## Folder structure
The folder is divided into two projects.
aiaugmentation contains the implementation of the different model interfaces used to augment the given requirements.
aiaugmentation_evaluation contains the evaluation tools used during the experiment.

## Installation
Installment requirements and instructions on how to use can be found in the README.md documents located in the corresponding project folders.
It is not necessary to download additional packages for the aiaugmentation_evaluation project, if the installment of the aiaugmentation project has been successfully completed.

## Dependencies
Warning! Fairseq is incompatible on windows OS. It is required to use OS systems based on Linux, such as MacOS

## References
This repository contains source code used for augmentation in the following papers:

Korfmann, R., Beyersdorffer, P., Gerlich, R., Münch, J., & Kuhrmann, M. (2025). Overcoming Data Shortage in Critical Domains With Data Augmentation for Natural Language Software Requirements. Journal of Software: Evolution and Process, 37(5), [10.1002/smr.70027](https://onlinelibrary.wiley.com/doi/10.1002/smr.70027).

Korfmann, R., Beyersdorffer, P., Münch, J., & Kuhrmann, M. (2024, September). Using data augmentation to support AI-based requirements evaluation in large-scale projects. In European Conference on Software Process Improvement (pp. 97-111). Cham: Springer Nature Switzerland [10.1007/978-3-031-71139-8_7](https://link.springer.com/chapter/10.1007/978-3-031-71139-8_7).

The code was originally developed in scope of the Bachelor Thesis  "Anwendung von Verfahren zur Erweiterung eines bestehenden Datensatzes an Anforderungsdokumenten für Software-Projekte der Raumfahrtdomäne" that was submitted on August 30th.

 This Repository is part of the ANuKI project. Further information can be found in the Github [overview page](https://anuki-rt.github.io/) or the on the [HHZ](https://www.hhz.de/forschung/forschungsprojekte/anuki) website.
