# Investigating Machine Unlearning Effects on Legal Reasoning Capabilities
Does unlearned targeted names from a model's training data negatively affect the model's performance on legal reasoning tasks?

## Table of Contents
[Project Description](#description)\
[Getting Started](#getting-started)\
[AI Use Statement](#ai-use-statement)\
[Authors](#authors)\
[Licensing Information](#license)

## Description

As LLMs grow in capability and popularity, issues surrounding the intake of confidential data like names or SSNs arise. In the legal world, models are trained on large collections of legal documents that often contain sensitive information. Although the integration of an LLM into businesses and corporations can be very beneficial, there are concerns about trading privacy for efficiency.

Machine unlearning is a potential solution to limiting the risk of a model using your sensitive information as part of its training. In basic applications, machine unlearning has proven to be a strong contender for mitigating these risks, but it's efficacy has not been applied to legal domain-specific applications.

This project seeks to investigate how machine unlearning affects the utility and legal reasoning performance of a model built for legal applications. This repo specifically provides a foundation for research into legal applications of machine unlearning and protecting one's privacy when it comes to LLMs. The code in this repository is structured for reproducibility.

We curate a list of targeted names to unlearn from the training data of our chosen model. After loading the SaulLM-7B-Instruct model, we evaluate its performance before and after applying Negative Preference Optimization, a relatively novel unlearning technique. Then, we evaluate the performance of the model after unlearning. Based on the results, you can directly see how effective unlearning can be and whether there is a tradeoff between unlearning the information and having a high performing model.

## Getting Started
This project was run in the Google Colab Pro Python 3.12 environment with the A100 GPU. Please account for the compute requirements due to the large size of the model.

### Dependencies
- Install dependencies and all the required python packages:
`pip install -r requirements.txt`

### Execution
- There are two ways to run this program:
1. You can run `main.py` directly, using the following command `python3 main.py`. This is preferred if you have the compute support and only want to view the final results.
2. You can open the `final_project_run.ipynb` and run the program through the Jupyter notebook to see the output of each stage of the process. This can be done in a virtual environment like Google Colab if you do not have the compute support. I recommend running the project in the notebook as I have not tested `main.py` as my computer system does not support the memory requirements for running this program.


## AI Use Statement
This project used ChatGPT during the development process. It was used for generating the code for the NPO training loop, the metric computations, and the graph generations of results. Any code it produced has been edited, formatted, and adapted for improved readability and flexibility.

## Authors
- Kynnedy Armstrong (kynrosea)
- Ben Noyes (bnoyes)

## Version History
- 0.1
    - Initial Release

## License
This project is licensed under the MIT License - see the LICENSE file for details.
