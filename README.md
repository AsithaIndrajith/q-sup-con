# Q-SupCon: Quantum-Enhanced Supervised Contrastive Learning Architecture within the Representation Learning Framework

In an era of evolving data privacy regulations, the demand for robust deep classification models confronts the challenge of acquiring extensive training data. The efficacy of these models heavily relies on the volume of training data due to the numerous parameters requiring fine-tuning. However, obtaining such vast data proves arduous, especially in fields like medical applications, where robust models for early disease detection are imperative yet labeled data is scarce. Despite this, classical supervised contrastive learning models have shown promise in mitigating this challenge, albeit to a certain extent, by leveraging deep encoder models. Nonetheless, recent advancements in quantum machine learning offer the potential to extract meaningful representations from even extremely limited and simple data. Thus, replacing classical counterparts in classical or hybrid quantum-classical supervised contrastive models enhances feature learning capability with minimal data. Consequently, this work introduces the Q-SupCon model, a fully quantum-powered supervised contrastive learning model comprising a quantum data augmentation circuit, quantum encoder, quantum projection head, and quantum variational classifier, enabling efficient image classification with minimal labeled data. Furthermore, the novel model achieves 80%, 60%, and 80% test accuracy on MNIST, KMNIST, and FMNIST datasets, respectively, marking a significant advancement in addressing the data scarcity challenge.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Published Research Paper](#published-research-paper)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The Q-SupCon research addresses the challenge of data scarcity in classification models, especially in medical applications. By integrating quantum computing principles into supervised contrastive learning, it aims to enhance feature learning with minimal labeled data. Quantum data augmentation, encoding, and classification components enable efficient image classification. This approach promises to revolutionize classification models, offering robust solutions in data-limited scenarios.

## Features

- **Generalized Multi-class Classification Model:** A novel multi-class classification model specifically designed to operate efficiently with fewer labeled images. This model comprises four key components: Quantum Data Augmentation Module, Quantum Encoder, Quantum Projection Head, and Quantum Classifier.

- **Maximize the Potential of QAE for Enhanced Feature Learning:** The objective of this study was to leverage quantum gates to enhance feature learning using Quantum Auto-encoders (QAE), surpassing the capabilities of traditional supervised learning models and even novel hybrid quantum-classical models.

- **Quantum Image Processing and Compression:** A versatile data augmentation circuit designed for quantum image processing, coupled with an adaptable quantum autoencoder circuit for data compression, is suitable for diverse application domains.

## Getting Started

To get started with this project, follow the steps below to set up and run the code on your local machine.

### Prerequisites

Before running the code, make sure you have the following software and dependencies installed:

1. **Conda:**
   - If you don't have Conda installed, follow the installation instructions for your operating system [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create a Conda Environment:**
   - Create a new Conda environment:
     ```bash
     conda env create -n <env_name> python=3.10
     ```

   - Activate the Conda environment:
     ```bash
     conda activate your_environment_name
     ```

3. **PyTorch:**
   - Install PyTorch within your Conda environment:
     ```bash
     conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
     ```

4. **IBM Qiskit:**
   - Install IBM Qiskit within your Conda environment:
     ```bash
     conda install -c conda-forge qiskit
     ```

   - Additional installation for IBM Qiskit Aer (optional):
     ```bash
     conda install -c conda-forge qiskit-aer
     ```

   - Additional installation for IBM Qiskit Aqua (optional):
     ```bash
     conda install -c conda-forge qiskit-aqua
     ```

Make sure to replace `your_environment_name` with the desired name for your Conda environment.

Now, you're ready to run the project with the specified dependencies.

### Installation

To install and run the project on your local machine, follow these simple steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AsithaIndrajith/q-sup-con.git

## Usage

To use this project, follow the steps outlined below:

1. **Open the Jupyter Notebook:**
   - Navigate to the project directory using the terminal or file explorer.
   - Launch the Jupyter Notebook by running:
     ```bash
     jupyter notebook
     ```
   - In the Jupyter interface, open the `main.ipynb` notebook.

2. **Run the Code:**
   - Within the Jupyter Notebook, navigate to the specific code cell you want to run.
   - Execute the cell by clicking the "Run" button or using the keyboard shortcut `Shift + Enter`.

Note: Ensure that you have all the necessary dependencies installed in your Python environment. If you encounter any issues, refer to the project documentation or README for additional instructions.

## Contributing

We welcome contributions from the community to enhance and improve this project. If you'd like to contribute, please follow these guidelines:

### Reporting Issues

If you encounter any bugs, issues, or have suggestions for improvements, please check if the issue has already been reported in the [Issues](https://github.com/AsithaIndrajith/q-sup-con/issues) section. If not, feel free to open a new issue with details on the problem or enhancement request.

### Feature Requests

If you have ideas for new features or improvements, you can submit them by opening an issue in the [Issues](https://github.com/AsithaIndrajith/q-sup-con/issues) section. Provide a clear description of the proposed feature and its benefits.

### Pull Requests

We encourage you to contribute directly by submitting pull requests. Follow these steps:

1. Fork the repository to your GitHub account.
2. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/AsithaIndrajith/q-sup-con.git
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature_branch_name
   ```
4. Make your changes, commit them, and push to your forked repository:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature_branch_name
   ```
5. Open a pull request on the [Pull Requests](https://github.com/AsithaIndrajith/q-sup-con/pulls) page. Provide a clear title and description for your changes.

## Published Research Paper

The Q-SupCon model is associated with the following published research paper:

- **Title:** Q-SupCon: Quantum-Enhanced Supervised Contrastive Learning Architecture within the Representation Learning Framework
- **Authors:** Asitha Kottahachchi Kankanamge Don; Ibrahim Khalil
- **Journal/Conference:** ACM Transactions on Quantum Computing
- **Year:** 2024
- **DOI or Link:** http://dx.doi.org/10.1145/3660647

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is supported by the Australian Research Council Discovery Project (DP210102761). We would like to extend our acknowledgment to Robert Shen from RACE (RMIT AWS Cloud Supercomputing Hub) and Jeff Paine from Amazon Web Services (AWS) for their invaluable provision of computing resources.