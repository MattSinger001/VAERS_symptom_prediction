
## Missing Symptom Prediction

This repository shows a tool which predicts a undiagnosed, lethal patient symptom given a set of up to 10 other known symptoms. 
A 'lethal' symptom is a MedDRA terms such that the probability of a patient dying given the 'lethal' symptom is above 20% in the VAERS database.
The base nerual network model was trained utilizing patient observations from the VAERS dataset.

The model is trained to output a set of softmax probabilities for each candidate symptom. Utilizing the technique of conformal prediction the set of softmax probabilies
is able to be converted into a prediction set such that the prediction set contains the correct response at a desired level of coverage. This basic proof of concept
allows for the user to select a coverage level of 80%, 90%, 95%, 99%, or 99.9%. They may also select what 'non-conformity score' the conformal prediction algorithm will use.
Although all choices of non-conformity score result in achieving the correct coverage (validity), they varry in their average prediction set size (efficiency).

We recommend the utilization of the 'probability' non-conformity score.

The model has two modes of input: 'Manual' and 'Test Observations'. The 'Manual' option will allow a user to enter into 10 possible MedDRA High Level Group Terms.
The 'Test Obserrvations' option allows for a user to examine observations from the test dataset inorder to evaluate how accurate the model is.

# Installation
To run the above code, simply download the given folder and edit run_streamlit.py. The variable 'base_path' needs to be modified to the location of the downloaded folder.

Additionally your python environment needs to contain the following libraries:
  - Streamlit
  - Numpy
  - Pandas
  - Tensorflow

# Test Observations Example
![streamlit_example](https://github.com/user-attachments/assets/4ba13244-6bf9-43f7-b01b-82378577f908)



# Manual Input Example
![Streamlit_example_manual](https://github.com/user-attachments/assets/fe2fb341-c30b-4728-b829-9d45c4180858)
