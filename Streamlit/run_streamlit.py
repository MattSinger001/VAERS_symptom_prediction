
import streamlit as st

import os
import numpy as np
import pandas as pd

from tensorflow import keras


#%%

st.set_page_config(layout="wide")
base_path = os.getcwd() + '/Streamlit/'


# load in the model, test data, non-conformity score limits, and vocab dictionaries
@st.cache_resource
def load_model():
    base_model = keras.models.load_model(base_path + 'prediction_model')
    #base_model = keras.layers.TFSMLayer(base_path + 'prediction_model',call_endpoint='serving_default')
    
    test_data = pd.read_csv(base_path + 'test_data.csv')
    
    test_inputs = test_data.iloc[:,:10]
    test_labels = test_data.iloc[:,-1]
    
    limits = pd.read_csv(base_path + 'limits.csv')
    coverages = pd.read_csv(base_path + 'coverages.csv')
    
    vocab = pd.read_csv(base_path + 'vocab_hlgt.csv')

    vocab_hlgt_idx= {k:v for k, v in zip(vocab['symptom_hlgt'],vocab['uid'])}
    vocab_idx_hlgt = {}
    
    for k,v in vocab_hlgt_idx.items():
        vocab_idx_hlgt[v] = k
    
    return(base_model,test_inputs,test_labels,limits,coverages, vocab_hlgt_idx,vocab_idx_hlgt)
    
base_model,test_inputs,test_labels,limits,coverages,vocab_hlgt_idx,vocab_idx_hlgt = load_model()

    
#%%

# make the sidebar
with st.sidebar:
    
    # select input type
    st.title('Model Options')

    input_data_select = st.selectbox('Input Data',['Manual Input', 'Test Observations'])
    if input_data_select == '':
        input_data_select = 'Manual Input'

    
    if input_data_select != 'Manual Input':
        input_idx = st.number_input(label = 'Select a test observation (0-{})'.format(len(test_inputs)),min_value = 0 ,max_value = len(test_inputs),value = 0,step = 1)
    
    desired_quants = [.8,.85,.9,.95,.99,.999]
    prob_conf = st.select_slider('Select a confidence level',options = desired_quants)
    prob_idx = desired_quants.index(prob_conf)
    
    desired_nc = ['probability','cumulative probability','index']
    nc_score = st.select_slider('Select a nonconformity score',options = desired_nc)

    if nc_score == 'probability':
        avg_size = coverages[coverages.q == prob_conf].p_len.values[0]
    elif nc_score == 'cumulative probability':
        avg_size = coverages[coverages.q == prob_conf].cp_len.values[0]
    else:
        avg_size = coverages[coverages.q == prob_conf].idx_len.values[0]
    
    st.markdown('The average prediction set size while using the {} nonconformity score at {}% confidence level is {}'.format(nc_score,prob_conf*100,round(avg_size,3)))

    



if input_data_select == 'Manual Input':
    
    key_list = list(vocab_hlgt_idx.keys())
    key_list.sort()
    
    A = st.multiselect('Select up to 10 patient symptoms',options = key_list)
    
    selected_numbers = [0]*10
    
    for idx,x in enumerate(A):
        selected_numbers[idx] = vocab_hlgt_idx[x]
        
    model_input = pd.DataFrame(selected_numbers).T
        
    model_output = base_model.predict(model_input)

else:

    
    col1, col2 = st.columns(2)
    
    model_input = pd.DataFrame(test_inputs.iloc[input_idx,:]).T
    
    model_output = base_model.predict(model_input)
    
    model_input_text = []
    
    for x in model_input.iloc[0,:]:
        symp_name = vocab_idx_hlgt[x]
        if symp_name != 'No-symp':
            model_input_text.append(symp_name)
            
    
    true_output = vocab_idx_hlgt[test_labels[input_idx]]
    if true_output == 'No-symp':
        true_output = 'No life threatening symptom predicted'
    
    symptoms = pd.DataFrame(model_input_text)
    symptoms.columns = ['Input Symptoms']
    
    with col1:
        st.markdown('# Input Symptoms')
        st.table(symptoms)
        
        st.markdown('# Desired Prediction')
        st.markdown(true_output)

sorted_idx = list(np.argsort(model_output[0]))
sorted_idx.reverse()
sorted_idx = np.array(sorted_idx)

sorted_output = [model_output[0][x] for x in sorted_idx]
sorted_cum_prob = []
cp = 0
for x in sorted_output:
    cp += x
    sorted_cum_prob.append(cp)


p_pred = []


if nc_score == 'probability':
    p_temp = 1
    q_p = limits[limits.quant == prob_conf].reset_index().p[0]
    y = 0
    while p_temp >= q_p and (y < len(model_output[0])) :
        p_temp = sorted_output[y]
            
        if p_temp >= q_p:
            p_pred.append(sorted_idx[y])
    
        y += 1
        
elif nc_score == 'cumulative probability':
    p_temp = 0
    q_p = limits[limits.quant == prob_conf].reset_index().cp[0]
    y = 0
    while p_temp <= q_p and (y < len(model_output[0])) :
        p_temp = sorted_cum_prob[y]
            
        if p_temp <= q_p:
            p_pred.append(sorted_idx[y])
    
        y += 1 
else:
    p_temp = 0
    q_p = limits[limits.quant == prob_conf].reset_index().idx[0]
    y = 0
    while p_temp <= q_p and (y < len(model_output[0])) :
        p_temp = y
            
        if p_temp <= q_p:
            p_pred.append(sorted_idx[y])
    
        y += 1


non_0_values = [x for x in model_input.values[0] if x != 0]

unique_pred = [x for x in p_pred if x not in non_0_values]



predicted_sicknesses = []
for x in unique_pred:
    symp_name = vocab_idx_hlgt[x]
    if symp_name == 'No-symp':
        symp_name = 'No life threatening symptom predicted'
    
    predicted_sicknesses.append(symp_name)
    

  
if len(predicted_sicknesses) != 0:
    symptoms = pd.DataFrame(predicted_sicknesses)
    symptoms.columns = ['Predited Symptoms']
else:
    symptoms = pd.DataFrame([],columns=['Predited Symptoms'])


symptoms.index.name = 'Index'

if input_data_select == 'Manual Input':

    st.markdown('# {} Predited MedDRA terms:'.format(len(symptoms)))
    
    st.table(symptoms)
else:
    with col2:
        
        if true_output in symptoms.values:
            st.markdown('# Correct prediction: YES ')
            st.markdown('# Prediction Index: {} '.format(list(symptoms.values).index(true_output)))
        else:
            st.markdown('# Correct prediction: NO ')
            st.markdown('# Prediction Index: {} '.format(list(sorted_idx).index(test_labels[input_idx])))
            
        st.markdown('# {} Predited MedDRA terms:'.format(len(symptoms)))
        
        st.table(symptoms)


