# The representation of objects across different levels of abstraction in deep convolutional neural networks and human behavior

This repository containes code, data and stimuli for the manuscript "The representation of objects across different levels of abstraction in deep convolutional neural networks and human behavior" submitted to the Journal of Vision.
With the material contained in this repository all of the results in the manuscript can be reproduced. 
 
## Models 

The experiments in the manuscript use three different models, one for each of the experiments: 
- For experiment 1 the VGG-16 model pretrained on ImageNet (which comes with the pytorch package) was used.
- For experiment 2 the VGG-16 model pretrained on the stylized ImageNet dataset (which can be retrieved from: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK) was used. 
- For experiment 3 the VGG-16 model pretrained on ImageNet and finetuned on the ImageNet-Sketch dataset (https://github.com/HaohanWang/ImageNet-Sketch) was used. The model weights for the finetuned model can be retrieved from (https://datashare.rzg.mpg.de/s/vUiWZ3oIV3QikZH).

## Analysis Structure 

The structure of the analysis and the corresponding scripts are: 

- before running the analysis, the activations of the networks need to be extracted using the following script: extract_activations_all_VGG16.py , which can be used to extract the activations for the experimental stimuli for the network VGG16 pretrained on ImageNet, for VGG16 trained on stylized ImageNet, and the VGG16 pretrained on ImageNet but finetuned on ImageNet-Sketch (finetuned using the script finetuning.py) 
- to obtain top-1 accuracies for the stimuli from the experiment separately for each depiction use the script get_top_1_accs_all_VGG16.py (loading models follows the same logic as for extracting activations)
- run_analyis_VGG16.m runs the whole analysis on the activations from one network. The analysis steps follow the logic of the results section in the paper for one experiment and contain: 
  - computing RDMs for different depictions across layers
  - computing the RDM similarity between depictions across layers 
  - computing the RDMs for all depictions combined across layers + computing the MDS-solutions for these RDMs and aligning the MDS-solutions to each other using the procrustes method
  - compute the fit between network RDMs and human behavioral RDMs for each depiction and layer seperately 
  - compute the manmade vs. natural decoding (training and testing on the same depictions) and crossdecoding (training on one depiction but testing on another one) accuracies

 - to compute the statistical comparisons for the results of the analysis the following scripts were used:
    - to compare network accuracies against each other use run_stats_accuracies_all_VGG16.py 
    - to compare human accuracies (obtained using the analyze_label_data.m script followed by the get_stats_for_human_labelling.m script in the \behavioral folder) against each other and against network accuracies use run_stats_accuracies.m
    - to statistically evaluate RDM similarities and compare them against each other use run_stats_RDM_sims.m
    - to compare RDM similarities of different networks use run_RDM_sim_stats_between_networks.m 
    - to statistically evaluate the decoding/cross-decoding accuracies of the network use run_stats_decoding.m 
