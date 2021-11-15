# From photos to sketches - how humans and deep neural networks process objects across different levels of visual abstraction

This repository containes code, data and stimuli for the manuscript "From photos to sketches - how humans and deep neural networks process objects across different levels of visual abstraction" submitted to the Journal of Vision. 
With the material contained in this repository all of the results in the manuscript can be reproduced. 
Link to preprint: https://psyarxiv.com/xg2uy/. Link to paper will be made available upon publication. 

## Models 

The experiments in the manuscript use three different models, one for each of the experiments: 
- For experiment 1 the VGG-16 model pretrained on ImageNet (which comes with the pytorch package) was used.
- For experiment 2 the VGG-16 model pretrained on the stylized ImageNet dataset (which can be retrieved from: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK) was used. 
- For experiment 3 the VGG-16 model pretrained on ImageNet and finetuned on the ImageNet-Sketch dataset (https://github.com/HaohanWang/ImageNet-Sketch) was used. The model weights for the finetuned model can be retrieved from (https://datashare.rzg.mpg.de/s/vUiWZ3oIV3QikZH). Alternatively, the model can be finetuned using the finetuning.py script and the ImageNet-Sketch dataset. 

## Analysis Structure 

The structure of the analysis and the corresponding scripts are: 

- before running the analysis, the activations of the networks need to be extracted using the following script: extract_activations_all_VGG16.py (specify which model you want to use in the script before runnning)
- to obtain top-1 accuracies for the stimuli from the experiment separately for each depiction use the script get_top_1_accs_all_VGG16.py (loading models follows the same logic as for extracting activations)
- run_analyis_VGG16.m runs the whole analysis on the activations from one network. The analysis steps follow the logic of the results section in the paper for one experiment and contain: 
  - computing RDMs for different depictions across layers
  - computing the RDM similarity between depictions across layers 
  - computing the RDMs for all depictions combined across layers + computing the MDS-solutions for these RDMs and aligning the MDS-solutions to each other using the procrustes method
  - compute the manmade vs. natural decoding (training and testing on the same depictions) and crossdecoding (training on one depiction but testing on another one) accuracies
  - compute the fit between network RDMs and human behavioral RDMs for each depiction and layer seperately 

 - to compute the statistical comparisons for the results of the analysis the following scripts were used:
    - to compare network accuracies against each other use run_stats_accuracies_all_VGG16.py 
    - to compare human accuracies (contained in the folder /data) against each other and against network accuracies use run_stats_accuracies.m
    - to statistically evaluate RDM similarities and compare them against each other use run_stats_RDM_sims.m
    - to compare RDM similarities of different networks use run_RDM_sim_stats_between_networks.m 
    - to statistically evaluate the decoding/cross-decoding accuracies of the network use run_stats_decoding.m 

- the folder SVM 42 contains additional code for the SVM classification analysis in the Appendix
    - for this analysis the ImageNet-Sketch images need to be downloaded locally 
    - next the VGG-16 activations for different layers for the ImageNet-Sketch images need to be extracted with the script extract_all_activations_imagenet_sketch_batch_gpu.py
    - next the VGG-16 activations for different layers for the experimental stimuli need to be extracted with the scipt extract_all_activations_VGG16.py
    - finally, a SVM classifier can be trained for every layer separately on the ImageNet-Sketch activations. Importantly, we only use the activations corresponding to the 42 object categories 
      in the experimental stimulus set to train the SVM. After training the SVM, the classifier is evaluated on the drawing and sketch activations separately