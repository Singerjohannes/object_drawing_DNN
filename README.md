# The representation of objects across different levels of abstraction in deep convolutional neural networks 

Drawings are universal in human culture and serve as tools to efficiently convey meaning with little visual information. Humans are adept at recognizing even highly abstracted drawings of objects, and their visual system has been shown to respond similarly to different object depictions. Yet, the processing of object drawings in deep convolutional neural networks (CNNs) has yielded conflicting results. While CNNs have been shown to perform poorly on drawings, there is evidence that representations in CNNs are similar for object photographs and drawings. This project is aimed at resolving these seemingly disparate findings by probing the representations and classification decisions of a CNN on object images across different levels of abstraction. 

## Structure 

The repository contains matlab and python code for different parts of the project. The structure of the analysis and the corresponding scripts are: 

- before running the analysis, the activations of the networks need to be extracted using the following script: extract_activations_all_VGG16.py , which can be used to extract the activations for the experimental stimuli for the network VGG16 pretrained on ImageNet, for VGG16 trained on stylized ImageNet (network needs to be loaded with the script load_geirhos_model.py), and the VGG16 pretrained on ImageNet but finetuned on ImageNet-Sketch (ref.) (finetuned using the script finetuning.py) 
- run_analyis_VGG16.m runs the whole analysis on the activations from one network. The analysis steps for one network contain: 
  - computing RDMs for different depictions across layers
  - computing the RDM similarity between depictions across layers 
  - computing the RDMs for all depictions combined across layers + computing the MDS-solutions for these RDMs and aligning the MDS-solutions to each other using the procrusted     method
  - compute the fit between network RDMs and human behavioral RDMs (obtained using the analyze_triplet_results60.m scripts in the \behavioral folder) for each depiction and layer seperately 
  - compute the manmade vs. natural decoding (training and testing on the same depictions) and crossdecoding (training on one depiction but testing on another one) accuracies
 
 - to analyse the classification accuracies for the networks the following scripts were used: 
    - to obtain the top-1 accuracies for one network the script get_top_1_accs_all_VGG16.py can be used (if the geirhos model needs to be evaluated the model must be loaded using the load_geirhos_model.py script) 
    
 - to compute the statistical comparisons for the results of the analysis the following scripts were used:
    - to compare network accuracies against each other use run_stats_accuracies_all_VGG16.py 
    - to compare human accuracies (obtained using the analyze_label_data.m script followed by the get_stats_for_human_labelling.m script in the \behavioral folder) against each other and against network accuracies use run_stats_accuracies.m
    - to statistically evaluate RDM similarities and compare them against each other use run_stats_RDM_sims.m
    - to compare RDM similarities of different networks use run_RDM_sim_stats_between_networks.m 
    - to statistically evaluate the decoding/cross-decoding accuracies of the network use run_stats_decoding.m 


### TODO: create selection of behavioral RDMS and label data for the stimuli used in the experiment (42 instead of 60) and remove the selection part in the git scripts 
###       check necessary files for the analysis steps -> everything needs to be reproducible from scratch 
###       maybe integrate the load_geirhos_model.py script into the extract_activations... and get_top1... scripts
