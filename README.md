# Tumor Inflammation Signature Prediction Project
![TIS](data/TIS.jpg)
* Checkpoint inhibitor immunotherapy is a form of cancer treatment that blocks specific proteins on immune cells or cancer cells, known as immune checkpoints. These checkpoints act as brakes to dampen the
  immune system's response to cancer. By using antibodies or other agents to inhibit these checkpoint proteins, these drugs release the brakes on the immune system, enhancing its ability to recognize and attack
  cancer cells. Key checkpoint proteins targeted by these drugs include PD-1, PD-L1, and CTLA-4. Blocking these pathways augments anti-tumor immunity, leading to durable clinical responses in some patients.
  Checkpoint inhibitors such as pembrolizumab, nivolumab, and ipilimumab have demonstrated efficacy against melanoma, lung cancer, and other malignancies. However, only a subset of patients respond, highlighting
  the need for biomarkers to guide patient selection.Ongoing research aims to expand the utility of checkpoint blockade across cancer types and improve patient outcomes through combination therapies.

* Despite promising results, only a small proportion of patients with advanced cancer respond favorably to checkpoint inhibitor immunotherapy. The majority of patients undergo unnecessary exposure to expensive
  yet ineffective drugs with potentially toxic side effects. The medical community is actively investigating biomarkers that can reliably predict patient responsiveness to these biological therapies. To date,
  the only FDA-approved predictor is PD-L1 expression measured by immunohistochemistry (IHC). Another emerging IHC-based test is ImmunoScore, which quantifies immune cell densities within the tumor
  microenvironment. Additional biomarkers under investigation include mismatch repair (MMR) deficiency, microsatellite instability (MSI), and tumor mutational burden (TMB). Though these tests show early promise,
  larger validation studies are needed to confirm their utility for response prediction and guide more personalized, precise use of checkpoint blockade therapy.

* In this project, my partner and I researched a biomarker called TIS - Tumor Inflammation Signature.
  + Developed by Ayers et. al. (2017) - IFN-Gamma-related mRNA profile predicts clinical response to PD-1 blockade. https://www.jci.org/articles/view/91190
  + Formula made of RNA expression of 18 genes (Antigen presentation, chemokine expression, cytotoxic activity, adaptive immune resistance, interferon gamma activity)
  + PSMB10, HLA-DQA1, HLA-DRB1, CMKLR1, HLA-E, NKG7, CD8A, CCL5, CXCL9, CD27, CXCR6, IDO1, STAT1, TIGIT, LAG3, CD274, PDCD1LG2, CD276
  + High score should indicate success of using Pembrolizumab.

# TCGA - The Cancer Genome Atlas (TCGA) database

* TCGA was a landmark cancer genomics program led by the National Cancer Institute and National Human Genome Research Institute from 2006-2015. 
* It systematically mapped the genomic changes in over 20,000 primary cancer samples across 33 tumor types.
* TCGA performed integrated multi-dimensional analysis including whole exome sequencing, copy number, methylation, expression analysis and proteomics on each sample.
* All TCGA data is freely available through the Genomic Data Commons portal and has led to many insights into cancer biology and precision medicine. 

* The idea is to use the TCGA high resolution SVS images of patients, together with their genomic data, and using deep neural network to try predicting the TIS values.
* The main problem when working with high resolution biopsies data of this kind, is how to handle the size of the image. SVS image can be as large as 3 Giga bytes, and it does not fit the current architectures of
  neural networks. What all researchers do is split the images to areas of interest (aoi) in the image, usually these are areas where there are lots of lymphocytes, or where the tumor is located, and train the neural net only on these areas, or patches. Finally, another model is trained in order to combine the results of all the different patches. Along the years various architectures have been developed around these
  two different models, the classifier and the combiner.

# TIS Project Pipelin
![TIS flow][data/TIS flow.jpg]
* The first step is downloading the SVS files from TCGA. This can be done using the manifests. We decided to download BRCA patients. You can use gdc-client.exe executable and manifests files in the code directory as well as the file_download.py script for example.