# craNNium


craNNium is a convolutional neural network that identifies patients with mild-to-severe dementia.

## Background
Magnetic resonance imaging (MRI) is one of the primary tools used by physicians to identify dementia. Patients with most types of dementia (Alzheimer's disease, vascular dementia, and frontotemporal lobar degeneration) exhibit structural atrophy in certain regions of the brain. However, identifying the subtle differences in brain structure can be a time-consuming process.

Deep learning methods offer an appealing way to identify dementia patients as a first-pass, but in order to be helpful in a clinical setting, these methods must have high recall and short run-time<sup>[1]</sup>.

In this project, we train a convolutional neural network on T1w MRI data of 200 anonymized patients from the OASIS-3 project. OASIS-3 is a compilation of data obtained through the 30-year WUSTL Knight ADRC program of both cognitively normal adults and adults at various stages of cognitive decline<sup>[2]</sup>. OASIS-3 provides a rich and balanced dataset that may be ideal for deep learning methods.

## Methodology

We select patients from OASIS-3 by identifying equal numbers of patients with Clinical Dementia Rating (CDR) of zero (cognitively normal) and >= 1 (mild-to-severe). We select equally from female and male patients. From each patient, we take one T1-weighted, 3.0T MRI 3-dimensional scan, resulting in a total of 200 total scans, 100 cognitively normal and 100 with some level of dementia.

We pre-pre-process each scan by running each image through [fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat)<sup>[3]</sup>, a common pipeline for processing MRI scans. [fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat) re-orients and crops each scan, aligns the patient head onto a common centerpoint, and extracts the brain.

From each brain, we collect a central slice from a "top-down" orientation. We then pre-process each image by normalizing by the 99th percentile pixel value, to account for noise-related spikes in pixel value. A subset of the pre-processed images are shown below.

![](brains.png)

We run the images through Google's Inception_v4 convolutional neural network architecture <sup>[4]</sup>. Inception_v4 consists of many individual networks run in series, then consolidated at the end. Inception_v4 has been shown to have high accuracy with relatively low computational cost when  run on 2-D image classification. A diagram of the architecture (originally from [1]) is shown below. After the images are run through each step, the outputs are run through a fully-connected dense layer with a `softmax` activation function.

We split our data into 20% test, 80% training. From the training set, we extract 30% of the images for a validation set.

To stop the network from getting caught training on unwanted features in the images, we apply random transformations to each training image, keeping the original as well. We apply rotations, horizontal flips, and zooms to each training image.

Finally, we run the network for 100 epochs, validating against a recall score. We choose recall as our metric since physicians likely want to identify as many true-dementia patients as possible, even if it brings along false positives.

![](inception_v4.png)

## Results

We find that the network is not able to learn well from the images and generalize to the validation set. Regardless of epoch, the recall score of both the training and validation sets hover around 50%, as seen below.

![](performance.png)

Our results suggest a few things.
1. In order to fit the training sample better, more complex pre-processing (e.g, including multiple slices from each patient), or a more specialized model construction+tuning phase may be necessary.
2. In order to generalize better to untrained data, more patient data may be required. Deep learning techniques benefit from larger sample sizes, and 200 images are likely not enough.
3. [1] use a similar method, but in the final layer of their network they add bias-features in the form of patient age, sex, and slice location. These features likely help the network since real-life dementia diagnoses rely on all three parameters.

## Reproducing Results

Since data were provided by OASIS-3, we cannot upload the raw scans. However, we do include the pre-processed test slices.  With the full data set, `src/run_craNNium.py` runs the full processing and training steps. 

The final trained model can be found [here](https://drive.google.com/file/d/1Z4BETLc7Q1GfsbRlBHW2i0vflYsE3iRU/view?usp=sharing). Unpack `model.tar.gz` into the `models/` directory, and run `src/evaluate_model.py`to run on the test data. Test data and labels can be found in `data/images/test/`.

## Acknowledgements

Data were provided by OASIS-3

[1] Bae, J.B., Lee, S., Jung, W. et al. Identification of Alzheimer's disease using a convolutional neural network model based on T1-weighted magnetic resonance imaging. Sci Rep 10, 22252 (2020). https://doi.org/10.1038/s41598-020-79243-9

[2] OASIS-3: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.

[3]  M.W. Woolrich, S. Jbabdi, B. Patenaude, M. Chappell, S. Makni, T. Behrens, C. Beckmann, M. Jenkinson, S.M. Smith. Bayesian analysis of neuroimaging data in FSL. NeuroImage, 45:S173-86, 2009 

[4] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi)