# ArtSeg and VasMorph

## *Applying Machine Learning to Assist in the Morphometric Assessment of Brain Arteriosclerosis Through Automation*

Jerry J. Lou*, Peter Chang, Kiana D. Nava, Chanon Chantaduly, Hsin-Pei Wang, William H. Yong, Viharkumar Patel, Ajinkya J. Chaudhari, La Rissa Vasquez, Edwin Monuki, Elizabeth Head, Harry V. Vinters, Shino Magaki, Danielle J. Harvey, Chen-Nee Chuah, Charles S. DeCarli, Christopher K. Williams, Michael Keiser, Brittany N. Dugger*

*Corresponding authors: Jerry J. Lou and Brittany N. Dugger

*Published in Free Neuropathology on June 2, 2025*

**[Publication](https://www.uni-muenster.de/Ejournals/index.php/fnp/article/view/6387)** | **[Download Models](https://zenodo.org/records/14955269?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUyZDYxNDNiLThlM2YtNDk4Yi05ZWMwLTRmYWZlM2UzYzk0NCIsImRhdGEiOnt9LCJyYW5kb20iOiIwZWFmMTZiMTliOTFlYmFhY2Y1NDRhODVjZjZjZGZlNSJ9.zsZ-DW-_8-tBOc-6PK0XHYZOM7rOL6fG-evXOBE0IPMYVA8hdTGorCMkoWJR70xJMSI68F2fSv8COx5gt7K6ZA)** | **[Datasets](#datasets)** | **[Cite](#reference)** 

**Abstract**:\
<img src="Figures/Blood_vessel_chip.png" width="300px" align="right" />
Objective quantification of brain arteriolosclerosis remains an area of ongoing refinement in neuropathology, with current methods primarily utilizing semi-quantitative scales completed through manual histological examination. These approaches offer modest inter-rater reliability and do not provide precise quantitative metrics. To address this gap, we present a prototype end-to-end machine learning (ML)-based algorithm – Arteriolosclerosis Segmentation (ArtSeg) followed by Vascular Morphometry (VasMorph) – to assist persons in the morphometric analysis of arteriolosclerotic vessels on whole slide images (WSIs). We digitized hematoxylin and eosin-stained glass slides (13 participants, total 42 WSIs) of human brain frontal or occipital lobe cortical and/or periventricular white matter collected from three brain banks (University of California Davis, Irvine, and Los Angeles Alzheimer’s Disease Research Centers). ArtSeg comprises three ML models for blood vessel detection, arteriolosclerosis classification, and segmentation of arteriolosclerotic vessel walls and lumens. For blood vessel detection, ArtSeg achieved area under the receiver operating characteristic curve (AUC-ROC) values of 0.79 (internal hold-out testing) and 0.77 (external testing), Dice scores of 0.56 (internal hold-out) and 0.74 (external), and Hausdorff distances of 2.53 (internal hold-out) and 2.15 (external). Arteriolosclerosis classification demonstrated accuracies of 0.94 (mean, 3-fold cross-validation), 0.86 (internal hold-out), and 0.77 (external), alongside AUC-ROC values of 0.69 (mean, 3-fold cross-validation), 0.87 (internal hold-out), and 0.83 (external). For arteriolosclerotic vessel segmentation, ArtSeg yielded Dice scores of 0.68 (mean, 3-fold cross-validation), 0.73 (internal hold-out), and 0.71 (external); Hausdorff distances of 7.63 (mean, 3-fold cross-validation), 6.93 (internal hold-out), and 7.80 (external); and AUC-ROC values of 0.90 (mean, 3-fold cross-validation), 0.92 (internal hold-out), and 0.87 (external). VasMorph successfully derived sclerotic indices, vessel wall thicknesses, and vessel wall to lumen area ratios from ArtSeg-segmented vessels, producing results comparable to expert assessment. This integrated approach shows promise as an assistive tool to enhance current neuropathological evaluation of brain arteriolosclerosis, offering potential for improved inter-rater reliability and quantification.

## What is ArtSeg and VasMorph?

Together, ArtSeg and VasMorph compose a prototype end-to-end ML-based pipeline that can assist neuropathologists in the morphometric analysis of arteriolosclerotic vessels on whole slide images. The ML component Arteriolosclerosis Segmentation (ArtSeg) receives WSIs as input and outputs segmentations of arteriolosclerotic blood vessel walls and lumens, which are in turn input into the non-ML component Vascular Morphometry (VasMorph) that outputs quantitative metrics for the sclerotic index, vessel wall thickness, and vessel wall to lumen area ratio.

**Why use ArtSeg and VasMorph?**  

> Objective quantification of brain arteriolosclerosis remains an area of ongoing refinement in neuropathology, with current methods primarily utilizing semi-quantitative scales completed through manual histological examination. These approaches offer modest inter-rater reliability and do not provide precise quantitative metrics. ArtSegand VasMorph show promise as an assistive tool to enhance current neuropathological evaluation of brain arteriolosclerosis, offering potential for improved inter-rater reliability and quantification.

## Algorithm Architecture and Walkthrough

### Phase 1: Arteriolosclerosis Segmentation (ArtSeg)

ArtSeg comprises four algorithms that complete four sequential steps (Figure 2). After WSI tiling, the first step (Phase 1a) is to detect blood vessels and keep tiles that contain a blood vessel and discard those that do not. The second step (Phase 1b) is to recursively shift tiles until the detected blood vessel appears at the center of the tile. The third step (Phase 1c) is to keep tiles that contain a blood vessel with arteriolosclerosis and discard those that do not. The fourth step (Phase 1d) is to segment the walls and lumens of blood vessels with arteriolosclerosis.

![Insert Figure Here](Figures/Figure%202-%20ArtSeg%20overview.jpg)
**Figure 2: The ML pipeline received WSIs of H&E-stained cortical and/or periventricular white matter brain tissue as input.** Each WSI was tiled into tens of thousands of (512 x 512) pixel image tiles. (Phase 1a) The blood vessel detection ML model sorted tiles into those with blood vessels and those without. (Phase 1b) Object of interest Recursive Centering Algorithm (ORCA) generated new tiles centered onto the detected blood vessels. (Phase 1c) An arteriolosclerosis classification model separated tiles with centered blood vessels into those with arteriolosclerosis and those without. (Phase 1d) A modified Attention U-Net segmented the arteriolosclerotic vessel walls and lumens to produce the final output. All models within ArtSegtake advantage of fixed ImageNet pretrained parameters from Google’s EfficientV2L to extract low-level features prior to learning vessel specific features de-novo. 

<ins>Phase 1a</ins>: Blood vessel detection. The blood vessel detection neural network consisted of an Attention U-net architecture33 with an encoder composed of an EfficientNetV2L34 backbone with five semi-trainable convolution layers followed by two fully trainable convolution layers and a decoder composed of seven trainable convolution layers generated through the concatenation of a 2D transpose convolution of the prior layer and an attention gate33 that filters features propagated from the skip connections.

<ins>Phase 1b</ins>: Blood vessel centering. Blood vessel centering was achieved by a custom recursive algorithm – Object of interest Recursive Centering Algorithm (ORCA) – wrapping the blood vessel detection neural network (Figure 3). The wrapper algorithm inputs raw (512 x 512) tiles into the blood vessel detection neural network, which segments blood vessels. Subsequently the wrapper algorithm generates a new (512 x 512) tile with shifted boundaries such that the detected blood vessel resides closer to the center of the tile. This process is repeated until the detected blood vessel lies in the center of the final output tile (Figure 3). ORCA detects when the patch has been centered onto the blood vessel(s) by comparing the coordinates of the new patch with shifted boundaries to the original input patch; if the shift in boundaries is less than a preset threshold, then the patch is considered blood vessel(s) centered. The average runtime per WSI was approximately 37 minutes.

![Insert Figure Here](Figures/Figure%203-%20ORCA.JPG)\
**Figure 3: Object of interest Recursive Centering Algorithm (ORCA).** Starting at step (1), the algorithm inputs a raw (512 x 512) image tile through the embedded blood vessel detection model which produces an output segmentation (2). (3) ORCA creates a new patch from the input WSI used modified shifted coordinates based on the previous segmentation. (4-6) Steps 1 through 3 are repeated until the vessel is centered. 

![Insert Figure Here](Figures/centered_vessel.jpg)\
**Figure 8: Results of vessel centering by ORCA.** (a) WSI tiling generates tiles that often contain blood vessel(s) at the tile edge, with some vessel cropped (arrow). (b) ORCA generates new tiles with the detected blood vessel(s) located in the center.

<ins>Phase 1c</ins>: Arteriolosclerosis classification. The arteriolosclerosis classification neural network consisted of an EfficientNetV2L34 backbone with five semi-trainable convolution layers topped by two fully trainable convolution layers, followed by three dense layers.

<ins>Phase 1d</ins>: Arteriolosclerotic vessel segmentation. The arteriolosclerotic vessel segmentation network used the same architecture as the model for blood vessel detection.

![Insert Figure Here](Figures/ArtS_neural_network_architectures.jpeg)\
**Supplementary Figure 3: Detailed architecture of the classification and segmentation ML models.** “Conv2D” operations consisted of separable 2D convolutions with kernel size = (3, 3), stride = 1, and padding = 1 followed by batch normalization and Leaky ReLU. For subsampling, we utilized separable 2D convolutions with kernel size = (3, 3), stride = 2, and padding = 1. (a) The arteriolosclerosis classification model comprised an EfficientNetV2L34 backbone with five semi-trainable convolution layers followed by two fully trainable convolution layers and three dense fully connected layers. The semi-trainable Conv2D layers involved concatenating the subsampling of the previous layer and an EfficientNet2VL layer with frozen parameters followed by a “Conv2D” operation. The model contained 11 layers with 118,123,101 total parameters, 95,789,413 trainable parameters, and 22,333,688 untrainable parameters. (b) The blood vessel detection and arteriolosclerotic vessel segmentation models comprised an Attention U-net architecture33 with an encoder composed of an EfficientNetV2L34 backbone with five semi-trainable convolution layers followed by two fully trainable convolution layers and a decoder composed of seven trainable convolution layers generated by concatenating the 2D transpose convolution of the previous 

### Example Results

![Insert Figure Here](Figures/Arts.Mild.Arteriolosclerosis.jpg)\
**Figure 9: Example segmentation results for blood vessels with mild arteriolosclerosis as classified by a neuropathology fellow (JL) and annotated by non-experts (KN, HSW, JL)**
Six example instances with input image, human annotation mask, model segmentation output, and an overlap image of input image and model segmentation are shown here. The blood vessels shown here were classified as having arteriolosclerosis by a human annotator. (a, b) Example instances of good model performance. (c, d) Example instances of intermediate model performance. (e, f) Example instances of poor model performance.

![Insert Figure Here](Figures/ArtS.Moderate.Arteriolosclerosis.jpg)\
**Figure 10: Example segmentation results for blood vessels with moderate arteriolosclerosis as classified by a neuropathology fellow (JL) and annotated by non-experts (KN, HSW, JL)**
Six example instances with input image, human annotation mask, model segmentation output, and an overlap image of input image and model segmentation are shown here. The blood vessels shown here were classified as having arteriolosclerosis by a human annotator. (a, b) Example instances of good model performance. (c, d) Example instances of intermediate model performance. (e, f) Example instances of poor model performance.

![Insert Figure Here](Figures/ArtS.Severe.Arteriolosclerosis.jpg)\
**Figure 11: Example segmentation results for blood vessels with severe arteriolosclerosis as classified by a neuropathology fellow (JL) and annotated by non-experts (KN, HSW, JL)**
Six example instances with input image, human annotation mask, model segmentation output, and an overlap image of input image and model segmentation are shown here. The blood vessels shown here were classified as having arteriolosclerosis by a human annotator. (a, b) Example instances of good model performance. (c, d) Example instances of intermediate model performance. (e, f) Example instances of poor model performance.

![Insert Figure Here](Figures/ArtS.Difficult.Instances.jpg)\
**Figure 12: Example segmentation results for challenging instances as classified by a neuropathology fellow (JL) and annotated by non-experts (KN, HSW, JL)**
Six example instances with input image, human annotation mask, model segmentation output, and an overlap image of input image and model segmentation are shown here. The blood vessels shown here were classified as having arteriolosclerosis by a human annotator. (a) A corpora amylacea was mistaken for a vessel lumen by ArtSeg. (b) ArtSegmisidentified a non-arteriolosclerotic vessel as having arteriolosclerosis and a target for segmentation. (c) Conversely, a vessel with arteriolosclerosis is misidentified as a non-arteriolosclerotic vessel and omitting for segmentation. (d) ArtSegfails to identify moderate hyaline vessel wall thickening and lumen stenosis. (e) Image tiles with clusters of blood vessels present a particular challenge to ArtSegand VasMorph. (f) Vessels within leptomeninges are mistakenly segmented by ArtSeg.

### Walkthrough

> **Note:** Replace all `#Insert path` placeholders with the appropriate file paths for your setup.

#### Step 1: Organize WSIs
Organize whole slide images (WSIs) intended for analysis into a folder.

WSIs used in this study are accessible via the below Zenodo records.

Links under construction.

#### Step 2: Patch Extraction with ORCA
Use `ORCA.py` to obtain patches centered on detected blood vessels.

Input WSIs into ORCA through fnames, the list of paths to your WSIs that ORCA will process.

```python
if __name__ == '__main__':
    # --- Autoselect GPU
    gpus.autoselect()

    fnames = [
        '#Insert paths to WSIs'
    ]
```

#### Step 3: Arteriolosclerosis Classification
Input patches centered onto detected blood vessels into the arteriosclerosis classification model `Unfrozen_EfficientNetV2L_v0.64` to separate patches with arteriolosclerotic blood vessels from those without.

*Set up the environment*
```python
import glob, numpy as np, pandas as pd, tensorflow as tf, os, scipy, random
from tensorflow.keras import Input, Model, layers

import sys
sys.path.append('#path to folder containing jerry_utils.py')
from jerry_utils import load_dataset_v1
```

*Load the model*
```python
name = 'Unfrozen_EfficientNetV2L_v0.64'
model_path = '#Insert path to model hdf5 file'
model = tf.keras.models.load_model(model_path)
```

*Load the dataset*
```python
input_path = glob.glob('#path to where you saved the blood vessel centered patches generated by ORCA')
input = load_dataset_v1(input_path)
```

*Run inference*
The prediction output of the arteriolosclerosis model is output as the variable `pred`.

- The  labels for the classification output are:
  - `0`: No arteriolosclerosis  
  - `1`: Arteriolosclerosis

```python
for x, y in external:
    logits = model.predict(x)
    m = np.argmax(logits, axis=-1)
    pred = tf.cast(np.squeeze(m), tf.uint8)
```

Save patches with and without arteriolosclerotic vessels into separate folders.

#### Step 4: Arteriolosclerotic Vessel Wall and Lumen Segmentation
Use `Attention_Unfrozen_EfficientNetV2L_data.test.v2_v0.7_try` to segment arteriosclerotic vessel walls and lumens.

*Set up the environment*
```python
import glob, numpy as np, pandas as pd, tensorflow as tf, os, scipy, random
from tensorflow.keras import Input, Model, layers
from scipy.spatial.distance import directed_hausdorff

import sys
sys.path.append('#path to folder containing jerry_utils.py')
from jerry_utils import load_dataset_v1
import jerry_losses, jerry_metrics
```

*Load the model*
```python
name = 'Attention_Unfrozen_EfficientNetV2L_data.test.v2_v0.7_try'
model_path = '#Insert path to model hdf5 file'

custom_objects = {
    'dice_all': jerry_metrics.dice_metric(cls=1),
    'hausdorff_all': jerry_metrics.hausdorff_metric(cls=1),
    'focal_dice_like_loss_multiclass_weighted': jerry_losses.focal_dice_like_loss_multiclass_weighted
}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
```

*Load the dataset*
```python
input_path = glob.glob('#path to where you saved the patches with arteriolosclerotic blood vessels')
input = load_dataset_v1(input_path)
```

*Run inference*
- The mask labels for the segmentation output are:
  - `0` for background  
  - `1` for vessel (both wall and lumen)  
  - `2` for vessel lumen only  

- The segmentation output is stored in the variable `pred`.
  
```python
for x, y in input:
    logits = model.predict(x)
    pred = np.argmax(logits, axis=-1)
```

Save each segmentation output into a specific folder for use in VasMorph.

## Phase 2: Vascular Morphometry (VasMorph)

For each arteriolosclerotic blood vessel analyzed, VasMorph outputs the sclerotic index, vessel wall thickness, and vessel wall to lumen area ratio, which have previously been used as an indicator of the degree of vascular stenosis. The final sclerotic index and vessel wall thickness outputs include the median, mean, standard deviation, minimum, and maximum of sclerotic indices and vessel wall thicknesses calculated in a 360-degree rotation around the center of the blood vessel lumen. We pilot test two methods for calculating blood vessel wall thickness: a radii-based method and a tangent line-based method. The tangent line-based method appears to generate metrics with greater proximity to human expert measurements, and so we only include this method here.

![Insert Figure Here](Figures/SI.Calculation.jpg)\
**Figure 4: Sclerotic Index Calculation.** An image patch (a) centered onto an arteriolosclerotic blood vessel is input into the segmentation model of ArtSegwhich outputs the vessel wall and lumen segmentation (b). VasMorph measures the internal diameter (D_(i(θ))) and external diameter (D_(e(θ))) of the vessel to calculate the sclerotic index (〖SI〗_θ) at each degree angle over a 180-degree rotation in a half circle around the centroid35 of the lumen segmentation to obtain a set of sclerotic indices (b and c). The output of VasMorph is the median, mean, standard deviation, minimum, and maximum of this set of vessel wall thicknesses. VasMorph find the center of the lumen segmentation.

![Insert Figure Here](Figures/Vessel.wall.thickness_tangent.line.based.jpg)\
**Figure 5: Vessel Wall Thickness Based on a Line Tangent to Every Point on the Lumen Contour.** An image patch (a) centered onto an arteriolosclerotic blood vessel is input into the segmentation model of ArtSegwhich outputs the vessel wall and lumen segmentation (b). VasMorph measures the vessel wall thickness at each point in the lumen contour to obtain a set of thicknesses, where thickness at point alpha (T_α)  is defined as the distance between a line tangent to point alpha (L_α) on the curve of the lumen contour and the external contour of the vessel (E): T_α=E- L_α (b and c). VasMorph then outputs the median, mean, standard deviation, minimum, and maximum of this set of vessel wall thicknesses.

## Walkthrough

> **Note:** Replace all `#Insert path` placeholders with the appropriate file paths for your setup.

### Step 1: Load segmented vessels
```python
path = '#path to ArtSeg-segmented arteriolosclerotic vessels'
test = tf.data.Dataset.load(path)
```

### Step 2: Change save path for VasMorph output
```python
df = pd.DataFrame(metrics_dict)
df.to_csv('#save_path.csv', index=False)
```

### Step 3: Run VasMorph

## Datasets

### ArtSeg-VasMorph de-identified WSIs
- **[ArtSeg-VasMorph WSIs Part 1](https://zenodo.org/records/15692208?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA0NDgwM2MxLWIzOGUtNDE1Ny05MWFhLTdmOTc3OTMwOGYwNCIsImRhdGEiOnt9LCJyYW5kb20iOiJkNGNkOTI5NzhjYzJjY2UzMzk5NDcwNGM1YmFjYmU4OCJ9.xGTmZ3AIHJyvUtnR-4ao7I7cLT4ZIDW2prmoYSoHe-M0RbbLa50Bur56VNO5oWiOufLBZNJOO6rW6-wtVD84aA)**
- **[ArtSeg-VasMorph WSIs Part 2](https://zenodo.org/records/15692396?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI0ZThiNTRlLTExNjMtNDRjZi04ZmQ5LTJmODhmZjFhMWM5NCIsImRhdGEiOnt9LCJyYW5kb20iOiIwZTZhN2VjNDUzZTRkYTI0Y2IxODUwYTYxNzMyMWRlMyJ9.qV3zkppaMy3UOvrbrb1XINUibYnT1mRbOzt4WAYr-OWUrrks32WWcNQZovPgCPvlUNttBEJH6wkj5jtd9_1pxg)**
- **[ArtSeg-VasMorph WSIs Part 3](https://zenodo.org/records/15694655?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQwMWMzNmY3LTRmZWMtNGYwMS05YjI3LTA5MzQxZjJlZWQzNCIsImRhdGEiOnt9LCJyYW5kb20iOiJjMjE5MWE5OGYxYTMyZDkzYWMxODE1MzJjNDk3NzQ4MCJ9.c3nWT3cSmduarGNah0jAYflDJMr5PMUlAgfL6jKmQFLXc9Q28ck3Bst0dTSvVJ_Q9j3UA-D-kjTkmgSVjFEVww)**
- **[ArtSeg-VasMorph WSIs Part 4](https://zenodo.org/records/15694693?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0NGJlNjk3LTU0NDMtNDU4ZC1hZDEyLWNjOTNmY2NiMTg3ZCIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjA1NjU1YmE4MWZhNGU1ZDdlNTE4YjEyODU3NDc2MCJ9.BjBMkUIJT-W7124waieEHEznbuLCX5SZ7sgETDWz0AOwq02ikAXiU2VdWhFa_PD8IzvuY8fNaVn5dww-oMNZ3w)**
- **[ArtSeg-VasMorph WSIs Part 5](https://zenodo.org/records/15748082?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcyNGZiZjRhLTcyY2YtNDhhNi05NWQwLWU2OTlkODU4ODIzZCIsImRhdGEiOnt9LCJyYW5kb20iOiIzNTJmOTUyYWU5ZGYxMmI2YTU4NjBkNGVmZTQ2OTVkMSJ9.5JFAG97YmT0dasZwR9iMDoAdZdsyNpHyaB9PcYwALHjgyhkfW2K1qdxI7hwB_A8gbvD5i3Xw7y5Ah9xU-rVzpg)**

### VasMorph 
- **[VasMorph Test Dataset](https://zenodo.org/records/14885417?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU3OGY0MGYxLTY2NzUtNGNjYS05NzNhLTY2NDllMjcyODJlMyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ODQ2NWM1ODM5YzIyMDNjZGFkMjVlMmFiOGU1ODkyNyJ9.qj1_vt2Cfe9ly7Jj5yz5y4ynWUVnu_shwRex-LuRKyJzQYbAckhi5baZ-fat_4JzAshjYmbheqD9mON0eRnXkg)**

## License and Terms of Use
The models and associated code are released under the CC BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the ArtSegmodels and their derivatives, which include models trained on outputs from the ArtSegmodels or datasets created from the ArtSegmodels, is prohibited and requires prior approval. Any commercial use, sale, or other monetization of VasMorph is prohibited and requires prior approval. By downloading the models, you agree not to distribute, publish or reproduce a copy of the models. If another user within your organization wishes to use the ArtSegmodels, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying models. If you are a commercial entity, please contact the corresponding authors.

## Reference
Lou, J. J., Chang, P., Nava, K. D., Chantaduly, C., Wang, H.-P., Yong, W. H., Patel, V., Chaudhari, A. J., Vasquez, L. R., Monuki, E., Head, E., Vinters, H. V., Magaki, S., Harvey, D. J., Chuah, C.-N., DeCarli, C. S., Williams, C. K., Keiser, M., & Dugger, B. N. (2025). Applying machine learning to assist in the morphometric assessment of brain arteriolosclerosis through automation. Free Neuropathology, 6, 12. https://doi.org/10.17879/freeneuropathology-2025-6387

