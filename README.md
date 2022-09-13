# Data Science

The goals of this project include:

* Application of data science techniques to discover information in two distinct problems (datasets).
* Explore the datasets and select, learn and compare the models for the available data.
* Criticize the results achieved
* Hypothesize causes for the limited performance of the learned models

---

## Data

### [Parkinson Disease](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification) (pd_speech_features.csv)

The data used in this study was gathered from 188 patients with PD at the Department of Neurology in CerrahpaÅŸa Faculty of Medicine, Istanbul University. The control group consists of 64 healthy individuals. During the data collection process, the sustained phonation of the vowel /a/ was collected from each subject with three repetitions.

Various speech signal processing algorithms including Time Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform based Features, Vocal Fold Features and TWQT features have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment.

* Number of features = $754$
* Number of instances = $756 (3 * (188+64))$
* Target: binary (0 = no Parkinson, 1 = has Parkinson)


### [Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype) (covtype.info + covtype.data)

The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data.



---

## Methodology

<ol>
  <li>Statistical Description (Atts x Instances, values Domain, Distributions, Outliers)</li>
  <li>Unsupervised Learning
    <ol>
      <li>Preprocessing</li>
      <li>Pattern Mining</li>
      <li>Clustering</li>
    </ol>
  </li>
  <li>Classification</li>
    <ol>
      <li>Naïve Bayes</li>
      <li>Decision Trees</li>
      <li>kNN</li>
      <li>Random Forests</li>
      <li>XGBoost</li>
    </ol>
</ol>
