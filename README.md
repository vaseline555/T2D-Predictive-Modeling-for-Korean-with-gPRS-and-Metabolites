# Predictive Modeling of Type 2 Diabetes for South Korean with Genome-wide Polygenic Risk Scores (gPRS) and Serum Metabolites
* This is the source code of the paper: [**Prediction of Type 2 Diabetes using Genome-wide Polygenic Risk Score and Metabolic Profiles: A Machine Learning Analysis of Population-based 10-year Prospective Cohort Study**](https://#) published in #### (2022).


## Abstract
### Background
Previous works on predicting type 2 diabetes were mostly on the Western population, integrating clinical and genetic factors. We additionally incorporate genome-wide polygenic risk score (gPRS) and serum metabolites for type 2 diabetes risk prediction on the Asian population. 
### Methods
Participants from the Korean Genome and Epidemiology Study (KoGES) Ansan-Ansung cohort (n=1,425) were included for type 2 diabetes risk prediction. For gPRS, genotypic and clinical information from KoGES Health Examinee (n=58,701) and KoGES Cardiovascular Disease Association Study (n=8,105) were included. For linkage-disequilibrium, 239,062 genetic variants were used to determine the gPRS, and metabolites are selected through Boruta algorithm. The performances of logistic regression and random forest (RF)-based machine learning model were evaluated by bootstrapped cross-validation. Finally, associations of gPRS and metabolites with the homeostatic model assessment of beta-cell function (HOMA-B) and insulin resistance (HOMA-IR) were further estimated. 
### Findings
In follow-up periods (8·3±2·8 years), 331 participants (23·2%) were diagnosed with type 2 diabetes. The area under the curves of RF-based model with demographic and clinical factors, gPRS-added model, and gPRS and metabolites-added model were 0·844, 0·876, and 0·883, and the latter two models improved the net classification by 11·7% and 4·2%. While gPRS was significantly associated with HOMA-B, most metabolites had a significant association with HOMA-IR. 
### Interpretation
Embodying both gPRS and metabolites enhanced type 2 diabetes risk prediction by capturing distinct etiology of type 2 diabetes development. RF-based model with clinical factors, gPRS, and metabolites

## Dataset
* The Korean Genome and Epidemiology Study (KoGES) Ansan-Ansung dataset used in this study is **not** opened to the public. 
* Dataset is only accessible with consent from Korea Biobank. (http://koreabiobank.re.kr; TEL: +82-1661-9070)

## Commands
* For detailed description of command arguments, please check `main.py` or run command `python main.py -h`.
```
sh run.sh
```

## Contact
If you have any questions, feel free to contact
- Seok-Ju Hahn ([seokjuhahn@unist.ac.kr](mailto:seokjuhahn@unist.ac.kr))
- Suhyeon Kim ([suhyeonkim@unist.ac.kr](mailto:suhyeonkim@unist.ac.kr))
