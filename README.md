# ML:click_prediction 
By anaylzing online advertisement click data, this project aims to forecast if a click leads to a target behavior such as purchasing.
The dataset consists of time-series of large size categorical features. However, this repository does not contain the original data. I mainly descrbie how to implement categorical embedding by Tensorflow, as well as PCA to describe feature importance.

Repository contents:
- Task_description.pdf: General explaination in this project as a presentation format.
- prediction_result.csv: Prediction result with ID column and probability to be labeled as one.


/Files
- EDAreport.html: EDA report generated by R summarytools library. One should download to visaulize the html result.
- category_embedding_validation.ipynb: the main modeling code with comments. It mainly use Tensorflow
- feature_importance.xlsx: feature importance tables written in Excel. Explaination included.
- pyspark_script.py: script to train logistic regression with Spark on GCP dataproc
- pca_modeling.ipynb: modeling with PCA features
