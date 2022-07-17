# Flax_text_prediction

The dataset presented here contains argumentative essays written by U.S students in grades 6-12. These essays were annotated by expert raters for discourse elements commonly found in argumentative writing:  

Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis  
Position - an opinion or conclusion on the main question  
Claim - a claim that supports the position  
Counterclaim - a claim that refutes another claim or gives an opposing reason to the position  
Rebuttal - a claim that refutes a counterclaim  
Evidence - ideas or examples that support claims, counterclaims, or rebuttals.  
Concluding Statement - a concluding statement that restates the claims  
Your task is to predict the quality rating of each discourse element. Human readers rated each rhetorical or argumentative element, in order of increasing quality, as one of:  

Ineffective  
Adequate  
Effective  
For more information on the annotation scheme and scoring rubric, please see: Argumentation Annotation Scheme and Descriptions.  

Note that this is a Code Competition, in which you will submit code that will be run against an unseen test set. The unseen test set comprises about 3,000 essays. A small public test sample has been provided for testing your submission notebooks.  

This dataset is a subset of the dataset from the Feedback Prize - Evaluating Student Writing competition. You are welcome to make use of this earlier dataset, if you like.  

Training Data  
The training set consist of a .csv file containing the annotated discourse elements each essay, including the quality ratings, together with .txt files containing the full text of each essay. It is important to note that some parts of the essays will be unannotated (i.e., they do not fit into one of the classifications above) and they will lack a quality rating. We do not include the unannotated parts in train.csv.  

train.csv - Contains the annotated discourse elements for all essays in the test set.  
discourse_id - ID code for discourse element  
essay_id - ID code for essay response. This ID code corresponds to the name of the full-text file in the train/ folder.  
discourse_text - Text of discourse element.  
discourse_type - Class label of discourse element.  
discourse_type_num - Enumerated class label of discourse element.  
discourse_effectiveness - Quality rating of discourse element, the target.  
Example Test Data  
To help you author submission code, we include a few example instances selected from the test set. When you submit your notebook for scoring, this example data will be replaced by the actual test data, including the sample_submission.csv file.  

test/ - A folder containing an example essay from the test set. The actual test set comprises about 3,000 essays in a format similar to the training set essays. The test set essays are distinct from the training set essays.  
test.csv - Annotations for the test set essays, containing all of the fields of train.csv except the target, discourse_effectiveness.  
sample_submission.csv - A sample submission file in the correct format. See the Evaluation page for more details.
