# Speaker Identiity Model
This model identifies speaker of the uploaded audio using a Deep Neural Network Architecture. 

#### About Dataset:
Datasource: https://data.mendeley.com/datasets/zw4p4p7sdh/1
<br>The dataset is used from above resource. The dataset contains total 3000 sample of 150 speakers of Arabic accent and English language. Each speaker's 10 sample digit pronounce and 10 sample of different phrase. Each sample lengths 5-6 seconds.
<p>For preparing train test data entire dataset is splited in train and test folder. See train.txt and test.txt file for list of files used in training an evaluation. None of test data point used in training or validation. </p>


### Feature Extraction
Various wave features extracted which can be relavant to speech recognition. Some of these features are chromagram, Mel-frequency cepstral coefficients (MFCCs), spectral centroid, spectral bandwidth, spectral bandwidth, LPC etc.<br>

#### Model Architecture 
<img src="https://github.com/mdmazharali786/Speaker-Identity-Model/blob/main/model.png">

#### Evaluation
This model performs <b>96%</b> accuracy on test data.

#### Future scope
<ul><li>Unknown speaker like with different accent has to be handled. 
    <li> Multi Speaker Identity in single audio file</ul>
 <p>
Facing challanges of collecting multilable speaker dataset. Complexity in speaker diarization.

### How to Run the Model?
Step1. Download the repository. <br>
Step2. Install all the packages listed in requirement.txt file. <br>
Step3. Go to your CMD and type - python app.py <br>
Step4. Open the local link showing there. <br>
Step5. Select audio file from Test folder and click upload. <br>

You will see the predicted result in the html page. You can verify predicted and true value by the name of file and value suffixed to Speaker_. <br>
For example if file name is 25-2.flac the true label is 25 and if predicted speaker is "Speaker_25" the predicted value is 25.
