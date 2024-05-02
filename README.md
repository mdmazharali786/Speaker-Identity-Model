# Multi Speaker Identity Model
Datasource- Some audios extracted from youtube videos of ipl cricbuz discussion. Audios extracted from below links for training. 
<pre>
{'https://www.youtube.com/watch?v=ag1XMfs1FZw' : 'Simon_Doull Joy_Bhattacharjya Gaurav_Kapur' ,
           'https://www.youtube.com/watch?v=oAUpMOPYBj0' : 'Simon_Doull Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=Mruf9HB3Czo' : 'Harsha_Bhogle Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=GayBRgMK5Ig' : 'Harsha_Bhogle Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=5l5aCBLwkN0' : 'Simon_Doull Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=PrvynAkro7o' : 'Michael_Vaughan Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=k-IKGQTQ4ns' : 'Michael_Vaughan Simon_Doull Gaurav_Kapur',
           'https://www.youtube.com/watch?v=8-QsDzveZWM' : 'Michael_Vaughan, Harsha_Bhogle Gaurav_Kapur',
           'https://www.youtube.com/watch?v=MZEJLxg9vAc' : 'Michael_Vaughan Joy_Bhattacharjya Gaurav_Kapur',
           'https://www.youtube.com/watch?v=XZGPRVC7kH8' : 'Harsha_Bhogle Simon_Doull Gaurav_Kapur',
           'https://www.youtube.com/watch?v=4THplodQNp0' : 'Harsha_Bhogle Simon_Doull Gaurav_Kapur'}
</pre>
These are the only link from which audio is extracted and used for training model. Model can identity these speaker and other than that it indentify as unknown.

## Recommendation for testing the model api.
Download any audio from cricbuzz youtube video having english speech. Do not upload more than 15 min audio response will be more delayed as the server is colab free tier and gdrive to upload download, read etc activity so response time is not setisfactory. It takes aprox 2 minutes to display the result on 15 min audio. Best will be to uplaod up 5 or 10 min audio in any of the format - "mp3, mp4, m4a, wav, flac etc". 

### Instruction on how to run the mmodel.

Model uses <b>GPU</b> for converting audio to vector. So the provided notebook has to be run on google colab. It leverages the colab GPU resource. It also requires to give access to your gdrive. 
Steps-
<ul>
  <li>Open the Initial notebook.ipynb in google colab. </li>
  <li>Connect to GPU runtime in colab notebook.</li>
  <li>Run the cell in the sequence as given.</li>
  <li>Once tha last cell runs and showing like this. <br><img src = "https://github.com/mdmazharali786/Speaker-Identity-Model/assets/75331860/f0ac4f1e-2844-4fbc-957e-dd7d5800314b">
</li>
  <li>Click the link above this cell to run the api of the model. See below. <br> <img src = "https://github.com/mdmazharali786/Speaker-Identity-Model/assets/75331860/b04a63d1-0d10-4a68-8412-5ad9167b9e56">
</li>
</ul>
