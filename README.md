

# General-Search-Index
General Search Index (GSI) pulls data from google trend and creates a search index for a topic of the users choice.

Note that the project is still being worked on so this is not the final verion yet.

To use the GSI please follow these steps: </br></br>
&emsp; Step 1. Download Google trend API and matplotlib or simply uncomment the fist two lines in the GSI.py file: </br>

&emsp; > !pip install pytrends </br>
&emsp; > !pip install matplotlib </br>

&emsp; Step 2.Import: </br>
&emsp; > import GSI </br>
&emsp; > import matplotlib.pyplot </br>

&emsp; Note: for ease of use put the GSI.py file in the same folder as the file you are calling the GSI from.

&emsp; Step 3. Use the following command to use the search index: </br>
&emsp; >  GSI.search_index_tinator( 'your search term' , ['google location code'] , model selection )


</br>
</br>
The idea was to have different set of models to create the search index ie. model selection =  1. for PCA, 2. weighted average etc. However, only PCA works for now, and im still making improvements to the model interms of dealing with surge in search level in short periods and low volume search data... but If you have any ideas or suggestions with google data transformation im all ears! 

</br>
</br>
Do take a look at the GSI example.ipynb file if you want to get an idea of how the search index works.
