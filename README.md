# Lyrical Analysis by Popularity 
### Contributors
Tanner Sims: u1159642, tannerjeffreysims@gmail.com
<br>
Ethan Burrows: u1102916, ethandburrows@gmail.com
<br>
Lass Omar: u1179231, u1179231@utah.edu
<br>

## Introduction 
Can AI beat the music industry? The music industry is heavily invested in decoding what makes a popular song, and the ability to predict popularity would be invaluable. But is it doable? What about a song makes it popular. Modern deep learning models may be able to answer that question.

We believe that lyrics are an important aspect of a song, and that a portion of the popularity of a song will stem from the lyrics. Our project uses modern transformer networks and RNNs to predict the popularity of a song, given only the lyrical content. Further, we will attempt to generate specific models which predict the popularity of a song by lyrics within genres. By analyzing genres and their lyrics, we can make predictions about the importance of lyrics within that genre. Finally, we hope to identify good and bad sections within the lyrics using the fully trained models.

### Background
The past 5 years have seen an explosion in the capabilities of deep networks in the application of natural language tasks. These models account for all of the recent advances in technology like Siri, Amazon’s Alexa, and even Google translate. After moving to deep learning, Google translate saw a sudden and large improvement, especially with difficult to translate or obscure languages.

Recent models, such as Google’s BERT and the Open-AI GPT-2, have consistently passed turing tests over a variety of tasks, demonstrating remarkable flexibility in modeling meaning-dependent problems. In fact, these models are so effective that concerns have even been raised about the ethics of releasing such models publicly due to their ability to mass produce believable fake news and other misinformation. Simply put, these models are rapidly approaching (or exceeding) human performance on all language tasks. 

In addition to translation and generative tasks, these models have also been shown to perform well on categorization and regression tasks, such as sentiment analysis or forecasting problems. Because of this success in literature, we expect that such models will perform well predicting the popularity, provided sufficient information within the lyrics text. We hope to see that the model is able to predict song popularity with reasonable accuracy, and we would like to use this measure to analyze songs in order to identify strong and weak lyrical sections.

## Method
### Data Aquisition
Two primary pieces of information are necessary to perform the analysis for each song in the data set: popularity measures and song lyrics. Since we are performing a regression, we need to use a measure that is quantitative and as continuous as possible. Measures like top Billboard position, are out of the question; while we can interpret ordered data like this as quantitative to some extent, there is no information about how much more popular the best song is relative to the second best. In fact, if two songs were released at different times, both of them may have hit the top. With this in mind, we chose to use the number of listens a song has on the streaming platform Spotify, since this data was available to the general public, and is reasonably continuous. Although the number of listens is also not a continuous variable as there can be no fractional values, the numbers tend to be very large, so the data is granular relative to its range. Additionally, it is very unlikely that two songs will have the same number of listens. Further, Spotify also contains a measure called hotness which, (according to their documentation) is proprietary time weighted measure of listens, among other undisclosed values. 

While the Spotify hotness value could have also been a valid choice for a measure of popularity, we had two reservations in using this measure. First, as mentioned previously, the method by which they calculate the hotness is proprietary, so it is unclear if the measure includes factors we wouldn’t want to consider. Second, since hotness is time weighted, the release date of a song is introduced as a confounding factor; release date might not be reflected in lyrical content, so older songs would naturally suffer as a result. Therefore, we believe the model has a greater chance of explaining variability in lyrical content using the number of listens instead. However, despite our misgivings regarding the hotness value, we decided to gather the hotness data on the songs as it is readily accessed and provides an additional data point; including the hotness value allows us to see how well it correlates with the number of listens as a measure of popularity, thus further validating its use.

Unfortunately, Spotify does not contain the lyrics for a song, and so we sought an alternate source for that data. To obtain the song lyrics, we chose to use the site Genius, since it houses a large library of various songs and lyrics. Ideally, the larger our source of data, the more it begins to resemble songs as a whole. This is important for maintaining the generalizability of our final model and results. In addition to lyrics, we would also like to know the title of the song, the artist, and the track’s genres. Luckily, Genius also contains genre tags for every song, so all of this desired song content data is readily and easily accessed from one location.

Both Genius and Spotify are large companies with expansive databases and services, so naturally they both have extensive and well documented APIs. Our initial hope was that all of the necessary data could be obtained by querying each API. As it turns out, the data that we wanted to collect, was not exposed in the API for either services. Genius APIs do not give lyrics or genre, and Spotify APIs do not provide listens. As a result, we had to turn to scraping. Genius is relatively easy to scrape. Their service is hosted entirely in html, and every song has its own page. The lyrics are available immediately, and the genre tags are hidden away in some metadata json that are easy to access. Spotify, however, is not that simple. It functions more like an application; even the web listening service requires a login. Spotify content is written entirely in JavaScript, which is far more difficult to extract data from. Fortunately, there is a github user by the name of evilarceus who has already solved this problem. His project interfaces with the spotify client application, and returns a json file with the number of listens and hotness value. For more information, please visit his [repo](https://github.com/evilarceus/Spotify-PlayCount).
At this point, with all of the necessary data gathering components in place, we needed a way of ensuring that our data set was representative of the music industry as a whole. Beyond collecting as large a dataset as feasible, we needed to select our data points randomly. Fortunately, Genius uses integers between 0 and 4,000,000 to identify each of the songs in their database. Thus, to select a song, we generated a random Genius ID between 0 and 4,000,000. Using the Genius API, we then found the song and scraped its lyrics and genre. Finally, we used the Spotify search API to find the same song within the Spotify database, and then scrape the number of listens and hotness value. This entire scraping process was performed by a stand-alone python script run in the command line. The procedure is described in detail in the jupyter notebook found [here](Data%20Collection/Project%20Data%20Aquisition.ipynb), and scraping script found [here](Data%20Collection/scrapingscript.py). In total, we collected 118,000 unique song data points. The scraping was done over the course of about 2 weeks, on and off, at a rate of about 1200 songs per hour.

### Data Cleanup
Once the complete dataframe of songs were obtained, in order to analyze the songs based on their lyrics, we needed to remove those that were either non-english or were simply instrumentals. Our model will be based off of english words, and including songs that either contain none nor have any actual lyrics will obviously affect the outcome. To check for english speaking songs, we used the package “langdetect” to check the lyrics within each song and got rid of those that were not english. This was done by using a for loop and using the lyrics column for each song to detect any different languages used. To check for instrumental songs, it is noted that on Spotify, any instrumental songs have their titles noted as, “Instrumental”. 

Similarly for checking a song’s language, we use a for loop and check the title within each song to see if it contains the word, “Instrumental”. Not only that, but we use a try block for each song to see if the lyrics feature any words or not, since a song may or may not contain any phrases or words whatsoever, which is what we want to avoid. After that, we used str.replace() to help remove any unwanted punctuation marks such as exclamation marks, question marks, periods, and other symbols. Since we're gonna analyze each word within our corpus list of words used within every song, we want words like, “love!” and “love?” to be the same word. 

Finally, we needed to remove the words, “Genius” from our genres columns as the various types of music contained the word, “Genius”, which of course sounds really weird to say (R&B Genius doesn’t sound like a popular genre). Thus we once again used str.replace() to remove all the “Genius”’s from the column. Once we had all that done, we had finally cleaned all of the data. Our Data Cleanup script can be found [here](Data%20Cleanup/Data%20Cleaning.ipynb)

### Exploratory Analysis
The dataset we collected contained a total of 120,000 of songs. Once those songs were restricted to only English and non-Instrumental tracks, we were left with a total of 85,000 to serve as training points. Our Exploratory Analysis script can be found [here](Data%20Cleanup/Exploratory%20Analysis.ipynb). The image below shows a sample view of some of the data set:

#### Figure 1: Sample View of Data Set
(image here)
Each of the tracks contains 10 features: the title and artist, the Spotify popularity metrics (hotness and number of listens), the platform IDs, genres, and the language and instrumental tags. We first looked at the distribution of listens metric, as this will be our main label for the training stages of the project. We would like to verify some of our assumptions about the large skew in the Spotify plays, since many of our design decisions were chosen with that in mind.

#### Figure 2: Distribution of Listens
(image here)
As evident in the figure above, the data is shaped precisely as we assumed. Only a small number of songs had a large number of listens, and as can be expected, most songs are not popular. Only very few break away from the low thousands of listens. Also of interest, is form of the distribution; the distribution is unimodal with no identifiable groups. It would appear that all of these songs are subject to similar conditions of popularity, since we do not see modes higher in the distribution. (Features which may be present if certain songs were buoyed by advertising or other market forces.) Ideally, this also translates to low rates of confounding behaviour and factors in the data, increasing the end model accuracy.

We examined the hotness metric to see if it followed a similar trend (see Figure 3). While there is a similar pattern of skew in the hotness data, the spread of the distribution within the hotness is far greater than what we see among the listens. While the range is larger in the listens (our most simplistic measure of spread), the vast majority of data points lie squarely to the left axis.

#### Figure 3: Distribution of Hotness
(image here)
To demonstrate these differences, see the following table of normalized variances for listens and hotness. 
(table here)
Unsurprisingly, the variance is far higher in the hotness. However, what's striking is that it is greater by many orders of magnitude. While at first glance these distributions may look very similar, they are in fact quite different. We suspect this may be indicative of a lack of correlation between the two measures (see Figure 5).

This is concerning, since originally, we wanted to utilize hotness as a validating metric for the number of listens as a measure of popularity. A lack of correlation suggests that if one is a good measure of popularity, the other would not be.

To explore this, we utilized a simple linear model to regress the hotness of a song, by the number of listens. Below is that regression plotted against those two features.

#### Figure 4: Song Length
(image here)
The above is a distribution of song length of the English, non-instrumental dataset. Luckily the average song is 254 words long, and is far within our capabilities to train the RNNs against. Additionally, most songs fall very closely within this reasonable range. The longest song in the dataset, however, is a total of 9969 words long (not pictured in the visualization), which is definitely larger than we would like to tackle in the scope of this project. (At this point, we need to find a literature basis for the upper effective size for these models, and then filter our dataset accordingly.)

#### Figure 5: Scatter Plot of Number of Hotness by Number of Listens
(image here)
And as we suspected, the model is absolutely terrible. Looking at the scatter plot, there seems to be very little relation between the metrics, and it is hard to imagine a model that would be successful given this data. The R^2 value for this model is an abysmally low .06.
We were hoping that by correlating the number of listens with the hotness value, we could further validate its use as a popularity measure. Unfortunately, the lack of correlations negates this use of hotness as a source of validation for the popularity of a song. For lack of a more salient metric, however, we will continue forward with the number of listens as our final regression target, but must keep in mind that it is not necessarily indicative of the song’s popularity, as we did not manage to validate such a claim.

Beyond viewing the various distribution statistics for each portion of our dataset, there are several key pieces of information we need to know about the dataset before performing the main analysis. The models which we will be using (Transformers, LSTMs, etc), are the most adept models currently in use for language tasks. They are able to remember key contextual information for far longer periods of time than any of their predecessors, thus expanding their ability to reference distant dependencies. However, the longer we extend their input, the longer and more difficult training such a network becomes. Thus the last numeric value within our dataset, but certainly not the least important, is song length.

## Analysis
### Corpus and Embeddings
While it is standard to use a pre trained embedding without modification in many language tasks, we will be fine tuning our embedding to the dataset. We believe that the use of the English language within songs varies from general use. Within a song, a word can be chosen not just for its meaning, but also its rhythm or cadence. Sometimes a word is chosen with complete disregard for meaning. We expect that fine tuning the embeddings on a corpus of songs will lead to an increased performance in the final model.

There are many ways to generate an embedding, but for our embedding training, we will be using a process called “Continuous Bag of Words,” or CBOW. This method is used by the Word2Vec model, and has resulted in meaningful embeddings within the literature. This model predicts our target word by taking in pairs of context and target words.

To generate the corpus, we will utilize the lyrics from each of the songs within our dataset. Each word must be paired with a unique numeric identifier, which will replace that word within the corpus. It is by using this list that we will select the context and targets for our CBOW model. For our embeddings, we selected a context window size of 3 words in either direction of the target word; this was an important consideration because more context does not necessarily mean more accurate results as words farther away become less related to the target. Our embedding script can be found [here](Embeddings/embedding%20(2).ipynb).

### Training
The primary model which we will utilize is a Transformer network. This network has excelled experimentally at natural language tasks, and trains quickly in parallel processing situations (read GPU). (We will need to decide whether we will be using multi-headed attention or not.) Additionally, we will train a simple recurrent neural network (RNN) and a long-short-term memory (LSTM) on the dataset; these will provide a baseline performance for the more complicated Transformer network. However, the size and parameters of the networks are yet to be determined. In addition to the recurrent models, we will perform the regression task using even simpler fully connected models. For those which cannot handle data of variable length, we will attempt padding or truncating the lyrical embeddings.

We expect that the main model we will be training for this regression task will be a Transformer structure with a fully connected regression head. The training on such a model will be the most time consuming portion of the project. We have chosen the U of U Center for High Performance Computing to undertake this task. Additionally, we will explore utilizing transfer learning from GPT-2 (OpenAI) or BERT (Google), since this would reduce the learning time for our model.
Once the models have been trained on the data, we will cross-validate with a testing set of data to select the most performant model for our regression task. This will become the broad model. We will then refit the model to five selected genres, looking at the performance of these models as an indicator of how important lyrics are within that genre when compared to the general market.
Finally, as an extension of the project, we aim to create the lyric analysis tool described previously. This tool will be able to utilize the general and genre specific models which we will train to identify which lyrics have the greatest negative impact on the final score given by the model. Further, if the genre of the lyrics is one of the five which we select, then a specific model for that genre will be used in place of the general model.

## Limitations
### Validity
While it is tempting to regard the measure of Spotify listens as an absolute indicator of a song’s popularity, we were not able to validate against any other metrics. That is not to say that the number of listens is not related to popularity, but that future investigation would be needed to support this claim thoroughly. Only once this has been given rigorous consideration can the results of our eventual model be interpreted as predictions of popularity.

### Generalization
As with any model, it is hard to measure how well it will generalize outside of the domain under which it was trained. In our case, the domain consists only of songs which were both on Spotify and Genius, and only a subsection of those which were chosen by our scraping process. While randomly choosing our data, and collecting such a large dataset, brings our set towards being a representative sample, it will only be a representative sample of those songs which are on Genius and Spotify. Any predictions which are outside of the domain will be maximally as accurate as we have measured within the domain, but likely far worse.

### Ethical Considerations
While we are not utilizing private or sensitive data outside of the public sphere, there are still some ethical implications which arise from the application of such techniques to judge the quality of an individual’s effort, especially when the product is art. Such a tool which evaluates the ‘quality’ of lyrical work is not necessarily representative of a song’s artistic merit. If such a tool, geared to increase a raw and speculative metric like the number of Spotify listens, were to be used for measuring the worth of an Artist’s work or career, it might cause unjust devaluing of otherwise capable individuals. If those who manage such artists, such as Record Labels, feel that they should rely on these types of metrics and suggestions entirely, they may require conformity to those standards on the part of the musicians, hampering the diversity of music which they produce.

### Project Progress
As of the Project Milestone benchmark we have collected and cleaned the data, performed preliminary analysis on the song data and we are currently training the embeddings. Our next steps include choosing our pretrained model, this going to be a transformer and a RNN as stated in the intro of this paper. Below is a project schedule for the rest of the semester.
Weekly Schedule: 
-March 27th: All data collected,  progress embeddings
-March 29th - April 2nd: Have clean and explored data, complete embeddings, model types chosen.
-April 2nd: Models created and training of the main model in progress.
-April 9th: Training completed and models evaluated. Begin creating project video.
-April 19th (Entire Project): *Can identify the problematic areas of lyrics using models






