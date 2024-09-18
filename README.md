# Fake News Detector  (part 1)
This is “Part One” of the two-part “Fake News Detector” project.
Part One provides an overview of the challenges and strategies currently deployed in the fake news detection process. It then delves into different aspects of using machine learning techniques to address these challenges. Curated true and fake news datasets are first obtained from Kaggle, and the project begins with an Exploratory Data Analysis (EDA) to understand the data, followed by preparing the dataset for Machine Learning models to determine the authenticity of the news text.
Part Two focuses on a Machine Learning-based real-time decision support system. Here, the dataset is split into training and testing sets. Various models, including Logistic Regression, K-Nearest Neighbors, Decision Tree Classifier, and SVM, are trained and tested on the dataset. These models are then used to evaluate real-time text inputs to determine their truthfulness.  Part Two is presented in a separate document.

### Repository Structure
	./data_fakenews/True.csv, Fake.csv - curated fake/true news data files
	FakeNewsModel.ini - program configuration file for processing the dataset
	Fakenews-EDA-title.ipynb - Fakenews Detector performing EDA on the title column
	Fakenews-EDA-text.ipynb - Fakenews Detector performing EDA on the text column

### DataSet
Obtained from SimpliLearn, 2 2-file set, True.csv contains the curated true news file, and Fake.csv 
curated fake news.  The files contain the following common format:
	Title:	 	text, headline of the article
	Text: 		text, news article
	Subject:	text, subject of the article
	Date: 		date time of article’s publishing time

## What is Fake News?
Fake news refers to false or misleading information presented as news.  It can take the form of articles, images or videos that are designed to deceive and manipulate others to gain control or power in a situation. Over time, these actions can lead to chronic manipulation and potential destructive consequences.  Here are some key points about fake news:
### Types of Fake News:
Misinformation: False information shared without harmful intent.
Disinformation: Deliberately false information spread with the intent to deceive.
Propaganda: Information, often biased or misleading, used to promote a political cause or point of view.
Hoaxes: Fabricated stories meant to trick people.
### Motivations Behind Fake News
#### Political Gain
To influence public opinion or discredit opponents.
#### Financial Gain
To attract clicks and generate advertising revenue.
#### Social Influence
To shape societal views or create confusion.
### Traditional Means of Detecting Fake News
Identifying fake news is incredibly challenging and time-consuming. It requires dedicated and knowledgeable personnel to constantly review new information and update knowledge. Here are some of the strategies to spot fake news:
#### Check the source
Verify the data source, the credibility of the article’s author, the website, or the publication medium of the publisher. 
#### Read Beyond the Headlines
Headlines can be misleading.
#### Look for Evidence and Supporting Documents
Reliable news stories provide evidence and cite sources
#### Examine the Date
Sometimes old news stories are recirculated as if they are current.
#### Use Fact-Checking Websites
Fact checking websites like FactCheck.org can assist in determining the authenticity of the article.
### Machine Learning
Machine learning models can significantly enhance the ability to detect fake news by analyzing the presented text information's patterns and features. Fake News Detection involves several steps and techniques to analyze and classify news articles. Here’s an overview of the process:
#### Data Collection
Gather a large dataset of news articles, including both fake and true news.
#### Data Preprocessing
Clean the data by removing irrelevant information, normalizing text, and handling missing values. Natural Language Processing techniques such as tokenization, stemming and lemmatization are often used.
#### Feature Extraction
Extract features from the text that can help in classification. Common techniques include:

##### Bag of Words (BoW)
Represents text as a set of word frequencies.

##### Term Frequency-Inverse Document Frequency (TF-IDF)
Measures the importance of a word in a document relative to a collection of documents.

### EDA (Exploratory Data Analysis)
Data Cleaning
Handling missing data, remove duplicates and correct errors
Descriptive Statistics
Summarizing the dataset’s main features, such as mean, median, mode, and standard deviation
Data Visualization
Creating plots and charts to understand the distribution and relationships with the data
Identifying Patterns
Look for trends, correlations and anomalies in the data


## Part Two
Model Selection:
Choose appropriate machine learning models. Commonly used models include:
Decision Tree Classifier
Support Vector Machines (SVM)
KNNearest Neighbors
Logistic Regression
Training the Model:
Split the dataset into training and testing sets.
Evaluation:
Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
Continuous Improvement:
Continuously update the model with new data and retrain it to improve accuracy


## Exploratory Data Analysis (EDA)
Traditional data analysis, such as finding and counting null and missing values, dropping bad data, and unusable features, and presenting the types and information on the features columns are performed in the accompanying jupyter notebooks and not presented here.  


The dataset is further processed where the ‘publisher’ information is extracted from the source text and a new feature column ‘published’ created.  Here is a snippet of the processed data:


***** Fake Data *****
   title \
0 Donald Trump Sends Out Embarrassing New Year’...
1 Drunk Bragging Trump Staffer Started Russian ...
2 Sheriff David Clarke Becomes An Internet Joke...
3 Trump Is So Obsessed He Even Has Obama’s Name...
4 Pope Francis Just Called Out Donald Trump Dur...


text subject \
0 Donald Trump just couldn t wish all Americans ... News
1 House Intelligence Committee Chairman Devin Nu... News
2 On Friday, it was revealed that former Milwauk... News
3 On Christmas day, Donald Trump announced that ... News
4 Pope Francis used his annual Christmas Day mes... News


date class publisher
0 December 31, 2017 0 None
1 December 31, 2017 0 None
2 December 30, 2017 0 None
3 December 29, 2017 0 None
4 December 25, 2017 0 None


***** True Data *****
title \
0 As U.S. budget fight looms, Republicans flip t...
1 U.S. military to accept transgender recruits o...
2 Senior U.S. Republican senator: 'Let Mr. Muell...
3 FBI Russia probe helped by Australian diplomat...
4 Trump wants Postal Service to charge 'much mor…


text subject \
0 The head of a conservative Republican faction... politicsNews
1 Transgender people will be allowed for the fi... politicsNews
2 The special counsel investigation of links be... politicsNews
3 Trump campaign adviser George Papadopoulos to... politicsNews
4 President Donald Trump called on the U.S. Pos... politicsNews


date class publisher
0 December 31, 2017 1 WASHINGTON (Reuters)
1 December 29, 2017 1 WASHINGTON (Reuters)
2 December 31, 2017 1 WASHINGTON (Reuters)
3 December 30, 2017 1 WASHINGTON (Reuters)
4 December 29, 2017 1 SEATTLE/WASHINGTON (Reuters)
### Natural Language Processing
The Natural Language Toolkit (NLTK) is used to prepare text data for later machine learning stages. The text is first stripped of unwanted escape character sequences, tokenized, stemmed, and then lemmatized.


Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) techniques are used to convert the text into numerical format for machine learning algorithms processing.


Due to the fact that the title of the news can be just as misleading as the actual text, two separate passes  programs are prepared.  One examining only the title, and the other the text field of the news dataset.

### Publisher Counts
The following plots illustrate how fake and true news are separated based on the source of the publisher.





### True News BoW / TF-IDF - TITLE

### True News BoW / TF-IDF - TEXT
### Fake News BoW / TF-IDF - TITLE
### Fake News BoW / TF-IDF - TEXT
For brevity, word plots below shown only for the ‘Title’ feature, using the TF-IDF for the Fake and True News.  Please refer to the jupyter for the detailed plots for each feature.
## Word Cloud
### Fake News
### True News
## Pie Chart
### Fake News
### True News
## Tree Map
### Fake News
### True News
## Radial Bar
### Fake News
### True News

