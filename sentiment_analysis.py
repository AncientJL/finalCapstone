"""This is the link to my colab Note Book - Birth of Capstone Project
Name of File : 4A_ Sentiment_Analysis_ipynb
For a full comprehensive digital report.
https://colab.research.google.com/drive/14HZn-qV3ACnfE-vBAlq30HVMzDWtNqcW#scrollTo=zW7yyvi6EZG- """

# Code Block 4: Initial Setup and Library Imports
#%matplotlib inline # (Switch on For Jupyter Notebooks)
# Import Libraries
import spacy # NLP Librabry

# Librabry use to handle data manipulation and analysis
import numpy as np # Support for efficient numerical operations
import pandas as pd # Tools for handling structured data

# Libraries, which are used for data visualization
import seaborn as sns #  Creates high-level informative statistical graphics.
import matplotlib.pyplot as plt # Creates different plots
import matplotlib.dates as mdates # Functionailty working with DT in plots.

from scipy import stats # Offers various stats func and distrib for D.Anly
from textblob import TextBlob # NPL proc sentiment analysis on textual data
from wordcloud import WordCloud # Create Visual representions of Freq Text
from spacytextblob.spacytextblob import SpacyTextBlob # Extentension Lib.

#-----------------------------------------------------------------------

# Code Block 5: Data Loading and Preprocessing
data = pd.read_csv('Amazon_product_reviews.csv') # Load and read dataset
data = data.drop_duplicates() # Remove any duplicate rows in dataset
data['reviews_date_backup'] = data['reviews.date'].copy() # Bkup reviews.date

#print(data.head()) # Check that the .csv loaded
#print(data.columns) # Check the coloum has reviews.text, just in case.

reviews_data = data['reviews.text'] # Select the review.text colomn
clean_data = reviews_data.dropna() # Removes rows with missing values (NaN)

nlp = spacy.load('en_core_web_sm') # Load Small Eng NLP
nlp.add_pipe('spacytextblob') # Load sentiment analysis analyser

# Define function for text processing
def text_preprocessing(text):
  doc= nlp(text) # nlp() func tokenizes text & performs var linguistic task
  clean_tokens = [
    token.lemma_.lower() for token in doc

    if not token.is_stop and not token.is_punct and not token.like_num
] #initialize a list clean-tokens (Home to all clean tokens)

  clean_text = ' '.join(clean_tokens) # Join clean_token to a single string.
  return clean_text # Return the processed and cleaned text.

data['clean_review_text'] = clean_data.apply(text_preprocessing) # Store in DF
num_rows = data.shape[0]
print("Count rows left after data cleaning:", num_rows)

#pd.set_option('display.max_colwidth', None) # Check cleaned txts as intended.
print(data[['reviews.text', 'clean_review_text']].head())
#print(data.columns) # Check reviews.date has not disappered.

#-----------------------------------------------------------------------

# Code Block 6: Sentiment Analysis Function
# Define the sentiment analysis function
def sentiment_analysis(clean_review_text):
    # Calculate the polarity score using the TextBlob library
    polarity = TextBlob(clean_review_text).sentiment.polarity

    # Calculate the sentiment label
    sentiment = ('Positive' if polarity > 0 else
                 'Negative' if polarity < 0 else
                 'Neutral')

    # Return a Series with the sentiment label and polarity score
    return pd.Series([sentiment, polarity], index=['sentiment', 'polarity'])

# Apply sentiment_analysis on clean_review_text
results = data['clean_review_text'].apply(sentiment_analysis)

# Add the sentiment label and polarity score columns to the data DataFrame
data = data.join(results)
print(data.head()) # Checking the results has completed the task.

#-----------------------------------------------------------------------

# Code Block 7: Sentiment Over Time Analysis
# Restore 'reviews.date' from backup if 'reviews.date' not found
if 'reviews_date_backup' in data.columns and 'reviews.date' not in data.columns:
    data['reviews.date'] = data['reviews_date_backup']

# Normalization
data['normalized_polarity'] = (data['polarity'] + 1) * 5

# Convert 'reviews.date' to DatetimeIndex
data.set_index('reviews.date', inplace=True, drop=False)
data.index = pd.to_datetime(data.index)

# Resample, calculate mean polarity over time, and create a DataFrame
if 'normalized_polarity' in data.columns:
    mean_polarity_over_time = data['normalized_polarity'].resample('M').mean()
    df_mean_polarity = mean_polarity_over_time.reset_index()
    df_mean_polarity.columns = ['Date', 'Mean Normalized Polarity']

    # Remove NaN values
    df_mean_polarity.dropna(inplace=True)

    # Convert Date to numbers for fitting trend line
    dates_num = mdates.date2num(df_mean_polarity['Date'])
    x = np.array(dates_num).reshape(-1, 1)

    # Fit linear regression line
    A = np.hstack((x, np.ones_like(x)))
    coefficients = np.linalg.lstsq(A, df_mean_polarity['Mean Normalized Polarity'],
                                   rcond=None)[0]

    # Create polynomial function for plotting
    p = np.poly1d(coefficients)

    # Plot with Trend Line
    plt.figure(figsize=(8, 4))
    plt.plot(df_mean_polarity['Date'],
             df_mean_polarity['Mean Normalized Polarity'],
             marker='o', label='Monthly Mean Polarity')

    # Plot the trend line
    plt.plot(df_mean_polarity['Date'], mdates.num2date(p(x)), linestyle='--',
             linewidth=2, color='red', label='Trend Line')

    plt.title('Sentiment Polarity Trend Over Time with Trend Line') # Title
    plt.xlabel('Time') # x-axis name
    plt.ylabel('Average Normalized Polarity') # y-axis name
    plt.text(0.5, -0.2, '\nFig.1 - Time Series Analysis of Sentiment Polarity',
             transform=plt.gca().transAxes, ha='center')
    plt.legend() # Legend
    plt.show() # Show graph

else:
    print("Column 'normalized_polarity' does not exist. ")

#-----------------------------------------------------------------------

# Code Block 8: Pie Chart for Sentiment Distribution
#Pie Chart for Sentiment Distribution:

# Count all 'sentiment' column and cal the freq of each unique value.
sentiment_counts = data['sentiment'].value_counts()

# Create a DataFrame with the sentiment counts
df_sentiment_counts = pd.DataFrame({'Sentiment': sentiment_counts.index,
                                    'Count': sentiment_counts})
print(df_sentiment_counts.to_string(index=False)) # Show Df

# 5 inches by 5 inches
plt.figure(figsize=(6, 5))

# Custom formatting function to place percentages inside the slices
# This section is included because the percentages on Pie was overlapping.
def autopct_format(pct):
    # Calculate the angle for each percentage
    angle = 2 * np.pi * pct / 100
    # If it's the negative slice, adjust the position of the label
    if pct == 4.995:
        return f'{pct:.3f}%\n\n\n\n'
    else:
        return f'{pct:.3f}%\n'

#Plot the data S-M, specs the labels per slice, in % , set to 140 degrees
plt.pie(sentiment_counts, labels=sentiment_counts.index,
        autopct=autopct_format, startangle=140, textprops={'rotation': 65})

plt.title('Sentiment Distribution\n') # Pie chart name
plt.text(0.5, -0.1, '\nFig.2 - Sentiment Distribution Pie Chart: '
            '\nDepict the proportion of different sentiment categories',
         transform=plt.gca().transAxes, ha='center')
plt.show() # Show Pie Graph

#-----------------------------------------------------------------------

# Code Block 9: Bar Chart for Sentiment Counts
# Print the data frame
print(" Sentiment Counts:\n")
print(sentiment_counts)

plt.figure(figsize=(5, 5)) # Size 4in by 3 in
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red']) # Add Colour
plt.title('Frequency of Sentiment Categories\n') # Bar char title
plt.xlabel('Sentiment Catogory') # x-axis name
plt.ylabel('Frequency of Sentiments') # y-axis name
plt.xticks(rotation=360)  # Adjust the rotation to 360 degrees
plt.subplots_adjust(bottom=0.3)  # Adjust the bottom margin
plt.text(0.5, -0.2, 'Fig.3 - Frequency of each Sentiment:'
 '\nDepict the Polarity frequency per sentiment',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show bar chart

#-----------------------------------------------------------------------

# Code Block 10: Word Clouds for Positive Sentiments

# Joining +ve reviews into a single string, breaking into multiple lines
positive_reviews = ' '.join(
    data[data['sentiment'] == 'Positive']['clean_review_text']
)

# Generating a word cloud for positive sentiment reviews
positive_wordcloud = WordCloud(
    background_color='white'
).generate(positive_reviews)

# Store the positive word cloud in a list
wordcloud_list = [positive_wordcloud]

# Store the positive word cloud in a list
wordcloud_list = [positive_wordcloud]
sorted_word_freq = []

print("Top 20 words based on frequency in positive sentiment:\n")
for word, freq in sorted_word_freq[:20]:
    print(f"{word}: {freq}")

# Print the top 20 words based on freq for each word cloud in the list
for i, wordcloud in enumerate(wordcloud_list):
    word_freq = wordcloud.words_
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1],
                              reverse=True)
    for word, freq in sorted_word_freq[:20]:
        print(f"{word}: {freq}")
    print("\n")


plt.figure(figsize=(6, 4)) # Size 6in by 4in
plt.imshow(positive_wordcloud, interpolation='bilinear') # Show +ve image
plt.axis('off') # Remove axis and ticks
plt.title('Word Cloud for Positive Sentiments\n') # Title
plt.text(0.5, -0.2, 'Fig.4 - Word Clouds with Positive Sentiments:'
 '\nDepict the Highest frequency per positive word',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show image

#-----------------------------------------------------------------------

# Code Block 11: Word Clouds for Negative Sentiments
# Filtering and joining negative reviews into a single string
negative_reviews = ' '.join(
    data[data['sentiment'] == 'Negative']['clean_review_text']
)

# Generating a word cloud for negative sentiment reviews
negative_wordcloud = WordCloud(
    background_color='white',
    width=800,  # optional, to define width
    height=400  # optional, to define height
).generate(negative_reviews)

# Store the negative word cloud in a list
wordcloud_list.append(negative_wordcloud)

# Get the word frequency for the negative word cloud
word_freq = negative_wordcloud.words_

# Sort the word frequency in descending order
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1],
                          reverse=True)

# Print the top 20 words based on frequency
print("Top 20 words based on frequency in negative sentiment:\n")
for word, freq in sorted_word_freq[:20]:
    print(f"{word}: {freq}")

# Displaying the word cloud for negative reviews
plt.figure(figsize=(6, 4))  # Adjusting figure size
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis and ticks
plt.title('Word Cloud for Negative Sentiments\n') # Title
plt.text(0.5, -0.2, 'Fig.5 - Word Clouds with Negative Sentiments:'
 '\nDepict the Highest frequency per negative word',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show image

#-----------------------------------------------------------------------

# Code Block 13: Basic Statistical Measures and Sentiment Label Counts
#Statistical Analysis:
print("Count of Sentiment Labels:\n", data['sentiment'].value_counts())

# Bar Chart for Sentiment Counts
plt.figure(figsize=(5, 3)) # Size 5in by 3 in
sns.countplot(x='sentiment', data=data, palette='viridis') # Bar Chart from data
plt.xlabel('Sentiment') # x-axis name
plt.ylabel('Count') #y-axis name
plt.title('Count of Sentiment Labels') # Title
plt.text(0.5, -0.2, 'Fig.6 - Count of Sentiments:'
 '\nDepict the Highest count per sentiment',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show bar Chart

#-----------------------------------------------------------------------

# Code Block 14: Pie Chart for Sentiment Proportions
print("\nProportions of Sentiment Labels:\n",
      data['sentiment'].value_counts(normalize=True))

# Pie Chart for Sentiment Proportions
sentiment_proportions = data['sentiment'].value_counts(normalize=True)
plt.figure(figsize=(5, 5))
sentiment_proportions.plot.pie(autopct='%1.1f%%', startangle=140,
                               colors=['skyblue', 'orange', 'green'])
plt.title('Proportions of Sentiment Labels')
plt.ylabel('')  # Hide the y-label
plt.text(0.5, -0.0, 'Fig.7 - Proportions of Sentiment Labels:\n'
                   'Display the highest count per sentiment category',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Pie Chart

#-----------------------------------------------------------------------

# Code Block 15: Histogram for Polarity Scores Distribution
mean_polarity = data['polarity'].mean() # Calculate the mean for 'polarity'
std_polarity = data['polarity'].std() # Calculate the std for 'polarity'
print("\nMean Polarity:", mean_polarity)
print("\nStandard deviation Polarity:", std_polarity)

# Histogram for Polarity Scores
plt.figure(figsize=(6, 4)) # 6in by 4in
sns.histplot(data['polarity'], bins=20, kde=True, color='purple')
plt.title('Distribution of Polarity Scores') # Title
plt.xlabel('Polarity') # x-axis name
plt.ylabel('Frequency') # y-axis name
plt.text(0.5, -0.1, '\nFig.8 - Distribution of Polarity Scores:'
         '\nRepresenting the frequency of polarity values in the dataset',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Histogram

#-----------------------------------------------------------------------

# Code Block 16: Box Plot for Polarity Score Distribution
print("Standard Deviation of Polarity:", std_polarity) #

# Box Plot for Polarity Scores
plt.figure(figsize=(6, 4)) # Size 6in by 4 in
sns.boxplot(x='polarity', data=data, palette='coolwarm')
plt.title('Box Plot of Polarity Score Distribution') # Title
plt.xlabel('Polarity Scores') # x-axis Name
plt.text(0.5, -0.15, '\nFig.9 - Frequency Distribution of Polarity Scores:'
         '\nRepresenting the Spread of Polarity Values',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Box Plot

#-----------------------------------------------------------------------

# Code Block 17: Histogram for Distribution of Polarity Scores
# Count the occurrence of each polarity than resets to regular column
polarity_counts = data['polarity'].value_counts().reset_index()
polarity_counts.columns = ['Polarity', 'Frequency'] # Access the two cols
polarity_counts = polarity_counts.sort_values('Polarity') # Sort to Bins
print(polarity_counts)

min_polarity = data['polarity'].min() # Cal the min polarity scores
max_polarity = data['polarity'].max() # Cal the min polarity scores
print(f"\nRange of Polarity Scores: {min_polarity} to {max_polarity}")

plt.figure(figsize=(6, 4)) # Size 6in by 4in
plt.hist(data['polarity'], bins=20, color='blue', alpha=0.7) # Create fig
plt.title('Distribution of Polarity Scores') #Title
plt.xlabel('Polarity') # x-axis name
plt.ylabel('Frequency') # yx-axis name
plt.grid(True)
plt.text(0.5, -0.2, '\nFig.10 - Distribution of Polarity Scores\n:'
         '\nSpread of polarity scores by sentiment',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Histogram

#-----------------------------------------------------------------------

# Code Block 18: Quartiles and Interquartile Range Calculation
quartiles = data['polarity'].quantile([0.25, 0.5, 0.75]) # Cal Quartiles
IQR = quartiles[0.75] - quartiles[0.25] # Cal interqutaile Range
print("Quartiles:\n", quartiles)
print("Interquartile Range:", IQR)

plt.figure(figsize=(6, 4)) # Size 6in by 4in
sns.boxplot(x=data['polarity'], color='yellow')
plt.title('Box Plot of Polarity Scores') # Title
plt.xlabel('Polarity') # x -axis name
plt.grid(True)
plt.text(0.5, -0.2, '\nFig.11 - Polarity Scores\n:'
         'Distribution and spread of sentiment in the dataset',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Box plot

#-----------------------------------------------------------------------

# Code Block 19: Identifying Outliers in Polarity Scores
# Identifying outliers
lower_bound = quartiles[0.25] - (1.5 * IQR) # Cal Lower bound
upper_bound = quartiles[0.75] + (1.5 * IQR) # Cal Lower bound
print(f"Outliers are values below {lower_bound} and above {upper_bound}.")

# Identify extreme sentiment scores
outliers = data[(data['polarity'] < lower_bound) | (data['polarity'] > upper_bound)]

# Plotting the original distribution with outliers highlighted
plt.figure(figsize=(6, 4))
sns.histplot(data['polarity'], kde=True, color="purple",
             label="Polarity Scores", bins=30) #  create a KDE
plt.scatter(outliers['polarity'], np.zeros_like(outliers['polarity']) - 0.01,
            color='red', s=50, label='Outliers')
plt.title('Distribution of Polarity Scores with Outliers') # Title
plt.xlabel('Polarity') # x -axis name
plt.ylabel('Frequency') # y -axis name
plt.legend() # Legend
plt.text(0.5, -0.2, '\nFig.12 -Dist. & Spread of Sentiment in the Dataset\n:'
         '\nMean Polarity Scores',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Histogram and KDE Line

#-----------------------------------------------------------------------

# Code Block 20: Mean Polarity by Sentiment Category

category_means = data.groupby('sentiment')['polarity'].mean() # Gp & cal Mean
print("\nMean Polarity by Sentiment Category:\n", category_means)

plt.figure(figsize=(5, 3)) # size 5in by 3in
category_means.plot(kind='bar', color=['lightblue', 'green', 'pink']) # Colour
plt.title('Mean Polarity by Sentiment Category') # Title
plt.xlabel('Sentiment Category') # X -axis
plt.ylabel('Mean Polarity') # y - axis
plt.xticks(rotation=45)
plt.text(0.5, -0.3, '\n\nFig.13 -Mean Polarity Scores\n:'
         'Distribution and Spread of Sentiment in the Dataset',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Bar Chart

#-----------------------------------------------------------------------

# Code Block 21: Median Polarity by Sentiment Category
category_medians = data.groupby('sentiment')['polarity'].median() # Gp & cal Median
print("\nMedian Polarity by Sentiment Category:\n", category_medians)

plt.figure(figsize=(5, 3)) #Size 5in by 3 in
category_medians.plot(kind='bar', color=['red', 'green', 'blue']) #Colour
plt.title('Median Polarity by Sentiment Category') # Title
plt.xlabel('Sentiment Category') # X-axis
plt.ylabel('Median Polarity') # y -axis
plt.xticks(rotation=45)
plt.text(0.5, -0.3, '\nFig.14 -Median Polarity by Sentiment Category:\n'
         'Distribution and Spread of Sentiment in the Dataset',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Bar Chart

#-----------------------------------------------------------------------

# Code Block 22: Confidence Interval for Mean Polarity
mean_polarity = data['polarity'].mean() # Cal Mean Polarity
confidence_level = 0.95
degrees_freedom = data['polarity'].count() - 1 # Deg of freedom
std_err = stats.sem(data['polarity']) # Standard Error
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       mean_polarity, std_err)
print(f"\n95% Confidence Interval for Mean Polarity: {confidence_interval}")

# Visualizing the mean polarity with confidence interval
plt.figure(figsize=(5,3)) # Size 5in by 3in
plt.bar('Mean Polarity', mean_polarity, color='skyblue', 
        yerr=[[mean_polarity - confidence_interval[0]], 
         [confidence_interval[1] - mean_polarity]], capsize=10)
plt.errorbar('Mean Polarity', mean_polarity, 
             yerr=[[mean_polarity - confidence_interval[0]], 
              [confidence_interval[1] - mean_polarity]], 
             fmt='o', color='darkblue', capsize=10)
plt.ylabel('Polarity Score') # X-axis
plt.title('Mean Polarity with 95% Confidence Interval') # Title
plt.tight_layout()
plt.text(0.5, -0.2, '\nFig.15 -Mean Polarity Score with 95% C.Interval:\n:'
         '\nMean polarity score and the level of confidence in the estimate',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Bar Chart with Errors Bars

#-----------------------------------------------------------------------

# Code Block 23: Correlation between Polarity and Review Length
data['review_length'] = data['clean_review_text'].apply(len) # Cal len store
print("\nCorrelation between Polarity and Review Length:\n",
      data[['polarity', 'review_length']].corr())

# Plotting scatter plot with regression line
plt.figure(figsize=(6, 4)) # Size 6in by 4in
sns.regplot(x='review_length', y='polarity', data=data,
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Review Length(RL) vs. Polarity(SP)') # Title
plt.xlabel('Review Length') # X-axis
plt.ylabel('Polarity') # y-axis
plt.text(0.5, -0.2, '\nFig.16 -Correlation between RL and SP:\n:'
         '\nCorrelation between Review Length and Sentiment Polarity',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Scatter plot with regression line

#-----------------------------------------------------------------------

# Code Block 24: Correlation between Polarity and Review Length
# Printing DataFrame for Correlation
correlation_stats = data[['review_length', 'polarity']].corr() # Cal RL vs SP
print(correlation_stats)

# Creating a DataFrame for visualization (if needed)
df_for_visualization = pd.DataFrame({
    'Review Length': data['review_length'],
    'Polarity': data['polarity']
})

print(df_for_visualization.head())

bins = np.linspace(0, 1, 11)  # 11 bins from 0 to 1
polarity_hist, _ = np.histogram(data['polarity'], bins=bins) # CBuils Hist

for i, count in enumerate(polarity_hist): # Count each bin
    print(f"{bins[i]:.2f} - {bins[i+1]:.2f}\t{count}")

# Assuming 'review_length' and 'polarity' are already in your DataFrame
plt.figure(figsize=(3, 5)) # Size 4in by 5in
sns.jointplot(x='review_length', y='polarity', data=data, kind='hex',
              gridsize=20, color='blue', space=0)
plt.suptitle('Review Length vs. Polarity (Hexbin Plot)',
             y=1.02) # Title
plt.xlabel('Review Length') # x- axis
plt.ylabel('Polarity') # y- axis
plt.text(0.5, -0.1, '\nFig.17 -Density between RL and SP:\n:'
         '\nDensity & Correlation between Review Length and Sentiment Polarity',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Hexbin plot

#-----------------------------------------------------------------------

# Code Block 26: Normalization of Polarity Scores
# Applying a consistent normalization procedure for Scaling & standardization.
data['normalized_polarity'] = data['polarity'].apply(lambda x: (x + 1) * 5)

# Display the results
print(data[[
    'clean_review_text',
    'sentiment',
    'polarity',
    'normalized_polarity'
]].head())

#-----------------------------------------------------------------------

# Code Block 27: Categorization of Sentiment Scores
# Create bins for the 0-10 scale, assuming you want to categorize to 10 lvls
bins = np.linspace(0, 10, 11) # Increase View of Dist Scores

# Create the 'sentiment_group' column based on the 'reviews.rating' column
data['sentiment_group'] = pd.cut(data['reviews.rating'], bins,
                                 labels=np.arange(10), include_lowest=True)

# Print the DataFrame with the 'sentiment_group' column
print(data[['reviews.rating', 'sentiment_group']])

print(data['reviews.rating'].value_counts().sort_index())

# Create a bar plot of sentiment scores
plt.figure(figsize=(6, 4))
plt.bar(np.arange(1, 6), data['reviews.rating'].value_counts(),
        align='center')
plt.xticks(np.arange(1, 6))
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.text(0.5, -0.1, '\nFig.18 -Distribution of Sentiment Scores:\n:'
         '\nFrequency of Sentiment Scores',
         transform=plt.gca().transAxes, ha='center', va='top')
plt.show() # Show Bar Plot

#-----------------------------------------------------------------------

# Code Block 28: Calculating Similarity between Two Reviews
nlp = spacy.load('en_core_web_md')  # Make sure to download this model first

# Function to calculate similarity between two texts
def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

try: # This is included "be cautious not to use an index that is out of bounds"
    # Attempt to access an element using an index
    my_review_of_choice1 = data['reviews.text'][1985]
    my_review_of_choice2 = data['reviews.text'][2985]
except IndexError:
    # Handle the case where the index is out of bounds
    print(f"One of the indices is out of bounds for the dataset.")

similarity_score = calculate_similarity(my_review_of_choice1, my_review_of_choice2)
print(f"Review 1 length: {len(my_review_of_choice1)}")
print(f"Review 2 length: {len(my_review_of_choice2)}")
print(f"Similarity score: {similarity_score:.6f}")

#-----------------------------------------------------------------------

# Code Block 29: Sentiment Analysis & Similarity Score btw Custom Reviews Code
# Function to calculate similarity between two texts
def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# Example usage with two reviews
review1 = ("I absolutely love the great features of the Amazon Kindle tablet.  "
           "It's a fantastic device that I use every day, and its lightweight "
           "design makes it perfect for easy portability.")
review2 = ("I'm extremely disappointed with the small size of this app on the "
           "device. The book downloads don't work properly, and the tablet's "
           "outdated version causes numerous problems with updates, making it an "
           "incredibly frustrating experience to use this tablet.")

similarity_score = calculate_similarity(review1, review2)
print(f"Review 1 length: {len(review1)}")
print(f"Review 2 length: {len(review2)}")
print(f"Similarity score: {similarity_score}")
