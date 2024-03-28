Amazon Reviews Sentiment Analysis

>> Description

The Amazon Reviews Sentiment Analysis project leverages advanced Natural Language Processing (NLP) techniques to evaluate and analyze customer reviews of Amazon products, mainly focusing on consumer electronics. This project aims to distil vast quantities of textual data into actionable insights, determining consumers' prevailing sentiment—positive, neutral, or negative—. Utilizing Python and powerful libraries like spaCy and TextBlob, the project processes, classifies, and visualizes sentiment data. This endeavour not only underscores the significance of sentiment analysis in interpreting consumer feedback but also showcases the potential of NLP in extracting meaningful patterns and trends from textual data, thereby aiding businesses in understanding customer satisfaction and product perception.

>> Installation

Before running this project, ensure you have Python 3.x installed on your system. 
You can download Python [here](https://www.python.org/downloads/).

To set up the project environment, run the following commands in your terminal:

```bash
Clone the project repository
git clone https://github.com/AncientJL/finalCapstone.git

Navigate to the project directory
cd finalCapstone

Install required Python packages
pip install pandas numpy spacy textblob spacytextblob matplotlib seaborn wordcloud

Download and install the spaCy small English model
python -m spacy download en_core_web_sm

Download additional data required by TextBlob
python -m textblob.download_corpora

>> Usage
After installing the necessary Python libraries, you can execute the sentiment analysis scripts to analyze Amazon product reviews. For a detailed exploration, refer to the Colab Notebooks in the repository, which illustrate the step-by-step preprocessing, analysis, and visualization processes. 
python sentiment_analysis.py

>> Project Structure
Amazon_product_reviews.csv: The dataset containing Amazon product reviews.
sentiment_analysis.py: Python script for conducting sentiment analysis.
sentiment_analysis_report.pdf: Detailed report of the analysis findings and insights.

>> Credits
Joanne ZY Liaw: Primary researcher and developer responsible for the project's conception, methodology design, data analysis, and reporting.
This project was developed as part of the coursework for Cohort C5 Nov at Hyperion Dev, under the guidance and mentorship of academic and industry professionals.
For further inquiries or contributions to the project, don't hesitate to contact Joanne ZY Liaw.
This project is a testament to the collaborative effort and expertise shared among peers and mentors, pushing the boundaries of what's possible in sentiment analysis and NLP.
