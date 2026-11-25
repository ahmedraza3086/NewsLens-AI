# Project Proposal: Fake News Headline Detector

**Course:** Artificial Intelligence Lab  
**Leader:** Ahmed Raza (K24-3086)  
**Team Member:** Shahmir Ali (K24-3048)

## 1. Introduction

The Fake News Headline Detector is an AI-based project that classifies news headlines as likely real or fake using machine-learning techniques. The system also explains its predictions by highlighting influential words and identifies groups of similar headlines through clustering. The application is intended as an educational prototype for exploring natural language processing, supervised learning, unsupervised learning, and explainable AI.

## 2. Objectives

- Analyze and preprocess a labelled news-headline dataset.
- Apply Term Frequency–Inverse Document Frequency (TF-IDF) to convert text into numerical features.
- Train and compare Naive Bayes and Logistic Regression models.
- Use K-Means clustering to identify groups of similar headline patterns.
- Explain predictions using influential words.
- Develop an interactive application using Streamlit.

## 3. Methodology

The project uses a publicly available labelled news dataset. Headlines are cleaned and prepared before feature extraction. TF-IDF represents the text numerically, after which Naive Bayes and Logistic Regression are trained as supervised classifiers. K-Means is used to group headlines according to their TF-IDF representations. Model performance is evaluated using accuracy, precision, recall, and F1-score.

The application also includes an investigation workflow that searches for related information, retrieves accessible article text, and uses a locally hosted language model to prepare an evidence-oriented summary. This investigation feature supplements, rather than replaces, the machine-learning prediction.

## 4. Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib
- Streamlit
- LangChain and LangChain-Ollama
- Ollama local models
- DuckDuckGo search through `ddgs`
- Requests and BeautifulSoup for article extraction

## 5. Expected Outcome

The expected outcome is an interactive application that predicts whether a headline is likely real or fake, explains factors influencing the prediction, displays analytical visualizations, explores headline patterns through clustering, and provides an evidence-oriented investigation workflow.

## 6. Conclusion

The project provides practical experience in natural language processing, supervised and unsupervised machine learning, model evaluation, explainable AI, and the integration of locally hosted language models. It is an educational prototype: its classifier output is not definitive verification of news authenticity, and investigation conclusions depend on the availability and quality of retrieved evidence.
