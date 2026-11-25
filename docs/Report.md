# NewsLens AI: Fake News Headline Detection and Investigation

**Course:** Artificial Intelligence Lab  
**Project Leader / Backend:** Ahmed Raza (K24-3086)  
**Frontend Developer:** Shahmir Ali (K24-3048)  
**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES)

---

## Abstract

NewsLens AI is an educational application for analyzing news headlines through machine learning and an evidence-oriented investigation workflow. Its supervised learning component represents cleaned headlines using TF-IDF features and compares Multinomial Naive Bayes with Logistic Regression. It also includes K-Means clustering, influential-word explanations, a Streamlit interface, and an investigation assistant that can search for related information and extract accessible article text using locally hosted language models.

The supplied dataset summary contains 20,000 headlines: 10,056 labelled fake and 9,944 labelled real. On the reported evaluation, Naive Bayes achieved 0.4893 accuracy and Logistic Regression achieved 0.4485 accuracy. Both results are below the 0.5028 majority-class baseline implied by the supplied class counts. These results indicate that the current headline-only classification setup has weak predictive performance and should not be used to determine whether a real-world claim is true. The project is therefore presented as a prototype that demonstrates an end-to-end AI workflow while making its limitations explicit.

## 1. Introduction

The rapid circulation of online news makes it difficult for readers to assess the reliability of short headlines. A headline may omit context, exaggerate an event, or use language that resembles previously labelled examples. Automated text classification can help identify patterns in labelled data, but a classifier cannot independently establish whether a news claim is true.

NewsLens AI addresses this problem as a learning-oriented prototype. It combines a conventional machine-learning classifier with explanatory features, topic exploration, and a separate evidence-gathering workflow. The machine-learning result and the investigation result are kept conceptually distinct: the first estimates a label based on patterns learned from the dataset, while the second gathers external material relevant to the submitted claim.

## 2. Problem Formulation

The supervised learning task is binary text classification. Given a headline \(h\), the classifier produces one of two dataset labels:

- **Fake (0):** the headline belongs to the fake-labelled class in the dataset.
- **Real (1):** the headline belongs to the real-labelled class in the dataset.

The output is a model prediction, not a verified factual judgement. The model learns statistical associations between headline text and training labels. It does not independently verify events, dates, named entities, sources, or the full article context.

The investigation component addresses a different question: what relevant material can be found to support, contradict, partially support, or leave a submitted claim inconclusive? Search relevance is not proof, and retrieved content must be assessed critically.

## 3. Objectives and Scope

The project objectives are:

1. Load, validate, clean, and analyze a labelled headline dataset.
2. Convert text into numerical TF-IDF features.
3. Train and compare Naive Bayes and Logistic Regression classifiers.
4. Evaluate models using accuracy, precision, recall, and F1-score.
5. Explain individual predictions through influential terms.
6. Explore groups of similar headlines using K-Means.
7. Provide an interactive Streamlit interface.
8. Add a separate AI-assisted investigation workflow using web search and article extraction.

The system is limited to headline-level classification. It is not a professional fact-checking service, does not guarantee access to every publisher, and cannot guarantee that search results are complete or unbiased.

## 4. Dataset Description

The supplied dataset summary reports the following distribution:

| Dataset statistic | Count |
|---|---:|
| Total headlines | 20,000 |
| Fake headlines | 10,056 |
| Real headlines | 9,944 |

The two classes are close to balanced. Fake headlines account for approximately 50.28% of the supplied records and real headlines account for approximately 49.72%. The project proposal identifies the dataset as a publicly available labelled news dataset. The dataset source recorded for the project is the Kaggle dataset **Fake News Detection Dataset**: https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset.

The backend expects at least two CSV columns: `title` and `label`. The implementation normalizes the labels into numeric values, with fake mapped to 0 and real mapped to 1. It cleans headline text, removes rows with unusable cleaned text, and removes duplicate cleaned headlines. Consequently, the number of records used for model training can be lower than the original 20,000 if cleaning or duplicate removal removes rows. The exact post-cleaning count was not supplied and is not asserted here.

## 5. Methodology

### 5.1 Text cleaning

The backend converts headline text to lowercase, removes URLs, removes characters outside the English alphabet and whitespace, collapses repeated whitespace, and trims leading or trailing spaces. Missing titles are replaced with empty strings before cleaning. Rows with empty cleaned headlines are excluded, and duplicate cleaned headlines are removed.

### 5.2 TF-IDF feature extraction

The cleaned headlines are transformed into sparse numerical vectors using `TfidfVectorizer`. The implementation uses English stop-word removal, a maximum vocabulary size of 20,000 features, unigrams and bigrams, `min_df=2`, `max_df=0.95`, and sublinear term-frequency scaling. The vectorizer is fitted on the training data and then applied to the test data, preventing the test set from determining the vocabulary.

### 5.3 Supervised classifiers

Two classifiers are trained using the same TF-IDF representation:

- **Multinomial Naive Bayes:** configured with `alpha=0.5`.
- **Logistic Regression:** configured with `max_iter=1000`, `C=2.0`, and `random_state=42`.

Both models are evaluated on the same held-out test set. The backend uses scikit-learn metrics for accuracy, precision, recall, F1-score, confusion matrix, and a class-wise classification report. The reported precision, recall, and F1 values are the default binary metrics with label 1 (Real) treated as the positive class.

### 5.4 Train/test split

The code uses an 80/20 train/test split with `random_state=42` and stratification by label. Stratification aims to preserve the class proportions in both subsets. Because cleaning and duplicate removal occur before splitting, the final subset sizes depend on the number of rows remaining after preprocessing.

### 5.5 K-Means clustering

The backend transforms the cleaned headlines using the fitted TF-IDF vectorizer and applies K-Means with six clusters, `random_state=42`, and `n_init=10`. The ten highest-weight terms from each cluster centre are used as indicative topic terms. These terms help describe patterns in the clusters; they do not establish that a cluster represents a verified real-world topic.

## 6. System Design and Implementation

### 6.1 Main processing flow

The system follows this sequence:

1. Load the CSV dataset and validate the required columns.
2. Normalize labels and clean headline text.
3. Remove unusable and duplicate cleaned headlines.
4. Generate exploratory dataset statistics.
5. Split the data into training and test subsets.
6. Fit TF-IDF on training headlines and transform both subsets.
7. Train Naive Bayes and Logistic Regression.
8. Calculate evaluation metrics and retain the trained models.
9. Run K-Means clustering and extract representative terms.
10. Make the trained components available to the Streamlit application.

### 6.2 Prediction explanations

For a submitted headline, the backend applies the same cleaning and TF-IDF transformation used during training. It obtains the predicted class and class probabilities. It then calculates term-level contributions using the model coefficients for Logistic Regression or the difference in class log probabilities for Naive Bayes. The highest-scoring terms are returned with their estimated influence toward the Real or Fake class.

These explanations describe how the model's learned weights contribute to its output. They are not explanations of why the headline is factually true or false.

### 6.3 AI-assisted investigation

The backend includes an investigation agent built around a local Ollama language model (`qwen3.5:2b`). It can call two tools:

- `search_news`: searches for related news using DuckDuckGo through the `ddgs` package. Search snippets are embedded using the local `nomic-embed-text:latest` model and ranked by cosine similarity to the query.
- `fetch_article`: requests an article URL and uses BeautifulSoup to remove common page elements and extract readable text.

The investigation prompt asks the model to search the exact headline and related wording, identify relevant reporting, inspect accessible articles, and assign one of four evidence-oriented verdicts: SUPPORTED, CONTRADICTED, PARTIALLY SUPPORTED, or INCONCLUSIVE. The prompt explicitly warns that machine-learning predictions and semantic similarity are not proof, and that sources, quotations, or evidence must not be invented.

### 6.4 User interface

The proposal specifies Streamlit as the interactive interface. The interface connects user interaction with the backend's prediction, analysis, clustering, and investigation functionality. The application is intended to make the underlying techniques accessible through a single project demonstration.

## 7. Experimental Setup

The model comparison uses the same preprocessed headline data, train/test split, and TF-IDF representation for both classifiers. The code configuration is:

| Setting | Value |
|---|---|
| Test fraction | 0.20 |
| Random state | 42 |
| Stratified split | Yes |
| TF-IDF maximum features | 20,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Sublinear TF | True |
| Naive Bayes smoothing (`alpha`) | 0.5 |
| Logistic Regression iterations | 1,000 |
| Logistic Regression `C` | 2.0 |
| K-Means clusters | 6 |

The numerical results below are the values supplied for this report. The original run logs, confusion matrices, per-class reports, and exact post-cleaning sample count were not supplied. They are therefore not reconstructed or fabricated.

## 8. Experimental Results

### 8.1 Model comparison

| Metric | Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 0.4893 | 0.4485 |
| Precision | 0.4910 | 0.4605 |
| Recall | 0.4853 | 0.4662 |
| F1-score | 0.4875 | 0.4736 |

Naive Bayes performs better than Logistic Regression on all four supplied metrics. Its accuracy is 48.93%, compared with 44.85% for Logistic Regression. Its F1-score is 0.4875, compared with 0.4736. However, the difference does not make Naive Bayes a strong classifier: both models remain below the simple majority-class baseline.

### 8.2 Baseline comparison

Using the supplied original class counts, always predicting the majority class (Fake) would yield:

\[
\frac{10,056}{20,000}=0.5028=50.28\%
\]

Naive Bayes accuracy is 1.35 percentage points below this baseline. Logistic Regression accuracy is 5.43 percentage points below it. This comparison indicates that the current models do not outperform a trivial majority-class strategy on the supplied figures.

The baseline is calculated from the provided 20,000-record class distribution. If preprocessing changes the class counts by removing duplicate or empty headlines, a baseline calculated on the final cleaned dataset should also be reported for a fully matched comparison.

### 8.3 Interpretation

The results suggest that the current feature representation and headline-only dataset are insufficient for reliable discrimination in this experiment. Possible factors include limited context in headlines, noisy or weak labels, overlap in wording between classes, duplicate or near-duplicate records, dataset-specific language patterns, and preprocessing choices. These are plausible explanations rather than proven causes; further experiments are required to identify the main source of error.

Because the positive class is Real (label 1) in the backend metric calls, the supplied precision, recall, and F1-score values should be interpreted as binary metrics for the Real class. A future evaluation should also report per-class metrics and a confusion matrix to show how often fake and real headlines are misclassified.

## 9. Testing and Evaluation

The implementation includes validation for the presence of the `title` and `label` columns, checks that both classes are present, handles missing headline text, removes unusable cleaned text, and rejects empty user input during prediction. The prediction function also checks that the trained model contains the expected binary classes.

The model evaluation code calculates accuracy, precision, recall, F1-score, confusion matrix, and a classification report. The investigation workflow handles search and article extraction failures by returning error messages rather than silently treating failed retrieval as evidence.

The supplied results confirm that the two models were evaluated and that four summary metrics were recorded. No separate test log, UI test checklist, article-retrieval success rate, or confusion matrix was provided for this report. Those results should be added after they are actually run.

## 10. Limitations and Future Improvements

### 10.1 Current limitations

- The classifier uses headlines only and does not understand the full article context.
- The reported model scores are low and fall below the majority-class baseline calculated from the supplied dataset distribution.
- Dataset labels may contain noise or reflect the dataset's original collection process.
- The exact number of records remaining after cleaning and deduplication was not supplied.
- The supplied aggregate metrics do not show per-class errors or the confusion matrix.
- Search results and article extraction may be incomplete, unavailable, or biased toward indexed sources.
- Semantic similarity ranks relevance; it does not establish factual correctness.
- Local language-model output can still be incorrect and requires human review.
- The application is an educational prototype, not a substitute for professional fact-checking.

### 10.2 Future improvements

1. Inspect the dataset for duplicates, label quality, source leakage, and class-specific patterns.
2. Evaluate a majority-class baseline and dummy classifier on the exact post-cleaning test set.
3. Report per-class precision, recall, F1-score, confusion matrix, and cross-validation results.
4. Test alternative n-gram ranges, vocabulary sizes, regularization strengths, and Naive Bayes smoothing values using a validation procedure.
5. Compare TF-IDF models with stronger contextual text representations.
6. Add full-article analysis where reliable article text is available.
7. Record source URLs, retrieval timestamps, and evidence excerpts in investigation reports.
8. Test the interface with empty inputs, invalid URLs, unavailable Ollama models, and search failures.
9. Preserve a reproducible experiment script and save model evaluation outputs.

## 11. Team Contributions

| Member | Roll number | Contribution |
|---|---|---|
| Ahmed Raza | K24-3086 | Backend development: dataset loading and preprocessing, TF-IDF feature extraction, Naive Bayes and Logistic Regression training/evaluation, prediction explanations, EDA and K-Means clustering, and the Ollama/LangChain investigation and retrieval workflow. |
| Shahmir Ali | K24-3048 | Frontend development: Streamlit interface and integration of the user-facing application with backend functionality. |

This division reflects the project contribution split provided by the team: Ahmed Raza handled the backend and Shahmir Ali handled the frontend.

## 12. Conclusion

NewsLens AI demonstrates how conventional NLP and machine-learning methods can be combined with clustering, prediction explanations, an interactive interface, and an AI-assisted evidence-gathering workflow. The implementation compares Multinomial Naive Bayes and Logistic Regression using TF-IDF features and explores headline groups through K-Means.

The reported evaluation is an important limitation: Naive Bayes achieved 48.93% accuracy and Logistic Regression achieved 44.85%, both below the 50.28% majority-class baseline derived from the supplied class counts. The current classifier therefore cannot be considered reliable. The project's value is its end-to-end educational implementation and the opportunity it provides to investigate data quality, model evaluation, evidence retrieval, and responsible communication of uncertainty.

## References

1. Mahdi Mashayekhi, *Fake News Detection Dataset*, Kaggle. https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset
2. scikit-learn documentation, *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
3. scikit-learn documentation, *MultinomialNB*. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
4. scikit-learn documentation, *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
5. scikit-learn documentation, *KMeans*. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
6. Streamlit documentation. https://docs.streamlit.io/
7. Ollama documentation. https://ollama.com/
