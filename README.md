# Text-Based Recommender System using Document Embeddings

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a text-based recommender system that predicts a user's star rating for a restaurant based on the text of their review. The core of the project is a comparative analysis of various document embedding techniques to find the most effective feature representation for a final regression model. The entire pipeline, from data preparation to hyperparameter tuning, is implemented in Python.

---

## 📋 Project Pipeline

1.  **Data Preparation:**
    * A specific subset of the **Yelp Open Dataset** was extracted, focusing on active users and popular restaurants.
    * The dataset of reviews was split into training, validation, and test sets based on the review creation date to simulate a real-world scenario.

2.  **Document Embedding Strategies:**
    The main goal was to find the best vector representation for each user review. Six different embedding strategies were trained and evaluated:
    * **word2vec** (Skip-Gram with Negative Sampling) + Word Vector Aggregation
    * **word2vec** (Continuous Bag of Words) + Word Vector Aggregation
    * **fastText** (Skip-Gram) + Word Vector Aggregation
    * **fastText** (CBOW) + Word Vector Aggregation
    * **doc2vec** (Distributed Memory - DM)
    * **doc2vec** (Distributed Bag of Words - DBOW)

3.  **Regression Modeling:**
    * The document embeddings generated from each of the six methods were used as input features for a regression model from the scikit-learn library.
    * The model was trained to predict the review's star rating (a value from 1 to 5).

4.  **Hyperparameter Tuning with Optuna:**
    * The **Optuna** library was used to perform efficient hyperparameter optimization.
    * A separate tuning process was run for each of the six embedding configurations to find the optimal set of parameters (e.g., embedding vector size, window size, regressor parameters).

5.  **Comprehensive Evaluation:**
    * The six final, optimized models were evaluated on the test set from two perspectives:
        * **Regression Performance:** Measured with metrics like R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
        * **Ranking Performance:** Measured with Information Retrieval metrics like Normalized Discounted Cumulative Gain (NDCG@k), Spearman's Rank Correlation, and Kendall's Tau.

---

## 🛠️ Technologies Used

* Python
* pandas
* NLTK
* gensim (for word2vec, fastText, and doc2vec)
* scikit-learn (for regression models)
* optuna (for hyperparameter tuning)
* torchmetrics (for evaluation)
* Jupyter Notebook

---

## 🚀 Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare the Dataset:**
    Download the Yelp Open Dataset and place the necessary JSON files in a `data` directory.

4.  **Execute the Analysis:**
    Open and run the cells in the main Jupyter Notebook to execute the entire pipeline.

---

## 📄 License
This project is licensed under the MIT License.