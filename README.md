# Sentiment_Analysis
This repository contains Python code for sentiment analysis using an LSTM (Long Short-Term Memory) neural network. The project involves training a model to classify the sentiment of airline tweets as positive, neutral, or negative.

Introductin to LSTM's

LSTM (Long Short-Term Memory) networks are particularly effective for sentiment analysis due to several key reasons:

1. **Handling Sequence Data**: Sentences and texts are inherently sequential data where the order of words matters. LSTMs excel in capturing dependencies and patterns in sequences, making them suitable for tasks like sentiment analysis where context is crucial.

2. **Long-Term Dependencies**: Traditional neural networks struggle with capturing long-term dependencies in sequences because of the vanishing gradient problem. LSTMs address this issue by maintaining a cell state that can retain information over long periods, selectively forgetting or updating information as new inputs are processed.

3. **Contextual Understanding**: Sentiment analysis requires understanding the context in which words and phrases are used. LSTMs, with their ability to capture context from preceding words and sentences, can infer nuances in language that affect sentiment, such as sarcasm, negation, or subtle sentiment shifts.

4. **Variable Length Input**: LSTMs can handle variable-length inputs by padding sequences to a fixed length, which is common in NLP tasks. This flexibility allows them to process texts of different lengths efficiently without losing information.

5. **Feature Extraction**: LSTMs automatically learn useful features from raw text data through their hierarchical structure of gates and cells. This feature extraction capability is beneficial in sentiment analysis, as it reduces the need for manual feature engineering.

6. **Effective for Deep Learning**: In deep learning architectures, LSTMs can be stacked to create deeper networks, allowing them to learn more complex patterns and representations in text data. This depth enhances their ability to model intricate relationships between words and sentiments.

Summary of the code
### Data Preparation and Preprocessing

1. **Loading Data**:
   - Reads a CSV file (`Tweets.csv`) containing tweets and their associated sentiment labels into a Pandas DataFrame (`df`).
   - Selects only the 'text' and 'airline_sentiment' columns from the DataFrame.

2. **Data Cleaning**:
   - Drops any rows with missing values (`NaN`) from the DataFrame.

3. **Label Encoding**:
   - Uses `LabelEncoder` from Scikit-learn to convert categorical sentiment labels ('positive', 'neutral', 'negative') into numeric labels (`0`, `1`, `2`).

4. **Train-Test Split**:
   - Splits the dataset into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets using `train_test_split` from Scikit-learn.
   - The text data (`X_train` and `X_test`) is used for training and evaluating the model, while `y_train` and `y_test` contain the corresponding numeric sentiment labels.

### Text Tokenization and Padding

5. **Tokenization**:
   - Uses `Tokenizer` from Keras to tokenize the text data (`X_train`) and fit it on the training set.

6. **Sequence Padding**:
   - Converts the tokenized sequences (`X_train_seq` and `X_test_seq`) into sequences of uniform length (`max_seq_length`) using `pad_sequences` from Keras. This ensures all input sequences have the same length for model training.

### Model Building

7. **LSTM Model Definition**:
   - Constructs a Sequential model (`model`) using Keras.
   - Adds layers: 
     - Embedding layer for word embeddings.
     - SpatialDropout1D layer for regularization.
     - LSTM layer with dropout to learn from sequences.
     - Dense layer with softmax activation for multi-class classification (3 classes: positive, neutral, negative).

8. **Model Compilation**:
   - Compiles the model with 'sparse_categorical_crossentropy' loss function for multi-class classification and 'adam' optimizer.
   - Specifies 'accuracy' as the metric to monitor during training.

### Model Training

9. **Training**:
   - Trains the LSTM model (`model`) on the training data (`X_train_pad`, `y_train`) for a specified number of epochs (`epochs`) and batch size (`batch_size`).
   - Validates the model on the testing data (`X_test_pad`, `y_test`).

10. **Model Evaluation**:
    - Evaluates the trained model on the testing data and prints the test accuracy.

11. **Model and Tokenizer Saving**:
    - Saves the trained LSTM model (`model`) as a `.h5` file ('sentiment_lstm_model.h5').
    - Saves the `Tokenizer` object (`tokenizer`) used for tokenization as a pickle file ('tokenizer.pickle').

### Sentiment Prediction on New Dataset

12. **Loading Trained Model and Tokenizer**:
    - Loads the saved model ('sentiment_lstm_model.h5') and tokenizer ('tokenizer.pickle').

13. **Processing New Dataset**:
    - Reads another CSV file (`01tweet.csv`) containing new tweets into a Pandas DataFrame (`df_01tweet`).
    - Retrieves only the 'text' column from this DataFrame.

14. **Tokenization and Padding for New Data**:
    - Tokenizes and pads the sequences of new tweets (`texts_01tweet`) using the loaded tokenizer and `pad_sequences`.

15. **Sentiment Prediction**:
    - Uses the loaded LSTM model to predict sentiment probabilities (`predictions`) for the new tweets.
    - Determines the predicted sentiment labels (`predicted_labels`) by selecting the class with the highest probability.

### Clustering and Rating

16. **Clustering Using K-Means**:
    - Applies K-Means clustering to the predicted sentiment probabilities (`predictions`) to group tweets into clusters.
    - Uses the elbow method to determine the optimal number of clusters (`optimal_k`) based on the sum of squared distances.

17. **Assigning Ratings**:
    - Assigns ratings (`1`, `2`, `3`, etc.) to each tweet based on the cluster it belongs to.

18. **Output**:
    - Constructs a DataFrame (`result_df`) with columns for tweet text, predicted sentiment, assigned cluster, and rating.
    - Prints each tweet along with its predicted sentiment and assigned rating.

### Visualization

19. **Elbow Method Visualization**:
    - Plots the sum of squared distances for different numbers of clusters (`K`) to visualize the optimal number of clusters using the elbow method.

### Summary

This code performs end-to-end tasks for sentiment analysis:
- It prepares the data by cleaning and encoding sentiment labels.
- Builds and trains an LSTM model for sentiment classification.
- Saves the trained model and tokenizer for future use.
- Predicts sentiments on new tweets and assigns ratings based on clustering.
- Visualizes the elbow method to determine the optimal number of clusters.

