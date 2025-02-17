import os
import urllib.request
import zipfile
import pandas as pd
import tarfile

def download_and_extract_sst2(file_path):
    """
    Downloads and extracts the SST-2 dataset if not already present.
    """
    url = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
    zip_path = os.path.join(file_path, "stanfordSentimentTreebank.zip")
    extract_path = os.path.join(file_path, "stanfordSentimentTreebank")
    
    if not os.path.exists(extract_path):
        print("Downloading SST-2 dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(file_path)
        os.remove(zip_path)
        print("Download and extraction complete.")
    else:
        print("SST-2 dataset already exists.")

def download_and_extract_imdb(file_path):
    """
    Downloads and extracts the IMDB dataset if not already present.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = os.path.join(file_path, "aclImdb_v1.tar.gz")
    extract_path = os.path.join(file_path, "aclImdb")
    
    os.makedirs(file_path, exist_ok=True)
    
    if not os.path.exists(extract_path):
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=file_path)
        os.remove(tar_path)
        print("Download and extraction complete.")
    else:
        print("IMDB dataset already exists.")

def preprocess_imdb_data(file_path, save_path):
    """
    Processes IMDB dataset and saves the data as CSV files.
    """
    dataset_path = os.path.join(file_path, "aclImdb")
    os.makedirs(save_path, exist_ok=True)
    
    def process_folder(folder_path, label, output_file):
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    data.append({'sentence': content, 'label': label})
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, output_file), index=False)
    
    process_folder(os.path.join(dataset_path, "train/pos"), 1, "IMDB_train_pos.csv")
    process_folder(os.path.join(dataset_path, "train/neg"), 0, "IMDB_train_neg.csv")
    process_folder(os.path.join(dataset_path, "test/pos"), 1, "IMDB_test_pos.csv")
    process_folder(os.path.join(dataset_path, "test/neg"), 0, "IMDB_test_neg.csv")

def read_csv_data(file_path, dataset):
    """
    Reads CSV files for the specified dataset and returns training,
    development, and test data as lists.
    """
    if dataset == "imdb":
        download_and_extract_imdb(file_path)
        save_path = os.path.join(file_path, "aclImdb")
        preprocess_imdb_data(file_path, save_path)
        train_dev_neg_path = os.path.join(save_path, "IMDB_train_neg.csv")
        train_dev_pos_path = os.path.join(save_path, "IMDB_train_pos.csv")
        train_dev_pos = pd.read_csv(train_dev_neg_path)
        train_dev_neg = pd.read_csv(train_dev_pos_path)
        train_df = pd.concat([train_dev_pos[:10000], train_dev_neg[:10000]], ignore_index=True)
        dev_df = pd.concat([train_dev_pos[10000:], train_dev_neg[10000:]], ignore_index=True)
        test_neg_path = os.path.join(save_path, "IMDB_test_neg.csv")
        test_pos_path = os.path.join(save_path, "IMDB_test_pos.csv")
        test_pos = pd.read_csv(test_neg_path)
        test_neg = pd.read_csv(test_pos_path)
        test_df = pd.concat([test_pos, test_neg], ignore_index=True)
        x_train = train_df['sentence']
        y_train = train_df['label']
        x_dev = dev_df['sentence']
        y_dev = dev_df['label']
        x_test = test_df['sentence']
        y_test = test_df['label']
    
    elif dataset == 'sst2':
        download_and_extract_sst2(file_path) 
        sst2_path = os.path.join(file_path, "stanfordSentimentTreebank")
        
        dataset_sentences_path = os.path.join(sst2_path, "datasetSentences.txt")
        dataset_split_path = os.path.join(sst2_path, 'datasetSplit.txt')
        dictionary_path = os.path.join(sst2_path, 'dictionary.txt')
        sentiment_labels_path = os.path.join(sst2_path, 'sentiment_labels.txt')
        
        sentences_df = pd.read_csv(dataset_sentences_path, sep='\t')
        sentences_df.columns = ['sentence_index', 'sentence']
        splits_df = pd.read_csv(dataset_split_path, sep=',')
        splits_df.columns = ['sentence_index', 'splitset_label']
        sentences_df = sentences_df.merge(splits_df, on='sentence_index')
        dictionary_df = pd.read_csv(dictionary_path, sep='|', names=['phrase', 'phrase_id'])
        sentiment_labels_df = pd.read_csv(sentiment_labels_path, sep='|')
        sentiment_labels_df.columns = ['phrase_id', 'sentiment_value']
        dictionary_df = dictionary_df.merge(sentiment_labels_df, on='phrase_id')
        sentences_df = sentences_df.merge(dictionary_df, left_on='sentence', right_on='phrase')
        
        def sentiment_class(sentiment_value):
            if sentiment_value <= 0.4: 
                return 'negative'
            elif sentiment_value > 0.6:
                return 'positive'
        
        sentences_df['sentiment_class'] = sentences_df['sentiment_value'].apply(sentiment_class)
        sentences_df = sentences_df.dropna(subset=['sentiment_class'])
        train_df = sentences_df[sentences_df['splitset_label'] == 1]
        test_df = sentences_df[sentences_df['splitset_label'] == 2]
        dev_df = sentences_df[sentences_df['splitset_label'] == 3]
        
        def balance_dataset(df):
            positive_df = df[df['sentiment_class'] == 'positive']
            negative_df = df[df['sentiment_class'] == 'negative']
            min_size = min(len(positive_df), len(negative_df))
            positive_truncated = positive_df.iloc[:min_size]
            negative_truncated = negative_df.iloc[:min_size]
            return pd.concat([positive_truncated, negative_truncated])

        train_df = balance_dataset(train_df)
        train_df['sentiment_class'] = train_df['sentiment_class'].map({'negative': 0, 'positive': 1})
        dev_df['sentiment_class'] = dev_df['sentiment_class'].map({'negative': 0, 'positive': 1})
        test_df['sentiment_class'] = test_df['sentiment_class'].map({'negative': 0, 'positive': 1})
        
        x_train = train_df['sentence']
        y_train = train_df['sentiment_class']
        x_dev = dev_df['sentence']
        y_dev = dev_df['sentiment_class']
        x_test = test_df['sentence']
        y_test = test_df['sentiment_class']
    
    return x_train.to_list(), y_train.to_list(), x_dev.to_list(), y_dev.to_list(), x_test.to_list(), y_test.to_list()
