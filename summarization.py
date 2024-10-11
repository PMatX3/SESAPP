import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import networkx as nx

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, summary_ratio=0.5):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    num_sentences_to_include = max(1, int(total_sentences * summary_ratio))
    
    # Preprocess sentences
    stop_words = set(stopwords.words('english'))
    def preprocess_sentence(sentence):
        words = word_tokenize(sentence.lower())
        return [word for word in words if word.isalnum() and word not in stop_words]

    processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]
    sentence_similarity_matrix = create_similarity_matrix(processed_sentences)

    # Rank sentences
    sentence_ranking = nx.pagerank(nx.from_numpy_array(sentence_similarity_matrix))
    ranked_sentences = sorted(((sentence_ranking[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top-ranked sentences
    summarized_sentences = [sentence for rank, sentence in ranked_sentences[:num_sentences_to_include]]
    summary = ' '.join(summarized_sentences)
    
    return summary

def create_similarity_matrix(sentences):
    # Create a TF-IDF matrix
    count_vectorizer = CountVectorizer().fit_transform([' '.join(sentence) for sentence in sentences])
    tfidf_transformer = TfidfTransformer().fit(count_vectorizer)
    tfidf_matrix = tfidf_transformer.transform(count_vectorizer)
    
    # Calculate the similarity matrix
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    return similarity_matrix
