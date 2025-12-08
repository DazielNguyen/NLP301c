# **NLP301c - Xử Lý Ngôn Ngữ Tự Nhiên - Kỳ 6 - FALL 2025**

Chào mừng bạn đến với repository học tập môn Natural Language Processing! 👋

## **Giới thiệu**

Xin chào! Mình là **Nguyễn Văn Anh Duy**, sinh viên ngành Trí Tuệ Nhân Tạo tại Đại học FPT TP.HCM. Repository này lưu trữ toàn bộ tài liệu học tập, bài tập và ghi chú của mình trong môn NLP301c.

## **Cấu trúc Repository**

### **Module 01 - Natural Language Processing with Classification and Vector Spaces**

_Xử lý ngôn ngữ tự nhiên với phân loại và không gian vector_

#### **Week 1: Sentiment Analysis with Logistic Regression**

- Phân tích tình cảm (Sentiment Analysis) sử dụng Logistic Regression
- Biểu diễn văn bản dưới dạng vector
- Xây dựng bộ phân loại tích cực/tiêu cực cho tweets
- Supervised Machine Learning và hàm chi phí
- **Bài tập**: Xây dựng model phân loại sentiment cho Twitter data

#### **Week 2: Sentiment Analysis with Naive Bayes**

- Probability và Bayes' Rule (Quy tắc Bayes)
- Conditional Probability (Xác suất có điều kiện)
- Naive Bayes Classifier cho phân loại văn bản
- Laplacian Smoothing để xử lý từ chưa gặp
- **Bài tập**: Triển khai Naive Bayes cho sentiment analysis

#### **Week 3: Vector Space Models**

- Word by Word và Word by Document design
- Co-occurrence matrices (Ma trận đồng xuất hiện)
- Euclidean Distance và Cosine Similarity
- PCA (Principal Component Analysis) để giảm chiều
- **Bài tập**: Xây dựng vector space model để tìm từ tương tự

#### **Week 4: Machine Translation and Document Search**

- Transforming word vectors giữa các ngôn ngữ
- Locality Sensitive Hashing (LSH)
- K-Nearest Neighbors search
- Dịch máy cơ bản sử dụng word embeddings
- **Bài tập**: Xây dựng hệ thống dịch thuật Anh-Pháp đơn giản

---

### **Module 02 - Natural Language Processing with Probabilistic Models**

_Xử lý ngôn ngữ tự nhiên với các mô hình xác suất_

#### **Week 1: Autocorrect and Minimum Edit Distance**

- Xây dựng model Autocorrect (tự động sửa lỗi chính tả)
- Minimum Edit Distance (Khoảng cách chỉnh sửa tối thiểu)
- Dynamic Programming cho edit distance
- Insert, Delete, Switch, Replace operations
- **Bài tập**: Triển khai autocorrect system

#### **Week 2: Part of Speech Tagging and Hidden Markov Models**

- Part of Speech (POS) Tagging
- Markov Chains (Chuỗi Markov)
- Hidden Markov Models (HMMs)
- Viterbi Algorithm
- Named Entity Recognition
- **Bài tập**: Xây dựng POS tagger sử dụng HMM

#### **Week 3: Autocomplete and Language Models**

- N-grams (Unigrams, Bigrams, Trigrams)
- Language Models và tính xác suất câu
- Perplexity để đánh giá model
- Smoothing techniques cho unseen n-grams
- Out-of-vocabulary words handling
- **Bài tập**: Xây dựng autocomplete system sử dụng N-gram model

#### **Week 4: Word Embeddings with Neural Networks**

- Basic Word Representations (One-hot vectors)
- Word2Vec và Continuous Bag of Words (CBOW)
- Training word embeddings
- Word analogies và semantic relationships
- GloVe embeddings
- **Bài tập**: Training word embeddings từ text corpus

---

### **Module 03 - Natural Language Processing with Sequence Models**

_Xử lý ngôn ngữ tự nhiên với các mô hình chuỗi_

#### **Week 1: Recurrent Neural Networks for Language Modeling**

- Neural Networks cho Sentiment Analysis
- Recurrent Neural Networks (RNNs)
- Forward Propagation trong RNNs
- Dense Layers và ReLU activation
- Backpropagation Through Time (BPTT)
- **Bài tập**: Xây dựng RNN cho sentiment classification

#### **Week 2: LSTMs and Named Entity Recognition**

- Long Short-Term Memory (LSTM) networks
- Vanishing và Exploding Gradients problem
- LSTM Architecture (gates: forget, input, output)
- Named Entity Recognition (NER)
- Gated Recurrent Units (GRUs)
- **Bài tập**: Triển khai LSTM cho NER task

#### **Week 3: Siamese Networks**

- Siamese Network Architecture
- Similarity và distance metrics
- One-shot learning
- Duplicate question detection
- Triplet Loss function
- **Bài tập**: Xây dựng Siamese network cho question similarity

---

### **Module 04 - Natural Language Processing with Attention Models**

_Xử lý ngôn ngữ tự nhiên với các mô hình attention_

#### **Week 1: Neural Machine Translation**

- Sequence-to-Sequence (Seq2Seq) models
- Encoder-Decoder architecture
- Attention Mechanism
- Teacher Forcing
- BLEU Score để đánh giá dịch máy
- **Bài tập**: Xây dựng neural machine translation system

#### **Week 2: Text Summarization**

- Transformers Architecture
- Multi-Head Attention
- Positional Encoding
- Self-Attention mechanism
- Text Summarization (extractive và abstractive)
- **Bài tập**: Triển khai text summarization với Transformers

#### **Week 3: Question Answering**

- Transfer Learning trong NLP
- BERT (Bidirectional Encoder Representations from Transformers)
- T5 (Text-to-Text Transfer Transformer)
- Fine-tuning pre-trained models
- Context-based Question Answering
- **Bài tập**: Fine-tune BERT cho question answering task

---

## **Tài liệu bổ sung**

- **`utils.py`**: Các hàm tiện ích dùng chung cho các bài tập
- **`requirements.txt`**: Danh sách thư viện Python cần thiết
- **`npl-env/`**: Virtual environment cho project
- **`Ôn tập PE/`**: Tài liệu ôn tập cho các kỳ thi Practice Exam

## **Công nghệ sử dụng**

- **Python 3.11**: Ngôn ngữ lập trình chính
- **NumPy**: Tính toán số học và ma trận
- **NLTK**: Natural Language Toolkit
- **TensorFlow/Keras**: Deep Learning frameworks
- **Jupyter Notebook**: Môi trường phát triển tương tác

## **Kết nối với mình**

Nếu bạn muốn trao đổi hoặc có thắc mắc gì, đừng ngại liên hệ nhé:

- **Email**: duynguyenvananh@gmail.com
- **Phone**: 0387883041
- **GitHub**: [@DazielNguyen](https://github.com/DazielNguyen)
- **Linkedin**: [Văn Anh Duy](https://www.linkedin.com/in/dazielvad/)

## License

Copyright © 2025 Nguyen Van Anh Duy - AWS FCJ Internship Report

---

**Cảm ơn bạn đã ghé thăm! Have a good day!**
