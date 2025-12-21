# **NLP301c - Xử Lý Ngôn Ngữ Tự Nhiên - Kỳ 6 - FALL 2025**

![Coursera_NLP](https://github.com/DazielNguyen/NLP301c/blob/main/NLP_Coursera.png)

## **Chào mừng bạn đến với repository học tập môn Natural Language Processing!** 👋

## **Giới thiệu**

Xin chào! Mình là **Nguyễn Văn Anh Duy**, sinh viên ngành Trí Tuệ Nhân Tạo tại Đại học FPT TP.HCM. Repository này lưu trữ toàn bộ tài liệu học tập, bài tập và ghi chú của mình trong môn NLP301c.

## **Về khóa học**

Khóa học **Natural Language Processing Specialization** được phát triển bởi:

- **[Younes Bensouda Mourri](https://www.linkedin.com/in/younes-bensouda-mourri/)** - Instructor, Stanford University
- **[Łukasz Kaiser](https://www.linkedin.com/in/lukasz-kaiser/)** - Staff Research Scientist, Google Brain
- **[Eddy Shyu](https://www.linkedin.com/in/eddy-shyu/)** - Curriculum Product Manager, deeplearning.ai
- **[Andrew Ng](https://www.linkedin.com/in/andrewyng/)** - Founder, DeepLearning.AI & Co-founder, Coursera

Khóa học này là một phần của **DeepLearning.AI** và được cung cấp trên nền tảng **Coursera**.

## **Cấu trúc Repository**

### [Module 01 - Natural Language Processing with Classification and Vector Spaces](https://github.com/DazielNguyen/NLP301c/tree/main/Module%2001)

_Xử lý ngôn ngữ tự nhiên với phân loại và không gian vector_

#### [Week 1: Sentiment Analysis with Logistic Regression](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Lessons_Notes_Module_01/Week_01.md)

- Phân tích tình cảm (Sentiment Analysis) sử dụng Logistic Regression
- Biểu diễn văn bản dưới dạng vector
- Xây dựng bộ phân loại tích cực/tiêu cực cho tweets
- Supervised Machine Learning và hàm chi phí
- **Bài tập**: Xây dựng model phân loại sentiment cho Twitter data

#### [Week 2: Sentiment Analysis with Naive Bayes](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Lessons_Notes_Module_01/Week_02.md)

- Probability và Bayes' Rule (Quy tắc Bayes)
- Conditional Probability (Xác suất có điều kiện)
- Naive Bayes Classifier cho phân loại văn bản
- Laplacian Smoothing để xử lý từ chưa gặp
- **Bài tập**: Triển khai Naive Bayes cho sentiment analysis

#### [Week 3: Vector Space Models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Lessons_Notes_Module_01/Week_03.md)

- Word by Word và Word by Document design
- Co-occurrence matrices (Ma trận đồng xuất hiện)
- Euclidean Distance và Cosine Similarity
- PCA (Principal Component Analysis) để giảm chiều
- **Bài tập**: Xây dựng vector space model để tìm từ tương tự

#### [Week 4: Machine Translation and Document Search](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Lessons_Notes_Module_01/Week_04.md)

- Transforming word vectors giữa các ngôn ngữ
- Locality Sensitive Hashing (LSH)
- K-Nearest Neighbors search
- Dịch máy cơ bản sử dụng word embeddings
- **Bài tập**: Xây dựng hệ thống dịch thuật Anh-Pháp đơn giản

---

### [Module 02 - Natural Language Processing with Probabilistic Models](https://github.com/DazielNguyen/NLP301c/tree/main/Module%2002)

_Xử lý ngôn ngữ tự nhiên với các mô hình xác suất_

#### [Week 1: Autocorrect and Minimum Edit Distance](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Lessons_Notes_Module_02/Week_01.md)

- Xây dựng model Autocorrect (tự động sửa lỗi chính tả)
- Minimum Edit Distance (Khoảng cách chỉnh sửa tối thiểu)
- Dynamic Programming cho edit distance
- Insert, Delete, Switch, Replace operations
- **Bài tập**: Triển khai autocorrect system

#### [Week 2: Part of Speech Tagging and Hidden Markov Models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Lessons_Notes_Module_02/Week_02.md)

- Part of Speech (POS) Tagging
- Markov Chains (Chuỗi Markov)
- Hidden Markov Models (HMMs)
- Viterbi Algorithm
- Named Entity Recognition
- **Bài tập**: Xây dựng POS tagger sử dụng HMM

#### [Week 3: Autocomplete and Language Models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Lessons_Notes_Module_02/Week_03.md)

- N-grams (Unigrams, Bigrams, Trigrams)
- Language Models và tính xác suất câu
- Perplexity để đánh giá model
- Smoothing techniques cho unseen n-grams
- Out-of-vocabulary words handling
- **Bài tập**: Xây dựng autocomplete system sử dụng N-gram model

#### [Week 4: Word Embeddings with Neural Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Lessons_Notes_Module_02/Week_04.md)

- Basic Word Representations (One-hot vectors)
- Word2Vec và Continuous Bag of Words (CBOW)
- Training word embeddings
- Word analogies và semantic relationships
- GloVe embeddings
- **Bài tập**: Training word embeddings từ text corpus

---

### [Module 03 - Natural Language Processing with Sequence Models](https://github.com/DazielNguyen/NLP301c/tree/main/Module%2003)

_Xử lý ngôn ngữ tự nhiên với các mô hình chuỗi_

#### [Week 1: Recurrent Neural Networks for Language Modeling](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Lessons_Notes_Module_03/Week_01.md)

- Neural Networks cho Sentiment Analysis
- Recurrent Neural Networks (RNNs)
- Forward Propagation trong RNNs
- Dense Layers và ReLU activation
- Backpropagation Through Time (BPTT)
- **Bài tập**: Xây dựng RNN cho sentiment classification

#### [Week 2: LSTMs and Named Entity Recognition](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Lessons_Notes_Module_03/Week_02.md)

- Long Short-Term Memory (LSTM) networks
- Vanishing và Exploding Gradients problem
- LSTM Architecture (gates: forget, input, output)
- Named Entity Recognition (NER)
- Gated Recurrent Units (GRUs)
- **Bài tập**: Triển khai LSTM cho NER task

#### [Week 3: Siamese Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Lessons_Notes_Module_03/Week_03.md)

- Siamese Network Architecture
- Similarity và distance metrics
- One-shot learning
- Duplicate question detection
- Triplet Loss function
- **Bài tập**: Xây dựng Siamese network cho question similarity

---

### [Module 04 - Natural Language Processing with Attention Models](https://github.com/DazielNguyen/NLP301c/tree/main/Module%2004)

_Xử lý ngôn ngữ tự nhiên với các mô hình attention_

#### [Week 1: Neural Machine Translation](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Lessons_Notes_Module_04/Week_01.md)

- Sequence-to-Sequence (Seq2Seq) models
- Encoder-Decoder architecture
- Attention Mechanism
- Teacher Forcing
- BLEU Score để đánh giá dịch máy
- **Bài tập**: Xây dựng neural machine translation system

#### [Week 2: Text Summarization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Lessons_Notes_Module_04/Week_02.md)

- Transformers Architecture
- Multi-Head Attention
- Positional Encoding
- Self-Attention mechanism
- Text Summarization (extractive và abstractive)
- **Bài tập**: Triển khai text summarization với Transformers

#### [Week 3: Question Answering](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Lessons_Notes_Module_04/Week_03.md)

- Transfer Learning trong NLP
- BERT (Bidirectional Encoder Representations from Transformers)
- T5 (Text-to-Text Transfer Transformer)
- Fine-tuning pre-trained models
- Context-based Question Answering
- **Bài tập**: Fine-tune BERT cho question answering task

---

## **Ôn tập PE (Practice Exam)**

Repository này bao gồm tài liệu ôn tập và đề thi từ các kỳ trước, giúp bạn chuẩn bị tốt cho kỳ thi:

### [Knowledge Base - Tổng hợp kiến thức NLP](https://github.com/DazielNguyen/NLP301c/blob/main/%C3%94n%20t%E1%BA%ADp%20PE/Knowledge.md)

Tài liệu tổng hợp toàn diện các kỹ thuật xử lý chuỗi và câu trong NLP:
- **Tách từ và Tokenization**: word_tokenize, sent_tokenize, NLTK
- **Xử lý và Làm sạch Chuỗi**: lowercase, remove punctuation, strip whitespace
- **Đếm và Thống kê Từ**: frequency distribution, sorting
- **Lọc và Trích xuất Từ**: filtering, pattern matching
- **Kiểm tra và Xác thực Chuỗi**: palindrome, isogram, validation
- **Chuyển đổi Format**: snake_case, camelCase, title case
- **Sửa lỗi Chính tả**: TextBlob spell correction
- **Thao tác với List và Dictionary**: slicing, sorting, lambda functions
- **N-grams: Bigrams, Trigrams và Anagrams**: Language models, text generation, spell checking
- **NLTK Library Cheatsheet**: Tokenization, stemming, lemmatization, POS tagging, NER, sentiment analysis

### Extra Practice Questions (PQ1-PQ20)

[Bộ câu hỏi thực hành bổ sung](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/Extra%20-%20NLP301c) với 20 câu hỏi từ cơ bản đến nâng cao:
- **PQ1**: Tokenize paragraph thành sentences và words
- **PQ2**: Normalize text (lowercase, remove whitespace)
- **PQ3**: Remove punctuation không dùng thư viện string
- **PQ4**: Count word frequency, filter và sort
- **PQ5**: Top 3 most frequent words
- **PQ6**: Average word length (exclude short words)
- **PQ7**: Extract words starting with vowel
- **PQ8**: Extract words with vowel và digit
- **PQ9**: Extract unique words, exclude stop words
- **PQ10**: Extract email addresses
- **PQ11**: Find palindrome words
- **PQ12**: Check isogram
- **PQ13**: Validate Python variable names
- **PQ14**: Convert snake_case to camelCase
- **PQ15**: Smart title case preserving acronyms
- **PQ16**: Reverse word order
- **PQ17**: Extract every nth word
- **PQ18**: Group words by length
- **PQ19**: Multi-criteria sorting
- **PQ20**: Longest common prefix

### Đề thi các kỳ trước

#### **Fall 2024 (FA24)**
- **[FA24 - Practice Exam](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/FA24%20-%20NLP301c%20-%20PE)**
  - Mã đề: NLP301c-FA24-PE
  - Nội dung: Tokenization, sentiment analysis, text processing
  
- **[FA24 - Retake Exam](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/FA24%20-%20NLP301c%20-%20RE)**
  - Mã đề: NLP301c-FA24-RE
  - Nội dung: String manipulation, word frequency, filtering

#### **Fall 2025 (FA25)**
- **[FA25 - Practice Exam](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/FA25%20-%20NLP301c%20-%20PE)**
  - Mã đề: NLP301c-FA25-PE
  - Nội dung: Advanced text processing, pattern matching
  
- **[FA25 - Practice Exam 2](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/FA25%20-%20NLP301c%20-%20PE2)**
  - Mã đề: NLP301c-FA25-PE2
  - Nội dung: NLTK applications, word embeddings

#### **Spring 2024 (SP24)**
- **[SP24 - Practice Exam 1](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/SP24%20-%20NLP301c%20-%20PE1)**
  - Mã đề: NLP301c-SP24-PE1
  - Nội dung: Basic tokenization, frequency analysis
  
- **[SP24 - Practice Exam 2](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/SP24%20-%20NLP301c%20-%20PE2)**
  - Mã đề: NLP301c-SP24-PE2
  - Nội dung: Text cleaning, validation, formatting

#### **Summer 2024 (SU24)**
- **[SU24 - Practice Exam 1](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/SU24%20-%20NLP301c%20-%20PE1)**
  - Mã đề: NLP301c-SU24-PE1
  - Nội dung: String operations, text transformation

#### **Summer 2025 (SU25)**
- **[SU25 - Practice Exam](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/SU25%20-%20NLP301c%20-%20PE)**
  - Mã đề: NLP301c-SU25-PE
  - Nội dung: Comprehensive NLP techniques
  
- **[SU25 - Retake Exam](https://github.com/DazielNguyen/NLP301c/tree/main/%C3%94n%20t%E1%BA%ADp%20PE/SU25%20-%20NLP301c%20-%20RE)**
  - Mã đề: NLP301c-SU25-RE
  - Nội dung: Problem solving with NLTK

### Tips cho Practice Exam

1. **Nắm vững cơ bản**: Tokenization, lowercase, split(), join()
2. **Thành thạo NLTK**: word_tokenize(), sent_tokenize(), stopwords
3. **Xử lý dictionary**: Đếm tần suất, sorting với lambda
4. **List comprehension**: Viết code ngắn gọn và hiệu quả
5. **String methods**: strip(), replace(), startswith(), endswith()
6. **Regex patterns**: Cho pattern matching phức tạp
7. **Practice coding**: Làm hết các đề PQ1-PQ20 và đề các kỳ trước

---

## **Ôn tập FE (Final Exam)**

Tổng hợp các đề thi Final Exam từ 2023-2025 trên Quizlet để ôn tập lý thuyết:

### Đề thi Final Exam các kỳ

- **[NLP301c - SU23 - FE](https://quizlet.com/vn/1124363658/nlp301c-su23-fe-flash-cards/)** - Summer 2023
  - Lý thuyết cơ bản về NLP, tokenization, word embeddings
  
- **[NLP301c - SP24 - FE](https://quizlet.com/vn/1124364086/nlp301c-sp24-fe-flash-cards/)** - Spring 2024
  - Vector spaces, probabilistic models, sentiment analysis
  
- **[NLP301c - SP24 - FE Retake](https://quizlet.com/vn/1124361416/nlp301c-sp24-fe-retake-flash-cards/)** - Spring 2024 Retake
  - Autocorrect, edit distance, language models
  
- **[NLP301c - FA24 - FE 1](https://quizlet.com/vn/1124365513/nlp301c-fa24-fe-1-flash-cards/)** - Fall 2024 Exam 1
  - RNNs, LSTMs, sequence models
  
- **[NLP301c - FA24 - FE 2](https://quizlet.com/vn/1124366281/nlp301c-fa24-fe-2-flash-cards/)** - Fall 2024 Exam 2
  - Attention mechanisms, transformers
  
- **[NLP301c - SU25 - FE](https://quizlet.com/vn/1124366683/nlp301c-su25-fe-flash-cards/)** - Summer 2025
  - Comprehensive review, all modules
  
- **[NLP301c - FA25 - FE](https://quizlet.com/vn/1124381752/nlp301c-fe-fa25-flash-cards/)** - Fall 2025
  - Latest exam, BERT, T5, question answering

### Tips cho Final Exam

1. **Ôn lý thuyết**: Nắm vững concepts từ cả 4 modules
2. **Flashcards**: Sử dụng Quizlet để học thuộc định nghĩa và công thức
3. **So sánh models**: Hiểu rõ ưu/nhược điểm của từng model (Naive Bayes vs Logistic Regression, RNN vs LSTM, etc.)
4. **Math formulas**: Ghi nhớ các công thức xác suất, cosine similarity, perplexity
5. **Architectures**: Vẽ và giải thích được kiến trúc của RNN, LSTM, Transformer
6. **Applications**: Biết ứng dụng của từng technique trong thực tế
7. **Practice**: Làm hết các đề từ 2023-2025 trên Quizlet

---

## **Tài liệu bổ sung**

- **`utils.py`**: Các hàm tiện ích dùng chung cho các bài tập
- **`requirements.txt`**: Danh sách thư viện Python cần thiết
- **`npl-env/`**: Virtual environment cho project

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

Copyright © 2025 Nguyen Van Anh Duy - NLP301c Documents

---

**Cảm ơn bạn đã ghé thăm! Have a good day!**
