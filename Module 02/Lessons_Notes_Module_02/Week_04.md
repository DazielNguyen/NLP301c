# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 4: Word Embeddings with Neural Network**
---
### **Overview**
---

Nội dung tuần này tập trung vào **word vectors**, còn được gọi là **word embeddings**, và cách `training` (huấn luyện) chúng từ đầu.

#### Understanding Word Vectors

- **Word vectors** rất cần thiết cho các ứng dụng khác nhau trong `natural language processing` (`NLP`), chẳng hạn như `sentiment analysis` (phân tích tình cảm) và `machine translation` (dịch máy).
- Chúng cho phép biểu diễn số học của các từ, tạo điều kiện cho việc sử dụng chúng trong các `mathematical models`.

#### Training Word Vectors

- Khóa học sẽ đề cập đến các phương pháp tạo **word embeddings**, bao gồm **continuous bag-of-words model** (`CBOW`).
- Các kỹ thuật khác như **GloVe** và **Word2Vec** cũng sẽ được đề cập, nhưng trọng tâm sẽ là **continuous bag-of-words model**.

#### Preparing Text for Machine Learning

- Người học sẽ biết cách biến đổi dữ liệu văn bản thành một `training set` phù hợp cho các `machine learning models`.
- Lời khuyên thực tế sẽ được cung cấp để làm việc với các `text corpora` đa dạng, chẳng hạn như sách và `tweets`.

> **Word embeddings** (nhúng từ) được sử dụng trong hầu hết các `NLP applications`. Bất cứ khi nào bạn xử lý văn bản, trước tiên bạn phải tìm cách để `encode` (mã hóa) các từ dưới dạng số. `Word embedding` là một kỹ thuật rất phổ biến cho phép bạn làm điều đó.

> Dưới đây là một vài `applications` (ứng dụng) của `word embeddings` mà bạn sẽ có thể triển khai khi hoàn thành chuyên ngành này.

![01_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/01_Overview.png)

![02_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/02_Overview.png)

#### Mục tiêu học tập trong tuần

Đến cuối tuần này, bạn sẽ có thể:

- Xác định các khái niệm chính của **word representations** (biểu diễn từ).
- Tạo ra **word embeddings**.
- Chuẩn bị văn bản cho **machine learning**.
- Triển khai **continuous bag-of-words model**.

---
### **Basic Word Representations**
---
Nội dung tập trung vào việc biểu diễn các từ trong một `vocabulary` (từ vựng) bằng các `numerical vectors` (véc-tơ số), cụ thể thông qua khái niệm **one-hot vectors**.

#### Understanding One-Hot Vectors

- Mỗi từ trong một `vocabulary` được gán một số nguyên (`integer`) duy nhất, nhưng phương pháp này thiếu **semantic meaning** (ý nghĩa ngữ nghĩa).
- **One-hot vectors** biểu diễn các từ dưới dạng các `binary vectors` (véc-tơ nhị phân), trong đó '1' cho biết sự hiện diện của một từ và '0' cho biết sự vắng mặt.

#### Advantages and Limitations of One-Hot Vectors

- **One-hot vectors** đơn giản và không ngụ ý bất kỳ mối quan hệ nào giữa các từ.
- Tuy nhiên, chúng có thể trở nên rất lớn và **không nắm bắt được ý nghĩa** hoặc sự tương đồng giữa các từ, dẫn đến những hạn chế trong các `natural language processing tasks`.

#### Transition to Word Embeddings

- Cuộc thảo luận tạo tiền đề cho việc giới thiệu **word embeddings**, cái mà nhằm mục đích giải quyết những hạn chế của `one-hot vectors` bằng cách nắm bắt các mối quan hệ ngữ nghĩa (`semantic relationships`) giữa các từ.

> Các biểu diễn từ cơ bản có thể được phân loại thành các dạng sau:

- **Integers** (Số nguyên)
- **One-hot vectors**
- **Word embeddings**

![03_Basic_Word_Representations](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/03_Basic_Word_Representations.png)

> Ở bên trái, bạn có một ví dụ trong đó bạn sử dụng số nguyên (`integers`) để biểu diễn một từ. Vấn đề ở đó là không có lý do gì khiến từ này tương ứng với một số lớn hơn từ khác. Để khắc phục vấn đề này, chúng ta giới thiệu **one hot vectors** (sơ đồ bên phải). Để triển khai `one hot vectors`, bạn phải `initialize` (khởi tạo) một `vector` toàn số không (`zeros`) có **dimension $V$** và sau đó đặt số **1** vào `index` tương ứng với từ bạn đang biểu diễn.

**Ưu điểm** (`Pros`) của `one-hot vectors`:
- Đơn giản.
- Không yêu cầu thứ tự ngụ ý (`implied ordering`).

**Nhược điểm** (`Cons`) của `one-hot vectors`:
- Rất lớn (`huge`).
- Không `encode` (mã hóa) được ý nghĩa (`meaning`).

---
### **Word Embeddings**
---



---
### **How to Create Word Embeddings**
---


---
### **Word Embedding Methods**
---



---
### **Continuous Bag-of-Words Model**
---


---
### **Sliding Window of Words in Python**
---


---
### **Transforming Words into Vectors**
---



---
### **Architecture of the CBOW Model**
---


---
### **Architecture of the CBOW Model: Dimensions**
---

---
### **Architecture of the CBOW Model: Dimensions 2**
---

---
### **Architecture of the CBOW Model: Activation Functions**
---


---
### **Training a CBOW Model: Cost Function**
---

---
### **Training a CBOW Model: Forward Propagation**
---


---
### **Training a CBOW Model: Backpropagation and Gradient Descent**
---


---
### **Extracting Word Embedding Vectors**
---


---
### **Evaluating Word Embeddings: Intrinsic Evaluation**
---


---
### **Evaluating Word Embeddings: Extrinsic Evaluation**
---



