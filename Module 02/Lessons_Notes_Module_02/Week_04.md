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

> **Ưu điểm** (`Pros`) của `one-hot vectors`:
- Đơn giản.
- Không yêu cầu thứ tự ngụ ý (`implied ordering`).

> **Nhược điểm** (`Cons`) của `one-hot vectors`:
- Rất lớn (`huge`).
- Không `encode` (mã hóa) được ý nghĩa (`meaning`).

---
### **Word Embeddings**
---

Nội dung này tập trung vào khái niệm **word embeddings**, một phương pháp để `encode` (mã hóa) ý nghĩa của các từ trong một **low-dimensional vector space** (không gian véc-tơ chiều thấp).

#### Understanding Word Embeddings

- **Word embeddings** biểu diễn các từ dưới dạng các `vectors` theo cách nắm bắt được ý nghĩa của chúng, cho phép so sánh dựa trên sự gần gũi trong `vector space`.
- Các từ có thể được định vị dọc theo hai trục: một cho **sentiment** (từ tích cực đến tiêu cực) và một cho **concreteness** (từ cụ thể đến trừu tượng).

#### Creating Word Vectors

- Một `two-dimensional vector` (véc-tơ hai chiều) có thể biểu diễn các từ, trong đó các `coordinates` (tọa độ) chỉ ra `sentiment` và mức độ trừu tượng của chúng.
- Biểu diễn này cho phép xác định sự tương đồng giữa các từ, chẳng hạn như "happy" và "excited" gần nhau hơn "paper."

#### Applications and Importance

- **Word embeddings** tạo điều kiện cho các `natural language processing` (`NLP`) `tasks` khác nhau, bao gồm **analogies** (sự tương tự) và **sentence meaning encoding** (mã hóa ý nghĩa câu).
- Bài giảng nhấn mạnh rằng việc tạo `word embeddings` là một mục tiêu chính của mô-đun này, dẫn đến các `NLP applications` phức tạp hơn như **question answering** và **translation**.

> Vậy tại sao lại sử dụng **word embeddings**? Hãy cùng xem.

![04_Word_Embeddings](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/04_Word_Embeddings.png)

> Từ sơ đồ trên, bạn có thể thấy rằng khi `encode` (mã hóa) một từ trong không gian **2D**, các từ tương tự có xu hướng nằm gần nhau. Có lẽ **coordinate** đầu tiên đại diện cho việc một từ là tích cực hay tiêu cực. **Coordinate** thứ hai cho bạn biết từ đó là **abstract** (trừu tượng) hay **concrete** (cụ thể). Đây chỉ là một ví dụ, trong thế giới thực, bạn sẽ tìm thấy các `embeddings` với hàng trăm **dimensions** (chiều). Bạn có thể coi mỗi **coordinate** là một con số cho bạn biết điều gì đó về từ đó.

> Ưu điểm của Word Embeddings

- **Low dimensions** (Chiều thấp) (ít hơn $V$, kích thước `vocabulary`).
- Cho phép bạn `encode` (mã hóa) ý nghĩa (`meaning`).

---
### **How to Create Word Embeddings**
---

Nội dung này tập trung vào quá trình tạo **word embeddings** trong `natural language processing` (`NLP`).

#### Các Thành phần Thiết yếu

Để tạo **word embeddings** cần hai thành phần chính:

- **Corpus** (Kho ngữ liệu) văn bản.
- **Embedding method** (Phương pháp nhúng).

`Corpus` phải liên quan đến ngữ cảnh. Ví dụ, để tạo `Shakespearean embeddings`, bạn cần sử dụng văn bản gốc của Shakespeare chứ không phải chỉ là các ghi chú tóm tắt.

#### Tầm quan trọng của Context

- **Context** (Ngữ cảnh) đề cập đến các từ xung quanh cung cấp ý nghĩa cho mỗi **word embedding**.
- Một `vocabulary list` đơn giản là không đủ; cần có một `corpus` toàn diện để nắm bắt các sắc thái ngữ nghĩa.

#### Phương pháp và Giám sát

- **Embedding method**, thường dựa trên các `machine learning models`, tạo ra **word embeddings** từ `corpus`.
- `Learning task` có thể là **self-supervised** (tự giám sát), tận dụng dữ liệu không có nhãn trong khi `model` tự cung cấp ngữ cảnh của riêng nó để giám sát.

#### Hyperparameters và Biểu diễn Toán học

- **Word embeddings** có thể được điều chỉnh bằng **hyperparameters** (siêu tham số), chẳng hạn như **dimension** (chiều) của các `embedding vectors`, thường dao động từ hàng trăm đến hàng nghìn.
- `Corpus` phải được biến đổi thành một **biểu diễn toán học** phù hợp cho `model`, thường sử dụng **integer-based indices** hoặc **one-hot vectors**.

Nội dung sắp tới sẽ giới thiệu các `word embedding methods` khác nhau, bao gồm cách tiếp cận **continuous bag-of-words** (`CBOW`), cái mà sẽ được triển khai trong bài tập tiếp theo.

> Để tạo **word embeddings**, bạn luôn cần một **corpus** (kho ngữ liệu) văn bản và một **embedding method** (phương pháp nhúng). **Context** (Ngữ cảnh) của một từ cho bạn biết loại từ nào có xu hướng xảy ra gần từ cụ thể đó. **Context** là quan trọng vì đây là yếu tố sẽ mang lại ý nghĩa cho mỗi `word embedding`.


#### Phương pháp Embeddings và Tự giám sát

> Có nhiều loại phương pháp có thể cho phép bạn học các **word embeddings**. `Machine learning model` thực hiện một `learning task` (nhiệm vụ học tập), và sản phẩm phụ chính của `task` này là các `word embeddings`. `Task` có thể là học cách dự đoán một từ dựa trên các từ xung quanh trong một câu của `corpus`, như trong trường hợp của **continuous bag-of-words** (`CBOW`).

> `Task` là **self-supervised** (tự giám sát): nó vừa là **unsupervised** (không giám sát) ở chỗ dữ liệu đầu vào — `corpus` — là **unlabelled** (không có nhãn), và vừa là **supervised** (có giám sát) ở chỗ bản thân dữ liệu cung cấp `context` cần thiết mà thông thường sẽ tạo thành các `labels` (nhãn).

> Khi `training word vectors`, có một số **hyperparameters** (siêu tham số) bạn cần điều chỉnh (ví dụ: **dimension** (chiều) của `word vector`).

![05_How_to_Create_Word_Embeddings](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/05_How_to_Create_Word_Embeddings.png)

---
### **Word Embedding Methods**
---

Nội dung này tập trung vào các **word embedding methods** khác nhau được sử dụng trong `natural language processing`.

#### Mô hình Word2Vec

* **Word2Vec** sử dụng một **shallow neural network** (mạng nơ-ron nông) với hai `architectures` (kiến trúc): **continuous bag-of-words** (**CBOW**) và **continuous skip-gram**.
    * **CBOW** dự đoán một từ bị thiếu dựa trên các từ xung quanh.
    * **Skip-gram** dự đoán các từ xung quanh từ một từ `input` cho trước.

#### Các Kỹ thuật Embeddings Nâng cao

* **GloVe** (`Global Vectors`) phân tích ma trận **word co-occurrence matrix** (đồng xuất hiện từ) để nắm bắt ý nghĩa của từ.
* **FastText** cải tiến `skip-gram` bằng cách biểu diễn các từ dưới dạng **character n-grams**, cho phép nó xử lý hiệu quả các **unseen words** (từ chưa từng thấy).

### Contextual Word Embeddings

* Các `models` tiên tiến như **BERT**, **ELMo**, và **GPT-2** tạo ra các `embeddings` khác nhau cho các từ dựa trên **context** (ngữ cảnh) của chúng, hỗ trợ **polysemy** (đa nghĩa).
* Các `models` này có thể được tìm thấy dưới dạng **pretrained versions** (phiên bản được huấn luyện trước) trực tuyến và có thể được **fine-tuned** (tinh chỉnh) với các `corpora` cụ thể để có `performance` tốt hơn.

### 📚 Phương pháp Word Embedding

#### Phương pháp Cổ điển (`Classical Methods`)

* **word2vec** (Google, 2013):
    * **Continuous bag-of-words (CBOW)**: `model` học cách **dự đoán** từ trung tâm (`center word`) cho trước các `context words` (từ ngữ cảnh).
    * **Continuous skip-gram / Skip-gram with negative sampling (SGNS)**: `model` học cách **dự đoán** các từ xung quanh (`surrounding words`) cho trước một từ `input`.

* **Global Vectors (GloVe)** (Stanford, 2014): Phân tích `logarithm` của **word co-occurrence matrix** (ma trận đồng xuất hiện từ) của `corpus`, tương tự như `count matrix` bạn đã sử dụng trước đây.
* **fastText** (Facebook, 2016): Dựa trên `skip-gram model` và tính đến cấu trúc của từ bằng cách biểu diễn các từ dưới dạng `n-gram` của ký tự. Nó hỗ trợ các từ **out-of-vocabulary (OOV)**.

#### Deep Learning, Contextual Embeddings

Trong các `models` tiên tiến hơn này, các từ có các `embeddings` khác nhau tùy thuộc vào **context** (ngữ cảnh) của chúng. Bạn có thể tải xuống các `pre-trained embeddings` (embeddings được huấn luyện trước) cho các `models` sau:

* **BERT** (Google, 2018)
* **ELMo** (Allen Institute for AI, 2018)
* **GPT-2** (OpenAI, 2018)

---
### **Continuous Bag-of-Words Model**
---

Nội dung tập trung vào việc triển khai **continuous bag-of-words model** (**CBOW**) để tạo `word embeddings` trong `natural language processing`.

#### Quá trình tổng thể của Word Embeddings

* **Word embeddings** được tạo ra thông qua một `machine learning model` học từ một `corpus` (kho ngữ liệu).
* **Continuous bag-of-words model** dự đoán một từ bị thiếu (**center word**) dựa trên các **context words** (từ ngữ cảnh) xung quanh nó.

#### Tạo Dữ liệu Huấn luyện

* **Context words** được định nghĩa là các từ bao quanh một **center word**, với một **hyperparameter $C$** xác định số lượng `context words` (bán kính cửa sổ ngữ cảnh).
* `Model` sử dụng **sliding windows** (cửa sổ trượt) để tạo các `training examples` (ví dụ huấn luyện), trong đó `context words` là `inputs` và `center word` là `target` (mục tiêu) để dự đoán.

#### Kiến trúc Mô hình và Học tập

* **Model architecture** bao gồm `context words` là `inputs` và `center words` là `outputs`.
* Khi `model` học, nó tạo ra `word embeddings` như một sản phẩm phụ của `prediction task` (nhiệm vụ dự đoán), nắm bắt được **semantic relationships** (mối quan hệ ngữ nghĩa) giữa các từ.

> Để tạo **word embeddings**, bạn cần một `corpus` và một `learning algorithm` (thuật toán học tập). Sản phẩm phụ của `task` này sẽ là một tập hợp các `word embeddings`. Trong trường hợp của **continuous bag-of-words model** (**CBOW**), `objective` (mục tiêu) của `task` là **dự đoán một từ bị thiếu** dựa trên các từ xung quanh nó.

![06_Continuous_Bag-of-Words_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/06_Continuous_Bag-of-Words_Model.png)

> Dưới đây là một **visualization** (hình ảnh trực quan) cho bạn thấy `model` hoạt động như thế nào.

![07_Continuous_Bag-of-Words_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/07_Continuous_Bag-of-Words_Model.png)

> Như bạn có thể thấy, **window size** (kích thước cửa sổ) trong hình ảnh phía trên là 5. **Context size** (kích thước ngữ cảnh), $C$, là 2. $C$ thường cho bạn biết có bao nhiêu từ trước hoặc sau **center word** (từ trung tâm) mà `model` sẽ sử dụng để đưa ra **prediction** (dự đoán).

> Dưới đây là một **visualization** khác cho thấy tổng quan về `model`.

![08_Continuous_Bag-of-Words_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/08_Continuous_Bag-of-Words_Model.png)

---
### **Cleaning and Tokenization**
---

Nội dung tập trung vào các quy trình **cleaning** (làm sạch) và **tokenization** (tạo token) trong `natural language processing` (`NLP`).

#### Cleaning và Tokenization

- Các từ nên được xử lý dưới dạng **case insensitive** (không phân biệt chữ hoa/thường), nghĩa là chúng nên được chuyển đổi thành một định dạng duy nhất (chữ thường hoặc chữ hoa) để đồng nhất.
- **Punctuation** (Dấu câu) cần được xử lý cẩn thận; dấu câu gây ngắt quãng có thể được biểu thị bằng một `special word` (từ đặc biệt) duy nhất, trong khi dấu câu không gây ngắt quãng có thể bị bỏ qua.

#### Xử lý Số và Ký tự Đặc biệt

- **Numbers** (Số) có thể bị bỏ đi nếu chúng không quan trọng, nhưng các số quan trọng nên được giữ lại hoặc thay thế bằng một `special token` như **\<NUMBER\>**.
- Các **Special characters** (Ký tự đặc biệt), chẳng hạn như ký hiệu toán học và `emojis`, nên được quản lý dựa trên mức độ liên quan của chúng với `model`.

### Ví dụ Thực hành

Một `Python example` minh họa cách `clean` một `corpus` bằng cách gộp `punctuation` và **tokenizing** văn bản bằng cách sử dụng `NLTK library`, tạo ra một `array of tokens` sẵn sàng để phân tích thêm.

Điều này tạo tiền đề cho chủ đề tiếp theo về **continuous bag-of-words model**.

> Trước khi triển khai bất kỳ thuật toán `natural language processing` (`NLP`) nào, bạn có thể muốn `clean` (làm sạch) dữ liệu và `tokenize` (tạo token) nó. Dưới đây là một vài điều cần lưu ý khi xử lý `data` của bạn.

![09_Cleaning_and_Tokenization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/09_Cleaning_and_Tokenization.png)

> Bạn có thể `clean data` bằng `Python` như sau:

![10_Cleaning_and_Tokenization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/10_Cleaning_and_Tokenization.png)

> Bạn có thể thêm bao nhiêu điều kiện tùy thích vào các dòng tương ứng với hình chữ nhật màu xanh lá cây phía trên.

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



