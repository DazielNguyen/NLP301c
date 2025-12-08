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

Nội dung tập trung vào việc trích xuất **center words** (từ trung tâm) và **context words** (từ ngữ cảnh) để `training` (huấn luyện) **continuous bag-of-words model** trong `natural language processing`.

#### Trích xuất Center và Context Words

* Quá trình bắt đầu với một `cleaned and tokenized corpus` (kho ngữ liệu đã làm sạch và tạo token), được biểu diễn dưới dạng một `array` (mảng) các từ.
* Một `function` (hàm) gọi là `get_windows` được định nghĩa để trích xuất `center words` và `context words` của chúng dựa trên một `context size` được chỉ định.

#### Triển khai Function

* `Function` nhận một `array of words` và một **context size ($C$)**, cái mà xác định có bao nhiêu từ sẽ được xem xét ở mỗi bên của `center word`.
* Nó `initialize` (khởi tạo) một vòng lặp để lặp qua `array`, trích xuất `center words` và các `context words` tương ứng của chúng.

#### Sử dụng Function

* `Function` sử dụng **yield keyword** để trả về các giá trị một cách **iteratively** (lặp lại), cho phép **data generation** (tạo dữ liệu) hiệu quả.
* Một vòng lặp được sử dụng để hiển thị `context` và `center words`, cái mà cần thiết cho **continuous bag-of-words model**.

Tổng quan, bài giảng này cung cấp một cách tiếp cận thực tế để chuẩn bị `data` cho `training word embeddings` bằng `Python`.

![11_Sliding_Window_of_Words_in_Python](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/11_Sliding_Window_of_Words_in_Python.png)

> `Code` phía trên cho thấy một `function` (hàm) nhận hai `parameters` (tham số).

* `Words`: một `list` (danh sách) các từ.
* $C$: **context size** (kích thước ngữ cảnh).

> Chúng ta bắt đầu bằng cách đặt $i$ bằng $C$. Sau đó, chúng ta tách **center\_word** (từ trung tâm) và **context\_words** (các từ ngữ cảnh). Chúng ta sau đó **yield** (trả về) các giá trị này và **increment** (tăng) $i$ lên.


---
### **Transforming Words into Vectors**
---

Nội dung tập trung vào việc chuẩn bị `data` (dữ liệu) cho **continuous bag-of-words model** (**CBOW**) trong `natural language processing`.

#### Chuẩn bị Dữ liệu cho Mô hình CBOW

* **Context và Central Words**: Quá trình bắt đầu bằng việc xác định **context words** và **central word** (từ trung tâm) từ một `sliding window` (cửa sổ trượt) trên `corpus` (kho ngữ liệu).
* **Vocabulary Creation**: Một **vocabulary** (từ vựng) được hình thành từ các từ độc nhất (`unique words`) trong `corpus`, sau đó được sử dụng để tạo **one-hot vectors** cho các `central words`.

#### Biểu diễn Vector

* **One-Hot Encoding**: Mỗi từ trong `vocabulary` được biểu diễn dưới dạng **one-hot vector**, trong đó '1' cho biết sự hiện diện của một từ và '0' cho biết sự vắng mặt.
* **Averaging Context Vectors**: Đối với **context words**, một `vector` duy nhất được tạo ra bằng cách **averaging** (tính trung bình) các `one-hot vectors` của mỗi `context word`, cung cấp một biểu diễn cho `context`.

#### Chuẩn bị Dữ liệu Huấn luyện

* **Final Vector Representation**: Các `final vectors` (véc-tơ cuối cùng) cho cả `central words` và `context words` được chuẩn bị để `training` (huấn luyện) **CBOW model**.
* **Transition to Model Learning**: Với `data` đã được biểu diễn đầy đủ, bước tiếp theo là tìm hiểu về **architecture** (kiến trúc) của **CBOW model** và áp dụng các kỹ năng vào các bài tập sắp tới.

> Để biến đổi các **context vectors** (véc-tơ ngữ cảnh) thành một **single vector** (véc-tơ đơn lẻ), bạn có thể sử dụng công thức/phương pháp sau:

![12_Transforming_Words_into_Vectors](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/12_Transforming_Words_into_Vectors.png)

> Như bạn có thể thấy, chúng ta bắt đầu với các **one-hot vectors** cho các từ ngữ cảnh và biến đổi chúng thành một **single vector** bằng cách lấy **average** (trung bình). Kết quả là bạn nhận được các `vectors` sau đây mà bạn có thể sử dụng cho việc **training** (huấn luyện) của mình.

![13_Transforming_Words_into_Vectors](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/13_Transforming_Words_into_Vectors.png)

---
### **Architecture of the CBOW Model**
---

Nội dung tập trung vào **architecture** (kiến trúc) của **Continuous Bag of Words** (**CBOW**) `model` được sử dụng trong `natural language processing`.

#### Tổng quan Kiến trúc

* **CBOW model** bao gồm một **shallow dense neural network** (mạng nơ-ron dày đặc nông) với một `input layer`, một `hidden layer` (lớp ẩn), và một `output layer`.
* `Input` là một `vector` của **context words** (các từ ngữ cảnh), trong khi `output` là **center word** (từ trung tâm) được dự đoán, cả hai đều có kích thước theo **vocabulary** ($V$).

#### Chi tiết các Lớp

* Kích thước của `hidden layer` được xác định bởi **dimension** (chiều) đã chọn của **word embeddings** ($N$), thường dao động từ $100$ đến $1,000$.
* `Network` là **fully connected** (kết nối đầy đủ), với các **weight matrices** ($W_1$ và $W_2$) và **bias vectors** ($b_1$ và $b_2$) mà `model` học trong quá trình `training`.

#### Hàm Kích hoạt (Activation Functions)

* `Hidden layer` sử dụng **Rectified Linear Units** (**ReLU**) `activation function` (hàm kích hoạt).
* `Output layer` sử dụng **softmax function** để đưa ra **predictions** (dự đoán).

Bản tóm tắt này cung cấp sự hiểu biết ngắn gọn về cấu trúc và các thành phần của `CBOW model`.

> `Architecture` (Kiến trúc) cho **CBOW model** có thể được mô tả như sau:

![14_Architecture_of_the_CBOW_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/14_Architecture_of_the_CBOW_Model.png)

> Bạn có một `input`, $X$, là giá trị trung bình (`average`) của tất cả các **context vectors** (véc-tơ ngữ cảnh). Sau đó, bạn nhân nó với $W_1$ và cộng thêm $b_1$. Kết quả này đi qua một **ReLU function** để tạo ra **hidden layer** (lớp ẩn) của bạn. Lớp đó sau đó được nhân với $W_2$ và bạn cộng thêm $b_2$. Kết quả này đi qua một **softmax** để cung cấp cho bạn một **distribution** (phân phối) trên $V$ (kích thước `vocabulary`). Bạn chọn `vocabulary word` tương ứng với **arg-max** của `output`.

---
### **Architecture of the CBOW Model: Dimensions**
---

Nội dung tập trung vào việc hiểu **dimensions** (chiều) của các lớp trong một `neural network model`, cụ thể là **continuous bag of words** (**CBOW**) `model`.

#### Kiến trúc Neural Network

* `Input layer` được biểu diễn bằng một **column vector** ($x$) với các số không (`zeros`), trong đó $V$ là **vocabulary size** (kích thước từ vựng). ($x$ có dimension $V \times 1$).
* `Hidden layer` ($h$) được tính bằng **weighted sum** ($W_1 x + b_1$), trong đó $W_1$ là **weight matrix** (ma trận trọng số) (dimension $N \times V$) và $b_1$ là **bias vector** (véc-tơ độ lệch) (dimension $N \times 1$). ($N$ là embedding dimension).

#### Tính toán Output

* Các giá trị **output layer** được suy ra từ `hidden layer` ($h$) bằng cách sử dụng ($W_2 h + b_2$), trong đó $W_2$ là **weight matrix** cho `output layer` (dimension $V \times N$) và $b_2$ là **bias vector** tương ứng (dimension $V \times 1$).
* `Output` cuối cùng ($\hat{y}$) được thu được bằng cách áp dụng **softmax activation function** (hàm kích hoạt softmax) cho các giá trị `output layer`. ($\hat{y}$ có dimension $V \times 1$).

#### Xử lý các Loại Vector

* Nếu sử dụng **row vectors** (véc-tơ hàng) thay vì **column vectors** (véc-tơ cột), các phép tính `matrix` phải được điều chỉnh tương ứng, chẳng hạn như **transposing matrices** (chuyển vị ma trận) trong quá trình nhân.
* Hiểu rõ các `dimensions` này là rất quan trọng để tránh các lỗi **dimension mismatch errors** (lỗi không khớp chiều) trong các `programming assignments`.

> Các phương trình cho `model` trước là:

$$z_1 = W_1 x + b_1$$

$$h = \text{ReLU}(z_1)$$

$$z_2 = W_2 h + b_2$$

$$\hat{y} = \text{softmax}(z_2)$$

> Ở đây, bạn có thể thấy các **dimensions** (chiều):

![15_Architecture_of_the_CBOW_Model_Dimensions](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/15_Architecture_of_the_CBOW_Model_Dimensions.png)

> Hãy đảm bảo rằng bạn xem kỹ các phép **matrix multiplications** (nhân ma trận) và hiểu tại sao các **dimensions** (chiều) lại hợp lý.

---
### **Architecture of the CBOW Model: Dimensions 2**
---

Nội dung tập trung vào khái niệm **batch processing** (xử lý theo lô) trong `Continuous Bag of Words` (**CBOW**) `model` được sử dụng trong `neural networks`.

#### Batch Processing trong CBOW

* Thay vì cung cấp các `individual examples` (ví dụ riêng lẻ), nhiều `input examples` có thể được xử lý đồng thời, điều này giúp tăng tốc quá trình học tập.
* **Batch size** ($M$) là một **hyperparameter** (siêu tham số) được định nghĩa trong quá trình `training`, cho phép hình thành một **matrix** ($X$) từ các `input vectors` này.

#### Các Phép toán Matrix

* Các giá trị **hidden layer** ($H$) được tính bằng cách áp dụng **ReLU activation function** cho `weighted input matrix` ($Z_1$), cái mà bao gồm một **bias matrix** ($B_1$).
* **Output matrix** ($\hat{Y}$) được suy ra từ `hidden layer` và bao gồm một **replicated bias matrix** ($B_2$), biến đổi các `input vectors` thành các `output vectors` tương ứng.

#### Hàm Kích hoạt

* Bài giảng gợi ý về việc giới thiệu các **activation functions** (hàm kích hoạt) được sử dụng trong `CBOW model`, cho thấy người học đang tiến tới việc xây dựng một `model` chức năng.

> Khi xử lý **batch input** (đầu vào theo lô), bạn có thể **stack** (xếp chồng) các ví dụ thành các **columns** (cột). Sau đó, bạn có thể tiến hành nhân các **matrices** (ma trận) như sau:

![16_Architecture_of_the_CBOW_Model_Dimensions_2](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/16_Architecture_of_the_CBOW_Model_Dimensions_2.png)

> Trong sơ đồ phía trên, bạn có thể thấy các **dimensions** (chiều) của mỗi **matrix**. Lưu ý rằng $\hat{Y}$ của bạn có **dimension** $V$ nhân $M$. Mỗi **column** là **prediction** (dự đoán) của `column` tương ứng với các **context words**. Vì vậy, `column` đầu tiên trong $\hat{Y}$ là **prediction** tương ứng với `column` đầu tiên của $X$.

---
### **Architecture of the CBOW Model: Activation Functions**
---

Nội dung này tập trung vào hai **activation functions** (hàm kích hoạt) quan trọng được sử dụng trong trí tuệ nhân tạo: **Rectified Linear Unit** (`ReLU`) và **Softmax function**.

#### ⚙️ ReLU Function

* **ReLU** là một **activation function** được sử dụng rộng rãi, nó chỉ kích hoạt một **neuron** khi `weighted input` là dương, thiết lập tất cả các `inputs` âm về 0.
* Công thức của `ReLU` là:

$$f(z) = \max(0, z)$$

* Ví dụ, nếu `input vector` chứa các giá trị âm, những giá trị đó sẽ trở thành số 0 trong `output`, trong khi các giá trị dương vẫn giữ nguyên.

#### 📊 Softmax Function

* **Softmax function** nhận một `vector of real numbers` (véc-tơ các số thực) làm `input` và `output` ra một **probability distribution** (phân phối xác suất), trong đó tổng các giá trị bằng một.

* Công thức cho `Softmax` (đối với phần tử thứ $i$ trong vector $z$) là:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

* Nó đặc biệt hữu ích trong các vấn đề **multi-class classification** (phân loại đa lớp), vì nó cung cấp `probabilities` của mỗi `class`, cho phép giải thích `model's predictions`.

Tóm lại, `ReLU` giúp quản lý kích hoạt **neuron** ở `hidden layer`, trong khi `Softmax` là cần thiết để tạo **probabilities** ở `output layer` trong các `classification tasks`.

> ReLU function

**ReLU function** (`Rectified Linear Unit`), là một trong những `activation functions` phổ biến nhất. Khi bạn đưa một `vector`, cụ thể là $x$, vào một `ReLU function`. Bạn kết thúc với phép tính:

$$x = \max(0, x)$$

Đây là hình vẽ minh họa `ReLU`.

![17_Architecture_of_the_CBOW_Model_Dimensions_AF](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/17_Architecture_of_the_CBOW_Model_Dimensions_AF.png)

> Softmax function

**Softmax function** nhận một `vector` và biến đổi nó thành một **probability distribution** (phân phối xác suất). Ví dụ, cho trước `vector` $z$ sau, bạn có thể biến đổi nó thành một **probability distribution** như sau.

![18_Architecture_of_the_CBOW_Model_Dimensions_AF](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/18_Architecture_of_the_CBOW_Model_Dimensions_AF.png)

Như bạn có thể thấy, bạn có thể tính `probability` ($\hat{y}_i$) của phần tử $i$ như sau:

$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

Trong đó $V$ là kích thước của `vector` $z$ (tức là kích thước `vocabulary`).

---
### **Training a CBOW Model: Cost Function**
---

Nội dung này tập trung vào **cost function** (hàm chi phí) cho **Softmax** trong `machine learning`, đặc biệt trong bối cảnh dự đoán từ bằng **continuous bag of words model** (**CBOW**).

#### Tổng quan Cost Function

* **Cost function** rất cần thiết để dự đoán một trong những từ có thể có bằng cách tối thiểu hóa một chi phí cụ thể.
* Một **training example** (ví dụ huấn luyện) đơn lẻ bao gồm một `input`, một **true target** (mục tiêu thực tế), và giá trị dự đoán của `model`.

#### Loss Function và Tham số

* **Loss function** (hàm mất mát) đo lường sai số giữa `prediction` và **true value** cho một `training example`.
* Trong **CBOW model**, các **parameters** (tham số) được điều chỉnh bao gồm **weight matrices** ($W_1, W_2$) và **bias factors** ($b_1, b_2$).

#### Cross Entropy Loss

* **Cross entropy loss** (mất mát entropy chéo) thường được sử dụng với các **classification models** và có liên quan đến lớp `output Softmax`.
* Công thức cho **cross entropy loss** ($J$) liên quan đến tổng âm của tích giữa **true value** ($y_i$) và `log` của **predicted value** ($\hat{y}_i$):
$$J = - \sum_{i=1}^{V} y_i \log(\hat{y}_i)$$

#### Ví dụ Dự đoán

* **Loss function** thưởng cho các `predictions` đúng và phạt các `predictions` không chính xác (cho thấy `loss` tăng lên với các `predictions` không chính xác), với ý nghĩa đối với **model performance**.

![19_Training_a_CBOW_Model_Cost_Function](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/19_Training_a_CBOW_Model_Cost_Function.png)

> Mức **chi phí (cost)** là $4.61$ trong ví dụ trên là do **mô hình đã dự đoán một xác suất rất thấp cho từ đúng (true target)**.

> Giá trị $4.61$ không phải là ngẫu nhiên; nó thể hiện mối quan hệ nghịch đảo giữa chi phí và xác suất dự đoán thông qua hàm $\log$ tự nhiên ($\ln$).

Dưới đây là lý do chi tiết:

#### 1. Công thức Đơn giản hóa

Công thức **Cross-Entropy Loss** cho một ví dụ huấn luyện đơn lẻ, nơi **true target ($y$)** là một **one-hot vector** (chỉ có 1 ở vị trí từ đúng $k^*$) được đơn giản hóa thành:

$$J = - \sum_{k=1}^{V} y_k \log(\hat{y}_k) = - \log(\hat{y}_{k^*})$$

Trong đó:
* $V$ là kích thước từ vựng.
* $y_{k^*}$ là $1$ (xác suất $100\%$ rằng từ $k^*$ là đúng).
* $\hat{y}_{k^*}$ là xác suất mà mô hình dự đoán cho từ đúng $k^*$.

#### 2. Tính toán Xác suất Dự đoán

Nếu chi phí được tính là $J = 4.61$ (sử dụng $\log$ tự nhiên, $\ln$, là tiêu chuẩn):

$$4.61 = - \ln(\hat{y}_{k^*})$$

Chúng ta có thể giải phương trình này để tìm xác suất dự đoán $\hat{y}_{k^*}$:

$$\ln(\hat{y}_{k^*}) = -4.61$$
$$\hat{y}_{k^*} = e^{-4.61} \approx 0.010$$

#### Kết luận

Giá trị $4.61$ cho thấy **mô hình chỉ dự đoán xác suất khoảng $0.01$ (tức $1\%$) cho từ lẽ ra phải là đáp án đúng** trong ví dụ này. Chi phí cao (như $4.61$) là cách **loss function** trừng phạt mô hình vì đã đưa ra một dự đoán sai lệch cao (xác suất thấp) cho kết quả đúng.

---
### **Training a CBOW Model: Forward Propagation**
---

Nội dung tập trung vào quá trình **forward propagation** (lan truyền tiến) trong `Continuous Bag-of-Words` (**CBOW**) `model` được sử dụng trong `neural networks`.

#### Tổng quan Forward Propagation

* **Forward propagation** bao gồm việc truyền các giá trị `input` qua `neural network` từ `input` đến `output`, tính toán các giá trị ở mỗi lớp.
* Một **batch of examples** (lô ví dụ) được biểu diễn dưới dạng một `matrix`, và **output matrix** được tạo ra bằng cách lan truyền `input` này qua `network`.

#### Tính toán Cost

* **Cost function** (hàm chi phí) là một phần mở rộng của **loss function** (hàm mất mát), được sử dụng để đo lường sai số cho một `batch of training examples`.
* **Cross-entropy cost** cho một `batch` là **mean** (giá trị trung bình) của các **cross-entropy losses** riêng lẻ cho mỗi ví dụ, cho phép hình dung `cost` như là một giá trị trung bình của các `losses`.

#### Quá trình Optimization

* Sau khi tính toán `cost`, **back propagation** (lan truyền ngược) và **gradient descent** (giảm độ dốc) được sử dụng để điều chỉnh các `parameters` (tham số) của `network` nhằm cải thiện `predictions`.
* Các bước tiếp theo bao gồm `training word vectors` bằng cách sử dụng **cost function** để nâng cao `model's performance`.

> Forward Propagation (Lan truyền tiến)

> **Forward Propagation** được định nghĩa là:

$$Z_1 = W_1 X + B_1$$

$$H = \text{ReLU}(Z_1)$$

$$Z_2 = W_2 H + B_2$$

$$\hat{Y} = \text{softmax}(Z_2)$$

> Trong đó $X$ là ma trận `input` (đầu vào theo lô), $W_1, W_2$ là ma trận `weights` (trọng số), $B_1, B_2$ là ma trận `bias` (độ lệch), $H$ là **hidden layer** (lớp ẩn), và $\hat{Y}$ là ma trận dự đoán `output` (`predicted output matrix`).

> Trong hình ảnh dưới đây, bạn bắt đầu từ bên trái và **forward propagate** (lan truyền tiến) suốt tới bên phải.

![20_Training_a_CBOW_Model_FP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/20_Training_a_CBOW_Model_FP.png)

> Batch Cost Function (Hàm chi phí theo lô)

> Để tính **loss** (tổn thất) của một **batch** (lô), bạn phải tính công thức sau. Công thức này là giá trị trung bình (`mean`) của các **Cross-Entropy losses** trên $M$ ví dụ trong `batch`:

$$J_{\text{batch}} = -\frac{1}{M}\sum_{i=1}^{M}\sum_{j=1}^{V}y_{j}^{(i)}\log\hat{y}_{j}^{(i)}$$

> Trong đó:

* $M$: Kích thước `batch`.
* $V$: Kích thước `vocabulary`.
* $y_{j}^{(i)}$: Giá trị thực tế (`actual`) của từ thứ $j$ trong ví dụ thứ $i$.
* $\hat{y}_{j}^{(i)}$: Giá trị dự đoán (`predicted`) của từ thứ $j$ trong ví dụ thứ $i$.

> Cho ma trận **predicted center word** của bạn ($\hat{Y}$) và ma trận **actual center word** ($Y_{\text{true}}$), bạn có thể tính `loss`.

![21_Training_a_CBOW_Model_FP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W4/21_Training_a_CBOW_Model_FP.png)

---
### **Training a CBOW Model: Backpropagation and Gradient Descent**
---

Nội dung này tập trung vào các kỹ thuật để tối thiểu hóa **cost** (chi phí) trong `neural networks`, cụ thể thông qua **backpropagation** và **gradient descent**.

#### Backpropagation (Lan truyền ngược)

* **Backpropagation** là một `algorithm` (thuật toán) tính toán các **partial derivatives** (đạo hàm riêng) của `cost` đối với các `weights` (trọng số) và `biases` (độ lệch) của `neural network`.
* Nó sử dụng **chain rule** (quy tắc chuỗi) cho các đạo hàm, bắt đầu từ `output layer` và tính toán ngược trở lại qua các lớp.

#### Gradient Descent (Giảm độ dốc)

* **Gradient Descent** là một phương pháp điều chỉnh các `weights` và `biases` bằng cách sử dụng các **gradients** (độ dốc) đã tính toán để tối thiểu hóa `cost`.
* **Learning rate** ($\alpha$) là một **hyperparameter** (siêu tham số) kiểm soát kích thước của các `updates` (cập nhật) đối với các `weights` và `biases`.

#### Công thức cập nhật Trọng số và Độ lệch

Các công thức sau được sử dụng để điều chỉnh `weights` ($W$) và `biases` ($b$) trong mỗi bước lặp:

* **Cập nhật Trọng số:**

$$W := W - \alpha \frac{\partial J}{\partial W}$$

* **Cập nhật Độ lệch:**

$$b := b - \alpha \frac{\partial J}{\partial b}$$

`Learning rates` **nhỏ hơn** cho phép `updates` dần dần và chính xác, trong khi `rates` **lớn hơn** cho phép `updates` nhanh hơn, nhưng có nguy cơ bỏ lỡ điểm tối thiểu.

Bản tóm tắt này gói gọn các khái niệm chính liên quan đến `training` một **continuous bag of words model** (`CBOW`) trong bối cảnh `neural networks`.

Quá trình **Backpropagation** (lan truyền ngược) và **Gradient Descent** (giảm độ dốc) được tóm tắt như sau:

> Backpropagation (Lan truyền ngược)

> **Backpropagation** là quá trình tính toán **partial derivatives** (đạo hàm riêng) của hàm chi phí (`cost function`) $J_{\text{batch}}$ đối với tất cả các tham số (`parameters`) của mô hình: $\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2$.

> Khi thực hiện **back-prop** trong mô hình **CBOW** này, bạn cần tính toán các đạo hàm sau:

$$\frac{\partial J_{\text{batch}}}{\partial \mathbf{W}_1}, \quad \frac{\partial J_{\text{batch}}}{\partial \mathbf{W}_2}, \quad \frac{\partial J_{\text{batch}}}{\partial \mathbf{b}_1}, \quad \frac{\partial J_{\text{batch}}}{\partial \mathbf{b}_2}$$

> Gradient Descent (Giảm độ dốc)

> **Gradient Descent** sử dụng các đạo hàm đã tính ở trên để cập nhật các tham số, nhằm tối thiểu hóa chi phí $J_{\text{batch}}$. Các công thức cập nhật được lặp lại (`iterate`) như sau:

$$\mathbf{W}_{1} := \mathbf{W}_{1} - \alpha \frac{\partial J_{\text {batch }}}{\partial \mathbf{W}_{1}}$$

$$\mathbf{W}_{2} := \mathbf{W}_{2} - \alpha \frac{\partial J_{\text {batch }}}{\partial \mathbf{W}_{2}}$$

$$\mathbf{b}_{1} := \mathbf{b}_{1} - \alpha \frac{\partial J_{\text {batch }}}{\partial \mathbf{b}_{1}}$$

$$\mathbf{b}_{2} := \mathbf{b}_{2} - \alpha \frac{\partial J_{\text {batch }}}{\partial \mathbf{b}_{2}}$$

> **Learning rate** ($\alpha$) là một **hyperparameter** (siêu tham số) quan trọng kiểm soát tốc độ học:

* **$\alpha$ nhỏ hơn** cho phép các cập nhật **gradual** (dần dần) đối với các `weights` và `biases`.
* **$\alpha$ lớn hơn** cho phép cập nhật **faster** (nhanh hơn).

> **Lưu ý:** Nếu $\alpha$ quá lớn, bạn có thể **vượt quá** điểm tối thiểu và không học được gì; nếu nó quá nhỏ, `model` của bạn sẽ mất rất nhiều thời gian để `training`.

---
### **Extracting Word Embedding Vectors**
---


---
### **Evaluating Word Embeddings: Intrinsic Evaluation**
---


---
### **Evaluating Word Embeddings: Extrinsic Evaluation**
---



