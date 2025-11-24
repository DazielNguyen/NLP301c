# **Module 04 - Natural Language Processing with Attention Models Models**
## **Week 3: Question Answering**
---
### **Week 3 Overview**
---
**Transfer Learning**

* `Transfer learning` cho phép các `models` tận dụng kiến thức thu được từ `task` này để cải thiện `performance` trên `task` khác, giảm thời gian `training` và yêu cầu về dữ liệu.
* Một ví dụ là sử dụng một `pre-trained model` trên các đánh giá phim để dự đoán `ratings` cho các đánh giá khóa học, bắt đầu với các `weights` hiện có thay vì khởi tạo từ đầu.

**Question Answering**

* Khóa học đề cập đến hai loại `question answering`: `context-based` (dựa trên ngữ cảnh), sử dụng `context` được cung cấp để tìm câu trả lời, và `closed book` (sách đóng), tạo ra câu trả lời mà không cần `context`.
* Các `models` như `BERT` sử dụng `bi-directional context` để nâng cao `performance`, trong khi `T5` có thể xử lý nhiều `tasks`, chẳng hạn như trả lời câu hỏi và dự đoán `ratings`, bằng cách sử dụng một `model` duy nhất.

**Key Takeaways**

* Các đổi mới trong các phương pháp `training`, chẳng hạn như `transfer learning`, có thể cải thiện đáng kể `model performance`.
* Càng có nhiều dữ liệu sẵn có để `training`, `model performance` càng có xu hướng tốt hơn, như thấy với `T5 model` được `trained` trên một `large dataset`.

> Chào mừng đến với Tuần 3! Trong tuần này, bạn sẽ tìm hiểu về `transfer learning` và cụ thể bạn sẽ hiểu cách `T5` và `BERT` thực sự hoạt động.

![01_Week_3_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/01_Week_3_Overview.png)

> Trong hình ảnh phía trên, bạn có thể thấy cách một `model` ban đầu được `trained` trên một loại `sentiment classification`, giờ đây có thể được sử dụng cho `question answering`.

> Một `model state of the art` khác sử dụng `multi tasking`. Ví dụ, cùng một `model` có thể được sử dụng cho `sentiment analysis`, `question answering`, và nhiều thứ khác.

![02_Week_3_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/02_Week_3_Overview.png)

> Các loại `models` mới này sử dụng rất nhiều dữ liệu. Ví dụ, `C4` (`colossal cleaned crawled corpus`) có dung lượng khoảng 800 GB trong khi toàn bộ `Wikipedia` tiếng Anh chỉ là 13 GB!


---
### **Transfer Learning in NLP**
---

Nội dung bài giảng này tập trung vào khái niệm `transfer learning` trong `natural language processing` (`NLP`) và ứng dụng của nó trong các `models` khác nhau.

**Transfer Learning Overview**

* `Transfer learning` liên quan đến việc sử dụng các `pre-trained models` để cải thiện `performance` trên các `tasks` cụ thể.
* Nó có thể được triển khai thông qua `feature-based learning` (ví dụ: `word vectors`) hoặc `fine-tuning` các `models` hiện có.


**Feature-Based Learning vs. Fine-Tuning**

* `Feature-based learning` sử dụng các `embeddings` làm `input features` cho các `models` khác nhau để đưa ra `predictions`.
* `Fine-tuning` liên quan đến việc điều chỉnh các `weights` của một `pre-trained model` cho một `downstream task` cụ thể, chẳng hạn như `sentiment analysis`.

**Data and Performance**

* Số lượng và chất lượng của dữ liệu ảnh hưởng đáng kể đến `model performance`; nhiều dữ liệu hơn thường dẫn đến kết quả tốt hơn.
* `Labeled data` thường khan hiếm so với `unlabeled data`, cái mà có thể được tận dụng trong các `self-supervised tasks` để tạo `input features` và `targets`.

**Pre-Training and Downstream Tasks**

* Các `pre-training tasks` có thể bao gồm `language modeling`, nơi các `models` dự đoán các `masked words` hoặc câu tiếp theo.
* `Fine-tuning` có thể được áp dụng cho các `tasks` khác nhau, bao gồm dịch thuật (`translation`), tóm tắt (`summarization`), và `question answering`, sử dụng `pre-trained model`.

> Có ba lợi ích chính của `transfer learning`:

* Giảm `training time` (thời gian huấn luyện)
* Cải thiện `predictions` (dự đoán)
* Cho phép bạn sử dụng các `datasets` nhỏ hơn

> Hai phương pháp mà bạn có thể sử dụng cho `transfer learning` là:

![03_Transfer_Learning_in_NLP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/03_Transfer_Learning_in_NLP.png)

> Trong `feature based transfer learning`, bạn học `word embeddings` bằng cách `training` một `model` và sau đó bạn sử dụng các `word embeddings` đó trong một `model` khác cho một `task` khác.

> Khi `fine tuning`, bạn có thể sử dụng chính xác cùng một `model` và chỉ chạy nó trên một `task` khác. Đôi khi khi `fine tuning`, bạn có thể giữ các `model weights` cố định và chỉ thêm một `layer` mới mà bạn sẽ `train`. Những lúc khác bạn có thể từ từ `unfreeze` (rã đông) các `layers` từng cái một. Bạn cũng có thể sử dụng `unlabelled data` khi `pre-training`, bằng cách `masking` các từ và cố gắng `predict` từ nào đã bị `mask`.

![04_Transfer_Learning_in_NLP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/04_Transfer_Learning_in_NLP.png)

> Ví dụ, trong hình vẽ trên, chúng ta cố gắng `predict` từ "friend". Điều này cho phép `model` của bạn nắm bắt được cấu trúc tổng thể của dữ liệu và giúp `model` học được một số mối quan hệ trong các từ của một `sentence`.

---
### **ELMo, GPT, BERT, T5**
---

Nội dung thảo luận về sự tiến hóa của các `models natural language processing` (`NLP`) khác nhau cùng các ưu điểm và nhược điểm tương ứng của chúng.

**Model Evolution**

* Dòng thời gian bao gồm các `models` như `Continuous Bag of Words`, `ELMo`, `GPT`, `BERT`, và `T5`, làm nổi bật sự phát triển của chúng và các vấn đề mà chúng giải quyết.
* Mỗi `model` xây dựng dựa trên các khái niệm trước đó, dẫn đến những cải tiến trong việc hiểu `context` và `word embeddings`.

**Contextual Understanding**

* Tầm quan trọng của `context` trong việc hiểu các từ được nhấn mạnh, với các phương pháp như `fixed window sizes` và `bi-directional LSTMs` được khám phá.
* Cách tiếp cận `bi-directional` của `BERT` cho phép sử dụng `context` tốt hơn so với các `models` trước đó.

**Model Architectures**

* Các `architectures` khác nhau được thảo luận: `GPT` sử dụng `decoder-only stack`, trong khi `BERT` sử dụng `encoder-only stack`.
* `T5` kết hợp cả `encoder` và `decoder stacks`, cho thấy `performance` được cải thiện trong các `tasks` khác nhau thông qua các `multi-task training strategies`.

Bản tóm tắt này gói gọn các điểm chính liên quan đến sự phát triển và chức năng của các `NLP models`.

> Các `models` được đề cập trong video trước đã được khám phá theo thứ tự sau:

![05_ELMo_GPT_BERT_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/05_Transfer_Learning_in_NLP.png)


> Trong `CBOW`, bạn muốn mã hóa một từ dưới dạng một `vector`. Để làm điều này, chúng ta sử dụng `context` trước từ và `context` sau từ và chúng ta sử dụng `model` đó để học và tạo ra các `features` cho từ. Tuy nhiên, `CBOW` sử dụng một `fixed window C` (cho `context`).

> `ElMo` sử dụng một `bi-directional LSTM`, đây là một phiên bản khác của `RNN` và bạn có các `inputs` từ bên trái và bên phải.

> Sau đó `Open AI` đã giới thiệu `GPT`, một `uni-directional model` sử dụng `transformers`. Mặc dù `ElMo` là `bi-directional`, nó vẫn gặp một số vấn đề như nắm bắt các `longer-term dependencies`, điều mà `transformers` giải quyết tốt hơn nhiều.

> Sau đó, `Bi-directional Encoder Representation from Transformers` (`BERT`) được giới thiệu, cái mà tận dụng các `bi-directional transformers` như tên gọi của nó.

> Cuối cùng, `T5` được giới thiệu, cái mà sử dụng `transfer learning` và sử dụng cùng một `model` để `predict` (dự đoán) trên nhiều `tasks`. Dưới đây là một minh họa về cách nó hoạt động:

![06_ELMo_GPT_BERT_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/06_Transfer_Learning_in_NLP.png)

---
### **Bidirectional Encoder Representations from Transformers (BERT)**
---

Nội dung tập trung vào `Bidirectional Encoder Representations from Transformers` (`BERT`), một `model` sử dụng `transformer architecture` để xử lý các `inputs` theo cả hai hướng (`bidirectional`).

**BERT Architecture**

* `BERT` sử dụng một `multi-layer bidirectional transformer` với `positional embeddings`.
* `Base model` bao gồm 12 `layers`, 12 `attention heads`, và 110 triệu `parameters`.

**Pre-training Process**

* `BERT` được `pre-trained` trên dữ liệu không có nhãn (`unlabeled data`) bằng cách sử dụng các `tasks` như `masked language modeling` và `next sentence prediction`.
* Trong quá trình `pre-training`, 15% số từ trong `input` được `masked` (che), và `model` dự đoán các từ bị `masked` này.

**Fine-tuning**

* Sau `pre-training`, `BERT` được `fine-tuned` trên dữ liệu có nhãn (`labeled data`) cho các `downstream tasks` cụ thể.
* Các `parameters` của `model` được điều chỉnh để cải thiện `performance` trên các `tasks` này, chẳng hạn như `classification` hoặc `prediction`.

> Bây giờ bạn sẽ tìm hiểu về `BERT architecture` và hiểu cách `pre-training` hoạt động.

![07_BERT](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/07_BERT.png)


> Có hai bước trong `BERT framework`: `pre-training` và `fine-tuning`. Trong quá trình `pre-training`, `model` được `trained` trên dữ liệu không có nhãn (`unlabeled data`) thông qua các `pre-training tasks` khác nhau. Đối với `fine-tuning`, `BERT model` đầu tiên được `initialize` (khởi tạo) với các `pre-trained parameters`, và tất cả các `parameters` đều được `fine-tuned` bằng cách sử dụng dữ liệu có nhãn (`labeled data`) từ các `downstream tasks`.

> Ví dụ, trong hình ảnh phía trên, bạn nhận được các `embeddings` tương ứng cho các từ `input`, bạn chạy nó qua một vài `transformer blocks`, và sau đó bạn đưa ra `prediction` tại mỗi `time point` $T_i$.

> Trong quá trình `pre-training`:
* Chọn 15% `tokens` một cách ngẫu nhiên: `mask` chúng 80% thời gian, thay thế chúng bằng một `random token` 10% thời gian, hoặc giữ nguyên 10% thời gian.
* Có thể có nhiều `masked spans` (khoảng che) trong một câu.
* `Next sentence prediction` cũng được sử dụng khi `pre-training`.

---
### **BERT Objective**
---

Nội dung tập trung vào `input representation` và `objective` của `BERT model` trong `natural language processing`.

**Input Representation**

* `BERT` sử dụng `position embeddings` để chỉ ra vị trí của mỗi từ trong một câu.
* `Segment embeddings` phân biệt giữa `sentence A` và `sentence B`, điều này cần thiết cho `next sentence prediction`.

**Combining Inputs**

* `Token embeddings`, bao gồm một **CLS token** cho điểm bắt đầu và một **SEP token** cho điểm kết thúc của câu, được cộng tổng với `position` và `segment embeddings` để tạo ra `final input`.
* `Input` được chuyển đổi thành các `embeddings` được xử lý qua các `transformer blocks`.

**BERT Objective**

* `Model` sử dụng một **Multi-Mask language model** dùng `cross-entropy loss` để dự đoán các từ bị `masked`.
* Một **binary loss** được thêm vào cho **next sentence prediction**, xác định xem hai câu có theo sau nhau hay không.

> Chúng ta sẽ bắt đầu bằng cách trực quan hóa (`visualizing`) `input`.

![08_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/08_BERT_Objective.png)

> Các `input embeddings` là tổng của các `token embeddings`, các `segmentation embeddings` và các `position embeddings`.
* Các `input embeddings`: bạn có một **CLS token** để chỉ ra điểm bắt đầu của câu và một **SEP** để chỉ ra điểm kết thúc của câu.
* Các `segment embeddings`: cho phép bạn chỉ ra đó là `sentence A` hay `B`.
* Các `positional embeddings`: cho phép bạn chỉ ra vị trí của từ trong câu.

![09_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/09_BERT_Objective.png)


> `C token` trong hình ảnh phía trên có thể được sử dụng cho các mục đích phân loại (`classification purposes`). Cặp `unlabeled sentence A/B` sẽ phụ thuộc vào những gì bạn đang cố gắng `predict` (dự đoán), nó có thể bao gồm từ `question answering` đến `sentiment` (trong trường hợp đó, câu thứ hai có thể chỉ là trống). `BERT objective` được định nghĩa như sau:

![10_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/10_BERT_Objective.png)

> Bạn chỉ cần kết hợp các `losses`!

---
### **Fine tuning BERT**
---


---
### **Transformer: T5**
---


---
### **Multi-Task Training Strategy**
---



---
### **GLUE Benchmark**
---


---
### **Hugging Face Introduction**
---


---
### **Hugging Face I**
---



---
### **Hugging Face II**
---




---
### **Hugging Face III**
---



---
### **Andrew Ng with Quoc Le**
---


