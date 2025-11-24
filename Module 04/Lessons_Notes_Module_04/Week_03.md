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



---
### **Bidirectional Encoder Representations from Transformers (BERT)**
---


---
### **BERT Objective**
---



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


