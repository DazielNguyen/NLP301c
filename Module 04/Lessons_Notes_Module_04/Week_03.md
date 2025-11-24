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


