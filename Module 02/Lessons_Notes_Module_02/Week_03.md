# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 3: Autocomplete and Language Models**
---
### **N-Grams: Overview**
---
#### Understanding N-grams

- **N-grams** là các chuỗi từ (`sequences of words`) được sử dụng trong các `NLP tasks` khác nhau như nhận dạng giọng nói (`speech recognition`) và sửa lỗi chính tả (`spelling correction`).
- Một `language model` tính toán **xác suất** (`probabilities`) của các câu và dự đoán từ tiếp theo dựa trên các từ trước đó.

#### Building an N-gram Language Model

- Bạn sẽ tạo một **n-gram language model** từ một **text corpus** (một tập hợp lớn các tài liệu văn bản).
- `Model` sẽ được sử dụng để **auto-completing** (tự động hoàn thành) các câu bằng cách gợi ý các từ có khả năng xảy ra dựa trên `input` của người dùng.

#### Techniques and Applications

- Khóa học sẽ đề cập đến việc xử lý các từ **out-of-vocabulary** (ngoài từ vựng) và sử dụng các **smoothing techniques** (kỹ thuật làm mịn) để ước tính `probabilities` cho các từ chưa từng thấy (`unseen words`).
- `Language models` cũng được tận dụng trong các hệ thống **augmentative communication** (giao tiếp tăng cường), giúp người dùng bị suy giảm khả năng nói hình thành câu.

Đến cuối tuần, bạn sẽ triển khai một `model auto-completion` câu sử dụng các kỹ năng đã học.

> `N-grams` là nền tảng và cung cấp cho bạn cơ sở để hiểu các `models` phức tạp hơn trong chuyên ngành này. Các `models` này cho phép bạn tính toán `probabilities` (xác suất) của các từ nhất định xảy ra trong một `sequence` cụ thể. Sử dụng điều đó, bạn có thể xây dựng một công cụ `auto-correct` (tự động sửa lỗi) hoặc thậm chí là một công cụ gợi ý tìm kiếm (`search suggestion tool`).

> Các ứng dụng khác của `N-gram language modeling` bao gồm:

![01_N-Grams](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/01_N-Grams.png)

> Tuần này bạn sẽ học cách:

- `Process` một `text corpus` (kho ngữ liệu văn bản) thành `N-gram language model`.
- Xử lý các từ **out of vocabulary** (ngoài từ vựng).
- Triển khai `smoothing` (làm mịn) cho các `N-grams` chưa từng thấy trước đây.
- `Evaluation` (đánh giá) `language model`.

---
### **N-grams and Probabilities**
---






---
### **Sequence Probabilities**
---



---
### **Starting and Ending Sentences**
---


---
### **The N-gram Language Model**
---



---
### **Language Model Evaluation**
---


---
### **Out of Vocabulary Words**
---


---
### **Smoothing**
---



---
### **Week Summary**
---

