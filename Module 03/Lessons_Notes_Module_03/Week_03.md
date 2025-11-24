# **Module 03 - Natural Language Processing with Sequence Models**
## **Week 3: Siamese Networks**
---
### **Siamese Networks**
---

Trong bài giảng này, bạn sẽ tìm hiểu về `Siamese networks`, một loại `neural network architecture` cụ thể bao gồm hai `networks` giống hệt nhau được hợp nhất ở phần cuối, với nhiều ứng dụng khác nhau trong `natural language processing` (`NLP`).

**Siamese Networks Overview**

* `Siamese networks` được thiết kế để so sánh sự tương đồng (`similarity`) giữa hai `inputs`, chẳng hạn như các câu hỏi hoặc câu nói.
* Chúng đặc biệt hữu ích trong việc xác định các bản sao (`duplicates`) trong các câu hỏi, ngay cả khi được diễn đạt khác nhau.

**Applications in NLP**

* Các `networks` này có thể được sử dụng để xác thực chữ ký bằng cách so sánh hai chữ ký viết tay về độ tương đồng.
* Chúng giúp xác định các câu hỏi trùng lặp trên các nền tảng như Quora hoặc Stack Overflow, đảm bảo người dùng không đăng cùng một câu hỏi nhiều lần.

**Sentiment Analysis and Classification**

* Trong `sentiment analysis`, `Siamese networks` có thể xác định `sentiment` của các phát biểu bằng cách so sánh các `features` biểu thị `sentiment` tích cực hoặc tiêu cực.
* Một `similarity score` được tính toán để đánh giá mối quan hệ giữa hai `inputs`, hỗ trợ trong các tác vụ phân loại (`classification tasks`).

Bài giảng đặt nền tảng cho việc khám phá sâu hơn về `architecture` và các ứng dụng thực tế của `Siamese networks` trong `NLP`.

> Tốt nhất là mô tả một `Siamese network` thông qua một ví dụ.

![01_Siamese_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/01_Siamese_Networks.png)

> Lưu ý rằng trong ví dụ đầu tiên ở trên, hai câu có nghĩa giống nhau nhưng có các từ hoàn toàn khác nhau. Trong khi ở trường hợp thứ hai, hai câu có nghĩa hoàn toàn khác nhau nhưng chúng có các từ rất giống nhau.

> `Classification`: học điều gì làm cho một `input` là chính nó (what it is).

> `Siamese Networks`: học điều gì làm cho hai `inputs` giống nhau.

> Dưới đây là một vài ứng dụng của `siamese networks`:

![02_Siamese_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/02_Siamese_Networks.png)

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

