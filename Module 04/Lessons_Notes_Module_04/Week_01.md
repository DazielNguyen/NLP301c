# **Module 04 - Natural Language Processing with Attention Models Models**
## **Week 1: Neural Machine Translation**
---
### **Seq2seq**
---

Nội dung này cung cấp một cái nhìn tổng quan về `neural machine translation` (dịch máy thần kinh) và `seq2seq model` được sử dụng cho tác vụ này.

**Neural Machine Translation Overview**

* `Neural machine translation` sử dụng `encoder-decoder architecture` để dịch văn bản từ ngôn ngữ này sang ngôn ngữ khác.
* `Seq2seq model` truyền thống, được Google giới thiệu vào năm 2014, ánh xạ các `input sequences` có độ dài thay đổi thành bộ nhớ có độ dài cố định, mã hóa ý nghĩa tổng thể của các câu.



**Encoder and Decoder Structure**

* `Encoder` bao gồm một `embedding layer` và một `LSTM module`, chuyển đổi các `word tokens` thành các `vectors` và trả về `final hidden state` giúp mã hóa ý nghĩa của câu.
* `Decoder` cũng có một `embedding layer` và một `LSTM`, tạo ra câu đã dịch từng bước một, sử dụng `output` trước đó làm `input` cho bước tiếp theo.

**Limitations of the Seq2seq Model**

* Một hạn chế lớn là `information bottleneck` (nút thắt thông tin), trong đó các `input sequences` dài có thể dẫn đến `model performance` thấp hơn do các `hidden states` có độ dài cố định.
* `Bottleneck` này hạn chế lượng thông tin được truyền từ `encoder` sang `decoder`, gây khó khăn cho việc dịch chính xác các `sequences` dài hơn.

**Attention Mechanism**

* Để giải quyết các hạn chế này, một `attention mechanism` cho phép `model` tập trung vào các từ quan trọng nhất tại mỗi `decoding step`.
* Cơ chế này tăng cường khả năng xử lý thông tin hiệu quả của `model`, cải thiện `translation accuracy`.

---
### **Seq2seq Model with Attention**
---



---
### **Building the model**
---



---
### **Building the model II**
---


---
### **Minimum edit distance**
---



---
### **Minimum edit distance algorithmn**
---


---
### **Minimum edit distance algorithmn II**
---


---
### **Minimum edit distance algorithmn III**
---



---
### **Minimum edit distance III**
---


