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

Nội dung tập trung vào khái niệm `attention` trong `machine learning`, đặc biệt là trong bối cảnh `natural language processing`.

**Understanding Attention**

* `Attention` cho phép các `models` tập trung vào các phần cụ thể của `input` khi đưa ra các `predictions`, cải thiện các tác vụ như dịch thuật.
* Nó được giới thiệu trong một bài báo bởi Bahdanau, Cho, và Bengio để nâng cao `performance` của các `sequence-to-sequence models` trong việc dịch các câu dài hơn.

**Performance Comparison**

* Các `attention-based models` hoạt động tốt hơn các `sequence-to-sequence models` truyền thống, đặc biệt là với các câu dài hơn.
* Các `RNN search models` với `attention` duy trì `performance` mà không bị suy giảm đáng kể khi độ dài câu tăng lên.

**Mechanism of Attention**

* Các `models` truyền thống sử dụng `final hidden state` của `encoder`, điều này có thể hạn chế `performance`; `attention` cho phép sử dụng tất cả các `hidden states`.
* `Context vector` được tạo ra bằng cách đánh trọng số các `encoder states`, tập trung vào các `inputs` liên quan nhất cho các `predictions` của `decoder`.

**Calculating Weights**

* `Weights` được xác định bằng cách so sánh các `previous hidden states` của `decoder` với các `encoder states` để xác định các `inputs` quan trọng.
* `Attention layer` tính toán các `alignment scores`, các điểm này được chuyển đổi thành `weights` bằng cách sử dụng một `softmax function`, dẫn đến một `context vector` tóm tắt thông tin liên quan.

**Next Steps**

* Nội dung tiếp theo sẽ giải thích các khái niệm về `keys`, `queries`, và `values` trong các `attention mechanisms`.

---
### **Background on seq2seq**
---
Các `recurrent models` thường nhận vào một `sequence` theo thứ tự nó được viết và sử dụng nó để xuất ra một `sequence`. Mỗi phần tử trong `sequence` được liên kết với bước của nó trong thời gian tính toán $t$. (tức là nếu một từ nằm ở phần tử thứ ba, nó sẽ được tính toán tại $t_3$). Các `models` này tạo ra một `sequence` các `hidden states` $h_t$, như một hàm của `hidden state` trước đó $h_{t-1}$ và `input` cho vị trí $t$.

Bản chất tuần tự (`sequential nature`) của các `models` bạn đã học trong khóa học trước (`RNNs`, `LSTMs`, `GRUs`) không cho phép `parallelization` (song song hóa) trong các `training examples`, điều này trở nên quan trọng ở các độ dài `sequence` dài hơn, vì các hạn chế về bộ nhớ giới hạn việc `batching` giữa các ví dụ.



Nói cách khác, nếu bạn dựa vào các `sequences` và bạn cần biết phần đầu của văn bản trước khi có thể tính toán điều gì đó về phần cuối của nó, thì bạn không thể sử dụng `parallel computing` (tính toán song song). Bạn sẽ phải đợi cho đến khi các tính toán ban đầu hoàn tất. Điều này không tốt, bởi vì nếu văn bản của bạn quá dài, thì 
- 1) sẽ mất nhiều thời gian để bạn `process` nó và 
- 2) bạn sẽ mất một lượng lớn thông tin được đề cập trước đó trong văn bản khi bạn tiến về phía cuối.

Do đó, các `attention mechanisms` đã trở nên quan trọng đối với `sequence modeling` trong các tác vụ khác nhau, cho phép mô hình hóa các `dependencies` mà không cần quan tâm quá nhiều về khoảng cách của chúng trong các `input` hoặc `output sequences`.

---
### **Queries, Keys, Values, and Attention**
---

Nội dung này tập trung vào các khái niệm `queries`, `keys`, và `values` trong bối cảnh các `attention mechanisms` trong `natural language processing`.

**Understanding Attention Mechanisms**

* `Attention mechanism` được giới thiệu vào năm 2014, với những tiến bộ đáng kể như `transformer model` vào năm 2017.
* `Queries`, `keys`, và `values` là các thành phần thiết yếu, trong đó `queries` được khớp với `keys` để truy xuất các `values` liên quan.

**Alignment and Similarity**

* `Keys` và `values` có thể được coi như một `lookup table`, với `alignment` đại diện cho sự tương đồng (`similarity`) giữa các từ trong các ngôn ngữ khác nhau.
* Các `alignment scores` được tính toán để xác định mức độ khớp nhau giữa `queries` và `keys`, sau đó được sử dụng để tạo ra các `attention vectors`.

**Scale Dot-Product Attention**

* Phương pháp này liên quan đến các `matrix multiplications` để tính toán hiệu quả (`efficient computation`), cho phép xử lý đồng thời các `queries`.
* Việc `scaling` các điểm số và việc sử dụng `softmax function` giúp tạo ra các `weights` có tổng bằng một, tạo điều kiện cho việc tạo ra các `attention vectors` cho mỗi `query`.

---
### **Setup for Machine Translation**
---

Nội dung này tập trung vào cách các từ được biểu diễn (`represented`) trong bối cảnh `neural machine translation` (dịch máy thần kinh) và cấu trúc của `dataset` được sử dụng để `training` các `models`.

**Understanding Word Representation**

* Các từ ban đầu được biểu diễn bằng cách sử dụng `one-hot vectors`, sau đó được chuyển đổi thành các chỉ số (`indices`) để xử lý.
* Một từ điển `word-to-index` và `index-to-word` được duy trì để tạo điều kiện cho sự chuyển đổi này.

**Dataset Structure**

* Dữ liệu đầu vào (`Input data`) bao gồm các câu tiếng Anh được ghép nối với các bản dịch tiếng Pháp của chúng, hiển thị các ví dụ như "I'm hungry" và "I watch the soccer game."
* Một `end-of-sequence` (`EOS`) `token` được thêm vào, và các `sequences` được `padded` với các số không để khớp với độ dài của `sequence` dài nhất.

**Model Training Preparation**

* Sau khi biểu diễn các từ và cấu trúc hóa `dataset`, người học có thể tiến hành `train` `models` của họ.
* Các bước tiếp theo sẽ liên quan đến việc `training` thực tế của `model` dựa trên dữ liệu đã chuẩn bị.

---
### **Teacher Forcing**
---

Phần này tập trung vào việc `training` các hệ thống `neural machine translation` (`NMT`), đặc biệt là sử dụng kỹ thuật được gọi là `teacher forcing`.

**Understanding Teacher Forcing**

* `Teacher forcing` liên quan đến việc sử dụng các từ mục tiêu (`target words`) thực tế làm `inputs` cho `decoder` trong quá trình `training`, thay vì sử dụng các dự đoán (`predictions`) của chính `model`.
* Phương pháp này giúp `model` học hiệu quả hơn, đặc biệt là trong các giai đoạn đầu khi nó có xu hướng đưa ra các `incorrect predictions` (dự đoán sai).

**Challenges in NMT Training**

* Các dự đoán sớm (`early predictions`) có thể dẫn đến `compounding error` (lỗi tích lũy), nơi các `outputs` sai lệch ngày càng xa so với `target sequence`.
* Một ví dụ minh họa cách một `wrong prediction` có thể dẫn đến một ý nghĩa hoàn toàn khác trong bản dịch.

**Curriculum Learning**

* Một biến thể của `teacher forcing` là `curriculum learning`, nơi `model` chuyển đổi dần dần từ việc sử dụng các `target words` sang sử dụng các `outputs` của chính nó theo thời gian.
* Cách tiếp cận này giúp `fine-tuning` (tinh chỉnh) `performance` của `model` và cải thiện `accuracy` trong các bản dịch.

---
### **Minimum edit distance algorithmn II**
---


---
### **Minimum edit distance algorithmn III**
---



---
### **Minimum edit distance III**
---


