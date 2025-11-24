# **Module 04 - Natural Language Processing with Attention Models Models**
## **Week 2: Text Summarization**
---
### **Transformers vs RNNs**
---

Bài giảng tập trung vào `transformer model`, một `architecture` hoàn toàn dựa trên `attention` được phát triển bởi Google để giải quyết các vấn đề với `recurrent neural networks` (`RNNs`).

**Understanding RNN Limitations**

* `RNNs` xử lý `input` một cách tuần tự (`sequentially`), điều này hạn chế `parallel computation` (tính toán song song) và làm tăng thời gian xử lý cho các câu dài hơn.
* Khi `sequence length` tăng lên, thông tin có thể bị mất, dẫn đến các vấn đề `vanishing gradient`, ngay cả với các `architectures` tiên tiến như `LSTMs` và `GRUs`.

**Introduction to Transformers**

* `Transformers` sử dụng các `attention mechanisms` một cách độc quyền, loại bỏ sự cần thiết của các `recurrent networks`.
* `Model` này cho phép xử lý hiệu quả hơn các `long sequences`, giải quyết các vấn đề về `context retention` (lưu giữ ngữ cảnh) mà `RNNs` phải đối mặt.

**Next Steps**

* Bài giảng đặt nền móng cho việc khám phá sâu hơn về `transformers` trong video tiếp theo, hứa hẹn một cái nhìn tổng quan cụ thể về `structure` và `functionality` của chúng.

![01_Transformers_vs_RNNs](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/01_Transformers_vs_RNNs.png)

> Trong hình ảnh phía trên, bạn có thể thấy một `RNN` điển hình được sử dụng để dịch câu tiếng Anh "How are you?" sang câu tương đương tiếng Pháp, "Comment allez-vous?".



> Một trong những vấn đề lớn nhất với các `RNNs` này, là chúng sử dụng `sequential computation` (tính toán tuần tự). Điều đó có nghĩa là, để `code` của bạn `process` từ "you", trước tiên nó phải đi qua "How" và "are". Hai vấn đề khác với `RNNs` là:
* `Loss of information` (mất thông tin): Ví dụ, khó hơn để theo dõi xem chủ ngữ là số ít hay số nhiều khi bạn di chuyển ra xa chủ ngữ.
* `Vanishing Gradient` (biến mất gradient): khi bạn `back-propagate`, các `gradients` có thể trở nên thực sự nhỏ và kết quả là, `model` của bạn sẽ không học được nhiều.

> Ngược lại, `transformers` dựa trên `attention` và không yêu cầu bất kỳ `sequential computation` nào trên mỗi `layer`, chỉ cần một bước duy nhất. Ngoài ra, các `gradient steps` cần thực hiện từ `last output` đến `first input` trong một `transformer` chỉ là một. Đối với `RNNs`, số lượng các bước tăng lên với các `sequences` dài hơn. Cuối cùng, `transformers` không bị các vấn đề `vanishing gradients` liên quan đến độ dài của các `sequences`.

> Chúng ta sẽ nói thêm về cách `attention component` hoạt động với `transformers`. Vì vậy đừng lo lắng về nó lúc này :)

---
### **Transformers overview**
---



---
### **Transformer Applications**
---



---
### **Hidden Markov Models**
---


---
### **Calculating Probabilities**
---



---
### **Populating the Transition Matrix**
---


---
### **Populating the Emission Matrix**
---


---
### **The Viterbi Algorithm**
---



---
### **Viterbi: Initialization**
---


---
### **Viterbi: Forward Pass**
---

---
### **Viterbi: Backward Pass**
---
