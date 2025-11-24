# **Module 03 - Natural Language Processing with Sequence Models**
## **Week 2: LSTMs and Named Entity Recognition**
---
### **RNNs and Vanishing Gradients**
---

Bài giảng này tập trung vào các `Long Short-Term Memory` (`LSTM`) `cells` và những thách thức mà các `Recurrent Neural Networks` (`RNNs`) thông thường phải đối mặt, đặc biệt là các vấn đề `vanishing` và `exploding gradients`.

**Understanding RNNs**

* `RNNs` mô hình hóa các `sequences` bằng cách gợi nhớ thông tin từ quá khứ ngay trước đó (`immediate past`), nắm bắt các `dependencies` ở một mức độ nào đó.
* Chúng `lightweight` so với các `models` khác nhưng gặp khó khăn với các `long-term dependencies` và dễ bị `vanishing` và `exploding gradients`.

**Vanishing and Exploding Gradients**

* Những vấn đề này nảy sinh trong quá trình `backpropagation through time`, nơi thông tin từ các bước trước đó bị suy giảm hoặc tăng trưởng không kiểm soát.
* `Vanishing gradients` dẫn đến việc các bước đầu tiên bị bỏ qua, trong khi `exploding gradients` gây ra các vấn đề về `convergence` trong quá trình `training`.

**Solutions to Gradient Problems**

* Để giảm thiểu `vanishing gradients`, các `weights` có thể được khởi tạo thành `identity matrix` và có thể sử dụng `ReLU activation`.
* `Gradient clipping` có thể giới hạn độ lớn của các `gradients` để ngăn chặn `exploding gradients`.
* `Skip connections` có thể cung cấp các liên kết trực tiếp đến các `layers` sớm hơn, tăng cường ảnh hưởng của các `early activations` lên `cost function`.

Bài giảng đặt nền móng cho việc giới thiệu `LSTMs` như một giải pháp cho những thách thức này trong video tiếp theo.

> **Advantages of RNNs**

- `RNNs` cho phép chúng ta nắm bắt các `dependencies` trong phạm vi ngắn và chúng chiếm ít `RAM` hơn các `n-gram models` khác.

> **Disadvantages of RNNs**
- `RNNs` gặp khó khăn với các `dependencies` dài hạn hơn và rất dễ bị `vanishing` hoặc `exploding gradients`.

![01_RNNs_and_Vanishing_Gradients](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/01_RNNs_and_Vanishing_Gradients.png)

> Lưu ý rằng khi bạn thực hiện `back-propagating through time`, bạn sẽ nhận được kết quả như sau:

> Lưu ý rằng các hàm `sigmoid` và `tanh` bị giới hạn bởi 0 và 1, và -1 và 1 tương ứng. Điều này cuối cùng dẫn chúng ta đến một vấn đề. Nếu bạn có nhiều số nhỏ hơn $|1|$, thì khi bạn đi qua nhiều `layers`, và bạn lấy tích của những số đó, cuối cùng bạn sẽ nhận được một `gradient` rất gần với 0. Điều này dẫn đến vấn đề `vanishing gradients`.

> **Solutions to Vanishing Gradient Problems**

![02_RNNs_and_Vanishing_Gradients](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/02_RNNs_and_Vanishing_Gradients.png)

---
### **Introduction to LSTMs**
---

Nội dung tập trung vào các mạng `Long Short-Term Memory` (`LSTM`), được thiết kế để giải quyết vấn đề `vanishing gradients` trong `recurrent neural networks` (`RNNs`).

**LSTM Architecture**

* Bao gồm `cell state` (bộ nhớ) và `hidden state` (nơi các `computations` diễn ra).
* Sử dụng nhiều `gates` để quản lý luồng thông tin, cho phép các `gradients` lưu thông hiệu quả.

**Gates in LSTM**

* `Forget Gate`: Quyết định thông tin nào cần loại bỏ khỏi `cell state`.
* `Input Gate`: Xác định thông tin mới nào cần thêm vào `cell state`.
* `Output Gate`: Chọn thông tin từ `cell state` để được `outputted` tại mỗi `timestep`.

**Applications of LSTMs**

* Hữu ích trong `language modeling`, chẳng hạn như dự đoán ký tự tiếp theo trong văn bản hoặc xây dựng `chatbots`.
* Có thể áp dụng trong `music composition`, `automatic image captioning`, và `speech recognition`.
* `LSTMs` đã thúc đẩy đáng kể các khả năng của `natural language processing` (`NLP`).

> `LSTM` cho phép `model` của bạn ghi nhớ và quên một số `inputs` nhất định. Nó bao gồm một `cell state` và một `hidden state` với ba `gates`. Các `gates` cho phép các `gradients` lưu thông mà không bị thay đổi. Bạn có thể hình dung về ba `gates` như sau:

* `Input gate`: cho bạn biết lượng thông tin cần đưa vào (`input`) tại bất kỳ `time point` nào.
* `Forget gate`: cho bạn biết lượng thông tin cần quên tại bất kỳ `time point` nào.
* `Output gate`: cho bạn biết lượng thông tin cần truyền qua tại bất kỳ `time point` nào.

> Có nhiều ứng dụng bạn có thể sử dụng `LSTMs`, chẳng hạn như:

> Dưới đây là một bài viết kinh điển về `LSTMs`[https://colah.github.io/posts/2015-08-Understanding-LSTMs/] với các giải thích trực quan và sơ đồ, để bổ sung cho tài liệu của tuần này.

![03_Introduction_to_LSTMs](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/03_Introduction_to_LSTMs.png)


---
### **Markov Chains and POS Tags**
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
