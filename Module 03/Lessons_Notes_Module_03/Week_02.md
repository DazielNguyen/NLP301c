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
### **LSTM Architecture**
---

Video bài giảng tập trung vào `architecture` và các `computations` của các mạng `Long Short-Term Memory` (`LSTM`), vốn rất quan trọng để xử lý các `sequences` trong `natural language processing`.

**LSTM Architecture**

* `LSTMs` bao gồm một `cell state`, `hidden state`, `input`, và `output`, trong đó `cell state` đóng vai trò là bộ nhớ của `network`.
* `Architecture` bao gồm ba `gates`: `forget gate` (quyết định cái gì cần loại bỏ), `input gate` (chọn thông tin liên quan), và `output gate` (xác định `output`).

**Gate Functions**

* Các `sigmoid activation functions` được sử dụng cho các `gates`, đảm bảo các giá trị nằm trong khoảng từ 0 đến 1, trong đó 0 nghĩa là `gate` đóng và 1 nghĩa là nó mở.
* `Candidate cell state` được tính toán bằng cách sử dụng một `hyperbolic tangent activation function`, giúp chuyển đổi thông tin để cải thiện `training performance`.

**Updating States**

* `Cell state` mới được cập nhật bằng cách kết hợp thông tin từ `candidate cell state` và `cell state` trước đó, được lọc qua các `forget` và `input gates`.
* `Hidden state` mới được suy ra từ `cell state` mới, cái mà có thể đi qua `output gate`, đôi khi trực tiếp mà không cần `hyperbolic tangent transformation`.

Nhìn chung, bài giảng cung cấp một sự hiểu biết toàn diện về cách `LSTMs` hoạt động và chuẩn bị cho người học để triển khai `LSTMs` trong các ứng dụng thực tế.

> Kiến trúc `LSTM` có thể trở nên phức tạp và đừng lo lắng về điều đó nếu bạn không hiểu nó. Cá nhân tôi thích nhìn vào phương trình hơn, nhưng tôi sẽ cố gắng đưa ra một `visualization` (trực quan hóa) cho bạn bây giờ và cuối tuần này chúng ta sẽ xem xét các phương trình.

![04_LSTM_Architecture](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/04_LSTM_Architecture.png)


> Lưu ý `forget gate` (1), `input gate` (2) và `output gate` (3) được đánh dấu màu xanh dương. Ngược lại với `vanilla RNNs`, có `cell state` bên cạnh `hidden state`. Ý tưởng của `forget gate` là loại bỏ thông tin không còn quan trọng nữa. Nó sử dụng `hidden state` trước đó $h^{<t_0>}$ và `input` $x^{<t_1>}$. `Input gate` đảm bảo giữ lại các thông tin liên quan cần được lưu trữ. Cuối cùng `output gate` tạo ra một `output` được sử dụng tại bước hiện tại.

> Các phương trình `LSTM` (tùy chọn):
Để hiểu rõ hơn, hãy xem các phương trình `LSTM` và liên hệ chúng với hình trên.

> `Forget gate`:
$$f=\sigma(W_f[h_{t-1};x_t]+b_f)$$
(được đánh dấu bằng số 1 màu xanh dương)

> `Input gate`:
$$i=\sigma(W_i[h_{t-1};x_t]+b_i)$$
(được đánh dấu bằng số 2 màu xanh dương)

> `Gate gate` (candidate memory cell):
$$g=\tanh(W_g[h_{t-1};x_t]+b_g)$$

> `Cell state`:
$$c_t=f \odot c_{t-1} + i \odot g$$

> `Output gate`:
$$o=\sigma(W_o[h_{t-1};x_t]+b_o)$$
(được đánh dấu bằng số 3 màu xanh dương)

> `Output` của `LSTM unit`:
$$h_t=o_t \odot \tanh(c_t)$$

---
### **Introduction to Named Entity Recognition**
---

Video bài giảng tập trung vào `named entity recognition` (`NER`), một thành phần quan trọng trong các hệ thống `natural language processing` (`NLP`).

**Understanding Named Entity Recognition**

* Các hệ thống `NER` xác định và trích xuất các `named entities` từ văn bản, có thể bao gồm con người, tổ chức, địa điểm, ngày tháng, và nhiều hơn nữa.
* Các ví dụ về `named entities` bao gồm "Sharon" (người), "Miami" (thực thể địa lý), và "Friday" (chỉ báo thời gian).

**Applications of NER**

* `NER` tăng cường hiệu quả của `search engine` bằng cách quét và gắn thẻ hàng triệu trang web, cho phép khớp nhanh các `search queries`.
* Nó cũng được sử dụng trong `customer service` để khớp người dùng với các `agents` phù hợp dựa trên yêu cầu của họ.

**Real-World Use Cases**

* `NER` có thể tối ưu hóa các `recommendations` bằng cách so sánh lịch sử người dùng và đề xuất các mục liên quan.
* Trong tài chính, `NER` có thể được áp dụng trong `automatic trading` bằng cách phân tích các bài báo tin tức liên quan đến các cổ phiếu hoặc tiền điện tử cụ thể.

Bài giảng kết luận bằng cách nêu bật các ứng dụng khác nhau của `NER` trong `deep learning` và tầm quan trọng của nó trong `NLP`.

> `Named Entity Recognition` (`NER`) định vị và trích xuất các `entities` (thực thể) được xác định trước từ văn bản. Nó cho phép bạn tìm các địa điểm, tổ chức, tên, thời gian và ngày tháng. Dưới đây là một ví dụ về `model` mà bạn sẽ xây dựng:

![05_Introduction_to_Named_Entity_Recognition](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/05_Introduction_to_Named_Entity_Recognition.png)

> Các hệ thống `NER` đang được sử dụng trong `search efficiency` (hiệu quả tìm kiếm), `recommendation engines` (công cụ đề xuất), `customer service` (dịch vụ khách hàng), `automatic trading` (giao dịch tự động), và nhiều hơn nữa.

---
### **Training NERs: Data Processing**
---

Nội dung này tập trung vào các bước cần thiết để `train` một hệ thống `Named Entity Recognition` (`NER`), chi tiết hóa quá trình chuyển đổi văn bản thành các `numerical representations`.

**Data Preparation**

* Gán các số duy nhất cho mỗi `entity class` (ví dụ: tên người, địa điểm).
* Chuyển đổi các từ trong câu thành các `numerical arrays` tương ứng, với các từ không nhận dạng được đánh dấu là 'O'.

**Sequence Handling**

* Đảm bảo tất cả các `numerical arrays` có cùng kích thước bằng cách `padding` các `sequences` ngắn hơn với một `generic token`.
* Tạo các `tensors` cho `inputs` và `labels`, và tạo các `batches` để xử lý.

**Model Training**

* Đưa các `batches` vào một `LSTM unit`, theo sau là một `dense layer`.
* Sử dụng `LogSoftmax` cho các `predictions` để nâng cao `numerical performance` và `gradient optimization`.

Tóm tắt này gói gọn các bước thiết yếu trong việc chuẩn bị và `training` một hệ thống `NER`, nhấn mạnh tầm quan trọng của `data representation` và `model architecture`.

> `Processing data` (xử lý dữ liệu) là một trong những nhiệm vụ quan trọng nhất khi `training` các thuật toán `AI`. Đối với `NER`, bạn phải:

* Chuyển đổi các từ và `entity classes` thành các mảng (`arrays`):
* `Pad` với các `tokens`: Đặt `sequence length` thành một số nhất định và sử dụng `<PAD>` `token` để lấp đầy các khoảng trống.
* Tạo một `data generator`.

> Khi bạn đã có điều đó, bạn có thể gán cho mỗi `class` một con số, và mỗi từ một con số.

![06_Training_NERs_Data_Processing](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/06_Training_NERs_Data_Processing.png)

`Training` một hệ thống `NER`:
1.  Tạo một `tensor` cho mỗi `input` và số tương ứng của nó
2.  Đặt chúng vào một `batch` ==> 64, 128, 256, 512 ...
3.  Đưa nó vào một `LSTM unit`
4.  Chạy `output` qua một `dense layer`
5.  Dự đoán (`Predict`) sử dụng một `log softmax` trên K `classes`

Dưới đây là một ví dụ về `architecture`:

![07_Training_NERs_Data_Processing](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W2/07_Training_NERs_Data_Processing.png)

Lưu ý rằng đây chỉ là một ví dụ về hệ thống `NER`. Bạn có thể có các `architectures` khác nhau.



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
