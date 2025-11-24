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

Nội dung cung cấp một cái nhìn tổng quan về `transformer model`, một bước tiến quan trọng trong `natural language processing` (NLP) được giới thiệu bởi các nhà nghiên cứu của Google vào năm 2017.

**Transformer Model Overview**

* `Transformer architecture` đã trở thành tiêu chuẩn cho các `large language models` như `BERT`, `T5`, và `GPT-3`, cách mạng hóa `NLP`.
* Nó sử dụng `scale dot-product attention`, vốn hiệu quả trong việc tính toán (`computation`) và bộ nhớ (`memory`), cho phép các `models` lớn hơn và phức tạp hơn.

**Encoder and Decoder Structure**

* `Encoder` bao gồm các `multi-head attention layers` thực hiện `self-attention`, theo sau là các `residual connections` và `feed-forward layers`, được lặp lại nhiều lần.
* `Decoder` tương tự nhưng bao gồm `masked attention` để ngăn chặn luồng thông tin từ các vị trí tương lai, cho phép nó chú ý (`attend`) đến các `outputs` của `encoder`.

**Positional Encoding**

* `Positional encoding` rất quan trọng để duy trì thứ tự của các từ trong các `sequences`, vì `transformers` không sử dụng `recurrent neural networks`.
* Loại mã hóa này có thể được học (`learned`) hoặc cố định (`fixed`) và được thêm vào `word embeddings` để giữ lại thông tin tuần tự (`sequential information`).

**Advantages Over RNNs**

* `Transformers` giải quyết các vấn đề mà `recurrent neural networks` (`RNNs`) gặp phải, chẳng hạn như những khó khăn trong `parallel computing` (tính toán song song) và mất thông tin trong các `sequences` dài.
* Chúng hiệu quả hơn cho việc `training` trên các `large datasets` và có thể xử lý nhiều tác vụ một cách hiệu quả.


---
### **Transformer Applications**
---

Nội dung tập trung vào các ứng dụng và khả năng của `transformer models` trong `Natural Language Processing` (`NLP`).

**Applications of Transformers**

* `Transformers` là các `models` linh hoạt được sử dụng trong nhiều `NLP tasks` khác nhau như `automatic text summarization` (tóm tắt văn bản tự động), `autocompletion` (tự động hoàn thành), `named entity recognition`, `machine translation` (dịch máy), và `question answering` (trả lời câu hỏi).
* Các `transformer models` đáng chú ý bao gồm `GPT-2`, `BERT`, và `T5`, mỗi `model` được thiết kế cho các `tasks` và chức năng cụ thể.

**Overview of T5 Model**

* `T5` (`Text-to-Text Transfer Transformer`) có thể thực hiện nhiều `tasks` như dịch thuật, phân loại (`classification`), và `question answering` bằng cách sử dụng một `model` duy nhất.
* `Model` diễn giải các `input strings` chỉ định `task` và dữ liệu, cho phép nó tạo ra các `outputs` phù hợp cho các `tasks` khác nhau.

**Task Examples with T5**

* Đối với dịch thuật, một `input string` chỉ định sự chuyển đổi ngôn ngữ, trong khi đối với `classification`, nó xác định xem một câu có thể chấp nhận được hay không.
* `T5` cũng có thể xử lý các `regression tasks`, đo lường sự tương đồng giữa các câu, và `summarization` (tóm tắt), cô đọng các văn bản dài thành các phát biểu ngắn gọn.

> Dưới đây là bản tóm tắt ngắn gọn về tất cả các ứng dụng khác nhau mà bạn có thể xây dựng bằng cách sử dụng `transformers`:

![02_Transformer_Applications](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/02_Transformer_Applications.png)

> Sẽ rất tuyệt nếu bạn thực sự có thể chơi đố vui (`trivia`) với một `transformer` tại đây: [https://t5-trivia.glitch.me/](https://t5-trivia.glitch.me/)

> Một lĩnh vực nghiên cứu thú vị khác là việc sử dụng `transfer learning` với `transformers`. Ví dụ, để `train` một `model` dịch tiếng Anh sang tiếng Đức, bạn chỉ cần đặt trước (`prepend`) văn bản "translate English to German" vào `inputs` mà bạn sắp đưa vào `model`. Sau đó, bạn có thể giữ nguyên `model` đó để `detect sentiment` (phát hiện cảm xúc) bằng cách đặt trước một `tag` khác. Hình ảnh sau đây tóm tắt `T5 model` sử dụng khái niệm này:

![03_Transformer_Applications](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/03_Transformer_Applications.png)


> `GPT`, `BERT`, và `T5` là một số `transformer models` mới nhất.

---
### **Scaled and Dot-Product Attention**
---

Nội dung tập trung vào `scale dot-product attention mechanism`, một thành phần chính của `transformer models` trong `natural language processing`.

**Scale Dot-Product Attention**

* Cơ chế này sử dụng `queries`, `keys`, và `values` để `compute` (tính toán) các `context vectors` cho mỗi `query`.
* Độ tương đồng (`similarity`) giữa `queries` và `keys` xác định các `weights` được gán cho `values`, với `SoftMax` đảm bảo tổng `weights` bằng 1.

**Matrix Operations**

* Quá trình này liên quan đến `matrix multiplication` giữa `query` và chuyển vị (`transpose`) của `key matrix`, được `scale` bằng nghịch đảo của căn bậc hai `key dimension`.
* `Weight matrix` thu được được nhân với `value matrix` để tạo ra các `context vectors`.

**Efficiency and Implementation**

* `Scale dot-product attention` hiệu quả do dựa vào `matrix multiplication` và `SoftMax`, làm cho nó phù hợp để triển khai trên `GPU` hoặc `TPU`.
* `Attention mechanism` là nền tảng cho `transformers`, cho phép xử lý hiệu quả các `sequences` trong các `NLP tasks`.

---
### **Masked Self Attention**
---

Nội dung này tập trung vào các loại `attention mechanisms` khác nhau trong `transformer model`, đặc biệt làm nổi bật `masked self-attention`.

**Các loại Attention Mechanism**

* **Encoder-Decoder Attention**: Trong cơ chế này, `queries` đến từ một câu (ví dụ: câu đích trong `decoder`) trong khi `keys` và `values` đến từ câu khác (ví dụ: câu nguồn trong `encoder`), cho phép các từ trong câu đích chú ý (`attend`) đến tất cả các từ trong câu nguồn.
* **Self-Attention**: Ở đây, `queries`, `keys`, và `values` đến từ cùng một câu, cho phép mỗi từ chú ý (`attend`) đến mọi từ khác trong chính câu đó, điều này giúp `model` hiểu `context` của mỗi từ.


**Masked Self-Attention**

* Trong **Masked Self-Attention**, `queries`, `keys`, và `values` cũng đến từ cùng một câu, nhưng `queries` **không thể chú ý (`attend`) đến các vị trí trong tương lai** (`future positions`). Điều này rất quan trọng để đảm bảo rằng các `predictions` (dự đoán) chỉ phụ thuộc vào các `outputs` đã biết (thông tin ở hiện tại và quá khứ).
* Việc triển khai bằng toán học liên quan đến việc thêm một **mask matrix** vào phép tính `softmax`, cái mà sẽ đặt trọng số bằng $0$ cho các vị trí tương lai, ngăn các vị trí này được chú ý.

Tổng thể, video này phác thảo ba loại `attention mechanisms` chính và nhấn mạnh các đặc điểm độc đáo của `masked self-attention`—một thành phần cốt lõi của `decoder` trong `Transformer model`.

---
### **Multi-head Attention**
---

**Multi-Head Attention Overview**

* `Multi-head attention` cho phép `attention mechanism` được áp dụng song song (`in parallel`) cho nhiều tập hợp `query`, `key`, và `value matrices`.
* Mỗi `head` trong `model` sử dụng các biểu diễn khác nhau của các `embeddings` gốc, cho phép `model` học các mối quan hệ đa dạng giữa các từ.

**Process of Multi-Head Attention**

* Các `input matrices` (`queries`, `keys`, `values`) được chuyển đổi thành nhiều `vector spaces` dựa trên số lượng `heads` trong `model`.
* `Scaled dot-product attention mechanism` được áp dụng cho mỗi tập hợp các `matrices` đã được chuyển đổi, dẫn đến các `output matrices` riêng biệt cho mỗi `head`.

**Final Output Generation**

* Các `outputs` từ mỗi `attention head` được `concatenated` (nối) thành một `single matrix`, cái mà sau đó được biến đổi tuyến tính (`linearly transformed`) để tạo ra các `context vectors` cuối cùng.
* Việc lựa chọn kích thước `transformation matrix` phù hợp đảm bảo `computational cost` vẫn hiệu quả, tương tự như `single-head attention`.

> Bài giảng này chuẩn bị cho người học triển khai `multi-head attention` một cách hiệu quả trong việc xây dựng các `transformer models`.

![04_Multi-head_Attention](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/04_Multi-head_Attention.png)


> Trong bài đọc này, bạn sẽ thấy bản tóm tắt về `intuition` (trực giác) đằng sau `multi-head attention` và `scaled dot product attention`.

> Với một từ, bạn lấy `embedding` của nó sau đó bạn nhân nó với `matrix` $W_Q, W_K, W_V$ để có được các `queries`, `keys` và `values` tương ứng. Khi bạn sử dụng `multi-head attention`, mỗi `head` thực hiện cùng một `operation`, nhưng sử dụng các `matrices` riêng của nó và có thể học các mối quan hệ khác nhau giữa các từ so với một `head` khác.

> Dưới đây là hướng dẫn từng bước, đầu tiên bạn nhận được các `matrices` $Q, K, V$:

![05_Multi-head_Attention](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/05_Multi-head_Attention.png)

> Đối với mỗi từ, bạn nhân nó với các `matrices` $W_Q, W_K, W_V$ tương ứng để có được `word embedding` tương ứng. Sau đó, bạn phải tính toán `scores` với các `embedding` đó như sau:

![06_Multi-head_Attention](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/06_Multi-head_Attention.png)

> Lưu ý rằng `computation` trên được thực hiện cho một `head`. Nếu bạn có nhiều `heads`, cụ thể là $n$, thì bạn sẽ có $Z_1, Z_2, \dots, Z_n$. Trong trường hợp đó, bạn chỉ cần `concatenate` (nối) chúng lại và nhân với một `matrix` $W_O$ như sau:

![07_Multi-head_Attention](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/07_Multi-head_Attention.png)


> Trong hầu hết các trường hợp, `dimensionality` của các $Z$ được cấu hình để căn chỉnh với $d_{model}$ (trong đó `head size` được xác định bởi $d_{head}=d_{model}/h$), đảm bảo tính nhất quán với `input dimensions`. Do đó, các `representations` (embeddings) được kết hợp thường trải qua một `final projection` bởi $W_O$ thành một `attention embedding` mà không thay đổi `dimensions`.

![08_Multi-head_Attention](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W1/08_Multi-head_Attention.png)

> Ví dụ, nếu $d_{model}$ là 16, với hai `heads`, việc `concatenate` $Z_1$ và $Z_2$ dẫn đến một `dimension` là 16 (8 + 8). Tương tự, với bốn `heads`, việc `concatenate` $Z_1, Z_2, Z_3,$ và $Z_4$ cũng dẫn đến một `dimension` là 16 (4 + 4 + 4 + 4). Trong ví dụ này, và trong hầu hết các `architectures` phổ biến, đáng chú ý là số lượng `heads` không làm thay đổi `dimensionality` của `concatenated output`. Điều này vẫn đúng ngay cả sau `final projection` với $W_O$, cái mà cũng thường duy trì các `dimensions` nhất quán.


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
