# **Module 03 - Natural Language Processing with Sequence Models**
## **Week 1: Recurrent Neural Network for Language Modeling**
---
### **Neural Networks for Sentiment Analysis**
---
- Nội dung này tập trung vào cấu trúc và chức năng của mạng nơ-ron, đặc biệt trong bối cảnh phân tích cảm xúc.

- **Cấu trúc Mạng Nơ-ron**
    + Mạng nơ-ron mô phỏng khả năng nhận dạng mẫu của não người và hiệu quả trong nhiều ứng dụng AI, bao gồm xử lý ngôn ngữ tự nhiên (NLP).

    + Một mạng nơ-ron đơn giản bao gồm các tham số đầu vào, lớp ẩn và đơn vị đầu ra, xử lý dữ liệu thông qua lan truyền thuận.

- **Triển khai Phân tích Cảm xúc**
    + Mạng nơ-ron dùng để phân tích cảm xúc sẽ biểu diễn một vectơ của các tweet, sử dụng một lớp nhúng và một lớp ẩn với hàm kích hoạt ReLU.
    + Lớp đầu ra sử dụng hàm softmax để dự đoán cảm xúc (tích cực hoặc tiêu cực) của các tweet.

- **Chuẩn bị Dữ liệu**
    + Các tweet được biểu diễn dưới dạng các vectơ nguyên, trong đó mỗi từ được gán một chỉ mục từ danh sách từ vựng.
    + Đệm được sử dụng để đảm bảo tất cả các vectơ có cùng kích thước, phù hợp với các tweet có độ dài khác nhau.

- Tổng quan này đặt nền tảng cho việc triển khai mạng nơ-ron để phân loại cảm xúc trong các tweet phức tạp, điều này sẽ được tìm hiểu thêm trong video tiếp theo.

> Trước đây trong khóa học, bạn đã thực hiện `sentiment analysis` (phân tích cảm xúc) với `logistic regression` và `naive Bayes`. Những mô hình đó theo một nghĩa nào đó thì `naive` (ngây thơ) hơn, và không có khả năng nắm bắt `sentiment` từ một `tweet` như: "I am not happy" hoặc "If only it was a good day". Khi sử dụng một `neural network` để dự đoán `sentiment` của một câu, bạn có thể sử dụng cách sau đây. Lưu ý rằng hình ảnh bên dưới có ba `outputs`, trong trường hợp này bạn có thể muốn dự đoán "positive", "neutral", hoặc "negative".

![01_Neural_Networks_for_Sentiment_Analysis](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/01_Neural_Networks_for_Sentiment_Analysis.png)

> Lưu ý rằng `network` ở trên có ba `layers`. Để đi từ `layer` này sang `layer` khác bạn có thể sử dụng một `matrix` $W$ để `propagate` tới `layer` tiếp theo. Do đó, chúng ta gọi khái niệm đi từ `input` cho đến `final layer` này là `forward propagation`. Để biểu diễn một `tweet`, bạn có thể sử dụng cách sau:

![02_Neural_Networks_for_Sentiment_Analysis](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/02_Neural_Networks_for_Sentiment_Analysis.png)

> Lưu ý rằng, chúng ta thêm các số không (zeros) để `padding` nhằm khớp với kích thước của `tweet` dài nhất.

> Một `neural network` trong thiết lập mà bạn có thể thấy ở trên chỉ có thể `process` một `tweet` như vậy tại một thời điểm. Để làm cho việc `training` hiệu quả hơn (nhanh hơn), bạn muốn `process` nhiều `tweets` song song (`in parallel`). Bạn đạt được điều này bằng cách đặt nhiều `tweets` cùng nhau vào một `matrix` và sau đó truyền `matrix` này (thay vì các `tweets` riêng lẻ) qua `neural network`. Sau đó `neural network` có thể thực hiện các `computations` của nó trên tất cả các `tweets` cùng một lúc.

---
### **Dense Layers and ReLU**
---
- Nội dung tập trung vào hai `layers` thiết yếu thường được sử dụng trong `neural networks`: `dense layer` và `ReLU layer`.

- **Dense Layer**
    + `Dense layer` tạo điều kiện cho các kết nối giữa các `layers` trong một `neural network` bằng cách thực hiện một `dot product` giữa `weights` và `activations` từ `layer` trước đó.
    + Nó tính toán tất cả các `dot products` đồng thời bằng cách sử dụng một `weight matrix` có thể được học trong quá trình `training`.

- **ReLU Layer**
    + `ReLU` (Rectified Linear Unit) `layer` áp dụng một `non-linear function` lên `output` của `dense layer`, ánh xạ các giá trị âm về không (zero).
    + Nó giữ lại các giá trị dương một cách hiệu quả trong khi chuyển đổi các giá trị âm thành không, điều này giúp ổn định `performance` của `network`.

- Tổng quan này làm nổi bật các thành phần cơ bản của `neural networks` đóng góp vào chức năng của chúng.

- Lưu ý rằng `Dense` và `ReLU` được mô tả là hai `layers` khác nhau ở đây. Trong `TensorFlow`, bạn có thể sử dụng mỗi cái như một `layer` riêng biệt như thế này: 

```
tf.keras.layers.Dense(32)` và `tf.keras.layers.ReLU()
```    

- Trong đó dòng `code` đầu tiên định nghĩa một `Dense layer` với 32 `units` và dòng thứ hai định nghĩa một `ReLU layer`.
- Tuy nhiên, thông thường bạn sẽ muốn `ReLU layer` ngay sau `dense layer`, vì vậy bạn có thể thực hiện điều này theo một cách nhanh hơn, bằng cách chỉ sử dụng một `dense layer`, nhưng truyền `ReLU activation` như một `parameter`:

```
tf.keras.layers.Dense(32, activation='relu')
```
> `Dense layer` là sự tính toán của `inner product` (tích trong) giữa một tập hợp các `trainable weights` (`weight matrix`) và một `input vector`. `Visualization` (trực quan hóa) của `dense layer` có thể được nhìn thấy trong hình ảnh bên dưới.

![03_Dense_Layers_and_ReLU](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/03_Dense_Layers_and_ReLU.png)

> Hộp màu cam trong hình ảnh phía trên hiển thị `dense layer`. Một `activation layer` là tập hợp các `nodes` màu xanh dương được hiển thị cùng với hộp màu cam trong hình ảnh bên dưới. Cụ thể, một trong những `activation layers` được sử dụng phổ biến nhất là `rectified linear unit` (`ReLU`).


![04_Dense_Layers_and_ReLU](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/04_Dense_Layers_and_ReLU.png)

> `ReLU(x)` được định nghĩa là `max(0,x)` cho bất kỳ `input` x nào.


---
### **Embedding and Mean Layers**
---

- Nội dung này giới thiệu hai `layers` quan trọng được sử dụng trong `neural networks` cho `natural language processing` (NLP): `embedding layers` và `mean layers`.

- **Embedding Layers**

    + Một `embedding layer` ánh xạ các từ duy nhất từ một `vocabulary` sang một `vector representation` của một `dimension` được chỉ định, cho phép `model` học được các `word representations` hiệu quả.
    + Kích thước của `embedding` là một `hyperparameter`, và `layer` này học một `matrix of weights` giúp cải thiện `performance` cho các tác vụ `NLP` cụ thể.

- **Purpose of Embedding Layers**

    + Mục đích của một `embedding layer` là chuyển đổi `categorical data`, cụ thể là các từ từ một `vocabulary`, thành các `continuous vector representations`. Dưới đây là các chức năng chính mà nó phục vụ:

    + **Mapping Words to Vectors**: Nó lấy một `index` được gán cho mỗi từ và ánh xạ nó sang một `dense vector` có kích thước cố định, cho phép `model` biểu diễn các từ theo cách nắm bắt được ý nghĩa và mối quan hệ của chúng.
    + **Learning Representations**: `Embedding layer` học `representation` tốt nhất cho các từ trong quá trình `training`, tối ưu hóa các `vectors` để cải thiện `performance` trên các tác vụ `NLP` cụ thể, chẳng hạn như `sentiment analysis` hoặc `text classification`.
    + **Reducing Dimensionality**: Bằng cách chuyển đổi `categorical data` nhiều chiều (như `one-hot encoded vectors`) thành các `continuous vectors` ít chiều hơn, nó giúp giảm độ phức tạp của `model`.

- Nhìn chung, `embedding layers` rất quan trọng để cho phép `neural networks` hiểu và xử lý `natural language` một cách hiệu quả.



- **Mean Layers**

    + Một `mean layer` tính toán giá trị trung bình của các `word embeddings`, giúp giảm số lượng `parameters` cần `train` trong khi vẫn duy trì cùng số lượng `features` như `embedding size`.
    + `Layer` này không có các `trainable parameters`, vì nó chỉ đơn giản là tính toán `mean` của các `embeddings`.

- Tóm lại, `embedding layers` giúp ích trong việc học `word representations`, trong khi `mean layers` đơn giản hóa `output` bằng cách tính trung bình các `embeddings`, cả hai đều thiết yếu để xây dựng các `neural networks` hiệu quả trong các tác vụ `NLP`.

- **Some functions of Mean Layers**


    + Chức năng của một `mean layer` là tính toán giá trị trung bình của các `word embeddings` từ một `embedding layer`. Dưới đây là các khía cạnh chính trong chức năng của nó:

    + * **Averaging Embeddings**: Nó lấy một `matrix` các `word embeddings` (ví dụ: từ một `sequence` các từ) và tính toán `mean` cho mỗi `feature` trên toàn bộ các `embeddings`, dẫn đến kết quả là một `vector representation` duy nhất.
    + **Reducing Parameters**: Không giống như các `layers` khác, `mean layer` không có các `trainable parameters`. Nó đơn giản hóa `model` bằng cách giảm số lượng `parameters` cần được `trained`, điều này có thể giúp ngăn chặn `overfitting`.
    + **Maintaining Feature Size**: `Output` của `mean layer` giữ lại cùng số lượng `features` như `embedding size`, giúp dễ dàng đưa vào các `layers` tiếp theo của `neural network`.

- Tóm lại, `mean layer` giúp tóm tắt thông tin từ nhiều `word embeddings` thành một `vector` duy nhất, tạo điều kiện thuận lợi cho việc `processing` các `sequences` văn bản trong khi giữ cho `model` hiệu quả.

- **Nếu bạn không sử dụng một `mean layer` sau một `embedding layer`, những điều sau có thể xảy ra:**

    + **Increased Complexity**: `Output` từ `embedding layer` sẽ là một `matrix` các `word embeddings` cho mỗi từ trong một `sequence`. `Matrix` này có thể trở nên lớn, đặc biệt là đối với các `sequences` dài, dẫn đến sự gia tăng `complexity` trong `model`.
    + **More Parameters to Train**: Nếu không có `mean layer`, các `layers` tiếp theo sẽ cần xử lý toàn bộ `matrix` các `embeddings`, dẫn đến số lượng lớn hơn các `parameters` cần `train`. Điều này có thể làm cho `model` dễ bị `overfitting` hơn, đặc biệt là với `training data` hạn chế.
    + **Difficulty in Handling Variable-Length Inputs**: Nếu các `input sequences` thay đổi về độ dài, việc quản lý `matrix` các `embeddings` mà không tóm tắt chúng có thể làm phức tạp hóa `architecture` của `neural network`, vì nó sẽ cần phải thích ứng với các kích thước `input` khác nhau.
    + **Loss of Contextual Information**: `Mean layer` giúp tóm tắt thông tin từ nhiều `embeddings` thành một `vector` duy nhất. Nếu không có nó, bạn có thể mất khả năng biểu diễn hiệu quả ý nghĩa tổng thể của một `sequence` các từ.

- Tóm lại, việc không sử dụng một `mean layer` có thể dẫn đến một `model` phức tạp hơn với nguy cơ `overfitting` cao hơn và gặp thách thức trong việc xử lý các `variable-length sequences`.

> Sử dụng một `embedding layer`, bạn có thể học các `word embeddings` cho mỗi từ trong `vocabulary` của bạn như sau:

![05_Embedding_and_Mean_Layers](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/05_Embedding_and_Mean_Layers.png)

- `Mean layer` cho phép bạn tính trung bình các `embeddings`. Bạn có thể trực quan hóa nó như sau:

![06_Embedding_and_Mean_Layers](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/06_Embedding_and_Mean_Layers.png)

- `Layer` này không có bất kỳ `trainable parameters` nào.

---
### **Traditional Language models**
---

- Nội dung này tập trung vào các hạn chế của `N-gram language models` trong `natural language processing`.

- **N-gram Language Models**

    + Các `N-gram models` tính toán `probability` của một `sequence` các từ dựa trên N-1 từ trước đó.
    + Ví dụ, trong một `bigram model`, `probability` của một từ phụ thuộc vào từ đứng ngay trước nó (`preceding word`).

- **Limitations of N-gram Models**

    + Các `N-gram models` yêu cầu `memory` và `storage space` đáng kể để tính đến tất cả các `word combinations` có thể xảy ra.
    + Chúng gặp khó khăn trong việc nắm bắt các `long-range dependencies` giữa các từ, khiến chúng trở nên không thực tế đối với các `datasets` lớn hơn.

- **Introduction to Recurrent Neural Networks (RNNs)**

    + `RNNs` được đề xuất như một sự thay thế hiệu quả hơn cho các `N-gram models` đối với các tác vụ như `machine translation`.
    + Chúng có thể xử lý các `sequences` văn bản dài hơn mà không gặp các hạn chế về `memory` tương tự như các `N-gram models`.

- **The main limitations of N-gram models (hạn chế chính của `N-gram models`) include**

    + **Memory Requirements**: `N-gram models` yêu cầu rất nhiều bộ nhớ để lưu trữ các `probabilities` của tất cả các tổ hợp từ có thể, đặc biệt là khi giá trị của N tăng lên.
    + **Long-Range Dependencies**: Chúng gặp khó khăn trong việc nắm bắt các `dependencies` giữa các từ nằm cách xa nhau trong một câu, vì chúng chỉ xem xét một `context` giới hạn ($N-1$ từ trước đó).
    + **Data Sparsity**: Khi kích thước `vocabulary` tăng lên, số lượng các `N-grams` có thể có sẽ tăng theo cấp số nhân, dẫn đến nhiều tổ hợp có thể không xuất hiện trong `training data`, dẫn đến các vấn đề về dữ liệu thưa thớt (`sparse data`).
    + **Inflexibility**: `N-gram models` rất cứng nhắc và không thích ứng tốt với các `contexts` mới hoặc các biến thể trong ngôn ngữ, khiến chúng kém hiệu quả hơn đối với các tác vụ ngôn ngữ phức tạp.

- Những hạn chế này làm cho các `N-gram models` ít thực tế hơn đối với các `datasets` lớn hơn và các tác vụ `natural language processing` phức tạp hơn.


> Các `traditional language models` sử dụng `probabilities` để giúp xác định xem câu nào có nhiều khả năng xảy ra nhất.

![07_Traditional_Language_models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/07_Traditional_Language_models.png)

> Trong ví dụ ở trên, câu thứ hai là câu có nhiều khả năng xảy ra nhất vì nó có `probability` xảy ra cao nhất. Để tính toán các `probabilities`, bạn có thể làm như sau:

![08_Traditional_Language_models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/08_Traditional_Language_models.png)

Các `Large N-grams` nắm bắt các `dependencies` giữa các từ ở xa nhau và cần rất nhiều không gian và `RAM`. Do đó, chúng ta phải dùng đến các loại `alternatives` khác nhau.

---
### **Recurrent Neural Networks**
---

- Nội dung này tập trung vào các ưu điểm của `recurrent neural networks` (`RNNs`) trong `natural language processing`.

- **Understanding RNNs**

    + `RNNs` có thể nắm bắt các `dependencies` trong các `sequences` mà các `n-gram models` truyền thống không thể, cho phép đưa ra các `predictions` tốt hơn trong các `language tasks`.
    + Không giống như các `n-gram models`, `RNNs` `propagate` (lan truyền) thông tin từ đầu câu đến cuối câu, giúp cải thiện việc hiểu `context`.

- **RNN Functionality**

    + `RNNs` `compute` (tính toán) các giá trị cho mỗi từ trong một `sequence`, sử dụng thông tin từ tất cả các từ trước đó để đưa ra `predictions`.
    + Các `weights` giống nhau được áp dụng trên toàn bộ `sequence`, cho phép `RNNs` chia sẻ `parameters` và học tập một cách hiệu quả.

- **Advantages of RNNs**

    + `RNNs` có thể xử lý các `sequences` dài hơn mà không gặp phải tính thiếu thực tế của các `n-gram models` truyền thống.
    + Chúng được thiết kế để đưa ra các `predictions` dựa trên toàn bộ `context` của một câu, dẫn đến việc `language modeling` chính xác hơn.


> Trước đây, chúng ta đã thử sử dụng các `traditional language models`, nhưng hóa ra chúng chiếm rất nhiều không gian và `RAM`. Ví dụ, trong câu bên dưới:

![09_Recurrent_Neural_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/09_Recurrent_Neural_Networks.png)

> Một `N-gram` (`trigram`) sẽ chỉ nhìn vào "did not" và sẽ cố gắng hoàn thành câu từ đó. Kết quả là, `model` sẽ không thể nhìn thấy phần đầu của câu "I called her but she". Có lẽ từ có khả năng nhất là "have" sau "did not". `RNNs` giúp chúng ta giải quyết vấn đề này bằng cách có khả năng theo dõi các `dependencies` nằm cách xa nhau hơn nhiều. Khi `RNN` đi qua một 
`text corpus`, nó thu thập một số thông tin như sau:

![10_Recurrent_Neural_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/10_Recurrent_Neural_Networks.png)

> Lưu ý rằng khi bạn đưa thêm thông tin vào `model`, sự lưu giữ của từ trước đó trở nên yếu hơn, nhưng nó vẫn ở đó. Hãy nhìn vào hình chữ nhật màu cam ở trên và xem nó trở nên nhỏ hơn như thế nào khi bạn đi qua văn bản. Điều này cho thấy rằng `model` của bạn có khả năng nắm bắt các `dependencies` và nhớ một từ trước đó mặc dù nó nằm ở đầu câu hoặc đoạn văn. Một ưu điểm khác của `RNNs` là rất nhiều tính toán chia sẻ các `parameters`.


---
### **Applications of RNNs**
---

Nội dung tập trung vào các loại `Recurrent Neural Network` (`RNN`) `architectures` (kiến trúc) khác nhau và các ứng dụng của chúng trong các tác vụ `AI` khác nhau.

**Types of AI Tasks** (Các loại tác vụ AI)

* **One to One**: Liên quan đến một `input` duy nhất và một `output` duy nhất, chẳng hạn như dự đoán vị trí của một đội dựa trên điểm số. `RNNs` không đặc biệt hữu ích ở đây.
* **One to Many**: Một `RNN` lấy một `input` duy nhất (như một hình ảnh) và tạo ra nhiều `outputs` (như một `caption` mô tả hình ảnh).

**Different RNN Architectures** (Các kiến trúc RNN khác nhau)

* **Many to One**: Được sử dụng trong các tác vụ như `sentiment analysis` (phân tích cảm xúc), trong đó một `sequence` (chuỗi) các từ được đưa vào, và `model` xuất ra một `sentiment` duy nhất (tích cực hoặc tiêu cực).
* **Many to Many**: Liên quan đến nhiều `inputs` và `outputs`, chẳng hạn như `machine translation` (dịch máy), trong đó một `sequence` bằng ngôn ngữ này được dịch sang ngôn ngữ khác. `Architecture` `encoder-decoder` thường được sử dụng ở đây.



`RNNs` là những công cụ linh hoạt trong `Natural Language Processing` (`NLP`) cho các tác vụ như `machine translation` và `caption generation`, thích ứng với các tình huống khác nhau dựa trên yêu cầu của tác vụ.

`RNNs` có thể được sử dụng trong nhiều tác vụ khác nhau, từ `machine translation` đến `caption generation`. Có nhiều cách để triển khai một `RNN model`:

- `One to One`: cho một vài điểm số của một giải vô địch, bạn có thể dự đoán người chiến thắng.
- `One to Many`: cho một hình ảnh, bạn có thể dự đoán `caption` sẽ là gì.
- `Many to One`: cho một `tweet`, bạn có thể dự đoán `sentiment` của `tweet` đó.
- `Many to Many`: cho một câu tiếng Anh, bạn có thể dịch nó sang câu tương đương tiếng Đức.

---
### **Math in Simple RNNs**
---

Nội dung tập trung vào các nguyên tắc cơ bản của `Recurrent Neural Networks` (`RNNs`) và các `computations` của chúng.

**Understanding RNNs**

* `RNNs` được thiết kế để xử lý các `sequences` dữ liệu, cho phép thông tin được `propagated` theo thời gian.
* `Architecture` bao gồm các `inputs` ($X$), `hidden states` ($H$), và `predictions` ($\hat{Y}$) tại mỗi `time step`.

**Mathematical Foundations**

* `Hidden state` tại thời điểm $T$ được tính toán bằng cách sử dụng một `activation function` và liên quan đến `hidden state` trước đó và `input` hiện tại.
* `Prediction` $\hat{Y}$ được suy ra từ `hidden state` hiện tại và các `parameters` bổ sung.

**Computation Process**

* `Cell` đầu tiên của `RNN` lấy `hidden state` trước đó và `input` hiện tại để tính toán `hidden state` hiện tại.
* Quy trình này liên quan đến các `matrix multiplications` và `summations`, theo sau là việc đi qua một `activation function` để tạo ra các `predictions`.

> Tốt nhất là giải thích toán học đằng sau một `simple RNN` với một sơ đồ:

![11_Math_in_Simple_RNNs](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/11_Math_in_Simple_RNNs.png)

> Lưu ý rằng:

$$h^{<t>}=g(W_h[h^{<t-1>},x^{<t>}]+b_h)$$

> Giống như việc nhân $W_{hh}$ với $h$ và $W_{hx}$ với $x$. Nói cách khác, bạn có thể `concatenate` nó như sau:

$$h^{<t>}=g(W_{hh}h^{<t-1>} \oplus W_{hx}x^{<t>} + b_h)$$

> Đối với `prediction` tại mỗi `time step`, bạn có thể sử dụng công thức sau:

$$\hat{y}^{<t>}=g(W_{yh}h^{<t>}+b_y)$$

> Lưu ý rằng cuối cùng bạn sẽ `training` $W_{hh},W_{hx},W_{yh},b_h,b_y$. Đây là một `visualization` của `model`.

![12_Math_in_Simple_RNNs](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/12_Math_in_Simple_RNNs.png)

---
### **Cost Function for RNNs**
---

Nội dung tập trung vào `cost function` cho `Recurrent Neural Networks` (`RNNs`) và cách điều chỉnh `cross-entropy loss` để tính đến nhiều `time steps`.

**Cost Function for RNNs**

* `Sequential model` bao gồm một `input vector`, nhiều `hidden units`, và các `output units` dự đoán các `class probabilities`.
* `Cross-entropy loss` được tính toán bằng cách so sánh các `predicted probabilities` với các `true labels` cho mỗi quan sát.

**Adapting Cross-Entropy Loss**

* `Loss` cho một quan sát đơn lẻ được tính trung bình qua các `time steps`, sử dụng phép tính tổng theo thời gian và chia cho tổng số bước.
* `Average loss` này nắm bắt `individual loss` trong `model` qua tất cả các `time steps`.

**Implementation Notes**

* `Overall cost` của `model` đạt được bằng cách cộng tổng `average cross-entropy loss` trên tất cả các ví dụ trong `dataset`.
* Nội dung tương lai sẽ bao gồm các `RNN architectures` phức tạp hơn và các `implementations` của chúng.

> `Cost function` được sử dụng trong một `RNN` là `cross entropy loss`. Nếu bạn muốn trực quan hóa nó về cơ bản bạn đang tính tổng trên tất cả các `classes` và sau đó nhân $y_j$ với $\log \hat{y}_j$. 

![13_Cost_Function_for_RNNs](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/13_Cost_Function_for_RNNs.png)

> Nếu bạn muốn tính toán `loss` qua nhiều `time steps`, hãy sử dụng công thức sau:

$$J=-\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{K}y_j^{<t>}\log\hat{y}_j^{<t>}$$

> Lưu ý rằng chúng ta chỉ đơn giản là tính tổng trên tất cả các `time steps` và chia cho $T$, để có được `average cost` trong mỗi `time step`. Do đó, chúng ta chỉ đang lấy trung bình theo thời gian.

---
### **Implementation Note**
---

Nội dung tập trung vào việc triển khai `Recurrent Neural Networks` (`RNNs`) sử dụng `scan functions` trong `TensorFlow` để `computation` (tính toán) hiệu quả.

**Scan Functions in RNNs**

* `Scan functions` là các sự trừu tượng hóa (`abstractions`) cho phép `computation` nhanh hơn bằng cách áp dụng một `function` lên một danh sách các phần tử (`list of elements`) một cách tuần tự.
* Trong `TensorFlow`, `scan function` nhận vào một `function` (`fn`) và một danh sách các `inputs` (`elems`), với một `initializer` tùy chọn cho lần `computation` đầu tiên.

**Implementation Process**

* `RNN` khởi tạo `hidden state` và chuẩn bị một danh sách trống cho các `predictions`.
* Đối với mỗi `input` trong danh sách, `function` được gọi với `input` hiện tại và `hidden state` trước đó, tính toán các `predictions` tại mỗi `time step`.

**Importance of Scan Functions**

* Mặc dù nó có vẻ giống như một vòng lặp (`loop`) đơn giản, `scan functions` cho phép các `deep learning frameworks` thực hiện các `parallel computations` (tính toán song song) và tận dụng `GPUs` một cách hiệu quả.
* Sự trừu tượng hóa (`abstraction`) này rất quan trọng để tối ưu hóa `performance` của `RNNs` trong các ứng dụng `deep learning`.

> `Scan function` được xây dựng như sau:

![14_Implementation_Note](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/14_Implementation_Note.png)

> Lưu ý rằng, đó về cơ bản là những gì một `RNN` đang làm. Nó lấy `initializer`, và trả về một danh sách các `outputs` (`ys`), và sử dụng `current value`, để lấy `next y` và `next current value`. Những loại `abstractions` này cho phép `computation` nhanh hơn nhiều.

---
### **Gated Recurrent Units**
---

Nội dung này giới thiệu `Gated Recurrent Units` (`GRUs`) như một `advanced model` để xử lý các `long sequences` trong `natural language processing`, giải quyết các hạn chế của `vanilla RNNs`.

**Understanding GRUs**

* `GRUs` giữ lại thông tin liên quan trong `hidden state` qua các `long sequences`, cho phép đưa ra các `predictions` tốt hơn trong các tác vụ như hoàn thành câu.
* Chúng sử dụng `relevance and update gates` để xác định thông tin nào cần giữ lại hoặc cập nhật từ các `previous hidden states`.

**Comparison with Vanilla RNNs**

* Không giống như `vanilla RNNs`, vốn có thể gặp phải `vanishing gradient problem`, `GRUs` thực hiện các `additional computations` để quản lý luồng thông tin một cách hiệu quả.
* `GRUs` tính toán nhiều `operations` hơn, điều này có thể dẫn đến thời gian xử lý lâu hơn nhưng tăng cường khả năng của `model` trong việc học và lưu giữ thông tin quan trọng.

**Applications and Future Learning**

* `GRUs` đặc biệt hữu ích cho các tác vụ `NLP` khác nhau và đóng vai trò là phiên bản đơn giản hóa của `LSTMs`, sẽ được khám phá sau trong khóa học.
* Chủ đề tiếp theo sẽ bao gồm `bidirectional and deep RNNs`, mở rộng các khái niệm được giới thiệu với `GRUs`.

> `Gated recurrent units` rất giống với `vanilla RNNs`, ngoại trừ việc chúng có một "relevance" và "update" `gate` cho phép `model` cập nhật và lấy thông tin liên quan. Cá nhân tôi thấy dễ hiểu hơn bằng cách nhìn vào các công thức:

![15_Gated_Recurrent_Units](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W1/15_Gated_Recurrent_Units.png)



> Ở bên trái, bạn có sơ đồ và các phương trình cho một `simple RNN`. Ở bên phải, chúng tôi giải thích về `GRU`. Lưu ý rằng chúng ta thêm 3 `layers` trước khi tính toán $h$ và $y$.

$$\begin{aligned} \Gamma_{u} &=\sigma\left(W_{u}\left[h^{<t_{0}>}, x^{<t_{1}>}\right]+b_{u}\right) \\ \Gamma_{r} &=\sigma\left(W_{r}\left[h^{<t_{0}>}, x^{<t_{1}>}\right]+b_{r}\right) \\ h^{\prime<t_{1}>}=& \tanh \left(W_{h}\left[\Gamma_{r} * h^{<t_{0}>}, x^{<t_{1}>}\right]+b_{h}\right) \end{aligned}$$

> `Gate` đầu tiên $\Gamma_u$ cho phép bạn quyết định mức độ bạn muốn `update` các `weights`. `Gate` thứ hai $\Gamma_r$, giúp bạn tìm ra một `relevance score`. Bạn có thể tính toán $h$ mới bằng cách sử dụng `relevance gate`. Cuối cùng bạn có thể tính toán $h$, sử dụng `update gate`. `GRUs` “quyết định” cách `update` `hidden state`. `GRUs` giúp bảo tồn thông tin quan trọng.




---
### **Deep and Bi-directional RNNs**
---
