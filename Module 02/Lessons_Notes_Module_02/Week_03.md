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

Nội dung này tập trung vào `N-gram language models`, cái mà cần thiết để tạo văn bản tự động.

#### Understanding N-grams

- Một **N-gram** là một chuỗi từ (`sequence of words`) trong đó thứ tự quan trọng; nó cũng có thể đề cập đến ký tự hoặc các yếu tố khác.
- **Unigrams** bao gồm các từ đơn duy nhất, **bigrams** là các cặp từ liền kề, và **trigrams** là các bộ ba từ.

#### Calculating Probabilities

- `Probability` của một `unigram` được tính bằng cách chia số lần đếm của từ đó cho tổng số từ trong `corpus`.
- Đối với **bigrams**, `probability` của một từ ($w_i$) đi sau một từ khác ($w_{i-1}$) được xác định bằng số lần đếm của `bigram` chia cho số lần đếm của `unigram` đứng trước:
$$P(w_i | w_{i-1}) = \frac{Count(w_{i-1} w_i)}{Count(w_{i-1})}$$

#### Generalizing to N-grams

- `Probability` của một từ đi sau một chuỗi từ có thể được tổng quát hóa cho bất kỳ `N-gram` nào, sử dụng số lần đếm của `N-gram` đó và `prefix` ($N-1$ grams) của nó:
$$P(w_i | w_{i-(N-1)} \dots w_{i-1}) = \frac{Count(w_{i-(N-1)} \dots w_i)}{Count(w_{i-(N-1)} \dots w_{i-1})}$$
- `Framework` này cho phép tính toán `probabilities` cho các chuỗi có độ dài khác nhau, nâng cao khả năng tạo văn bản.

> Trước khi chúng ta bắt đầu tính toán `probabilities` của các `sequences` nhất định, đầu tiên chúng ta cần định nghĩa `N-gram language model` là gì:

![02_N-grams_and_Probabilities](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/02_N-grams_and_Probabilities.png)

> Bây giờ với những định nghĩa đó, chúng ta có thể gán nhãn cho một câu như sau:

![03_N-grams_and_Probabilities](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/03_N-grams_and_Probabilities.png)

> Bằng ký hiệu khác, bạn có thể viết:

$$w_{1}^{m}=w_{1}w_{2}w_{3}\dots w_{m}$$

$$w_{1}^{3}=w_{1}w_{2}w_{3}$$

$$w_{m-2}^{m}=w_{m-2}w_{m-1}w_{m}$$

> Cho `corpus` sau: "I am happy because I am learning." Kích thước `corpus` $m = 7$.

$$P(\text{I}) = \frac{2}{7}$$

$$P(\text{happy}) = \frac{1}{7}$$

> Để tổng quát hóa, `probability` của một `unigram` là:

$$P(w) = \frac{C(w)}{m}$$

#### Bigram Probability

![04_N-grams_and_Probabilities](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/04_N-grams_and_Probabilities.png)

> `Bigram Probability` được tính như sau (áp dụng công thức tổng quát cho $N=2$):

$$P(w_2 \mid w_1) = \frac{C(w_1 w_2)}{C(w_1)}$$

#### Trigram Probability

> Để tính `probability` của một `trigram`:

$$P(w_3 \mid w_{1}^{2}) = \frac{C(w_{1}^{2}w_{3})}{C(w_{1}^{2})}$$

> Trong đó `count` là:

$$C(w_{1}^{2}w_{3})=C(w_{1}w_{2}w_{3})=C(w_{1}^{3})$$

#### N-gram Probability

> `N-gram Probability` được tổng quát hóa như sau:

$$P(w_{N} \mid w_{1}^{N-1}) = \frac{C(w_{1}^{N-1}w_{N})}{C(w_{1}^{N-1})}$$

> Trong đó `count` là:

$$C(w_{1}^{N-1}w_{N})=C(w_{1}^{N})$$

---
### **Sequence Probabilities**
---

Nội dung này tập trung vào việc mô hình hóa toàn bộ câu bằng cách sử dụng `n-gram probabilities`, điều này cần thiết để tạo văn bản.

#### Understanding Conditional Probability

- `Conditional probability` (xác suất có điều kiện) của một từ phụ thuộc vào (các) từ đứng trước, được thể hiện thông qua **chain rule** (quy tắc chuỗi).
- Công thức cho `conditional probability` có thể được sắp xếp lại để tính **joint probability** (xác suất đồng thời) của các chuỗi từ.

#### Applying the Chain Rule

- `Probability` của một câu ($w_1, \dots, w_m$) được tính bằng tích của `probabilities` của mỗi từ với điều kiện là các từ đứng trước nó:

$$P(w_1, \dots, w_m) = P(w_1) \prod_{i=2}^{m} P(w_i | w_1, \dots, w_{i-1})$$

- Khi các câu dài hơn, khả năng tìm thấy các `sequences` chính xác trong `training corpus` (kho ngữ liệu huấn luyện) giảm xuống.

#### Using the Markov Assumption

- **Markov assumption** (giả định Markov) đơn giản hóa các phép tính bằng cách chỉ xem xét một lịch sử giới hạn của các từ trước đó ($n-1$).
- Dưới giả định Markov, `conditional probability` của $w_i$ được tính:

$$P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-(N-1)}, \dots, w_{i-1})$$

- Đối với `bigrams` (N=2), công thức tập trung vào từ đứng ngay trước nó ($w_{i-1}$), cho phép ước tính `sentence probabilities` dễ dàng hơn.

Bản tóm tắt này gói gọn các khái niệm chính về `n-gram modeling` và ứng dụng của nó trong `natural language processing`.

> Bạn vừa thấy cách tính toán `sequence probabilities`, những thiếu sót của chúng, và cuối cùng là cách xấp xỉ `N-gram probabilities`. Khi làm như vậy, bạn cố gắng xấp xỉ `probability` (xác suất) của một câu. Ví dụ, `probability` của câu sau là gì: "The teacher drinks tea."

> Để tính toán nó, bạn sẽ sử dụng những điều sau:

- $$P(B \mid A)=\frac{P(A,B)}{P(A)}\Longrightarrow P(A,B)=P(A)P(B\mid A)$$

- $$P(A,B,C,D)=P(A)P(B\mid A)P(C\mid A,B)P(D\mid A,B,C)$$

> Để tính `probability` của một `sequence`, bạn có thể tính như sau:

$$\begin{array}{r}P(\text { the teacher drinks tea })= \begin{array}{r}P(\text {the}) P(\text { teacher } \mid \text {the}) P(\text { drinks } \mid \text {the teacher}) P(\text {tea} \mid \text {the teacher drinks })\end{array}\end{array}$$

> Một trong những vấn đề chính khi tính `probabilities` ở trên là `corpus` hiếm khi chứa chính xác các cụm từ giống như những cụm từ bạn đã tính `probabilities`. Do đó, bạn có thể dễ dàng nhận được `probability` bằng 0. **Markov assumption** (giả định Markov) chỉ ra rằng chỉ từ cuối cùng mới quan trọng. Do đó:

- $$\text{Bigram } P(w_n \mid w_{1}^{n-1})\approx P(w_n \mid w_{n-1})$$

- $$\text{N-gram } P(w_n \mid w_{1}^{n-1})\approx P(w_n \mid w_{n-N+1}^{n-1})$$

> Bạn có thể mô hình hóa toàn bộ `sentence` như sau:

- $$P(w_{1}^{n})\approx\prod_{i=1}^{n}P(w_i\mid w_{i-1})$$

- $$P(w_{1}^{n})\approx P(w_1)P(w_2\mid w_1)\dots P(w_n\mid w_{n-1})$$

---
### **Starting and Ending Sentences**
---

Nội dung này tập trung vào việc xử lý điểm bắt đầu và kết thúc của câu khi triển khai `N-gram language models`.

#### Understanding Sentence Boundaries

- Giới thiệu các ký hiệu đặc biệt để biểu thị `start` (bắt đầu) và `end` (kết thúc) của một `sentence`, những ký hiệu này rất quan trọng để tính toán `probabilities` trong `N-gram models`.
- Giải thích cách sửa đổi các câu bằng cách thêm `start tokens` (ví dụ: "S") và `end tokens` (ví dụ: "/S") để tạo điều kiện thuận lợi cho các phép tính `bigram` và `N-gram`.

#### Calculating Probabilities

- Thảo luận về những thách thức trong việc tính toán `probabilities` cho từ đầu tiên trong một `sentence` do thiếu `context`, và cách việc thêm `start tokens` giải quyết vấn đề này.
- Mô tả cách xử lý từ cuối cùng của một `sentence`, nhấn mạnh sự cần thiết của một `end-of-sentence token` để duy trì các phép tính `probability` chính xác.

#### Generalizing to N-grams

- Giải thích rằng đối với **N-grams**, việc thêm $n-1$ `start tokens` ở đầu và một `end token` ở cuối mỗi `sentence` cho phép ước tính `probability` chính xác.
- Minh họa bằng các ví dụ về cách tính `probabilities` cho các độ dài `sentence` khác nhau, đảm bảo tổng `probability` cộng lại bằng 1 trên tất cả các `sentences` có thể có.

> Chúng ta thường bắt đầu và kết thúc một câu với các `tokens` sau tương ứng: `<s>` và `</s>`.

> Khi tính toán `probabilities` bằng cách sử dụng một `unigram`, bạn có thể thêm một `<s>` vào đầu câu. Để tổng quát hóa cho một **N-gram language model**, bạn có thể thêm $N-1$ `start tokens` `<s>`.

> Đối với `end of sentence token` `</s>`, bạn chỉ cần một cái ngay cả khi đó là một `N-gram`.

> Dưới đây là một ví dụ:

![05_Starting_and_Ending_Sentences](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/05_Starting_and_Ending_Sentences.png)

> Hãy đảm bảo rằng bạn biết cách tính toán `probabilities` ở trên!

---
### **The N-gram Language Model**
---

Nội dung này tập trung vào việc tạo và sử dụng một **count matrix** (ma trận đếm) cho `n-grams` trong `natural language processing`.

#### Count Matrix Creation

- **Count matrix** ghi lại sự xuất hiện của `n-grams`, với các `rows` (hàng) đại diện cho các `n-1 grams` độc nhất (`unique`) và các `columns` (cột) đại diện cho các từ độc nhất.
- Đối với `bigrams`, phương pháp **sliding window** (cửa sổ trượt) được sử dụng để đếm sự xuất hiện khi bạn `process` `corpus`.

#### Probability Matrix Transformation

- **Count matrix** được chuyển đổi thành **probability matrix** (ma trận xác suất) bằng cách `normalizing` (chuẩn hóa) mỗi ô dựa trên tổng của `row` tương ứng.
- `Matrix` này cung cấp **conditional probabilities** (xác suất có điều kiện) của `n-grams`, điều cần thiết cho `language modeling`.

#### Language Model Implementation

- `Language model` sử dụng `probability matrix` để ước tính `sentence probabilities` và dự đoán từ tiếp theo trong một `sequence`.
- Nó giải quyết các vấn đề **numerical underflow** (tràn số âm) phát sinh từ việc nhân các `probabilities` nhỏ, thường sử dụng các **logarithmic transformations** để đảm bảo `stability` (ổn định).

> Bạn đã xem qua rất nhiều khái niệm trong video trước. Bạn đã thấy:

- **Count matrix** (Ma trận đếm)
- **Probability matrix** (Ma trận xác suất)
- **Language model** (Mô hình ngôn ngữ)
- **Log probability** (Xác suất log) để tránh tràn số âm (`underflow`)
- **Generative language model** (Mô hình ngôn ngữ sinh)

#### Count Matrix và Chuyển đổi sang Probability Matrix

> Trong **Count matrix**:

- Các **Rows** (Hàng) tương ứng với các `N-1 grams` độc nhất (`unique`) của `corpus`.
- Các **Columns** (Cột) tương ứng với các từ độc nhất của `corpus`.

> Dưới đây là một ví dụ về **count matrix** của một `bigram` (Ngụ ý trong hình trên).

![06_The_N-gram_Language_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/06_The_N-gram_Language_Model.png)


> Để chuyển đổi nó thành **probability matrix**, bạn có thể sử dụng công thức sau:

- $$P(w_{n} \mid w_{n-N+1}^{n-1}) = \frac{C(w_{n-N+1}^{n-1}, w_{n})}{C(w_{n-N+1}^{n-1})}$$

> Trong đó, tổng của mỗi hàng chính là số lần đếm của `prefix` ($N-1$ gram), được sử dụng để chuẩn hóa (`normalize`):

- $$sum(row)=\sum_{w \in V} C(w_{n-N+1}^{n-1}, w) = C(w_{n-N+1}^{n-1})$$

#### Mô hình hóa và Khử Underflow

> Bây giờ với **probability matrix**, bạn có thể tạo ra `language model`. Bạn có thể tính toán `sentence probability` (xác suất câu) và `next word prediction` (dự đoán từ tiếp theo).

> Để tính `probability` của một `sequence` (chuỗi) ($w_1, \dots, w_n$), bạn cần tính:

- $$P(w_{1}^{n}) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-1})$$

> Để tránh **underflow** (tràn số âm), bạn có thể nhân bằng `log` (logarit):

- $$\log(P(w_{1}^{n})) \approx \sum_{i=1}^{n} \log(P(w_i \mid w_{i-1}))$$

> Cuối cùng, đây là bản tóm tắt để tạo ra **generative model**:

![07_The_N-gram_Language_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/07_The_N-gram_Language_Model.png)

---
### **Language Model Evaluation**
---

Nội dung này tập trung vào việc đánh giá `language models` bằng cách sử dụng `perplexity metric`.

#### Understanding Data Splits

- `Text corpus` được chia thành các tập **training**, **validation**, và **test sets** để đảm bảo đánh giá `model` hiệu quả.
- Một tỷ lệ chia phổ biến cho các `datasets` nhỏ hơn là **80%** cho `training`, **10%** cho `validation`, và **10%** cho `testing`.

#### Perplexity as a Metric

- **Perplexity** đo lường độ phức tạp của một văn bản và cho biết mức độ `language model` dự đoán một tập hợp các câu tốt như thế nào.
- **Perplexity scores** thấp hơn cho thấy văn bản xuất hiện tự nhiên và giống con người hơn, trong khi `scores` cao hơn cho thấy sự ngẫu nhiên.

#### Calculating Perplexity

- **Perplexity** ($PP$) được tính bằng cách xác định `probability` của các câu trong `test set` và `normalizing` (chuẩn hóa) nó theo số lượng từ ($N$).
- Công thức tổng quát cho Perplexity của một chuỗi từ $W = w_1, \dots, w_N$ là:
$$PP(W) = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}}$$
- Công thức có thể được đơn giản hóa cho các `bigram models`, và **log perplexity** thường được sử dụng để tính toán dễ dàng hơn.

#### Examples of Language Models

- Các `language models` khác nhau mang lại các `perplexity scores` khác nhau, với các `models` tốt thường đạt từ **20 đến 60**.
- Nội dung minh họa cách các **unigram**, **bigram**, và **trigram models** hoạt động về mặt `perplexity`, cho thấy sự cải thiện về tính **coherence** (mạch lạc) của văn bản với các `models` phức tạp hơn.

> Bây giờ chúng ta sẽ thảo luận về các `train/val/test splits` (chia tập huấn luyện/xác thực/kiểm thử) và `perplexity`.

#### Train/Val/Test splits

- **Corpora (Kho ngữ liệu) nhỏ hơn:**
    + 80% train
    + 10% val
    + 10% test
- **Corpora lớn hơn:**
    + 98% train
    + 1% val
    + 1% test
> Có 2 phương pháp cho chia tập dữ liệu

![08_Language_Model_Evaluation](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/08_Language_Model_Evaluation.png)

#### Perplexity

> `Perplexity` được sử dụng để cho chúng ta biết liệu một tập hợp các câu có vẻ được viết bởi con người hay không, thay vì được tạo ra bởi một chương trình đơn giản chọn từ ngẫu nhiên. Một văn bản được viết bởi con người có nhiều khả năng có `perplexity` thấp hơn, trong khi một văn bản được tạo ra bởi việc chọn từ ngẫu nhiên sẽ có `perplexity` cao hơn.

> Cụ thể, đây là các công thức để tính `perplexity`.

$$PP(W)=P(s_1,s_2,\dots,s_m)^{-\frac{1}{m}}$$

$$PP(W) = \sqrt[m]{ \prod_{i=1}^{m} \prod_{j=1}^{|s_i|} \frac{1}{P\left(w_j^{(i)} \mid w_{j-1}^{(i)}\right)} }$$

> $w_j^{(i)} \to j$ tương ứng với từ thứ $j$ trong câu thứ $i$. Nếu bạn nối tất cả các câu lại, thì $w_i$ là từ thứ $i$ trong `test set`.

$$PP(W) = \sqrt[m]{ \prod_{i=1}^{m} \frac{1}{P(w_i \mid w_{i-1})} }$$

> Để tính **log perplexity**, bạn chuyển từ công thức trên thành:

$$\log PP(W)=-\frac{1}{m}\sum_{i=1}^{m}\log_{2}(P(w_i\mid w_{i-1}))$$

---
### **Out of Vocabulary Words**
---

Nội dung này tập trung vào việc xử lý các từ **out-of-vocabulary** (`OOV`) trong `language models`.

#### Understanding Out-of-Vocabulary Words

- **OOV words** là những từ không có mặt trong `training vocabulary` của `model`, thường gặp trong các `tasks` như `speech recognition`.
- Một **closed vocabulary** (từ vựng đóng) giới hạn `model` trong một tập hợp từ cố định, trong khi **open vocabulary** (từ vựng mở) cho phép các từ mới, chưa từng thấy.

#### Using the UNK Token

- Để quản lý `OOV words`, chúng có thể được thay thế bằng một `special token` (**UNK**), trong quá trình tính toán `probability`.
- `Vocabulary` được định nghĩa dựa trên `word frequency` (tần suất từ), trong đó các từ xuất hiện ít hơn một số lần được chỉ định sẽ được thay thế bằng **UNK**.

#### Building Vocabulary

- `Vocabulary` có thể được tạo bằng cách đặt một **minimum frequency threshold** (ngưỡng tần suất tối thiểu) hoặc một giới hạn kích thước tối đa.
- Sự hiện diện của `UNK tokens` có thể ảnh hưởng đến `perplexity` của `model`, thường làm nó có vẻ hiệu quả hơn, nhưng quá nhiều `UNKs` có thể dẫn đến `outputs` vô nghĩa.

Tóm lại, bài giảng nhấn mạnh tầm quan trọng của việc quản lý hiệu quả `OOV words` để cải thiện `language model performance`.

> Nhiều khi, bạn sẽ phải xử lý các từ không xác định trong `corpus`. Vậy làm thế nào để bạn chọn **vocabulary** (từ vựng) của mình?

#### Định nghĩa và Loại Vocabulary

> **Vocabulary** là một tập hợp các từ độc nhất được hỗ trợ bởi `language model` của bạn.

- Trong một số `tasks` như `speech recognition` hoặc `question answering`, bạn sẽ gặp và tạo ra các từ chỉ từ một tập hợp từ cố định. Do đó, đây là **closed vocabulary** (từ vựng đóng).
- **Open vocabulary** (từ vựng mở) có nghĩa là bạn có thể gặp các từ bên ngoài `vocabulary`, như tên của một thành phố mới trong `training set`.

#### "Công thức" xử lý từ không xác định

> Dưới đây là một "công thức" cho phép bạn xử lý các từ không xác định:

1.  Tạo `vocabulary` $V$.
2.  Thay thế bất kỳ từ nào trong `corpus` và không có trong $V$ bằng **`<UNK>`** (Unknown token).
3.  Đếm `probabilities` với `<UNK>` như với bất kỳ từ nào khác.

![09_Out_of_Vocabulary_Words](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W3/09_Out_of_Vocabulary_Words.png)

> Ví dụ trên cho thấy cách bạn có thể sử dụng **min\_frequency** (tần suất tối thiểu) và thay thế tất cả các từ xuất hiện ít hơn `min_frequency` bằng **UNK**. Sau đó bạn có thể coi **UNK** như một từ thông thường.

#### Tiêu chí để tạo Vocabulary

- **Min word frequency $f$**: Chọn một tần suất từ tối thiểu.
- **Max $|V|$**: Đặt giới hạn kích thước tối đa của `vocabulary`, bao gồm các từ theo tần suất.
- Sử dụng `<UNK>` một cách **tiết kiệm** (Vì việc sử dụng quá nhiều `<UNK>` có thể làm giảm ý nghĩa của `output`).
- **Perplexity**: Chỉ so sánh `LM` (Language Models) có cùng `vocabulary` ($V$) để có kết quả đánh giá công bằng.

---
### **Smoothing**
---



---
### **Week Summary**
---

