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

