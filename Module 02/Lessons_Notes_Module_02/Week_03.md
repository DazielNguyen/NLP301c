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
Để tổng quát hóa, `probability` của một `unigram` là:
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

