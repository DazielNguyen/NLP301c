# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 2: Sentiment Analysis with Naive Bayes
### Probability and Bayes' Rule

> Video này cung cấp cái nhìn tổng quan về **Quy tắc Bayes** (Bayes' rule) và **xác suất** (probability).

![01_Probability_and_Bayes'_Rule](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/01_Probability_and_Bayes'_Rule.png)

- Xác suất là cơ bản cho nhiều ứng dụng trong **NLP**, ví dụ như phân loại tweet (tích cực hay tiêu cực).
    1. Xem xét **xác suất** và **xác suất có điều kiện** (conditional probability).
    2. Rút ra **Quy tắc Bayes** từ xác suất có điều kiện.
- Quy tắc Bayes được áp dụng rộng rãi (y học, giáo dục, **NLP**).
- Bạn sẽ dùng nó để thực hiện **phân tích tình cảm** (sentiment analysis) (nhiệm vụ tuần này).
- Trong khóa học tiếp theo, bạn cũng sẽ dùng nó để **tự động sửa** (auto-correct).
> Ví dụ: Một kho (corpus) lớn các tweet được phân loại là **tích cực** hoặc **tiêu cực**, nhưng **không phải cả hai**. Từ "happy" (hạnh phúc) có thể xuất hiện ở cả hai loại.

![02_Probability_and_Bayes'_Rule_Example](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/02_Probability_and_Bayes'_Rule_Example.png)

- Một cách nghĩ về xác suất là đếm **tần suất** (frequencies).
- **Sự kiện A** (Event A): Tweet được gắn nhãn tích cực.
- **Xác suất của A** ($P(A)$) = (Số tweet tích cực) / (Tổng số tweet).
- Ví dụ: $13/20 = 0.65$ (hoặc 65%).
- **Xác suất bổ sung** (complementary probability): $P(\text{tiêu cực}) = 1 - P(\text{tích cực})$. (Điều này đúng vì các tweet chỉ được phân loại là tích cực hoặc tiêu cực, không phải cả hai).
- **Sự kiện B** (Event B): Tweet chứa từ "happy" (hạnh phúc). Ví dụ: $N_{\text{happy}} = 4$.
- **Giao điểm** (intersection): Các tweet được gắn nhãn tích cực VÀ chứa từ "happy".
- Xác suất của giao điểm = (Diện tích giao điểm) / (Diện tích toàn bộ corpus).
- Ví dụ: 3 tweet (tích cực VÀ "happy") trên tổng số 20 tweet $\rightarrow$ Xác suất = $3/20 = 0.15$.
- Video tiếp theo sẽ nói về **Naive Bayes**.
---
### Bayes' Rule
- Video này xem xét **xác suất có điều kiện** (conditional probability) để hiểu **quy tắc Bayes** (Bayes' rule).
> Ví dụ về xác suất có điều kiện: Đoán thời tiết dễ hơn nếu biết điều kiện (ví dụ: ở California vào mùa đông).

![03_Bayes'_Rule_Example](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/03_Bayes'_Rule_Example.png)

> **CONDITIONNAL PROBABILITIES** giúp chúng ta **GIẢM SAMPLE SEARCH SPACE**
- Để rút ra quy tắc Bayes, chúng ta bắt đầu bằng cách xem xét các tweet chỉ chứa từ "happy" (hạnh phúc) (vòng tròn màu xanh lam), thay vì toàn bộ tài liệu.
- **Xác suất một tweet là tích cực, *do* nó chứa từ "happy"** $P(\text{tích cực} | \text{"happy"})$ = (Số tweet tích cực VÀ chứa "happy") / (Tổng số tweet chứa "happy").
- Ví dụ: 75% khả năng tweet là tích cực nếu nó chứa từ "happy".
- Tương tự (vùng màu tím), **xác suất một tweet tích cực chứa từ "happy"** $P(\text{"happy"} | \text{tích cực})$ = (Số tweet tích cực VÀ chứa "happy") / (Tổng số tweet tích cực).
 
- Ví dụ: 3 trên 13 (0.231).
- **Xác suất có điều kiện** $P(B|A)$ là xác suất của kết quả B, biết rằng sự kiện A đã xảy ra.

![04_Bayes'_Rule_Equations](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/04_Bayes'_Rule_Equations.png)

- Sử dụng biểu đồ Venn, phương trình là:
    $P(\text{tích cực} | \text{"happy"}) = P(\text{giao điểm tích cực và "happy"}) / P(\text{"happy"})$
- Bạn có thể viết một phương trình tương tự bằng cách hoán đổi các điều kiện:
    $P(\text{"happy"} | \text{tích cực}) = P(\text{giao điểm tích cực và "happy"}) / P(\text{tích cực})$
- **Giao điểm** (intersection) là như nhau trong cả hai phương trình. Bằng cách "thao tác đại số" (algebraic manipulation), bạn có thể kết hợp chúng.
- Đây là biểu hiện của **Quy tắc Bayes**.
- **Công thức chung:** $P(x | y) = P(y | x) \times \frac{P(x)}{P(y)}$ (xác suất của x cho y bằng xác suất của y cho x nhân tỷ lệ xác suất của x trên xác suất của y).
- **Điểm rút ra chính:** Quy tắc Bayes dựa trên công thức toán học của xác suất có điều kiện. Nó cho phép bạn tính $P(x | y)$ nếu bạn biết $P(y | x)$ và tỷ lệ $P(x) / P(y)$.
- **CONDITIONAL PROBABILITIES** "LÀ" **BAYES' RULE**
- Video tiếp theo sẽ áp dụng quy tắc Bayes cho một mô hình gọi là **Naive Bayes** (Bayes ngây thơ) để xây dựng **bộ phân loại phân tích tâm lý** (sentiment analysis classifier).
---
### Naive Bayes Introduction

- Tuần này, bạn sẽ học cách phân loại tweet bằng **Naive Bayes** (Bayes ngây thơ), thay vì dùng **hồi quy logistic** (logistic regression) như tuần trước.
- Naive Bayes là một cơ sở (baseline) "nhanh chóng và bẩn thỉu" (quick and dirty) cho các nhiệm vụ **phân loại văn bản** (text classification).
- Đây là một ví dụ về **học máy có giám sát** (supervised machine learning).
- Nó được gọi là **"ngây thơ" (naive)** vì đưa ra giả định rằng các **tính năng (features)** (từ) là **độc lập (independent)** với nhau, điều mà trong thực tế hiếm khi xảy ra nhưng vẫn hoạt động tốt.


> Quy trình của Naive Bayes

![05_Naive_Bayes_Introduction](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/05_Naive_Bayes_Introduction.png)


1.  **Chuẩn bị dữ liệu**:
    - Bắt đầu với hai **corpora** (kho dữ liệu): một cho tweet tích cực và một cho tweet tiêu cực.
    - Trích xuất **từ vựng** (vocabulary) (tất cả các từ khác nhau) và **số lượng (counts)** của chúng trong mỗi kho (tích cực và tiêu cực).
2.  **Tính Tổng số từ**:
    - Đếm tổng số từ trong kho tích cực (ví dụ: 13) và tổng số từ trong kho âm (ví dụ: 12).
    - Đây là bước mới quan trọng cho Naive Bayes, cho phép tính **xác suất có điều kiện** (conditional probability).

> Bảng xác suất có điều kiện

![06_Naive_Bayes_Introduction](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/06_Naive_Bayes_Introduction.png)

3.  **Tính Bảng Xác suất Có điều kiện**:
    - Tính $P(\text{từ} | \text{lớp})$ bằng cách: (tần suất của từ trong lớp) / (tổng số từ trong lớp đó).
    - Ví dụ: $P(\text{"I"} | \text{positive}) = 3/13 \approx 0.24$. $P(\text{"I"} | \text{negative}) = 3/12 = 0.25$.
    - Lưu các giá trị này vào một bảng xác suất có điều kiện.
    - Tổng tất cả các xác suất cho mỗi lớp sẽ bằng 1.
4.  **Phân tích Bảng**:
    - Các từ có xác suất gần giống nhau ở cả hai lớp (ví dụ: "I", "am", "learning", "NLP") là các **từ trung lập (neutral words)** và không đóng góp nhiều vào tình cảm.
    - Các từ có **sự khác biệt đáng kể** (ví dụ: "happy", "sad", "not") là **"từ quyền lực" (powerful words)** giúp xác định tình cảm.
5.  **Vấn đề và Giải pháp (Sơ lược)**:
    * Nếu một từ (ví dụ: "because") chỉ xuất hiện trong kho tích cực, xác suất của nó ở kho âm sẽ là 0.
    * Điều này gây ra vấn đề tính toán. Để tránh điều này, bạn cần **làm mịn (smoothing)**.


#### Áp dụng (Suy luận)

![07_Naive_Bayes_Introduction](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/07_Naive_Bayes_Introduction.png)

- Giả sử có tweet mới: “Hôm nay tôi hạnh phúc, tôi đang học hỏi.”
- Sử dụng **quy tắc điều kiện suy luận Naive Bayes (Naive Bayes inference conditional rule)** để phân loại nhị phân.
- Quy tắc này lấy **tích (product)** của tỷ lệ (xác suất dương / xác suất âm) của mỗi từ trong tweet:
    $\prod \frac{P(\text{word}_i | \text{positive})}{P(\text{word}_i | \text{negative})}$
- **Ví dụ tính toán**:
    + "I": 0.2 / 0.2
    + "am": 0.2 / 0.2
    + "happy": 0.14 / 0.10
    + "Today": Không có trong từ vựng, vì vậy **không bao gồm (not include)** trong điểm số.
    + "I" (lần 2): 0.2 / 0.2
    + "I'm" (lần 2): 0.2 / 0.2
    + "learning": 0.10 / 0.10
- Các từ trung lập (tỷ lệ = 1) sẽ tự **hủy bỏ (cancel out)**.
- Kết quả cuối cùng là $0.14 / 0.10 = 1.4$.
- Vì giá trị này **cao hơn một** (1.4 > 1), tweet được kết luận là tích cực.
- Video tiếp theo sẽ xem xét các vấn đề và **đơn giản hóa các tính toán (simplify calculations)**.

---
### Laplacian Smoothing
> Công thức tính Laplacian Smoothing

![08_Laplacian_smoothing](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/08_Laplacian_smoothing.png)

- **Vấn đề**: Khi tính xác suất, nếu hai từ không bao giờ xuất hiện cạnh nhau trong tập tài liệu đào tạo, bạn sẽ nhận được **xác suất bằng 0**. Điều này có thể khiến xác suất của toàn bộ chuỗi bằng 0.
- **Giải pháp**: **Làm mịn Laplacian** (Laplacian smoothing), một kỹ thuật để tránh xác suất bằng không.
- **Công thức gốc**: 
    + Xác suất có điều kiện $P(\text{từ} | \text{lớp})$ được tính bằng:
    + $\text{Tần số}(\text{từ}_i, \text{lớp}) / N_{\text{lớp}}$ (Số từ trong lớp đó).
- **Công thức làm mịn (Smoothing)**:
    1.  **Tử số**: Thêm 1 vào tần số:
        $\text{Tần số}(\text{từ}_i, \text{lớp}) + 1$.
        (Sự "biến đổi nhỏ" này tránh xác suất bằng 0).
    2.  **Mẫu số**: Để chuẩn hóa (normalize) lại, bạn thêm **V** vào tổng số từ:
        $N_{\text{lớp}} + V$
    3.  **V** là **số lượng từ duy nhất trong từ vựng** (number of unique words in the vocabulary).
- Quá trình này được gọi là **Làm mịn Laplacian** (Laplacian smoothing).
- **Ví dụ tính toán**:
    * Đầu tiên, tính **V**. Trong ví dụ, $V = 8$ (tám từ duy nhất).
    * $P(\text{"I"} | \text{tích cực}) = (3 + 1) / (13 + 8) = 0.19$ (đã làm tròn).
    * $P(\text{"I"} | \text{tiêu cực}) = (3 + 1) / (12 + 8) = 0.2$.
    * Quá trình này được tiếp tục cho phần còn lại của bảng.
- **Kết quả**: Tổng xác suất trong bảng vẫn là 1, và quan trọng là (ví dụ: từ "because" ở script trước) **không còn xác suất bằng 0**.
- **Kết luận**: Bạn đã học về làm mịn Laplacian và hiểu tầm quan trọng của nó để xác suất không bằng không.
- Video tiếp theo sẽ nói về **khả năng ghi nhật ký** (log likelihood).

---
### Log Likelihood

#### Part 1:

- Video này giới thiệu về **khả năng ghi lại** (log likelihoods), chính là **logarit (logarithm)** của xác suất đã tính ở video trước. Chúng thuận tiện hơn khi làm việc.
- Từ ngữ được đơn giản hóa thành ba loại (trung lập, tích cực, tiêu cực) bằng cách **chia các xác suất có điều kiện** (tạo thành một tỷ lệ).
> Cách tính Ratio

![09_Log_Likelihood_Part_1](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/09_Log_Likelihood_Part_1.png)

- **Quy tắc của tỷ lệ ($P(\text{word}|\text{pos}) / P(\text{word}|\text{neg})$):**
    + **Trung lập (Neutral):** Tỷ lệ = 1 (ví dụ: "I", "am", "learning").
    + **Tích cực (Positive):** Tỷ lệ > 1 (ví dụ: "happy" = 1.4). Tỷ lệ càng lớn, từ càng tích cực.
    + **Tiêu cực (Negative):** Tỷ lệ < 1 (ví dụ: "sad", "not" = 0.6). Giá trị càng nhỏ, từ càng âm.
- Tỷ lệ này rất cần thiết cho **phân loại nhị phân (binary classification)** của Naive Bayes.
- **Quy tắc phân loại:**
    + Một tweet là tích cực nếu **sản phẩm (product)** của các tỷ lệ (của các từ trong tweet) > 1.
    + Sản phẩm này được gọi là **khả năng (likelihood)**.
- **Công thức Naive Bayes đầy đủ:** $\text{Score} = \text{(Tỷ lệ trước)} \times \text{(Khả năng)}$.
    + **Tỷ lệ trước (Prior ratio)** là tỷ lệ giữa các tweet tích cực và tiêu cực ($P(\text{pos}) / P(\text{neg})$).
    + Trong ví dụ này, tỷ lệ trước là 1 (vì đây là **bộ dữ liệu cân bằng - balanced dataset**), nhưng nó rất quan trọng đối với các **tập dữ liệu không cân bằng (unbalanced datasets)**.
- **Vấn đề khi triển khai:** Tính **khả năng (likelihood)** yêu cầu nhân nhiều số nhỏ (từ 0 đến 1), dẫn đến nguy cơ bị **thiếu số (underflow)** (số trả về quá nhỏ không thể lưu trữ).

> Giải pháp tính toán trong slide

![10_Log_Likelihood_Part_1](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/10_Log_Likelihood_Part_1.png)


- **Giải pháp (Mẹo toán học):** Sử dụng **nhật ký điểm số (log of the score)** thay vì điểm thô.
    + Sử dụng thuộc tính của logarit: $\log(\text{product}) = \text{sum}(\log)$.
    + $\log(\text{Score}) = \log(\text{Prior}) + \log(\text{Likelihood})$
    + $\log(\text{Likelihood})$ = **Tổng (sum)** của logarit của các tỷ lệ.

> Sử dụng Lambda

![11_Log_Likelihood_Part_1](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/11_Log_Likelihood_Part_1.png)

- **Lambda ($\lambda$):**
    + Định nghĩa: $\lambda = \log(\text{tỷ lệ}) = \log(P(\text{word}|\text{pos}) / P(\text{word}|\text{neg}))$.
    + Ví dụ: $\lambda(\text{"I"}) = \log(1) = 0$ (trung lập). $\lambda(\text{"happy"}) = 2.2$ (> 0, tích cực).
- **Kết luận:** Điểm nhật ký (log score) của tweet có thể được tính bằng cách **tổng các Lambdas (summing the Lambdas)**.
- Việc lấy logarit của tỷ lệ giúp giảm nguy cơ "tràn đầy số" (underflow) khi sản phẩm của các xác suất tiến quá gần đến 0.

#### Part 2: 

- Video này chỉ cách thực hiện **suy luận (inference)** (dự đoán) bằng cách sử dụng từ điển **lambda** ($\lambda$) đã có.
- **Quy trình tính toán:**
    * Bạn tính **khả năng ghi nhật ký (log likelihood)** của tweet bằng cách **tổng các lambda** ($\sum \lambda$) từ mỗi từ có trong tweet.
> **Ví dụ:**

![12_Log_Likelihood_Part_2](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/12_Log_Likelihood_Part_2.png)
- Tweet mẫu
    + "I" $\rightarrow$ 0
    + "am" $\rightarrow$ 0
    + "happy" (hạnh phúc) $\rightarrow$ 2.2
    + "because" (vì) $\rightarrow$ 0
    + "I" $\rightarrow$ 0
    + "learning" (học) $\rightarrow$ 1.1
    + **Tổng (Khả năng ghi nhật ký):** 3.3
- **Ngưỡng quyết định (Decision threshold):**
    + Ngưỡng bây giờ là **0** (thay vì 1, vì $\log(1) = 0$).
    + Giá trị **dương** (trên 0) $\rightarrow$ Tweet tích cực.
    + Giá trị **âm** (dưới 0) $\rightarrow$ Tweet tiêu cực.
- **Kết luận ví dụ:**
    + 3.3 lớn hơn 0, do đó tweet được phân loại là **tích cực**.
    + Điểm số này hoàn toàn dựa trên các từ "happy" và "learning" (từ tích cực), vì các từ trung lập (neutral words) có lambda bằng 0 và không đóng góp vào điểm số.
- **Tóm tắt nhanh:**
    + Bạn dự đoán tình cảm bằng cách tổng hợp các lambda.
    + Điểm số này gọi là **khả năng log (log likelihood)**.
    + Tweet tích cực có khả năng log > 0; tweet tiêu cực có khả năng log < 0.
- **Tiếp theo:** Bạn sẽ học cách **đào tạo (train)** một mô hình Naive Bayes.
---
### Training Naive Bayes

- Video này chỉ cách **đào tạo (train)** bộ phân loại **Naive Bayes** (Bayes ngây thơ).
- "Đào tạo" trong bối cảnh này khác với hồi quy logistic hay học sâu; không có **độ dốc dốc (gradient descent)**, mà chỉ **đếm tần số (counting frequencies)** của từ.

---

> Các bước đào tạo Naive Bayes

1.  **Bước 1: Thu thập dữ liệu**
- Lấy một tập hợp các tweet và chia nó thành hai nhóm: **tích cực (positive)** và **tiêu cực (negative)**.

2.  **Bước 2: Tiền xử lý (Preprocessing)**
- Đây là bước nền tảng, bao gồm 5 bước nhỏ:
    1.  Viết thường (lowercase).
    2.  Loại bỏ dấu câu, URL, và tay cầm (handles).
    3.  Xóa các từ dừng (stop words).
    4.  Tạo gốc (stemming).
    5.  Mã hóa (tokenization) (chia tài liệu thành từ/mã thông báo).
- Trong "thế giới thực", việc thu thập và xử lý văn bản chiếm phần lớn thời gian.

3.  **Bước 3: Tính Tần suất và Xác suất - freq(w, class) - P(w|pos), P(w|neg)**

![13_Training_Naive_Bayes](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/13_Training_Naive_Bayes.png)

- Từ kho (corpus) đã xử lý, tính toán **từ vựng (vocabulary)** (bảng tần số) cho từng từ trong mỗi lớp.
- Tính tổng các từ trong mỗi kho.
- Từ bảng tần số, tính **xác suất có điều kiện (conditional probabilities)** bằng **công thức làm mịn Laplacian (Laplacian smoothing formula)**.
- Lưu ý: **V** (số từ duy nhất) chỉ tính các từ trong bảng (ví dụ: V = 6), không phải tổng số từ trong kho gốc.
- Điều này tạo ra một bảng xác suất chỉ chứa các giá trị lớn hơn 0.

4.  **Bước 4: Tính Lambda**
    - Nhận **điểm Lambda (Lambda score)** cho mỗi từ.
    - Lambda là **nhật ký của tỷ lệ xác suất có điều kiện (log of the ratio of conditional probabilities)**.

5.  **Bước 5: Ước tính Nhật ký Trước (Estimate Log Prior)**
    - Đếm số lượng tweet tích cực và tiêu cực.
    - **Nhật ký trước (Log prior)** là nhật ký của tỷ lệ (số lượng tweet tích cực / số lượng tweet tiêu cực).
    - Trong các bài tập tới (bộ dữ liệu cân bằng - **balanced dataset**), log prior sẽ bằng 0.
    - Thuật ngữ này quan trọng đối với **tập dữ liệu không cân bằng (unbalanced datasets)**.

> Tóm tắt (6 bước logic)

Việc đào tạo một mô hình Naive Bayes có thể được chia thành sáu bước logic:
1.  Chú thích (annotate) một tập dữ liệu (tweet tích cực và tiêu cực).
2.  Xử lý văn bản thô để có kho mã thông báo sạch.
3.  Tính tần số từ điển (dictionary frequencies) cho từng từ trong lớp.
4.  Tính xác suất có điều kiện (dùng làm mịn Laplacian).
5.  Tính hệ số lambda cho mỗi từ.
6.  Ước tính nhật ký trước (log prior) của mô hình.

* **Kết luận:** Bây giờ bạn đã thấy cách xây dựng bảng xác suất cần thiết.
* **Tiếp theo:** Phân loại câu của bạn.
---
### Testing Naive Bayes









---
### Applications of Naive Bayes
---
### Naive Bayes Assumption
---
### Error Analysis


