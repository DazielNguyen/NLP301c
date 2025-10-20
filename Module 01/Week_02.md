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

- **Kết luận:** Bây giờ bạn đã thấy cách xây dựng bảng xác suất cần thiết.
- **Tiếp theo:** Phân loại câu của bạn.

---
### Testing Naive Bayes

- Video này nói về việc áp dụng bộ phân loại **Naive Bayes** (Bayes ngây thơ) trên các **ví dụ thử nghiệm (test examples)** thực tế và đề cập đến một số "trường hợp góc đặc biệt" (special corner cases).
- Sau khi **đào tạo (train)** mô hình (có được bảng Lambda và logprior), bước tiếp theo là **kiểm tra (test)** nó.
- Bạn sử dụng mô hình để dự đoán tình cảm của các **tweet chưa được nhìn thấy (unseen tweets)**.

![14_Testing_Naive_Bayes](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/14_Testing_Naive_Bayes.png)

> Quy trình dự đoán (Suy luận) trên Tweet mới

1.  **Ví dụ Tweet mới:** “Tôi đã vượt qua cuộc phỏng vấn NLP.”
2.  **Bước 1: Tiền xử lý (Pre-process)**
    * Văn bản phải được xử lý trước (loại bỏ dấu câu, tạo gốc từ, mã hóa) để tạo ra một **vectơ từ (vector of words)**.
3.  **Bước 2: Tính điểm (Score)**
    * Tra cứu từng từ trong vectơ này trong **bảng Lambda** (bảng khả năng ghi).
    * **Tính tổng** (sum) tất cả các **thuật ngữ Lambda** (Lambda terms) tương ứng cho các từ *được tìm thấy* (ví dụ: "I", "pass", "the", "NLP").
4.  **Bước 3: Xử lý từ không xác định (Trường hợp góc)**
    * Các từ **không tìm thấy** trong bảng Lambda (ví dụ: "phỏng vấn") được coi là **trung lập (neutral)** và **không đóng góp bất cứ điều gì vào điểm số** (tức là giá trị lambda bằng 0).
    * Mô hình chỉ có thể chấm điểm các từ mà nó đã thấy trước đây.
5.  **Bước 4: Thêm Log Prior**
    * Thêm **nhật ký trước (logprior)** vào tổng điểm (để tính đến sự mất cân bằng của các lớp).
6.  **Bước 5: Quyết định**
    * Tổng điểm trong ví dụ là 0.48.
    * Nếu điểm số **lớn hơn 0**, tweet có cảm xúc **tích cực (positive)**. (Nếu nhỏ hơn 0, tweet là âm).

> Quy trình Kiểm tra (Đánh giá) mô hình

- Để kiểm tra hiệu suất, bạn sử dụng một **bộ xác nhận (validation set)** (gồm `X_val` - tweet thô và `Y_val` - tình cảm tương ứng).
- Bạn cần triển khai hàm **độ chính xác (accuracy)**.
- **Các bước tính độ chính xác:**
    1.  Tính **điểm (score)** cho mỗi tweet trong `X_val` (như quy trình dự đoán ở trên).
    2.  Đánh giá xem mỗi điểm có **lớn hơn 0** hay không.
    3.  Thao tác này tạo ra một **vectơ dự đoán** (vector of predictions) chứa các số 0 (âm) và 1 (dương).
    4.  So sánh **vectơ dự đoán** này với các giá trị thực (labels) trong `Y_val`.
    5.  Nếu dự đoán = nhãn thực $\rightarrow$ 1 (chính xác). Nếu không $\rightarrow$ 0 (không chính xác).
    6.  **Độ chính xác** = (Tổng của vectơ so sánh này) / (Tổng số ví dụ trong bộ xác thực).
    * (Quy trình này giống như bạn đã làm cho hồi quy logistic).

- **Tóm lại:** Bạn kiểm tra mô hình bằng cách dự đoán trên bộ xác thực, so sánh dự đoán với nhãn thực để có được **tỷ lệ phần trăm tweet được dự đoán chính xác**.
- **Tiếp theo:** Bạn sẽ áp dụng Naive Bayes trong bài tập mã hóa và xem các ứng dụng khác của nó.

---

### Applications of Naive Bayes

- Bạn đã sử dụng **Naive Bayes** (Bayes ngây thơ) để phân loại tweet, nhưng nó cũng có thể được dùng để **xác định ai là tác giả** (identify author) của một văn bản.
- Khi dùng Naive Bayes, bạn đang ước tính xác suất cho mỗi lớp. Công thức là **tỷ lệ** (ratio) giữa hai xác suất này (tích của **tiền trước - prior** và **khả năng xảy ra - likelihood**).
- Tỷ lệ này có thể được sử dụng cho nhiều ứng dụng hơn là **phân tích tình cảm** (sentiment analysis).

> Các ứng dụng khác của Naive Bayes

1.  **Xác định tác giả (Author Identification)**
    + Nếu bạn có hai **hạ sĩ lớn (corpora)** (ví dụ: một của Shakespeare, một của Hemingway), bạn có thể đào tạo mô hình để nhận ra ai đã viết một tài liệu mới.
    + Bạn sẽ **tính toán Lambda** ($\lambda$) cho mỗi từ để dự đoán khả năng.
    + Phương pháp này cho phép **Xác định Danh tính tác giả** (Author Identity).

2.  **Lọc thư rác (Spam Filtering)**
    + Sử dụng thông tin từ người gửi, chủ đề và nội dung để quyết định xem email có phải là **spam** hay không.

3.  **Truy xuất thông tin (Information Retrieval)**
    + Đây là một trong những ứng dụng sớm nhất.
    + Nó lọc giữa các tài liệu **liên quan (relevant)** và **không liên quan (irrelevant)** trong cơ sở dữ liệu cho một **truy vấn (query)**.
    + Trong trường hợp này, bạn **tính toán khả năng (calculate the likelihood)** của các tài liệu được cung cấp truy vấn (bạn không thể biết trước xác suất "trước").
    + Bạn có thể lưu trữ (sắp xếp) tài liệu dựa trên khả năng và chọn **kết quả m đầu tiên** (top m results) hoặc những kết quả trên một ngưỡng nhất định.

4.  **Phân biệt từ (Word Disambiguation)**
    + Làm rõ nghĩa của từ dựa theo ngữ cảnh.
    + Ví dụ: từ "bank" (ngân hàng) có thể là **bờ sông (river bank)** hoặc **tổ chức tài chính (financial institution)**.
    + Để phân biệt, bạn **tính điểm (calculate the score)** của tài liệu, *vì nó đề cập đến từng ý nghĩa có thể*.
    + Nếu văn bản đề cập đến khái niệm "sông" (river) thay vì "tiền" (money), điểm số sẽ lớn hơn một.

> Tóm lại

- **Quy tắc Bayes (Bayes' rule)** và **tính gần đúng ngây thơ (naive approximation)** của nó có nhiều ứng dụng (phân tích tình cảm, xác định tác giả, truy xuất thông tin, phân biệt từ).
- Nó phổ biến vì **tương đối đơn giản** (relatively simple) để đào tạo, sử dụng và diễn giải.
- **Tiếp theo:** Bạn sẽ học về những giả định (assumptions) làm nền tảng cho phương pháp Naive Bayes.

---
### Naive Bayes Assumption

- Video này nói về các **giả định (assumptions)** làm nền tảng cho phương pháp **Naive Bayes (bayes ngây thơ)**.
- Giả định chính là **sự độc lập của các từ (independence of words)** trong một câu.
- Phương pháp này được gọi là **"ngây thơ" (naive)** vì những giả định nó đưa ra về dữ liệu:
    1. **Sự độc lập giữa các yếu tố dự đoán (independence between predictors)** (tính năng) liên quan đến mỗi lớp.
    2. (Giả định thứ hai liên quan đến bộ xác thực, nhưng kịch bản tập trung vào vấn đề phân phối dữ liệu).

![15_Naive_Bayes_Assumptions_01](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/15_Naive_Bayes_Assumptions_01.png)


> Vấn đề 1: Giả định về sự độc lập

- **Ví dụ:** Câu "Trời nắng và nóng ở sa mạc Sahara" (It is sunny and hot in the Sahara desert). Naive Bayes giả định các từ này **độc lập** với nhau.
- **Thực tế:** Điều này thường không xảy ra. "Nắng" (sunny) và "nóng" (hot) thường xuất hiện cùng nhau và có liên quan đến "sa mạc" (desert).
- **Hệ quả:** Giả định ngây thơ này có thể khiến bạn **đánh giá thấp (underestimate)** hoặc **đánh giá quá cao (overestimate)** xác suất có điều kiện của từng từ.
- **Ví dụ (Hạn chế):** Nếu hoàn thành câu "it always cold and snow is white in...", Naive Bayes có thể gán xác suất bằng nhau cho các mùa (xuân, hè, thu, đông), mặc dù "mùa đông" (winter) là có khả năng nhất theo ngữ cảnh.
- (Các khóa học tiếp theo sẽ giới thiệu các phương pháp tinh vi hơn để giải quyết vấn đề này).

![16_Naive_Bayes_Assumptions_02](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/16_Naive_Bayes_Assumptions_02.png)

> Vấn đề 2: Phân phối dữ liệu (Data Distribution)

- Naive Bayes dựa vào sự **phân phối của các bộ dữ liệu đào tạo (distribution of the training datasets)**.
- Một tập dữ liệu tốt nên có tỷ lệ (tích cực/tiêu cực) giống như một mẫu ngẫu nhiên.
= **Thực tế:** Hầu hết các **cơ thể có chú thích (annotated corpora)** có sẵn đều được **cân bằng nhân tạo (artificially balanced)** (giống như tập dữ liệu bạn dùng).
- Trong **"tweet thực sự" (real tweet)** (thực tế), các tweet tích cực có xu hướng xảy ra thường xuyên hơn tweet tiêu cực.
    + Lý do: Các tweet tiêu cực (ví dụ: **từ vựng không phù hợp hoặc xúc phạm - inappropriate or offensive vocabulary**) có thể bị nền tảng cấm hoặc người dùng tắt tiếng.
- **Hệ quả:** Nếu dữ liệu đào tạo (ví dụ: cân bằng) không phản ánh thực tế (ví dụ: nhiều tích cực hơn), mô hình có thể trở nên rất **lạc quan (optimistic)** hoặc **bi quan (pessimistic)**.

> Tóm tắt

- Giả định về sự độc lập (ngây thơ) rất khó đảm bảo, nhưng mô hình vẫn hoạt động khá tốt trong một số tình huống.
- Đối với các bài tập trong mô-đun này, tần suất tương đối (tích cực/tiêu cực) trong bộ dữ liệu đào tạo cần được **cân bằng** để mang lại kết quả chính xác.
- **Video tiếp theo:** Sẽ chỉ cho bạn biết phải làm gì khi mô hình hoạt động không tốt trong một số trường hợp.

---
### Error Analysis

Video này chỉ cho bạn cách **phân tích lỗi** (analyze errors) khi một phương pháp NLP (như Naive Bayes) **phân loại sai** (misclassify) một câu.

> Các Nguồn Lỗi Tiềm Ẩn

Có ba nguyên nhân chính gây ra lỗi dự đoán:
1.  **Ý nghĩa ngữ nghĩa** (Semantic meaning) bị mất trong **bước xử lý trước** (preprocessing).
2.  **Thứ tự từ** (Word order) ảnh hưởng đến ý nghĩa của câu.
3.  Những **điều kỳ quặc về ngôn ngữ** (language quirks) mà các mô hình ngây thơ (naive models) nhầm lẫn.


> Lỗi 1: Xử lý trước (Preprocessing)

Một trong những cân nhắc chính là văn bản thực sự trông như thế nào sau khi được xử lý.

![17_Error_Analysis_01](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/17_Error_Analysis_01.png)

- **Dấu câu (Punctuation):**
    + Ví dụ: Tweet "bà ngoại yêu dấu của tôi :(". Dấu câu khuôn mặt buồn (`:(`) rất quan trọng đối với tình cảm.
    + Nếu bạn **xóa dấu câu**, văn bản được xử lý ("người bà yêu quý") trông giống như một tweet rất **tích cực** (positive).
- **Từ trung lập (Neutral Words/Stop Words):**
    + Ví dụ: "Điều này **không** tốt bởi vì thái độ của bạn..." (This **not** good...).
    + Nếu bạn loại bỏ các từ trung lập như **"không" (not)**, bạn sẽ còn lại: "Tốt, thái độ, gần gũi, tốt đẹp". Bộ phân loại sẽ suy ra đây là một nội dung rất tích cực.
- Kết luận: Luôn kiểm tra kỹ văn bản đã xử lý. **Đường ống đầu vào** (Input pipeline) là một nguồn rắc rối tiềm ẩn.

---

> Lỗi 2: Thứ tự từ (Word Order)

**Bộ phân loại cơ sở ngây thơ** (Naive base classifier) bỏ lỡ tầm quan trọng của thứ tự từ.

![18_Error_Analysis_02](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/18_Error_Analysis_02.png)

- Ví dụ 1: "Tôi **hạnh phúc** vì tôi đã **không** đi." (Đây là tweet tích cực).
- Ví dụ 2: "Tôi **không hạnh phúc** vì tôi **không** đi." (Đây là cảm xúc tiêu cực).
- Thứ tự từ (vị trí của từ "không") rất quan trọng, nhưng Naive Bayes không nắm bắt được điều này.

> Lỗi 3: "Cuộc tấn công đối thủ" (Adversarial Attack)

Thuật ngữ này mô tả các hiện tượng ngôn ngữ phổ biến mà máy móc rất tệ trong việc xử lý, nhưng con người lại nhanh chóng nhận ra:

- **Châm biếm (Sarcasm)**
- **Mỉa mai (Irony)**
- **Ẩn dụ (Metaphors)**

**Ví dụ:** Một bài đánh giá phim: "Đây là một bộ phim mạnh mẽ đến mức **lố bịch**. Cốt truyện rất hấp dẫn và tôi đã **khóc** cho đến khi kết thúc."

- Đây là một bài đánh giá **tích cực**.
- Tuy nhiên, nếu bạn xử lý trước tweet này, bạn sẽ nhận được một danh sách các từ chủ yếu là tiêu cực (như "lố bịch", "khóc").
- Naive Bayes, khi áp dụng trên danh sách từ này, sẽ cho **điểm rất tiêu cực** (very negative score).

> Kết luận

- Naive Bayes đưa ra **giả định độc lập** (independence assumption), điều này có thể dẫn đến sai sót.
- Bạn đã biết cách phân tích những lỗi này.
- Mặc dù vậy, NaVve Bayes vẫn là một **cơ sở rất mạnh mẽ** (strong baseline) vì nó dựa vào **số lượng tần số từ** (word frequency counts).
- Tuần tới, bạn sẽ học cách sử dụng **vectơ từ** (word vectors), điều này có thể cho kết quả tốt hơn.


