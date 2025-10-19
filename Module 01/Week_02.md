# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 2: Sentiment Analysis with Naive Bayes
### Probability and Bayes' Rule

> Video này cung cấp cái nhìn tổng quan về **Quy tắc Bayes** (Bayes' rule) và **xác suất** (probability).

![01_Probability_and_Bayes'_Rule]()

- Xác suất là cơ bản cho nhiều ứng dụng trong **NLP**, ví dụ như phân loại tweet (tích cực hay tiêu cực).
    1. Xem xét **xác suất** và **xác suất có điều kiện** (conditional probability).
    2. Rút ra **Quy tắc Bayes** từ xác suất có điều kiện.
- Quy tắc Bayes được áp dụng rộng rãi (y học, giáo dục, **NLP**).
- Bạn sẽ dùng nó để thực hiện **phân tích tình cảm** (sentiment analysis) (nhiệm vụ tuần này).
- Trong khóa học tiếp theo, bạn cũng sẽ dùng nó để **tự động sửa** (auto-correct).
> Ví dụ: Một kho (corpus) lớn các tweet được phân loại là **tích cực** hoặc **tiêu cực**, nhưng **không phải cả hai**. Từ "happy" (hạnh phúc) có thể xuất hiện ở cả hai loại.

![02_Probability_and_Bayes'_Rule_Example]()

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









### Bayes' Rule
### Naive Bayes Introduction
### Laplacian Smoothing
### Log Likelihood
### Training Naive Bayes
### Testing Naive Bayes
### Applications og Naive Bayes
### Naive Bayes Assumption
### Error Analysis


