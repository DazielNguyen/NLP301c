# Module 01 - Natural Language Processing with Classification and Vector Spaces
## Week 1: Sentiment Analysis with Logistic Regression

- Biểu diễn văn bản dưới dạng text thành vector và xây dựng một bộ phân loại sẽ phân loại văn bản mẫu thành hai loại (Tâm lý tích cực hoặc Tâm lý tiêu cực). Sử dụng Logistic Regression. 

### Logistic Regression
#### 1. Supervised Machine Learning (Học có giám sát)

- Trong máy học giám sát bạn có các tính năng đầu vào **X** và tập hợp các nhãn **Y**
- Để đảm bảo rằng bạn nhận được **dự đoán chính xác nhất** dựa trên dữ liệu của bạn.
- Mục tiêu của bản là **giảm thiểu** tỷ lệ **lỗi** của bạn hoặc **chi phí** càng nhiều càng tốt. 
- Và để làm được điều này, phải chạy được **Prediction Function** của bạn, cái mà lấy trong dữ liệu tham số để gán các Feature của bạn để đầu ra nhãn Y^
- Mapping tốt nhất từ các Features đến nhãn đạt được khi sự khác biệt giữa các giá trị kỳ vongh Y và giá trị dự đoán Y^ được **giảm thiểu**.
- **Hàm chi phí** thực hiện bằng cách so sánh mức độ gần gũi giữa Output Y^ của bạn với nhãn Y.
- Sau đó bạn có thể cập nhật tham số và lặp lại toàn bộ quá trình xử lý cho đến khi tối ưu được chi phí thấp nhất. 

![M1_W1_01_Supervised ML](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_01_Supervised%20ML.png)
> Kiến trúc của Supervise ML

#### 2. Sentiment Analysis (Phân tích tình cảm)

- Trong ví dụ này bạn có: 

    ```
    Tweet: Tôi rất hạnh phúc bởi vì tôi đang học NLP
    ```

- Và mình cần chứng minh rằng câu nói này đang là **tâm lý tích cực** hay **tâm lý tiêu cực**
- Để thực hiện bạn cần chuẩn bị tập training: 

    ```
    Postive (Tâm lý tích cực): -> Lable: 1
    Negative (Tâm lý tiêu cực): -> Lable: 0
    ```

- Dùng Logistic Regression đã dán nhãn, gán quan sát của nó cho hai lớp khác biệt

![Sentiment analysis Ví dụ](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_02_Sentiment%20analysis.png)

>Ví dụ về Sentiment analysis 

- Để xây dựng một bộ phân loại hồi quy logistic, có khả năng dự đoán tình cảm của một tweet tùy ý. 
- Xử lý các tweet thô trong training set và trích xuất các tính năng hữu ích
- Sau đó train Logistic Regression của bạn cùng với đó phải giảm thiểu chi phí. 
- Classify, cuối cùng bạn sẽ có thể đưa ra dự đoán của bạn


![Cách xây dựng Sentiment analysis](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2_02_Sentiment%20analysis%20ho%E1%BA%A1t%20%C4%91%E1%BB%99ng.png)
> Cách xây dựng Sentiment analysis







## Week 2: 
- Sử dụng phân loại Naive Bayes trên cùng một vấn đề

## Week 3: 
- Tìm hiểu về các mô hình khôn gian Vector. 
- Tim hiểu về các biểu diễn các tài liệu văn bản như tweet, artical, truy vấn hoặc bất kỳ đối tượng nào có chứa văn bản dưới dạng một Vector. Điều này quan trọng trong việc truy xuất thông tin trong tập chỉ mục, trong xếp hạng liên quan và cũng như trong lọc thông tin.

## Week 4: 
- Xây dựng hệ thống dịch máy cơ bản đầu tiền của bạn và bạn sẽ sử dụng locality sensitive hashing để cải thiện hiệu suất của Nearest Neighbor Search. 