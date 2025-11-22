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

![01_Neural_Networks_for_Sentiment_Analysis]()

> Lưu ý rằng `network` ở trên có ba `layers`. Để đi từ `layer` này sang `layer` khác bạn có thể sử dụng một `matrix` $W$ để `propagate` tới `layer` tiếp theo. Do đó, chúng ta gọi khái niệm đi từ `input` cho đến `final layer` này là `forward propagation`. Để biểu diễn một `tweet`, bạn có thể sử dụng cách sau:

![02_Neural_Networks_for_Sentiment_Analysis]()

> Lưu ý rằng, chúng ta thêm các số không (zeros) để `padding` nhằm khớp với kích thước của `tweet` dài nhất.

> Một `neural network` trong thiết lập mà bạn có thể thấy ở trên chỉ có thể `process` một `tweet` như vậy tại một thời điểm. Để làm cho việc `training` hiệu quả hơn (nhanh hơn), bạn muốn `process` nhiều `tweets` song song (`in parallel`). Bạn đạt được điều này bằng cách đặt nhiều `tweets` cùng nhau vào một `matrix` và sau đó truyền `matrix` này (thay vì các `tweets` riêng lẻ) qua `neural network`. Sau đó `neural network` có thể thực hiện các `computations` của nó trên tất cả các `tweets` cùng một lúc.

---
### **Dense Layers and ReLU**
---



---
### **Embedding and Mean Layers**
---
- Nội dung tập trung vào hai `layers` thiết yếu thường được sử dụng trong `neural networks`: `dense layer` và `ReLU layer`.

- **Dense Layer**
    + `Dense layer` tạo điều kiện cho các kết nối giữa các `layers` trong một `neural network` bằng cách thực hiện một `dot product` giữa `weights` và `activations` từ `layer` trước đó.
    + Nó tính toán tất cả các `dot products` đồng thời bằng cách sử dụng một `weight matrix` có thể được học trong quá trình `training`.

- **ReLU Layer**
    + `ReLU` (Rectified Linear Unit) `layer` áp dụng một `non-linear function` lên `output` của `dense layer`, ánh xạ các giá trị âm về không (zero).
    + Nó giữ lại các giá trị dương một cách hiệu quả trong khi chuyển đổi các giá trị âm thành không, điều này giúp ổn định `performance` của `network`.

- Tổng quan này làm nổi bật các thành phần cơ bản của `neural networks` đóng góp vào chức năng của chúng.


---
### **Traditional Language models**
---


---
### **Recurrent Neural Networks**
---



---
### **Applications of RNNs**
---


---
### **Math in Simple RNNs**
---


---
### **Cost Function for RNNs**
---



---
### **Implementation Note**
---





---
### **Gated Recurrent Units**
---





---
### **Deep and Bi-directional RNNs**
---
