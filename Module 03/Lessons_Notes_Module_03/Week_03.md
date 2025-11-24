# **Module 03 - Natural Language Processing with Sequence Models**
## **Week 3: Siamese Networks**
---
### **Siamese Networks**
---

Trong bài giảng này, bạn sẽ tìm hiểu về `Siamese networks`, một loại `neural network architecture` cụ thể bao gồm hai `networks` giống hệt nhau được hợp nhất ở phần cuối, với nhiều ứng dụng khác nhau trong `natural language processing` (`NLP`).

**Siamese Networks Overview**

* `Siamese networks` được thiết kế để so sánh sự tương đồng (`similarity`) giữa hai `inputs`, chẳng hạn như các câu hỏi hoặc câu nói.
* Chúng đặc biệt hữu ích trong việc xác định các bản sao (`duplicates`) trong các câu hỏi, ngay cả khi được diễn đạt khác nhau.

**Applications in NLP**

* Các `networks` này có thể được sử dụng để xác thực chữ ký bằng cách so sánh hai chữ ký viết tay về độ tương đồng.
* Chúng giúp xác định các câu hỏi trùng lặp trên các nền tảng như Quora hoặc Stack Overflow, đảm bảo người dùng không đăng cùng một câu hỏi nhiều lần.

**Sentiment Analysis and Classification**

* Trong `sentiment analysis`, `Siamese networks` có thể xác định `sentiment` của các phát biểu bằng cách so sánh các `features` biểu thị `sentiment` tích cực hoặc tiêu cực.
* Một `similarity score` được tính toán để đánh giá mối quan hệ giữa hai `inputs`, hỗ trợ trong các tác vụ phân loại (`classification tasks`).

Bài giảng đặt nền tảng cho việc khám phá sâu hơn về `architecture` và các ứng dụng thực tế của `Siamese networks` trong `NLP`.

> Tốt nhất là mô tả một `Siamese network` thông qua một ví dụ.

![01_Siamese_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/01_Siamese_Networks.png)

> Lưu ý rằng trong ví dụ đầu tiên ở trên, hai câu có nghĩa giống nhau nhưng có các từ hoàn toàn khác nhau. Trong khi ở trường hợp thứ hai, hai câu có nghĩa hoàn toàn khác nhau nhưng chúng có các từ rất giống nhau.

> `Classification`: học điều gì làm cho một `input` là chính nó (what it is).

> `Siamese Networks`: học điều gì làm cho hai `inputs` giống nhau.

> Dưới đây là một vài ứng dụng của `siamese networks`:

![02_Siamese_Networks](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/02_Siamese_Networks.png)

---
### **Architecture**
---

Nội dung tập trung vào `architecture` và cách thức hoạt động của `Siamese networks`, được sử dụng để so sánh các `inputs` và tạo ra các `similarity scores`.

**Siamese Network Architecture**

* Bao gồm hai `subnetworks` giống hệt nhau xử lý hai `inputs` (ví dụ: các câu hỏi) để tạo ra các `outputs`.
* Mỗi `subnetwork` chuyển đổi `input` của nó thành một `embedding` và chuyển nó qua một `LSTM layer` để nắm bắt ý nghĩa.


**Parameter Sharing and Output**

* Các `subnetworks` chia sẻ các `parameters` giống hệt nhau, cho phép `training` chỉ một tập hợp các `weights`.
* `Outputs` từ cả hai `subnetworks` được so sánh bằng cách sử dụng `cosine similarity` để xác định mức độ tương tự của các `inputs`.

**Cosine Similarity and Thresholding**

* `Cosine similarity` đo góc giữa hai `vectors`, biểu thị sự tương đồng của chúng.
* Một `threshold` (tau) được thiết lập để phân loại các `inputs` là tương tự hoặc khác biệt dựa trên `cosine similarity score`.

> `Model architecture` của một `siamese network` điển hình có thể trông như sau:

![03_Architecture](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/03_Architecture.png)


![04_Architecture](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/04_Architecture.png)


> Hai `sub-networks` này là `sister-networks` kết hợp với nhau để tạo ra một `similarity score`. Không phải tất cả `Siamese networks` sẽ được thiết kế để chứa `LSTMs`. Một điều cần nhớ là các `sub-networks` chia sẻ các `identical parameters`. Điều này có nghĩa là bạn chỉ cần `train` một tập hợp các `weights` chứ không phải hai.

> `Output` của mỗi `sub-network` là một `vector`. Sau đó bạn có thể chạy `output` qua một `cosine similarity function` để lấy `similarity score`. Trong video tiếp theo, chúng ta sẽ nói về `cost function` cho một `network` như vậy.


---
### **Cost Function**
---

Nội dung tập trung vào `triplet loss function` được sử dụng trong `Siamese networks` để xác định độ tương đồng giữa các câu hỏi.

**Siamese Network Overview**

* `Siamese network` dự đoán xem hai câu hỏi là tương tự hay khác biệt.
* Nó sử dụng một `anchor` question để so sánh với các câu hỏi `positive` (nghĩa tương tự) và `negative` (nghĩa khác biệt).

**Cosine Similarity**

* `Cosine similarity` đo lường độ tương đồng giữa hai `vectors`, nằm trong khoảng từ -1 (hoàn toàn khác biệt) đến 1 (gần như giống hệt nhau).
* Một `model` được `trained` tốt nhắm tới độ tương đồng gần bằng 1 cho các cặp `anchor-positive` và gần bằng -1 cho các cặp `anchor-negative`.

**Loss Function Development**

* `Loss function` được tạo ra bằng cách trừ độ tương đồng của `anchor` và câu hỏi `negative` từ độ tương đồng của `anchor` và câu hỏi `positive`.
* Việc giảm thiểu (`minimizing`) `loss` này trong quá trình `training` giúp `model` phân biệt hiệu quả giữa các câu hỏi tương tự và khác biệt.

> Hãy xem kỹ `slide` sau:

![05_Cost_Function](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/05_Cost_Function.png)


> Lưu ý rằng khi cố gắng tính toán `cost` cho một `siamese network` bạn sử dụng `triplet loss`. `Triplet loss` xem xét một ví dụ `Anchor`, một `Positive` và một `Negative`. Điều quan trọng cần lưu ý là bạn nhắm đến việc điều chỉnh các `weights` của `model` theo cách mà `anchor` và ví dụ `positive` có một `cosine similarity score` gần bằng 1. Ngược lại, `anchor` và ví dụ `negative` nên có một `cosine similarity score` gần bằng -1. Cụ thể hơn, bạn tìm cách giảm thiểu (`minimize`) phương trình sau:
$-\cos(A,P)+\cos(A,N)\le 0$

> Lưu ý rằng nếu $\cos(A,P)=1$ và $\cos(A,N)=-1$, thì phương trình chắc chắn nhỏ hơn 0. Tuy nhiên, khi $\cos(A,P)$ lệch khỏi 1 và $\cos(A,N)$ lệch khỏi -1, thì cuối cùng bạn có thể nhận được một `cost` > 0. Đây là một `visualization` sẽ giúp bạn hiểu những gì đang diễn ra. Hãy thoải mái thử nghiệm với các con số khác nhau.

![06_Cost_Function](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/06_Cost_Function.png)

---
### **Triplets**
---

Nội dung này tập trung vào khái niệm `triplet loss` trong việc `training` các `models` để xác định các `inputs` tương đương.

**Triplet Loss and Its Components**

* `Triplet loss` bao gồm ba thành phần: một `anchor` (A), một ví dụ `positive` (P), và một ví dụ `negative` (N).
* Mục tiêu là tối đa hóa sự tương đồng (`similarity`) giữa `anchor` và ví dụ `positive`, trong khi giảm thiểu sự tương đồng giữa `anchor` và ví dụ `negative`.

**Understanding the Loss Function**

* `Triplet loss function` đơn giản nhất giảm thiểu sự khác biệt giữa các độ tương đồng của A và P, và A và N.
* Một `margin` (alpha) được đưa vào để đảm bảo `model` học cho đến khi sự khác biệt giữa các độ tương đồng đạt đến một giá trị cụ thể, ngăn chặn `loss` trở nên âm.

**Selecting Triplets for Training**

* Việc chọn `hard triplets`, nơi `anchor` và các ví dụ `negative` có độ tương đồng gần nhau, giúp `model` học tập hiệu quả.
* Cách tiếp cận này tập trung quá trình `training` vào các trường hợp thách thức, cải thiện `performance` của `model` bằng cách cung cấp nhiều ví dụ mang tính thông tin hơn.

> Để có được `full cost function` bạn sẽ thêm một `margin` vào `cost function` trước đó.

![07_Triplets](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/07_Triplets.png)

> Lưu ý $\alpha$ trong phương trình ở trên, đại diện cho `margin`. Điều này cho phép bạn có một chút "safety", khi so sánh các câu. Khi tính toán `full cost`, bạn lấy `max` của kết quả $-\cos(A,P)+\cos(A,N)+\alpha$ và 0. Lưu ý, chúng ta không muốn lấy một số âm làm `cost`.

> Dưới đây là một tóm tắt nhanh:

> $\alpha$: kiểm soát $\cos(A,P)$ cách xa $\cos(A,N)$ bao nhiêu

> `Easy negative triplet`: $\cos(A,N) < \cos(A,P)$

> `Semi-hard negative triplet`: $\cos(A,N) < \cos(A,P) < \cos(A,N) + \alpha$

> `Hard negative triplet`: $\cos(A,P) < \cos(A,N)$


---
### **Computing The Cost I**
---
Bài giảng này tập trung vào việc xây dựng một `cost function` và tối ưu hóa nó bằng cách sử dụng `gradient descent` trong bối cảnh của một `Siamese network` cho `natural language processing`.

**Batch Preparation**

* Dữ liệu được tổ chức thành các `batches`, với mỗi câu hỏi có các bản sao (`duplicates`) tương ứng.
* Mỗi `batch` chứa các câu hỏi duy nhất, đảm bảo không có `duplicates` trong cùng một `batch`.

**Model Output**

* `Model` xử lý `batch` để tạo ra một `vector output`, với các kích thước (`dimensions`) được xác định bởi `embedding layer`.
* `Output` là một `matrix` của các `stacked vectors`, đại diện cho nhiều quan sát (`observations`) trong `batch`.

**Similarity Calculation**

* `Model` tính toán độ tương đồng (`similarity`) giữa các cặp `vector` từ hai `batches`, xác định các `duplicates` thông qua các `similarity scores` cao hơn.
* Các ví dụ `positive` (`duplicates`) hiển thị các giá trị `similarity` cao hơn so với các ví dụ `negative` (`non-duplicates`).

**Cost Function and Training**

* `Overall cost` cho `Siamese network` được suy ra từ các `losses` riêng lẻ trên các `training sets`.
* `Hard negative mining` được giới thiệu để nâng cao `model performance` bằng cách sử dụng các `duplicates` hiện có làm các ví dụ `positive` và các `non-duplicates` làm các ví dụ `negative`.

> Để tính toán `cost`, bạn chuẩn bị các `batches` như sau:

![08_Computing_The_Cost_I](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/08_Computing_The_Cost_I.png)

> Lưu ý rằng mỗi ví dụ ở bên trái có một ví dụ tương tự ở bên phải nó, nhưng không có ví dụ nào khác ở trên hoặc dưới nó có nghĩa giống như vậy.
> Sau đó, bạn có thể tính toán `similarity matrix` giữa mỗi cặp có thể từ các cột bên trái và bên phải.

![09_Computing_The_Cost_I](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/09_Computing_The_Cost_I.png)

> `Diagonal line` (đường chéo) tương ứng với các điểm số của các câu tương tự, (thông thường chúng nên là `positive`). Các `off-diagonals` (phần nằm ngoài đường chéo) tương ứng với các `cosine scores` giữa `anchor` và các ví dụ `negative`.

---
### **Computing The Cost II**
---

Nội dung tập trung vào các khái niệm `diagonals` và `off-diagonals` trong một `cost matrix` được sử dụng trong việc `training` `Siamese networks`.

**Understanding Diagonals and Off-Diagonals**

* Các `diagonal values` đại diện cho các `similarities` cho các câu hỏi trùng lặp (`duplicate questions`), cái mà nên cao hơn các `off-diagonal values` đối với một `well-trained model`.
* Các `off-diagonal values` đại diện cho các `similarities` cho các câu hỏi không trùng lặp (`non-duplicate questions`) và có thể được sử dụng để nâng cao `model performance`.

**Mean Negative and Closest Negative Concepts**

* `Mean negative` đề cập đến trung bình của các `off-diagonal values` trong mỗi hàng, giúp giảm `noise` trong quá trình `training`.
* `Closest negative` là `off-diagonal value` gần nhất với `diagonal value`, cung cấp các cơ hội học tập giá trị cho `model`.

**Loss Function Modifications**

* `Triplet loss function` có thể được cải thiện bằng cách kết hợp `mean negative` và `closest negative` để tạo ra một `loss function` mới.
* Sự điều chỉnh này giúp `model` `converge` (hội tụ) nhanh hơn bằng cách tập trung vào các ví dụ thách thức hơn, cuối cùng là nâng cao `training efficiency`.

> Bây giờ bạn đã có `matrix` với các `cosine similarity scores`, vốn là `product` (tích) của hai `matrices`, chúng ta tiếp tục tính toán `cost`.

![10_Computing_The_Cost_II](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2003/Image_Module_03/M3_W3/10_Computing_The_Cost_II.png)

> Bây giờ chúng ta giới thiệu hai khái niệm, `mean_neg`, là `mean negative` của tất cả các `off diagonals` khác trong `row`, và `closest_neg`, tương ứng với số cao nhất trong các `off diagonals`.

$$Cost=\max(-\cos(A,P)+\cos(A,N)+\alpha,0)$$

> Vì vậy bây giờ chúng ta sẽ có hai `costs`:

$$Cost1=\max(-\cos(A,P)+mean\_neg+\alpha,0)$$

$$Cost2=\max(-\cos(A,P)+closest\_neg+\alpha,0)$$

> `Full cost` được định nghĩa là: `Cost 1` + `Cost 2`.


---
### **Out of Vocabulary Words**
---


---
### **Smoothing**
---



---
### **Week Summary**
---

