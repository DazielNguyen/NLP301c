# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 4: Machine Translation and Document Search
- Xây dựng hệ thống dịch máy cơ bản đầu tiền của bạn và bạn sẽ sử dụng locality sensitive hashing để cải thiện hiệu suất của Nearest Neighbor Search. 
---
### **Transforming word vectors**
---

- Tuần này, sẽ sử dụng **vectơ từ (word vectors)** (hay **nhúng từ - word embeddings**) để **sắp xếp (align)** các từ trong hai ngôn ngữ khác nhau, tạo ra chương trình **dịch thuật cơ bản (basic translation)** đầu tiên.

> Biểu đồ hóa về việc sử dụng Transformation Matrix

![01_Visualization_Transformation_Matrix]()

#### 1. Ý tưởng Dịch máy

- **Cách làm của máy:**
    1.  Tính toán **nhúng từ tiếng Anh** và **nhúng từ tiếng Pháp**.
    2.  Lấy một từ nhúng tiếng Anh (ví dụ: "cat").
    3.  Tìm cách **biến đổi (transform)** từ nhúng tiếng Anh này sang không gian vectơ từ tiếng Pháp.
    4.  Tìm kiếm vectơ **tương tự nhất (similar)** trong không gian tiếng Pháp. Từ tương tự nhất (ví dụ: "chat") là ứng cử viên cho bản dịch.
- Để thực hiện phép biến đổi này, chúng ta cần tìm một **ma trận biến đổi (transformation matrix)**, ký hiệu là **R**.
- Phép biến đổi là phép nhân ma trận (ví dụ: `numpy.dot(x, R)`).

#### 2. Học Ma trận R (Training R)

- **Dữ liệu đào tạo:**
    - Chúng ta cần một **tập hợp con (subset)** các từ tiếng Anh và các từ tiếng Pháp tương đương đã biết.
    - Xếp các vectơ từ này vào hai ma trận: **X** (tiếng Anh) và **Y** (tiếng Pháp).
    - **Chìa khóa:** Các hàng phải được **sắp xếp (align)** (ví dụ: nếu hàng 1 của X là "cat", hàng 1 của Y phải là "chat").
- **Lợi ích:** Bạn chỉ cần đào tạo R trên một *tập hợp con*. Nếu R tốt, nó có thể dịch các từ *không* nằm trong bộ đào tạo ban đầu.
- **Mục tiêu:** Tìm R sao cho $X \cdot R$ (dự đoán) gần với $Y$ (thực tế) nhất.

> Hình ảnh minh họa cho bạn thấy các vectơ đã được căn chỉnh:

![02_Visualization_Aligned_Vectors]()

#### 3. Hàm tổn thất (Loss Function) và Định mức Frobenius

- Chúng ta đo lường khoảng cách (lỗi) bằng một **hàm tổn thất (loss function)**.
- Công thức tổn thất: $L = ||X \cdot R - Y||_F$
- Ký hiệu $||...||_F$ là **định mức Frobenius (Frobenius norm)**.
- **Định mức Frobenius** là căn bậc hai của tổng bình phương của tất cả các phần tử trong ma trận.
    * Ví dụ: Ma trận A là `[[2, 2], [2, 2]]`.
    * Định mức = $\sqrt{2^2 + 2^2 + 2^2 + 2^2} = \sqrt{16} = 4$.
- **Mẹo tối ưu:** Trên thực tế, việc giảm thiểu **bình phương của định mức Frobenius** ($||X \cdot R - Y||_F^2$) sẽ dễ dàng hơn, vì nó **hủy bỏ căn bậc hai (cancel the square root)**. Điều này giúp việc tính **đạo hàm (derivative)** đơn giản hơn.

#### 4. Vòng lặp đào tạo (Gradient Descent)

- Chúng ta tìm ma trận R tốt nhất bằng cách sử dụng **gradient descent**:
    1.  Bắt đầu với một **ma trận R ngẫu nhiên (random matrix R)**.
    2.  Tính **gradient** (đạo hàm của hàm tổn thất đối với R). (Script lưu ý rằng bạn không cần biết giải tích, bạn có thể tra cứu các đạo hàm này).
    3.  **Cập nhật ma trận R**: $R = R - (\alpha \times \text{gradient})$ (trong đó $\alpha$ là **tốc độ học - learning rate**).
    4.  Lặp lại quá trình này (trong một số lần lặp cố định hoặc cho đến khi tổn thất giảm xuống dưới một ngưỡng).

- **Kết luận:** Chỉ với một ma trận R, bạn có thể học cách căn chỉnh các vectơ từ giữa các ngôn ngữ.

![03_Calculate_R]()


---
### **K-nearest neighbors**
---



---
### **Hash tables and hash functions**
---



---
### **Locality sensitive hashing**
---


---
### **Multiple Planes**
---



---
### **Approximate nearest neighbors**
---


---
### **Searching Documents**
---


---
### Acknowledgements
---



---
### **Bibliography**
---


---
### **Andrew Ng with Kathleen McKeown**
---