# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 4: Machine Translation and Document Search
- Xây dựng hệ thống dịch máy cơ bản đầu tiền của bạn và bạn sẽ sử dụng locality sensitive hashing để cải thiện hiệu suất của Nearest Neighbor Search. 
---
### **Transforming word vectors**
---

- Tuần này, sẽ sử dụng **vectơ từ (word vectors)** (hay **nhúng từ - word embeddings**) để **sắp xếp (align)** các từ trong hai ngôn ngữ khác nhau, tạo ra chương trình **dịch thuật cơ bản (basic translation)** đầu tiên.

> Biểu đồ hóa về việc sử dụng Transformation Matrix

![01_Visualization_Transformation_Matrix](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/01_Visualization_Transformation_Matrix.png)

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

![02_Visualization_Aligned_Vectors](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/02_Visualization_Aligned_Vectors.png)

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
> Tổng hợp các bước và công thức tính 

![03_Calculate_R](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/03_Calculate_R.png)

---
### **K-nearest neighbors**
---

- Một thao tác chính cần thiết là tìm **k hàng xóm gần nhất (k-nearest neighbors)** của một vectơ.
- Sau khi **biến đổi** (transformation) (ví dụ: nhân với ma trận R), một vectơ (ví dụ: "hello") sẽ nằm trong không gian vectơ tiếng Pháp, nhưng nó sẽ không nhất thiết phải **giống hệt** (identical) với bất kỳ vectơ từ tiếng Pháp thực tế nào.
- Do đó, bạn cần **tìm kiếm** (search) qua các vectơ từ tiếng Pháp để tìm một từ tương tự (ví dụ: "salut" hoặc "bonjour").
- Câu hỏi đặt ra là: Làm thế nào để tìm các vectơ từ tương tự một cách hiệu quả?

> Ví dụ về ứng dụng K-nearest neighbor 
![04_Example_Visualization_K_nearest](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/04_Example_Visualization_K_nearest.png)


#### Phép loại suy (Analogy) về việc Tìm kiếm

- Một câu hỏi liên quan: Làm thế nào để bạn tìm thấy những người bạn đang sống gần đó khi bạn đang đến thăm **San Francisco**?
- **Phương pháp chậm (Tìm kiếm tuyến tính):** Xem qua *toàn bộ* sổ địa chỉ của bạn, tính toán khoảng cách của *mỗi* người bạn (ví dụ: Thượng Hải, Bangalore, Los Angeles) đến San Francisco, sau đó sắp xếp họ.
- **Vấn đề:** Nếu bạn có nhiều bạn bè, đây là một quá trình **rất tốn thời gian** (very time-consuming).
- **Giải pháp (Ý tưởng):** Bạn có thể nhận ra rằng không cần thiết phải xem xét tất cả bạn bè (ví dụ: chỉ tìm kiếm những người ở Hoa Kỳ/Bắc Mỹ).
- Ý tưởng là nếu bạn có thể chia không gian thành các **khu vực (regions)**, bạn chỉ có thể tìm kiếm trong các khu vực đó.

#### Hướng tới Bảng băm

- Việc tổ chức các tập hợp con một cách hiệu quả (đặt dữ liệu vào các **thùng chứa - buckets**) dẫn đến ý tưởng về **bảng băm (hash tables)**.
- Bảng băm là công cụ hữu ích cho công việc liên quan đến dữ liệu.
- Video này giới thiệu bảng băm, một cấu trúc dữ liệu hữu ích.
- **Video tiếp theo:** Sẽ tìm hiểu về **hashing** (băm), một kỹ thuật hiệu quả cho phép tra cứu nhanh hơn nhiều so với **tìm kiếm tuyến tính đơn giản** (simple linear search).

#### Tóm tắt
- K-nearest neighbor, cho cái matches gần nhất

---
### **Hash tables and hash functions**
---
- Một thao tác chính cần thiết cho việc dịch thuật (từ video trước) là tìm **k hàng xóm gần nhất** (k-nearest neighbors - kNN) của một vectơ.
- **Vấn đề:** Khi một vectơ tiếng Anh được biến đổi (bằng ma trận R), nó sẽ nằm trong không gian vectơ tiếng Pháp, nhưng **không nhất thiết phải giống hệt** (not necessarily identical) với bất kỳ vectơ từ nào có sẵn.
- **Mục tiêu:** Bạn cần tìm kiếm các vectơ từ tiếng Pháp thực tế **tương tự nhất** (most similar) với vectơ đã biến đổi. (Ví dụ: vectơ "hello" được biến đổi có thể gần nhất với "salut" hoặc "bonjour").
- **Câu hỏi:** Làm thế nào để tìm các vectơ tương tự một cách **hiệu quả (efficiently)**?

#### Phương pháp chậm (Tìm kiếm tuyến tính)

- Script sử dụng một ví dụ: Bạn đang ở **San Francisco (SF)** và muốn thăm những người bạn ở gần.
- Cách chậm (tương đương **tìm kiếm tuyến tính - linear search**) là xem qua *toàn bộ* sổ địa chỉ của bạn (bao gồm Thượng Hải, Bangalore, Los Angeles), tính toán khoảng cách của *từng người* đến SF, rồi sắp xếp họ.
- Việc này **rất tốn thời gian** (very time-consuming) nếu bạn có nhiều bạn bè.

#### Phương pháp hiệu quả (Hashing)

- Bạn có thể nhận ra rằng một số người bạn ở lục địa khác và **lọc (filter)** họ ra.
- Ý tưởng là **chia không gian địa lý thành các khu vực** (divide the geographical space into regions) (ví dụ: chỉ tìm kiếm những người bạn ở Bắc Mỹ).
- Khi bạn tổ chức dữ liệu thành các "tập hợp con" (subsets) hoặc "thùng chứa" (buckets), bạn nên nghĩ đến **bảng băm (hash tables)**.
- Bảng băm là một cấu trúc dữ liệu hữu ích để "tra cứu" (lookup) nhanh hơn nhiều so với tìm kiếm tuyến tính.
- **Video tiếp theo:** Sẽ tìm hiểu về hashing.

> Hãy tưởng tượng bạn phải phân loại các con số sau vào các nhóm khác nhau:

![05_Example_Hash_01](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/05_Example_Hash_01.png)

> Lưu ý rằng các hình màu xanh dương, đỏ và xám sẽ được nhóm lại với nhau.

![06_Example_Hash_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/06_Example_Hash_02.png)

> Bạn có thể nghĩ hàm băm như một hàm nhận dữ liệu có **arbitrary sizes** (kích thước tùy ý) và maps (ánh xạ) nó thành một **fixed value** (giá trị cố định). Các giá trị trả về được gọi là **hash values** (giá trị băm) hoặc thậm chí là **hashes** (các băm).

![07_Hash_Value](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/07_Hash_Value.png)

> Đoạn mã ở trên tạo ra một bảng băm (hash table) cơ bản bao gồm các giá trị đã được băm (hashed values) bên trong các `bucket` (thùng) của chúng. Hàm `hash_function` nhận vào `value_l` (một danh sách các giá trị cần băm) và `n_buckets`, sau đó thực hiện phép chia lấy dư (mods) giá trị cho số lượng `bucket`. Bây giờ để tạo `hash_table`, đầu tiên bạn khởi tạo một danh sách (list) có kích thước bằng `n_buckets` (mỗi giá trị sẽ đi vào một `bucket`). Với mỗi giá trị trong danh sách các giá trị của bạn, bạn sẽ đưa nó vào hàm `hash_function`, nhận về `hash_value` (giá trị băm), và nối (append) nó vào danh sách các giá trị trong `bucket` tương ứng.

> Giờ đây, với một `input` (đầu vào) cho trước, bạn không cần phải so sánh nó với tất cả các ví dụ khác, bạn chỉ cần so sánh nó với tất cả các giá trị trong cùng `hash_bucket` mà `input` đó đã được băm vào.

> Khi thực hiện băm, đôi khi bạn muốn các từ tương tự hoặc các số tương tự được băm vào cùng một `bucket`. Để làm điều này, bạn sẽ sử dụng “locality sensitive hashing” (băm nhạy cảm với vị trí). `Locality` là một từ khác của “location” (vị trí). Vì vậy, `locality sensitive hashing` là một phương pháp băm rất chú trọng đến việc gán các mục (items) dựa trên vị trí của chúng trong không gian véc-tơ (vector space).
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