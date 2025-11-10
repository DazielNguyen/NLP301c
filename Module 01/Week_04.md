# **Module 01 - Natural Language Processing with Classification and Vector Spaces**
## **Week 4: Machine Translation and Document Search**
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
- `Locality sensitive hashing` (băm nhạy cảm với vị trí) là một kỹ thuật cho phép bạn băm các `input` (đầu vào) tương tự vào cùng một `bucket` (thùng) với xác suất cao.

> Ví dụ dễ hình dung

![08_Example_Locality_Sensitive_Hashing](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/08_Example_Locality_Sensitive_Hashing.png)

- Một phương pháp để giảm chi phí tính toán khi tìm kiếm hàng xóm (neighbor search) trong không gian chiều cao là **băm nhạy cảm với địa phương** (Locality-Sensitive Hashing - LSH).
- Video này giới thiệu về **hash** (băm) là gì.
- **Ví dụ (2D):** Giả sử bạn có các vectơ (chấm xanh, chấm xám). Bạn muốn biết các chấm xanh ở gần nhau và các chấm xám cũng gần nhau.
- **Ý tưởng:** Chia không gian bằng các đường đứt nét, được gọi là **máy bay** (planes).
- **Quan sát:** Các vectơ liên quan (ví dụ: các chấm xanh) thường nằm ở cùng một phía của một "máy bay" cụ thể. Điều này giúp xếp các vectơ thành các tập con dựa trên vị trí của chúng.
- Một hàm băm nhạy cảm với **vị trí (locality)** của các mục được gọi là **băm nhạy cảm với địa phương**.
- **Định nghĩa "Máy bay":** Một "máy bay" (plane) (ví dụ: đường màu đỏ tươi) có thể được xác định bằng một **vectơ bình thường** (normal vector - P) duy nhất, là vectơ **vuông góc** (perpendicular) với "máy bay" đó.

> Thay vì các `bucket` (thùng) thông thường mà chúng ta đã sử dụng, bạn có thể nghĩ đến việc phân cụm (clustering) các điểm bằng cách quyết định xem chúng ở trên hay dưới một đường thẳng (line). Bây giờ khi chúng ta đi đến các chiều cao hơn (ví dụ các véc-tơ n-chiều), bạn sẽ sử dụng các mặt phẳng (planes) thay vì các đường thẳng. Chúng ta hãy xem một ví dụ cụ thể:

![09_Example_Locality_Sensitive_Hashing_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/09_Example_Locality_Sensitive_Hashing_02.png)

> Cho một `point` (điểm) tại $(1,1)$ và ba `vectors` (véc-tơ) $V_1=(1,2)$, $V_2=(-1,1)$, $V_3=(-2,-1)$
bạn sẽ thấy điều gì xảy ra khi chúng ta thực hiện `dot product` (tích vô hướng).
Đầu tiên lưu ý rằng `dashed line` (đường nét đứt) là `plane` (mặt phẳng) của chúng ta.
`Vector` (véc-tơ) với `point` $P=(1,1)$ thì `perpendicular` (vuông góc) với `line` (`plane`) đó.
Bây giờ bất kỳ `vector` nào ở trên `dashed line` mà được `multiplied by` (nhân với) $(1,1)$ sẽ có một `positive number` (số dương).
Bất kỳ `vector` nào bên dưới `dashed line` khi được `dotted with` $(1,1)$ sẽ có một `negative number` (số âm).
Bây giờ bất kỳ `vector` nào trên `dashed line` được `multiplied by` (nhân với) $(1,1)$ sẽ cho bạn một `dot product` bằng 0.
- **Câu hỏi:** Làm thế nào để biết một vectơ nằm ở bên nào của "máy bay" bằng toán học?
- **Giải pháp:** Sử dụng **tích chấm** (dot product) của vectơ (V) với vectơ bình thường (P).
- **Ý nghĩa của Dấu (Sign):**
    + Tích chấm **dương** (positive) (ví dụ: V1 · P = 3) $\rightarrow$ Vectơ nằm ở một bên của "máy bay".
    + Tích chấm **âm** (negative) (ví dụ: V3 · P = -3) $\rightarrow$ Vectơ nằm ở phía đối diện.
    + Tích chấm **bằng 0** (zero) (ví dụ: V2 · P = 0) $\rightarrow$ Vectơ nằm trên "máy bay".

> Trực quan hóa Tích chấm

![10_Visualization_Production_Dot](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/10_Visualization_Production_Dot.png)

- **Trực quan hóa Tích chấm:** Tích chấm có thể được coi là **phép chiếu** (projection) của vectơ V lên vectơ P.
- **Dấu hiệu của tích chấm** (sign of the dot product) cho biết hướng của phép chiếu (so với P), từ đó cho biết vectơ nằm ở bên nào của "máy bay".

> Tính side_of_plan bằng Python

![11_Python_Code](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/11_Python_Code.png)

- **Python:** Một hàm (`side_of_plane`) được minh họa, sử dụng `np.dot` (tích chấm) và `np.sign` (dấu) để trả về +1 (dương), -1 (âm), hoặc 0.
- **Điểm rút ra chính:** Dấu hiệu của phép chiếu (tích chấm) cho bạn biết điểm nằm ở phần nào của đường thẳng (trên hoặc dưới).

---
### **Multiple Planes**
---

> Phần này chỉ bạn cách kết hợp thông tin từ **nhiều mặt phẳng** (multiple planes) để có được một **giá trị băm** (hash value).

![12_Multi_Planes](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/12_Multi_Planes.png)

- Video trước đã chỉ ra **dấu của tích chấm** (sign of the dot product) cho biết vị trí tương đối của vectơ so với *một* mặt phẳng.
- Bạn sử dụng nhiều hơn một mặt phẳng để chia **không gian vectơ** (vector space) thành các **vùng** (regions) nhỏ hơn, dễ quản lý hơn.
- Bạn cần kết hợp các "tín hiệu" (signals) (mỗi mặt phẳng một tín hiệu) thành một **giá trị băm duy nhất** (single hash value) để xác định một vùng cụ thể (hoặc "thùng" - bucket).

#### Quy trình tạo giá trị băm

1.  **Lấy tín hiệu từ mỗi mặt phẳng:**
    - **Quy tắc:** Dấu của tích chấm giữa vectơ (V) và vectơ bình thường của mặt phẳng ($P_i$) quyết định **giá trị băm trung gian** ($h_i$).
    - Nếu (tích chấm) $\ge 0 \rightarrow$ $h_i = 1$.
    - Nếu (tích chấm) $< 0 \rightarrow$ $h_i = 0$.

2.  **Kết hợp các tín hiệu:**
    - Bạn kết hợp các giá trị băm trung gian ($h_1, h_2, h_3, ...$) thành một giá trị băm duy nhất bằng một công thức tổng trọng số.
    - **Công thức:** $\text{Hash} = (2^0 \times h_1) + (2^1 \times h_2) + (2^2 \times h_3) + \dots$
    - **Ví dụ:** Nếu một vectơ có $h_1=1$ (dương ở P1), $h_2=1$ (dương ở P2), và $h_3=0$ (âm ở P3), giá trị băm kết hợp là:
        $(2^0 \times 1) + (2^1 \times 1) + (2^2 \times 0) = 1 + 2 + 0 = 3$.

$$hash_v \text{ alue} = \sum_i^H 2^i \times h_i$$

*(Lưu ý: Dường như có một lỗi đánh máy nhỏ trong hình ảnh gốc. Dựa trên nội dung các video trước đó, công thức này có thể được dự định là $\text{hash\_value}$ (giá trị băm).)*
3.  **Code (Logic):**
> Code hash_multiple_plane

![13_Code_hash_multiple_plane](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/13_Code_hash_multiple_plane.png)


- Bắt đầu với `hash_value = 0`.
- Lặp qua các mặt phẳng (i = 0, 1, 2, ...).
- Tính dấu của tích chấm.
- Gán `hash_i` là 1 (nếu dấu $\ge 0$) hoặc 0 (nếu dấu $< 0$).
- Cộng dồn vào giá trị băm: `hash_value += h_i * (2**i)`.
- `P_l` là danh sách các mặt phẳng (list of planes). Bạn khởi tạo giá trị bằng 0, và sau đó bạn duyệt qua (iterate) tất cả các mặt phẳng (`P`), và bạn theo dõi `index` (chỉ số). Bạn lấy `sign` (dấu) bằng cách tìm dấu của `dot product` (tích vô hướng) giữa `v` và mặt phẳng `P` của bạn. Nếu nó `positive` (dương) bạn gán nó bằng 1, ngược lại bạn gán nó bằng 0. Sau đó, bạn cộng `score` (điểm) cho mặt phẳng thứ `i` vào `hash value` (giá trị băm) bằng cách tính $2^i \times h_i$.
- **Kết luận:** Đây là cách bạn có được **hàm băm nhạy cảm với địa phương** (locality-sensitive hash function).

---
### **Approximate nearest neighbors**
---

- Phần này giải thích cách sử dụng **Hashing Nhạy cảm với Địa phương (Locality-Sensitive Hashing - LSH)** để tăng tốc độ tìm kiếm **k hàng xóm gần nhất (k-nearest neighbors)**.

#### Vấn đề và Giải pháp

- **Vấn đề:** Một bộ mặt phẳng (planes) có thể chia không gian vectơ, nhưng chúng ta không biết đó có phải là cách tốt nhất hay không.
- **Giải pháp:** Thay vì chỉ dùng một bộ, hãy tạo **nhiều bộ mặt phẳng ngẫu nhiên** (multiple sets of random planes). Điều này giống như tạo ra nhiều **bảng băm độc lập** (independent hash tables) (video gọi đây là "đa vũ trụ" - multiverse).

> Approximate nearest neighbors không cung cấp full nearest neighbors nhưng cung cấp cho mình một xấp xỉ aproximate. Nó thường đánh đổi độ chính xác để lấy hiệu quả. Tham khảo vào biểu đồ sau: 

![14_Example_ANN](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/14_Example_ANN.png)


#### Cách thức hoạt động

1.  Giả sử bạn có một vectơ truy vấn (query vector) (ví dụ: chấm đỏ tươi).
2.  **Bộ mặt phẳng 1 (Vũ trụ 1):** Vectơ đỏ và các vectơ xanh lá cây được gán vào **cùng một túi băm** (same hash bucket).
3.  **Bộ mặt phẳng 2 (Vũ trụ 2):** Vectơ đỏ và các vectơ xanh lam rơi vào cùng một túi.
4.  **Bộ mặt phẳng 3 (Vũ trụ 3):** Vectơ đỏ và các vectơ màu cam rơi vào cùng một túi.
5.  Bằng cách sử dụng nhiều bộ mặt phẳng ngẫu nhiên, bạn có một cách mạnh mẽ hơn để tìm kiếm một tập hợp các vectơ có thể là **ứng cử viên gần nhất** (closest candidates).

=>  Vì vậy, bạn có thể thấy rằng khi làm nhiều lần hơn, bạn có khả năng nhận được tất cả các hàng xóm. Đây là mã cho một tập hợp các mặt phẳng ngẫu nhiên. Hãy đảm bảo rằng bạn hiểu những gì đang diễn ra.
#### Hàng xóm gần nhất Xấp xỉ (ANN)

- Phương pháp này được gọi là **Hàng xóm gần nhất Xấp xỉ** (Approximate Nearest Neighbors - ANN).
- Nó "xấp xỉ" (approximate) bởi vì bạn không tìm kiếm toàn bộ không gian vectơ mà chỉ tìm kiếm một **tập hợp con** (subset) của nó (chỉ những vectơ rơi vào cùng túi băm với vectơ truy vấn).
- **Sự đánh đổi:** Bạn **hy sinh một số độ chính xác** (sacrifice some accuracy) để đạt được **hiệu quả** (efficiency) (tốc độ) cao hơn nhiều trong tìm kiếm.

#### Triển khai trong Code

- Để tạo (ví dụ) 3 mặt phẳng ngẫu nhiên trong không gian 2 chiều, bạn có thể sử dụng `np.random.normal` để tạo một ma trận (ví dụ: 3 hàng x 2 cột).
- Thay vì sử dụng vòng lặp `for`, bạn có thể sử dụng `np.dot` để tính tích chấm của vectơ (V) với *tất cả* các mặt phẳng trong một bước, nhằm xác định vectơ nằm ở phía dương hay âm của mỗi mặt phẳng.

![15_Code_ANN](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/15_Code_ANN.png)

---
### **Searching Documents**
---

- Bạn có thể sử dụng **k-near láng giềng (k-nearest neighbors)** nhanh để **tìm kiếm tài liệu** (document search) liên quan đến một truy vấn trong một bộ sưu tập lớn.
> Video trước đã cho bạn thấy một ví dụ minh họa về cách bạn thực sự có thể biểu diễn một tài liệu dưới dạng vector.

![16_Searching_Document](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/16_Searching_Document.png)

- Bạn cần tạo vectơ cho cả truy vấn và tài liệu.
- **Cách biểu diễn tài liệu (document) dưới dạng vectơ:**
    + Ví dụ: Tài liệu "Tôi thích học tập".
    + Lấy **vectơ từ (word vector)** cho từng từ riêng lẻ ("Tôi", "thích", "học tập").
    + **Kết hợp chúng lại** (combine them). Cụ thể, **tổng (sum)** của tất cả các vectơ từ này trở thành **vectơ tài liệu (document vector)**.
    + Vectơ tài liệu này có **cùng chiều (same dimension)** với các vectơ từ (ví dụ: 3 chiều).

- **Code (logic):**
    + Tạo một từ điển để **nhúng từ (word embeddings)**.
    + Khởi tạo **nhúng tài liệu (document embedding)** là một mảng zero.
    + Lặp qua mỗi từ trong tài liệu, lấy vectơ của nó (nếu tồn tại) và **cộng (add)** vào vectơ nhúng tài liệu.
    + Trả về vectơ nhúng tài liệu (tổng).
- **Kết luận:** Đây là một phương pháp chung: văn bản có thể được **nhúng vào không gian vectơ** (embedded into vector space). Trong không gian đó, **hàng xóm gần nhất (nearest neighbors)** đề cập đến văn bản có **ý nghĩa tương tự (similar meaning)**.

- **Cấu trúc cơ bản** (basic structure) này được sử dụng lặp đi lặp lại trong toàn bộ **NLP hiện đại** (modern NLP).

> Trong ví dụ này, bạn chỉ cần cộng các vector từ của một tài liệu để nhận được vector của tài liệu đó. Tóm lại, bây giờ bạn nên quen thuộc với các khái niệm sau:

![17_Tong_hop](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W4/17_Tong_hop.png)

---
### **Andrew Ng with Kathleen McKeown**
---

Dưới đây là tóm tắt các ý chính từ cuộc phỏng vấn của Andrew Ng với Kathy McKeown:

#### Giới thiệu

* Andrew Ng giới thiệu **Kathy McKeown**, Giáo sư Khoa học Máy tính tại **Đại học Columbia**, Giám đốc sáng lập **Viện Khoa học Dữ liệu và Kỹ thuật** (Data Science and Engineering Institute), và là **Học giả Amazon** (Amazon Scholar).
* Bà được biết đến nhiều nhất với công trình về **tóm tắt văn bản** (text summarization).

#### Con đường đến với NLP

* Andrew hỏi về hành trình của Kathy, lưu ý rằng bà học chuyên ngành **Văn học So sánh** (Comparative Literature) tại **Đại học Brown**, mặc dù cũng có định hướng toán học.
* Kathy giải thích rằng bà đã làm công việc lập trình sau khi tốt nghiệp và thấy nó "rất nhàm chán".
* Một người bạn (chuyên ngành ngôn ngữ học) đã giới thiệu cho bà về **ngôn ngữ học tính toán** (computational linguistics).
* Bà đã dành một năm trong thư viện *tự mình* đọc về AI và NLP. Bà nộp đơn vào trường cao học (Cao học) vì nó kết hợp cả hai mối quan tâm của bà là ngôn ngữ và toán học.

#### Tự học và Hội chứng Kẻ mạo danh

* Khi mới bắt đầu tự học, bà không có hướng dẫn và tự mình tìm kiếm các tài liệu tham khảo.
* Khi vào Cao học tại **Penn** (điều mà bà thừa nhận là "hoàn toàn may mắn" vì đó là nơi tốt nhất về NLP vào thời điểm đó), bà cảm thấy "rất đáng sợ" và như một **"kẻ mạo danh"** (imposter).
* Bà khuyên những người học ngày nay đang cảm thấy cô đơn hoặc bị cô lập hãy tiếp cận và nói chuyện với mọi người, tham gia các nhóm đọc trực tuyến, hoặc tham gia các khóa học (như của Andrew).

#### Công việc hiện tại và Tóm tắt tiểu thuyết

* Công việc chính của bà trong những năm gần đây là **tóm tắt** (summarization).
* Một dự án thú vị bà làm với Amazon là **tóm tắt các chương tiểu thuyết** (summarizing novel chapters).
* Đây là một nhiệm vụ rất khó khăn vì hai lý do chính:
    1.  Các chương dài hơn nhiều so với các bài báo tin tức (nơi hầu hết các công việc tóm tắt hiện tại được thực hiện).
    2.  Có một **"sự diễn giải cực kỳ"** (extreme paraphrasing) giữa đầu vào (tiểu thuyết thế kỷ 19) và đầu ra (bản tóm tắt bằng ngôn ngữ ngày nay).
* Bà quan tâm đến **tóm tắt trừu tượng** (abstractive summarization), nơi bản tóm tắt sử dụng các từ và cấu trúc cú pháp khác với bản gốc.

#### Về việc chọn các Vấn đề Nghiên cứu Mới

* Kathy thích **công việc liên ngành** (interdisciplinary work) (với báo chí, y tế, v.v.) vì nó mang lại một góc nhìn khác.
* Bà khuyên các nhà nghiên cứu nên chọn những vấn đề *khác biệt* và *quan trọng*.
* Bà chỉ trích công việc tập trung vào **tóm tắt tài liệu đơn của tin tức** (single document news summarization). Mặc dù có nhiều dữ liệu (như **CNN Daily Mail**) và **bảng xếp hạng** (leaderboards), nhưng đó không phải là một nhiệm vụ thực sự cần thiết, vì **câu dẫn đầu** (hai câu đầu tiên) của một bài báo đã là một bản tóm tắt tốt và rất khó bị đánh bại.
* Bà thích đi theo hướng mới (như tóm tắt tiểu thuyết hoặc tóm tắt câu chuyện cá nhân sau thảm họa) vì nó giải quyết một vấn đề quan trọng và bạn sẽ là "người đầu tiên đi đến giải pháp".
* Thách thức của việc này là các bài báo rất khó được đánh giá do không có **điểm chuẩn** (benchmarks) hoặc công việc trước đó.

#### Về các Số liệu (Metrics)

* Cả hai đều đồng ý rằng các **số liệu tự động** (automatic metrics) (ví dụ: **Rouge** trong tóm tắt, hoặc trong dịch máy) thường bị **sai sót** (flawed), nhưng mọi người vẫn tiếp tục sử dụng chúng vì lý do lịch sử (để so sánh).

#### Tác động xã hội và Thuật toán Không thiên vị

* Kathy đang làm việc với các nhà nghiên cứu công tác xã hội và ngôn ngữ học để phân tích ngôn ngữ từ **cộng đồng da đen** (Black community) ở **Harlem**.
* Họ đang xem xét các phản ứng (cảm xúc) đối với các sự kiện lớn như **Black Lives Matter** và **COVID-19**.
* Các mục tiêu bao gồm:
    1.  Hiểu cách **ngôn ngữ người Mỹ gốc Phi** (African American language) thể hiện cảm xúc so với tiếng Anh tiêu chuẩn.
    2.  Phát triển các **thuật toán không thiên vị** (unbiased algorithms), vì hầu hết các hệ thống hiện tại được đào tạo trên ngôn ngữ tin tức (ví dụ: Tạp chí Phố Wall).
    3.  Xem xét tác động của **chấn thương** (trauma) (cá nhân hoặc khi chứng kiến người khác bị hại).
* Bà cũng đề cập đến các công việc có tác động xã hội trước đây, như phân tích các bài đăng trên mạng xã hội liên quan đến **băng đảng** và tạo **cập nhật thảm họa** (sau cơn bão Sandy).

#### Sự phát triển của NLP

* Kathy (lấy bằng Ph.D. năm '82) mô tả lĩnh vực NLP thời kỳ đầu:
    1.  Rất **liên ngành** (interdisciplinary), lấy lý thuyết từ ngôn ngữ học, triết học và tâm lý học.
    2.  Bà bị ảnh hưởng rất nhiều bởi các **phụ nữ cao cấp** (senior women) trong lĩnh vực, bao gồm **Bonnie Weber**, **Eva Hychova**, **Barbara Gross**, và **Karen Spark Jones**.
    3.  Họ đã rút ra các lý thuyết (ví dụ: "trọng tâm của sự chú ý" (focus of attention) từ ngôn ngữ học, hoặc lý thuyết của Grice từ triết học) và cố gắng thể hiện chúng trong các hệ thống.

#### Suy nghĩ cuối cùng

* Các lĩnh vực bà thấy thú vị nhất bao gồm tóm tắt trừu tượng, phân tích ngôn ngữ từ **cộng đồng đa dạng** (diverse communities), giải quyết **thiên vị** (bias), và hiểu **nghĩa ngôn ngữ/thực dụng** (pragmatic meaning) (như cảm xúc và ý định).
* Bà đề cập đến một bài báo yêu thích cũ về "Floating Constraints on Lexical Choice", lưu ý rằng sự **kiểm soát** (control) là thứ còn thiếu trong các mô hình tạo sinh học sâu (deep learning generation) ngày nay.
* BDY nhấn mạnh rằng dù học sâu đã có những tiến bộ lớn, NLP vẫn còn nhiều hướng đi. Bà muốn thấy nhiều **công việc liên ngành** (interdisciplinary work) hơn và muốn các nhà nghiên cứu nhìn vào **dữ liệu và đầu ra** (data and output) chứ không chỉ là **số** (numbers).