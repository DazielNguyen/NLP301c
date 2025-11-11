# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 1: Autocorrect and Minimum Edit Distance** -> Consist of 2 small labs and 1 Asgm
---
### **Overview**
---
#### Mục tiêu của tuần 2 
- Hiểu được autocorrect là gì? 
- Xây dựng được model để tự động sửa lỗi chính tả từ 
- Hiểu được về chỉnh sửa khoảng cách tối thiểu giữa các chuỗi từ (Minimum edit distance)
- Lập trình động (Dynamic Programming) dùng các thuật toán chính sửa khoảng cách tối thiểu -> Giải quyết các vấn đề về tối ưu hóa 
> Overview Tuần 2 

![01_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/01_Overview.png)

---
### **Autocorrect (Tự động sửa lỗi)**
---
- Autocorrect là một ứng dụng thay đổi các từ sai chính tả thành các từ chính xác
- Autocorrect hoạt động như thế nào? 

    1. Xác định một từ không chính xác. Như sai chính tả....
    2. Tìm các chuỗi từ 1 đến n để chỉnh sửa khoảng cách xa
    3. Lọc các chuỗi từ thực được đánh vần chính tả chính xác
    4. Tính toán xác suất từ, cho biết khả năng mỗi từ xuất hiện trong ngữ cảnh khác nhau và chọn ứng cử viên có khả năng thay thế nhất.

---
### **Building the model**
---

#### Bước 1: Xác định từ sai chính tả (Identify the misspelled word)

- **Làm thế nào?** Kiểm tra xem một từ có tồn tại trong **từ điển** (dictionary) hay không.
- Nếu từ đó **không có** trong từ điển, nó sẽ bị đánh dấu là sai chính tả và cần sửa.
- **Lưu ý:** Phương pháp này chỉ xác định **lỗi chính tả** (spelling errors), không phải **lỗi ngữ cảnh** (contextual errors). (Ví dụ: "hươu" (deer) là một từ đúng chính tả, nó sẽ vượt qua bộ lọc ngay cả khi ngữ cảnh có thể sai).


#### Bước 2: Tìm các chuỗi trong khoảng cách n chỉnh sửa (Find strings n edit distance away)

- **Chỉnh sửa (Edit):** Là một thao tác (operation) được thực hiện trên một chuỗi để thay đổi nó.
- **Khoảng cách chỉnh sửa n (n edit distance):** Là số lượng *n* thao tác cần thiết để biến một chuỗi thành chuỗi khác.
- Bốn loại chỉnh sửa cơ bản được sử dụng:

    1.  **Chèn (Insert):** Thêm một chữ cái vào bất kỳ vị trí nào.
        * *Ví dụ:* `to` $\rightarrow$ `top` (chèn 'p') hoặc `two` (chèn 'w').
    2.  **Xóa (Delete):** Loại bỏ một chữ cái.
        * *Ví dụ:* `hat` $\rightarrow$ `ha` (xóa 't'), `at` (xóa 'h'), `ht` (xóa 'a').
    3.  **Hoán đổi (Switch):** Hoán đổi hai chữ cái **liền kề** (adjacent).
        * *Ví dụ:* `eta` $\rightarrow$ `eat` (hoán đổi 't' và 'a') hoặc `tea` (hoán đổi 'e' và 't').
        * *Lưu ý:* Điều này *không* bao gồm các chữ cái không liền kề (ví dụ: `eta` $\rightarrow$ `ate` không phải là một lần "hoán đổi").
    4.  **Thay thế (Replace):** Thay đổi một chữ cái này thành một chữ cái khác.
        * *Ví dụ:* `jaw` $\rightarrow$ `jar` (thay 'w' bằng 'r').

- Đối với tự động sửa lỗi, *n* thường là 1 đến 3. Bằng cách kết hợp các chỉnh sửa này, bạn tạo ra một danh sách tất cả các chuỗi có thể có trong vòng *n* khoảng cách chỉnh sửa.


#### Bước 3: Lọc ứng cử viên (Filter candidates)

- Nhiều chuỗi được tạo ra ở Bước 2 không phải là từ có thật.
- **Làm thế nào?** Bạn **lọc (filter)** danh sách ứng cử viên bằng cách kiểm tra lại chúng với **từ điển** (giống như ở Bước 1).
- Nếu một chuỗi ứng cử viên **không** xuất hiện trong từ điển, nó sẽ bị **xóa (remove)** khỏi danh sách.
- Sau bước này, bạn chỉ còn lại một danh sách các từ có thật, đúng chính tả.

#### Tóm tắt

- Ba bước đầu tiên: 
    + (1) Xác định lỗi
    + (2) Tạo các chỉnh sửa, và 
    + (3) Lọc các ứng cử viên. Bước thứ tư và cũng là bước cuối cùng—**tính toán xác suất**

---
### **Building the model II**
---


---
### **Minimum edit distance**
---



---
### **Minimum edit distance algorithmn**
---


---
### **Minimum edit distance algorithmn II**
---


---
### **Minimum edit distance algorithmn III**
---



---
### **Minimum edit distance III**
---


