# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 2: Part of Speech Tagging and Hidden Markov Models**
---
### **Part of Speech Tagging**
---

#### Tổng quan về Tuần 2

- Bạn sẽ tìm hiểu các ứng dụng của nó và cách tính toán **độ chính xác** (accuracy) của trình gắn thẻ.
- Tuần này, bạn sẽ học về **Chuỗi Markov** (Markov Chains), **Mô hình Markov ẩn** (Hidden Markov Models - HMMs), và **thuật toán Viterbi** (Viterbi algorithm) để tạo các thẻ này.


#### Gắn thẻ Phần của Lời nói (POS Tagging) là gì?

- **Phần của lời nói** (Part of speech) đề cập đến loại từ hoặc thuật ngữ từ vựng trong một ngôn ngữ (ví dụ: **danh từ, động từ, tính từ, trạng từ, đại từ, giới từ**).
- Vì việc viết đầy đủ tên các thuật ngữ này (ví dụ: "Tại sao không học một cái gì đó?") rất **"cồng kềnh"** (cumbersome), bạn sẽ sử dụng một biểu diễn ngắn gọi là **thẻ** (tags).
- Quá trình gán các thẻ này cho các từ trong câu được gọi là **gắn thẻ phần của lời nói** (hoặc **POS tagging**).

#### Ứng dụng của POS Tagging

Thẻ POS mô tả cấu trúc và giúp đưa ra các giả định về ngữ nghĩa. Các ứng dụng bao gồm:

1.  **Xác định các thực thể được đặt tên** (Identifying named entities).
    * Ví dụ: Trong "Tháp Eiffel nằm ở Paris", "Tháp Eiffel" và "Paris" là các thực thể được đặt tên.
2.  **Độ phân giải đồng tham chiếu** (Co-reference resolution).
    * Ví dụ: Trong "Tháp Eiffel... **nó** cao 324 mét", POS tagging giúp suy ra "nó" (it) đề cập đến "Tháp Eiffel".
3.  **Nhận dạng giọng nói** (Speech recognition).
    * Được sử dụng để kiểm tra xem một chuỗi từ có xác suất cao hay không.
> Gán nhãn từ loại (POS) là quá trình gán một từ loại cho một từ. Khi làm vậy, bạn sẽ học được những điều sau:

- Markov Chains
- Hidden Markov Models
- Viterbi algorithm

> Dưới đây là một ví dụ cụ thể:

![01_Part_of_Speech_Tagging](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/01_Part_of_Speech_Tagging.png)

> Bạn có thể sử dụng gán nhãn từ loại cho:

- Xác định thực thể được đặt tên
- Nhận dạng giọng nói
- Giải quyết tham chiếu

> Bạn có thể sử dụng xác suất xuất hiện gần nhau của các nhãn POS để đưa ra kết quả hợp lý nhất.
---
### **Markov Chains**
---



---
### **Markov Chains and POS Tags**
---



---
### **Hidden Markov Models**
---


---
### **Calculating Probabilities**
---



---
### **Populating the Transition Matrix**
---


---
### **Populating the Emission Matrix**
---


---
### **The Viterbi Algorithm**
---



---
### **Viterbi: Initialization**
---


---
### **Viterbi: Forward Pass**
---

---
### **Viterbi: Backward Pass**
---
