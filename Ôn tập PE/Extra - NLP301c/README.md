# 🎓 Practice Exercises - Extra
## Bài Tập Luyện Tập Bổ Sung NLP301c

Đây là bộ 30 bài tập luyện tập bổ sung cho môn NLP301c, tập trung vào **xử lý chuỗi (String)**, **mảng (Array)** và **từ điển (Dictionary)** trong Python.

---

## 📚 Cấu Trúc Bài Tập

### 🔤 Phần 1: Xử Lý Chuỗi (Q5-Q10) - 6 câu
| Câu | Tiêu đề | Độ khó | Điểm |
|-----|---------|--------|------|
| Q5  | Đảo ngược từng từ trong câu | Dễ | 2 |
| Q6  | Tìm từ palindrome | Dễ | 2 |
| Q7  | Chuyển đổi snake_case sang camelCase | Trung bình | 3 |
| Q8  | Đếm nguyên âm và phụ âm | Trung bình | 3 |
| Q9  | Tìm substring xuất hiện nhiều nhất | Khó | 4 |
| Q10 | Loại bỏ ký tự trùng lặp liên tiếp | Khó | 4 |

### 📊 Phần 2: Xử Lý Mảng (Q11-Q20) - 10 câu
| Câu | Tiêu đề | Độ khó | Điểm |
|-----|---------|--------|------|
| Q11 | Tìm cặp số có tổng bằng target | Dễ | 2 |
| Q12 | Xoay mảng trái/phải | Dễ | 2 |
| Q13 | Tìm phần tử xuất hiện nhiều nhất | Trung bình | 3 |
| Q14 | Merge nhiều mảng đã sắp xếp | Trung bình | 3 |
| Q15 | Tìm số thiếu trong dãy | Trung bình | 3 |
| Q16 | Chia mảng thành 2 phần tổng bằng nhau | Khó | 4 |
| Q17 | Tìm dãy số liên tiếp dài nhất | Khó | 4 |
| Q18 | Loại bỏ duplicates giữ thứ tự | Khó | 4 |
| Q19 | Tìm intersection của nhiều mảng | Khó | 4 |
| Q20 | Sliding window maximum | Khó | 4 |

### 📖 Phần 3: Xử Lý Dictionary (Q21-Q30) - 10 câu
| Câu | Tiêu đề | Độ khó | Điểm |
|-----|---------|--------|------|
| Q21 | Merge dictionaries với tổng values | Dễ | 2 |
| Q22 | Đảo ngược key-value | Dễ | 2 |
| Q23 | Lọc dictionary theo điều kiện | Trung bình | 3 |
| Q24 | Nhóm items theo value | Trung bình | 3 |
| Q25 | Đếm tần suất values | Trung bình | 3 |
| Q26 | Flatten nested dictionary | Khó | 4 |
| Q27 | Top K keys có values lớn nhất | Khó | 4 |
| Q28 | Dictionary comprehension có điều kiện | Khó | 4 |
| Q29 | So sánh và tìm differences | Khó | 4 |
| Q30 | Tạo inverted index | Khó | 4 |

---

## 🎯 Cách Sử Dụng

### 1. **Đọc đề bài**
Mỗi file Python chứa:
- Mô tả bài toán chi tiết
- Input/Output mẫu
- Hints giải quyết
- Test cases

### 2. **Implement solution**
```python
def function_name(parameters):
    """
    TODO: Implement this function
    1. Step 1
    2. Step 2
    3. Step 3
    """
    pass  # Replace with your code
```

### 3. **Chạy test**
```bash
# Activate virtual environment
source ../npl-env/bin/activate

# Run specific question
python Q5.py
python Q11.py
python Q21.py
```

### 4. **Verify kết quả**
So sánh output với expected output trong comments

---

## 💡 Tips Học Tập

### String Processing
- Sử dụng string methods: `split()`, `join()`, `lower()`, `upper()`
- String slicing: `str[start:end:step]`
- Character checking: `isalpha()`, `isdigit()`, `isalnum()`
- List comprehension cho string manipulation

### Array Processing
- Hiểu về slicing: `arr[start:end]`, `arr[:k]`, `arr[-k:]`
- Set operations: `union`, `intersection`, `difference`
- Dictionary cho frequency counting
- Two-pointer technique

### Dictionary Processing
- Dictionary methods: `keys()`, `values()`, `items()`
- Dictionary comprehension: `{k: v for k, v in ...}`
- `get()` method với default value
- Nested dictionary navigation
- `collections.defaultdict` cho advanced use cases

---

## 📝 Đánh Giá & Điểm Số

| Mức độ | Số câu | Tổng điểm |
|--------|--------|-----------|
| Dễ (2 điểm) | 6 | 12 |
| Trung bình (3 điểm) | 9 | 27 |
| Khó (4 điểm) | 15 | 60 |
| **TỔNG** | **30** | **99** |

---

## 🚀 Lộ Trình Học

### Week 1: String Processing (Q5-Q10)
- [ ] Q5: Reverse words
- [ ] Q6: Find palindromes
- [ ] Q7: Snake to camel case
- [ ] Q8: Count vowels/consonants
- [ ] Q9: Most frequent substring
- [ ] Q10: Remove consecutive duplicates

### Week 2: Array Processing Part 1 (Q11-Q15)
- [ ] Q11: Pairs with sum
- [ ] Q12: Rotate array
- [ ] Q13: Most frequent element
- [ ] Q14: Merge sorted arrays
- [ ] Q15: Find missing number

### Week 3: Array Processing Part 2 (Q16-Q20)
- [ ] Q16: Split equal sum
- [ ] Q17: Longest consecutive sequence
- [ ] Q18: Remove duplicates
- [ ] Q19: Find intersection
- [ ] Q20: Sliding window max

### Week 4: Dictionary Processing Part 1 (Q21-Q25)
- [ ] Q21: Merge dictionaries
- [ ] Q22: Invert dictionary
- [ ] Q23: Filter dictionary
- [ ] Q24: Group by value
- [ ] Q25: Value frequency

### Week 5: Dictionary Processing Part 2 (Q26-Q30)
- [ ] Q26: Flatten dictionary
- [ ] Q27: Top K items
- [ ] Q28: Dictionary comprehension
- [ ] Q29: Compare dictionaries
- [ ] Q30: Inverted index

---

## 📚 Tài Liệu Tham Khảo

- [Python String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [Python List Operations](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
- [Python Collections](https://docs.python.org/3/library/collections.html)

---

## ✅ Checklist Hoàn Thành

Đánh dấu ✅ khi hoàn thành mỗi bài:

**String Processing:**
- [ ] Q5 - [ ] Q6 - [ ] Q7 - [ ] Q8 - [ ] Q9 - [ ] Q10

**Array Processing:**
- [ ] Q11 - [ ] Q12 - [ ] Q13 - [ ] Q14 - [ ] Q15
- [ ] Q16 - [ ] Q17 - [ ] Q18 - [ ] Q19 - [ ] Q20

**Dictionary Processing:**
- [ ] Q21 - [ ] Q22 - [ ] Q23 - [ ] Q24 - [ ] Q25
- [ ] Q26 - [ ] Q27 - [ ] Q28 - [ ] Q29 - [ ] Q30

---

## 🎉 Good Luck!

Chúc bạn học tốt và đạt điểm cao trong kỳ thi NLP301c!

---

**Created:** 16/12/2025  
**Author:** GitHub Copilot  
**Version:** 1.0
