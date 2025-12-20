# **Tổng Hợp Kỹ Thuật Xử Lý Chuỗi và Câu trong NLP**

## Mục Lục
1. [Tách từ và Tokenization](#1-tách-từ-và-tokenization)
2. [Xử lý và Làm sạch Chuỗi](#2-xử-lý-và-làm-sạch-chuỗi)
3. [Đếm và Thống kê Từ](#3-đếm-và-thống-kê-từ)
4. [Lọc và Trích xuất Từ](#4-lọc-và-trích-xuất-từ)
5. [Kiểm tra và Xác thực Chuỗi](#5-kiểm-tra-và-xác-thực-chuỗi)
6. [Chuyển đổi Format](#6-chuyển-đổi-format)
7. [Sửa lỗi Chính tả](#7-sửa-lỗi-chính-tả)
8. [Thao tác với List và Dictionary](#8-thao-tác-với-list-và-dictionary)

---

## 1. Tách từ và Tokenization

### 1.1. Tokenization với NLTK
**Mục đích**: Tách câu thành các từ và dấu câu riêng biệt

```python
import nltk
from nltk.tokenize import word_tokenize

# Download corpus nếu chưa có
nltk.download('punkt_tab')

text = "This is a sample sentence. It's a simple task."
tokens = word_tokenize(text)
# Kết quả: ['This', 'is', 'a', 'sample', 'sentence', '.', 'It', "'s", 'a', 'simple', 'task', '.']
```

**Giải thích**:
- `word_tokenize()` tách văn bản thành tokens (từ và dấu câu)
- Xử lý được các trường hợp đặc biệt như viết tắt ("It's" → "It", "'s")
- Giữ lại dấu câu như các token riêng biệt

### 1.2. Tách từ đơn giản với split()
**Mục đích**: Tách câu thành các từ theo khoảng trắng

```python
text = "Hello world from Python"
words = text.split()  # Tách theo khoảng trắng
# Kết quả: ['Hello', 'world', 'from', 'Python']

# Tách theo ký tự cụ thể
text = "apple,banana,orange"
fruits = text.split(',')
# Kết quả: ['apple', 'banana', 'orange']
```

**Giải thích**:
- `split()` không có tham số: tách theo khoảng trắng và loại bỏ khoảng trắng thừa
- `split(' ')`: tách theo đúng một khoảng trắng
- `split(delimiter)`: tách theo ký tự phân cách cụ thể

---

## 2. Xử lý và Làm sạch Chuỗi

### 2.1. Chuyển đổi chữ hoa/thường
```python
text = "Cat DOG bird"

# Chuyển tất cả sang chữ thường
lower_text = text.lower()  # "cat dog bird"

# Chuyển tất cả sang chữ HOA
upper_text = text.upper()  # "CAT DOG BIRD"

# Chuyển chữ cái đầu thành hoa
title_text = text.title()  # "Cat Dog Bird"

# Chuyển chữ cái đầu câu thành hoa
capitalize_text = text.capitalize()  # "Cat dog bird"
```

**Giải thích**:
- `lower()`: chuyển toàn bộ sang chữ thường (dùng khi so sánh không phân biệt hoa/thường)
- `upper()`: chuyển toàn bộ sang chữ HOA
- `title()`: chữ cái đầu mỗi từ viết hoa
- `capitalize()`: chỉ chữ cái đầu tiên của chuỗi viết hoa

### 2.2. Loại bỏ khoảng trắng thừa
```python
text = "  hello    world     this    is   a   test   "

# Cách 1: Dùng split() và join()
cleaned = ' '.join(text.split())
# Kết quả: "hello world this is a test"

# Cách 2: Loại bỏ khoảng trắng đầu/cuối
trimmed = text.strip()

# Loại bỏ khoảng trắng bên trái
left_trimmed = text.lstrip()

# Loại bỏ khoảng trắng bên phải
right_trimmed = text.rstrip()
```

**Giải thích**:
- `strip()`: loại bỏ khoảng trắng ở đầu và cuối chuỗi
- `split()` + `join()`: loại bỏ tất cả khoảng trắng thừa (kể cả giữa các từ)

### 2.3. Loại bỏ ký tự đặc biệt
```python
# Cách 1: Giữ lại chỉ chữ và số
text = "Having lots of fun #goa #vaction @Beachbay restro:"
cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
words = [word for word in cleaned.split() if word]
# Kết quả: ['Having', 'lots', 'of', 'fun', 'goa', 'vaction', 'Beachbay', 'restro']

# Cách 2: Loại bỏ dấu câu với string.punctuation
import string
text = "Hello, world! How are you?"
no_punct = text.translate(str.maketrans('', '', string.punctuation))
# Kết quả: "Hello world How are you"
```

**Giải thích**:
- `isalnum()`: kiểm tra ký tự là chữ cái hoặc số
- `isspace()`: kiểm tra ký tự là khoảng trắng
- `string.punctuation`: chứa tất cả dấu câu (!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~)
- `translate()` + `maketrans()`: cách hiệu quả để loại bỏ nhiều ký tự

---

## 3. Đếm và Thống kê Từ

### 3.1. Đếm tần suất từ
```python
text = "This is a sample text. This text is for testing purposes."
words = text.split()

# Cách 1: Dùng dictionary thủ công
freq = {}
for word in words:
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1

# Cách 2: Dùng get() (gọn hơn)
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1

# Cách 3: Dùng Counter (đơn giản nhất)
from collections import Counter
freq = Counter(words)
```

**Giải thích**:
- `dict.get(key, default)`: trả về giá trị của key, nếu không có trả về default
- `Counter`: class chuyên dụng để đếm tần suất

### 3.2. Sắp xếp theo tần suất
```python
# Sắp xếp theo tần suất giảm dần
sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

# Sắp xếp theo tần suất tăng dần
sorted_freq = sorted(freq.items(), key=lambda x: x[1])

# Sắp xếp theo alphabet
sorted_alpha = dict(sorted(freq.items()))
```

**Giải thích**:
- `sorted()`: trả về list đã sắp xếp
- `key=lambda x: x[1]`: sắp xếp theo giá trị (tần suất)
- `reverse=True`: sắp xếp giảm dần
- `dict.items()`: trả về các cặp (key, value)

### 3.3. Tính độ dài trung bình của từ
```python
text = "The quick brown fox"
words = text.split()

# Tính tổng độ dài tất cả các từ
total_length = sum(len(word) for word in words)

# Tính trung bình
average_length = total_length / len(words)

# Làm tròn 2 chữ số thập phân
average_length = round(average_length, 2)
```

**Giải thích**:
- `sum()`: tính tổng các phần tử
- `len(word)`: độ dài của từ
- `round(num, n)`: làm tròn đến n chữ số thập phân

---

## 4. Lọc và Trích xuất Từ

### 4.1. Lọc từ theo điều kiện
```python
# Lọc từ kết thúc bằng 'lia'
flowers = ['camellia', 'pendulum', 'dahlia', 'hostas']
lia_words = [word for word in flowers if word.endswith('lia')]
# Kết quả: ['camellia', 'dahlia']

# Lọc từ bắt đầu bằng 'h'
h_words = [word for word in flowers if word.startswith('h')]

# Lọc từ dài hơn 8 ký tự
long_words = [word for word in flowers if len(word) > 8]
```

**Giải thích**:
- `endswith(suffix)`: kiểm tra chuỗi kết thúc bằng suffix
- `startswith(prefix)`: kiểm tra chuỗi bắt đầu bằng prefix
- List comprehension: cách ngắn gọn để tạo list mới từ list cũ

### 4.2. Lọc từ chứa điều kiện phức tạp
```python
# Lọc từ chứa ít nhất 1 chữ số và chỉ chứa chữ-số (không có dấu)
text = "I bought 2apples and 3bananas but item#4 was missing"
words = text.split()
result = [word for word in words 
          if any(char.isdigit() for char in word) and word.isalnum()]
# Kết quả: ['2apples', '3bananas']
```

**Giải thích**:
- `any()`: trả về True nếu ít nhất 1 phần tử thỏa điều kiện
- `isdigit()`: kiểm tra ký tự là chữ số
- `isalnum()`: kiểm tra ký tự là chữ cái hoặc số (không có ký tự đặc biệt)

### 4.3. Trích xuất từ duy nhất (unique)
```python
text = "machine learning is fun and machine learning improves skills"
words = text.split()

# Giữ thứ tự xuất hiện đầu tiên
seen = set()
unique_words = []
for word in words:
    word_lower = word.lower()
    if word_lower not in seen:
        seen.add(word_lower)
        unique_words.append(word_lower)

# Không quan tâm thứ tự
unique_words = list(set(words))
```

**Giải thích**:
- `set()`: tập hợp không chứa phần tử trùng lặp
- Dùng set để track các từ đã thấy, giữ thứ tự bằng list riêng

### 4.4. Trích xuất email theo pattern
```python
text = "Please contact us at contact@fpt.edu.vn for further information. You can also give feedback at feedback@gmail.com"
words = text.split()

# Trích xuất email FPT
fpt_emails = [word for word in words if '@fpt' in word]
# Kết quả: ['contact@fpt.edu.vn']

# Hoặc dùng regex cho pattern phức tạp
import re
all_emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
```

**Giải thích**:
- `in`: kiểm tra substring có trong chuỗi
- Regex pattern phức tạp hơn nhưng chính xác hơn

---

## 5. Kiểm tra và Xác thực Chuỗi

### 5.1. Kiểm tra Palindrome (từ đối xứng)
```python
def is_palindrome(word):
    cleaned = word.lower()
    return cleaned == cleaned[::-1]

# Tìm tất cả palindrome trong text
text = "madam Anna racecar hello level world Deed"
words = text.split()
palindromes = []
seen = set()

for word in words:
    word_lower = word.lower()
    if word_lower not in seen and word_lower == word_lower[::-1]:
        seen.add(word_lower)
        palindromes.append(word_lower)
# Kết quả: ['madam', 'anna', 'racecar', 'level', 'deed']
```

**Giải thích**:
- `[::-1]`: slice notation để đảo ngược chuỗi
- So sánh chuỗi với bản đảo ngược của nó

### 5.2. Kiểm tra Isogram (không có chữ lặp)
```python
def is_isogram(sentence):
    # Chỉ giữ các chữ cái, chuyển về lowercase
    cleaned = ''.join(char.lower() for char in sentence if char.isalpha())
    
    # Kiểm tra có chữ nào lặp không
    seen = set()
    for char in cleaned:
        if char in seen:
            return False
        seen.add(char)
    return True

# Hoặc cách ngắn gọn hơn
def is_isogram_short(sentence):
    cleaned = ''.join(char.lower() for char in sentence if char.isalpha())
    return len(cleaned) == len(set(cleaned))
```

**Giải thích**:
- `isalpha()`: kiểm tra ký tự là chữ cái
- So sánh độ dài chuỗi với độ dài set (set loại bỏ trùng lặp)

### 5.3. Kiểm tra Python Identifier hợp lệ
```python
import keyword

def is_valid_identifier(name):
    # Kiểm tra là keyword của Python
    if keyword.iskeyword(name):
        return False
    
    # Kiểm tra ký tự đầu (phải là chữ hoặc _)
    if name[0].isdigit():
        return False
    
    # Kiểm tra các ký tự (chỉ chữ, số, _)
    for char in name:
        if not (char.isalnum() or char == '_'):
            return False
    
    return True

# Hoặc dùng method có sẵn
name.isidentifier() and not keyword.iskeyword(name)
```

**Giải thích**:
- Python identifier: tên biến, hàm, class hợp lệ
- Quy tắc: bắt đầu bằng chữ/_, theo sau là chữ/số/_, không là keyword

---

## 6. Chuyển đổi Format

### 6.1. Chuyển snake_case sang camelCase
```python
def snake_to_camel(snake_str):
    words = snake_str.split('_')
    # Từ đầu giữ nguyên lowercase
    # Các từ sau capitalize chữ cái đầu
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

# Ví dụ
result = snake_to_camel("hello_world_from_python")
# Kết quả: "helloWorldFromPython"
```

**Giải thích**:
- `capitalize()`: viết hoa chữ cái đầu, lowercase phần còn lại
- Từ đầu giữ nguyên, các từ sau capitalize và nối lại

### 6.2. Chuyển sang Title Case
```python
text = "  hello    world     this    is   a   test   "

# Cách 1: Dùng title()
title_text = text.title()

# Cách 2: Tự xử lý (loại bỏ khoảng trắng thừa)
words = text.split()
title_words = [word.capitalize() for word in words]
result = ' '.join(title_words)
# Kết quả: "Hello World This Is A Test"
```

### 6.3. Đảo ngược từ trong câu
```python
def reverse_words(sentence):
    words = sentence.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)

# Ví dụ
text = "Hello world from Python"
result = reverse_words(text)
# Kết quả: "olleH dlrow morf nohtyP"
```

**Giải thích**:
- Tách câu thành từ, đảo ngược từng từ, nối lại

---

## 7. Sửa lỗi Chính tả

### 7.1. Dùng TextBlob
```python
from textblob import TextBlob

# Cài đặt: pip install textblob
# Download corpus: python -m textblob.download_corpora

def correct_spelling(text):
    blob = TextBlob(text)
    corrected = blob.correct()
    return str(corrected)

# Ví dụ
text = "He is a gret person. He beleives in bod"
result = correct_spelling(text)
# Kết quả: "He is a great person. He believes in god"
```

**Giải thích**:
- TextBlob sử dụng thuật toán probability-based để sửa lỗi
- `correct()`: tự động sửa các từ sai chính tả

---

## 8. Thao tác với List và Dictionary

### 8.1. Slicing - Lấy phần tử
```python
words = ['She', 'received', 'the', 'news', 'of', 'discovery']

# Lấy 3 từ đầu
first_three = words[:3]  # ['She', 'received', 'the']

# Lấy từ vị trí 2 đến 4
middle = words[2:5]  # ['the', 'news', 'of']

# Lấy từ cuối
last_word = words[-1]  # 'discovery'

# Lấy 2 từ cuối
last_two = words[-2:]  # ['of', 'discovery']

# Đảo ngược list
reversed_list = words[::-1]

# Lấy mọi phần tử thứ 2
every_second = words[::2]  # ['She', 'the', 'of']
```

**Giải thích**:
- `list[start:end]`: lấy từ start đến end-1
- `list[:n]`: lấy n phần tử đầu
- `list[-n:]`: lấy n phần tử cuối
- `list[::step]`: lấy mỗi step phần tử

### 8.2. Sắp xếp và loại bỏ duplicate
```python
# Sắp xếp theo tần suất tăng dần, loại bỏ duplicate
words = ['table']*10 + ['chair']*9
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1

# Sắp xếp theo tần suất
sorted_words = sorted(freq.items(), key=lambda x: x[1])
result = [word for word, count in sorted_words]
# Kết quả: ['chair', 'table']
```

### 8.3. Lambda function và sorting
```python
# Sắp xếp theo nhiều tiêu chí
data = [('apple', 5), ('banana', 2), ('orange', 5), ('grape', 2)]

# Sắp xếp theo count, sau đó theo tên
sorted_data = sorted(data, key=lambda x: (x[1], x[0]))

# Sắp xếp theo count giảm dần, tên tăng dần
sorted_data = sorted(data, key=lambda x: (-x[1], x[0]))
```

**Giải thích**:
- `lambda x: expression`: hàm ẩn danh (anonymous function)
- `lambda x: (x[1], x[0])`: sắp xếp theo phần tử thứ 2, sau đó phần tử thứ 1
- `-x[1]`: đảo dấu để sắp xếp giảm dần

---

## Các Kỹ Thuật Nâng Cao

### 1. List Comprehension
```python
# Syntax cơ bản
[expression for item in iterable if condition]

# Ví dụ
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]

# Với điều kiện
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]

# Nested
matrix = [[i+j for j in range(3)] for i in range(3)]
```

### 2. Generator Expression
```python
# Tiết kiệm bộ nhớ cho tập dữ liệu lớn
large_data = range(1000000)
sum_squares = sum(x**2 for x in large_data)
```

### 3. String Methods Tổng hợp
```python
text = "Hello World"

# Kiểm tra
text.isalpha()      # Chỉ chữ cái
text.isdigit()      # Chỉ chữ số
text.isalnum()      # Chữ cái hoặc số
text.isspace()      # Chỉ khoảng trắng
text.islower()      # Tất cả chữ thường
text.isupper()      # Tất cả chữ HOA

# Tìm kiếm
text.find('World')       # Trả về index, -1 nếu không tìm thấy
text.index('World')      # Trả về index, raise error nếu không tìm thấy
text.count('l')          # Đếm số lần xuất hiện
'World' in text          # Kiểm tra substring

# Thay thế
text.replace('World', 'Python')  # Thay thế substring
```

### 4. Regular Expression (Regex)
```python
import re

text = "Email: test@example.com, Phone: 123-456-7890"

# Tìm tất cả email
emails = re.findall(r'\S+@\S+', text)

# Tìm tất cả số điện thoại
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)

# Thay thế pattern
cleaned = re.sub(r'\d+', 'XXX', text)  # Thay tất cả số bằng XXX

# Match và extract
match = re.search(r'(\w+)@(\w+\.\w+)', text)
if match:
    username = match.group(1)
    domain = match.group(2)
```

---

## Cheat Sheet - Các Phương thức Thường Dùng

### String Methods
| Method | Mô tả | Ví dụ |
|--------|-------|-------|
| `split(sep)` | Tách chuỗi theo separator | `"a,b,c".split(',')` → `['a','b','c']` |
| `join(list)` | Nối list thành chuỗi | `','.join(['a','b'])` → `'a,b'` |
| `strip()` | Loại bỏ khoảng trắng đầu/cuối | `" hi ".strip()` → `'hi'` |
| `lower()` | Chuyển thành chữ thường | `"HI".lower()` → `'hi'` |
| `upper()` | Chuyển thành chữ HOA | `"hi".upper()` → `'HI'` |
| `capitalize()` | Viết hoa chữ đầu | `"hi".capitalize()` → `'Hi'` |
| `title()` | Viết hoa mỗi từ | `"hi world".title()` → `'Hi World'` |
| `replace(old, new)` | Thay thế substring | `"cat".replace('c','b')` → `'bat'` |
| `startswith(prefix)` | Kiểm tra bắt đầu | `"hello".startswith('he')` → `True` |
| `endswith(suffix)` | Kiểm tra kết thúc | `"hello".endswith('lo')` → `True` |

### List Methods
| Method | Mô tả | Ví dụ |
|--------|-------|-------|
| `append(item)` | Thêm phần tử cuối | `[1,2].append(3)` → `[1,2,3]` |
| `extend(list)` | Nối list | `[1,2].extend([3,4])` → `[1,2,3,4]` |
| `insert(i, item)` | Chèn tại vị trí i | `[1,3].insert(1,2)` → `[1,2,3]` |
| `remove(item)` | Xóa phần tử đầu tiên | `[1,2,1].remove(1)` → `[2,1]` |
| `pop(i)` | Xóa và trả về phần tử | `[1,2,3].pop(1)` → 2, list còn `[1,3]` |
| `sort()` | Sắp xếp tại chỗ | `[3,1,2].sort()` → `[1,2,3]` |
| `sorted()` | Trả về list mới | `sorted([3,1,2])` → `[1,2,3]` |

### Dictionary Methods
| Method | Mô tả | Ví dụ |
|--------|-------|-------|
| `get(key, default)` | Lấy giá trị an toàn | `d.get('a', 0)` |
| `keys()` | Lấy tất cả keys | `d.keys()` |
| `values()` | Lấy tất cả values | `d.values()` |
| `items()` | Lấy cặp (key, value) | `d.items()` |

---

## Tips và Best Practices

1. **Xử lý Case-Insensitive**: Luôn chuyển về `.lower()` khi so sánh
2. **Loại bỏ Duplicates**: Dùng `set()` để track, giữ order bằng list riêng
3. **Đếm tần suất**: Dùng `Counter` hoặc `dict.get(key, 0)`
4. **Sắp xếp phức tạp**: Dùng `lambda` với tuple để sắp xếp nhiều tiêu chí
5. **Xử lý chuỗi rỗng**: Luôn kiểm tra `if word` trước khi xử lý
6. **Performance**: Dùng list comprehension thay vì loop khi có thể
7. **Regular Expression**: Cho pattern matching phức tạp
8. **NLTK/TextBlob**: Cho NLP tasks nâng cao

---

## Thư viện Hữu ích

```python
# Tokenization và NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Spell checking
from textblob import TextBlob

# Regex
import re

# Đếm tần suất
from collections import Counter

# Xử lý string
import string

# Kiểm tra keyword Python
import keyword
```

---

## NLTK Library Cheatsheet

### 1. Cài đặt và Download Corpus
```python
# Cài đặt NLTK
# pip install nltk

import nltk

# Download tất cả corpus (khuyến nghị cho lần đầu)
nltk.download('all')

# Download từng corpus cụ thể (tiết kiệm dung lượng)
nltk.download('punkt')           # Tokenizer models
nltk.download('punkt_tab')       # Tokenizer models (version mới)
nltk.download('stopwords')       # Stop words
nltk.download('wordnet')         # WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # Named Entity Chunker
nltk.download('words')           # Word list
nltk.download('vader_lexicon')   # Sentiment analysis
nltk.download('brown')           # Brown corpus
nltk.download('reuters')         # Reuters corpus
```

### 2. Tokenization

#### 2.1. Word Tokenization
```python
from nltk.tokenize import word_tokenize, wordpunct_tokenize, TreebankWordTokenizer

# Word tokenize - Phương pháp cơ bản (khuyến nghị)
text = "Hello, world! It's a beautiful day. Cost: $10.50"
tokens = word_tokenize(text)
# ['Hello', ',', 'world', '!', 'It', "'s", 'a', 'beautiful', 'day', '.', 'Cost', ':', '$', '10.50']

# WordPunct tokenize - Tách theo dấu câu
tokens = wordpunct_tokenize(text)
# ['Hello', ',', 'world', '!', 'It', "'", 's', 'a', 'beautiful', 'day', '.', 'Cost', ':', '$', '10', '.', '50']

# TreebankWord tokenizer - Chuẩn Penn Treebank
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)
```

#### 2.2. Sentence Tokenization
```python
from nltk.tokenize import sent_tokenize

# Tách văn bản thành câu
text = "Hello world. How are you? I am fine!"
sentences = sent_tokenize(text)
# ['Hello world.', 'How are you?', 'I am fine!']

# Xử lý viết tắt
text = "Dr. Smith works at U.S. Bank. He lives in N.Y."
sentences = sent_tokenize(text)
# ['Dr. Smith works at U.S. Bank.', 'He lives in N.Y.']
```

#### 2.3. Tokenization Nâng cao
```python
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer, BlanklineTokenizer

# Regex tokenizer - Tùy chỉnh pattern
tokenizer = RegexpTokenizer(r'\w+')  # Chỉ lấy chữ và số
tokens = tokenizer.tokenize("Hello, world! 123")
# ['Hello', 'world', '123']

# Whitespace tokenizer - Tách theo khoảng trắng
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize("Hello   world\t\nPython")
# ['Hello', 'world', 'Python']

# Blankline tokenizer - Tách theo dòng trống
tokenizer = BlanklineTokenizer()
paragraphs = tokenizer.tokenize("Para 1\n\nPara 2\n\nPara 3")
```

### 3. Stop Words

```python
from nltk.corpus import stopwords

# Lấy stop words tiếng Anh
stop_words = set(stopwords.words('english'))
# {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', ...}

# Lọc bỏ stop words
text = "This is a sample sentence with some stop words"
words = word_tokenize(text.lower())
filtered = [word for word in words if word not in stop_words]
# ['sample', 'sentence', 'stop', 'words']

# Stop words các ngôn ngữ khác
print(stopwords.fileids())  # List tất cả ngôn ngữ
# ['arabic', 'azerbaijani', 'bengali', 'catalan', 'chinese', 'danish', ...]

# Tiếng Việt (nếu có)
# vietnamese_stops = set(stopwords.words('vietnamese'))
```

### 4. Stemming - Rút gọn từ về gốc

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Porter Stemmer (phổ biến nhất)
porter = PorterStemmer()
words = ["running", "runs", "ran", "runner", "easily", "fairly"]
stems = [porter.stem(word) for word in words]
# ['run', 'run', 'ran', 'runner', 'easili', 'fairli']

# Lancaster Stemmer (aggressive hơn)
lancaster = LancasterStemmer()
stems = [lancaster.stem(word) for word in words]
# ['run', 'run', 'ran', 'run', 'easy', 'fair']

# Snowball Stemmer (hỗ trợ nhiều ngôn ngữ)
snowball = SnowballStemmer('english')
stems = [snowball.stem(word) for word in words]
# ['run', 'run', 'ran', 'runner', 'easili', 'fair']

# Snowball hỗ trợ các ngôn ngữ
print(SnowballStemmer.languages)
# ['arabic', 'danish', 'dutch', 'english', 'finnish', ...]
```

### 5. Lemmatization - Chuẩn hóa từ

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Lemmatize (cần chỉ định part-of-speech)
words = ["running", "runs", "ran", "better", "geese"]

# Mặc định là noun
lemmas = [lemmatizer.lemmatize(word) for word in words]
# ['running', 'run', 'ran', 'better', 'goose']

# Chỉ định part-of-speech
lemmatizer.lemmatize("running", pos='v')  # 'run' (verb)
lemmatizer.lemmatize("running", pos='n')  # 'running' (noun)
lemmatizer.lemmatize("better", pos='a')   # 'good' (adjective)

# POS tags: n=noun, v=verb, a=adjective, r=adverb
```

### 6. Part-of-Speech (POS) Tagging

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# POS tagging
text = "Python is a powerful programming language"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
# [('Python', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('powerful', 'JJ'), 
#  ('programming', 'NN'), ('language', 'NN')]

# Các POS tags phổ biến:
# NN  - Noun (danh từ)
# NNP - Proper noun (danh từ riêng)
# VB  - Verb (động từ)
# VBZ - Verb, 3rd person singular present
# JJ  - Adjective (tính từ)
# DT  - Determiner (từ hạn định)
# RB  - Adverb (trạng từ)
# PRP - Personal pronoun (đại từ nhân xưng)

# Lọc theo POS tag
nouns = [word for word, pos in tagged if pos.startswith('NN')]
# ['Python', 'programming', 'language']
```

### 7. Named Entity Recognition (NER)

```python
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "Apple Inc. is located in Cupertino, California. Tim Cook is the CEO."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)

# Trích xuất named entities
for chunk in entities:
    if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))
# ORGANIZATION Apple Inc.
# GPE Cupertino
# GPE California
# PERSON Tim Cook

# Entity types:
# PERSON - Người
# ORGANIZATION - Tổ chức
# GPE - Địa danh (Countries, cities, states)
# DATE - Ngày tháng
# TIME - Thời gian
```

### 8. N-grams

```python
from nltk import ngrams
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is amazing"
tokens = word_tokenize(text)

# Bigrams (2-grams)
bigrams = list(ngrams(tokens, 2))
# [('Natural', 'Language'), ('Language', 'Processing'), 
#  ('Processing', 'is'), ('is', 'amazing')]

# Trigrams (3-grams)
trigrams = list(ngrams(tokens, 3))
# [('Natural', 'Language', 'Processing'), 
#  ('Language', 'Processing', 'is'), 
#  ('Processing', 'is', 'amazing')]

# Unigrams (1-gram)
unigrams = list(ngrams(tokens, 1))

# N-grams tùy chỉnh
fourgrams = list(ngrams(tokens, 4))
```

### 9. Frequency Distribution

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize

text = "the quick brown fox jumps over the lazy dog the fox"
tokens = word_tokenize(text.lower())

# Tạo frequency distribution
fdist = FreqDist(tokens)

# Lấy tần suất của từ
print(fdist['the'])  # 3
print(fdist['fox'])  # 2

# Top N từ phổ biến nhất
most_common = fdist.most_common(3)
# [('the', 3), ('fox', 2), ('quick', 1)]

# Tất cả từ và tần suất
for word, freq in fdist.items():
    print(f"{word}: {freq}")

# Vẽ biểu đồ frequency
fdist.plot(30)  # Top 30 từ

# Tổng số tokens
print(fdist.N())

# Số từ unique
print(len(fdist))
```

### 10. Chunking - Phân tích cú pháp

```python
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "the little yellow dog barked at the cat"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Định nghĩa grammar pattern
grammar = "NP: {<DT>?<JJ>*<NN>}"
parser = RegexpParser(grammar)

# Parse
tree = parser.parse(tagged)
print(tree)
# (S
#   (NP the/DT little/JJ yellow/JJ dog/NN)
#   barked/VBD
#   at/IN
#   (NP the/DT cat/NN))

# Trích xuất noun phrases
for subtree in tree.subtrees():
    if subtree.label() == 'NP':
        print(' '.join(word for word, tag in subtree.leaves()))
# the little yellow dog
# the cat
```

### 11. Sentiment Analysis với VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Khởi tạo
sia = SentimentIntensityAnalyzer()

# Phân tích sentiment
text1 = "This movie is absolutely fantastic and amazing!"
scores1 = sia.polarity_scores(text1)
# {'neg': 0.0, 'neu': 0.359, 'pos': 0.641, 'compound': 0.8516}

text2 = "This movie is terrible and boring."
scores2 = sia.polarity_scores(text2)
# {'neg': 0.491, 'neu': 0.509, 'pos': 0.0, 'compound': -0.5574}

# Compound score: -1 (rất negative) đến +1 (rất positive)
# > 0.05: positive
# < -0.05: negative
# -0.05 to 0.05: neutral

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'
```

### 12. Corpus Access - Truy cập dữ liệu

```python
from nltk.corpus import brown, reuters, gutenberg, wordnet

# Brown Corpus
brown.words()[:10]
brown.sents()[:3]
brown.categories()

# Reuters Corpus
reuters.fileids()
reuters.words('training/9865')

# Gutenberg Corpus (sách cổ điển)
gutenberg.fileids()
# ['austen-emma.txt', 'austen-persuasion.txt', ...]
emma = gutenberg.words('austen-emma.txt')

# WordNet - Lexical database
synsets = wordnet.synsets('dog')
print(synsets[0].definition())  # 'a member of the genus Canis...'
print(synsets[0].examples())    # ['the dog barked all night']
```

### 13. Text Similarity

```python
from nltk.metrics import edit_distance, jaccard_distance
from nltk.util import ngrams

# Edit distance (Levenshtein distance)
dist = edit_distance("hello", "hallo")  # 1
dist = edit_distance("python", "pithon")  # 1

# Jaccard distance (cho sets)
set1 = set(ngrams("hello", 2))
set2 = set(ngrams("hallo", 2))
similarity = 1 - jaccard_distance(set1, set2)
```

### 14. Collocations - Cụm từ hay xuất hiện cùng nhau

```python
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize

text = "natural language processing and machine learning are subfields of artificial intelligence"
tokens = word_tokenize(text.lower())

# Tìm bigram collocations
finder = BigramCollocationFinder.from_words(tokens)
bigram_measures = BigramAssocMeasures()

# Top 5 collocations
collocations = finder.nbest(bigram_measures.pmi, 5)
# [('natural', 'language'), ('language', 'processing'), ...]
```

### 15. Useful Functions

```python
# Tokenize và lowercase
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return tokens

# Remove stop words và punctuation
from nltk.corpus import stopwords
import string

def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens 
              if token not in stop_words 
              and token not in string.punctuation]
    return tokens

# Lemmatize với POS tagging
from nltk.stem import WordNetLemmatizer

def lemmatize_with_pos(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Convert POS tag to WordNet format
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'  # adjective
        elif tag.startswith('V'):
            return 'v'  # verb
        elif tag.startswith('N'):
            return 'n'  # noun
        elif tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
              for word, pos in tagged]
    return lemmas
```

### 16. NLTK Quick Reference Table

| Chức năng | Module/Function | Ví dụ |
|-----------|----------------|-------|
| Word tokenize | `word_tokenize()` | `word_tokenize("Hi there")` |
| Sentence tokenize | `sent_tokenize()` | `sent_tokenize("Hi. How are you?")` |
| Stemming | `PorterStemmer()` | `porter.stem("running")` |
| Lemmatization | `WordNetLemmatizer()` | `lemmatizer.lemmatize("better", 'a')` |
| POS tagging | `pos_tag()` | `pos_tag(tokens)` |
| NER | `ne_chunk()` | `ne_chunk(tagged)` |
| N-grams | `ngrams()` | `ngrams(tokens, 2)` |
| Frequency | `FreqDist()` | `FreqDist(tokens)` |
| Stop words | `stopwords.words()` | `stopwords.words('english')` |
| Sentiment | `SentimentIntensityAnalyzer()` | `sia.polarity_scores(text)` |

### 17. Common NLTK Patterns

```python
# Pattern 1: Tokenize và clean
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def process_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stop words và punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens 
              if t not in stop_words 
              and t not in string.punctuation 
              and t.isalpha()]
    
    return tokens

# Pattern 2: Stem hoặc Lemmatize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def normalize_text(tokens, method='lemma'):
    if method == 'stem':
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

# Pattern 3: Pipeline hoàn chỉnh
def nlp_pipeline(text):
    # 1. Tokenize
    tokens = word_tokenize(text.lower())
    
    # 2. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 3. Remove punctuation
    tokens = [t for t in tokens if t.isalpha()]
    
    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens
```

---

## NLTK Best Practices

1. **Always download required corpus first**: `nltk.download('punkt')`, `nltk.download('stopwords')`
2. **Use word_tokenize over split()**: Xử lý tốt hơn với punctuation và contractions
3. **Lemmatization > Stemming**: Lemmatization cho kết quả chính xác hơn
4. **POS tagging before lemmatization**: Cải thiện độ chính xác của lemmatization
5. **Use set() for stop words**: Tìm kiếm nhanh hơn với set
6. **FreqDist for frequency analysis**: Hiệu quả hơn manual counting
7. **VADER for quick sentiment**: Không cần training, phù hợp cho social media text

---

**Ghi chú**: Tài liệu này tổng hợp từ các đề thi NLP301c các kỳ FA24, FA25, SP24, SU24, SU25. Tất cả code examples đều được test và chạy thành công.
