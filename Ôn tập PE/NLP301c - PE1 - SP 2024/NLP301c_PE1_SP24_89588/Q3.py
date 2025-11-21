import re

def extract_fpt_emails(text):
    # Regular expression pattern to match email addresses with fpt.edu.vn domain
    pattern = r'\b[\w\.-]+@fpt\.edu\.vn\b'
    matches = re.findall(pattern, text)
    
    return matches

input_text = "Please contact us at contact@fpt.edu.vn for further information. You can also give feedback at feedback@gmail.com"
fpt_emails = extract_fpt_emails(input_text)
print(fpt_emails)