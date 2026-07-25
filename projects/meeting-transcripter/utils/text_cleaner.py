import re

def clean_transcript(text):
    # Remove timestamps like [00:04:23] or (00:04:23)
    text = re.sub(r'[\[\(]\d{2}:\d{2}:\d{2}[\]\)]', '', text)
    
    # Standardize speaker labels to "Name:" format
    text = re.sub(r'(?i)^([\w\s]+)\s*[-–]\s*', r'\1: ', text, flags=re.MULTILINE)
    
    # Remove filler words
    fillers = r'\b(um|uh|you know|like|basically|literally|actually|so uh|uh so)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    
    # Remove multiple spaces left behind
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text