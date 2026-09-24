"""Compare Delphi token IDs with the publisher's Hugging Face tokenizer.json."""
import subprocess, sys
from pathlib import Path
from tokenizers import Tokenizer
ref=Tokenizer.from_file('reference/tokenizer.json')
texts=['Hello world!', 'Quanto e 2 + 2? Responda apenas o numero.',
       'a\u0301, Ol\u00e1! a\u00e7\u00e3o caf\u00e9 \U0001f600\n123456',
       "We're testing\t spaces\n\n", '<|im_start|>user\nOi<|im_end|>\n',
       '  leading and trailing  ', '\u4f60\u597d\uff0c\u4e16\u754c!',
       '<think>\n</think>\n\n', '0 123 456789 3.14']
model = sys.argv[1] if len(sys.argv)>1 else 'D:/Qwen3.5-0.8B.gguf'
for i,s in enumerate(texts):
    Path('build/tokenizer-input.txt').write_bytes(s.encode('utf-8'))
    p=subprocess.run(['build/Qwen35Smoke.exe',model,'build/tokenizer-input.txt','--tokenize-file'],capture_output=True,check=True)
    actual=[int(x) for x in p.stdout.splitlines()[0].decode().split(',') if x]
    expected=ref.encode(s,add_special_tokens=False).ids
    assert actual==expected,(i,actual,expected)
    print('tokenizer case',i,'PASS')
