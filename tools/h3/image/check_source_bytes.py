"""Reject raw NUL bytes and physical newlines inside ordinary C++ string/character literals."""
import argparse
from pathlib import Path


def check(path):
    raw=path.read_bytes()
    if b"\0" in raw: raise ValueError(f"{path}: raw NUL byte")
    if path.suffix not in (".cpp",".hpp",".hip",".h",".inc"): return
    s=raw.decode("utf-8-sig"); i=0
    while i<len(s):
        if s.startswith("//",i):
            end=s.find("\n",i+2);i=len(s) if end<0 else end+1
        elif s.startswith("/*",i):
            end=s.find("*/",i+2)
            if end<0: raise ValueError(f"{path}: unfinished comment")
            i=end+2
        elif s.startswith('R"',i):
            paren=s.find("(",i+2)
            if paren<0: raise ValueError(f"{path}: malformed raw literal")
            delimiter=s[i+2:paren]; end=s.find(")"+delimiter+'"',paren+1)
            if end<0: raise ValueError(f"{path}: unfinished raw literal")
            i=end+len(delimiter)+2
        elif s[i] in ('"',"'"):
            if s[i]=="'" and i>0 and s[i-1].isdigit() and i+1<len(s) and s[i+1].isdigit():
                i+=1;continue
            quote=s[i];start=i;i+=1
            while i<len(s) and s[i]!=quote:
                if s[i] in "\r\n": raise ValueError(f"{path}:{s.count(chr(10),0,start)+1}: raw newline in literal")
                if s[i]=="\\":
                    i+=1
                    if i<len(s) and s[i] in "\r\n": raise ValueError(f"{path}: physical newline escape in literal")
                i+=1
            if i==len(s): raise ValueError(f"{path}: unfinished literal")
            i+=1
        else: i+=1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files",nargs="+",type=Path)
    a=p.parse_args()
    for file in a.files: check(file)
    print(f"source byte/literal checks passed: {len(a.files)} files")
