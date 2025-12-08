import os
import urllib.request
import sys

BASE_URL = "https://github.com/ParamTatva-org/sanskrit/releases/download/Alpha1.001/"
FILES = [
    "model.pt",
    "video_encoder.pt",
    "vision_encoder.pt",
    "vocab.json"
]

DEST_DIR = os.path.dirname(os.path.abspath(__file__))

def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = f"\rDownloading... {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB)"
        sys.stdout.write(s)
        sys.stdout.flush()
        if read_so_far >= total_size:
            sys.stdout.write("\n")

def main():
    print(f"Downloading weights to {DEST_DIR}...")
    
    for filename in FILES:
        file_path = os.path.join(DEST_DIR, filename)
        if os.path.exists(file_path):
            print(f"File {filename} already exists. Skipping.")
            continue
            
        url = BASE_URL + filename
        print(f"Downloading {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, file_path, reporthook)
            print(f"Successfully downloaded {filename}")
        except urllib.error.HTTPError as e:
             print(f"Failed to download {filename}: HTTP {e.code} {e.reason}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
