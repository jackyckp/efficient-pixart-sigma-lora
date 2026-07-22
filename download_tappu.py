"""
download_tappu.py — Scrape & download ink-wash paintings from tappu.com
=======================================================================
Crawls the gallery at https://tappu.com/bokuga/category/gallery/ (17 pages),
downloads one representative image per article, translates the Japanese
description to English using Google Translate, and saves both the image and
the translated caption to organised sub-folders.

Output structure
----------------
  data/ink/
    animal/   ← articles whose tags match Japanese animal keywords
    plant/    ← articles whose tags match Japanese plant keywords
    others/   ← everything else

  Each article produces two files:
    <num>.jpg  (or .png, etc.) — the article's hero / eye-catch image
    <num>.txt                  — English translation of the article body

  Image numbers come from "#NNN" in the article title; articles without a
  number get a fallback name like "no_num_1001".

Dependencies
------------
  pip install requests beautifulsoup4 deep-translator

Usage
-----
  # Run from the project root — images are saved to data/ink/
  python download_tappu.py

Notes
-----
  - A 1-second polite delay is inserted between each article request.
  - If translation fails the original Japanese text is saved instead.
  - The script is idempotent: re-running overwrites existing files.
  - Total pages scraped is controlled by `total_pages` in main() (default 17).
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

# Create directories
base_dir = "data/ink"
categories = ["animal", "plant", "others"]
for cat in categories:
    os.makedirs(os.path.join(base_dir, cat), exist_ok=True)

animal_keywords = ['虫', '動物', '鳥', 'カエル', '犬', '猫', '魚', '蟹', '海老', '蛙']
plant_keywords = ['花', '樹木', '植物', '葉', '果実', '野菜', '春', '夏', '秋', '冬', '根', '実', '草', '木', '蘭', '竹', '菊', '梅', '桜', '蓮', '松', '牡丹']

def get_category(tags):
    for tag in tags:
        for ak in animal_keywords:
            if ak in tag: return "animal"
    for tag in tags:
        for pk in plant_keywords:
            if pk in tag: return "plant"
    return "others"

def process_detail_page(url, title, num_str):
    try:
        res = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(res.content, "html.parser")
    
    # tags
    tag_list = soup.find("div", class_="entry-categories-tags")
    tags = []
    if tag_list:
        tag_elements = tag_list.find_all("a")
        tags = [t.get_text(strip=True) for t in tag_elements]
        
    cat = get_category(tags)
    
    # image
    img_tag = soup.find("img", class_="eye-catch-image")
    img_url = None
    if img_tag:
        img_url = img_tag.get("src")
    else:
        # Fallback to other wp-post-image
        img_tag = soup.find("img", class_="wp-post-image")
        if img_tag:
            img_url = img_tag.get("src")
            
    if not img_url:
        print(f"No image found for {url}")
        return
        
    # Description
    entry_content = soup.find("div", class_="entry-content")
    desc_text = ""
    if entry_content:
        desc_paragraphs = entry_content.find_all("p")
        desc_text = "\n".join([p.get_text(strip=True) for p in desc_paragraphs if p.get_text(strip=True)])
        
    # Translate
    desc_en = ""
    if desc_text:
        try:
            translator = GoogleTranslator(source='ja', target='en')
            desc_en = translator.translate(desc_text)
        except Exception as e:
            print(f"Translation failed for {url}: {e}")
            desc_en = desc_text # fallback to jp if fails
            
    # Save
    cat_dir = os.path.join(base_dir, cat)
    
    # Extension
    ext = os.path.splitext(img_url)[1]
    if not ext or '?' in ext:
        ext = ".jpg"
        
    img_path = os.path.join(cat_dir, f"{num_str}{ext}")
    txt_path = os.path.join(cat_dir, f"{num_str}.txt")
    
    # download image
    try:
        img_res = requests.get(img_url, timeout=10)
        with open(img_path, "wb") as f:
            f.write(img_res.content)
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")
        return
        
    # save text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(desc_en)
        
    print(f"Saved {num_str} to {cat}")

def main():
    total_pages = 17
    global_idx = 1000
    for page in range(1, total_pages + 1):
        print(f"Processing page {page}")
        url = f"https://tappu.com/bokuga/category/gallery/page/{page}/"
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            articles = soup.find_all("a", class_="entry-card-wrap")
            for a in articles:
                title_tag = a.find("h2")
                href = a.get("href")
                if title_tag and href:
                    title = title_tag.get_text(strip=True)
                    m = re.search(r'#(\d+)', title)
                    if m:
                        num_str = m.group(1)
                    else:
                        global_idx += 1
                        num_str = f"no_num_{global_idx}"
                        
                    print(f"Found {num_str} -> {href}")
                    process_detail_page(href, title, num_str)
                    time.sleep(1) # be polite
        except Exception as e:
            print(f"Error on page {page}: {e}")

if __name__ == "__main__":
    main()
