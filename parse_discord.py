import json
from bs4 import BeautifulSoup
from datetime import datetime

html_path = "../Discord_chat_Wed Jan 01 2025 00_00_00 GMT+0530 (India Standard Time)_Sun Jun 07 2026 00_00_00 GMT+0530 (India Standard Time).html"

with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

messages = []
chat_content = soup.find("ul", class_="chatContent")
if chat_content:
    for li in chat_content.find_all("li", recursive=False):
        name_span = li.find("span", class_="chatName")
        time_span = li.find("span", class_="time")
        
        name = name_span.get_text(strip=True) if name_span else "Unknown"
        time_str = time_span.get_text(strip=True) if time_span else ""
        
        # Message text is usually the last p tag or just inside titleInfo
        title_info = li.find("div", class_="titleInfo")
        text = ""
        images = []
        if title_info:
            paragraphs = title_info.find_all("p", recursive=False)
            if len(paragraphs) > 1:
                # First is time/name, subsequent ones are text
                text_parts = [p.get_text(" ", strip=True) for p in paragraphs[1:]]
                text = "\n".join(text_parts)
            
            # Images
            imgs = title_info.find_all("img")
            for img in imgs:
                src = img.get("src", "")
                if src: images.append(src)
                
        messages.append({
            "name": name,
            "time": time_str,
            "text": text,
            "images": images
        })

with open("discord_transcript.json", "w", encoding="utf-8") as f:
    json.dump(messages, f, indent=2)

print(f"Extracted {len(messages)} messages.")
