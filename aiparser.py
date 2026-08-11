import os
import csv
import re
import html
import json
import time
import random
import urllib3
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests as cffi_requests
from playwright.sync_api import sync_playwright
from openai import OpenAI

# ---------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------
INPUT_CSV = "cleaned_senior_centers.csv"
OUTPUT_CSV = "scraped_contacts2.csv"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Exact Blacklist
EXACT_BLACKLIST = {
    "info@seniorcenters.com",
    "newsletter@seniorcenters.com",
}

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

cf_solver = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
)


def fetch_with_playwright(url: str, timeout: int = 15000) -> str:
    """Renders dynamic JavaScript/Weebly/Gov pages via Playwright."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
            page.wait_for_timeout(2000)
            content = page.content()
            browser.close()
            return content
        except Exception:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                page.wait_for_timeout(1500)
                content = page.content()
                browser.close()
                return content
            except Exception as e:
                browser.close()
                raise Exception(f"Playwright fetch failed: {e}")


def safe_get(url: str, headers: dict, referer: str = None, timeout: int = 8):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    req_headers = dict(headers)
    if referer:
        req_headers["Referer"] = referer

    try:
        response = cffi_requests.get(url, headers=req_headers, timeout=timeout, impersonate="chrome120", verify=True)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass

    try:
        response = cf_solver.get(url, headers=req_headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass

    raise Exception("Fast HTTP blocked (403 / Cloudflare)")


def fetch_webpage_smart(url: str, headers: dict, referer: str = None) -> str:
    # Use Playwright for .gov or dynamic CMS sites immediately
    if any(domain in url.lower() for domain in [".gov", "weebly", "winooski", "lakestevens"]):
        return fetch_with_playwright(url)
    try:
        return safe_get(url, headers=headers, referer=referer, timeout=8)
    except Exception:
        print(f"   ⚠️ HTTP Blocked on '{url}'. Launching Playwright browser fallback...")
        return fetch_with_playwright(url)


def filter_and_clean_emails(raw_email_input) -> str:
    if not raw_email_input:
        return ""

    candidates = []
    if isinstance(raw_email_input, list):
        for item in raw_email_input:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                candidates.extend([str(v) for v in item.values()])
            else:
                candidates.append(str(item))
    elif isinstance(raw_email_input, str):
        candidates = [raw_email_input]
    elif isinstance(raw_email_input, dict):
        candidates = [str(v) for v in raw_email_input.values()]

    all_tokens = []
    for candidate in candidates:
        parts = re.split(r"[,\s;]+", candidate)
        all_tokens.extend([p.strip().lower() for p in parts if p.strip()])

    valid_emails = []
    for email in all_tokens:
        if email in EXACT_BLACKLIST:
            continue
        if EMAIL_REGEX.match(email):
            valid_emails.append(email)

    unique_emails = list(dict.fromkeys(valid_emails))
    return ", ".join(unique_emails)


def find_all_contact_subpages(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Expanded keywords specifically catching contact_us, staff, and directory routes."""
    target_keywords = [
        'contact', 'contact_us', 'contact-us', 'about', 'team', 'staff', 
        'directory', 'find us', 'find-us', 'find_us', 'location', 
        'reach-us', 'senior', 'seniors', 'departments'
    ]
    
    subpages = []
    for a in soup.find_all('a', href=True):
        text = a.get_text().strip().lower()
        href = a['href'].lower()
        
        # Normalize text and href to catch underscores like contact_us
        normalized_href = href.replace('_', '-').replace('/', ' ')
        normalized_text = text.replace('_', '-')
        
        if any(kw in normalized_text or kw in href or kw in normalized_href for kw in target_keywords):
            full_link = urljoin(base_url, a['href'])
            if full_link.rstrip('/') != base_url.rstrip('/') and full_link not in subpages:
                subpages.append(full_link)
    return subpages


def clean_html_to_text(html_content: str) -> str:
    """Extracts raw text and places mailto/embedded addresses at the top for Ollama."""
    if not html_content:
        return ""
        
    decoded_html = html.unescape(html_content)
    soup = BeautifulSoup(decoded_html, "html.parser")

    mailto_emails = []
    for a in soup.find_all('a', href=True):
        if 'mailto:' in a['href'].lower():
            email_part = a['href'].lower().replace('mailto:', '').split('?')[0].strip()
            if email_part:
                mailto_emails.append(email_part)

    for element in soup(["script", "style", "svg", "noscript", "iframe"]):
        element.decompose()

    text = soup.get_text(separator=" ")
    clean_text = re.sub(r"\s+", " ", text).strip()

    if mailto_emails:
        mailto_str = ", ".join(set(mailto_emails))
        return f"EMBEDDED EMAILS: {mailto_str} | CONTENT: {clean_text}"

    return clean_text


def extract_official_website(soup: BeautifulSoup, directory_url: str, center_name: str = "") -> str:
    social_and_ignored = [
        'seniorcenters.com', 'facebook.com', 'twitter.com', 'x.com', 
        'instagram.com', 'tiktok.com', 'linkedin.com', 'youtube.com', 
        'pinterest.com', 'snapchat.com', 'reddit.com', 'google.com', 
        'yelp.com', 'wikipedia.org', 'mapquest.com'
    ]

    if not center_name:
        h1 = soup.find('h1')
        center_name = h1.get_text().strip() if h1 else (soup.title.get_text().strip() if soup.title else "")

    stop_words = {'senior', 'center', 'centers', 'the', 'and', 'of', 'at', 'in', 'for'}
    center_tokens = [w.lower() for w in re.findall(r'\b\w+\b', center_name) if w.lower() not in stop_words and len(w) > 2]

    scored_candidates = []
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        text = a.get_text().strip().lower()
        full_url = urljoin(directory_url, href)

        if not full_url.startswith(('http://', 'https://')) or any(d in full_url.lower() for d in social_and_ignored):
            continue

        score = 0.0
        full_url_clean = re.sub(r'https?://(www\.)?', '', full_url.lower())
        for token in center_tokens:
            if token in text: score += 15.0
            if token in full_url_clean: score += 10.0

        if text:
            score += SequenceMatcher(None, center_name.lower(), text).ratio() * 20.0

        if any(kw in text for kw in ['visit website', 'official website', 'official site']):
            score += 10.0

        if '.gov' in full_url.lower(): score += 25.0
        elif '.org' in full_url.lower(): score += 15.0

        if score > 0:
            scored_candidates.append((score, full_url))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return scored_candidates[0][1]


def parse_all_with_ollama(snippet_text: str) -> dict:
    """Pure Ollama extraction focused on staying within context limits."""
    trimmed_snippet = snippet_text[:3500]

    prompt = f"Website Text:\n{trimmed_snippet}\n\nExtract contact details into JSON."

    try:
        response = client.chat.completions.create(
            model="llama3.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract all contact details from the text. "
                        "Return ONLY a JSON object with keys: "
                        "'emails' (array of strings), 'city' (string), 'state' (string), 'phone' (string), 'hours' (string), 'staff_names' (array of strings)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_reply = response.choices[0].message.content.strip()
        raw_reply = re.sub(r'^`{3}(?:json)?\n?|`{3}$', '', raw_reply, flags=re.MULTILINE).strip()
        data = json.loads(raw_reply)

        return {
            "emails": data.get("emails", []),
            "city": data.get("city", ""),
            "state": data.get("state", ""),
            "phone": data.get("phone", ""),
            "hours": data.get("hours", ""),
            "staff_names": data.get("staff_names", []),
        }
    except Exception:
        return {"emails": [], "city": "", "state": "", "phone": "", "hours": "", "staff_names": []}


def process_single_row(row, idx, total, url_col, name_col, col_to_remove, headers_req):
    raw_url = row.get(url_col, "").strip()
    center_name = row.get(name_col, "").strip() if name_col else ""
    clean_target = ("https://" + raw_url) if raw_url and not raw_url.startswith(("http://", "https://")) else raw_url

    output_row = dict(row)
    
    if col_to_remove and col_to_remove in output_row:
        del output_row[col_to_remove]

    if not clean_target:
        output_row.update({"Official_Website": "", "Extracted_Emails": "", "City": "", "State": "", "Phone": "", "Hours": "", "Staff_Names": "", "Status": "No URL"})
        return output_row

    time.sleep(round(random.uniform(1.0, 2.0), 2))

    try:
        # 1. Fetch Directory Page
        dir_html = fetch_webpage_smart(clean_target, headers=headers_req)
        dir_text = clean_html_to_text(dir_html)
        dir_soup = BeautifulSoup(dir_html, "html.parser")

        # 2. Extract Official Website Link
        official_url = row.get("override_url", "").strip() or extract_official_website(dir_soup, clean_target, center_name) or clean_target

        off_text = ""
        if official_url and official_url != clean_target:
            time.sleep(round(random.uniform(1.0, 2.0), 2))
            try:
                off_html = fetch_webpage_smart(official_url, headers=headers_req, referer=clean_target)
                off_text = clean_html_to_text(off_html)
                off_soup = BeautifulSoup(off_html, "html.parser")

                # DEEP CRAWL SUBPAGES (e.g., /contact_us or /staff)
                subpages = find_all_contact_subpages(off_soup, official_url)
                for sub_url in subpages[:3]:
                    try:
                        print(f"   🔍 Fetching subpage ({sub_url})")
                        sub_html = fetch_webpage_smart(sub_url, headers=headers_req, referer=official_url)
                        off_text += " " + clean_html_to_text(sub_html)
                    except Exception:
                        continue

            except Exception:
                print(f"   ⚠️ Could not fetch official site ({official_url}). Relying on directory HTML.")

        # 3. Combine text snippets for Ollama
        combined_text = f"{off_text} {dir_text}".strip()

        # 4. Pure Ollama Extraction
        llm_res = parse_all_with_ollama(combined_text)

        extracted_emails_str = filter_and_clean_emails(llm_res.get("emails", []))
        city = str(llm_res.get("city", "")).strip()
        state = str(llm_res.get("state", "")).strip()
        phone = str(llm_res.get("phone", "")).strip()
        hours = str(llm_res.get("hours", "")).strip()
        
        # SAFE TYPE CONVERSION FOR STAFF_NAMES (Fixes 'list' object has no attribute 'strip')
        staff_names = llm_res.get("staff_names", "")
        if isinstance(staff_names, list):
            staff_names = ", ".join([str(s).strip() for s in staff_names if s])
        elif isinstance(staff_names, str):
            staff_names = staff_names.strip()
        else:
            staff_names = str(staff_names)

        email_count = len(extracted_emails_str.split(",")) if extracted_emails_str else 0
        print(f"[{idx}/{total}] ✅ {official_url} | Ollama Emails: {email_count} ({extracted_emails_str}) | City: '{city}'")

        output_row.update({
            "Official_Website": official_url,
            "Extracted_Emails": extracted_emails_str,
            "City": city,
            "State": state,
            "Phone": phone,
            "Hours": hours,
            "Staff_Names": staff_names,
            "Status": "Success"
        })

    except Exception as e:
        print(f"[{idx}/{total}] ❌ Failed: {clean_target} ({e})")
        output_row.update({"Official_Website": clean_target, "Extracted_Emails": "", "City": "", "State": "", "Phone": "", "Hours": "", "Staff_Names": "", "Status": f"Failed: {e}"})

    return output_row


def process_csv_links(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV, max_workers: int = 2):
    if not os.path.exists(input_csv):
        print(f"❌ Input file '{input_csv}' not found.")
        return

    with open(input_csv, mode="r", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    header_map = {h.lower().strip(): h for h in headers}
    url_col = header_map.get("url") or header_map.get("link")
    name_col = header_map.get("name") or header_map.get("center_name") or header_map.get("title")

    col_to_remove = None
    for h in headers:
        clean_h = h.lower().strip()
        if clean_h in ["email", "original_email", "emails", "contact_email"]:
            col_to_remove = h
            break

    new_fields = ["Official_Website", "Extracted_Emails", "City", "State", "Phone", "Hours", "Staff_Names", "Status"]
    output_headers = [h for h in headers if h != col_to_remove] + new_fields

    headers_req = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }

    print(f"🚀 Reading '{input_csv}' -> Writing to '{output_csv}' for {len(rows)} links...\n")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_headers)
        writer.writeheader()
        outfile.flush()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_row, row, idx, len(rows), url_col, name_col, col_to_remove, headers_req)
                for idx, row in enumerate(rows, start=1)
            ]

            for future in as_completed(futures):
                result_row = future.result()
                writer.writerow(result_row)
                outfile.flush()

    print(f"\n🎉 Extraction Complete! Saved to '{output_csv}'")


if __name__ == "__main__":
    process_csv_links(INPUT_CSV, OUTPUT_CSV, max_workers=2)