"""
Sales Brochure Generator
=========================

Given a company name and its website URL, this script:
  1. Scrapes the landing page and collects its links.
  2. Asks the LLM to pick which links are actually relevant for a brochure
     (about page, careers, products, etc.) and discard junk (mailto:, #, terms).
  3. Fetches the contents of those relevant pages.
  4. Generates a short markdown sales/recruitment brochure and streams it out.

Requires:
  - scraper.py  (provides fetch_website_links and fetch_website_contents)
  - a .env file in the same folder containing:  OPENAI_API_KEY=sk-...
"""

import os
import json
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from openai import OpenAI

from scraper import fetch_website_links, fetch_website_contents


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv(override=True)

MODEL = "gpt-4o-mini"   # cheap + good enough for this task

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Create a .env file next to this script "
        "with a line like:  OPENAI_API_KEY=sk-..."
    )

openai = OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Step 1: clean the raw links from the scraper
# ---------------------------------------------------------------------------

def get_clean_links(url):
    """
    fetch_website_links returns raw href values, which include relative paths
    ('/about'), anchors ('#'), and mailto: links. Resolve them to absolute URLs
    and drop the junk so the LLM (and requests.get) only ever sees real pages.
    """
    raw_links = fetch_website_links(url)
    clean = []
    seen = set()
    for href in raw_links:
        href = href.strip()
        # skip anchors, mailto/tel, javascript pseudo-links
        if href.startswith("#") or href.lower().startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(url, href)          # resolves '/about' -> 'https://site.com/about'
        if not urlparse(absolute).scheme.startswith("http"):
            continue
        absolute = absolute.split("#")[0]       # strip fragments
        if absolute not in seen:
            seen.add(absolute)
            clean.append(absolute)
    return clean


# ---------------------------------------------------------------------------
# Step 2: let the LLM choose the relevant links
# ---------------------------------------------------------------------------

link_system_prompt = (
    "You are provided with a list of links found on a webpage. "
    "Decide which of the links are most relevant to include in a brochure about "
    "the company, such as an About page, Company page, or Careers/Jobs pages.\n"
    "You should respond in JSON as in this example:\n"
    "{\n"
    '    "links": [\n'
    '        {"type": "about page", "url": "https://full.url/goes/here/about"},\n'
    '        {"type": "careers page", "url": "https://another.full.url/careers"}\n'
    "    ]\n"
    "}\n"
    "Do not include Terms of Service, Privacy, or email links."
)


def get_links_user_prompt(url, links):
    prompt = f"Here is the list of links on the website of {url} - "
    prompt += (
        "please decide which of these are relevant web links for a brochure "
        "about the company. Respond with the full https URL in JSON format. "
        "Do not include Terms of Service, Privacy, or email links.\n"
    )
    prompt += "Links (some might be relative links):\n"
    prompt += "\n".join(links)
    return prompt


def select_relevant_links(url):
    links = get_clean_links(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url, links)},
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    return json.loads(result)


# ---------------------------------------------------------------------------
# Step 3: gather all the page contents
# ---------------------------------------------------------------------------

def get_all_details(url):
    """Landing page + every relevant link the LLM chose, concatenated."""
    print("  -> fetching landing page...", flush=True)
    result = "Landing page:\n"
    result += fetch_website_contents(url)

    print("  -> asking the model which links are relevant...", flush=True)
    links = select_relevant_links(url)
    print("  -> relevant links chosen:", links, flush=True)

    for link in links.get("links", []):
        print(f"  -> fetching {link['url']} ...", flush=True)
        result += f"\n\n{link['type']}:\n"
        try:
            result += fetch_website_contents(link["url"])
        except Exception as e:
            result += f"(could not fetch {link['url']}: {e})"
    return result


# ---------------------------------------------------------------------------
# Step 4: build the brochure
# ---------------------------------------------------------------------------

brochure_system_prompt = (
    "You are an assistant that analyzes the contents of several relevant pages "
    "from a company website and creates a short, engaging brochure about the "
    "company for prospective customers, investors, and recruits. Respond in "
    "markdown. Include details of company culture, customers, and careers/jobs "
    "if you have the information.\n"
    "Output the markdown content directly. Do NOT wrap your whole response in "
    "triple backticks or a ```markdown code fence."
)


def strip_code_fences(text):
    """
    If the model wrapped the whole brochure in a ```markdown ... ``` fence,
    remove it so the saved file renders as real markdown (not a code block).
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]                                   # drop opening ``` / ```markdown
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                              # drop closing ```
        t = "\n".join(lines).strip()
    return t


def get_brochure_user_prompt(company_name, url):
    prompt = f"You are looking at a company called: {company_name}\n"
    prompt += (
        "Here are the contents of its landing page and other relevant pages; "
        "use this information to build a short brochure of the company in markdown.\n"
    )
    prompt += get_all_details(url)
    return prompt[:5_000]   # keep prompt within a sensible size


def create_brochure(company_name, url):
    """Non-streaming: returns the full brochure string."""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
    )
    return response.choices[0].message.content


def stream_brochure(company_name, url):
    """Streaming: prints the brochure to stdout as it is generated."""
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
        stream=True,
    )
    full = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full += delta
        print(delta, end="", flush=True)
    print()
    return full


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    company_name = input("Company name: ").strip()
    url = input("Website URL (include https://): ").strip()
    print("\nGenerating brochure...\n")

    brochure = stream_brochure(company_name, url)
    brochure = strip_code_fences(brochure)

    output_path = "brochure.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(brochure)
    print(f"\n\nBrochure saved to {os.path.abspath(output_path)}")