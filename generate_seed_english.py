
import os, json, time, csv, requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.json")
OUT_CSV = os.getenv("OUT_CSV", "english_seeds_10k.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "1.5"))

SYSTEM = (
"You are a Young Kenyan English speaker writing realistic chat messages. "
"Constraints: 20-35 words each, natural WhatsApp tone, Nairobi Kenyan context, no emojis. "
)

PROMPT_TMPL = (
"Write {n} natural English chat sentences about the subtopic: '{subtopic}'. "
"Domain context: {domain}. Vary tone: friendly, serious, playful, flirty, annoyed. "
"Keep them distinct and non-redundant. Return ONLY a JSON array of strings."
)
def call_llm(n, domain, subtopic, retries=3):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT_TMPL.format(n=n, domain=domain, subtopic=subtopic)}
        ],
        "temperature": 1.05,
        "top_p": 0.95,
        "stream": False
    }
    delay = 1.2
    for attempt in range(retries):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            try:
                items = json.loads(content)
            except Exception:
                items = [s.strip("-• ").strip() for s in content.split("\n") if s.strip()]
            out = []
            seen = set()
            for s in items:
                s = (s or "").strip()
                if 5 <= len(s.split()) <= 15:
                    k = s.lower()
                    if k not in seen:
                        seen.add(k)
                        out.append(s)
            return out
        except Exception:
            time.sleep(delay)
            delay *= 1.7
    return []

def generate_all():
    if not API_KEY:
        raise SystemExit("Please set DEEPSEEK_API_KEY")
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        spec = json.load(f)

    rows = []
    for domain, cfg in spec.items():
        subs = cfg["subtopics"]
        for subtopic, target in subs.items():
            produced = 0
            while produced < target:
                need = min(BATCH_SIZE, target - produced)
                items = call_llm(need, domain, subtopic)
                unique = []
                seen = set()
                for s in items:
                    key = s.lower()
                    if key not in seen:
                        seen.add(key)
                        unique.append(s)
                for s in unique:
                    rows.append({"domain": domain, "subtopic": subtopic, "sentence": s})
                produced += len(unique)
                print(f"[{domain}/{subtopic}] {produced}/{target}")
                time.sleep(SLEEP_SEC)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "subtopic", "sentence"])
        w.writeheader()
        w.writerows(rows)
    print(f"Done. Wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    generate_all()
