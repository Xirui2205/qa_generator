import os, json, time, csv, argparse, requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM = (
"You are a Young Kenyan English speaker writing realistic chat messages. "
"Constraints: natural whatsApp tone, Nairobi Kenyan context, no emojis. "
)

PROMPT_TMPL = (
"Write {n} natural English chat sentences of 12 to 20 words about the subtopic: '{subtopic}'. "
"Domain context: {domain}. Vary tone: friendly, serious, playful, flirty, annoyed. "
"Keep them distinct and non-redundant. Return ONLY a JSON array of strings."
)
def call_llm(n, domain, subtopic, retries=3):
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
    delay = 1.5
    for attempt in range(retries):
        try:
            r = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=120)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--subtopic", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    produced = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "subtopic", "sentence"])
        while produced < args.target:
            need = min(args.batch, args.target - produced)
            items = call_llm(need, args.domain, args.subtopic)
            for s in items:
                w.writerow([args.domain, args.subtopic, s])
            produced += len(items)
            print(f"[{args.domain}/{args.subtopic}] {produced}/{args.target}")
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
