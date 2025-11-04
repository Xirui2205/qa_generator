import json, sys

TOPICS = sys.argv[1] if len(sys.argv) > 1 else "topics.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tasks.tsv"

spec = json.load(open(TOPICS, "r", encoding="utf-8"))
with open(OUT, "w", encoding="utf-8") as w:
    for domain, cfg in spec.items():
        for subtopic, count in cfg["subtopics"].items():
            outpath = f"out/{domain}__{subtopic}.csv"
            w.write(f"{domain}\t{subtopic}\t{count}\t{outpath}\n")
print(f"Wrote {OUT}")