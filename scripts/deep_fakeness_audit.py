import os
import re

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

patterns = [
    (r"\b(mock|dummy|fake|hardcode[d]?|placeholder|stub)\b", "Mock/Fake/Hardcoded identifier"),
    (r"\b(random\.(uniform|randint|random|choice)|np\.random\.(uniform|randint|rand|randn|choice))\b", "Random number generation in code"),
    (r"\b(return\s+\[\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)", "Hardcoded bounding boxes"),
    (r"\b(time\.sleep\(\d+\))\b", "Artificial sleep delay"),
]

exclude_dirs = {".git", ".system_generated", "__pycache__", "node_modules", "checkpoints_backups", "scratch"}
core_dirs = ["pipeline", "models", "api", "mlops", "data", "training", "scripts"]

findings = []

for cdir in core_dirs:
    dir_path = os.path.join(ENGINE_ROOT, cdir)
    if not os.path.isdir(dir_path):
        continue
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if not f.endswith(".py"):
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, ENGINE_ROOT)
            
            with open(fpath, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
                
            for idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#") or clean_line.startswith('"""') or clean_line.startswith("'''"):
                    continue
                # Skip test files if they are just asserting mock
                for pat, desc in patterns:
                    match = re.search(pat, line, re.IGNORECASE)
                    if match:
                        findings.append({
                            "file": rel_path,
                            "line": idx,
                            "type": desc,
                            "match": match.group(0),
                            "content": clean_line[:120]
                        })

print(f"Total potential occurrences found: {len(findings)}")
by_file = {}
for item in findings:
    by_file.setdefault(item["file"], []).append(item)

for f, flist in sorted(by_file.items()):
    # filter to interesting core files
    print(f"\n--- {f} ({len(flist)} matches) ---")
    for item in flist[:8]:
        print(f"  Line {item['line']:>4} [{item['type']}]: {item['content']}")
    if len(flist) > 8:
        print(f"  ... and {len(flist)-8} more")
