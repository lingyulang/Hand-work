# -*- coding: utf-8 -*-
"""性能评估报告 Markdown -> HTML（供 Edge headless 打印 PDF）。"""
import os

import markdown

here = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(here, "..", "docs", "性能评估报告.md")
out_path = os.path.join(here, "..", "docs", "性能评估报告.html")

with open(md_path, encoding="utf-8") as fp:
    body = markdown.markdown(
        fp.read(), extensions=["tables", "fenced_code", "toc"])

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<style>
  body {{ font-family: "Microsoft YaHei", "SimHei", sans-serif;
         font-size: 12px; line-height: 1.7; color: #222;
         max-width: 760px; margin: 24px auto; }}
  h1 {{ font-size: 20px; border-bottom: 2px solid #4a6fa5; padding-bottom: 6px; }}
  h2 {{ font-size: 16px; color: #2c4a75; margin-top: 22px; }}
  h3 {{ font-size: 13.5px; color: #2c4a75; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 5px 8px; text-align: left; }}
  th {{ background: #eef3fa; }}
  tr:nth-child(even) {{ background: #fafbfd; }}
  img {{ max-width: 100%; display: block; margin: 8px auto; }}
  blockquote {{ border-left: 3px solid #4a6fa5; margin: 8px 0;
               padding: 4px 12px; color: #555; background: #f5f8fc; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
         font-family: Consolas, monospace; font-size: 11px; }}
  pre code {{ display: block; padding: 8px; }}
</style></head><body>{body}</body></html>"""

with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(html)
print(f"HTML 已生成: {out_path}")
