### The llms bundle writes every URL in a plain-text-safe form

`llms-full.txt` is read as plain text by language-model tools, and their autolinkers take a URL up to the next
whitespace. So a markdown link inside parentheses, such as `([notes](https://.../README.md)).`, was requested as
`.../README.md)).`. GitHub's traffic showed those 404s.

`scripts/build_llms_bundle.py` now makes every URL safe:
- **A markdown link:** `[text](url)` becomes `text <url>`, and a badge becomes its alt text and link.
- **A bare URL:** one that ends in `)` or `.` becomes `<url>`, with the punctuation outside it. Any other bare URL is
  left byte-identical, including one inside a JSON string.
- **Section headings:** each heading's source URL is in angle brackets.
- **The check:** the build and `--check` refuse a bundle in which a URL is still directly followed by `)` or `.`.

The READMEs are unchanged. This is shared tooling, and experts4bit-qlora takes the same bytes next.
