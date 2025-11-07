# src/render.py

from datetime import date
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path


def render_newsletter(clusters, since_hours: int, template_dir: str = "newsletter", template_file: str = "template.md.j2") -> str:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(disabled_extensions=(".j2",), default=True),
        trim_blocks=True, lstrip_blocks=True,
    )
    tmpl = env.get_template(template_file)
    return tmpl.render(date=str(date.today()), clusters=clusters, since_hours=since_hours)


def save_output(markdown: str, out_dir: str = "output") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"newsletter_{date.today()}.md"
    out.write_text(markdown, encoding="utf-8")
    return str(out)
