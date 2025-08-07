from datetime import datetime

from jinja2 import Environment, PackageLoader, select_autoescape


def render_html(metrics, out_path: str):
    env = Environment(
        loader=PackageLoader("tabriskscore.reports", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("report.html.jinja")
    html = tpl.render(
        metrics=metrics, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
