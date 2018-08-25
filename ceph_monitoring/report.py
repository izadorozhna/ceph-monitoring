import shutil
import logging
import urllib.request
from pathlib import Path
from urllib.error import URLError
from typing import Optional, List, Any, Tuple

logger = logging.getLogger('report')

from . import html

class Report:
    def __init__(self, cluster_name: str, output_file_name: str) -> None:
        self.cluster_name = cluster_name
        self.output_file_name = output_file_name
        self.style: List[str] = []
        self.style_links: List[str] = []
        self.script_links: List[str] = []
        self.scripts: List[str] = []
        self.divs: List[Tuple[str, Optional[str], Optional[str], str]] = []
        self.onload: List[str] = ["onHashChanged()"]

    def add_block(self, name: str, header: Optional[str], block_obj: Any, menu_item: str = None):
        if menu_item is None:
            menu_item = header
        self.divs.append((name, header, menu_item, str(block_obj)))

    def save_to(self, output_dir: Path, pretty_html: bool = False,
                embed: bool = False):

        self.style_links.append("bootstrap.min.css")
        self.style_links.append("report.css")
        self.script_links.append("report.js")
        self.script_links.append("sorttable_utf.js")

        links: List[str] = []
        static_files_dir = Path(__file__).absolute().parent.parent / "html_js_css"

        def get_path(link: str) -> Tuple[bool, str]:
            if link.startswith("http://") or link.startswith("https://"):
                return False, link
            fname = link.rsplit('/', 1)[-1]
            return True, str(static_files_dir / fname)

        for link in self.style_links + self.script_links:
            local, fname = get_path(link)
            data = None

            if local:
                if embed:
                    data = open(fname, 'rb').read().decode("utf8")
                else:
                    shutil.copyfile(fname, output_dir / Path(fname).name)
            else:
                try:
                    data = urllib.request.urlopen(fname, timeout=10).read().decode("utf8")
                except (TimeoutError, URLError):
                    logger.warning(f"Can't retrieve {fname}")

            if data is not None:
                if link in self.style_links:
                    self.style.append(data)
                else:
                    self.scripts.append(data)
            else:
                links.append(link)

        css_links = links[:len(self.style_links)]
        js_links = links[len(self.style_links):]

        doc = html.Doc()
        with doc.html:
            with doc.head:
                doc.title("Ceph cluster report: " + self.cluster_name)

                for url in css_links:
                    doc.link(href=url, rel="stylesheet", type="text/css")

                if self.style:
                    doc.style("\n".join(self.style), type="text/css")

                for url in js_links:
                    doc.script(type="text/javascript", src=url)

                onload = "    " + ";\n    ".join(self.onload)
                self.scripts.append(f'function onload(){{\n{onload};\n}}')
                code = ";\n".join(self.scripts)

                if embed:
                    doc.script(code, type="text/javascript")
                else:
                    (output_dir / "onload.js").open("w").write(code)
                    doc.script(type="text/javascript", src="onload.js")

            with doc.body(onload="onload()"):
                with doc.div(_class="menu-ceph"):
                    with doc.ul():
                        for idx, div in enumerate(self.divs):
                            if div is not None:
                                name, _, menu, _ = div
                                if menu:
                                    if menu.endswith(":"):
                                        menu = menu[:-1]
                                    doc.li.span(menu,
                                                _class="menulink",
                                                onclick=f"clicked('{name}')",
                                                id=f"ptr_{name}")

                for div in self.divs:
                    doc("\n")
                    if div is not None:
                        name, header, menu_item, block = div
                        with doc.div(_class="data-ceph", id=name):
                            if header is None:
                                doc(block)
                            else:
                                doc.H3.center(header)
                                doc.br

                                if block != "":
                                    doc.center(block)

        index = f"<!doctype html>{doc}"
        index_path = output_dir / self.output_file_name

        try:
            if pretty_html:
                from bs4 import BeautifulSoup
                index = BeautifulSoup.BeautifulSoup(index).prettify()
        except:
            pass

        index_path.open("w").write(index)

