from typing import NamedTuple


Link = NamedTuple('Link', [('link', str), ('id', str)])


def host_link(hostname: str) -> Link:
    return Link(f"""<a class="link link-host" onclick="clicked('host-{hostname}')">{hostname}</a>""",
                f'host-{hostname}')


def osd_link(osd_id: int) -> Link:
    return Link(f"""<a class="link link-osd" onclick="clicked('osd-{osd_id}')">{osd_id}</a>""",
                f'osd-{osd_id}')


def mon_link(name: str) -> Link:
    return Link(f"""<a class="link link-mon" onclick="clicked('mon-{name}')">{name}</a>""",
                f'mon-{name}')


def err_link(reporter_id: str, mess: str = None) -> Link:
    return Link(f"""<a class="link link-err" onclick="clicked('err-{reporter_id}')">{mess}</a>""",
                f'err-{reporter_id}')


def rule_link(rule_name: str, rule_id: int) -> Link:
    return Link(f"""<a class="link link-rule" onclick="clicked('ruleset_info')">{rule_name}({rule_id})</a>""",
                'ruleset_info')


def pool_link(pool_name: str) -> Link:
    return Link(f"""<a class="link link-pool" onclick="clicked('pools_info')">{pool_name}</a>""",
                'pools_info')

