let curr_hash = "";

function clicked(el_id) {
    if (el_id === "")
        el_id = "cluster_summary";

    for(const old of document.getElementsByClassName("main-ceph")) {
        old.classList.remove('main-ceph');
        old.classList.add('data-ceph');
    }

    const el = document.getElementById(el_id);
    el.classList.remove('data-ceph');
    el.classList.add('main-ceph');
    curr_hash = el_id;
    window.location.hash = el_id;
    window.scrollTo(0, 0);
}

function onHashChanged() {
    if ("#" + curr_hash !== window.location.hash)
        clicked(window.location.hash.substr(1));
}

window.onload = onHashChanged;
window.onhashchange = onHashChanged;
