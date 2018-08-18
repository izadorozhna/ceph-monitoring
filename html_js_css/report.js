let curr_hash = "";

function clicked(el_id) {
    if (el_id === "")
        el_id = "cluster_summary";

    const el = document.getElementById(el_id);
    if (el === null) {
        alert("Element '" + el_id + "' is not defined")
    } else {
        for(const old of document.getElementsByClassName("main-ceph")) {
            old.classList.remove('main-ceph');
            old.classList.add('data-ceph');
        }

        el.classList.remove('data-ceph');
        el.classList.add('main-ceph');
        curr_hash = el_id;
        window.location.hash = el_id;
        window.scrollTo(0, 0);
    }
}

function onHashChanged() {
    if ("#" + curr_hash !== window.location.hash)
        clicked(window.location.hash.substr(1));
}

window.onload = onHashChanged;
window.onhashchange = onHashChanged;
