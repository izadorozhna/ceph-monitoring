function clicked(el_id) {
    for(const old of document.getElementsByClassName("main-ceph")) {
        old.classList.remove('main-ceph');
        old.classList.add('data-ceph');
    }
    const el = document.getElementById(el_id);
    el.classList.remove('data-ceph');
    el.classList.add('main-ceph');
}
