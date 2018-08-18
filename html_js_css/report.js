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

    if (el_id === "crush_canvas_div") {
        draw_crush(root1);
    }
}

function onHashChanged() {
    if ("#" + curr_hash !== window.location.hash)
        clicked(window.location.hash.substr(1));
}

window.onload = onHashChanged;
window.onhashchange = onHashChanged;

//
// function draw_crush(crush) {
//     let canvas = document.getElementById("crush_canvas");
//     let ctx = canvas.getContext("2d");
//     ctx.font = "10px Verdana";
//     do_draw_crush(ctx, crush, 10, 10, canvas.width - 20, 10);
// }
//
// const y_space = 100;
// const text_sp = 5;
//
// function do_draw_crush(ctx, crush, x0, y0, w, text_h) {
//     const name_w = ctx.measureText(crush['name']).width;
//     const weght = crush['weight'].toFixed(2);
//     const w_w = ctx.measureText(weght).width;
//     const node_w = (name_w > w_w ? name_w : w_w) + 2 * text_sp;
//     const node_h = text_h * 2 + 3 * text_sp;
//
//     const start_x = x0 + (w - node_w) / 2;
//     ctx.rect(start_x, y0, node_w, node_h);
//     ctx.stroke();
//
//     ctx.fillText(crush['name'],
//                  start_x + text_sp + (node_w - 2 * text_sp - name_w) / 2,
//                  y0 + text_h + text_sp);
//     ctx.fillText(weght,
//                  start_x + text_sp + (node_w - 2 * text_sp - w_w) / 2,
//                  y0 + text_h * 2 + 2 * text_sp);
//
//     if ('childs' in crush) {
//         const step = w / crush['childs'].length;
//         for (const child of crush['childs']) {
//             ctx.beginPath();
//             ctx.lineWidth = "1";
//             ctx.moveTo(start_x + node_w / 2, y0 + node_h + text_sp);
//             ctx.lineTo(x0 + step / 2, y0 + y_space - text_sp);
//             ctx.stroke();
//
//             do_draw_crush(ctx, child, x0, y0 + y_space, step, text_h);
//             x0 += step;
//         }
//     }
// }