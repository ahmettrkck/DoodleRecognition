var canvas, context, mouse_clicked = false,
    x_previous = 0,
    x_current = 0,
    y_previous = 0,
    y_current = 0

var style_stroke = "black", width_line = 2;

function init() {
    canvas = document.getElementById("drawing_canvas");
    context = canvas.getContext("2d");

    canvas.addEventListener("mousemove", function (event) {
        move_mouse_handler('move', event)
    }, false);
    canvas.addEventListener("mousedown", function (event) {
        move_mouse_handler('down', event)
    }, false);
    canvas.addEventListener("mouseup", function (event) {
        move_mouse_handler('up', event)
    }, false);
    canvas.addEventListener("mouseout", function (event) {
        move_mouse_handler('out', event)
    }, false);
}

function move_mouse_handler(action, event) {
    if (action == 'down') {
        mouse_clicked = true;

        x_previous = x_current;
        y_previous = y_current;
        x_current = event.clientX - canvas.offsetLeft;
        y_current = event.clientY - canvas.offsetTop;

        context.beginPath();
        context.fillStyle = x;
        context.fillRect(x_current, y_current, 2, 2);
        context.closePath();

    }
    if (action == 'up' || action == "out") {
        mouse_clicked = false;
    }
    if (action == 'move') {
        if (mouse_clicked) {
            x_previous = x_current;
            y_previous = y_current;
            x_current = event.clientX - canvas.offsetLeft;
            y_current = event.clientY - canvas.offsetTop;
            draw();
        }
    }
}

function draw() {
    context.beginPath();
    context.moveTo(x_previous, y_previous);
    context.lineTo(x_current, y_current);
    context.strokeStyle = style_stroke;
    context.lineWidth = width_line;
    context.stroke();
    context.closePath();
}

function predict() {
    // Convert dataURL to base64
    var image = canvas.toDataURL().split(",")[1];

    //  Replace Base64 characters '+ / =' with '. _ -'
    while (image.indexOf('+') > 0) {
        image = image.replace('+', '.');
    }
    while (image.indexOf('/') > 0) {
        image = image.replace('/', '_');
    }
    while (image.indexOf('=') > 0) {
        image = image.replace('=', '-');
    }

    window.location.href = '/predict?doodle=' + image;
}

function retry() {
    var confirmation = confirm("Are you sure you want to restart mate?");
    if (confirmation) {
        context.clearRect(0, 0, canvas.width, canvas.height);
    }
}

