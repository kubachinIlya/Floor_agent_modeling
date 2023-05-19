var ContinuousVisualization = function(width, height, context, bg_image) {
	var height = height;
	var width = width;
	var context = context;
	var bg_image = bg_image

	this.draw = function(objects) {
		for (var i in objects) {
			var p = objects[i];
			if (p.Shape == "rect")
				this.drawRectange(p.x, p.y, p.w, p.h, p.Color, p.Filled);
			if (p.Shape == "circle")
				this.drawCircle(p.x, p.y, p.r, p.Color, p.Filled);
		};

	};

	this.drawCircle = function(x, y, radius, color, fill) {
		var cx = x * width;
		var cy = y * height;
		var r = radius;

		context.beginPath();
		context.arc(cx, cy, r, 0, Math.PI * 2, false);
		context.closePath();

		context.strokeStyle = color;
		context.stroke();

		if (fill) {
			context.fillStyle = color;
			context.fill();
		}

	};

	this.drawRectange = function(x, y, w, h, color, fill) {
		context.beginPath();
		var dx = w * width;
		var dy = h * height;

		// Keep the drawing centered:
		var x0 = (x*width) - 0.5*dx;
		var y0 = (y*height) - 0.5*dy;

		context.strokeStyle = color;
		context.fillStyle = color;
		if (fill)
			context.fillRect(x0, y0, dx, dy);
		else
			context.strokeRect(x0, y0, dx, dy);
	};

	this.resetCanvas = function() {
		context.clearRect(0, 0, width, height);
		context.drawImage(bg_image, 0, 0, width, height);
		context.beginPath();
	};
};

var Simple_Continuous_Module = function(canvas_width, canvas_height, bg_src) {
	// Create canvas tag
	var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' style='border:1px dotted'></canvas>";
	var canvas = $(canvas_tag)[0];
	$("#elements").append(canvas);

	// Create image tag
	var image_tag = "<img id='bg_source' src='" + bg_src + "' width='" + canvas_width + "' height='" + canvas_height + "' style='display:none;'></img>";
	var image_elem = $(image_tag)[0];
	$("#elements").append(image_elem);

	// Create the context and the drawing controller:
	var context = canvas.getContext("2d");
	var image = document.getElementById("bg_source");
	var canvasDraw = new ContinuousVisualization(canvas_width, canvas_height, context, image);

	this.render = function(data) {
		canvasDraw.resetCanvas();
		canvasDraw.draw(data);
	};

	this.reset = function() {
		canvasDraw.resetCanvas();
	};

};