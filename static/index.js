// Adapted from: https://github.com/sleepokay/mnist-flask-app/blob/master/static/index.js
(function() {
	var canvas = document.querySelector("#canvas"); //get canvas from DOM
	var context = canvas.getContext("2d"); // as 2 dimensional
	canvas.width = 280; // set the width
	canvas.height = 280; // set the height

	var Mouse = {x:0, y:0}; 
	var lastMouse = {x:0, y:0}; 
	context.fillStyle = "white"; // fill colour for canvas = white
	context.fillRect(0, 0, canvas.width, canvas.height); // fill the entire canvas 
	context.color = "black"; // brush colour
	context.lineWidth = 7; // width of brush line
    context.lineJoin = context.lineCap = 'round'; // round top on brush
	
	debug(); //call debug function

	//Listening for user mouse movement in the canvas
	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft-15;
		Mouse.y = e.pageY - this.offsetTop-15;
	}, false);

	// Once the user clicks on the canvas and move around the mouse,
	//the function will call eventListener mousemove to start drawing on canvas
	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	// Once the user unclicks the mouse while drawing on canvas,
	// then remove the eventListener mousemove to stop drawing. 
	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	// function that enables drawing
	var onPaint = function() {	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath(); //begin to draw the path
		context.moveTo(lastMouse.x, lastMouse.y); //move to wherever co-ordinates the mouse moves
		context.lineTo(Mouse.x,Mouse.y ); // creates a line from new point from last point
		context.closePath(); // creates a path from current point back to starting point
		context.stroke(); // draw the path
	};

	// Clear canvas function
	function debug() {
		$("#clearButton").on("click", function() { //Once clear button is clicked, init function
			context.clearRect( 0, 0, 280, 280 ); //clears rectangle
			context.fillStyle="white"; // fill colour = white
			context.fillRect(0,0,canvas.width,canvas.height); //fill the canvas white
		});
	}
}());