// Example of alert() function:
alert("Welcome!");

// Example of confirm() function:
if (confirm("Wanna go to the Google?")) {
  document.location.href="http://google.com";
}
else {
  alert("Don't be so afraid...");
}

// Example of prompt() function:
var user_name = prompt("What is your name?");
document.write("Welcome to the page " + user_name + "!");

// Example of creating random numerals.
text = "";
max_value = 20;  // Actually it will generate till 19.(9)
for (let i = 0; i < 10; i++) {
  float_num = Math.random() * max_value;
  text += String(Math.floor(float_num)) + "\n";
}
alert(text);

// Get the path to the DOM element:
function handleClick(event) {
  event.stopPropagation();
  var node = event.target;
  var thisPath = node.nodeName;
  console.log(thisPath); // For testing only
  while(node.parentNode) {
    node = node.parentNode;
    thisPath = node.nodeName + " > " + thisPath;
  }
  alert(thisPath);
};
// Register the click event handler for all nodes.
function attachHandler(node) {
  if (node == null) return;
  node.onclick = handleClick;
  for (let i = 0; i < node.childNodes.length; ++i)
    attachHandler(node.childNodes[i]);
};
// Or use rootnode = getElementById('theBody')
attachHandler(document.documentElement.childNodes[2]);

// Just get all the DOM elements.
var rootElement = document.documentElement;
var firstTier = rootElement.childNodes;
for (var i = 0; i < firstTier.length; i++) {
   console.log(firstTier[i]);
}

// Get list of elements:
heading_elems = document.getElementsByTagName("h1");
console.log(heading_elems);

// Change a style
h1_heading = document.getElementById("theHeading");
h1_heading.setAttribute("style", "color: red");
