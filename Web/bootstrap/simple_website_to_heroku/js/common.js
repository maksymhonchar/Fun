$(document).ready(function() {

	function fullHeight(){
		var winHeight = $(window).height();
		$('.start_s').height(winHeight);
	};
	fullHeight();
	$(window).resize(function(){
		fullHeight();
	});

	document.getElementById("google-button").onclick = function() {
		window.open('https://www.google.com.ua', '_blank');
	};

});
