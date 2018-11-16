$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip();

  $('[data-toggle="popover"]').popover();

  $('.carousel').carousel({interval:500});

  $('#carousel-pause').click(function() {
    $('#mycarousel').carousel('pause');
  });

  $('#carousel-play').click(function() {
    $('#mycarousel').carousel('cycle');
  });

});
