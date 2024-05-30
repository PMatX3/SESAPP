$('.AIContent').slick({
  dots: false,
  infinite: true,
  speed: 500,
  slidesToShow: 1,
  slidesToScroll: 1,
  cssEase: 'linear',
  fade: true,

});
//   $('.fade').slick({
//   dots: true,
//   infinite: true,
//   speed: 500,
//   fade: true,
//   cssEase: 'linear'
// });


$('.single_item').slick({
  infinite: true,
  slidesToShow: 1,
  slidesToScroll: 1,
  dots: false,
  speed: 300,
  arrow: true,
  responsive: [
    {
      breakpoint: 769,
      settings: {
        autoplay: true
      }
    }
  ]
});