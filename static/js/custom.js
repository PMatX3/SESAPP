$(".our-clients-logo-box").slick({
  slidesToShow: 6,
  slidesToScroll: 1,
  autoplay: true,
  arrows: false,
  autoplaySpeed: 1000,
});

$(".review-slider-for").slick({
  slidesToShow: 3,
  slidesToScroll: 3,
  arrows: false,
  fade: true,
  asNavFor: ".review-slider-nav",
});
$(".review-slider-nav").slick({
  slidesToShow: 3,
  slidesToScroll: 3,
  asNavFor: ".review-slider-for",
  dots: false,
  arrows: false,
  centerMode: true,
  focusOnSelect: true,
});
