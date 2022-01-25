const speed = 0.5;
const slide = document.querySelector(".certificates-slide");

slide.addEventListener("wheel", (evt) => {
    evt.preventDefault();
    slide.scrollLeft += evt.deltaY * speed;
});