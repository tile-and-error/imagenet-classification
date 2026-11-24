document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea")) return;
  const link = document.querySelector(e.key === "ArrowRight" ? "a[rel=next]" : e.key === "ArrowLeft" ? "a[rel=prev]" : null);
  if (link) window.location.href = link.href;
});

// Slide counter: "N / total", read from the contents page (index.html), so adding a
// slide only needs its line in the contents. A slide's number is its position there.
// Opened from disk (file://) the browser blocks the fetch, and the header keeps the
// plain slide number written in the page.
const counter = document.querySelector("[data-slide-counter]");
if (counter) {
  const page = (path) => path.split("/").pop().replace(/\.html$/, "");
  fetch("index.html")
    .then((response) => response.text())
    .then((html) => {
      const contents = new DOMParser().parseFromString(html, "text/html");
      const slides = [...contents.querySelectorAll(".toc a")].map((a) => page(a.getAttribute("href")));
      const position = slides.indexOf(page(location.pathname));
      if (position >= 0) counter.textContent = `${position + 1} / ${slides.length}`;
    })
    .catch(() => {});
}
